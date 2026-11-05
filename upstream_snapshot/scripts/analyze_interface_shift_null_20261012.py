"""Permutation-null analysis of Derm7pt-to-PH2 predicted-interface shift.

The analysis uses 12 free coordinates, matches the PH2 ensemble to the five
Derm7pt out-of-fold repetitions, and includes a nevus-only Derm7pt negative
comparator.  No model is fitted or selected from PH2 outcomes.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata, wasserstein_distance
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from analyze_ph2_transfer_mechanism_20261011 import (
    probe_predict,
    read_json,
    reference_logit,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results/deadline_nested_20260912"
PH2 = ROOT / "paper_b_readiness/EXPLANATION_BASELINE_EXTERNAL_20260930/PH2"
OUT = ROOT / "paper_b_readiness/INTERFACE_SHIFT_NULL_20261012"
UNITS = [f"derm_r{r}_f{f}" for r in range(5) for f in range(5)]
BLOCK_WIDTHS = (3, 2, 3, 3, 3, 3, 2)
N_PERM = 10_000
SEED = 20261020


def stable_fold(case_id: str, repetition: int, offset: int = 0) -> int:
    token = int(hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:16], 16)
    return int((token + repetition + offset) % 5)


def free_columns() -> np.ndarray:
    columns: list[int] = []
    start = 0
    for width in BLOCK_WIDTHS:
        columns.extend(range(start + 1, start + width))
        start += width
    return np.asarray(columns, dtype=int)


def corr_matrix(x: np.ndarray) -> np.ndarray:
    ranks = rankdata(x, axis=0, method="average")
    corr = np.corrcoef(ranks, rowvar=False)
    return np.nan_to_num(corr, nan=0.0)


def direction_free_auc(a: np.ndarray, b: np.ndarray) -> float:
    y = np.r_[np.zeros(len(a), dtype=int), np.ones(len(b), dtype=int)]
    score = np.r_[a, b]
    auc = float(roc_auc_score(y, score))
    return max(auc, 1.0 - auc)


def statistics(a: np.ndarray, b: np.ndarray, za: np.ndarray, zb: np.ndarray) -> dict[str, float]:
    tri = np.triu_indices(a.shape[1], 1)
    ca, cb = corr_matrix(a), corr_matrix(b)
    return {
        "median_wasserstein": float(np.median([
            wasserstein_distance(a[:, j], b[:, j]) for j in range(a.shape[1])
        ])),
        "correlation_rms": float(np.sqrt(np.mean((ca[tri] - cb[tri]) ** 2))),
        "direction_free_z_auc": direction_free_auc(za, zb),
        "z_mean_difference_ph2_minus_derm": float(zb.mean() - za.mean()),
    }


def permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    za: np.ndarray,
    zb: np.ndarray,
    rng: np.random.Generator,
) -> list[dict]:
    observed = statistics(a, b, za, zb)
    x = np.vstack([a, b])
    z = np.r_[za, zb]
    na = len(a)
    null = {key: np.empty(N_PERM, dtype=float) for key in observed}
    for draw in range(N_PERM):
        order = rng.permutation(len(x))
        ia, ib = order[:na], order[na:]
        values = statistics(x[ia], x[ib], z[ia], z[ib])
        for key, value in values.items():
            null[key][draw] = value
    rows = []
    for key, value in observed.items():
        values = null[key]
        if key == "z_mean_difference_ph2_minus_derm":
            p = (1 + np.sum(np.abs(values) >= abs(value))) / (N_PERM + 1)
        else:
            p = (1 + np.sum(values >= value)) / (N_PERM + 1)
        rows.append({
            "metric": key,
            "observed": value,
            "null_mean": float(values.mean()),
            "observed_minus_null_mean": float(value - values.mean()),
            "null_p95": float(np.quantile(values, 0.95)),
            "permutation_p": float(p),
        })
    return rows


def build_inputs() -> dict[str, np.ndarray]:
    feature_path = PH2 / "private/ph2_biomedclip_features.npz"
    with np.load(feature_path) as data:
        ph2_ids = data["case_id"].astype(str)
        ph2_x = data["embeddings"].astype(float)
    ph2_x /= np.linalg.norm(ph2_x, axis=1, keepdims=True).clip(1e-12)
    ph2_cases = pd.read_csv(PH2 / "private/ensemble_predictions.csv")
    np.testing.assert_array_equal(ph2_ids, ph2_cases.case_id.astype(str).to_numpy())
    ph2_y = ph2_cases.y.to_numpy(dtype=int)

    first = read_json(PH2 / "private/recovered_models" / f"{UNITS[0]}.json")
    names = np.asarray([
        f"{model['name']}:{state}"
        for model in first["models"]
        for state in model["states"]
    ])
    assert len(names) == 19

    derm_rows: list[dict] = []
    ph2_q_models: list[np.ndarray] = []
    ph2_z_models: list[np.ndarray] = []
    for unit in UNITS:
        repetition = int(unit.split("_")[1][1:])
        fold = int(unit.split("_")[2][1:])
        panel_path = ARCHIVE / unit / "panel.npz"
        with np.load(panel_path) as panel:
            index = panel["e"].astype(int)
            derm_q = panel["val"].astype(float)
            derm_ids = panel["ids"][index].astype(str)
            derm_y = panel["y"][index].astype(int)
        heads = read_json(ARCHIVE / unit / "heads.json")
        derm_z = reference_logit(heads["raw_gold_nested"], derm_q)
        published = pd.read_csv(
            ROOT / "paper_b_readiness/p1_complete/units" / unit / "predictions.csv"
        ).set_index("case_id").loc[derm_ids]
        np.testing.assert_allclose(expit(derm_z), published.p_S.to_numpy(), atol=1e-10)

        recovered = read_json(PH2 / "private/recovered_models" / f"{unit}.json")
        assert recovered["panel_sha256"] == sha256(panel_path)
        ph2_q = np.concatenate([probe_predict(model, ph2_x) for model in recovered["models"]], axis=1)
        ph2_z = reference_logit(heads["raw_gold_nested"], ph2_q)
        ph2_q_models.append(ph2_q)
        ph2_z_models.append(ph2_z)

        for i, case_id in enumerate(derm_ids):
            row = {
                "case_id": case_id,
                "y": int(derm_y[i]),
                "repetition": repetition,
                "fold": fold,
                "z": float(derm_z[i]),
            }
            row.update({f"q{j}": float(derm_q[i, j]) for j in range(19)})
            derm_rows.append(row)

    ph2_q_models_arr = np.stack(ph2_q_models)
    ph2_z_models_arr = np.stack(ph2_z_models)
    np.testing.assert_allclose(
        expit(ph2_z_models_arr).mean(axis=0), ph2_cases.p_S.to_numpy(), atol=1e-10
    )

    derm_frame = pd.DataFrame(derm_rows)
    assert len(derm_frame) == 609 * 5
    q_columns = [f"q{j}" for j in range(19)]
    derm = derm_frame.groupby("case_id", sort=True)[["z", *q_columns]].mean()
    derm["y"] = derm_frame.groupby("case_id", sort=True).y.first()

    metadata = pd.read_csv(ROOT / "data/Derm7pt/derm7pt_dev609_only_v1_20260908.csv")
    diagnosis_by_id = {
        f"derm7pt:{int(row.metadata_index):04d}": str(row.diagnosis).strip().lower()
        for row in metadata.itertuples(index=False)
    }
    nevus_diagnoses = {
        "nevus", "blue nevus", "clark nevus", "combined nevus", "congenital nevus",
        "dermal nevus", "recurrent nevus", "reed or spitz nevus",
    }
    derm_nevus = np.asarray([diagnosis_by_id[case_id] in nevus_diagnoses for case_id in derm.index])

    out = {
        "coordinate_names": names,
        "derm_case_id": derm.index.to_numpy(dtype=str),
        "derm_scores": derm[q_columns].to_numpy(dtype=float),
        "derm_z": derm.z.to_numpy(dtype=float),
        "derm_y": derm.y.to_numpy(dtype=int),
        "derm_nevus": derm_nevus,
        "ph2_case_id": ph2_ids,
        "ph2_y": ph2_y,
        "ph2_scores_models": ph2_q_models_arr,
        "ph2_z_models": ph2_z_models_arr,
    }
    return out


def matched_ph2(data: dict[str, np.ndarray], offset: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = np.empty((5, len(data["ph2_case_id"])), dtype=int)
    for r in range(5):
        for i, case_id in enumerate(data["ph2_case_id"]):
            selected[r, i] = r * 5 + stable_fold(str(case_id), r, offset)
    case_index = np.arange(len(data["ph2_case_id"]))
    q = np.mean([
        data["ph2_scores_models"][selected[r], case_index] for r in range(5)
    ], axis=0)
    z = np.mean([
        data["ph2_z_models"][selected[r], case_index] for r in range(5)
    ], axis=0)
    return q, z, selected


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    complete = OUT / "COMPLETE.json"
    if complete.exists():
        raise RuntimeError(f"Refusing to overwrite completed analysis: {complete}")
    data = build_inputs()
    np.savez_compressed(OUT / "analysis_input.npz", **data)

    free = free_columns()
    ph2_q, ph2_z, selected = matched_ph2(data, offset=0)
    np.savez_compressed(OUT / "matched_primary.npz", ph2_scores=ph2_q, ph2_z=ph2_z, selected=selected)
    strata = [
        ("all", np.ones(len(data["derm_y"]), bool), np.ones(len(data["ph2_y"]), bool)),
        ("negative_all", data["derm_y"] == 0, data["ph2_y"] == 0),
        ("negative_nevus", (data["derm_y"] == 0) & data["derm_nevus"], data["ph2_y"] == 0),
        ("positive", data["derm_y"] == 1, data["ph2_y"] == 1),
    ]
    result_rows = []
    coordinate_rows = []
    rng = np.random.default_rng(SEED)
    for stratum, md, mp in strata:
        a, b = data["derm_scores"][md][:, free], ph2_q[mp][:, free]
        za, zb = data["derm_z"][md], ph2_z[mp]
        rows = permutation_test(a, b, za, zb, rng)
        for row in rows:
            result_rows.append({"stratum": stratum, "n_derm": int(md.sum()), "n_ph2": int(mp.sum()), **row})
        for local, original in enumerate(free):
            coordinate_rows.append({
                "stratum": stratum,
                "coordinate": str(data["coordinate_names"][original]),
                "derm_mean": float(a[:, local].mean()),
                "ph2_mean": float(b[:, local].mean()),
                "wasserstein": float(wasserstein_distance(a[:, local], b[:, local])),
            })

    results = pd.DataFrame(result_rows)
    coordinates = pd.DataFrame(coordinate_rows)
    results.to_csv(OUT / "permutation_results.csv", index=False)
    coordinates.to_csv(OUT / "coordinate_shift.csv", index=False)

    sensitivity_rows = []
    for offset in range(5):
        q, z, _ = matched_ph2(data, offset)
        for stratum, md, mp in strata:
            row = statistics(data["derm_scores"][md][:, free], q[mp][:, free], data["derm_z"][md], z[mp])
            sensitivity_rows.append({"offset": offset, "stratum": stratum, **row})
    pd.DataFrame(sensitivity_rows).to_csv(OUT / "matched_sensitivity.csv", index=False)

    (OUT / "PROTOCOL.md").write_text(
        "# Predicted-interface shift with a permutation null\n\n"
        "This post-result descriptive analysis reuses archived predictions and fits no model. "
        "Derm7pt uses the mean of five out-of-fold predictions per case. PH2 uses five source "
        "models per case, one from each repetition. Within repetition, the outer-fold model is "
        "assigned by a fixed SHA-256 hash of case ID plus repetition; five cyclic offsets provide "
        "a sensitivity analysis. Outcomes are not used for model assignment. The analysis drops "
        "one reference state per concept block, leaving 12 free coordinates and 66 correlation "
        "pairs. Cohort labels are permuted 10,000 times within each diagnostic stratum. The "
        "negative-class analysis is repeated with Derm7pt restricted to nevus diagnoses.\n"
    )
    artifacts = [
        OUT / "analysis_input.npz", OUT / "matched_primary.npz", OUT / "permutation_results.csv",
        OUT / "coordinate_shift.csv", OUT / "matched_sensitivity.csv", OUT / "PROTOCOL.md",
    ]
    complete.write_text(json.dumps({
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "n_permutations": N_PERM,
        "seed": SEED,
        "n_free_coordinates": int(len(free)),
        "n_correlation_pairs": int(len(free) * (len(free) - 1) / 2),
        "ph2_assignment": "case-specific deterministic hash; one outer fold per repetition",
        "artifacts": {p.name: sha256(p) for p in artifacts},
        "script_sha256": sha256(Path(__file__)),
    }, indent=2, sort_keys=True) + "\n")
    print(results.to_string(index=False))
    print("\nLargest coordinate shifts\n", coordinates.sort_values(["stratum", "wasserstein"], ascending=[True, False]).groupby("stratum").head(3).to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
