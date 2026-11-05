"""Post-result PH2 interface-shift analysis using frozen archived predictions."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from scipy.stats import rankdata, wasserstein_distance
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results/deadline_nested_20260912"
PH2 = ROOT / "paper_b_readiness/EXPLANATION_BASELINE_EXTERNAL_20260930/PH2"
OUT = ROOT / "paper_b_readiness/PH2_TRANSFER_MECHANISM_20261011"
UNITS = [f"derm_r{r}_f{f}" for r in range(5) for f in range(5)]
N_BOOT = 10_000
BOOT_SEED = 20261011


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def dump_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def probe_predict(model: dict, x: np.ndarray) -> np.ndarray:
    p = np.zeros((len(x), int(model["n_classes"])), dtype=float)
    if model["constant"] is not None:
        p[:, int(model["constant"])] = 1.0
    else:
        z = x @ np.asarray(model["coef"], dtype=float).T + np.asarray(model["intercept"], dtype=float)
        if len(model["classes"]) == 2:
            partial = np.column_stack([1.0 - expit(z[:, 0]), expit(z[:, 0])])
        else:
            partial = softmax(z, axis=1)
        p[:, np.asarray(model["classes"], dtype=int)] = partial
    return p / p.sum(axis=1, keepdims=True)


def reference_logit(head: dict, q: np.ndarray) -> np.ndarray:
    mean = np.asarray(head["mean"], dtype=float)
    scale = np.asarray(head["scale"], dtype=float)
    coef = np.asarray(head["coef"], dtype=float)
    intercept = np.asarray(head["intercept"], dtype=float)
    return (((q - mean) / scale) @ coef.T + intercept).ravel()


def spearman_matrix(x: np.ndarray) -> np.ndarray:
    ranks = rankdata(x, axis=0, method="average")
    ranks = ranks - ranks.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(ranks * ranks, axis=0))
    corr = (ranks.T @ ranks) / np.outer(denom, denom)
    return np.clip(corr, -1.0, 1.0)


def summaries(source_q: np.ndarray, external_q: np.ndarray,
              source_z: np.ndarray, external_z: np.ndarray) -> dict[str, float]:
    w1 = np.asarray([
        wasserstein_distance(source_q[:, j], external_q[:, j])
        for j in range(source_q.shape[1])
    ])
    source_corr = spearman_matrix(source_q)
    external_corr = spearman_matrix(external_q)
    tri = np.triu_indices(source_q.shape[1], 1)
    corr_delta = np.abs(external_corr[tri] - source_corr[tri])
    cohort = np.r_[np.zeros(len(source_z), dtype=int), np.ones(len(external_z), dtype=int)]
    score = np.r_[source_z, external_z]
    auc = float(roc_auc_score(cohort, score))
    return {
        "median_coordinate_w1": float(np.median(w1)),
        "max_coordinate_w1": float(np.max(w1)),
        "correlation_rms_change": float(np.sqrt(np.mean(corr_delta ** 2))),
        "correlation_median_abs_change": float(np.median(corr_delta)),
        "correlation_max_abs_change": float(np.max(corr_delta)),
        "z_cohort_auc_ph2_higher": auc,
        "z_cohort_auc_oriented": max(auc, 1.0 - auc),
        "z_mean_difference_ph2_minus_derm": float(np.mean(external_z) - np.mean(source_z)),
    }


def bootstrap(source_q: np.ndarray, external_q: np.ndarray,
              source_z: np.ndarray, external_z: np.ndarray,
              seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    keys = list(summaries(source_q, external_q, source_z, external_z))
    values = np.empty((N_BOOT, len(keys)), dtype=float)
    for b in range(N_BOOT):
        si = rng.integers(0, len(source_q), len(source_q))
        ei = rng.integers(0, len(external_q), len(external_q))
        row = summaries(source_q[si], external_q[ei], source_z[si], external_z[ei])
        values[b] = [row[key] for key in keys]
    point = summaries(source_q, external_q, source_z, external_z)
    return pd.DataFrame([
        {
            "metric": key,
            "estimate": point[key],
            "low": float(np.quantile(values[:, j], 0.025)),
            "high": float(np.quantile(values[:, j], 0.975)),
        }
        for j, key in enumerate(keys)
    ])


def coordinate_table(source_q: np.ndarray, external_q: np.ndarray,
                     names: list[str], stratum: str) -> pd.DataFrame:
    rows = []
    for j, name in enumerate(names):
        rows.append({
            "stratum": stratum,
            "coordinate": name,
            "derm_mean": float(source_q[:, j].mean()),
            "ph2_mean": float(external_q[:, j].mean()),
            "mean_difference_ph2_minus_derm": float(external_q[:, j].mean() - source_q[:, j].mean()),
            "wasserstein_1": float(wasserstein_distance(source_q[:, j], external_q[:, j])),
        })
    return pd.DataFrame(rows)


def correlation_table(source_q: np.ndarray, external_q: np.ndarray,
                      names: list[str], stratum: str) -> pd.DataFrame:
    source_corr = spearman_matrix(source_q)
    external_corr = spearman_matrix(external_q)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rows.append({
                "stratum": stratum,
                "coordinate_1": names[i],
                "coordinate_2": names[j],
                "derm_spearman": float(source_corr[i, j]),
                "ph2_spearman": float(external_corr[i, j]),
                "change_ph2_minus_derm": float(external_corr[i, j] - source_corr[i, j]),
                "absolute_change": float(abs(external_corr[i, j] - source_corr[i, j])),
            })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = OUT / "PROTOCOL.md"
    assert protocol.exists()
    final_marker = OUT / "COMPLETE.json"
    assert not final_marker.exists(), "Refusing to overwrite a completed analysis"

    feature_path = PH2 / "private/ph2_biomedclip_features.npz"
    with np.load(feature_path) as data:
        external_ids = data["case_id"].astype(str)
        external_x = data["embeddings"].astype(float)
    external_x /= np.linalg.norm(external_x, axis=1, keepdims=True).clip(1e-12)
    external_cases = pd.read_csv(PH2 / "private/ensemble_predictions.csv")
    np.testing.assert_array_equal(external_ids, external_cases.case_id.to_numpy(dtype=str))
    external_y = external_cases.y.to_numpy(dtype=int)

    first = read_json(PH2 / "private/recovered_models" / f"{UNITS[0]}.json")
    coordinate_names = [
        f"{model['name']}:{state}"
        for model in first["models"]
        for state in model["states"]
    ]
    assert len(coordinate_names) == 19

    source_rows: list[dict] = []
    external_q_units: list[np.ndarray] = []
    external_z_units: list[np.ndarray] = []
    unit_rows: list[dict] = []

    for unit in UNITS:
        repetition = int(unit.split("_")[1][1:])
        outer_fold = int(unit.split("_")[2][1:])
        panel_path = ARCHIVE / unit / "panel.npz"
        with np.load(panel_path) as panel:
            source_q = panel["val"].astype(float)
            source_index = panel["e"].astype(int)
            source_ids = panel["ids"][source_index].astype(str)
            source_y = panel["y"][source_index].astype(int)
        heads = read_json(ARCHIVE / unit / "heads.json")
        source_z = reference_logit(heads["raw_gold_nested"], source_q)

        published = pd.read_csv(
            ROOT / "paper_b_readiness/p1_complete/units" / unit / "predictions.csv"
        ).set_index("case_id").loc[source_ids]
        np.testing.assert_allclose(expit(source_z), published.p_S.to_numpy(), atol=1e-10)

        recovered = read_json(PH2 / "private/recovered_models" / f"{unit}.json")
        assert recovered["panel_sha256"] == sha256(panel_path)
        current_names = [
            f"{model['name']}:{state}"
            for model in recovered["models"]
            for state in model["states"]
        ]
        assert current_names == coordinate_names
        external_q = np.concatenate([
            probe_predict(model, external_x) for model in recovered["models"]
        ], axis=1)
        external_z = reference_logit(heads["raw_gold_nested"], external_q)
        external_q_units.append(external_q)
        external_z_units.append(external_z)

        for i, case_id in enumerate(source_ids):
            row = {
                "case_id": case_id,
                "y": int(source_y[i]),
                "repetition": repetition,
                "outer_fold": outer_fold,
                "z_S": float(source_z[i]),
            }
            row.update({name: float(source_q[i, j]) for j, name in enumerate(coordinate_names)})
            source_rows.append(row)

        for label, selector in [
            ("all", np.ones(len(source_y), dtype=bool)),
            ("class0", source_y == 0),
            ("class1", source_y == 1),
        ]:
            ext_selector = np.ones(len(external_y), dtype=bool) if label == "all" else external_y == int(label[-1])
            metric = summaries(
                source_q[selector], external_q[ext_selector],
                source_z[selector], external_z[ext_selector],
            )
            unit_rows.append({"unit": unit, "stratum": label, **metric})

    external_q_units_array = np.stack(external_q_units)
    external_z_units_array = np.stack(external_z_units)
    np.testing.assert_allclose(
        expit(external_z_units_array).mean(axis=0),
        external_cases.p_S.to_numpy(), atol=1e-10,
    )

    source_frame = pd.DataFrame(source_rows)
    assert len(source_frame) == 609 * 5
    assert source_frame.groupby("case_id").size().eq(5).all()
    assert source_frame.groupby("case_id").y.nunique().eq(1).all()
    aggregate_columns = ["z_S", *coordinate_names]
    source_aggregate = source_frame.groupby("case_id", sort=True)[aggregate_columns].mean()
    source_aggregate["y"] = source_frame.groupby("case_id", sort=True).y.first()
    assert len(source_aggregate) == 609

    external_q_aggregate = external_q_units_array.mean(axis=0)
    external_z_aggregate = external_z_units_array.mean(axis=0)
    source_q_aggregate = source_aggregate[coordinate_names].to_numpy()
    source_z_aggregate = source_aggregate.z_S.to_numpy()
    source_y_aggregate = source_aggregate.y.to_numpy(dtype=int)

    interval_frames = []
    coordinate_frames = []
    correlation_frames = []
    for label, source_selector, external_selector in [
        ("all", np.ones(len(source_y_aggregate), dtype=bool), np.ones(len(external_y), dtype=bool)),
        ("class0", source_y_aggregate == 0, external_y == 0),
        ("class1", source_y_aggregate == 1, external_y == 1),
    ]:
        sq = source_q_aggregate[source_selector]
        eq = external_q_aggregate[external_selector]
        sz = source_z_aggregate[source_selector]
        ez = external_z_aggregate[external_selector]
        intervals = bootstrap(sq, eq, sz, ez, BOOT_SEED)
        intervals.insert(0, "stratum", label)
        intervals.insert(1, "n_derm", len(sq))
        intervals.insert(2, "n_ph2", len(eq))
        interval_frames.append(intervals)
        coordinate_frames.append(coordinate_table(sq, eq, coordinate_names, label))
        correlation_frames.append(correlation_table(sq, eq, coordinate_names, label))

    intervals = pd.concat(interval_frames, ignore_index=True)
    coordinates = pd.concat(coordinate_frames, ignore_index=True)
    correlations = pd.concat(correlation_frames, ignore_index=True)
    units = pd.DataFrame(unit_rows)
    intervals.to_csv(OUT / "case_bootstrap_summary.csv", index=False)
    coordinates.to_csv(OUT / "coordinate_shift.csv", index=False)
    correlations.to_csv(OUT / "correlation_shift.csv", index=False)
    units.to_csv(OUT / "same_model_sensitivity.csv", index=False)

    unit_summary = []
    metrics = [column for column in units if column not in {"unit", "stratum"}]
    for stratum, part in units.groupby("stratum", sort=False):
        for metric in metrics:
            unit_summary.append({
                "stratum": stratum,
                "metric": metric,
                "median": float(part[metric].median()),
                "minimum": float(part[metric].min()),
                "maximum": float(part[metric].max()),
            })
    pd.DataFrame(unit_summary).to_csv(OUT / "same_model_summary.csv", index=False)

    top_coordinates = coordinates.sort_values(
        ["stratum", "wasserstein_1"], ascending=[True, False]
    ).groupby("stratum", sort=False).head(5)
    top_correlations = correlations.sort_values(
        ["stratum", "absolute_change"], ascending=[True, False]
    ).groupby("stratum", sort=False).head(5)
    top_coordinates.to_csv(OUT / "top_coordinate_shifts.csv", index=False)
    top_correlations.to_csv(OUT / "top_correlation_shifts.csv", index=False)

    artifacts = [
        OUT / "PROTOCOL.md",
        OUT / "case_bootstrap_summary.csv",
        OUT / "coordinate_shift.csv",
        OUT / "correlation_shift.csv",
        OUT / "same_model_sensitivity.csv",
        OUT / "same_model_summary.csv",
        OUT / "top_coordinate_shifts.csv",
        OUT / "top_correlation_shifts.csv",
    ]
    dump_json(final_marker, {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "post_result_descriptive_analysis": True,
        "new_fitting_selection_or_recalibration": False,
        "n_bootstrap": N_BOOT,
        "bootstrap_seed_by_stratum": {"all": BOOT_SEED, "class0": BOOT_SEED, "class1": BOOT_SEED},
        "n_derm_cases": int(len(source_aggregate)),
        "n_ph2_cases": int(len(external_y)),
        "n_source_models": len(UNITS),
        "n_coordinates": len(coordinate_names),
        "source_prediction_replay_pass": True,
        "external_ensemble_replay_pass": True,
        "script_sha256": sha256(Path(__file__)),
        "feature_sha256": sha256(feature_path),
        "artifacts": {str(path.relative_to(OUT)): sha256(path) for path in artifacts},
    })
    print(intervals.to_string(index=False))
    print("\nTop coordinate shifts\n", top_coordinates.to_string(index=False))
    print("\nTop correlation shifts\n", top_correlations.to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
