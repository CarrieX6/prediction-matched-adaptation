"""Relate PH2 score-vector deviation to M-versus-Cflex prediction changes.

This post-result analysis uses the deterministic five-model matched ensemble
from analyze_interface_shift_null_20261012.py.  It fits no diagnostic model.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.covariance import LedoitWolf

from analyze_interface_shift_null_20261012 import BLOCK_WIDTHS, free_columns
from analyze_ph2_transfer_mechanism_20261011 import read_json, reference_logit, sha256


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "paper_b_readiness/INTERFACE_SHIFT_NULL_20261012"
PH2 = ROOT / "paper_b_readiness/EXPLANATION_BASELINE_EXTERNAL_20260930/PH2/private"
ARCHIVE = ROOT / "results/deadline_nested_20260912"
OUT = ROOT / "paper_b_readiness/PH2_FAILURE_CONNECTION_20261012"
N_BOOT = 10_000
SEED = 20261023


def raw_head(head: dict) -> tuple[float, np.ndarray]:
    coef = np.asarray(head["coef"], dtype=float).reshape(-1)
    mean = np.asarray(head["mean"], dtype=float)
    scale = np.asarray(head["scale"], dtype=float)
    w = coef / scale
    b = float(np.asarray(head["intercept"], dtype=float).reshape(-1)[0] - np.dot(w, mean))
    return b, w


def reference_coded_head(head: dict) -> tuple[float, np.ndarray]:
    b, w = raw_head(head)
    free_weights: list[float] = []
    start = 0
    for width in BLOCK_WIDTHS:
        reference_weight = w[start]
        b += reference_weight
        free_weights.extend((w[start + 1:start + width] - reference_weight).tolist())
        start += width
    return b, np.asarray(free_weights, dtype=float)


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    point = float(spearmanr(x, y).statistic)
    draws = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        index = rng.integers(0, len(x), len(x))
        draws[i] = float(spearmanr(x[index], y[index]).statistic)
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def selected_average(values: np.ndarray, selected: np.ndarray) -> np.ndarray:
    case_index = np.arange(values.shape[1])
    return np.mean([values[selected[r], case_index] for r in range(5)], axis=0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    complete = OUT / "COMPLETE.json"
    if complete.exists():
        raise RuntimeError(f"Refusing to overwrite completed analysis: {complete}")
    with np.load(INPUT / "analysis_input.npz") as d:
        data = {key: d[key] for key in d.files}
    with np.load(INPUT / "matched_primary.npz") as d:
        ph2_scores, ph2_z, selected = d["ph2_scores"], d["ph2_z"], d["selected"]
    with np.load(PH2 / "individual_model_predictions.npz") as d:
        probabilities = d["probabilities"].astype(float)
        units = d["units"].astype(str)
        arms = d["arms"].astype(str)
        np.testing.assert_array_equal(d["case_id"].astype(str), data["ph2_case_id"].astype(str))
    arm_index = {arm: i for i, arm in enumerate(arms)}
    p_m = selected_average(probabilities[:, :, arm_index["M"]], selected)
    p_cflex = selected_average(probabilities[:, :, arm_index["Cflex"]], selected)
    probability_difference = p_m - p_cflex

    free = free_columns()
    derm_free = data["derm_scores"][:, free]
    ph2_free = ph2_scores[:, free]
    covariance = LedoitWolf().fit(derm_free)
    mahalanobis = np.sqrt(np.maximum(covariance.mahalanobis(ph2_free), 0.0))
    rng = np.random.default_rng(SEED)
    signed = bootstrap_spearman(mahalanobis, probability_difference, rng)
    absolute = bootstrap_spearman(mahalanobis, np.abs(probability_difference), rng)

    contribution = np.empty((5, len(data["ph2_case_id"]), len(free)), dtype=float)
    intercept_difference = np.empty((5, len(data["ph2_case_id"])), dtype=float)
    logit_difference = np.empty((5, len(data["ph2_case_id"])), dtype=float)
    case_index = np.arange(len(data["ph2_case_id"]))
    for repetition in range(5):
        model_index = selected[repetition]
        for index in np.unique(model_index):
            mask = model_index == index
            unit = str(units[index])
            heads = read_json(ARCHIVE / unit / "heads.json")
            b_m, w_m = reference_coded_head(heads["raw_soft_nested"])
            b_r, w_r = reference_coded_head(heads["raw_gold_nested"])
            q = data["ph2_scores_models"][index, mask][:, free]
            contribution[repetition, mask] = q * (w_m - w_r)
            intercept_difference[repetition, mask] = b_m - b_r
            z_m = reference_logit(heads["raw_soft_nested"], data["ph2_scores_models"][index, mask])
            z_r = reference_logit(heads["raw_gold_nested"], data["ph2_scores_models"][index, mask])
            logit_difference[repetition, mask] = z_m - z_r
            np.testing.assert_allclose(
                expit(z_m), probabilities[index, case_index[mask], arm_index["M"]], atol=1e-10
            )
    reconstructed = intercept_difference + contribution.sum(axis=2)
    np.testing.assert_allclose(reconstructed, logit_difference, atol=1e-10)
    mean_contribution = contribution.mean(axis=0)
    mean_logit_difference = logit_difference.mean(axis=0)
    np.testing.assert_allclose(
        mean_logit_difference,
        intercept_difference.mean(axis=0) + mean_contribution.sum(axis=1),
        atol=1e-10,
    )

    names = data["coordinate_names"][free].astype(str)
    mean_abs = np.mean(np.abs(mean_contribution), axis=0)
    denominator = float(mean_abs.sum())
    w1 = np.asarray([
        wasserstein_distance(derm_free[:, j], ph2_free[:, j]) for j in range(len(free))
    ])
    coordinate_table = pd.DataFrame({
        "coordinate": names,
        "wasserstein": w1,
        "mean_signed_logit_contribution": mean_contribution.mean(axis=0),
        "mean_absolute_logit_contribution": mean_abs,
        "share_of_absolute_coordinate_contribution": mean_abs / denominator,
    }).sort_values("mean_absolute_logit_contribution", ascending=False)
    coordinate_table.to_csv(OUT / "coordinate_contributions.csv", index=False)
    pd.DataFrame({
        "case_id": data["ph2_case_id"].astype(str),
        "y": data["ph2_y"].astype(int),
        "mahalanobis_distance": mahalanobis,
        "p_M_minus_p_Cflex": probability_difference,
        "absolute_p_M_minus_p_Cflex": np.abs(probability_difference),
        "M_minus_reference_logit": mean_logit_difference,
    }).to_csv(OUT / "case_level_connection.csv", index=False)

    coordinate_association = float(spearmanr(w1, mean_abs).statistic)
    largest_shift = np.argsort(w1)[::-1]
    top3_share = float(mean_abs[largest_shift[:3]].sum() / denominator)
    results = {
        "n_ph2": int(len(ph2_free)),
        "n_derm_reference": int(len(derm_free)),
        "mahalanobis_vs_signed_p_difference_spearman": {
            "estimate": signed[0], "ci95": [signed[1], signed[2]],
        },
        "mahalanobis_vs_absolute_p_difference_spearman": {
            "estimate": absolute[0], "ci95": [absolute[1], absolute[2]],
        },
        "coordinate_wasserstein_vs_mean_absolute_contribution_spearman": coordinate_association,
        "three_largest_shift_coordinates": names[largest_shift[:3]].tolist(),
        "three_largest_shift_coordinates_share_of_absolute_coordinate_contribution": top3_share,
        "mean_M_minus_reference_logit": float(mean_logit_difference.mean()),
        "mean_intercept_contribution": float(intercept_difference.mean()),
        "n_bootstrap": N_BOOT,
        "seed": SEED,
    }
    (OUT / "connection_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    (OUT / "PROTOCOL.md").write_text(
        "# PH2 failure-connection analysis\n\n"
        "This post-result analysis uses the same deterministic five-model matched ensemble as "
        "the permutation-null interface analysis. Mahalanobis distance is estimated from the "
        "609 Derm7pt case-level out-of-fold mean score vectors with Ledoit-Wolf covariance. "
        "Spearman associations with p_M-p_Cflex use 10,000 case bootstrap draws (seed 20261023). "
        "M-minus-reference logits are decomposed exactly in a reference-coded 12-coordinate "
        "parameterization of the seven simplex blocks. No PH2-specific fitting is performed.\n"
    )
    artifacts = [OUT / "connection_results.json", OUT / "coordinate_contributions.csv", OUT / "case_level_connection.csv", OUT / "PROTOCOL.md"]
    complete.write_text(json.dumps({
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "post_result_descriptive_analysis": True,
        "new_fitting": False,
        "script_sha256": sha256(Path(__file__)),
        "artifacts": {path.name: sha256(path) for path in artifacts},
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2))
    print("\n", coordinate_table.head(6).to_string(index=False))


if __name__ == "__main__":
    main()
