"""Audit the existing PH2 permutation family and within-class associations.

Uses archived, matched five-model predictions. No model is trained or selected.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_b_readiness/REVISION_STATISTICAL_AUDIT_20261013"
CASES = ROOT / "paper_b_readiness/PH2_FAILURE_CONNECTION_20261012/case_level_connection.csv"
PERM = ROOT / "paper_b_readiness/INTERFACE_SHIFT_NULL_20261012/permutation_results.csv"
SEED = 20261025
N_BOOT = 10000


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "COMPLETE.json").exists():
        raise RuntimeError("Sealed audit exists; inspect it rather than overwriting.")
    frame = pd.read_csv(CASES)
    assert len(frame) == 200 and frame.case_id.is_unique
    assert frame.y.value_counts().to_dict() == {0: 160, 1: 40}
    rng = np.random.default_rng(SEED)
    records = []
    for label in (0, 1):
        subset = frame.loc[frame.y == label]
        x = subset.mahalanobis_distance.to_numpy()
        signed = subset.p_M_minus_p_Cflex.to_numpy()
        absolute = subset.absolute_p_M_minus_p_Cflex.to_numpy()
        samples = rng.integers(0, len(subset), size=(N_BOOT, len(subset)))
        for name, y in (("signed_probability_difference", signed),
                        ("absolute_probability_difference", absolute)):
            values = np.array([spearmanr(x[ix], y[ix]).statistic for ix in samples])
            assert np.isfinite(values).all()
            lower, upper = np.quantile(values, [0.025, 0.975])
            records.append({"diagnosis": label, "n": len(subset), "contrast": name,
                            "rho": float(spearmanr(x, y).statistic),
                            "ci_lower": float(lower), "ci_upper": float(upper)})
    associations = pd.DataFrame(records)
    associations.to_csv(OUT / "classwise_associations.csv", index=False)
    permutations = pd.read_csv(PERM)
    assert len(permutations) == 16
    p = permutations.permutation_p.to_numpy()
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum(1.0, np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    restored = np.empty_like(p)
    restored[order] = adjusted
    permutations["holm_p"] = restored
    permutations["holm_reject_0_05"] = restored <= 0.05
    assert permutations.holm_reject_0_05.sum() == 12
    permutations.to_csv(OUT / "holm_permutation_results.csv", index=False)
    record = {"seed": SEED, "bootstrap_draws": N_BOOT,
              "prediction_ensemble": "matched five source models per PH2 case, offset 0",
              "bootstrap_unit": "case, resampled independently within diagnosis class",
              "covariance": "fixed archived Ledoit-Wolf fit on 609 Derm7pt OOF score vectors",
              "input_sha256": {str(CASES.relative_to(ROOT)): sha256(CASES),
                               str(PERM.relative_to(ROOT)): sha256(PERM)},
              "associations": records, "holm_family_size": len(p), "holm_alpha": 0.05}
    (OUT / "classwise_associations.json").write_text(json.dumps(record, indent=2)+"\n")
    (OUT / "PROTOCOL.md").write_text(
        "# Conditional statistical audit\n\n"
        "Archived matched-five-model PH2 predictions are used; neither the covariance "
        "estimate nor a diagnostic model is refitted. Within each diagnosis class, "
        "Spearman correlations of distance with signed and absolute M-minus-Cflex "
        "probability differences use 10,000 paired case-bootstrap draws (seed 20261025). "
        "Negative cases precede positive cases in the random-number stream; both "
        "contrasts share draws within a class. Percentile intervals condition on fitted "
        "models and are exploratory. Holm adjustment is applied at alpha 0.05 to the "
        "16 existing permutation tests (four statistics by four strata), without "
        "rerunning or changing the permutation results. The floor is 1/10,001.\n")
    outputs = sorted(p for p in OUT.iterdir() if p.is_file())
    (OUT / "COMPLETE.json").write_text(json.dumps({"outputs": {p.name: sha256(p) for p in outputs}}, indent=2)+"\n")
    print(associations.to_string(index=False))
    print(permutations[["stratum", "metric", "permutation_p", "holm_p", "holm_reject_0_05"]].to_string(index=False))


if __name__ == "__main__":
    main()
