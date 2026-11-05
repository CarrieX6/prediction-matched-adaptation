"""Independently replay archived reference-head selection, fitting and evaluation.

Reads development archives only; exports aggregate discrepancies, not case data.
The reference arm is already a freshly fitted, inner-selected diagnostic head.
"""
from pathlib import Path
import hashlib
import json
import platform
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
import threadpoolctl

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results/deadline_nested_20260912"
OUT = ROOT / "paper_b_readiness/REVIEW_RECHECK_20261014"
GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
TOLERANCE = 1e-7


def fit(x, y, c, max_iter):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        return LogisticRegression(
            C=c, class_weight="balanced", solver="lbfgs",
            max_iter=max_iter, random_state=20260912,
        ).fit(x, y)


def ce_nll(y, p):
    loss = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))
    assert set(y) == {0, 1}
    return float(np.mean([loss[y == k].mean() for k in (0, 1)]))


def verify_unit(path):
    with np.load(path / "panel.npz") as a:
        d = {k: a[k] for k in a.files}
    y, gold, t, e, val = (d[k] for k in ("y", "gold", "t", "e", "val"))
    assert not np.intersect1d(t, e).size
    selections = [x for x in json.loads((path / "selection.json").read_text())
                  if x["condition"] == "raw_gold_nested"]
    assert [x["C"] for x in selections] == GRID
    limit = 10000 if path.name.startswith("cbis_") else 2000
    risks, errors = [], []
    for c, archived in zip(GRID, selections):
        losses = []
        for h in range(4):
            f, v = d[f"f{h}"], d[f"h{h}"]
            assert not np.intersect1d(f, np.r_[v, e]).size
            assert np.array_equal(np.sort(np.r_[f, v]), np.sort(t))
            model = fit(gold[f], y[f], c, limit)
            losses.append(ce_nll(y[v], model.predict_proba(gold[v])))
        risks.append(float(np.mean(losses)))
        errors.extend(np.abs(np.asarray(losses) - archived["fold_losses"]))
        errors.append(abs(risks[-1] - archived["inner_nll"]))
    selected = GRID[int(np.argmin(risks))]
    stored = json.loads((path / "heads.json").read_text())["raw_gold_nested"]
    assert selected == stored["C"]
    model = fit(gold[t], y[t], selected, limit)
    frame = pd.read_csv(path / "predictions.csv")
    discrepancy = {}
    for condition, x in [("raw_gold_nested", gold[e]), ("raw_gold_substitution", val)]:
        rows = frame.loc[frame.condition == condition]
        assert np.array_equal(rows.label.to_numpy(), y[e])
        assert np.array_equal(rows.unit_id.astype(str).to_numpy(), d["ids"][e].astype(str))
        fresh = model.predict_proba(x)[:, 1]
        archived_prob = rows.p1.to_numpy()
        stored_prob = expit(((x - np.asarray(stored["mean"])) / stored["scale"])
                            @ np.asarray(stored["coef"])[0] + stored["intercept"][0])
        discrepancy[condition] = {
            "fresh_vs_archived_probability_max_abs": float(np.max(np.abs(fresh - archived_prob))),
            "stored_head_vs_archived_probability_max_abs": float(np.max(np.abs(stored_prob - archived_prob))),
        }
    record = {
        "outer_unit": path.name, "training_cases": len(t), "evaluation_cases": len(e),
        "selected_C": selected, "inner_fold_nll_max_abs_error": float(max(errors)),
        "coefficient_max_abs_error": float(np.max(np.abs(model.coef_ - stored["coef"]))),
        "intercept_max_abs_error": float(np.max(np.abs(model.intercept_ - stored["intercept"]))),
        "probability_errors": discrepancy,
        "source_sha256": {n: hashlib.sha256((path / n).read_bytes()).hexdigest()
                          for n in ["panel.npz", "selection.json", "heads.json", "predictions.csv"]},
    }
    numerical_errors = errors + [record["coefficient_max_abs_error"], record["intercept_max_abs_error"]]
    numerical_errors += [x for errors_by_arm in discrepancy.values() for x in errors_by_arm.values()]
    assert max(numerical_errors) <= TOLERANCE, (path.name, max(numerical_errors))
    return record


def main():
    assert (OUT / "reference_head_verification_protocol.json").is_file()
    units = [ARCHIVE / f"{domain}_r{rep}_f{fold}"
             for domain in ("cbis", "derm")
             for rep in range(1 if domain == "cbis" else 5) for fold in range(5)]
    with threadpoolctl.threadpool_limits(limits=1):
        records = []
        for unit in units:
            records.append(verify_unit(unit))
            print(f"Verified {unit.name}", flush=True)
    result = {
        "status": "PASS", "verified_outer_units": len(records),
        "independent_inner_fits": len(records) * 6 * 4,
        "independent_final_reference_fits": len(records), "new_scientific_arm": False,
        "selection_uses_outer_evaluation": False, "tolerance": TOLERANCE,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                        "threadpoolctl": threadpoolctl.__version__},
        "units": records,
    }
    (OUT / "reference_head_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("units", "environment")}))


if __name__ == "__main__":
    main()
