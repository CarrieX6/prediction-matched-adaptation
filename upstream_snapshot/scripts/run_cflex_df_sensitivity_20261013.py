#!/usr/bin/env python3
"""Sensitivity of the scalar spline control to 2--5 basis coefficients.

The analysis reuses the archived sample-out training logits and outer-fold
inputs.  It refits only the scalar calibration head, applies the original
inner-H selection rule and C grid, and validates the four-coefficient variant
against the archived primary predictions before reporting any result.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper_b_readiness/CFLEX_CONTROL_20260917/full/units"
OUT = ROOT / "paper_b_readiness/CFLEX_SENSITIVITY_20261013"
GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
VARIANTS = {
    2: (0.05, 0.50, 0.95),
    3: (0.05, 0.35, 0.65, 0.95),
    4: (0.05, 0.275, 0.50, 0.725, 0.95),
    5: (0.05, 0.23, 0.41, 0.59, 0.77, 0.95),
}
UNITS = tuple([f"cbis_r0_f{f}" for f in range(5)] +
              [f"derm_r{r}_f{f}" for r in range(5) for f in range(5)])
SEED = 20261024
N_BOOT = 10000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Basis:
    def __init__(self, knots: np.ndarray, mean: np.ndarray, sd: np.ndarray):
        self.knots, self.mean, self.sd = knots, mean, sd

    @staticmethod
    def raw(z: np.ndarray, knots: np.ndarray) -> np.ndarray:
        z, k = np.asarray(z, float), np.asarray(knots, float)
        if z.ndim != 1 or len(k) < 3 or not np.all(np.diff(k) > 0):
            raise ValueError("invalid or degenerate restricted-cubic basis")
        gap = k[-1] - k[-2]
        scale = (k[-1] - k[0]) ** 2
        nonlinear = []
        for j in range(len(k) - 2):
            h = (np.maximum(z-k[j], 0.0) ** 3
                 - (k[-1]-k[j])/gap * np.maximum(z-k[-2], 0.0) ** 3
                 + (k[-2]-k[j])/gap * np.maximum(z-k[-1], 0.0) ** 3) / scale
            nonlinear.append(h)
        return np.column_stack([z, *nonlinear])

    @classmethod
    def fit(cls, z: np.ndarray, quantiles: tuple[float, ...]) -> "Basis":
        knots = np.quantile(np.asarray(z, float), quantiles, method="linear")
        raw = cls.raw(z, knots)
        mean, sd = raw.mean(axis=0), raw.std(axis=0, ddof=0)
        if np.any(sd <= 0) or not np.isfinite(sd).all():
            raise ValueError("basis standardization is degenerate")
        if np.linalg.matrix_rank(np.column_stack([np.ones(len(raw)), (raw-mean)/sd])) != raw.shape[1]+1:
            raise ValueError("basis rank is deficient")
        return cls(knots, mean, sd)

    def transform(self, z: np.ndarray) -> np.ndarray:
        return (self.raw(z, self.knots) - self.mean) / self.sd


def fit_head(x: np.ndarray, y: np.ndarray, c: float, domain: str) -> LogisticRegression:
    model = LogisticRegression(
        C=c, class_weight="balanced", solver="lbfgs", tol=0.0001,
        random_state=20260912, max_iter=10000 if domain == "cbis" else 2000,
    )
    return model.fit(x, y)


def ceq_nll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), 1e-12, 1-1e-12)
    loss = -y*np.log(p) - (1-y)*np.log1p(-p)
    return 0.5 * (loss[y == 0].mean() + loss[y == 1].mean())


def refit_unit(name: str, df: int) -> tuple[pd.DataFrame, dict]:
    domain = name.split("_")[0]
    src = SOURCE / name
    train = pd.read_csv(src / "training_logits.csv", dtype={"case_id": str, "group_id": str})
    outer = pd.read_csv(src / "predictions.csv", dtype={"case_id": str, "group_id": str})
    with np.load(ROOT/"results/deadline_nested_20260912"/name/"panel.npz", allow_pickle=False) as panel:
        t_index = panel["t"].astype(int)
        h_index = [panel[f"h{h}"].astype(int) for h in range(4)]
        d_index = [panel[f"f{h}"].astype(int) for h in range(4)]
        e_index = panel["e"].astype(int)
    qspec = VARIANTS[df]
    candidate = []
    for c in GRID:
        losses = []
        for h in range(4):
            d0 = train[(train.stage == "D_local_fullchain_oof_new_reference") & (train.H == h)]
            v0 = train[(train.stage == "T_fullchain_oof_cached") & (train.H == h)]
            d = d0.set_index("row_index").loc[d_index[h]].reset_index()
            v = v0.set_index("row_index").loc[h_index[h]].reset_index()
            basis = Basis.fit(d.raw_reference_logit.to_numpy(float), qspec)
            model = fit_head(basis.transform(d.raw_reference_logit.to_numpy(float)), d.y.to_numpy(int), c, domain)
            p = model.predict_proba(basis.transform(v.raw_reference_logit.to_numpy(float)))[:, 1]
            losses.append(ceq_nll(v.y.to_numpy(int), p))
        candidate.append((c, float(np.mean(losses)), losses))
    selected_c, selected_loss, fold_losses = min(candidate, key=lambda row: row[1])
    t0 = train[train.stage == "T_fullchain_oof_cached"]
    if t0.row_index.duplicated().any():
        raise ValueError(f"duplicate sample-out training row in {name}")
    t = t0.set_index("row_index").loc[t_index].reset_index()
    basis = Basis.fit(t.raw_reference_logit.to_numpy(float), qspec)
    model = fit_head(basis.transform(t.raw_reference_logit.to_numpy(float)), t.y.to_numpy(int), selected_c, domain)
    if outer.row_index.tolist() != e_index.tolist():
        raise ValueError(f"outer evaluation order differs in {name}")
    q = outer.logit_S.to_numpy(float)
    p = model.predict_proba(basis.transform(q))[:, 1]
    result = outer[["domain", "rep", "fold", "row_index", "case_id", "group_id", "y", "p_M"]].copy()
    result["df"] = df
    result["p_Cflex"] = p
    result["selected_C"] = selected_c
    result["unit"] = name
    record = {
        "unit": name, "domain": domain, "df": df, "knot_quantiles": list(qspec),
        "selected_C": selected_c, "selected_mean_inner_class_equal_NLL": selected_loss,
        "selected_fold_class_equal_NLL": fold_losses,
    }
    return result, record


def auc_cache(y: np.ndarray, p: np.ndarray):
    _, inv = np.unique(p, return_inverse=True)
    return inv, int(inv.max()) + 1, y.astype(float), (1-y).astype(float)


def weighted_auc(cache, w: np.ndarray) -> float:
    inv, n, pos, neg = cache
    wp = np.bincount(inv, weights=w*pos, minlength=n)
    wn = np.bincount(inv, weights=w*neg, minlength=n)
    return float(np.dot(wp, np.cumsum(wn)-0.5*wn) / (wp.sum()*wn.sum()))


def ceq_weighted(y: np.ndarray, loss: np.ndarray, w: np.ndarray) -> float:
    return 0.5 * sum(np.dot(loss[y == c], w[y == c]) / w[y == c].sum() for c in (0, 1))


def make_context(frame: pd.DataFrame, domain: str):
    reps = [0] if domain == "cbis" else list(range(5))
    first = frame[frame.rep == 0].sort_values("case_id")
    ids = first.case_id.tolist()
    y = first.y.to_numpy(int)
    groups = first.group_id.tolist() if domain == "cbis" else ids
    names = sorted(set(groups))
    mapping = {g: i for i, g in enumerate(names)}
    cluster = np.asarray([mapping[g] for g in groups], int)
    probabilities = {"Cflex": [], "M": []}
    for rep in reps:
        block = frame[frame.rep == rep].sort_values("case_id")
        if block.case_id.tolist() != ids:
            raise ValueError(f"case order mismatch in {domain} repetition {rep}")
        for arm in probabilities:
            probabilities[arm].append(block[f"p_{arm}"].to_numpy(float))
    losses, caches = {}, {}
    for arm, ps in probabilities.items():
        clipped = np.clip(np.asarray(ps), 1e-12, 1-1e-12)
        losses[arm] = np.mean(-y*np.log(clipped)-(1-y)*np.log1p(-clipped), axis=0)
        caches[arm] = [auc_cache(y, p) for p in ps]
    return y, cluster, len(names), probabilities, losses, caches


def metrics(ctx, w: np.ndarray) -> dict[str, float]:
    y, _, _, probs, losses, caches = ctx
    out = {}
    for arm in ("Cflex", "M"):
        out[f"{arm}_class_equal_NLL"] = ceq_weighted(y, losses[arm], w)
        out[f"{arm}_natural_NLL"] = float(np.dot(losses[arm], w) / w.sum())
        out[f"{arm}_Brier"] = float(np.mean([
            np.dot((p-y)**2, w) / w.sum() for p in probs[arm]
        ]))
        out[f"{arm}_AUC"] = float(np.mean([weighted_auc(c, w) for c in caches[arm]]))
    out["M_over_Cflex_NLL_gain"] = out["Cflex_class_equal_NLL"] - out["M_class_equal_NLL"]
    out["M_over_Cflex_AUC_gain"] = out["M_AUC"] - out["Cflex_AUC"]
    return out


def summarize(frame: pd.DataFrame, domain: str, df: int, rng: np.random.Generator):
    ctx = make_context(frame, domain)
    y, cluster, nclusters, *_ = ctx
    point = metrics(ctx, np.ones(len(y)))
    draws = {key: [] for key in point}
    proposals = 0
    while len(draws[next(iter(draws))]) < N_BOOT:
        proposals += 1
        counts = np.bincount(rng.integers(0, nclusters, nclusters), minlength=nclusters)
        w = counts[cluster].astype(float)
        if any(w[y == c].sum() == 0 for c in (0, 1)):
            continue
        row = metrics(ctx, w)
        for key, value in row.items():
            draws[key].append(value)
    rows = []
    for key, estimate in point.items():
        lo, hi = np.quantile(draws[key], [0.025, 0.975], method="linear")
        rows.append({"domain": domain, "df": df, "quantity": key,
                     "estimate": estimate, "ci_low": lo, "ci_high": hi})
    return rows, proposals


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    pins = {}
    predictions, selections = [], []
    with threadpool_limits(limits=1):
        for name in UNITS:
            for filename in ("training_logits.csv", "predictions.csv"):
                path = SOURCE / name / filename
                pins[str(path.relative_to(ROOT))] = sha256(path)
            for df in VARIANTS:
                pred, record = refit_unit(name, df)
                predictions.append(pred)
                selections.append(record)
    all_predictions = pd.concat(predictions, ignore_index=True)
    primary = pd.concat([
        pd.read_csv(SOURCE/name/"predictions.csv", dtype={"case_id": str})
        .assign(unit=name)[["unit", "row_index", "p_Cflex"]]
        for name in UNITS
    ], ignore_index=True).rename(columns={"p_Cflex": "p_Cflex_archived"})
    check = all_predictions[all_predictions.df == 4].merge(primary, on=["unit", "row_index"], validate="one_to_one")
    max_abs = float(np.max(np.abs(check.p_Cflex-check.p_Cflex_archived)))
    # The archived primary run and this replay use the same data order, knots,
    # scaling and selected C.  Across the current and archived numerical
    # environments, lbfgs can stop at slightly different points within its
    # prespecified 1e-4 tolerance.  Fail if that difference is material, then
    # retain the archived predictions for the prespecified four-column arm so
    # that the sensitivity analysis cannot perturb the paper's primary result.
    if max_abs > 5e-3:
        raise RuntimeError(f"four-coefficient reproduction failed: max abs={max_abs}")
    archived_c = {
        name: float(json.loads((SOURCE/name/"selections.json").read_text())["Cflex"]["selected_C"])
        for name in UNITS
    }
    selected4 = {row["unit"]: float(row["selected_C"]) for row in selections if row["df"] == 4}
    if selected4 != archived_c:
        raise RuntimeError("four-coefficient selected-C replay differs from the archived primary run")
    lookup = primary.set_index(["unit", "row_index"])["p_Cflex_archived"]
    mask4 = all_predictions.df == 4
    all_predictions.loc[mask4, "p_Cflex"] = [
        lookup.loc[(unit, row)]
        for unit, row in zip(all_predictions.loc[mask4, "unit"], all_predictions.loc[mask4, "row_index"], strict=True)
    ]
    all_predictions.to_csv(OUT/"all_predictions.csv", index=False, float_format="%.17g")
    pd.DataFrame(selections).to_json(OUT/"unit_selections.json", orient="records", indent=2)
    rng = np.random.default_rng(SEED)
    summaries, proposal_counts = [], {}
    for domain in ("cbis", "derm"):
        for df in VARIANTS:
            block = all_predictions[(all_predictions.domain == domain) & (all_predictions.df == df)]
            rows, proposals = summarize(block, domain, df, rng)
            summaries.extend(rows)
            proposal_counts[f"{domain}_df{df}"] = proposals
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT/"metrics_and_paired_effects.csv", index=False, float_format="%.17g")
    pd.DataFrame(selections).groupby(["domain", "df", "selected_C"]).size().rename("units").reset_index().to_csv(
        OUT/"selected_C_counts.csv", index=False
    )
    record = {
        "status": "COMPLETE", "created": "2026-10-13", "units": list(UNITS),
        "variants": {str(k): list(v) for k, v in VARIANTS.items()}, "C_grid": list(GRID),
        "selection": "mean class-equal NLL over four inner H groups; first minimum",
        "bootstrap_seed": SEED, "bootstrap_draws_per_domain_and_df": N_BOOT,
        "bootstrap_scope": "patient clusters for CBIS-DDSM; cases for Derm7pt; conditional on archived fitted upstream predictors",
        "df4_replay_max_abs_probability_difference_from_archived_primary": max_abs,
        "df4_reported_predictions": "archived prespecified primary predictions after replay gate",
        "input_sha256": pins, "proposal_counts": proposal_counts,
        "official_test_accessed": False, "new_upstream_fits": 0,
    }
    (OUT/"run_record.json").write_text(json.dumps(record, indent=2)+"\n")
    for relative, digest in pins.items():
        if sha256(ROOT/relative) != digest:
            raise RuntimeError(f"input changed during analysis: {relative}")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
