#!/usr/bin/env python3
"""Run the frozen CLIP Analysis 1 and concept-quality summary."""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PANELS = HERE / "panels"
GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
QUANTILES = (0.05, 0.275, 0.5, 0.725, 0.95)
BOOTSTRAP_SEED = 20261010
BOOTSTRAP_DRAWS = 10_000

sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from paper2_evidence_handoff.data import CONCEPT_STATES  # noqa: E402


def sigmoid(z):
    z = np.asarray(z, float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1 + ez)
    return out


class Head:
    def __init__(self, c, max_iter):
        self.c, self.max_iter, self.model = float(c), int(max_iter), None

    def fit(self, x, y):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            self.model = LogisticRegression(
                C=self.c, class_weight="balanced", max_iter=self.max_iter,
                solver="lbfgs", random_state=20260912,
            ).fit(x, y)
        return self

    def logits(self, x):
        return np.asarray(self.model.decision_function(x), float)

    def probabilities(self, x):
        return sigmoid(self.logits(x))


def class_equal_nll(y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    loss = -(y * np.log(np.clip(p, 1e-12, 1)) + (1-y) * np.log(np.clip(1-p, 1e-12, 1)))
    return float(np.mean([loss[y == k].mean() for k in (0, 1)]))


def select_c(blocks, y, max_iter):
    curve = []
    for c in GRID:
        losses = [class_equal_nll(y[va], Head(c, max_iter).fit(xf, y[tr]).probabilities(xv))
                  for tr, va, xf, xv in blocks]
        curve.append(dict(C=c, fold_class_equal_NLL=losses,
                          mean_class_equal_NLL=float(np.mean(losses))))
    index = int(np.argmin([x["mean_class_equal_NLL"] for x in curve]))
    return GRID[index], curve


def fit_affine(z, y):
    z, y = np.asarray(z, float), np.asarray(y, int)
    counts = np.bincount(y, minlength=2).astype(float)
    weights = 1.0 / (2.0 * counts[y])

    def objective(theta):
        eta = theta[0] * z + theta[1]
        value = float(np.sum(weights * (np.logaddexp(0.0, eta) - y * eta)))
        residual = weights * (sigmoid(eta) - y)
        return value, np.asarray([np.sum(residual * z), np.sum(residual)], float)

    result = minimize(objective, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B",
                      bounds=((1e-6, None), (None, None)),
                      options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-10, "maxls": 50})
    if not result.success or result.x[0] <= 0:
        raise RuntimeError("affine calibration failed")
    return float(result.x[0]), float(result.x[1])


@dataclass(frozen=True)
class RestrictedCubicBasis:
    knots: np.ndarray
    mean: np.ndarray
    sd: np.ndarray

    @staticmethod
    def raw(logits, knots):
        z, k = np.asarray(logits, float), np.asarray(knots, float)
        if not np.all(np.diff(k) > 0):
            raise RuntimeError("restricted-cubic knot degeneracy")
        last_gap = k[4] - k[3]
        denominator = (k[4] - k[0]) ** 2
        nonlinear = []
        for j in range(3):
            nonlinear.append((np.maximum(z-k[j], 0.0) ** 3
                              - (k[4]-k[j])/last_gap * np.maximum(z-k[3], 0.0) ** 3
                              + (k[3]-k[j])/last_gap * np.maximum(z-k[4], 0.0) ** 3) / denominator)
        return np.column_stack([z, *nonlinear])

    @classmethod
    def fit(cls, logits):
        knots = np.quantile(np.asarray(logits, float), QUANTILES, method="linear")
        raw = cls.raw(logits, knots)
        mean, sd = raw.mean(0), raw.std(0, ddof=0)
        if np.any(sd <= 0):
            raise RuntimeError("restricted-cubic scale degeneracy")
        if np.linalg.matrix_rank(np.column_stack([np.ones(len(raw)), (raw-mean)/sd])) != 5:
            raise RuntimeError("restricted-cubic rank degeneracy")
        return cls(knots, mean, sd)

    def transform(self, logits):
        return (self.raw(logits, self.knots)-self.mean)/self.sd


def frozen_seed(base, *parts):
    payload = "|".join([str(base), *(str(p) for p in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") & 0x7fffffff


def derm_subfolds(rep, fold, h, frows):
    from sklearn.model_selection import KFold
    seed = frozen_seed(2026090829, "strict_meta_inner", rep, fold, h, "split")
    split = KFold(n_splits=4, shuffle=True, random_state=seed)
    return [(frows[tr], frows[va]) for tr, va in split.split(np.arange(len(frows)))]


def g_splits(domain, rep, fold, h, frows, hrows):
    members = [np.intersect1d(frows, block, assume_unique=False) for j, block in enumerate(hrows) if j != h]
    members = [x for x in members if len(x)]
    if domain == "cbis":
        if len(members) != 3:
            raise RuntimeError("CBIS G partition changed")
        return [(np.setdiff1d(frows, g), g) for g in members]
    return derm_subfolds(rep, fold, h, frows)


def reference_j_splits(splits, current_g):
    k = np.asarray(splits[current_g][0], int)
    result = [(np.setdiff1d(k, va), np.asarray(va, int))
              for g, (_, va) in enumerate(splits) if g != current_g]
    if set(np.concatenate([va for _, va in result])) != set(k):
        raise RuntimeError("deep reference folds changed")
    return result


def load_panel(domain, rep, fold):
    path = PANELS / f"{domain}_r{rep}_f{fold}" / "panel.npz"
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k].copy() for k in a.files}


def run_unit(domain, rep, fold):
    data = load_panel(domain, rep, fold)
    y = data["y"].astype(int)
    t, e = data["t"].astype(int), data["e"].astype(int)
    gold = data["gold"].astype(float)
    raw, val = data["raw"].astype(float), data["val"].astype(float)
    hrows = [data[f"h{h}"].astype(int) for h in range(4)]
    max_iter = 10000 if domain == "cbis" else 2000
    raw_blocks, q_t = [], np.full(len(t), np.nan)
    tpos = {int(row): pos for pos, row in enumerate(t)}
    selections = {"reference_H": [], "reference_D_local": []}

    # S, C, and M use the same nested H panels as the primary analysis.
    for h in range(4):
        frows, heldout = data[f"f{h}"].astype(int), hrows[h]
        xh, zh = data[f"x{h}"].astype(float), data[f"z{h}"].astype(float)
        splits = g_splits(domain, rep, fold, h, frows, hrows)
        blocks = [(tr, va, gold[tr], gold[va]) for tr, va in splits]
        c_ref, curve = select_c(blocks, y, max_iter)
        reference = Head(c_ref, max_iter).fit(gold[frows], y[frows])
        logits = reference.logits(zh)
        positions = [tpos[int(row)] for row in heldout]
        q_t[positions] = logits
        raw_blocks.append((frows, heldout, xh, zh))
        selections["reference_H"].append(dict(H=h, selected_C=c_ref, curve=curve))
    if not np.isfinite(q_t).all():
        raise RuntimeError("T OOF reference logits incomplete")

    c_gold, selections["outer_reference"] = select_c(
        [(tr, va, gold[tr], gold[va]) for tr, va, _, _ in raw_blocks], y, max_iter)
    c_m, selections["M"] = select_c(raw_blocks, y, max_iter)
    reference = Head(c_gold, max_iter).fit(gold[t], y[t])
    matched = Head(c_m, max_iter).fit(raw, y[t])
    logit_s = reference.logits(val)
    a, b = fit_affine(q_t, y[t])

    # Cflex repeats the exact D-local full chain used in the primary analysis.
    spline_blocks = []
    for h in range(4):
        d, heldout = data[f"f{h}"].astype(int), hrows[h]
        xd = data[f"x{h}"].astype(float)
        dpos = {int(row): pos for pos, row in enumerate(d)}
        q_d = np.full(len(d), np.nan)
        splits = g_splits(domain, rep, fold, h, d, hrows)
        h_records = []
        for g, (k, grows) in enumerate(splits):
            c_ref, curve = select_c([(tr, va, gold[tr], gold[va])
                                     for tr, va in reference_j_splits(splits, g)], y, max_iter)
            ref = Head(c_ref, max_iter).fit(gold[k], y[k])
            local = np.asarray([dpos[int(row)] for row in grows])
            q_d[local] = ref.logits(xd[local])
            h_records.append(dict(H=h, G=g, selected_C=c_ref, curve=curve))
        if not np.isfinite(q_d).all():
            raise RuntimeError("D-local OOF logits incomplete")
        basis = RestrictedCubicBasis.fit(q_d)
        q_h = np.asarray([q_t[tpos[int(row)]] for row in heldout])
        spline_blocks.append((d, heldout, basis.transform(q_d), basis.transform(q_h)))
        selections["reference_D_local"].extend(h_records)
    c_flex, selections["Cflex"] = select_c(spline_blocks, y, max_iter)
    final_basis = RestrictedCubicBasis.fit(q_t)
    flex = Head(c_flex, max_iter).fit(final_basis.transform(q_t), y[t])

    probs = {
        "S": sigmoid(logit_s),
        "C": sigmoid(a * logit_s + b),
        "Cflex": flex.probabilities(final_basis.transform(logit_s)),
        "M": matched.probabilities(val),
    }
    rows = []
    for pos, row in enumerate(e):
        item = dict(domain=domain, repetition=rep, outer_fold=fold, row_index=int(row),
                    case_id=str(data["ids"][row]), group_id=str(data["groups"][row]), label=int(y[row]))
        item.update({arm: float(p[pos]) for arm, p in probs.items()})
        rows.append(item)
    concept = dict(domain=domain, repetition=rep, outer_fold=fold, row_index=e,
                   gold=gold[e], predicted=val)
    return rows, concept, selections


def metric(domain, frame, arm):
    if domain == "cbis":
        y, p = frame.label.to_numpy(), frame[arm].to_numpy()
        return class_equal_nll(y, p), float(roc_auc_score(y, p))
    losses, aucs = [], []
    for _, part in frame.groupby("repetition", sort=True):
        y, p = part.label.to_numpy(), part[arm].to_numpy()
        sample_loss = -(y*np.log(np.clip(p, 1e-12, 1))+(1-y)*np.log(np.clip(1-p, 1e-12, 1)))
        losses.append(pd.DataFrame({"row_index": part.row_index, "label": y, "loss": sample_loss}))
        aucs.append(roc_auc_score(y, p))
    merged = pd.concat(losses).groupby(["row_index", "label"], as_index=False).loss.mean()
    nll = float(np.mean([merged.loc[merged.label == k, "loss"].mean() for k in (0, 1)]))
    return nll, float(np.mean(aucs))


def binary_auc(y, p):
    """Mann--Whitney AUC with average ranks for exact tie handling."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    n1, n0 = int(np.sum(y == 1)), int(np.sum(y == 0))
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = rankdata(p, method="average")
    return float((np.sum(ranks[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0))


def _fast_bootstrap_inputs(domain, frame):
    arms = ("S", "Cflex", "M")
    if domain == "cbis":
        groups = frame.group_id.drop_duplicates().to_numpy()
        indices = [np.flatnonzero(frame.group_id.to_numpy() == group) for group in groups]
        return dict(domain=domain, units=groups, indices=indices,
                    y=frame.label.to_numpy(int),
                    p={arm: frame[arm].to_numpy(float) for arm in arms})

    cases = frame.row_index.drop_duplicates().to_numpy()
    reps = np.sort(frame.repetition.unique())
    case_pos = {int(case): j for j, case in enumerate(cases)}
    rep_pos = {int(rep): j for j, rep in enumerate(reps)}
    y = np.full(len(cases), -1, int)
    p = {arm: np.full((len(reps), len(cases)), np.nan, float) for arm in arms}
    for row in frame.itertuples(index=False):
        j, r = case_pos[int(row.row_index)], rep_pos[int(row.repetition)]
        if y[j] not in (-1, int(row.label)):
            raise RuntimeError("Derm7pt label changed across repetitions")
        y[j] = int(row.label)
        for arm in arms:
            p[arm][r, j] = float(getattr(row, arm))
    if np.any(y < 0) or any(not np.isfinite(values).all() for values in p.values()):
        raise RuntimeError("Derm7pt repetition-by-case matrix is incomplete")
    return dict(domain=domain, units=cases, y=y, p=p)


def _fast_bootstrap_metrics(prepared, rng):
    arms = ("S", "Cflex", "M")
    if prepared["domain"] == "cbis":
        chosen = rng.integers(0, len(prepared["units"]), len(prepared["units"]))
        idx = np.concatenate([prepared["indices"][j] for j in chosen])
        y = prepared["y"][idx]
        values = {}
        for arm in arms:
            p = prepared["p"][arm][idx]
            values[arm] = (class_equal_nll(y, p), binary_auc(y, p))
        return values

    chosen = rng.integers(0, len(prepared["units"]), len(prepared["units"]))
    y = prepared["y"][chosen]
    values = {}
    for arm in arms:
        p = prepared["p"][arm][:, chosen]
        losses = -(y[None, :] * np.log(np.clip(p, 1e-12, 1))
                   + (1-y[None, :]) * np.log(np.clip(1-p, 1e-12, 1)))
        case_loss = losses.mean(axis=0)
        nll = float(np.mean([case_loss[y == k].mean() for k in (0, 1)]))
        auc = float(np.mean([binary_auc(y, p[r]) for r in range(p.shape[0])]))
        values[arm] = (nll, auc)
    return values


def summarize_predictions(frame):
    summary, contrasts = {}, {}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for domain, part in frame.groupby("domain", sort=True):
        summary[domain] = {arm: dict(zip(("class_equal_NLL", "AUC"), metric(domain, part, arm)))
                           for arm in ("S", "C", "Cflex", "M")}
        observed = {}
        for base in ("Cflex", "S"):
            observed[base] = {
                "NLL_gain": summary[domain][base]["class_equal_NLL"]-summary[domain]["M"]["class_equal_NLL"],
                "AUC_gain": summary[domain]["M"]["AUC"]-summary[domain][base]["AUC"],
            }
        draws = {base: {key: [] for key in ("NLL_gain", "AUC_gain")} for base in ("Cflex", "S")}
        prepared = _fast_bootstrap_inputs(domain, part)
        accepted = 0
        while accepted < BOOTSTRAP_DRAWS:
            values = _fast_bootstrap_metrics(prepared, rng)
            if any(not np.isfinite(value) for arm in values.values() for value in arm):
                continue
            for base in ("Cflex", "S"):
                draws[base]["NLL_gain"].append(values[base][0]-values["M"][0])
                draws[base]["AUC_gain"].append(values["M"][1]-values[base][1])
            accepted += 1
        contrasts[domain] = {}
        for base in ("Cflex", "S"):
            contrasts[domain][f"{base}_to_M"] = {}
            for key in ("NLL_gain", "AUC_gain"):
                lo, hi = np.quantile(draws[base][key], [0.025, 0.975])
                contrasts[domain][f"{base}_to_M"][key] = dict(
                    estimate=float(observed[base][key]), CI95=[float(lo), float(hi)])
    return summary, contrasts


def concept_quality(concepts):
    result = {}
    for domain in ("cbis", "derm"):
        selected = [x for x in concepts if x["domain"] == domain]
        if domain == "cbis":
            gold = np.concatenate([x["gold"] for x in selected])
            pred = np.concatenate([x["predicted"] for x in selected])
            labels = [f"concept_{j+1}" for j in range(gold.shape[1])]
            rows = []
            for j, name in enumerate(labels):
                rows.append(dict(concept=name, macro_OVR_AUC=float(roc_auc_score(gold[:, j], pred[:, j])),
                                 class_equal_concept_NLL=class_equal_nll(gold[:, j].astype(int), pred[:, j])))
        else:
            by_rep = []
            for rep in range(5):
                items = [x for x in selected if x["repetition"] == rep]
                order = np.concatenate([x["row_index"] for x in items])
                g = np.concatenate([x["gold"] for x in items])
                p = np.concatenate([x["predicted"] for x in items])
                idx = np.argsort(order)
                by_rep.append((g[idx], p[idx]))
            gold = by_rep[0][0]
            pred = np.mean([x[1] for x in by_rep], axis=0)
            widths = [len(states) for states in CONCEPT_STATES.values()]
            rows, start = [], 0
            for (name, _), width in zip(CONCEPT_STATES.items(), widths):
                gb, pb = gold[:, start:start+width], pred[:, start:start+width]
                y = gb.argmax(1)
                auc = roc_auc_score(gb, pb, average="macro", multi_class="ovr")
                losses = -np.log(np.clip(pb[np.arange(len(y)), y], 1e-12, 1))
                nll = float(np.mean([losses[y == k].mean() for k in range(width)]))
                rows.append(dict(concept=name, macro_OVR_AUC=float(auc), class_equal_concept_NLL=nll))
                start += width
        result[domain] = dict(
            concepts=rows,
            macro_OVR_AUC_range=[min(r["macro_OVR_AUC"] for r in rows), max(r["macro_OVR_AUC"] for r in rows)],
            class_equal_concept_NLL_range=[min(r["class_equal_concept_NLL"] for r in rows),
                                           max(r["class_equal_concept_NLL"] for r in rows)],
        )
    return result


def main():
    expected = [("cbis", 0, f) for f in range(5)] + [("derm", r, f) for r in range(5) for f in range(5)]
    rows, concepts, selections = [], [], {}
    with threadpool_limits(limits=1):
        for unit in expected:
            name = f"{unit[0]}_r{unit[1]}_f{unit[2]}"
            unit_rows, concept, selected = run_unit(*unit)
            rows.extend(unit_rows); concepts.append(concept); selections[name] = selected
            print(json.dumps({"complete": name}), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / "analysis1_predictions.csv", index=False)
    summary, contrasts = summarize_predictions(frame)
    quality = concept_quality(concepts)
    output = {
        "protocol": "ENCODER_ROBUSTNESS_CLIP_20261010/PROTOCOL.json",
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED,
                      "unit": {"cbis": "patient", "derm": "case with five repetitions retained"}},
        "absolute_metrics": summary,
        "paired_contrasts": contrasts,
        "concept_quality": quality,
    }
    (HERE / "RESULTS.json").write_text(json.dumps(output, indent=2) + "\n")
    (HERE / "SELECTIONS.json").write_text(json.dumps(selections, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
