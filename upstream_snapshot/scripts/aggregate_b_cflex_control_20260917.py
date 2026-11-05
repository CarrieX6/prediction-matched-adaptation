#!/usr/bin/env python3
"""Aggregate the 30 prespecified DEV C_flex units; never access Test."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "paper_b_readiness/CFLEX_CONTROL_20260917"
ARMS = ("S", "C", "Cflex", "M")
CONTRASTS = (("vector_over_flex", "Cflex", "M"),
             ("flex_over_affine", "C", "Cflex"),
             ("vector_over_substitution", "S", "M"))
UNITS = tuple([f"cbis_r0_f{f}" for f in range(5)] +
              [f"derm_r{r}_f{f}" for r in range(5) for f in range(5)])

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def ceq(y, loss, w):
    return 0.5 * sum(np.dot(loss[y == c], w[y == c]) / w[y == c].sum()
                     for c in (0, 1))

def auc_cache(y, p):
    """Tie groups in increasing score order, preserving full-precision ties."""
    _, inverse = np.unique(p, return_inverse=True)
    return inverse, int(inverse.max()) + 1, y.astype(float), (1-y).astype(float)

def weighted_auc(cache, w):
    inv, n, pos, neg = cache
    wp = np.bincount(inv, weights=w*pos, minlength=n)
    wn = np.bincount(inv, weights=w*neg, minlength=n)
    return float(np.dot(wp, np.cumsum(wn)-0.5*wn) / (wp.sum()*wn.sum()))

def context(frame, domain):
    reps = [0] if domain == "cbis" else list(range(5))
    frame = frame.copy()
    assert set(frame.rep) == set(reps)
    assert frame.y.isin([0, 1]).all()
    meta = frame.groupby("case_id").agg(y=("y", "nunique"), g=("group_id", "nunique"))
    assert (meta == 1).all().all(), "case label/group changes across repetitions"
    first = frame[frame.rep == 0].sort_values("case_id")
    ids = first.case_id.tolist()
    assert len(set(ids)) == len(ids)
    expected_n = 655 if domain == "cbis" else 609
    assert len(ids) == expected_n
    y = first.y.to_numpy(dtype=int)
    assert set(y) == {0, 1}
    groups = first.group_id.tolist() if domain == "cbis" else ids
    names = sorted(set(groups))
    mapping = {g:i for i,g in enumerate(names)}
    cluster = np.array([mapping[g] for g in groups], dtype=int)
    probabilities = {arm: [] for arm in ARMS}
    for r in reps:
        block = frame[frame.rep == r].sort_values("case_id")
        assert block.case_id.tolist() == ids
        assert set(block.fold) == set(range(5))
        for arm in ARMS:
            p = block[f"p_{arm}"].to_numpy(float)
            assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            probabilities[arm].append(p)
    loss = {}
    caches = {}
    points = {}
    w = np.ones(len(ids))
    for arm in ARMS:
        ps = probabilities[arm]
        clipped = np.clip(np.array(ps), 1e-12, 1-1e-12)
        loss[arm] = np.mean(-y*np.log(clipped)-(1-y)*np.log1p(-clipped), axis=0)
        caches[arm] = [auc_cache(y, p) for p in ps]
        points[f"{arm}_NLL"] = ceq(y, loss[arm], w)
        points[f"{arm}_AUC"] = np.mean([weighted_auc(c, w) for c in caches[arm]])
        points[f"{arm}_natural_NLL"] = loss[arm].mean()
        points[f"{arm}_Brier"] = np.mean([(p-y)**2 for p in ps])
    for name, baseline, candidate in CONTRASTS:
        points[name+"_NLL_gain"] = points[baseline+"_NLL"]-points[candidate+"_NLL"]
        points[name+"_AUC_gain"] = points[candidate+"_AUC"]-points[baseline+"_AUC"]
    case = pd.DataFrame({"domain": domain, "case_id":ids, "group_id":first.group_id.tolist(), "y":y})
    for arm in ARMS:
        case[f"NLL_{arm}"] = loss[arm]
    for name, baseline, candidate in CONTRASTS:
        case[name+"_NLL_gain"] = loss[baseline]-loss[candidate]
    return y, loss, caches, cluster, len(names), points, case

def bootstrap(ctx, rng):
    y, loss, caches, cluster, nclusters, _, _ = ctx
    draws = []
    proposals = 0
    while len(draws) < 10000:
        proposals += 1
        assert proposals <= 100000
        counts = np.bincount(rng.integers(0, nclusters, size=nclusters), minlength=nclusters)
        w = counts[cluster].astype(float)
        if any(w[y == c].sum() == 0 for c in (0, 1)):
            continue
        row = {"draw":len(draws), "proposal":proposals}
        for arm in ARMS:
            row[f"{arm}_NLL"] = ceq(y, loss[arm], w)
            row[f"{arm}_AUC"] = np.mean([weighted_auc(c, w) for c in caches[arm]])
        for name, baseline, candidate in CONTRASTS:
            row[name+"_NLL_gain"] = row[baseline+"_NLL"]-row[candidate+"_NLL"]
            row[name+"_AUC_gain"] = row[candidate+"_AUC"]-row[baseline+"_AUC"]
        draws.append(row)
    return pd.DataFrame(draws), proposals

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol-sha256", required=True)
    args = ap.parse_args()
    start = time.monotonic()
    protocol_file = BASE / "PROTOCOL.json"
    assert sha(protocol_file) == args.protocol_sha256
    protocol = json.loads(protocol_file.read_text())
    assert sha(Path(__file__)) == protocol["aggregate_script_sha256"]
    assert protocol["ordered_units"] == list(UNITS)
    assert protocol["status"] == "FROZEN_PRE_EXECUTION"
    complete = json.loads((BASE/"full/RUN_COMPLETE.json").read_text())
    assert complete["status"] == "COMPLETE" and complete["mode"] == "full"
    assert complete["protocol_sha256"] == args.protocol_sha256
    assert complete["units"] == list(UNITS)
    assert complete["reference_fits"] == 8380 and complete["spline_fits"] == 750
    assert complete["official_test_accessed"] is False
    frames = []
    pins = {}
    for name in UNITS:
        folder = BASE / "full/units" / name
        assert (folder / "COMPLETE.json").is_file(), f"incomplete unit {name}"
        receipt = json.loads((folder/"COMPLETE.json").read_text())
        assert receipt["status"] == "COMPLETE" and receipt["unit"] == name
        assert receipt["protocol_sha256"] == args.protocol_sha256
        assert receipt["official_test_accessed"] is False
        for filename, digest in receipt["artifact_sha256"].items():
            assert sha(folder/filename) == digest, f"changed artifact {name}/{filename}"
        p = folder / "predictions.csv"
        pins[str(p.relative_to(ROOT))] = sha(p)
        frame = pd.read_csv(p, dtype={"case_id":str,"group_id":str})
        if "rep" not in frame:
            frame = frame.rename(columns={"repetition":"rep", "outer_fold":"fold"})
        domain, rr, ff = name.split("_")
        if "domain" not in frame: frame["domain"] = domain
        if "rep" not in frame: frame["rep"] = int(rr[1:])
        if "fold" not in frame: frame["fold"] = int(ff[1:])
        assert (frame.domain == domain).all()
        assert (frame.rep == int(rr[1:])).all() and (frame.fold == int(ff[1:])).all()
        frames.append(frame)
    all_predictions = pd.concat(frames, ignore_index=True)
    assert len(all_predictions) == 3700
    output = BASE / "aggregate"
    output.mkdir(exist_ok=False)
    all_predictions.to_csv(output/"all_predictions.csv", index=False, float_format="%.17g")
    rng = np.random.default_rng(2026091517)
    summaries, cases, boots = [], [], []
    domains = {}
    for domain in ("cbis", "derm"):
        ctx = context(all_predictions[all_predictions.domain == domain], domain)
        draws, proposals = bootstrap(ctx, rng)
        draws.insert(0,"domain",domain)
        boots.append(draws)
        cases.append(ctx[-1])
        domains[domain] = {"cases":len(ctx[0]),"clusters":ctx[4],"bootstrap_proposals":proposals}
        for metric, point in ctx[-2].items():
            if metric in draws:
                lo, hi = np.quantile(draws[metric], [.025,.975], method="linear")
            else:
                lo, hi = None, None
            summaries.append({"domain":domain,"quantity":metric,"estimate":float(point),"ci_low":lo,"ci_high":hi})
    summary = pd.DataFrame(summaries)
    summary.to_csv(output/"paired_effects_and_metrics.csv",index=False,float_format="%.17g")
    pd.concat(cases,ignore_index=True).to_csv(output/"per_case_effects.csv",index=False,float_format="%.17g")
    pd.concat(boots,ignore_index=True).to_csv(output/"bootstrap_draws.csv",index=False,float_format="%.17g")
    primary = summary[(summary.domain == "cbis") & (summary.quantity == "vector_over_flex_NLL_gain")].iloc[0]
    if primary.estimate >= .005 and primary.ci_low > 0:
        gate = "GO_RESOURCE_ONLY"
    elif primary.ci_high < .005:
        gate = "NARROW"
    else:
        gate = "UNRESOLVED"
    record = {"status":"COMPLETE", "primary_resource_gate":gate,
              "gate_not_journal_innovation_or_acceptance":True,"Test_access":False,
              "protocol_sha256":args.protocol_sha256,"input_prediction_sha256":pins,
              "domains":domains,"units":30,"prediction_rows":3700,
              "bootstrap_draws_per_domain":10000,"bootstrap_seed":2026091517,
              "bootstrap_rng":"single stream; cbis then derm; sorted clusters",
              "CI_scope":"conditional on saved fitted DEV prediction panels, not retraining uncertainty",
              "wall_seconds":time.monotonic()-start}
    assert sha(protocol_file) == args.protocol_sha256
    for relative, digest in pins.items():
        assert sha(ROOT/relative) == digest, f"prediction changed during aggregation {relative}"
    (output/"run_record.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record,indent=2))

if __name__ == "__main__":
    main()
