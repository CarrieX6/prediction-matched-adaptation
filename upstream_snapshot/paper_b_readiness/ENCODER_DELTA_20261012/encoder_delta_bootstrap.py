#!/usr/bin/env python3
"""Paired cluster bootstrap for the encoder difference in the Cflex -> M effect.

Reports, on the SAME cases and the SAME resampled clusters:
  gain_enc  = L(Cflex) - L(M)           (class-equal NLL; positive favors M)
  gain_auc  = AUC(M) - AUC(Cflex)       (percentage points)
  delta_enc = gain_CLIP - gain_BiomedCLIP   (for both metrics)

INPUT (.npz):
  y      (n,)      diagnosis (0/1)
  group  (n,)      cluster id (patient for CBIS-DDSM; case id for Derm7pt)
  bmc_cflex, bmc_m, clip_cflex, clip_m : (R, n) predicted P(y=1)
      R = 1 for CBIS-DDSM, R = 5 repetitions for Derm7pt.
Conventions follow the manuscript: losses are averaged over repetitions within each case,
then class-equal NLL is computed; AUC is computed per repetition and averaged.
Draws without both classes are rejected.

USAGE
  python encoder_delta_bootstrap.py cbis.npz --n-boot 10000 --seed 20261021 --out cbis_delta.json
"""
import argparse
import json
import numpy as np
from sklearn.metrics import roc_auc_score

EPS = 1e-12


def case_loss(y, P):
    P = np.clip(P, EPS, 1 - EPS)
    L = -(y[None, :] * np.log(P) + (1 - y[None, :]) * np.log(1 - P))
    return L.mean(axis=0)  # average over repetitions within each case


def class_equal(loss, y, w):
    out = 0.0
    for k in (0, 1):
        m = y == k
        out += 0.5 * np.sum(w[m] * loss[m]) / np.sum(w[m])
    return out


def wauc(y, P, w):
    return float(np.mean([roc_auc_score(y, P[r], sample_weight=w) for r in range(P.shape[0])]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--out", default="encoder_delta.json")
    a = ap.parse_args()

    d = np.load(a.npz)
    y, g = d["y"].astype(int), d["group"]
    names = ["bmc_cflex", "bmc_m", "clip_cflex", "clip_m"]
    P = {k: np.atleast_2d(d[k]) for k in names}
    L = {k: case_loss(y, P[k]) for k in names}

    groups, inv = np.unique(g, return_inverse=True)
    G = len(groups)
    rng = np.random.default_rng(a.seed)

    def metrics(w):
        res = {}
        for enc in ("bmc", "clip"):
            nll = class_equal(L[f"{enc}_cflex"], y, w) - class_equal(L[f"{enc}_m"], y, w)
            auc = 100 * (wauc(y, P[f"{enc}_m"], w) - wauc(y, P[f"{enc}_cflex"], w))
            res[enc] = (nll, auc)
        return res

    w1 = np.ones(len(y))
    pt = metrics(w1)
    point = {"gain_bmc_nll": pt["bmc"][0], "gain_clip_nll": pt["clip"][0],
             "delta_nll": pt["clip"][0] - pt["bmc"][0],
             "gain_bmc_auc_pp": pt["bmc"][1], "gain_clip_auc_pp": pt["clip"][1],
             "delta_auc_pp": pt["clip"][1] - pt["bmc"][1]}

    draws = {k: [] for k in point}
    done = 0
    while done < a.n_boot:
        counts = np.bincount(rng.integers(0, G, G), minlength=G)
        w = counts[inv].astype(float)
        if len(np.unique(y[w > 0])) < 2:
            continue  # reject single-class draws
        m = metrics(w)
        vals = {"gain_bmc_nll": m["bmc"][0], "gain_clip_nll": m["clip"][0],
                "delta_nll": m["clip"][0] - m["bmc"][0],
                "gain_bmc_auc_pp": m["bmc"][1], "gain_clip_auc_pp": m["clip"][1],
                "delta_auc_pp": m["clip"][1] - m["bmc"][1]}
        for k, v in vals.items():
            draws[k].append(v)
        done += 1

    out = {"n_boot": a.n_boot, "seed": a.seed, "n_cases": int(len(y)), "n_clusters": int(G)}
    for k, v in point.items():
        lo, hi = np.percentile(draws[k], [2.5, 97.5])
        out[k] = {"estimate": float(v), "ci95": [float(lo), float(hi)]}
    json.dump(out, open(a.out, "w"), indent=2)
    for k in point:
        e = out[k]
        print(f"{k:18s} {e['estimate']:+.6f}  [{e['ci95'][0]:+.6f}, {e['ci95'][1]:+.6f}]")


if __name__ == "__main__":
    main()
