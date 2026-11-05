#!/usr/bin/env python3
"""Paired case-level comparison of concept quality across frozen encoders."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BIOMED = ROOT / "results" / "deadline_nested_20260912"
CLIP = HERE / "panels"
DRAW_COUNT = 10_000
BOOTSTRAP_SEED = 20261012

sys.path.insert(0, str(ROOT / "src"))
from paper2_evidence_handoff.data import CONCEPT_STATES  # noqa: E402


def binary_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    n1 = int(np.sum(y == 1))
    n0 = int(np.sum(y == 0))
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = rankdata(score, method="average")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_encoder(root: Path, domain: str) -> dict[str, np.ndarray]:
    repetitions = (0,) if domain == "cbis" else range(5)
    first = np.load(root / f"{domain}_r0_f0" / "panel.npz", allow_pickle=False)
    n, q = first["gold"].shape
    gold = first["gold"].astype(float)
    ids = first["ids"].astype(str)
    groups = first["groups"].astype(str)
    predictions = np.full((len(tuple(repetitions)), n, q), np.nan, float)
    seen = np.zeros((len(tuple(repetitions)), n), bool)
    for rpos, repetition in enumerate(repetitions):
        for fold in range(5):
            with np.load(root / f"{domain}_r{repetition}_f{fold}" / "panel.npz", allow_pickle=False) as panel:
                for key, expected in (("gold", gold), ("ids", ids), ("groups", groups)):
                    if not np.array_equal(panel[key].astype(expected.dtype), expected):
                        raise RuntimeError(f"{root.name}/{domain}: {key} changed in r{repetition}f{fold}")
                heldout = panel["e"].astype(int)
                if np.any(seen[rpos, heldout]):
                    raise RuntimeError(f"{root.name}/{domain}: duplicate outer predictions")
                predictions[rpos, heldout] = panel["val"].astype(float)
                seen[rpos, heldout] = True
    if not seen.all() or not np.isfinite(predictions).all():
        raise RuntimeError(f"{root.name}/{domain}: incomplete outer predictions")
    return {
        "gold": gold,
        "ids": ids,
        "groups": groups,
        "predictions": predictions.mean(axis=0),
    }


def validate_pair(biomed: dict[str, np.ndarray], clip: dict[str, np.ndarray], domain: str) -> None:
    for key in ("gold", "ids", "groups"):
        if not np.array_equal(biomed[key], clip[key]):
            raise RuntimeError(f"{domain}: paired encoder {key} mismatch")


def class_equal_binary_nll(y: np.ndarray, p: np.ndarray) -> float:
    loss = -(y * np.log(np.clip(p, 1e-12, 1)) + (1-y) * np.log(np.clip(1-p, 1e-12, 1)))
    values = [loss[y == state].mean() for state in (0, 1)]
    return float(np.mean(values)) if np.isfinite(values).all() else np.nan


def summarize(domain: str, gold: np.ndarray, predicted: np.ndarray, indices: np.ndarray) -> tuple[float, float]:
    g = gold[indices]
    p = predicted[indices]
    aucs, nlls = [], []
    if domain == "cbis":
        for j in range(g.shape[1]):
            y = g[:, j].astype(int)
            aucs.append(binary_auc(y, p[:, j]))
            nlls.append(class_equal_binary_nll(y, p[:, j]))
    else:
        start = 0
        for states in CONCEPT_STATES.values():
            width = len(states)
            block_gold = g[:, start:start + width]
            block_pred = p[:, start:start + width]
            y = block_gold.argmax(axis=1)
            state_aucs, state_losses = [], []
            selected = np.clip(block_pred[np.arange(len(y)), y], 1e-12, 1)
            losses = -np.log(selected)
            for state in range(width):
                state_aucs.append(binary_auc((y == state).astype(int), block_pred[:, state]))
                state_losses.append(losses[y == state].mean() if np.any(y == state) else np.nan)
            aucs.append(float(np.mean(state_aucs)))
            nlls.append(float(np.mean(state_losses)))
            start += width
    return float(np.mean(aucs)), float(np.mean(nlls))


def bootstrap_indices(domain: str, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if domain == "derm":
        return rng.integers(0, len(groups), len(groups))
    units = np.unique(groups)
    members = [np.flatnonzero(groups == unit) for unit in units]
    chosen = rng.integers(0, len(units), len(units))
    return np.concatenate([members[pos] for pos in chosen])


def analyze_domain(domain: str, rng: np.random.Generator) -> dict:
    biomed = load_encoder(BIOMED, domain)
    clip = load_encoder(CLIP, domain)
    validate_pair(biomed, clip, domain)
    base_indices = np.arange(len(biomed["groups"]))
    observed = {
        "BiomedCLIP": summarize(domain, biomed["gold"], biomed["predictions"], base_indices),
        "CLIP": summarize(domain, clip["gold"], clip["predictions"], base_indices),
    }
    draws = {"macro_OVR_AUC_difference_CLIP_minus_BiomedCLIP": [],
             "macro_class_equal_NLL_difference_CLIP_minus_BiomedCLIP": []}
    accepted = 0
    while accepted < DRAW_COUNT:
        indices = bootstrap_indices(domain, biomed["groups"], rng)
        b_auc, b_nll = summarize(domain, biomed["gold"], biomed["predictions"], indices)
        c_auc, c_nll = summarize(domain, clip["gold"], clip["predictions"], indices)
        values = (c_auc - b_auc, c_nll - b_nll)
        if not np.isfinite(values).all():
            continue
        draws["macro_OVR_AUC_difference_CLIP_minus_BiomedCLIP"].append(values[0])
        draws["macro_class_equal_NLL_difference_CLIP_minus_BiomedCLIP"].append(values[1])
        accepted += 1
    output = {
        "sample_count": int(len(base_indices)),
        "resampling_unit": "patient" if domain == "cbis" else "case",
        "absolute_macro_means": {
            encoder: {"macro_OVR_AUC": float(values[0]), "macro_class_equal_NLL": float(values[1])}
            for encoder, values in observed.items()
        },
        "paired_differences_CLIP_minus_BiomedCLIP": {},
    }
    for key, values in draws.items():
        estimate = (observed["CLIP"][0] - observed["BiomedCLIP"][0]
                    if "AUC" in key else observed["CLIP"][1] - observed["BiomedCLIP"][1])
        lo, hi = np.quantile(values, (0.025, 0.975))
        output["paired_differences_CLIP_minus_BiomedCLIP"][key] = {
            "estimate": float(estimate), "CI95": [float(lo), float(hi)]}
    return output


def main() -> None:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output = {
        "draw_count": DRAW_COUNT,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "pairing": "same cases and outer predictions; Derm7pt predictions averaged across five repetitions",
        "domains": {},
    }
    for domain in ("cbis", "derm"):
        output["domains"][domain] = analyze_domain(domain, rng)
        print(json.dumps({"complete": domain}), flush=True)
    (HERE / "PAIRED_CONCEPT_QUALITY.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
