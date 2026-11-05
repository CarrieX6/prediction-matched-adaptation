#!/usr/bin/env python3
"""Run the single frozen Paper-B P1 engineering pilot.

This program is intentionally incapable of selecting another domain, repetition,
or outer fold.  It consumes only the development assets named in the frozen P1
protocol, reuses the existing strict-nested concept predictions, and publishes a
unit only after all fail-closed lineage and numerical audits pass.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_info, threadpool_limits

import paper_b_cbis_p3 as cb


PROTOCOL_PATH = ROOT / "paper_b_readiness/P1_PROTOCOL.json"
PROTOCOL_SIDECAR = ROOT / "paper_b_readiness/P1_PROTOCOL.sha256"
FINAL_DIR = ROOT / "paper_b_readiness/p1_pilot/cbis_r0_f0"
VALIDATION_RECEIPT = FINAL_DIR.parent / f"{FINAL_DIR.name}_pilot_validation.json"
DOMAIN = "cbis"
REPETITION = 0
OUTER_FOLD = 0
ARMS = ("R", "S", "C", "M", "SP", "CP", "MP")
GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
HEAD_SEED = 20260912
HEAD_MAX_ITER = 10000
LOSS_CLIP = 1e-12
PRIOR_EPS = 1e-6
CONTRAST_SPECS = (
    ("C_minus_M", "C", "M", "negative_favours_C"),
    ("S_minus_SP", "S", "SP", "positive_favours_SP"),
    ("S_minus_C", "S", "C", "positive_favours_C"),
    ("CP_minus_MP", "CP", "MP", "negative_favours_CP"),
    ("M_minus_MP", "M", "MP", "positive_favours_MP"),
)
FROZEN_PROTOCOL_SHA256 = "bdaca53493521a400f6307088d887eafa035f7eed4a094d522d05ee2ea5a1c1b"
OUTCOME_INPUT_SPECS = {
    "pilot_archive.panel": ("pilot_archive", "panel", "npz"),
    "pilot_archive.generation": ("pilot_archive", "generation", "json"),
    "pilot_archive.predictions": ("pilot_archive", "predictions", "csv"),
    "pilot_archive.selection": ("pilot_archive", "selection", "json"),
    "pilot_archive.heads": ("pilot_archive", "heads", "json"),
    "cbis_development_cohort": (None, "cbis_development_cohort", "npz"),
}
CONTROL_INPUT_KEYS = (
    "p0_hash_manifest",
    "strict_nested_protocol",
    "strict_nested_runner",
    "cbis_source",
    "cbis_config",
)
EXPECTED_OUTPUT_FILES = (
    "predictions.csv",
    "calibration_oof.csv",
    "per_case_effects.csv",
    "metrics.csv",
    "paired_effects.csv",
    "selections.json",
    "calibrators.json",
    "prior_lineage.json",
    "lineage.json",
    "audit.json",
    "fit_cost.json",
    "run.log",
    "execution_manifest.json",
    "COMPLETE.json",
)
PRE_MANIFEST_FILES = EXPECTED_OUTPUT_FILES[:12]
CSV_OUTPUTS = EXPECTED_OUTPUT_FILES[:5]
PRE_MANIFEST_JSON_OUTPUTS = EXPECTED_OUTPUT_FILES[5:11]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ordered(values: Iterable[object]) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_float_array(values: np.ndarray) -> str:
    """Hash a float array with an explicit shape and little-endian representation."""
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(b"\n")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def stable_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                   allow_nan=False, default=json_default) + "\n",
        encoding="utf-8",
    )


def atomic_json(path: Path, value: object) -> None:
    """Publish a small JSON receipt atomically in its target directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        stable_json(temporary, value)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialise {type(value)!r}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def probability2(p1: np.ndarray) -> np.ndarray:
    p1 = np.asarray(p1, dtype=np.float64)
    return np.column_stack((1.0 - p1, p1))


def scope_record(rows: np.ndarray, ids: np.ndarray, groups: np.ndarray) -> dict[str, object]:
    rows = np.asarray(rows, dtype=np.int64)
    return {
        "n_rows": int(len(rows)),
        "n_groups": int(len(set(groups[rows].astype(str)))),
        "row_indices": rows.tolist(),
        "row_index_ordered_sha256": sha256_ordered(rows),
        "case_id_ordered_sha256": sha256_ordered(ids[rows].astype(str)),
        "case_id_generator_sorted_sha256": cb.sha256_values(ids[rows].astype(str)),
        "group_id_ordered_sha256": sha256_ordered(groups[rows].astype(str)),
        "group_id_generator_sorted_sha256": cb.sha256_values(groups[rows].astype(str)),
    }


def disjoint(a: np.ndarray, b: np.ndarray) -> bool:
    return not bool(set(np.asarray(a).astype(str)).intersection(np.asarray(b).astype(str)))


def class_equal_nll(y: np.ndarray, p1: np.ndarray) -> float:
    return cb.class_equal_nll(np.asarray(y, dtype=np.int64), probability2(p1))


def fit_head(x: np.ndarray, y: np.ndarray, c_value: float,
             fit_counter: dict[str, object], role: str) -> cb.BalancedBinaryLogistic:
    started = time.perf_counter()
    head = cb.BalancedBinaryLogistic(
        c=float(c_value), max_iter=HEAD_MAX_ITER, seed=HEAD_SEED
    ).fit(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.int64))
    elapsed = time.perf_counter() - started
    fit_counter["total_head_fits"] = int(fit_counter["total_head_fits"]) + 1
    fit_counter["total_head_fit_seconds"] = float(fit_counter["total_head_fit_seconds"]) + elapsed
    by_role = fit_counter["by_role"]
    assert isinstance(by_role, dict)
    rec = by_role.setdefault(role, {"fits": 0, "seconds": 0.0})
    rec["fits"] = int(rec["fits"]) + 1
    rec["seconds"] = float(rec["seconds"]) + elapsed
    return head


def head_logit(head: cb.BalancedBinaryLogistic, x: np.ndarray) -> np.ndarray:
    if head.model is None:
        raise RuntimeError("unfitted diagnosis head")
    z = np.asarray(head.model.decision_function(np.asarray(x, dtype=np.float64)), dtype=np.float64)
    if z.shape != (len(x),) or not np.isfinite(z).all():
        raise RuntimeError("malformed diagnosis logits")
    return z


def select_with_rows(
    *,
    blocks: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    y: np.ndarray,
    fit_counter: dict[str, object],
    role: str,
    ids: np.ndarray,
    groups: np.ndarray,
) -> tuple[float, dict[str, object]]:
    candidates: list[dict[str, object]] = []
    means: list[float] = []
    fold_scopes = [
        {
            "fit": scope_record(fit_rows, ids, groups),
            "validation": scope_record(val_rows, ids, groups),
        }
        for fit_rows, val_rows, _, _ in blocks
    ]
    for c_value in GRID:
        losses = []
        for fit_rows, val_rows, fit_x, val_x in blocks:
            if len(fit_rows) != len(fit_x) or len(val_rows) != len(val_x):
                raise RuntimeError("selection feature rows are misaligned")
            head = fit_head(fit_x, y[fit_rows], c_value, fit_counter, role)
            p = head.predict_proba(val_x)[:, 1]
            losses.append(class_equal_nll(y[val_rows], p))
        mean_loss = float(np.mean(losses))
        means.append(mean_loss)
        candidates.append({
            "C": float(c_value),
            "fold_class_equal_NLL": losses,
            "mean_class_equal_NLL": mean_loss,
        })
    index = int(np.argmin(np.asarray(means, dtype=np.float64)))
    tied = np.flatnonzero(np.asarray(means) == means[index])
    return float(GRID[index]), {
        "C_grid": list(GRID),
        "fold_scopes": fold_scopes,
        "candidates": candidates,
        "selected_index": index,
        "selected_C": float(GRID[index]),
        "tie_break_first_minimum": bool(index == int(tied[0])),
    }


def validate_generated_selection(
    selection: Mapping[str, object],
    *,
    expected_folds: int,
    role: str,
) -> dict[str, object]:
    """Fail closed on every generated selection curve before it can be used."""
    candidates = selection.get("candidates")
    scopes = selection.get("fold_scopes")
    if not isinstance(candidates, list) or not isinstance(scopes, list):
        raise RuntimeError(f"{role} selection records changed type")
    c_values = [float(item.get("C", np.nan)) for item in candidates]
    fold_losses = [
        np.asarray(item.get("fold_class_equal_NLL", []), dtype=np.float64)
        for item in candidates
    ]
    means = np.asarray(
        [float(item.get("mean_class_equal_NLL", np.nan)) for item in candidates],
        dtype=np.float64,
    )
    checks = {
        "candidate_count_equals_grid": len(candidates) == len(GRID),
        "C_values_equal_grid_in_order": c_values == list(GRID),
        "fold_scope_count_exact": len(scopes) == expected_folds,
        "loss_count_exact_for_every_candidate": all(
            values.shape == (expected_folds,) for values in fold_losses
        ),
        "all_candidate_values_finite": bool(np.isfinite(means).all()) and all(
            np.isfinite(values).all() for values in fold_losses
        ),
        "candidate_means_exact": len(candidates) == len(GRID) and all(
            np.isclose(means[index], float(np.mean(fold_losses[index])), atol=1e-15, rtol=0)
            for index in range(len(candidates))
        ),
        "selected_index_is_first_minimum": (
            len(candidates) == len(GRID)
            and int(selection.get("selected_index", -1)) == int(np.argmin(means))
        ),
        "selected_C_matches_grid_index": (
            len(candidates) == len(GRID)
            and int(selection.get("selected_index", -1)) in range(len(GRID))
            and float(selection.get("selected_C", np.nan))
            == float(GRID[int(selection.get("selected_index", -1))])
        ),
        "tie_break_flag_true": selection.get("tie_break_first_minimum") is True,
    }
    for index, scope in enumerate(scopes):
        if not isinstance(scope, dict) or not isinstance(scope.get("fit"), dict) or not isinstance(scope.get("validation"), dict):
            checks[f"fold_{index}_scope_structure"] = False
            continue
        fit_rows = set(scope["fit"].get("row_indices", []))
        val_rows = set(scope["validation"].get("row_indices", []))
        checks[f"fold_{index}_fit_validation_row_disjoint"] = not bool(fit_rows.intersection(val_rows))
        checks[f"fold_{index}_fit_nonempty"] = bool(fit_rows)
        checks[f"fold_{index}_validation_nonempty"] = bool(val_rows)
    if not all(checks.values()):
        raise RuntimeError(f"generated selection failed closed for {role}: {checks}")
    return {
        "role": role,
        "checks": checks,
        "C_values": c_values,
        "candidate_count": len(candidates),
        "fold_count": expected_folds,
        "selected_index": int(selection["selected_index"]),
        "selected_C": float(selection["selected_C"]),
    }


def concept_fit_records(manifest: Mapping[str, object], key: str) -> list[dict[str, object]]:
    records = manifest[key]
    if not isinstance(records, list):
        raise RuntimeError(f"generation manifest {key} changed")
    result = [dict(record) for record in records if isinstance(record, dict) and "concept" in record]
    if len(result) != len(cb.STABLE_CONCEPTS):
        raise RuntimeError(f"expected {len(cb.STABLE_CONCEPTS)} concept fits in {key}")
    return result


def validate_generation_collection(
    generation: Mapping[str, object],
    panel: Mapping[str, np.ndarray],
    source_fold: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    """Resolve H/final manifests by role and prove their frozen multiplicities."""
    manifests = generation.get("manifests")
    selection_manifests = generation.get("selection_manifests")
    if not isinstance(manifests, list) or not isinstance(selection_manifests, list):
        raise RuntimeError("generation manifest collections changed type")
    h_candidates = [
        dict(item) for item in manifests
        if isinstance(item, dict) and item.get("diagnosis_selection_heldout_fold") is not None
    ]
    final_candidates = [
        dict(item) for item in manifests
        if isinstance(item, dict) and item.get("role") == "final_outer_train_probe"
    ]
    if len(manifests) != 5 or len(h_candidates) != 4 or len(final_candidates) != 1:
        raise RuntimeError("expected exactly four H manifests and one final-probe manifest")
    if int(generation.get("probe_bundles", -1)) != 11:
        raise RuntimeError("frozen probe-bundle count changed")
    if len(selection_manifests) != 4 or any(
        not isinstance(block, list) or len(block) != 1 for block in selection_manifests
    ):
        raise RuntimeError("each H must have exactly one archived selection-manifest copy")

    by_h_fold: dict[int, dict[str, object]] = {}
    for manifest in h_candidates:
        heldout = int(manifest["diagnosis_selection_heldout_fold"])
        if heldout in by_h_fold:
            raise RuntimeError("duplicate H heldout fold in generation manifests")
        by_h_fold[heldout] = manifest
        if int(manifest["outer_fold"]) != OUTER_FOLD:
            raise RuntimeError("H generation manifest outer fold changed")
        nested = manifest.get("nested_probe_panels")
        if not isinstance(nested, list) or len(nested) != 3:
            raise RuntimeError("each CBIS H manifest must contain three G manifests")

    resolved_h: list[dict[str, object]] = []
    duplicate_checks: list[bool] = []
    h_folds: list[int] = []
    for h_index in range(4):
        hrows = np.asarray(panel[f"h{h_index}"], dtype=np.int64)
        folds = np.unique(source_fold[hrows])
        if len(folds) != 1:
            raise RuntimeError("panel H block no longer maps to one original patient fold")
        heldout = int(folds[0])
        h_folds.append(heldout)
        if heldout not in by_h_fold:
            raise RuntimeError("panel H block lacks its generation manifest")
        manifest = by_h_fold[heldout]
        matches = [
            block[0] == manifest for block in selection_manifests
            if isinstance(block, list) and len(block) == 1
        ]
        exact_one = sum(bool(value) for value in matches) == 1
        duplicate_checks.append(exact_one)
        if not exact_one:
            raise RuntimeError("H generation manifest does not have exactly one identical selection copy")
        resolved_h.append(manifest)
    if len(set(h_folds)) != 4:
        raise RuntimeError("H original folds are not unique")

    final_manifest = final_candidates[0]
    final_checks = {
        "unique_final_role": len(final_candidates) == 1,
        "final_is_fifth_archived_manifest": manifests.index(next(
            item for item in manifests
            if isinstance(item, dict) and item.get("role") == "final_outer_train_probe"
        )) == 4,
        "outer_fold_exact": int(final_manifest.get("outer_fold", -1)) == OUTER_FOLD,
        "has_exact_concept_fit_count": len(concept_fit_records(final_manifest, "concept_fits")) == len(cb.STABLE_CONCEPTS),
        "has_no_nested_probe_panels": not bool(final_manifest.get("nested_probe_panels", [])),
    }
    if not all(final_checks.values()):
        raise RuntimeError(f"final-probe manifest role/count checks failed: {final_checks}")
    evidence = {
        "manifest_count": len(manifests),
        "H_manifest_count": len(h_candidates),
        "H_original_folds_in_panel_order": h_folds,
        "selection_manifest_block_sizes": [len(block) for block in selection_manifests],
        "each_H_has_one_identical_selection_copy": all(duplicate_checks),
        "final_manifest_index": 4,
        "final_checks": final_checks,
        "probe_bundles": int(generation["probe_bundles"]),
    }
    return resolved_h, final_manifest, evidence


def validate_archived_selection(
    archived: object,
    condition: str,
    expected_selected_c: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Validate count, grid order, fold count, means and first-minimum selection."""
    if not isinstance(archived, list):
        raise RuntimeError("archived selection object changed type")
    rows = [dict(row) for row in archived if isinstance(row, dict) and row.get("condition") == condition]
    c_values = [float(row.get("C", np.nan)) for row in rows]
    fold_counts = [len(row.get("fold_losses", [])) if isinstance(row.get("fold_losses"), list) else -1 for row in rows]
    losses = [np.asarray(row.get("fold_losses", []), dtype=np.float64) for row in rows]
    inner = [float(row.get("inner_nll", np.nan)) for row in rows]
    checks = {
        "row_count_equals_grid": len(rows) == len(GRID),
        "C_values_equal_grid_in_order": c_values == list(GRID),
        "four_fold_losses_each": fold_counts == [4] * len(GRID),
        "all_losses_finite": all(array.shape == (4,) and np.isfinite(array).all() for array in losses),
        "inner_nll_is_fold_mean": len(rows) == len(GRID) and all(
            np.isclose(inner[i], float(np.mean(losses[i])), atol=1e-15, rtol=0)
            for i in range(len(rows))
        ),
    }
    if len(rows) == len(GRID) and all(array.shape == (4,) for array in losses):
        selected_index = int(np.argmin(np.asarray(inner, dtype=np.float64)))
        checks["selected_C_is_first_minimum"] = float(GRID[selected_index]) == float(expected_selected_c)
    else:
        selected_index = -1
        checks["selected_C_is_first_minimum"] = False
    if not all(checks.values()):
        raise RuntimeError(f"archived selection curve failed closed for {condition}: {checks}")
    return rows, {
        "condition": condition,
        "checks": checks,
        "C_values": c_values,
        "fold_counts": fold_counts,
        "selected_index": selected_index,
        "selected_C": float(expected_selected_c),
    }


def restore_cbis(
    *,
    scores: np.ndarray,
    fit_rows: np.ndarray,
    predict_rows: np.ndarray,
    gold: np.ndarray,
    valid: np.ndarray,
    ids: np.ndarray,
    groups: np.ndarray,
    frozen_concept_fits: Sequence[Mapping[str, object]],
    score_role: str,
    h_fold: int | None,
    g_fold: int | None,
    lineage_records: list[dict[str, object]],
    transform_counter: dict[str, object],
    verification_specs: list[dict[str, object]],
) -> np.ndarray:
    started = time.perf_counter()
    scores = np.asarray(scores, dtype=np.float64)
    fit_rows = np.asarray(fit_rows, dtype=np.int64)
    predict_rows = np.asarray(predict_rows, dtype=np.int64)
    if scores.shape != (len(predict_rows), len(cb.STABLE_CONCEPTS)):
        raise RuntimeError(f"score shape changed for {score_role}")
    by_name = {str(record["concept"]): record for record in frozen_concept_fits}
    if set(by_name) != set(cb.STABLE_CONCEPTS):
        raise RuntimeError("generation manifest concept names changed")
    restored = np.empty_like(scores)
    for j, concept in enumerate(cb.STABLE_CONCEPTS):
        concept_rows = fit_rows[np.asarray(valid[fit_rows, j], dtype=bool)]
        targets = np.asarray(gold[concept_rows, j], dtype=np.int64)
        if not np.all(np.isin(targets, (0, 1))):
            raise RuntimeError("CBIS binary concept target changed")
        counts = np.bincount(targets, minlength=2).astype(np.int64)
        if counts.sum() == 0:
            raise RuntimeError("empty concept-valid prior scope")
        prior = counts.astype(np.float64) / float(counts.sum())
        q = scores[:, j]
        state = np.column_stack((1.0 - q, q))
        unnormalised = np.maximum(state, PRIOR_EPS) * prior[None, :]
        denominator = unnormalised.sum(axis=1)
        if not np.isfinite(unnormalised).all() or not np.isfinite(denominator).all() or np.any(denominator <= 0):
            raise RuntimeError("prior restoration produced invalid normaliser")
        restored[:, j] = unnormalised[:, 1] / denominator
        frozen = by_name[concept]
        fit_units = ids[concept_rows].astype(str)
        generator_hash = cb.sha256_values(fit_units)
        assertions = {
            "fit_breasts_exact": int(frozen["fit_breasts"]) == len(concept_rows),
            "fit_patients_exact": int(frozen["fit_patients"]) == len(set(groups[concept_rows].astype(str))),
            "negative_exact": int(frozen["negative"]) == int(counts[0]),
            "positive_exact": int(frozen["positive"]) == int(counts[1]),
            "fit_unit_sha256_exact": str(frozen["fit_unit_sha256"]) == generator_hash,
        }
        if not all(assertions.values()):
            raise RuntimeError(f"prior provenance mismatch: {score_role}/{concept}/{assertions}")
        assertions["independent_empirical_prior_exact"] = False
        lineage_records.append({
            "domain": DOMAIN,
            "repetition": REPETITION,
            "outer_fold": OUTER_FOLD,
            "score_role": score_role,
            "H": h_fold,
            "G": g_fold,
            "concept": concept,
            "concept_index": j,
            "fit_row_indices": concept_rows.tolist(),
            "predict_row_indices": predict_rows.tolist(),
            "fit_rows_valid": int(len(concept_rows)),
            "predict_rows": int(len(predict_rows)),
            "fit_patients": int(len(set(groups[concept_rows].astype(str)))),
            "negative": int(counts[0]),
            "positive": int(counts[1]),
            "prior": [float(prior[0]), float(prior[1])],
            "input_score_sha256": sha256_float_array(q),
            "restored_score_sha256": sha256_float_array(restored[:, j]),
            "fit_unit_sha256": generator_hash,
            "fit_unit_ordered_sha256": sha256_ordered(fit_units),
            "fit_group_ordered_sha256": sha256_ordered(groups[concept_rows].astype(str)),
            "predict_unit_ordered_sha256": sha256_ordered(ids[predict_rows].astype(str)),
            "predict_group_ordered_sha256": sha256_ordered(groups[predict_rows].astype(str)),
            "frozen_generating_probe_manifest": frozen,
            "assertions": assertions,
        })
    if not np.isfinite(restored).all() or np.any(restored < 0) or np.any(restored > 1):
        raise RuntimeError("restored scores are malformed")
    elapsed = time.perf_counter() - started
    transform_counter["calls"] = int(transform_counter["calls"]) + 1
    transform_counter["concept_records"] = int(transform_counter["concept_records"]) + len(cb.STABLE_CONCEPTS)
    transform_counter["probability_values"] = int(transform_counter["probability_values"]) + int(scores.size)
    transform_counter["seconds"] = float(transform_counter["seconds"]) + elapsed
    by_role = transform_counter["by_role"]
    assert isinstance(by_role, dict)
    role_counter = by_role.setdefault(score_role, {
        "calls": 0, "concept_records": 0, "probability_values": 0, "seconds": 0.0,
    })
    role_counter["calls"] = int(role_counter["calls"]) + 1
    role_counter["concept_records"] = int(role_counter["concept_records"]) + len(cb.STABLE_CONCEPTS)
    role_counter["probability_values"] = int(role_counter["probability_values"]) + int(scores.size)
    role_counter["seconds"] = float(role_counter["seconds"]) + elapsed
    verification_specs.append({
        "score_role": score_role,
        "H": h_fold,
        "G": g_fold,
        "scores": scores.copy(),
        "fit_rows": fit_rows.copy(),
        "predict_rows": predict_rows.copy(),
        "restored": restored.copy(),
        "frozen_concept_fits": [dict(item) for item in frozen_concept_fits],
        "lineage_start": len(lineage_records) - len(cb.STABLE_CONCEPTS),
        "lineage_stop": len(lineage_records),
    })
    return restored


def verify_prior_restoration_independently(
    *,
    specifications: Sequence[Mapping[str, object]],
    lineage_records: list[dict[str, object]],
    gold: np.ndarray,
    valid: np.ndarray,
    ids: np.ndarray,
    groups: np.ndarray,
    transform_counter: Mapping[str, object],
) -> dict[str, object]:
    """Independently rebuild every analytic restoration and its provenance."""
    started = time.perf_counter()
    role_calls: dict[str, int] = {}
    role_records: dict[str, int] = {}
    role_probability_values: dict[str, int] = {}
    expected_probability_values = 0
    verified_records = 0
    for spec in specifications:
        role = str(spec["score_role"])
        role_calls[role] = role_calls.get(role, 0) + 1
        fit_rows = np.asarray(spec["fit_rows"], dtype=np.int64)
        predict_rows = np.asarray(spec["predict_rows"], dtype=np.int64)
        scores = np.asarray(spec["scores"], dtype=np.float64)
        expected_probability_values += int(scores.size)
        role_probability_values[role] = (
            role_probability_values.get(role, 0) + int(scores.size)
        )
        expected = np.asarray(spec["restored"], dtype=np.float64)
        frozen_by_name = {
            str(item["concept"]): dict(item)
            for item in spec["frozen_concept_fits"]
        }
        rebuilt = np.zeros(scores.shape, dtype=np.float64)
        record_slice = lineage_records[int(spec["lineage_start"]):int(spec["lineage_stop"])]
        if len(record_slice) != len(cb.STABLE_CONCEPTS):
            raise RuntimeError(f"independent prior record slice changed for {role}")
        for concept_index, concept in enumerate(cb.STABLE_CONCEPTS):
            mask = np.asarray(valid[fit_rows, concept_index], dtype=bool)
            valid_fit_rows = fit_rows[mask]
            labels = np.asarray(gold[valid_fit_rows, concept_index], dtype=np.int64)
            negative = int(np.count_nonzero(labels == 0))
            positive = int(np.count_nonzero(labels == 1))
            total = negative + positive
            if total != len(labels) or total == 0:
                raise RuntimeError(f"independent prior labels malformed for {role}/{concept}")
            pi0 = negative / total
            pi1 = positive / total
            q = scores[:, concept_index]
            left = np.maximum(1.0 - q, PRIOR_EPS) * pi0
            right = np.maximum(q, PRIOR_EPS) * pi1
            rebuilt[:, concept_index] = right / (left + right)
            frozen = frozen_by_name[concept]
            record = record_slice[concept_index]
            fit_units = ids[valid_fit_rows].astype(str)
            checks = {
                "concept_identity_exact": (
                    str(record["concept"]) == concept
                    and int(record["concept_index"]) == concept_index
                ),
                "fit_rows_exact": record["fit_row_indices"] == valid_fit_rows.tolist(),
                "predict_rows_exact": record["predict_row_indices"] == predict_rows.tolist(),
                "counts_exact": (
                    int(record["negative"]) == negative
                    and int(record["positive"]) == positive
                    and int(frozen["negative"]) == negative
                    and int(frozen["positive"]) == positive
                ),
                "prior_exact": np.array_equal(
                    np.asarray(record["prior"], dtype=np.float64),
                    np.asarray([pi0, pi1], dtype=np.float64),
                ),
                "fit_unit_hash_exact": (
                    str(record["fit_unit_sha256"]) == cb.sha256_values(fit_units)
                    and str(frozen["fit_unit_sha256"]) == cb.sha256_values(fit_units)
                ),
                "input_score_hash_exact": str(record["input_score_sha256"]) == sha256_float_array(q),
            }
            if not all(checks.values()):
                raise RuntimeError(f"independent prior provenance failed for {role}/{concept}: {checks}")
            record["assertions"]["independent_empirical_prior_exact"] = True
            record["independent_verification"] = checks
            verified_records += 1
            role_records[role] = role_records.get(role, 0) + 1
        if not np.array_equal(rebuilt, expected):
            max_delta = float(np.max(np.abs(rebuilt - expected)))
            raise RuntimeError(f"independent restored array mismatch for {role}: {max_delta}")
        for concept_index, record in enumerate(record_slice):
            if str(record["restored_score_sha256"]) != sha256_float_array(rebuilt[:, concept_index]):
                raise RuntimeError(f"independent restored score hash mismatch for {role}/{concept_index}")

    expected_role_calls = {
        "x_G_inside_D": 12,
        "z_H_and_raw_T_block_H": 4,
        "val_E": 1,
    }
    expected_role_records = {
        "x_G_inside_D": 120,
        "z_H_and_raw_T_block_H": 40,
        "val_E": 10,
    }
    recorded_by_role = transform_counter.get("by_role", {})
    recorded_role_counts_exact = isinstance(recorded_by_role, dict) and all(
        isinstance(recorded_by_role.get(role), dict)
        and int(recorded_by_role[role].get("calls", -1)) == role_calls[role]
        and int(recorded_by_role[role].get("concept_records", -1)) == role_records[role]
        and int(recorded_by_role[role].get("probability_values", -1))
        == role_probability_values[role]
        and np.isfinite(float(recorded_by_role[role].get("seconds", np.nan)))
        and float(recorded_by_role[role].get("seconds", -1.0)) >= 0.0
        for role in expected_role_calls
    ) and set(recorded_by_role) == set(expected_role_calls)
    checks = {
        "transform_model_fits_zero": int(transform_counter.get("model_fits", -1)) == 0,
        "calls_exact_17": int(transform_counter.get("calls", -1)) == 17,
        "concept_records_exact_170": int(transform_counter.get("concept_records", -1)) == 170,
        "probability_value_count_independently_recomputed": (
            int(transform_counter.get("probability_values", -1))
            == expected_probability_values
        ),
        "transform_timing_finite_nonnegative": (
            np.isfinite(float(transform_counter.get("seconds", np.nan)))
            and float(transform_counter.get("seconds", -1.0)) >= 0.0
        ),
        "specification_count_exact_17": len(specifications) == 17,
        "lineage_record_count_exact_170": len(lineage_records) == 170,
        "verified_record_count_exact_170": verified_records == 170,
        "role_calls_exact": role_calls == expected_role_calls,
        "role_records_exact": role_records == expected_role_records,
        "recorded_by_role_counters_and_timing_exact": recorded_role_counts_exact,
        "all_independent_flags_true": all(
            record["assertions"].get("independent_empirical_prior_exact") is True
            for record in lineage_records
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"independent prior-restoration audit failed: {checks}")
    return {
        "checks": checks,
        "role_calls": role_calls,
        "role_records": role_records,
        "role_probability_values": role_probability_values,
        "probability_values": expected_probability_values,
        "verification_seconds": time.perf_counter() - started,
        "implementation": "independent direct binary-state recomputation",
    }


def affine_objective(theta: np.ndarray, z: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    a, b = np.asarray(theta, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    counts = np.bincount(y.astype(np.int64), minlength=2)
    if np.any(counts == 0):
        raise RuntimeError("both classes required for affine calibration")
    weights = np.where(y == 0, 1.0 / (2.0 * counts[0]), 1.0 / (2.0 * counts[1]))
    eta = a * z + b
    objective = float(np.sum(weights * (np.logaddexp(0.0, eta) - y * eta)))
    residual = expit(eta) - y
    gradient = np.asarray([
        np.sum(weights * residual * z),
        np.sum(weights * residual),
    ], dtype=np.float64)
    return objective, gradient


def fit_affine(z: np.ndarray, y: np.ndarray, name: str) -> tuple[np.ndarray, dict[str, object]]:
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    identity_objective, identity_gradient = affine_objective(np.asarray([1.0, 0.0]), z, y)
    started = time.perf_counter()
    result = minimize(
        fun=lambda theta: affine_objective(theta, z, y)[0],
        x0=np.asarray([1.0, 0.0], dtype=np.float64),
        jac=lambda theta: affine_objective(theta, z, y)[1],
        method="L-BFGS-B",
        bounds=((1e-6, None), (None, None)),
        options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-10, "maxls": 50},
    )
    elapsed = time.perf_counter() - started
    fitted_objective, gradient = affine_objective(np.asarray(result.x), z, y)
    a, b = map(float, result.x)
    at_bound = a <= 1e-6 + 1e-10
    kkt_a = gradient[0] >= -1e-5 if at_bound else abs(gradient[0]) <= 1e-5
    kkt_b = abs(gradient[1]) <= 1e-5
    checks = {
        "optimizer_success": bool(result.success),
        "parameters_finite": bool(np.isfinite(result.x).all()),
        "positive_slope_bound": bool(a >= 1e-6),
        "objective_finite": bool(np.isfinite(fitted_objective)),
        "objective_not_above_identity": bool(fitted_objective <= identity_objective + 1e-10),
        "kkt_a": bool(kkt_a),
        "kkt_b": bool(kkt_b),
    }
    if not all(checks.values()):
        raise RuntimeError(f"{name} affine calibration failed closed: {checks}, result={result}")
    record = {
        "arm": name,
        "model": "sigmoid(a*z_ref+b)",
        "n_rows": int(len(z)),
        "class_counts": np.bincount(y, minlength=2).tolist(),
        "input_logit_ordered_sha256": sha256_ordered(format(v, ".17g") for v in z),
        "target_ordered_sha256": sha256_ordered(y),
        "optimizer": "scipy.optimize.minimize L-BFGS-B",
        "optimizer_options": {"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-10, "maxls": 50},
        "fallback": "none",
        "a": a,
        "b": b,
        "identity_objective": identity_objective,
        "identity_gradient": identity_gradient.tolist(),
        "fitted_objective": fitted_objective,
        "gradient": gradient.tolist(),
        "at_lower_bound": bool(at_bound),
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "njev": int(result.njev),
        "status": int(result.status),
        "message": str(result.message),
        "success": bool(result.success),
        "elapsed_seconds": elapsed,
        "checks": checks,
    }
    return np.asarray([a, b]), record


def metrics_for(y: np.ndarray, groups: np.ndarray, probabilities: Mapping[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    y = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y, minlength=2)
    for arm in ARMS:
        p = np.asarray(probabilities[arm], dtype=np.float64)
        true_p = np.where(y == 1, p, 1.0 - p)
        loss = -np.log(np.clip(true_p, LOSS_CLIP, 1.0))
        contribution = np.where(y == 0, loss / (2.0 * counts[0]), loss / (2.0 * counts[1]))
        ce = class_equal_nll(y, p)
        if not np.isclose(contribution.sum(), ce, atol=1e-13, rtol=0):
            raise RuntimeError(f"per-case class-equal contributions do not sum for {arm}")
        metric_rows.append({
            "domain": DOMAIN, "repetition": REPETITION, "outer_fold": OUTER_FOLD,
            "arm": arm, "class_equal_NLL": ce,
            "AUC": float(roc_auc_score(y, p)),
            "natural_frequency_NLL": float(loss.mean()),
            "Brier": float(np.mean((p - y) ** 2)),
            "n_cases": int(len(y)), "n_groups": int(len(set(groups.astype(str)))),
        })
        for i in range(len(y)):
            case_rows.append({
                "domain": DOMAIN, "repetition": REPETITION, "outer_fold": OUTER_FOLD,
                "row_index": None, "case_id": None, "group_id": str(groups[i]),
                "y": int(y[i]), "arm": arm,
                "negative_log_likelihood": float(loss[i]),
                "squared_probability_error": float((p[i] - y[i]) ** 2),
                "class_equal_nll_contribution": float(contribution[i]),
            })
    return pd.DataFrame(metric_rows), pd.DataFrame(case_rows)


def _json_round_trip(value: object) -> object:
    """Return the exact JSON-domain value that the on-disk encoder will expose."""
    return json.loads(json.dumps(
        value, sort_keys=True, ensure_ascii=False, allow_nan=False, default=json_default,
    ))


def _assert_published_frame_equal(
    name: str,
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    schema: Sequence[str],
) -> None:
    """Compare a reloaded CSV with its pre-publication value, including row order."""
    if list(observed.columns) != list(schema) or list(expected.columns) != list(schema):
        raise RuntimeError(f"post-publication schema mismatch for {name}")
    if len(observed) != len(expected):
        raise RuntimeError(f"post-publication row-count mismatch for {name}")
    try:
        pd.testing.assert_frame_equal(
            observed.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-14,
            check_like=False,
        )
    except AssertionError as exc:
        raise RuntimeError(f"post-publication value/order mismatch for {name}: {exc}") from exc


def validate_published_unit(
    *,
    protocol: Mapping[str, object],
    static: Mapping[str, object],
    protocol_mtime_ns: int,
    expected_frames: Mapping[str, pd.DataFrame],
    expected_json: Mapping[str, object],
    expected_log: str,
    execution_manifest: Mapping[str, object],
    completion: Mapping[str, object],
    expected_rows: np.ndarray,
    expected_ids: np.ndarray,
    expected_groups: np.ndarray,
    expected_y: np.ndarray,
) -> dict[str, object]:
    """Reload the renamed unit and independently verify every published artifact."""
    started = time.perf_counter()
    checks: dict[str, bool] = {}
    if not FINAL_DIR.is_dir() or FINAL_DIR.is_symlink():
        raise RuntimeError("post-publication target is not a regular directory")
    observed_names = tuple(sorted(path.name for path in FINAL_DIR.iterdir()))
    checks["exact_14_artifact_set"] = observed_names == tuple(sorted(EXPECTED_OUTPUT_FILES))
    checks["all_14_artifacts_regular_non_symlink_files"] = all(
        (FINAL_DIR / name).is_file() and not (FINAL_DIR / name).is_symlink()
        for name in EXPECTED_OUTPUT_FILES
    )
    if not all(checks.values()):
        raise RuntimeError(f"post-publication artifact-set audit failed: {checks}")

    schemas = protocol["output_contract"]["schemas"]
    reloaded_frames: dict[str, pd.DataFrame] = {}
    for name in CSV_OUTPUTS:
        frame = pd.read_csv(FINAL_DIR / name)
        _assert_published_frame_equal(name, frame, expected_frames[name], schemas[name])
        reloaded_frames[name] = frame
    checks["all_csv_schemas_values_and_row_order_exact"] = True

    predictions = reloaded_frames["predictions.csv"]
    exact_identity = (
        np.array_equal(predictions["domain"].astype(str).to_numpy(), np.repeat(DOMAIN, len(expected_rows)))
        and np.array_equal(predictions["repetition"].to_numpy(dtype=np.int64), np.repeat(REPETITION, len(expected_rows)))
        and np.array_equal(predictions["outer_fold"].to_numpy(dtype=np.int64), np.repeat(OUTER_FOLD, len(expected_rows)))
        and np.array_equal(predictions["row_index"].to_numpy(dtype=np.int64), expected_rows)
        and np.array_equal(predictions["case_id"].astype(str).to_numpy(), expected_ids.astype(str))
        and np.array_equal(predictions["group_id"].astype(str).to_numpy(), expected_groups.astype(str))
        and np.array_equal(predictions["y"].to_numpy(dtype=np.int64), expected_y)
    )
    checks["prediction_scope_order_labels_exact"] = bool(exact_identity)
    published_probabilities = {
        arm: predictions[f"p_{arm}"].to_numpy(dtype=np.float64) for arm in ARMS
    }
    checks["published_probabilities_finite_and_bounded"] = all(
        np.isfinite(values).all() and np.all((values >= 0.0) & (values <= 1.0))
        for values in published_probabilities.values()
    )
    probability_logit_pairs = (("S", "logit_S"), ("C", "logit_C"),
                               ("SP", "logit_SP"), ("CP", "logit_CP"))
    checks["published_probability_logit_relations_exact"] = all(
        np.allclose(
            published_probabilities[arm],
            expit(predictions[column].to_numpy(dtype=np.float64)),
            atol=1e-14,
            rtol=1e-12,
        )
        for arm, column in probability_logit_pairs
    )

    # Recompute all derived case, metric, effect and rank outputs from the
    # reloaded prediction rows rather than trusting the staged objects.
    recomputed_metrics, recomputed_cases = metrics_for(
        expected_y, expected_groups, published_probabilities,
    )
    recomputed_cases["row_index"] = np.tile(expected_rows, len(ARMS))
    recomputed_cases["case_id"] = np.tile(expected_ids.astype(str), len(ARMS))
    recomputed_cases = recomputed_cases[list(schemas["per_case_effects.csv"])]
    recomputed_metrics = recomputed_metrics[list(schemas["metrics.csv"])]
    _assert_published_frame_equal(
        "per_case_effects.csv independent recomputation",
        reloaded_frames["per_case_effects.csv"], recomputed_cases,
        schemas["per_case_effects.csv"],
    )
    _assert_published_frame_equal(
        "metrics.csv independent recomputation",
        reloaded_frames["metrics.csv"], recomputed_metrics,
        schemas["metrics.csv"],
    )
    recomputed_lookup = recomputed_metrics.set_index("arm")
    recomputed_paired = pd.DataFrame([{
        "domain": DOMAIN,
        "repetition": REPETITION,
        "outer_fold": OUTER_FOLD,
        "contrast": name,
        "class_equal_NLL_X_minus_Y": float(
            recomputed_lookup.loc[x, "class_equal_NLL"]
            - recomputed_lookup.loc[yarm, "class_equal_NLL"]
        ),
        "AUC_X_minus_Y": float(
            recomputed_lookup.loc[x, "AUC"] - recomputed_lookup.loc[yarm, "AUC"]
        ),
        "favourable_sign_NLL": sign,
        "pilot_descriptive_only": True,
    } for name, x, yarm, sign in CONTRAST_SPECS])
    recomputed_paired = recomputed_paired[list(schemas["paired_effects.csv"])]
    _assert_published_frame_equal(
        "paired_effects.csv independent recomputation",
        reloaded_frames["paired_effects.csv"], recomputed_paired,
        schemas["paired_effects.csv"],
    )
    checks["per_case_metrics_and_paired_effects_independently_recomputed"] = True
    checks["published_C_S_rank_invariance"] = bool(
        np.array_equal(rankdata(published_probabilities["C"], method="average"),
                       rankdata(published_probabilities["S"], method="average"))
        and abs(float(roc_auc_score(expected_y, published_probabilities["C"]))
                - float(roc_auc_score(expected_y, published_probabilities["S"]))) <= 1e-12
    )
    checks["published_CP_SP_rank_invariance"] = bool(
        np.array_equal(rankdata(published_probabilities["CP"], method="average"),
                       rankdata(published_probabilities["SP"], method="average"))
        and abs(float(roc_auc_score(expected_y, published_probabilities["CP"]))
                - float(roc_auc_score(expected_y, published_probabilities["SP"]))) <= 1e-12
    )

    reloaded_json: dict[str, object] = {}
    for name in PRE_MANIFEST_JSON_OUTPUTS:
        reloaded_json[name] = json.loads((FINAL_DIR / name).read_text(encoding="utf-8"))
        if reloaded_json[name] != _json_round_trip(expected_json[name]):
            raise RuntimeError(f"post-publication JSON mismatch for {name}")
    checks["all_six_payload_JSON_files_reread_exact"] = True
    if (FINAL_DIR / "run.log").read_text(encoding="utf-8") != expected_log:
        raise RuntimeError("post-publication run.log mismatch")
    checks["run_log_reread_exact"] = True

    published_prior = reloaded_json["prior_lineage.json"]
    published_selections = reloaded_json["selections.json"]
    published_lineage = reloaded_json["lineage.json"]
    published_audit = reloaded_json["audit.json"]
    published_fit_cost = reloaded_json["fit_cost.json"]
    prior_records = published_prior.get("records", [])
    prior_counter = published_prior.get("transform_counter", {})
    role_records: dict[str, int] = {}
    for record in prior_records:
        role = str(record.get("score_role"))
        role_records[role] = role_records.get(role, 0) + 1
    checks["published_prior_17_calls_170_records_zero_fits"] = (
        int(published_prior.get("record_count", -1)) == 170
        and len(prior_records) == 170
        and int(prior_counter.get("calls", -1)) == 17
        and int(prior_counter.get("concept_records", -1)) == 170
        and int(prior_counter.get("model_fits", -1)) == 0
        and role_records == {
            "x_G_inside_D": 120,
            "z_H_and_raw_T_block_H": 40,
            "val_E": 10,
        }
        and all(record.get("assertions", {}).get("independent_empirical_prior_exact") is True
                for record in prior_records)
    )
    selection_validation = published_selections.get("validation", {})
    generated_validation = selection_validation.get("generated", {})
    archived_validation = selection_validation.get("archived", {})
    checks["published_generation_and_selection_validation_lineage"] = (
        bool(published_lineage.get("generation_validation"))
        and len(generated_validation) == 7
        and all(all(item.get("checks", {}).values()) for item in generated_validation.values())
        and len(archived_validation) == 2
        and all(all(item.get("checks", {}).values()) for item in archived_validation.values())
    )
    source_records = published_lineage.get("sources", {})
    checks["published_exact_six_outcome_whitelist_and_access_counts"] = (
        source_records.get("outcome_whitelist_labels") == list(OUTCOME_INPUT_SPECS)
        and len(source_records.get("outcome_whitelist_records", [])) == 6
        and source_records.get("outcome_whitelist_access_counts")
        == {label: 1 for label in OUTCOME_INPUT_SPECS}
    )
    checks["published_audit_all_true"] = (
        published_audit.get("all_mandatory_audits_pass") is True
        and all(published_audit.get("checks", {}).values())
    )
    checks["published_fit_cost_marks_prior_as_nontraining"] = (
        published_fit_cost.get("prior_restoration", {}).get("model_fits") == 0
        and published_fit_cost.get("prior_restoration", {}).get("operation_type")
        == "analytic_nontraining_transform"
    )

    reloaded_manifest = json.loads(
        (FINAL_DIR / "execution_manifest.json").read_text(encoding="utf-8")
    )
    if reloaded_manifest != _json_round_trip(execution_manifest):
        raise RuntimeError("post-publication execution_manifest mismatch")
    manifest_records = reloaded_manifest.get("artifacts", [])
    checks["manifest_has_canonical_12_nonself_artifacts"] = (
        [record.get("path") for record in manifest_records] == list(PRE_MANIFEST_FILES)
        and int(reloaded_manifest.get("artifact_count_excluding_manifest_and_COMPLETE", -1)) == 12
    )
    checks["manifest_hashes_and_sizes_match_published_files"] = all(
        (FINAL_DIR / str(record.get("path"))).is_file()
        and not (FINAL_DIR / str(record.get("path"))).is_symlink()
        and (FINAL_DIR / str(record.get("path"))).stat().st_size == int(record.get("bytes", -1))
        and sha256_file(FINAL_DIR / str(record.get("path"))) == record.get("sha256")
        for record in manifest_records
    )
    reloaded_complete = json.loads((FINAL_DIR / "COMPLETE.json").read_text(encoding="utf-8"))
    if reloaded_complete != _json_round_trip(completion):
        raise RuntimeError("post-publication COMPLETE mismatch")
    checks["COMPLETE_identity_protocol_and_manifest_anchor_exact"] = (
        reloaded_complete.get("status") == "COMPLETE"
        and reloaded_complete.get("domain") == DOMAIN
        and int(reloaded_complete.get("repetition", -1)) == REPETITION
        and int(reloaded_complete.get("outer_fold", -1)) == OUTER_FOLD
        and reloaded_complete.get("protocol_sha256") == static["protocol_sha256"]
        and reloaded_complete.get("execution_manifest_sha256")
        == sha256_file(FINAL_DIR / "execution_manifest.json")
    )
    checks["COMPLETE_written_no_earlier_than_manifest"] = (
        (FINAL_DIR / "COMPLETE.json").stat().st_mtime_ns
        >= (FINAL_DIR / "execution_manifest.json").stat().st_mtime_ns
    )
    checks["protocol_hash_and_mtime_still_frozen_after_rename"] = (
        sha256_file(PROTOCOL_PATH) == static["protocol_sha256"]
        and PROTOCOL_PATH.stat().st_mtime_ns == protocol_mtime_ns
    )
    checks["runner_hash_matches_manifest_after_rename"] = (
        sha256_file(Path(__file__)) == reloaded_manifest.get("runner_sha256")
    )
    if not all(checks.values()):
        failed = {key: value for key, value in checks.items() if not value}
        raise RuntimeError(f"post-publication validation failed closed: {failed}")
    published_records = [{
        "path": name,
        "bytes": (FINAL_DIR / name).stat().st_size,
        "sha256": sha256_file(FINAL_DIR / name),
    } for name in EXPECTED_OUTPUT_FILES]
    return {
        "status": "VALIDATED",
        "validation_scope": "runner post-rename reread; independent external audit remains required",
        "unit": {"domain": DOMAIN, "repetition": REPETITION, "outer_fold": OUTER_FOLD},
        "final_directory": str(FINAL_DIR.relative_to(ROOT)),
        "protocol_sha256": static["protocol_sha256"],
        "runner_sha256": sha256_file(Path(__file__)),
        "published_artifact_count": len(published_records),
        "published_artifacts": published_records,
        "checks": checks,
        "all_checks_pass": True,
        "validation_seconds": time.perf_counter() - started,
        "validated_at_utc": utc_now(),
    }


class OutcomeAssetGate:
    """The only code path permitted to open the six frozen outcome-bearing inputs."""

    def __init__(self, protocol: Mapping[str, object]) -> None:
        fixed = protocol.get("fixed_inputs")
        if not isinstance(fixed, dict):
            raise RuntimeError("fixed_inputs changed type")
        self._records: dict[str, dict[str, object]] = {}
        self._loads: dict[str, int] = {label: 0 for label in OUTCOME_INPUT_SPECS}
        root = ROOT.resolve(strict=True)
        resolved_seen: set[Path] = set()
        for label, (parent, key, expected_format) in OUTCOME_INPUT_SPECS.items():
            container: object = fixed if parent is None else fixed.get(parent)
            if not isinstance(container, dict) or not isinstance(container.get(key), dict):
                raise RuntimeError(f"whitelisted outcome record missing: {label}")
            frozen = dict(container[key])
            relative = frozen.get("path")
            expected_hash = frozen.get("sha256")
            if not isinstance(relative, str) or not isinstance(expected_hash, str):
                raise RuntimeError(f"malformed whitelisted outcome record: {label}")
            lexical = ROOT / relative
            if lexical.is_symlink() or not lexical.is_file():
                raise RuntimeError(f"whitelisted outcome must be a regular non-symlink file: {label}")
            resolved = lexical.resolve(strict=True)
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise RuntimeError(f"whitelisted outcome escapes repository root: {label}") from exc
            if resolved in resolved_seen:
                raise RuntimeError(f"two outcome labels resolve to one file: {label}")
            resolved_seen.add(resolved)
            observed_hash = sha256_file(resolved)
            if observed_hash != expected_hash:
                raise RuntimeError(f"frozen outcome hash mismatch: {label}")
            self._records[label] = {
                "label": label,
                "path": relative,
                "sha256": observed_hash,
                "bytes": resolved.stat().st_size,
                "format": expected_format,
                "resolved_path": resolved,
            }
        if set(self._records) != set(OUTCOME_INPUT_SPECS):
            raise RuntimeError("outcome whitelist construction was not exact")

    def _claim(self, label: str, expected_format: str) -> Path:
        if label not in self._records:
            raise RuntimeError(f"attempted non-whitelisted outcome access: {label}")
        record = self._records[label]
        if record["format"] != expected_format:
            raise RuntimeError(f"wrong loader requested for whitelisted outcome: {label}")
        self._loads[label] += 1
        if self._loads[label] != 1:
            raise RuntimeError(f"whitelisted outcome loaded more than once: {label}")
        return Path(record["resolved_path"])

    def load_npz(self, label: str) -> dict[str, np.ndarray]:
        path = self._claim(label, "npz")
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    def load_json(self, label: str) -> object:
        return json.loads(self._claim(label, "json").read_text(encoding="utf-8"))

    def load_csv(self, label: str) -> pd.DataFrame:
        return pd.read_csv(self._claim(label, "csv"))

    def assert_complete(self) -> None:
        if self._loads != {label: 1 for label in OUTCOME_INPUT_SPECS}:
            raise RuntimeError(f"outcome whitelist was not consumed exactly once: {self._loads}")

    def public_records(self) -> list[dict[str, object]]:
        return [
            {key: value for key, value in self._records[label].items() if key != "resolved_path"}
            for label in OUTCOME_INPUT_SPECS
        ]

    def access_counts(self) -> dict[str, int]:
        return dict(self._loads)


def verify_static_inputs(
    protocol: Mapping[str, object],
) -> tuple[dict[str, object], OutcomeAssetGate]:
    """Verify only explicit controls and the six whitelisted outcome assets."""
    protocol_hash = sha256_file(PROTOCOL_PATH)
    sidecar_tokens = PROTOCOL_SIDECAR.read_text(encoding="utf-8").strip().split()
    if not sidecar_tokens or protocol_hash != sidecar_tokens[0]:
        raise RuntimeError("P1 protocol sidecar hash mismatch")
    if protocol_hash != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("P1 protocol hash differs from the runner's frozen hash")

    fixed = protocol.get("fixed_inputs")
    if not isinstance(fixed, dict):
        raise RuntimeError("fixed_inputs changed type")
    control_records: list[dict[str, object]] = []
    for key in CONTROL_INPUT_KEYS:
        value = fixed.get(key)
        if not isinstance(value, dict):
            raise RuntimeError(f"required control record missing: {key}")
        relative = value.get("path")
        expected_hash = value.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected_hash, str):
            raise RuntimeError(f"malformed control record: {key}")
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"control input must be a regular non-symlink file: {key}")
        observed = sha256_file(path)
        if observed != expected_hash:
            raise RuntimeError(f"frozen control hash mismatch: {key}")
        control_records.append({
            "label": key, "path": relative, "sha256": observed, "bytes": path.stat().st_size,
        })

    outcome_gate = OutcomeAssetGate(protocol)
    outcome_records = outcome_gate.public_records()
    p0_record = fixed["p0_hash_manifest"]
    p0_path = ROOT / str(p0_record["path"])
    p0 = json.loads(p0_path.read_text(encoding="utf-8"))
    expected_aggregate = p0_record["frozen_result_aggregate_sha256"]
    if p0.get("frozen_result_aggregate_sha256") != expected_aggregate:
        raise RuntimeError("P0 frozen aggregate identifier changed")
    p0_files = {
        record["path"]: record
        for record in p0.get("frozen_result_files", [])
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    }
    archive_audits: list[dict[str, object]] = []
    for label in OUTCOME_INPUT_SPECS:
        if not label.startswith("pilot_archive."):
            continue
        observed = next(record for record in outcome_records if record["label"] == label)
        p0_match = p0_files.get(observed["path"])
        exact = (
            p0_match is not None
            and p0_match.get("sha256") == observed["sha256"]
            and int(p0_match.get("bytes", -1)) == int(observed["bytes"])
        )
        archive_audits.append({
            "label": label, "path": observed["path"], "p0_record_exact": bool(exact),
        })
    if len(archive_audits) != 5 or not all(item["p0_record_exact"] for item in archive_audits):
        raise RuntimeError("used strict-nested archive lacks an exact P0 record")
    static = {
        "protocol_sha256": protocol_hash,
        "protocol_sidecar_exact": True,
        "control_input_records": control_records,
        "outcome_whitelist_records": outcome_records,
        "outcome_whitelist_labels": list(OUTCOME_INPUT_SPECS),
        "p0_manifest_sha256": sha256_file(p0_path),
        "p0_frozen_result_aggregate_sha256": expected_aggregate,
        "used_archive_p0_records": archive_audits,
    }
    return static, outcome_gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", required=True)
    parser.add_argument("--domain", choices=("cbis",), required=True)
    parser.add_argument("--repetition", type=int, choices=(0,), required=True)
    parser.add_argument("--outer-fold", type=int, choices=(0,), required=True)
    args = parser.parse_args()
    if not (args.pilot and args.domain == DOMAIN and args.repetition == REPETITION and args.outer_fold == OUTER_FOLD):
        raise SystemExit("only the frozen cbis r0/f0 pilot invocation is legal")
    if FINAL_DIR.exists():
        raise RuntimeError(f"refusing to overwrite existing pilot: {FINAL_DIR}")
    if VALIDATION_RECEIPT.exists():
        raise RuntimeError(f"refusing to overwrite existing pilot validation receipt: {VALIDATION_RECEIPT}")

    protocol_mtime_ns = PROTOCOL_PATH.stat().st_mtime_ns
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_PRE_OUTCOME" or protocol["scope"]["pilot_unit"] != {
        "domain": DOMAIN,
        "repetition": REPETITION,
        "outer_fold": OUTER_FOLD,
        "reason_selected_before_p1_outcomes": "one legal development unit with an existing strict-nested panel and patient-grouped lineage",
    }:
        raise RuntimeError("frozen pilot identity changed")
    if tuple(protocol["diagnostic_head"]["C_grid"]) != GRID:
        raise RuntimeError("frozen C grid changed")

    run_started = time.perf_counter()
    log: list[str] = [f"{utc_now()} START frozen cbis r0/f0 P1 engineering pilot"]
    static, outcome_gate = verify_static_inputs(protocol)
    log.append(f"{utc_now()} verified frozen protocol and all exact input hashes")
    threadpool_snapshot = threadpool_info()
    thread_counts = [record.get("num_threads") for record in threadpool_snapshot
                     if isinstance(record.get("num_threads"), int)]
    threadpool_exactly_one = bool(thread_counts) and all(count == 1 for count in thread_counts)
    if not threadpool_exactly_one:
        raise RuntimeError(f"diagnosis-head threadpool is not fixed at one: {threadpool_snapshot}")
    if (sha256_file(PROTOCOL_PATH) != static["protocol_sha256"] or
            PROTOCOL_PATH.stat().st_mtime_ns != protocol_mtime_ns):
        raise RuntimeError("protocol changed immediately before outcome loading")
    log.append(f"{utc_now()} verified one-thread execution and immutable protocol before outcome loading")

    load_started = time.perf_counter()
    panel = outcome_gate.load_npz("pilot_archive.panel")
    generation_object = outcome_gate.load_json("pilot_archive.generation")
    archived_predictions = outcome_gate.load_csv("pilot_archive.predictions")
    archived_selections = outcome_gate.load_json("pilot_archive.selection")
    archived_heads = outcome_gate.load_json("pilot_archive.heads")
    cohort = outcome_gate.load_npz("cbis_development_cohort")
    outcome_gate.assert_complete()
    if not isinstance(generation_object, dict):
        raise RuntimeError("generation outcome asset changed type")
    generation = generation_object
    if not isinstance(archived_selections, list) or not isinstance(archived_heads, dict):
        raise RuntimeError("archived selection/head outcome assets changed type")
    static["outcome_whitelist_access_counts"] = outcome_gate.access_counts()
    eligible = cohort["complete_concept_eligible"].astype(bool)
    source_names = cohort["concept_names"].astype(str)
    source_index = [int(np.flatnonzero(source_names == name)[0]) for name in cb.STABLE_CONCEPTS]
    source_ids = cohort["unit_id"].astype(str)[eligible]
    source_groups = cohort["patient_id"].astype(str)[eligible]
    source_y = cohort["outcome_id"].astype(np.int64)[eligible]
    source_fold = cohort["outer_fold"].astype(np.int64)[eligible]
    source_gold = cohort["concept_multi_hot"].astype(np.float64)[eligible][:, source_index]
    family_valid = cohort["concept_family_valid"].astype(bool)[eligible]
    source_valid = np.column_stack((
        np.repeat(family_valid[:, [0]], 5, axis=1),
        np.repeat(family_valid[:, [1]], 5, axis=1),
    ))
    load_seconds = time.perf_counter() - load_started

    y = panel["y"].astype(np.int64)
    ids = panel["ids"].astype(str)
    groups = panel["groups"].astype(str)
    gold = panel["gold"].astype(np.float64)
    t = panel["t"].astype(np.int64)
    e = panel["e"].astype(np.int64)
    raw = panel["raw"].astype(np.float64)
    val = panel["val"].astype(np.float64)
    if panel["widths"].size != 0:
        raise RuntimeError("pilot is no longer a binary-concept CBIS panel")
    source_checks = {
        "eligible_rows_655": len(source_ids) == 655,
        "panel_ids_equal_source": np.array_equal(ids, source_ids),
        "panel_groups_equal_source": np.array_equal(groups, source_groups),
        "panel_y_equal_source": np.array_equal(y, source_y),
        "panel_gold_equal_source": np.array_equal(gold, source_gold),
        "all_stable_concepts_valid": bool(source_valid.all()),
        "T_equals_nonouter0": np.array_equal(t, np.flatnonzero(source_fold != OUTER_FOLD)),
        "E_equals_outer0": np.array_equal(e, np.flatnonzero(source_fold == OUTER_FOLD)),
        "T_E_patient_disjoint": disjoint(groups[t], groups[e]),
    }
    if not all(source_checks.values()):
        raise RuntimeError(f"source/panel alignment failed: {source_checks}")
    resolved_h_manifests, resolved_final_manifest, generation_validation = (
        validate_generation_collection(generation, panel, source_fold)
    )

    fit_counter: dict[str, object] = {
        "total_head_fits": 0,
        "total_head_fit_seconds": 0.0,
        "by_role": {},
    }
    selection_outputs: dict[str, object] = {}
    selection_validations: dict[str, object] = {}
    prior_records: list[dict[str, object]] = []
    prior_verification_specs: list[dict[str, object]] = []
    prior_transform_counter: dict[str, object] = {
        "calls": 0,
        "concept_records": 0,
        "probability_values": 0,
        "model_fits": 0,
        "seconds": 0.0,
        "by_role": {},
    }
    partition_records: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    restored_raw = np.full_like(raw, np.nan)
    raw_stitched = np.full_like(raw, np.nan)
    calibration_raw = np.full(len(t), np.nan, dtype=np.float64)
    calibration_prior = np.full(len(t), np.nan, dtype=np.float64)
    calibration_seen = np.zeros(len(t), dtype=np.int64)
    t_lookup = {int(row): pos for pos, row in enumerate(t)}
    h_reference_heads: list[cb.BalancedBinaryLogistic] = []
    h_reference_c: list[float] = []
    h_block_data: list[dict[str, object]] = []

    selection_started = time.perf_counter()
    outer_reference_blocks = []
    m_blocks_raw = []
    m_blocks_prior = []
    h_concat: list[int] = []
    h_group_sets: list[set[str]] = []

    for h_index in range(4):
        d = panel[f"f{h_index}"].astype(np.int64)
        hrows = panel[f"h{h_index}"].astype(np.int64)
        x_raw = panel[f"x{h_index}"].astype(np.float64)
        z_raw = panel[f"z{h_index}"].astype(np.float64)
        manifest = resolved_h_manifests[h_index]
        h_fold = int(manifest["diagnosis_selection_heldout_fold"])
        if not np.array_equal(hrows, np.flatnonzero(source_fold == h_fold)):
            raise RuntimeError("H rows no longer equal their original patient fold")
        if not np.array_equal(d, np.setdiff1d(t, hrows, assume_unique=True)):
            raise RuntimeError("D is not exactly T minus H")
        if not disjoint(groups[d], groups[hrows]) or not disjoint(groups[d], groups[e]) or not disjoint(groups[hrows], groups[e]):
            raise RuntimeError("patient leakage in D/H/E")
        h_generator_checks = {
            "generation_fit_patient_sha256_exact": (
                str(manifest["fit_patient_sha256"]) == cb.sha256_values(groups[d].astype(str))
            ),
            "generation_heldout_patient_sha256_exact": (
                str(manifest["heldout_patient_sha256"]) == cb.sha256_values(groups[hrows].astype(str))
            ),
        }
        if not all(h_generator_checks.values()):
            raise RuntimeError(f"H generator patient-scope hashes changed: {h_generator_checks}")
        h_concat.extend(hrows.tolist())
        h_group_sets.append(set(groups[hrows].astype(str)))

        nested = manifest["nested_probe_panels"]
        if len(nested) != 3:
            raise RuntimeError("CBIS D must contain exactly three G folds")
        x_restored = np.full_like(x_raw, np.nan)
        g_seen = np.zeros(len(d), dtype=np.int64)
        d_lookup = {int(row): pos for pos, row in enumerate(d)}
        h_reference_blocks = []
        g_records = []
        for g_meta in nested:
            g_fold = int(g_meta["probe_heldout_fold"])
            grows = np.flatnonzero(source_fold == g_fold)
            dg = np.setdiff1d(d, grows, assume_unique=True)
            if not np.array_equal(np.sort(np.r_[dg, grows]), np.sort(d)):
                raise RuntimeError("G and D\\G do not partition D")
            if not disjoint(groups[dg], groups[grows]):
                raise RuntimeError("patient leakage in D\\G/G")
            if not bool(g_meta["diagnosis_validation_excluded_from_probe_fit"]) or not bool(g_meta["probe_validation_excluded_from_probe_fit"]):
                raise RuntimeError("frozen deeper-probe exclusion assertion failed")
            local_positions = np.asarray([d_lookup[int(row)] for row in grows], dtype=np.int64)
            g_seen[local_positions] += 1
            g_generator_checks = {
                "probe_fit_patient_sha256_exact": (
                    str(g_meta["probe_fit_patient_sha256"]) == cb.sha256_values(groups[dg].astype(str))
                ),
                "probe_validation_patient_sha256_exact": (
                    str(g_meta["probe_validation_patient_sha256"]) == cb.sha256_values(groups[grows].astype(str))
                ),
            }
            if not all(g_generator_checks.values()):
                raise RuntimeError(f"G generator patient-scope hashes changed: {g_generator_checks}")
            restored_g = restore_cbis(
                scores=x_raw[local_positions], fit_rows=dg, predict_rows=grows,
                gold=gold, valid=source_valid, ids=ids, groups=groups,
                frozen_concept_fits=concept_fit_records(g_meta, "concept_fits"),
                score_role="x_G_inside_D", h_fold=h_fold, g_fold=g_fold,
                lineage_records=prior_records,
                transform_counter=prior_transform_counter,
                verification_specs=prior_verification_specs,
            )
            x_restored[local_positions] = restored_g
            h_reference_blocks.append((dg, grows, gold[dg], gold[grows]))
            g_records.append({
                "G_original_fold": g_fold,
                "D_minus_G": scope_record(dg, ids, groups),
                "G": scope_record(grows, ids, groups),
                **g_generator_checks,
                "patient_disjoint": disjoint(groups[dg], groups[grows]),
            })
        if not np.all(g_seen == 1) or not np.isfinite(x_restored).all():
            raise RuntimeError("G blocks do not stitch D exactly once")

        z_restored = restore_cbis(
            scores=z_raw, fit_rows=d, predict_rows=hrows, gold=gold, valid=source_valid,
            ids=ids, groups=groups,
            frozen_concept_fits=concept_fit_records(manifest, "full_meta_fit_concept_fits"),
            score_role="z_H_and_raw_T_block_H", h_fold=h_fold, g_fold=None,
            lineage_records=prior_records,
            transform_counter=prior_transform_counter,
            verification_specs=prior_verification_specs,
        )
        raw_positions = np.asarray([t_lookup[int(row)] for row in hrows], dtype=np.int64)
        raw_stitched[raw_positions] = z_raw
        restored_raw[raw_positions] = z_restored

        c_h, h_sel = select_with_rows(
            blocks=h_reference_blocks, y=y, fit_counter=fit_counter,
            role="H_specific_reference_selection", ids=ids, groups=groups,
        )
        h_sel.update({"H_original_fold": h_fold, "D": scope_record(d, ids, groups),
                      "H": scope_record(hrows, ids, groups)})
        selection_validations[f"H_specific_reference_fold_{h_fold}"] = (
            validate_generated_selection(
                h_sel, expected_folds=3,
                role=f"H_specific_reference_fold_{h_fold}",
            )
        )
        selection_outputs.setdefault("H_specific_reference", []).append(h_sel)
        h_head = fit_head(gold[d], y[d], c_h, fit_counter, "H_specific_reference_final")
        h_reference_heads.append(h_head)
        h_reference_c.append(c_h)
        z_logit = head_logit(h_head, z_raw)
        zp_logit = head_logit(h_head, z_restored)
        calibration_raw[raw_positions] = z_logit
        calibration_prior[raw_positions] = zp_logit
        calibration_seen[raw_positions] += 1
        fit_scope_hashes = {str(record["fit_unit_sha256"]) for record in concept_fit_records(manifest, "full_meta_fit_concept_fits")}
        if len(fit_scope_hashes) != 1:
            raise RuntimeError("all-valid CBIS concepts unexpectedly have different D scopes")
        probe_scope_hash = next(iter(fit_scope_hashes))
        reference_scope_hash = cb.sha256_values(ids[d].astype(str))
        if probe_scope_hash != reference_scope_hash:
            raise RuntimeError("probe and reference D scope hashes differ")
        for row, lr, lp in zip(hrows, z_logit, zp_logit):
            calibration_rows.append({
                "domain": DOMAIN, "repetition": REPETITION, "outer_fold": OUTER_FOLD,
                "row_index": int(row), "case_id": str(ids[row]), "group_id": str(groups[row]),
                "y": int(y[row]), "H": h_fold, "reference_C_H": c_h,
                "raw_base_logit": float(lr), "prior_restored_base_logit": float(lp),
                "probe_fit_scope_sha256": probe_scope_hash,
                "reference_fit_scope_sha256": reference_scope_hash,
            })
        outer_reference_blocks.append((d, hrows, gold[d], gold[hrows]))
        m_blocks_raw.append((d, hrows, x_raw, z_raw))
        m_blocks_prior.append((d, hrows, x_restored, z_restored))
        partition_records.append({
            "H_index": h_index, "H_original_fold": h_fold,
            "D": scope_record(d, ids, groups), "H": scope_record(hrows, ids, groups),
            "D_H_patient_disjoint": disjoint(groups[d], groups[hrows]),
            "E_excluded_from_D_and_H": not np.intersect1d(e, np.r_[d, hrows]).size,
            **h_generator_checks,
            "G_partition_count": 3,
            "G_partitions_D_exactly_once": bool(np.all(g_seen == 1)),
            "G_blocks": g_records,
        })
        h_block_data.append({"D": d, "H": hrows, "H_original_fold": h_fold,
                             "x_raw": x_raw, "z_raw": z_raw,
                             "x_prior": x_restored, "z_prior": z_restored})

    if not np.array_equal(np.sort(np.asarray(h_concat)), np.sort(t)) or len(h_concat) != len(set(h_concat)):
        raise RuntimeError("H blocks do not partition T exactly once")
    if any(h_group_sets[i].intersection(h_group_sets[j]) for i in range(4) for j in range(i + 1, 4)):
        raise RuntimeError("a patient group crosses H blocks")
    if not np.array_equal(raw_stitched, raw):
        raise RuntimeError("stitched frozen z_H scores do not exactly reproduce raw_T")
    if not np.all(calibration_seen == 1) or not np.isfinite(calibration_raw).all() or not np.isfinite(calibration_prior).all():
        raise RuntimeError("calibration OOF logits do not cover T exactly once")

    reference_c, outer_ref_sel = select_with_rows(
        blocks=outer_reference_blocks, y=y, fit_counter=fit_counter,
        role="outer_reference_selection", ids=ids, groups=groups,
    )
    selection_outputs["outer_reference"] = outer_ref_sel
    selection_validations["outer_reference"] = validate_generated_selection(
        outer_ref_sel, expected_folds=4, role="outer_reference"
    )
    m_c, m_sel = select_with_rows(
        blocks=m_blocks_raw, y=y, fit_counter=fit_counter,
        role="M_selection", ids=ids, groups=groups,
    )
    selection_outputs["M"] = m_sel
    selection_validations["M"] = validate_generated_selection(
        m_sel, expected_folds=4, role="M"
    )
    mp_c, mp_sel = select_with_rows(
        blocks=m_blocks_prior, y=y, fit_counter=fit_counter,
        role="MP_selection", ids=ids, groups=groups,
    )
    selection_outputs["MP"] = mp_sel
    selection_validations["MP"] = validate_generated_selection(
        mp_sel, expected_folds=4, role="MP"
    )
    selection_seconds = time.perf_counter() - selection_started

    final_manifest = resolved_final_manifest
    if int(final_manifest["outer_fold"]) != OUTER_FOLD or str(final_manifest["role"]) != "final_outer_train_probe":
        raise RuntimeError("final concept predictor manifest changed")
    final_probe_checks = {
        "fit_patient_sha256_exact": (
            str(final_manifest["fit_patient_sha256"]) == cb.sha256_values(groups[t].astype(str))
        ),
        "validation_patient_sha256_exact": (
            str(final_manifest["validation_patient_sha256"]) == cb.sha256_values(groups[e].astype(str))
        ),
        "patient_disjoint": disjoint(groups[t], groups[e]),
    }
    if not all(final_probe_checks.values()):
        raise RuntimeError(f"final probe patient-scope lineage changed: {final_probe_checks}")
    val_prior = restore_cbis(
        scores=val, fit_rows=t, predict_rows=e, gold=gold, valid=source_valid,
        ids=ids, groups=groups,
        frozen_concept_fits=concept_fit_records(final_manifest, "concept_fits"),
        score_role="val_E", h_fold=None, g_fold=None,
        lineage_records=prior_records,
        transform_counter=prior_transform_counter,
        verification_specs=prior_verification_specs,
    )
    prior_independent_verification = verify_prior_restoration_independently(
        specifications=prior_verification_specs,
        lineage_records=prior_records,
        gold=gold,
        valid=source_valid,
        ids=ids,
        groups=groups,
        transform_counter=prior_transform_counter,
    )

    final_started = time.perf_counter()
    reference_head = fit_head(gold[t], y[t], reference_c, fit_counter, "outer_reference_final")
    m_head = fit_head(raw, y[t], m_c, fit_counter, "M_final")
    mp_head = fit_head(restored_raw, y[t], mp_c, fit_counter, "MP_final")
    p_r = reference_head.predict_proba(gold[e])[:, 1]
    logit_s = head_logit(reference_head, val)
    p_s = expit(logit_s)
    logit_sp = head_logit(reference_head, val_prior)
    p_sp = expit(logit_sp)
    p_m = m_head.predict_proba(val)[:, 1]
    p_mp = mp_head.predict_proba(val_prior)[:, 1]
    final_head_seconds = time.perf_counter() - final_started

    affine_started = time.perf_counter()
    theta_c, calibrator_c = fit_affine(calibration_raw, y[t], "C")
    theta_cp, calibrator_cp = fit_affine(calibration_prior, y[t], "CP")
    logit_c = theta_c[0] * logit_s + theta_c[1]
    logit_cp = theta_cp[0] * logit_sp + theta_cp[1]
    p_c = expit(logit_c)
    p_cp = expit(logit_cp)
    affine_seconds = time.perf_counter() - affine_started
    probabilities = {"R": p_r, "S": p_s, "C": p_c, "M": p_m,
                     "SP": p_sp, "CP": p_cp, "MP": p_mp}

    metrics_started = time.perf_counter()
    metrics_df, cases_df = metrics_for(y[e], groups[e], probabilities)
    cases_df["row_index"] = np.tile(e, len(ARMS))
    cases_df["case_id"] = np.tile(ids[e].astype(str), len(ARMS))
    metric_lookup = metrics_df.set_index("arm")
    paired_df = pd.DataFrame([{
        "domain": DOMAIN, "repetition": REPETITION, "outer_fold": OUTER_FOLD,
        "contrast": name,
        "class_equal_NLL_X_minus_Y": float(metric_lookup.loc[x, "class_equal_NLL"] - metric_lookup.loc[yarm, "class_equal_NLL"]),
        "AUC_X_minus_Y": float(metric_lookup.loc[x, "AUC"] - metric_lookup.loc[yarm, "AUC"]),
        "favourable_sign_NLL": sign,
        "pilot_descriptive_only": True,
    } for name, x, yarm, sign in CONTRAST_SPECS])
    metrics_seconds = time.perf_counter() - metrics_started

    old = archived_predictions
    archive_alignment_checks: dict[str, bool] = {}
    def archived_arm(condition: str) -> tuple[np.ndarray, np.ndarray]:
        rows = old[(old["domain"] == DOMAIN) & (old["repetition"] == REPETITION) &
                   (old["outer_fold"] == OUTER_FOLD) & (old["condition"] == condition)]
        checks = {
            "case_id_order_exact": np.array_equal(rows["unit_id"].astype(str).to_numpy(), ids[e].astype(str)),
            "group_id_order_exact": np.array_equal(rows["group_id"].astype(str).to_numpy(), groups[e].astype(str)),
            "label_order_exact": np.array_equal(rows["label"].to_numpy(dtype=np.int64), y[e]),
            "row_count_exact": len(rows) == len(e),
        }
        archive_alignment_checks.update({f"{condition}:{key}": value for key, value in checks.items()})
        if not all(checks.values()):
            raise RuntimeError(f"archived row alignment changed for {condition}: {checks}")
        return rows["p1"].to_numpy(dtype=float), rows["C"].to_numpy(dtype=float)
    old_r, old_r_c = archived_arm("raw_gold_nested")
    old_s, old_s_c = archived_arm("raw_gold_substitution")
    old_m, old_m_c = archived_arm("raw_soft_nested")
    archived_selected_m = float(archived_heads["raw_soft_nested"]["C"])
    archived_selected_ref = float(archived_heads["raw_gold_nested"]["C"])
    archived_selection_m_rows, archived_selection_m_validation = validate_archived_selection(
        archived_selections, "raw_soft_nested", archived_selected_m
    )
    archived_selection_ref_rows, archived_selection_ref_validation = validate_archived_selection(
        archived_selections, "raw_gold_nested", archived_selected_ref
    )

    invariance = {
        "C_vs_S_rank_identical": bool(np.array_equal(rankdata(p_c, method="average"), rankdata(p_s, method="average"))),
        "C_vs_S_AUC_absolute_delta": abs(float(roc_auc_score(y[e], p_c) - roc_auc_score(y[e], p_s))),
        "CP_vs_SP_rank_identical": bool(np.array_equal(rankdata(p_cp, method="average"), rankdata(p_sp, method="average"))),
        "CP_vs_SP_AUC_absolute_delta": abs(float(roc_auc_score(y[e], p_cp) - roc_auc_score(y[e], p_sp))),
    }
    expected_generated_roles = {
        f"H_specific_reference_fold_{record['H_original_fold']}"
        for record in partition_records
    }
    expected_generated_roles.update(("outer_reference", "M", "MP"))
    generation_validation_exact = (
        int(generation_validation["manifest_count"]) == 5
        and int(generation_validation["H_manifest_count"]) == 4
        and len(generation_validation["H_original_folds_in_panel_order"]) == 4
        and len(set(generation_validation["H_original_folds_in_panel_order"])) == 4
        and generation_validation["selection_manifest_block_sizes"] == [1, 1, 1, 1]
        and generation_validation["each_H_has_one_identical_selection_copy"] is True
        and int(generation_validation["final_manifest_index"]) == 4
        and int(generation_validation["probe_bundles"]) == 11
        and all(generation_validation["final_checks"].values())
    )
    generated_selection_validation_exact = (
        len(selection_validations) == 7
        and set(selection_validations) == expected_generated_roles
        and all(
            all(item["checks"].values())
            for item in selection_validations.values()
        )
    )
    archived_selection_validation_exact = (
        all(archived_selection_m_validation["checks"].values())
        and all(archived_selection_ref_validation["checks"].values())
    )
    prior_transform_counter_exact = (
        int(prior_transform_counter["calls"]) == 17
        and int(prior_transform_counter["concept_records"]) == 170
        and int(prior_transform_counter["model_fits"]) == 0
        and int(prior_transform_counter["probability_values"])
        == int(prior_independent_verification["probability_values"])
        and np.isfinite(float(prior_transform_counter["seconds"]))
        and float(prior_transform_counter["seconds"]) >= 0.0
    )
    prior_independent_exact = all(
        prior_independent_verification["checks"].values()
    )
    audit_started = time.perf_counter()
    audit_checks = {
        "input_hashes_equal_frozen_protocol": True,
        "used_archive_files_have_exact_P0_records": all(x["p0_record_exact"] for x in static["used_archive_p0_records"]),
        "outcome_asset_whitelist_exactly_six": (
            static["outcome_whitelist_labels"] == list(OUTCOME_INPUT_SPECS)
            and len(static["outcome_whitelist_records"]) == 6
            and static["outcome_whitelist_access_counts"]
            == {label: 1 for label in OUTCOME_INPUT_SPECS}
        ),
        "generation_manifest_collection_validated": generation_validation_exact,
        "all_seven_generated_selection_curves_validated": generated_selection_validation_exact,
        "two_archived_selection_curves_validated": archived_selection_validation_exact,
        "prior_restoration_exactly_17_calls_170_records_zero_fits": prior_transform_counter_exact,
        "prior_restoration_independent_recomputation_passed": prior_independent_exact,
        "pilot_invocation_exactly_cbis_r0_f0": True,
        "source_panel_alignment_exact": all(source_checks.values()),
        "threadpool_exactly_one": threadpool_exactly_one,
        "H_blocks_partition_T_exactly_once": np.array_equal(np.sort(np.asarray(h_concat)), np.sort(t)) and len(h_concat) == len(set(h_concat)),
        "H_patient_groups_do_not_cross_blocks": not any(h_group_sets[i].intersection(h_group_sets[j]) for i in range(4) for j in range(i + 1, 4)),
        "every_H_has_three_G_blocks_partitioning_D": all(
            len(r["G_blocks"]) == 3 and r["G_partitions_D_exactly_once"] and
            all(g["patient_disjoint"] for g in r["G_blocks"])
            for r in partition_records
        ),
        "all_H_generator_patient_scope_hashes_exact": all(
            r["generation_fit_patient_sha256_exact"] and
            r["generation_heldout_patient_sha256_exact"]
            for r in partition_records
        ),
        "all_G_generator_patient_scope_hashes_exact": all(
            g["probe_fit_patient_sha256_exact"] and
            g["probe_validation_patient_sha256_exact"]
            for r in partition_records for g in r["G_blocks"]
        ),
        "final_probe_patient_scope_hashes_and_disjointness_exact": all(final_probe_checks.values()),
        "all_E_rows_excluded_from_fits_and_selection": not np.intersect1d(e, t).size,
        "all_H_rows_excluded_from_generating_probe_and_reference_fit": all(r["D_H_patient_disjoint"] for r in partition_records),
        "all_G_rows_excluded_from_generating_probe_fit": all(all(g["patient_disjoint"] for g in r["G_blocks"]) for r in partition_records),
        "CBIS_patient_groups_disjoint_where_held_out": bool(source_checks["T_E_patient_disjoint"]) and all(r["D_H_patient_disjoint"] for r in partition_records),
        "every_prior_scope_hash_matches_frozen_generating_probe": all(all(v for v in r["assertions"].values()) for r in prior_records),
        "every_prior_independently_recomputed": all(r["assertions"]["independent_empirical_prior_exact"] for r in prior_records),
        "C_and_CP_calibration_OOF_cover_T_exactly_once_in_T_order": bool(np.all(calibration_seen == 1)),
        "C_E_base_logits_exactly_S_and_CP_E_base_logits_exactly_SP": np.array_equal(expit(logit_s), p_s) and np.array_equal(expit(logit_sp), p_sp),
        "all_seven_arms_aligned": all(np.asarray(probabilities[a]).shape == (len(e),) for a in ARMS),
        "all_seven_arms_one_finite_probability_per_E_row": all(np.isfinite(probabilities[a]).all() and np.all((probabilities[a] >= 0) & (probabilities[a] <= 1)) for a in ARMS),
        "archived_case_group_label_order_exact": all(archive_alignment_checks.values()),
        "R_reproduces_archived": bool(np.allclose(p_r, old_r, atol=1e-10, rtol=0)),
        "S_reproduces_archived": bool(np.allclose(p_s, old_s, atol=1e-10, rtol=0)),
        "M_reproduces_archived": bool(np.allclose(p_m, old_m, atol=1e-10, rtol=0)),
        "M_selected_C_reproduces_archived": m_c == archived_selected_m and np.all(old_m_c == m_c),
        "reference_selected_C_reproduces_archived": reference_c == archived_selected_ref and np.all(old_r_c == reference_c) and np.all(old_s_c == reference_c),
        "archived_selection_curves_reproduced_M": all(np.allclose(m_sel["candidates"][i]["fold_class_equal_NLL"], row["fold_losses"], atol=1e-13, rtol=0) for i, row in enumerate(archived_selection_m_rows)),
        "archived_selection_curves_reproduced_reference": all(np.allclose(outer_ref_sel["candidates"][i]["fold_class_equal_NLL"], row["fold_losses"], atol=1e-13, rtol=0) for i, row in enumerate(archived_selection_ref_rows)),
        "C_positive_affine_audits_pass": all(calibrator_c["checks"].values()),
        "CP_positive_affine_audits_pass": all(calibrator_cp["checks"].values()),
        "C_vs_S_rank_and_AUC_invariant_within_unit": invariance["C_vs_S_rank_identical"] and invariance["C_vs_S_AUC_absolute_delta"] <= 1e-12,
        "CP_vs_SP_rank_and_AUC_invariant_within_unit": invariance["CP_vs_SP_rank_identical"] and invariance["CP_vs_SP_AUC_absolute_delta"] <= 1e-12,
        "no_bootstrap_run_in_pilot": True,
        "no_nonpilot_unit_run": True,
        "no_official_test_asset_access_or_discovery": True,
        "protocol_hash_and_mtime_unchanged_during_computation": sha256_file(PROTOCOL_PATH) == static["protocol_sha256"] and PROTOCOL_PATH.stat().st_mtime_ns == protocol_mtime_ns,
        "expected_head_fit_count_151": int(fit_counter["total_head_fits"]) == 151,
        "exactly_two_affine_fits": True,
    }
    if not all(audit_checks.values()):
        failed = {key: value for key, value in audit_checks.items() if not value}
        raise RuntimeError(f"mandatory pilot audit failed: {failed}")
    audit_seconds = time.perf_counter() - audit_started

    predictions_df = pd.DataFrame({
        "domain": DOMAIN, "repetition": REPETITION, "outer_fold": OUTER_FOLD,
        "row_index": e, "case_id": ids[e].astype(str), "group_id": groups[e].astype(str),
        "y": y[e], "p_R": p_r, "p_S": p_s, "p_C": p_c, "p_M": p_m,
        "p_SP": p_sp, "p_CP": p_cp, "p_MP": p_mp,
        "logit_S": logit_s, "logit_C": logit_c,
        "logit_SP": logit_sp, "logit_CP": logit_cp,
    })
    calibration_df = pd.DataFrame(calibration_rows).set_index("row_index").loc[t].reset_index()
    calibration_df = calibration_df[[
        "domain", "repetition", "outer_fold", "row_index", "case_id", "group_id", "y", "H",
        "reference_C_H", "raw_base_logit", "prior_restored_base_logit",
        "probe_fit_scope_sha256", "reference_fit_scope_sha256",
    ]]
    prediction_columns = protocol["output_contract"]["schemas"]["predictions.csv"]
    calibration_columns = protocol["output_contract"]["schemas"]["calibration_oof.csv"]
    case_columns = protocol["output_contract"]["schemas"]["per_case_effects.csv"]
    metric_columns = protocol["output_contract"]["schemas"]["metrics.csv"]
    paired_columns = protocol["output_contract"]["schemas"]["paired_effects.csv"]
    for frame, columns, name in (
        (predictions_df, prediction_columns, "predictions.csv"),
        (calibration_df, calibration_columns, "calibration_oof.csv"),
        (cases_df, case_columns, "per_case_effects.csv"),
        (metrics_df, metric_columns, "metrics.csv"),
        (paired_df, paired_columns, "paired_effects.csv"),
    ):
        if list(frame.columns) != list(columns):
            raise RuntimeError(f"{name} schema mismatch: {frame.columns.tolist()}")

    lineage = {
        "unit": {"domain": DOMAIN, "repetition": REPETITION, "outer_fold": OUTER_FOLD,
                 "interpretation": "engineering pilot only; not a P1 scientific conclusion"},
        "sources": static,
        "cohort_source_checks": source_checks,
        "T": scope_record(t, ids, groups), "E": scope_record(e, ids, groups),
        "H_D_G": partition_records,
        "final_probe": {
            "fit": scope_record(t, ids, groups), "evaluation": scope_record(e, ids, groups),
            **final_probe_checks,
        },
        "arm_alignment": {
            "arms": list(ARMS), "row_count_each": int(len(e)),
            "row_index_ordered_sha256": sha256_ordered(e),
            "case_id_ordered_sha256": sha256_ordered(ids[e].astype(str)),
            "group_id_ordered_sha256": sha256_ordered(groups[e].astype(str)),
            "y_ordered_sha256": sha256_ordered(y[e]),
            "exact": True,
        },
        "software": {
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl")},
            "threadpool_snapshot_during_run": threadpool_snapshot,
            "threadpool_exactly_one": threadpool_exactly_one,
        },
        "generation_validation": generation_validation,
        "access_boundary": {
            "allowed_control_paths": [
                r["path"] for r in static["control_input_records"]
            ],
            "allowed_outcome_paths": [
                r["path"] for r in static["outcome_whitelist_records"]
            ],
            "outcome_asset_count": 6,
            "outcome_asset_access_counts": static["outcome_whitelist_access_counts"],
            "official_test_access": False,
            "official_test_discovery_or_listing": False,
        },
    }
    selection_outputs["unit"] = {"domain": DOMAIN, "repetition": REPETITION, "outer_fold": OUTER_FOLD}
    selection_outputs["tie_break_rule"] = "first minimum in ascending C grid"
    selection_outputs["archived_reproduction"] = {
        "M_selected_C": archived_selected_m, "reference_selected_C": archived_selected_ref,
    }
    selection_outputs["validation"] = {
        "generation_collection": generation_validation,
        "generated": selection_validations,
        "archived": {
            "raw_soft_nested": archived_selection_m_validation,
            "raw_gold_nested": archived_selection_ref_validation,
        },
    }
    calibrators = {"C": calibrator_c, "CP": calibrator_cp, "within_unit_invariance": invariance}
    prior_lineage = {
        "rule": protocol["concept_score_rules"],
        "operation_type": "analytic_nontraining_transform",
        "record_count": len(prior_records),
        "records": prior_records,
        "transform_counter": prior_transform_counter,
        "independent_verification": prior_independent_verification,
    }
    audit = {
        "unit": {"domain": DOMAIN, "repetition": REPETITION, "outer_fold": OUTER_FOLD},
        "pilot_descriptive_only": True,
        "hypothesis_judgment": "NOT_PERMITTED_FROM_PILOT",
        "checks": audit_checks,
        "observed": {
            "archived_probability_max_abs_delta": {
                "R": float(np.max(np.abs(p_r - old_r))),
                "S": float(np.max(np.abs(p_s - old_s))),
                "M": float(np.max(np.abs(p_m - old_m))),
            },
            "selected_C": {"reference": reference_c, "M": m_c, "MP": mp_c,
                           "H_specific_reference": h_reference_c},
            "invariance": invariance,
            "calibration_seen_min_max": [int(calibration_seen.min()), int(calibration_seen.max())],
            "prior_lineage_record_count": len(prior_records),
            "prior_restoration_transform_counter": prior_transform_counter,
            "prior_restoration_independent_verification": prior_independent_verification,
            "head_fit_count": int(fit_counter["total_head_fits"]),
        },
        "tolerances": protocol["numerical_tolerances"],
        "all_mandatory_audits_pass": True,
        "publication_semantics": "execution_manifest hashes every non-self artifact; COMPLETE hashes the execution_manifest and is written last before one atomic directory rename",
    }

    stage_parent = FINAL_DIR.parent
    stage_parent.mkdir(parents=True, exist_ok=True)
    stage = stage_parent / f".{FINAL_DIR.name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
    if stage.exists():
        raise RuntimeError("unexpected staging collision")
    stage.mkdir()
    output_started = time.perf_counter()
    try:
        predictions_df.to_csv(stage / "predictions.csv", index=False, float_format="%.17g", lineterminator="\n")
        calibration_df.to_csv(stage / "calibration_oof.csv", index=False, float_format="%.17g", lineterminator="\n")
        cases_df.to_csv(stage / "per_case_effects.csv", index=False, float_format="%.17g", lineterminator="\n")
        metrics_df.to_csv(stage / "metrics.csv", index=False, float_format="%.17g", lineterminator="\n")
        paired_df.to_csv(stage / "paired_effects.csv", index=False, float_format="%.17g", lineterminator="\n")
        stable_json(stage / "selections.json", selection_outputs)
        stable_json(stage / "calibrators.json", calibrators)
        stable_json(stage / "prior_lineage.json", prior_lineage)
        stable_json(stage / "lineage.json", lineage)
        stable_json(stage / "audit.json", audit)
        log.append(f"{utc_now()} all computational and lineage audits passed")
        log.append(f"{utc_now()} pilot is descriptive engineering evidence only; no bootstrap or hypothesis judgment")
        run_log_text = "\n".join(log) + "\n"
        (stage / "run.log").write_text(run_log_text, encoding="utf-8")
        initial_output_staging_seconds = time.perf_counter() - output_started
        fit_cost = {
            "unit": {"domain": DOMAIN, "repetition": REPETITION, "outer_fold": OUTER_FOLD},
            "probe_generation": {"fits": 0, "reused_frozen_probe_bundles": int(generation["probe_bundles"]),
                                 "explanation": "all concept predictions reused from the frozen strict-nested panel"},
            "head_fits": fit_counter,
            "affine_fits": {"fits": 2, "C_seconds": calibrator_c["elapsed_seconds"],
                            "CP_seconds": calibrator_cp["elapsed_seconds"], "total_seconds": affine_seconds},
            "prior_restoration": {
                "operation_type": "analytic_nontraining_transform",
                "model_fits": 0,
                "calls": int(prior_transform_counter["calls"]),
                "concept_records": int(prior_transform_counter["concept_records"]),
                "probability_values": int(prior_transform_counter["probability_values"]),
                "seconds": float(prior_transform_counter["seconds"]),
                "by_role": prior_transform_counter["by_role"],
                "independent_verification_seconds": float(
                    prior_independent_verification["verification_seconds"]
                ),
                "explanation": (
                    "Analytic prior restoration only; no trainable parameters "
                    "or model fitting."
                ),
            },
            "components_seconds": {
                "input_load": load_seconds, "selection_and_H_reference_work": selection_seconds,
                "final_head_work": final_head_seconds, "affine": affine_seconds,
                "metrics": metrics_seconds, "audits": audit_seconds,
                "initial_output_staging_before_fit_cost": initial_output_staging_seconds,
                "prepublication_total": time.perf_counter() - run_started,
            },
            "timing_scope_note": "initial_output_staging excludes fit_cost, execution_manifest, COMPLETE, verification, and atomic rename; total publication time is emitted by the runner",
            "bootstrap_draws": 0, "units_run": 1,
        }
        stable_json(stage / "fit_cost.json", fit_cost)

        protocol_required = tuple(protocol["output_contract"]["pilot"]["required"])
        if (len(protocol_required) != len(EXPECTED_OUTPUT_FILES)
                or len(set(protocol_required)) != len(EXPECTED_OUTPUT_FILES)
                or set(protocol_required) != set(EXPECTED_OUTPUT_FILES)):
            raise RuntimeError(
                "frozen protocol no longer requires the exact 14 pilot artifacts"
            )
        runner_hash = sha256_file(Path(__file__))
        artifact_names = tuple(sorted(path.name for path in stage.iterdir()))
        if artifact_names != tuple(sorted(PRE_MANIFEST_FILES)):
            raise RuntimeError(
                f"staged artifact set mismatch: {artifact_names} vs "
                f"{tuple(sorted(PRE_MANIFEST_FILES))}"
            )
        if not all(
            (stage / name).is_file() and not (stage / name).is_symlink()
            for name in PRE_MANIFEST_FILES
        ):
            raise RuntimeError("pre-manifest staging contains a non-regular artifact")
        artifact_records = [{
            "path": name,
            "bytes": (stage / name).stat().st_size,
            "sha256": sha256_file(stage / name),
        } for name in PRE_MANIFEST_FILES]
        execution_manifest = {
            "unit": {"domain": DOMAIN, "repetition": REPETITION, "outer_fold": OUTER_FOLD},
            "protocol_path": str(PROTOCOL_PATH.relative_to(ROOT)),
            "protocol_sha256": static["protocol_sha256"],
            "runner_path": str(Path(__file__).relative_to(ROOT)),
            "runner_sha256": runner_hash,
            "source_sha256": {
                "cbis_source": protocol["fixed_inputs"]["cbis_source"]["sha256"],
                "strict_nested_runner": protocol["fixed_inputs"]["strict_nested_runner"]["sha256"],
                "p0_hash_manifest": protocol["fixed_inputs"]["p0_hash_manifest"]["sha256"],
            },
            "artifacts": artifact_records,
            "artifact_count_excluding_manifest_and_COMPLETE": len(artifact_records),
            "self_hash_semantics": "A manifest cannot contain its own SHA-256. COMPLETE.json anchors the SHA-256 of this manifest.",
            "complete_excluded_because_written_last": True,
        }
        stable_json(stage / "execution_manifest.json", execution_manifest)
        staged_with_manifest = tuple(sorted(path.name for path in stage.iterdir()))
        if staged_with_manifest != tuple(sorted(EXPECTED_OUTPUT_FILES[:-1])):
            raise RuntimeError("staging set is not exact after manifest publication")
        for record in artifact_records:
            if sha256_file(stage / record["path"]) != record["sha256"] or (stage / record["path"]).stat().st_size != record["bytes"]:
                raise RuntimeError("execution manifest verification failed")
        manifest_hash = sha256_file(stage / "execution_manifest.json")
        if sha256_file(PROTOCOL_PATH) != static["protocol_sha256"] or PROTOCOL_PATH.stat().st_mtime_ns != protocol_mtime_ns:
            raise RuntimeError("protocol changed before COMPLETE publication")
        if sha256_file(Path(__file__)) != runner_hash:
            raise RuntimeError("runner changed before COMPLETE publication")
        completion = {
            "domain": DOMAIN, "repetition": REPETITION, "outer_fold": OUTER_FOLD,
            "status": "COMPLETE", "interpretation": "engineering pilot only; STOP_FOR_INDEPENDENT_AUDIT",
            "protocol_sha256": static["protocol_sha256"],
            "execution_manifest_sha256": manifest_hash,
            "completed_at_utc": utc_now(),
        }
        stable_json(stage / "COMPLETE.json", completion)
        completed_stage_names = tuple(sorted(path.name for path in stage.iterdir()))
        if completed_stage_names != tuple(sorted(EXPECTED_OUTPUT_FILES)):
            raise RuntimeError("COMPLETE was not added to the exact 14-artifact stage")
        if not all(
            (stage / name).is_file() and not (stage / name).is_symlink()
            for name in EXPECTED_OUTPUT_FILES
        ):
            raise RuntimeError("completed staging contains a non-regular artifact")
        if ((stage / "COMPLETE.json").stat().st_mtime_ns
                < (stage / "execution_manifest.json").stat().st_mtime_ns):
            raise RuntimeError("COMPLETE was not written after the manifest")
        if FINAL_DIR.exists():
            raise RuntimeError("final target appeared during staging")
        os.rename(stage, FINAL_DIR)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise

    expected_frames = {
        "predictions.csv": predictions_df,
        "calibration_oof.csv": calibration_df,
        "per_case_effects.csv": cases_df,
        "metrics.csv": metrics_df,
        "paired_effects.csv": paired_df,
    }
    expected_json = {
        "selections.json": selection_outputs,
        "calibrators.json": calibrators,
        "prior_lineage.json": prior_lineage,
        "lineage.json": lineage,
        "audit.json": audit,
        "fit_cost.json": fit_cost,
    }
    validation_receipt = validate_published_unit(
        protocol=protocol,
        static=static,
        protocol_mtime_ns=protocol_mtime_ns,
        expected_frames=expected_frames,
        expected_json=expected_json,
        expected_log=run_log_text,
        execution_manifest=execution_manifest,
        completion=completion,
        expected_rows=e,
        expected_ids=ids[e],
        expected_groups=groups[e],
        expected_y=y[e],
    )
    atomic_json(VALIDATION_RECEIPT, validation_receipt)
    reloaded_receipt = json.loads(VALIDATION_RECEIPT.read_text(encoding="utf-8"))
    if reloaded_receipt != _json_round_trip(validation_receipt):
        raise RuntimeError("external validation receipt reread mismatch")
    if (reloaded_receipt.get("all_checks_pass") is not True
            or int(reloaded_receipt.get("published_artifact_count", -1)) != 14
            or not all(reloaded_receipt.get("checks", {}).values())):
        raise RuntimeError("external validation receipt failed closed")
    validation_receipt_sha256 = sha256_file(VALIDATION_RECEIPT)

    output_seconds = time.perf_counter() - output_started
    total_seconds = time.perf_counter() - run_started
    print(json.dumps({
        "status": "COMPLETE", "unit": "cbis_r0_f0",
        "directory": str(FINAL_DIR), "protocol_sha256": static["protocol_sha256"],
        "runner_sha256": sha256_file(Path(__file__)), "head_fits": fit_counter["total_head_fits"],
        "affine_fits": 2, "total_seconds": total_seconds, "publication_seconds": output_seconds,
        "post_rename_validation": "PASS",
        "validation_receipt": str(VALIDATION_RECEIPT),
        "validation_receipt_sha256": validation_receipt_sha256,
        "next_gate": "STOP_FOR_INDEPENDENT_AUDIT",
    }, sort_keys=True))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
