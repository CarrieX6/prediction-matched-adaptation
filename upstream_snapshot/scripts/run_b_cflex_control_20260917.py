#!/usr/bin/env python3
"""One frozen, development-only flexible scalar control for Paper B.

No upstream predictor is fitted. The existing V5 module is imported only for
pure cache/isolation helpers and its reference logistic-head implementation.
This script never discovers units or accesses an official reserved test set.
Cost mode fits training-side objects only and never loads outer predictions.
It requires a separately frozen, exact-hash protocol before any scientific run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import stat
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
}
for _key, _value in THREAD_ENV.items():
    os.environ[_key] = _value

import numpy as np
from threadpoolctl import threadpool_limits

SCRIPT_PATH = Path(os.path.abspath(__file__))
ROOT = SCRIPT_PATH.parent.parent
CONTROL_ROOT = ROOT / "paper_b_readiness" / "CFLEX_CONTROL_20260917"
PROTOCOL_PATH = CONTROL_ROOT / "PROTOCOL.json"
HELPER_REL = "scripts/run_p1_formal_unit_v5_20260916.py"
P1_PROTOCOL_REL = "paper_b_readiness/P1_PROTOCOL.json"
P1_PROTOCOL_SHA256 = "bdaca53493521a400f6307088d887eafa035f7eed4a094d522d05ee2ea5a1c1b"
P0_SHA256 = "d00b9d304fd87d98319a1b85315c38accbb83f758abe97b62c02a2fdb91cfdcd"
CBIS_CONTROL_REL = "data/CBIS_DDSM/paper_b_cbis_mass_train_breast_cohort_v1_20260910.npz"
GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
QUANTILES = (0.05, 0.275, 0.5, 0.725, 0.95)
REFERENCE_FITS = {"cbis": 156, "derm": 304}
SPLINE_FITS_PER_UNIT = 25
PANEL_FIELDS = (
    "y", "ids", "groups", "gold", "widths", "t", "e", "raw", "val",
) + tuple(f"{key}{h}" for h in range(4) for key in ("f", "h", "x", "z"))
TRAIN_LOGIT_SCHEMA = (
    "domain", "repetition", "outer_fold", "stage", "H", "G", "row_index",
    "case_id", "group_id", "y", "raw_reference_logit", "reference_C",
    "reference_model_id", "concept_fit_scope_id", "reference_fit_scope_id",
    "H_excluded_from_generating_fits", "E_excluded_from_generating_fits",
)
PREDICTION_SCHEMA = (
    "domain", "rep", "fold", "repetition", "outer_fold", "row_index", "case_id", "group_id",
    "y", "logit_base", "logit_S", "logit_Cflex", "p_Cflex", "p_M", "p_C", "p_S",
    "Cflex_C", "source_prediction_sha256",
)


class ControlError(RuntimeError):
    """Fail-closed protocol, isolation, numerical, or resource failure."""


def expected_units() -> tuple[tuple[str, int, int], ...]:
    return tuple(("cbis", 0, f) for f in range(5)) + tuple(
        ("derm", r, f) for r in range(5) for f in range(5)
    )


def unit_name(unit: tuple[str, int, int]) -> str:
    return f"{unit[0]}_r{unit[1]}_f{unit[2]}"


def expected_input_relpaths() -> tuple[str, ...]:
    """Exact development whitelist; no glob, directory listing, or discovery."""
    paths = {HELPER_REL, P1_PROTOCOL_REL, CBIS_CONTROL_REL}
    for unit in expected_units():
        name = unit_name(unit)
        archive = f"results/deadline_nested_20260912/{name}"
        paths.update(f"{archive}/{filename}" for filename in (
            "panel.npz", "generation.json", "selection.json", "COMPLETE.json",
        ))
        formal = f"paper_b_readiness/p1_complete/units/{name}"
        paths.update(f"{formal}/{filename}" for filename in (
            "calibration_oof.csv", "predictions.csv", "selections.json",
            "lineage.json", "COMPLETE.json",
        ))
    return tuple(sorted(paths))


def frozen_settings() -> dict[str, object]:
    return {
        "scope": "same_30_P1_whitelisted_development_units",
        "basis": "restricted_cubic_4_nonintercept_columns_plus_separate_intercept",
        "knot_train_quantiles": list(QUANTILES), "quantile_method": "linear",
        "basis_columns": ["z", "restricted_cubic_h0", "restricted_cubic_h1", "restricted_cubic_h2"],
        "basis_nonlinear_denominator": "(kappa4-kappa0)^2",
        "standardization": "basis_train_mean_and_population_sd_ddof0",
        "penalty": "L2_all_four_standardized_basis_coefficients_unpenalized_intercept",
        "C_grid": list(GRID), "class_weight": "balanced", "solver": "lbfgs",
        "tol": 0.0001, "random_state": 20260912,
        "max_iter": {"cbis": 10000, "derm": 2000},
        "selection": "mean_four_H_class_equal_NLL_first_minimum_smallest_C",
        "reference_C_selection": "D_minus_G_gold_only_remaining_original_G_blocks",
        "reference_inner_blocks": {"cbis": 2, "derm": 3},
        "H_G_splits": "exact_P1_frozen_generation_splits",
        "final_train_logits": "P1_T_fullchain_oof_raw_base_logit",
        "outer_input_logits": "P1_predictions_logit_S",
        "reuse_without_refit": ["M", "C_affine", "S"],
        "degenerate_basis_action": "FAIL_no_df_or_knot_or_grid_change",
        "convergence_warning_action": "FAIL_no_retry",
        "monotonicity": "not_imposed_flexible_one_dimensional_remapping",
        "cost_mode": "cbis_r0_f0_and_derm_r0_f0_training_only_no_E_prediction_or_metrics",
        "cpu_only": True, "single_thread": True, "official_test_access": False,
        "new_upstream_fits": 0, "bootstrap_in_unit_runner": False,
    }


def frozen_budgets() -> dict[str, object]:
    return {
        "wall_seconds_full": 1800, "wall_seconds_cost_per_unit": 180,
        "reference_fits_full": 8380, "spline_fits_full": 750,
        "reference_fits_per_unit": dict(REFERENCE_FITS),
        "spline_fits_per_unit": SPLINE_FITS_PER_UNIT,
        "on_exceedance": "FAIL_no_automatic_scope_grid_or_df_change",
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plain_file(path: Path) -> None:
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ControlError("input is not a plain regular file")
    relative = path.relative_to(ROOT)
    current = ROOT
    for part in relative.parts[:-1]:
        current = current / part
        parent = os.lstat(current)
        if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
            raise ControlError("input directory chain contains a link or non-directory")


def verify_protocol(expected_hash: str) -> dict[str, Any]:
    plain_file(PROTOCOL_PATH)
    if sha256_file(PROTOCOL_PATH) != expected_hash:
        raise ControlError("protocol hash differs from the authorizing CLI hash")
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if protocol.get("status") != "FROZEN_PRE_EXECUTION":
        raise ControlError("protocol is not frozen before execution")
    if protocol.get("protocol_id") != "B_CFLEX_CONTROL_20260917":
        raise ControlError("protocol ID differs")
    if protocol.get("script_sha256") != sha256_file(SCRIPT_PATH):
        raise ControlError("runner source hash differs from frozen protocol")
    if protocol.get("settings") != frozen_settings() or protocol.get("budgets") != frozen_budgets():
        raise ControlError("frozen settings or budgets differ from this runner")
    if protocol.get("ordered_units") != [unit_name(u) for u in expected_units()]:
        raise ControlError("unit order differs from the exact whitelist")
    inputs = protocol.get("input_sha256")
    if not isinstance(inputs, dict) or set(inputs) != set(expected_input_relpaths()):
        raise ControlError("input whitelist differs")
    if inputs[P1_PROTOCOL_REL] != P1_PROTOCOL_SHA256:
        raise ControlError("P1 protocol anchor differs")
    if inputs[HELPER_REL] != protocol.get("helper_sha256"):
        raise ControlError("helper anchor differs")
    for relative, digest in inputs.items():
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ControlError("invalid input hash")
        path = ROOT / relative
        plain_file(path)
        if sha256_file(path) != digest:
            raise ControlError(f"whitelisted input changed: {relative}")
    return protocol


def import_v5() -> Any:
    """Import without executing V5 main, reading tickets, or opening data."""
    spec = importlib.util.spec_from_file_location("b_cflex_p1_v5_helpers", ROOT / HELPER_REL)
    if spec is None or spec.loader is None:
        raise ControlError("cannot import the frozen V5 helper module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if tuple(module.GRID) != GRID or module.PROTOCOL_SHA256 != P1_PROTOCOL_SHA256:
        raise ControlError("V5 helper constants differ")
    return module


@dataclass(frozen=True)
class RestrictedCubicBasis:
    knots: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    train_n: int
    train_rank: int

    @staticmethod
    def raw(logits: np.ndarray, knots: np.ndarray) -> np.ndarray:
        z = np.asarray(logits, float)
        k = np.asarray(knots, float)
        if z.ndim != 1 or k.shape != (5,) or not np.isfinite(z).all() or not np.isfinite(k).all():
            raise ControlError("invalid restricted-cubic basis inputs")
        if not np.all(np.diff(k) > 0):
            raise ControlError("restricted-cubic knot degeneracy")
        last_gap = k[4] - k[3]
        denominator = (k[4] - k[0]) ** 2
        nonlinear = []
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for j in range(3):
                h = (np.maximum(z-k[j], 0.0) ** 3
                     - (k[4]-k[j])/last_gap * np.maximum(z-k[3], 0.0) ** 3
                     + (k[3]-k[j])/last_gap * np.maximum(z-k[4], 0.0) ** 3) / denominator
                nonlinear.append(h)
        result = np.column_stack([z, *nonlinear])
        if not np.isfinite(result).all():
            raise ControlError("nonfinite restricted-cubic basis")
        return result

    @classmethod
    def fit(cls, train_logits: np.ndarray) -> "RestrictedCubicBasis":
        z = np.asarray(train_logits, float)
        if z.ndim != 1 or len(z) < 6 or not np.isfinite(z).all():
            raise ControlError("invalid basis training logits")
        knots = np.quantile(z, QUANTILES, method="linear")
        raw = cls.raw(z, knots)
        mean, sd = raw.mean(axis=0), raw.std(axis=0, ddof=0)
        if not np.isfinite(mean).all() or not np.isfinite(sd).all() or np.any(sd <= 0):
            raise ControlError("basis standardization degeneracy")
        standardized = (raw-mean)/sd
        rank = int(np.linalg.matrix_rank(np.column_stack([np.ones(len(z)), standardized])))
        if rank != 5:
            raise ControlError("basis with separate intercept has rank below five")
        return cls(knots, mean, sd, len(z), rank)

    def transform(self, logits: np.ndarray) -> np.ndarray:
        result = (self.raw(logits, self.knots)-self.mean)/self.sd
        if not np.isfinite(result).all():
            raise ControlError("nonfinite standardized basis")
        return result

    def record(self) -> dict[str, object]:
        return {"quantiles": list(QUANTILES), "quantile_method": "linear",
                "knots": self.knots.tolist(), "train_mean": self.mean.tolist(),
                "train_population_sd": self.sd.tolist(), "train_n": self.train_n,
                "train_rank_including_separate_intercept": self.train_rank,
                "nonintercept_basis_columns": 4, "separate_intercept": True,
                "monotonicity_constraint": False}


def assert_scope_disjoint(fit_rows: np.ndarray, predict_rows: np.ndarray,
                          blocked_rows: np.ndarray, groups: np.ndarray,
                          patient_grouped: bool) -> None:
    fit, predict, blocked = [np.asarray(rows, int) for rows in (fit_rows, predict_rows, blocked_rows)]
    if len(set(fit.tolist())) != len(fit) or len(set(predict.tolist())) != len(predict):
        raise ControlError("duplicate fit or prediction row")
    if not len(fit) or not len(predict):
        raise ControlError("empty fit or prediction scope")
    if set(fit).intersection(predict) or set(fit).intersection(blocked):
        raise ControlError("row exclusion failure")
    if patient_grouped and (set(groups[fit]).intersection(groups[predict])
                            or set(groups[fit]).intersection(groups[blocked])):
        raise ControlError("patient exclusion failure")


def reference_j_splits(g_splits: Sequence[tuple[np.ndarray, np.ndarray]],
                       current_g: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """CV only inside K (D minus G), using the remaining frozen G blocks."""
    k = np.asarray(g_splits[current_g][0], int)
    j_splits = [(np.setdiff1d(k, validation), np.asarray(validation, int))
                for g, (_, validation) in enumerate(g_splits) if g != current_g]
    validations = np.concatenate([va for _, va in j_splits])
    if len(validations) != len(k) or len(set(validations)) != len(k) or set(validations) != set(k):
        raise ControlError("deep reference CV does not partition D minus G")
    return j_splits


class Budget:
    def __init__(self, mode: str):
        self.mode = mode
        self.started = time.perf_counter()
        self.unit_started = self.started
        self.reference_fits = 0
        self.spline_fits = 0
        self.unit_reference = 0
        self.unit_spline = 0
        self.domain = ""

    def begin_unit(self, domain: str) -> None:
        self.domain = domain
        self.unit_started = time.perf_counter()
        self.unit_reference = self.unit_spline = 0
        self.check()

    def check(self) -> None:
        now = time.perf_counter()
        if self.mode == "cost" and now-self.unit_started > 180:
            raise ControlError("frozen cost-unit 180-second budget exceeded")
        if self.mode == "full" and now-self.started > 1800:
            raise ControlError("frozen full 1800-second budget exceeded")
        if self.reference_fits > 8380 or self.spline_fits > 750:
            raise ControlError("full fit-count ceiling exceeded")
        if self.domain and (self.unit_reference > REFERENCE_FITS[self.domain]
                            or self.unit_spline > SPLINE_FITS_PER_UNIT):
            raise ControlError("unit fit-count ceiling exceeded")

    def reserve_fit(self, kind: str) -> None:
        self.check()
        if kind == "reference":
            self.reference_fits += 1
            self.unit_reference += 1
        elif kind == "spline":
            self.spline_fits += 1
            self.unit_spline += 1
        else:
            raise ControlError("unrecognized fit role")
        self.check()


class UnitRecorder:
    def __init__(self, unit: tuple[str, int, int], ids: np.ndarray,
                 groups: np.ndarray, budget: Budget, v5: Any):
        self.unit, self.ids, self.groups = unit, ids, groups
        self.budget, self.v5 = budget, v5
        self.scopes: dict[str, dict[str, object]] = {}
        self.models: list[dict[str, object]] = []
        self.fit_scopes: list[dict[str, object]] = []
        self.reference_seconds = self.spline_seconds = 0.0

    def scope(self, rows: np.ndarray) -> str:
        rows = np.asarray(rows, int)
        key = hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest()
        if key not in self.scopes:
            self.scopes[key] = {"row_indices": rows.tolist(), "n_rows": len(rows),
                                "case_scope_sha256": self.v5.scope_sha(rows, self.ids),
                                "group_scope_sha256": self.v5.scope_sha(rows, self.groups)}
        elif self.scopes[key]["row_indices"] != rows.tolist():
            raise ControlError("scope hash collision")
        return key

    def fit(self, x: np.ndarray, y: np.ndarray, c: float, kind: str, role: str,
            fit_rows: np.ndarray, predict_rows: np.ndarray, blocked_rows: np.ndarray,
            *, context: Mapping[str, object], basis_id: str | None = None) -> tuple[Any, str]:
        assert_scope_disjoint(fit_rows, predict_rows, blocked_rows, self.groups, self.unit[0] == "cbis")
        if x.shape[0] != len(fit_rows) or y.shape != (len(fit_rows),) or set(y.tolist()) != {0, 1}:
            raise ControlError("model input/fit-scope or class mismatch")
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ControlError("invalid logistic-head matrix")
        if kind == "spline" and x.shape[1] != 4:
            raise ControlError("spline does not have four nonintercept columns")
        self.budget.reserve_fit(kind)
        started = time.perf_counter()
        head = self.v5.Head(c, 10000 if self.unit[0] == "cbis" else 2000).fit(x, y)
        seconds = time.perf_counter()-started
        self.budget.check()
        model_id = f"fit_{len(self.models):05d}"
        if kind == "reference":
            self.reference_seconds += seconds
        else:
            self.spline_seconds += seconds
        params = head.model.get_params()
        if (params["solver"] != "lbfgs" or params["class_weight"] != "balanced"
                or params["tol"] != 1e-4 or params["random_state"] != 20260912
                or params["penalty"] != "l2" or not params["fit_intercept"]):
            raise ControlError("fitted head does not match frozen logistic parameters")
        self.models.append({"model_id": model_id, "kind": kind, "role": role,
                            "seconds": seconds, "basis_id": basis_id, **head.record()})
        self.fit_scopes.append({"model_id": model_id, "kind": kind, "role": role,
                                "fit_scope_id": self.scope(fit_rows),
                                "predict_scope_id": self.scope(predict_rows),
                                "blocked_scope_id": self.scope(blocked_rows),
                                "row_and_patient_exclusion_verified": True,
                                "basis_id": basis_id, **dict(context)})
        return head, model_id


def read_csv(path: Path, schema: Sequence[str]) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(schema):
            raise ControlError("source CSV schema differs")
        return list(reader)


def build_g_splits(unit: tuple[str, int, int], data: Mapping[str, np.ndarray],
                   generation: Mapping[str, Any], v5: Any,
                   source_folds: np.ndarray | None,
                   cbis_manifests: list[Mapping[str, Any]] | None) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    domain, rep, outer = unit
    ids, groups = data["ids"].astype(str), data["groups"].astype(str)
    e, t = data["e"].astype(int), data["t"].astype(int)
    hrows = [data[f"h{h}"].astype(int) for h in range(4)]
    selection_manifests = v5.validate_generation_partition(generation, domain, t, e, hrows)
    answer = []
    for h in range(4):
        d, heldout = data[f"f{h}"].astype(int), hrows[h]
        if set(d) != set(t)-set(heldout):
            raise ControlError("D differs from T minus H")
        assert_scope_disjoint(d, heldout, e, groups, domain == "cbis")
        if data[f"x{h}"].shape != (len(d), data["gold"].shape[1]):
            raise ControlError("D-local predicted attribute matrix shape differs")
        if domain == "cbis":
            if source_folds is None or cbis_manifests is None:
                raise ControlError("CBIS generation controls missing")
            manifest = cbis_manifests[h]
            nested = manifest.get("nested_probe_panels")
            if not isinstance(nested, list) or len(nested) != 3:
                raise ControlError("CBIS nested prediction records differ")
            splits = []
            for item in nested:
                g = np.flatnonzero(source_folds == int(item.get("probe_heldout_fold", -1)))
                if not set(g).issubset(d):
                    raise ControlError("CBIS G is outside D")
                splits.append((np.setdiff1d(d, g), g))
            if (manifest.get("fit_patient_sha256") != v5.sha256_values(groups[d])
                    or manifest.get("heldout_patient_sha256") != v5.sha256_values(groups[heldout])
                    or manifest.get("fit_heldout_patient_disjoint") is not True):
                raise ControlError("CBIS H upstream generating scope differs")
            for item, (k, g) in zip(nested, splits, strict=True):
                if (item.get("probe_fit_patient_sha256") != v5.sha256_values(groups[k])
                        or item.get("probe_validation_patient_sha256") != v5.sha256_values(groups[g])
                        or item.get("diagnosis_validation_excluded_from_probe_fit") is not True
                        or item.get("probe_validation_excluded_from_probe_fit") is not True):
                    raise ControlError("CBIS G upstream generating scope differs")
        else:
            splits = v5._nested_subfolds(domain, rep, outer, h, d, [r for j, r in enumerate(hrows) if j != h])
            manifest = selection_manifests[h]
            if (not isinstance(manifest, dict)
                    or list(map(str, manifest.get("fit_case_ids", []))) != ids[d].tolist()
                    or list(map(str, manifest.get("heldout_case_ids", []))) != ids[heldout].tolist()
                    or not isinstance(manifest.get("inner"), list) or len(manifest["inner"]) != 4):
                raise ControlError("Derm H upstream generating scope differs")
            for item, (k, g) in zip(manifest["inner"], splits, strict=True):
                if (item.get("fit_case_id_sha256") != v5.sha256_values(ids[k])
                        or item.get("validation_case_id_sha256") != v5.sha256_values(ids[g])
                        or item.get("fit_validation_disjoint") is not True):
                    raise ControlError("Derm G upstream generating scope differs")
        all_g = np.concatenate([g for _, g in splits])
        if len(all_g) != len(d) or len(set(all_g)) != len(d) or set(all_g) != set(d):
            raise ControlError("frozen G blocks do not partition D")
        for k, g in splits:
            if set(k) != set(d)-set(g):
                raise ControlError("K differs from D minus G")
            assert_scope_disjoint(k, g, np.concatenate([heldout, e]), groups, domain == "cbis")
        answer.append(splits)
    return answer


def compute_training(unit: tuple[str, int, int], protocol: Mapping[str, Any],
                     budget: Budget, v5: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns fit artifacts and an in-memory final model, never E predictions."""
    started = time.perf_counter()
    domain, rep, outer = unit
    name = unit_name(unit)
    archive_prefix = f"results/deadline_nested_20260912/{name}"
    formal_prefix = f"paper_b_readiness/p1_complete/units/{name}"
    with np.load(ROOT / f"{archive_prefix}/panel.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(PANEL_FIELDS):
            raise ControlError("panel field set differs")
        # In neither mode do we decode raw/val/z arrays containing outer inputs.
        needed = ("y", "ids", "groups", "gold", "t", "e") + tuple(
            f"{key}{h}" for h in range(4) for key in ("f", "h", "x")
        )
        data = {key: archive[key].copy() for key in needed}
    y, ids, groups = data["y"].astype(int), data["ids"].astype(str), data["groups"].astype(str)
    gold, t, e = data["gold"].astype(float), data["t"].astype(int), data["e"].astype(int)
    if (gold.ndim != 2 or gold.shape[0] != len(y) or ids.shape != y.shape
            or groups.shape != y.shape or set(y[t].tolist()) != {0, 1}
            or not np.isfinite(gold[t]).all()):
        raise ControlError("panel training-side structure differs")
    generation = json.loads((ROOT / f"{archive_prefix}/generation.json").read_text())
    archive_complete = json.loads((ROOT / f"{archive_prefix}/COMPLETE.json").read_text())
    complete = json.loads((ROOT / f"{formal_prefix}/COMPLETE.json").read_text())
    lineage = json.loads((ROOT / f"{formal_prefix}/lineage.json").read_text())
    selections = json.loads((ROOT / f"{formal_prefix}/selections.json").read_text())
    for receipt in (archive_complete, complete):
        if receipt.get("domain") != domain or receipt.get("repetition") != rep or receipt.get("outer_fold") != outer:
            raise ControlError("source COMPLETE unit differs")
    if (complete.get("status") != "COMPLETE" or complete.get("all_formal_publication_audits_pass") is not True
            or complete.get("protocol_sha256") != P1_PROTOCOL_SHA256
            or lineage.get("anchors", {}).get("protocol_sha256") != P1_PROTOCOL_SHA256
            or lineage.get("anchors", {}).get("p0_manifest_sha256") != P0_SHA256
            or lineage.get("anchors", {}).get("engine_sha256") != protocol["helper_sha256"]
            or lineage.get("archive_generation_sha256") != protocol["input_sha256"][f"{archive_prefix}/generation.json"]
            or lineage.get("archive_selection_sha256") != protocol["input_sha256"][f"{archive_prefix}/selection.json"]
            or lineage.get("access_boundary", {}).get("official_test_access") is not False):
        raise ControlError("formal P1 accepted lineage differs")
    if selections.get("C_grid") != list(GRID):
        raise ControlError("P1 reference C grid differs")
    source_folds, cbis_manifests = None, None
    if domain == "cbis":
        _, source_folds = v5.load_cbis_control(ROOT / CBIS_CONTROL_REL, data, outer)
        cbis_manifests, _ = v5.resolve_cbis_manifests(generation, data, groups, source_folds, outer)
    g_blocks = build_g_splits(unit, data, generation, v5, source_folds, cbis_manifests)
    calibration = read_csv(ROOT / f"{formal_prefix}/calibration_oof.csv", v5.CSV_SCHEMAS["calibration_oof.csv"])
    q_t = np.full(len(t), np.nan)
    tpos = {int(row): pos for pos, row in enumerate(t)}
    write_count = np.zeros(len(t), int)
    source_ref_selections = selections.get("selections", {}).get("reference_H")
    if not isinstance(source_ref_selections, list) or len(source_ref_selections) != 4:
        raise ControlError("P1 H reference selections missing")
    ref_cs = {int(item["H"]): float(item["selected_C"]) for item in source_ref_selections}
    recorder = UnitRecorder(unit, ids, groups, budget, v5)
    train_rows: list[dict[str, object]] = []
    for row in calibration:
        index, h = int(row["row_index"]), int(row["H"])
        if h not in range(4) or index not in tpos or index not in set(data[f"h{h}"].astype(int)):
            raise ControlError("calibration H/row identity differs")
        d = data[f"f{h}"].astype(int)
        scope = v5.scope_sha(d, ids)
        if (row["domain"] != domain or int(row["repetition"]) != rep or int(row["outer_fold"]) != outer
                or row["case_id"] != ids[index] or row["group_id"] != groups[index] or int(row["y"]) != y[index]
                or row["probe_fit_scope_sha256"] != scope or row["reference_fit_scope_sha256"] != scope
                or float(row["reference_C_H"]) != ref_cs[h]):
            raise ControlError("calibration generating relation differs")
        assert_scope_disjoint(d, data[f"h{h}"].astype(int), e, groups, domain == "cbis")
        z = float(row["raw_base_logit"])
        if not np.isfinite(z):
            raise ControlError("nonfinite cached training logit")
        q_t[tpos[index]] = z
        write_count[tpos[index]] += 1
        train_rows.append(dict(zip(TRAIN_LOGIT_SCHEMA, (
            domain, rep, outer, "T_fullchain_oof_cached", h, -1, index,
            ids[index], groups[index], int(y[index]), z, ref_cs[h], f"P1_reference_H{h}",
            recorder.scope(d), recorder.scope(d), True, True,
        ), strict=True)))
    if len(calibration) != len(t) or not np.all(write_count == 1) or not np.isfinite(q_t).all():
        raise ControlError("cached T logits do not cover T exactly once")
    basis_records: dict[str, object] = {}
    reference_curves: list[dict[str, object]] = []
    spline_blocks = []
    for h in range(4):
        d, heldout = data[f"f{h}"].astype(int), data[f"h{h}"].astype(int)
        xd = data[f"x{h}"].astype(float)
        dpos = {int(row): pos for pos, row in enumerate(d)}
        q_d = np.full(len(d), np.nan)
        d_write_count = np.zeros(len(d), int)
        splits = g_blocks[h]
        for g, (k, g_rows) in enumerate(splits):
            blocked = np.concatenate([heldout, e, g_rows])
            j_splits = reference_j_splits(splits, g)
            if len(j_splits) != (2 if domain == "cbis" else 3):
                raise ControlError("deep reference CV block count differs")
            candidates = []
            for c in GRID:
                losses = []
                for j, (j_train, j_validation) in enumerate(j_splits):
                    if not set(j_train).issubset(k) or not set(j_validation).issubset(k):
                        raise ControlError("deep reference CV escaped K")
                    head, _ = recorder.fit(gold[j_train], y[j_train], c, "reference",
                                           f"H{h}_G{g}_reference_C_selection", j_train,
                                           j_validation, blocked, context={"H": h, "G": g, "J": j})
                    losses.append(v5.class_equal_nll(y[j_validation], head.probabilities(gold[j_validation])))
                candidates.append({"C": c, "fold_class_equal_NLL": losses,
                                   "mean_class_equal_NLL": float(np.mean(losses))})
            selected_index = int(np.argmin([r["mean_class_equal_NLL"] for r in candidates]))
            c_ref = GRID[selected_index]
            reference_curves.append({"H": h, "G": g, "C_grid": list(GRID),
                                     "selection_scope_id": recorder.scope(k), "candidates": candidates,
                                     "selected_index": selected_index, "selected_C": c_ref,
                                     "tie_break": "first_minimum_smallest_C"})
            head, model_id = recorder.fit(gold[k], y[k], c_ref, "reference",
                                          f"H{h}_G{g}_reference_final", k, g_rows,
                                          np.concatenate([heldout, e]), context={"H": h, "G": g, "J": -1})
            local = np.asarray([dpos[int(row)] for row in g_rows], int)
            logits = head.logits(xd[local])
            if logits.shape != (len(g_rows),) or not np.isfinite(logits).all():
                raise ControlError("invalid D-local reference logits")
            q_d[local] = logits
            d_write_count[local] += 1
            for row_index, z in zip(g_rows, logits, strict=True):
                train_rows.append(dict(zip(TRAIN_LOGIT_SCHEMA, (
                    domain, rep, outer, "D_local_fullchain_oof_new_reference", h, g,
                    int(row_index), ids[row_index], groups[row_index], int(y[row_index]),
                    float(z), c_ref, model_id, recorder.scope(k), recorder.scope(k), True, True,
                ), strict=True)))
        if not np.all(d_write_count == 1) or not np.isfinite(q_d).all():
            raise ControlError("D-local logits do not cover D exactly once")
        # Crucially, q_d is NOT a subset of the cached T-global OOF logits.
        basis = RestrictedCubicBasis.fit(q_d)
        basis_id = f"H{h}_train_D_basis"
        basis_records[basis_id] = {**basis.record(), "fit_scope_id": recorder.scope(d),
                                   "H_and_E_excluded": True}
        q_h = np.asarray([q_t[tpos[int(row)]] for row in heldout], float)
        spline_blocks.append((d, heldout, basis.transform(q_d), basis.transform(q_h), basis_id))
    candidates = []
    for c in GRID:
        losses = []
        for h, (d, heldout, x_d, x_h, basis_id) in enumerate(spline_blocks):
            head, _ = recorder.fit(x_d, y[d], c, "spline", "Cflex_H_selection",
                                   d, heldout, e, context={"H": h, "G": -1, "J": -1}, basis_id=basis_id)
            losses.append(v5.class_equal_nll(y[heldout], head.probabilities(x_h)))
        candidates.append({"C": c, "fold_class_equal_NLL": losses,
                           "mean_class_equal_NLL": float(np.mean(losses))})
    selected_index = int(np.argmin([r["mean_class_equal_NLL"] for r in candidates]))
    c_flex = GRID[selected_index]
    final_basis = RestrictedCubicBasis.fit(q_t)
    basis_records["T_final_basis"] = {**final_basis.record(), "fit_scope_id": recorder.scope(t), "E_excluded": True}
    # predict scope is recorded, not evaluated, in cost mode.
    final_head, final_model_id = recorder.fit(final_basis.transform(q_t), y[t], c_flex,
                                             "spline", "Cflex_final_T", t, e, e,
                                             context={"H": -1, "G": -1, "J": -1}, basis_id="T_final_basis")
    if budget.unit_reference != REFERENCE_FITS[domain] or budget.unit_spline != 25:
        raise ControlError("frozen exact unit fit count not attained")
    artifacts = {
        "models": recorder.models, "basis": basis_records,
        "scope_definitions": recorder.scopes, "fit_scopes": recorder.fit_scopes,
        "training_logits": train_rows,
        "selections": {"reference_D_local": reference_curves,
                       "Cflex": {"C_grid": list(GRID), "candidates": candidates,
                                 "selected_index": selected_index, "selected_C": c_flex,
                                 "tie_break": "first_minimum_smallest_C"}},
        "run_record": {"domain": domain, "repetition": rep, "outer_fold": outer,
                       "mode": budget.mode, "reference_fits": budget.unit_reference,
                       "spline_fits": budget.unit_spline,
                       "reference_fit_seconds": recorder.reference_seconds,
                       "spline_fit_seconds": recorder.spline_seconds,
                       "training_wall_seconds": time.perf_counter()-started,
                       "E_prediction_performed": False, "E_metrics_performed": False,
                       "official_test_accessed": False, "upstream_fits": 0,
                       "cached_global_T_OOF_used_for_H_training": False,
                       "protocol_id": protocol["protocol_id"],
                       "script_sha256": protocol["script_sha256"],
                       "input_sha256": {rel: digest for rel, digest in protocol["input_sha256"].items()
                                        if name in rel or rel in (HELPER_REL, P1_PROTOCOL_REL, CBIS_CONTROL_REL)},
                       "final_model_id": final_model_id, "selected_Cflex_C": c_flex},
    }
    state = {"head": final_head, "basis": final_basis, "selected_C": c_flex,
             "e": e, "y": y, "ids": ids, "groups": groups}
    return artifacts, state


def predict_outer(unit: tuple[str, int, int], state: Mapping[str, Any],
                  protocol: Mapping[str, Any], v5: Any, mode: str) -> list[dict[str, object]]:
    if mode != "full":
        raise ControlError("outer prediction is prohibited outside full mode")
    name = unit_name(unit)
    relative = f"paper_b_readiness/p1_complete/units/{name}/predictions.csv"
    source = read_csv(ROOT / relative, v5.CSV_SCHEMAS["predictions.csv"])
    e, ids, groups, y = (state[key] for key in ("e", "ids", "groups", "y"))
    if len(source) != len(e):
        raise ControlError("outer prediction source count differs")
    logits = []
    for row, index in zip(source, e, strict=True):
        if (row["domain"] != unit[0] or int(row["repetition"]) != unit[1]
                or int(row["outer_fold"]) != unit[2] or int(row["row_index"]) != index
                or row["case_id"] != ids[index] or row["group_id"] != groups[index]
                or int(row["y"]) != y[index]):
            raise ControlError("outer prediction identity/order differs")
        logit = float(row["logit_S"])
        existing_p = [float(row[f"p_{arm}"]) for arm in ("M", "C", "S")]
        if not np.isfinite(logit) or any(not np.isfinite(p) or not 0 <= p <= 1 for p in existing_p):
            raise ControlError("invalid frozen outer input")
        if not np.isclose(v5.sigmoid(np.asarray([logit]))[0], existing_p[2], atol=1e-12, rtol=1e-12):
            raise ControlError("frozen S probability disagrees with its logit")
        logits.append(logit)
    q_e = state["head"].logits(state["basis"].transform(np.asarray(logits)))
    p_e = v5.sigmoid(q_e)
    if q_e.shape != (len(e),) or not np.isfinite(q_e).all() or not np.isfinite(p_e).all():
        raise ControlError("invalid Cflex outer prediction")
    predictions = []
    for row, index, q, p in zip(source, e, q_e, p_e, strict=True):
        predictions.append(dict(zip(PREDICTION_SCHEMA, (
            unit[0], unit[1], unit[2], unit[1], unit[2], int(index), ids[index], groups[index], int(y[index]),
            float(row["logit_S"]), float(row["logit_S"]), float(q), float(p), float(row["p_M"]),
            float(row["p_C"]), float(row["p_S"]), state["selected_C"],
            protocol["input_sha256"][relative],
        ), strict=True)))
    return predictions


def write_new_json(path: Path, value: object) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_new_csv(path: Path, schema: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=schema, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cost", "full"), required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args(argv)
    # No optional unit/seed/grid/df flags: this is an exact frozen scope.
    overall_started = time.perf_counter()
    protocol = verify_protocol(args.protocol_sha256)
    v5 = import_v5()
    mode_root = CONTROL_ROOT / args.mode
    mode_root.mkdir(exist_ok=False)
    units_root = mode_root / "units"
    units_root.mkdir()
    units = expected_units() if args.mode == "full" else (("cbis", 0, 0), ("derm", 0, 0))
    budget = Budget(args.mode)
    current_unit = None
    try:
        with threadpool_limits(limits=1):
            for unit in units:
                current_unit = unit_name(unit)
                out = units_root / current_unit
                out.mkdir()
                budget.begin_unit(unit[0])
                artifacts, state = compute_training(unit, protocol, budget, v5)
                if args.mode == "full":
                    predictions = predict_outer(unit, state, protocol, v5, args.mode)
                    artifacts["run_record"]["E_prediction_performed"] = True
                    write_new_csv(out / "predictions.csv", PREDICTION_SCHEMA, predictions)
                budget.check()
                for key in ("models", "basis", "scope_definitions", "fit_scopes", "selections", "run_record"):
                    write_new_json(out / f"{key}.json", artifacts[key])
                write_new_csv(out / "training_logits.csv", TRAIN_LOGIT_SCHEMA, artifacts["training_logits"])
                # Record post-write wall time in receipt; run_record is immutable.
                write_new_json(out / "COMPLETE.json", {
                    "status": "TRAINING_COST_COMPLETE" if args.mode == "cost" else "COMPLETE",
                    "unit": current_unit, "mode": args.mode, "protocol_sha256": args.protocol_sha256,
                    "E_prediction_performed": args.mode == "full", "E_metrics_performed": False,
                    "unit_wall_seconds_including_write": time.perf_counter()-budget.unit_started,
                    "official_test_accessed": False, "upstream_fits": 0,
                    "artifact_sha256": {filename: sha256_file(out / filename) for filename in (
                        "models.json", "basis.json", "scope_definitions.json", "fit_scopes.json",
                        "selections.json", "run_record.json", "training_logits.csv",
                    ) + (("predictions.csv",) if args.mode == "full" else ())},
                })
                budget.check()
                print(json.dumps({"status": "UNIT_COMPLETE", "unit": current_unit, "mode": args.mode,
                                  "reference_fits": budget.unit_reference, "spline_fits": budget.unit_spline}), flush=True)
        # Pins all inputs and source again; no interpretation of outer rows.
        verify_protocol(args.protocol_sha256)
        budget.check()
        if args.mode == "full" and (budget.reference_fits != 8380 or budget.spline_fits != 750):
            raise ControlError("full exact fit count differs")
        write_new_json(mode_root / "RUN_COMPLETE.json", {
            "status": "COMPLETE", "mode": args.mode, "units": [unit_name(u) for u in units],
            "protocol_sha256": args.protocol_sha256, "script_sha256": protocol["script_sha256"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "reference_fits": budget.reference_fits, "spline_fits": budget.spline_fits,
            "compute_wall_seconds": time.perf_counter()-budget.started,
            "overall_wall_seconds_including_input_verification": time.perf_counter()-overall_started,
            "official_test_accessed": False, "E_metrics_performed": False,
            "E_prediction_performed": args.mode == "full", "bootstrap_performed": False,
            "hypothesis_judgment_performed": False,
        })
    except Exception as exc:
        write_new_json(mode_root / "RUN_FAILURE.json", {
            "status": "FAILED_CLOSED", "unit": current_unit, "mode": args.mode,
            "exception_type": type(exc).__name__, "reason": str(exc),
            "protocol_sha256": args.protocol_sha256, "official_test_accessed": False,
            "reference_fits_attempted": budget.reference_fits,
            "spline_fits_attempted": budget.spline_fits,
            "no_retry_or_automatic_protocol_change": True,
        })
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
