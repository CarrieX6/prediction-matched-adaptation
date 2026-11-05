#!/usr/bin/env python3
"""Unauthorised repair candidate for one frozen Paper-B P1 formal unit.

Do not execute this candidate on a scientific unit.  It must first receive a
separate result-blind source audit, a fixed-hash orchestration specification,
and a fresh authorization chain.

The orchestrator, rather than this process, verifies the independent pilot V4
receipt and the authorization chain.  This process consumes one authenticated
ticket on stdin, reads only the twelve project inputs named by that ticket, and
writes the twelve pre-publication payloads.  It never discovers units, reads a
pilot/formal result directory, bootstraps, or makes a scientific hypothesis
judgment.  F20 is deliberately left to the publishing orchestrator.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import stat
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

# These are set before importing numerical libraries.  A non-matching inherited
# value is rejected again in main; the orchestrator also pins this environment.
_THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "BLIS_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
}
for _key, _value in _THREAD_ENV.items():
    os.environ[_key] = _value

import numpy as np
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

ENGINE_PATH = Path(os.path.abspath(__file__))
ROOT = ENGINE_PATH.parent.parent
FORMAL_ROOT = ROOT / "paper_b_readiness" / "p1_complete" / "units"
PROTOCOL_SHA256 = "bdaca53493521a400f6307088d887eafa035f7eed4a094d522d05ee2ea5a1c1b"
P0_SHA256 = "d00b9d304fd87d98319a1b85315c38accbb83f758abe97b62c02a2fdb91cfdcd"
ORDERED_SCOPE_SHA256 = "76c7c9e2fbd8e9fe7318e7fa3d5a18aa349d7ccb90af8e6a4423c9c62dcd1e2b"
GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
ARMS = ("R", "S", "C", "M", "SP", "CP", "MP")
CONTRASTS = ("C_minus_M", "S_minus_SP", "S_minus_C", "CP_minus_MP", "M_minus_MP")
CONTRAST_SIGNS = {
    "C_minus_M": "negative_favours_C", "S_minus_SP": "positive_favours_SP",
    "S_minus_C": "positive_favours_C", "CP_minus_MP": "negative_favours_CP",
    "M_minus_MP": "positive_favours_MP",
}
F_CHECKS = tuple(f"F{i:02d}" for i in range(1, 20))

CSV_SCHEMAS = {
    "predictions.csv": ("domain", "repetition", "outer_fold", "row_index", "case_id", "group_id", "y", "p_R", "p_S", "p_C", "p_M", "p_SP", "p_CP", "p_MP", "logit_S", "logit_C", "logit_SP", "logit_CP"),
    "calibration_oof.csv": ("domain", "repetition", "outer_fold", "row_index", "case_id", "group_id", "y", "H", "reference_C_H", "raw_base_logit", "prior_restored_base_logit", "probe_fit_scope_sha256", "reference_fit_scope_sha256"),
    "per_case_effects.csv": ("domain", "repetition", "outer_fold", "row_index", "case_id", "group_id", "y", "arm", "negative_log_likelihood", "squared_probability_error", "class_equal_nll_contribution"),
    "metrics.csv": ("domain", "repetition", "outer_fold", "arm", "class_equal_NLL", "AUC", "natural_frequency_NLL", "Brier", "n_cases", "n_groups"),
    "paired_effects.csv": ("domain", "repetition", "outer_fold", "contrast", "class_equal_NLL_X_minus_Y", "AUC_X_minus_Y", "favourable_sign_NLL", "pilot_descriptive_only"),
}
JSON_PAYLOADS = ("selections.json", "calibrators.json", "prior_lineage.json", "lineage.json", "audit.json", "fit_cost.json")
PAYLOAD_FILES = tuple(CSV_SCHEMAS) + JSON_PAYLOADS + ("run.log",)
ARCHIVE_FILES = ("panel.npz", "generation.json", "predictions.csv", "selection.json", "heads.json", "COMPLETE.json")
CONTROLS = {
    "cbis": (
        "results/deadline_nested_20260912/protocol.json", "scripts/run_deadline_nested_20260912.py",
        "src/paper_b_cbis_p3.py", "configs/paper_b_cbis_p3_minimal_protocol_final_v1_20260910.json",
        "data/CBIS_DDSM/paper_b_cbis_mass_train_breast_cohort_v1_20260910.npz",
        "data_manifests/paper_b_cbis_mass_train_breast_cohort_v1_20260910.csv",
    ),
    "derm": (
        "results/deadline_nested_20260912/protocol.json", "scripts/run_deadline_nested_20260912.py",
        "src/paper2_mpjc/pilot.py", "configs/cf_mpjc_d1_derm7pt.json",
        "data/Derm7pt/derm7pt_biomedclip_dev609_only_v1_20260908.npz",
        "data/Derm7pt/derm7pt_dev609_only_v1_20260908.csv",
    ),
}
EXPECTED_COST = {
    "cbis": {"head_fits": 151, "affine_fits": 2, "prior_restoration_calls": 17, "prior_concept_records": 170},
    "derm": {"head_fits": 175, "affine_fits": 2, "prior_restoration_calls": 21, "prior_concept_records": 147},
}
CBIS_CONCEPTS = (
    "shape_round", "shape_oval", "shape_lobulated", "shape_irregular",
    "shape_architectural_distortion", "margin_circumscribed",
    "margin_microlobulated", "margin_obscured", "margin_ill_defined",
    "margin_spiculated",
)
TICKET_KEYS = (
    "contract", "unit", "ordered_scope_index", "ordered_scope_size", "ordered_scope_sha256",
    "stage_directory", "input_whitelist", "input_whitelist_sha256", "required_payload_files",
    "csv_schemas", "arms", "contrasts", "protocol_sha256", "p0_manifest_sha256",
    "orchestrator_sha256", "engine_sha256", "spec_sha256", "authorization_sha256",
    "engine_audit_sha256", "execution_rules", "ticket_sha256",
)
EXECUTION_RULES = {
    "development_only": True, "cpu_only": True, "single_thread": True,
    "one_unit_only": True, "no_bootstrap": True, "no_hypothesis_judgment": True,
    "no_path_discovery": True, "no_official_test_access_or_discovery": True,
}


FAILURE_STAGES = (
    "BOOT", "ARGUMENT_CONTRACT", "THREAD_ENV_CONTRACT",
    "TICKET_READ_PARSE", "TICKET_VALIDATE", "WHITELIST_PRE",
    "COMPUTE_ARCHIVE_LOAD", "COMPUTE_STRUCTURE_CONTRACT",
    "COMPUTE_GENERATION_LINEAGE", "COMPUTE_NESTED_FITS",
    "COMPUTE_OUTER_FITS", "COMPUTE_ARCHIVE_PREDICTION_REPRODUCTION",
    "COMPUTE_ARCHIVE_SELECTION_REPRODUCTION", "COMPUTE_COST_CONTRACT",
    "WHITELIST_POST", "PAYLOAD_BUILD", "PAYLOAD_WRITE",
    "SUCCESS_RECEIPT",
)
FAILURE_ERROR_CODES = (
    "E_FORMAL_CONTRACT",
    "E_ARCHIVE_SELECTION_CONTRACT",
    "E_UNEXPECTED_EXCEPTION",
)
_failure_stage = "BOOT"


def set_failure_stage(stage: str) -> None:
    """Set a fixed, non-data-bearing stage for a sanitized failure receipt."""
    global _failure_stage
    if stage not in FAILURE_STAGES:
        raise RuntimeError("failure-stage label is not allowlisted")
    _failure_stage = stage


class FormalError(RuntimeError):
    def __init__(self, message: str, *, code: str = "E_FORMAL_CONTRACT") -> None:
        if code not in FAILURE_ERROR_CODES:
            raise RuntimeError("formal error code is not allowlisted")
        super().__init__(message)
        self.code = code


def sanitized_failure_receipt(exc: Exception) -> dict[str, object]:
    """Return no exception text, paths, identifiers, data values, or traceback."""
    code = exc.code if isinstance(exc, FormalError) else "E_UNEXPECTED_EXCEPTION"
    return {
        "status": "PAYLOAD_FAILED_CLOSED",
        "stage": _failure_stage,
        "error_code": code,
        "official_test_accessed": False,
        "hypothesis_judgment_performed": False,
    }


def stable_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def sha256_json(value: object) -> str:
    return hashlib.sha256(stable_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_values(values: Sequence[object]) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, values))).encode()).hexdigest()


def scope_sha(rows: np.ndarray, ids: np.ndarray) -> str:
    return sha256_values(ids[np.asarray(rows, dtype=int)].astype(str))


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def require_exact_keys(value: object, expected: Sequence[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or len(value) != len(expected) or set(value) != set(expected):
        raise FormalError(f"{label} keys differ from the frozen contract")
    return value


def unit_name(unit: tuple[str, int, int]) -> str:
    return f"{unit[0]}_r{unit[1]}_f{unit[2]}"


def unit_object(unit: tuple[str, int, int]) -> dict[str, object]:
    return {"domain": unit[0], "repetition": unit[1], "outer_fold": unit[2]}


def expected_units() -> tuple[tuple[str, int, int], ...]:
    return tuple(("cbis", 0, f) for f in range(5)) + tuple(("derm", r, f) for r in range(5) for f in range(5))


def require_plain_file(path: Path, label: str) -> None:
    try:
        metadata = os.lstat(path)
    except OSError as exc:
        raise FormalError(f"{label} cannot be inspected") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise FormalError(f"{label} is not a non-symlink regular file")


def require_plain_directory_chain(path: Path, root: Path) -> None:
    if not path.is_absolute():
        raise FormalError("stage directory is not absolute")
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise FormalError("stage directory escapes the frozen formal root") from exc
    current = root
    for part in relative.parts:
        if part in ("", ".", ".."):
            raise FormalError("unsafe stage component")
        current = current / part
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise FormalError("stage ancestry is absent") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise FormalError("stage ancestry is not a plain directory")


def write_new_bytes(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    finally:
        os.close(fd)


def write_new_json(path: Path, value: object) -> None:
    write_new_bytes(path, stable_json_bytes(value))


def write_new_csv(path: Path, schema: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(schema), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if tuple(row) != tuple(schema):
            raise FormalError(f"row schema changed for {path.name}")
        writer.writerow(row)
    write_new_bytes(path, buffer.getvalue().encode())


def _safe_rel_path(value: object) -> str:
    if not isinstance(value, str) or value.startswith("/"):
        raise FormalError("whitelist path is not relative")
    parts = Path(value).parts
    if not parts or any(p in ("", ".", "..") for p in parts):
        raise FormalError("unsafe whitelist path")
    return value


def validate_ticket(raw: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[str, int, int], Path]:
    ticket = dict(raw)
    require_exact_keys(ticket, TICKET_KEYS, "ticket")
    claimed = ticket.pop("ticket_sha256")
    if sha256_json(ticket) != claimed:
        raise FormalError("ticket digest mismatch")
    ticket["ticket_sha256"] = claimed
    if ticket["contract"] != "P1_FORMAL_UNIT_TICKET_V1":
        raise FormalError("ticket contract mismatch")
    u = ticket["unit"]
    require_exact_keys(u, ("domain", "repetition", "outer_fold"), "ticket unit")
    unit = (u["domain"], u["repetition"], u["outer_fold"])
    units = expected_units()
    if unit not in units or ticket["ordered_scope_index"] != units.index(unit):
        raise FormalError("ticket unit is outside or misindexed in frozen scope")
    if ticket["ordered_scope_size"] != 30 or ticket["ordered_scope_sha256"] != ORDERED_SCOPE_SHA256:
        raise FormalError("ordered scope anchor mismatch")
    if ticket["protocol_sha256"] != PROTOCOL_SHA256 or ticket["p0_manifest_sha256"] != P0_SHA256:
        raise FormalError("protocol/P0 anchor mismatch")
    for key in ("orchestrator_sha256", "engine_sha256", "spec_sha256", "authorization_sha256", "engine_audit_sha256"):
        if not is_sha256(ticket[key]):
            raise FormalError(f"invalid {key}")
    if ticket["engine_sha256"] != sha256_file(ENGINE_PATH):
        raise FormalError("ticket does not pin the executing engine")
    if ticket["execution_rules"] != EXECUTION_RULES:
        raise FormalError("execution rules changed")
    if ticket["required_payload_files"] != list(PAYLOAD_FILES):
        raise FormalError("payload list changed")
    if ticket["csv_schemas"] != {k: list(v) for k, v in CSV_SCHEMAS.items()}:
        raise FormalError("CSV schemas changed")
    if ticket["arms"] != list(ARMS) or ticket["contrasts"] != list(CONTRASTS):
        raise FormalError("arms/contrasts changed")
    stage = Path(ticket["stage_directory"])
    require_plain_directory_chain(stage, FORMAL_ROOT)
    if stage.parent != FORMAL_ROOT or not stage.name.startswith(f".{unit_name(unit)}.staging-"):
        raise FormalError("stage name/scope mismatch")
    for name in PAYLOAD_FILES:
        if os.path.lexists(stage / name):
            raise FormalError("staging payload already exists")
    return ticket, unit, stage


def validate_whitelist(ticket: Mapping[str, Any], unit: tuple[str, int, int]) -> dict[str, Path]:
    records = ticket["input_whitelist"]
    if not isinstance(records, list) or len(records) != 12 or sha256_json(records) != ticket["input_whitelist_sha256"]:
        raise FormalError("input whitelist digest/count mismatch")
    prefix = f"results/deadline_nested_20260912/{unit_name(unit)}"
    expected = list(CONTROLS[unit[0]]) + [f"{prefix}/{name}" for name in ARCHIVE_FILES]
    observed: list[str] = []
    paths: dict[str, Path] = {}
    for record in records:
        require_exact_keys(record, ("path", "bytes", "sha256", "hash_source", "role"), "whitelist record")
        rel = _safe_rel_path(record["path"]); observed.append(rel)
        path = ROOT / rel
        require_plain_file(path, f"whitelisted input {rel}")
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise FormalError(f"whitelisted input hash/size mismatch: {rel}")
        expected_role = "complete_run_archive" if rel.startswith(prefix + "/") else "control"
        if record["role"] != expected_role or record["hash_source"] not in ("P0_HASH_MANIFEST", "P1_PROTOCOL.fixed_inputs"):
            raise FormalError("whitelist role/hash source changed")
        paths[rel] = path
    if observed != expected or len(set(observed)) != 12:
        raise FormalError("whitelist is not exact or ordered")
    return paths


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float)
    out = np.empty_like(z); pos = z >= 0
    out[pos] = 1 / (1 + np.exp(-z[pos])); ez = np.exp(z[~pos]); out[~pos] = ez / (1 + ez)
    return out


def logit(p: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    q = np.clip(np.asarray(p, float), epsilon, 1 - epsilon)
    return np.log(q) - np.log1p(-q)


def transform(x: np.ndarray, representation: str, widths: list[int] | None, epsilon: float) -> np.ndarray:
    x = np.asarray(x, float)
    if representation == "soft": return x.copy()
    if widths is None:
        if representation == "hard": return (x >= .5).astype(float)
        return logit(x, epsilon)
    blocks, start = [], 0
    for width in widths:
        block = x[:, start:start + width]; start += width
        if representation == "hard": blocks.append(np.eye(width)[np.argmax(block, axis=1)])
        else:
            z = np.log(np.clip(block, epsilon, 1.0)); blocks.append(z[:, 1:] - z[:, :1])
    if start != x.shape[1]: raise FormalError("concept block widths do not cover features")
    return np.concatenate(blocks, axis=1)


class Head:
    def __init__(self, c: float, max_iter: int):
        self.c = float(c); self.max_iter = int(max_iter); self.model: LogisticRegression | None = None
    def fit(self, x: np.ndarray, y: np.ndarray) -> "Head":
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            self.model = LogisticRegression(C=self.c, class_weight="balanced", max_iter=self.max_iter, solver="lbfgs", random_state=20260912).fit(x, y)
        return self
    def logits(self, x: np.ndarray) -> np.ndarray:
        assert self.model is not None
        return np.asarray(self.model.decision_function(x), float)
    def probabilities(self, x: np.ndarray) -> np.ndarray: return sigmoid(self.logits(x))
    def record(self) -> dict[str, object]:
        assert self.model is not None
        return {"C": self.c, "coef": self.model.coef_.tolist(), "intercept": self.model.intercept_.tolist()}


def class_equal_nll(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, int); loss = -(y * np.log(np.clip(p, 1e-12, 1)) + (1-y) * np.log(np.clip(1-p, 1e-12, 1)))
    if set(y.tolist()) != {0, 1}: raise FormalError("binary classes absent")
    return float((loss[y == 0].mean() + loss[y == 1].mean()) / 2)


def frozen_seed(base: int, *parts: object) -> int:
    """Byte-for-byte seed derivation used by the frozen Derm runner."""
    payload = "|".join([str(base), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") & 0x7FFFFFFF


def case_kfold_splits(n_cases: int, n_splits: int, seed: int) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    return tuple((np.asarray(tr, int), np.asarray(va, int)) for tr, va in splitter.split(np.arange(n_cases)))


def fit_head(x: np.ndarray, y: np.ndarray, c: float, max_iter: int, counter: dict[str, Any], role: str) -> Head:
    started = time.perf_counter(); head = Head(c, max_iter).fit(x, y)
    counter["head_fits"] += 1; counter["head_seconds"] += time.perf_counter() - started
    counter["head_by_role"][role] = counter["head_by_role"].get(role, 0) + 1
    return head


def select_c(blocks: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], y: np.ndarray, max_iter: int, counter: dict[str, Any], role: str) -> tuple[float, dict[str, object]]:
    candidates = []
    for c in GRID:
        losses = [class_equal_nll(y[va], fit_head(xf, y[tr], c, max_iter, counter, role).probabilities(xv)) for tr, va, xf, xv in blocks]
        candidates.append({"C": c, "fold_class_equal_NLL": losses, "mean_class_equal_NLL": float(np.mean(losses))})
    index = int(np.argmin([r["mean_class_equal_NLL"] for r in candidates]))
    return GRID[index], {"C_grid": list(GRID), "candidates": candidates, "selected_index": index, "selected_C": GRID[index], "tie_break": "first_minimum_smallest_C"}


def fit_affine(z: np.ndarray, y: np.ndarray, counter: dict[str, Any], role: str) -> tuple[float, float, dict[str, object]]:
    z = np.asarray(z, float); y = np.asarray(y, int); started = time.perf_counter()
    if z.ndim != 1 or y.shape != z.shape or set(y.tolist()) != {0, 1} or not np.isfinite(z).all():
        raise FormalError(f"invalid affine inputs: {role}")
    counts = np.bincount(y, minlength=2).astype(float)
    weights = 1.0 / (2.0 * counts[y])
    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = theta[0] * z + theta[1]
        value = float(np.sum(weights * (np.logaddexp(0.0, eta) - y * eta)))
        residual = weights * (sigmoid(eta) - y)
        return value, np.asarray([np.sum(residual * z), np.sum(residual)], float)
    identity_value = objective(np.asarray([1.0, 0.0]))[0]
    result = minimize(objective, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B",
                      bounds=((1e-6, None), (None, None)),
                      options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-10, "maxls": 50})
    if (not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.fun)
            or float(result.x[0]) <= 0 or float(result.fun) > identity_value + 1e-10):
        raise FormalError(f"affine calibration failed: {role}")
    gradient = objective(np.asarray(result.x, float))[1]
    kkt = max(abs(float(gradient[1])), abs(float(gradient[0])) if result.x[0] > 1.000001e-6 else max(0.0, -float(gradient[0])))
    if kkt > 1e-5: raise FormalError(f"affine KKT check failed: {role}")
    counter["affine_fits"] += 1; counter["affine_seconds"] += time.perf_counter() - started
    a, b = map(float, result.x)
    return a, b, {"role": role, "slope": a, "intercept": b, "optimizer_success": True,
                  "positive_slope": a > 0, "identity_class_equal_NLL": identity_value,
                  "oof_class_equal_NLL": float(result.fun), "kkt_residual": float(kkt)}


def restore(scores: np.ndarray, gold: np.ndarray, fit_rows: np.ndarray, widths: list[int] | None,
            epsilon: float, valid: np.ndarray | None = None) -> np.ndarray:
    scores = np.asarray(scores, float); fit_rows = np.asarray(fit_rows, int)
    if widths is None:
        q = np.asarray(scores, float)
        state = np.stack((1.0-q, q), axis=2)
        if valid is None or valid.shape != gold.shape:
            raise FormalError("CBIS concept-validity mask is absent or misaligned")
        priors = []
        for column in range(gold.shape[1]):
            concept_rows = fit_rows[valid[fit_rows, column]]
            if len(concept_rows) == 0:
                raise FormalError("CBIS prior fit has no valid concept labels")
            priors.append(float(np.mean(gold[concept_rows, column])))
        prior = np.asarray(priors, float)
        state_prior = np.stack((1.0-prior, prior), axis=1)
        un = np.maximum(state, epsilon) * state_prior[None, :, :]
        denominator = np.sum(un, axis=2)
        if not np.isfinite(un).all() or np.any(denominator <= 0): raise FormalError("invalid CBIS prior restoration")
        return un[:, :, 1] / denominator
    out, start = np.empty_like(scores), 0
    for width in widths:
        block = np.asarray(scores[:, start:start+width], float)
        states = np.argmax(gold[fit_rows, start:start+width], axis=1)
        prior = np.bincount(states, minlength=width).astype(float); prior /= prior.sum()
        un = np.maximum(block, epsilon) * prior[None, :]
        denominator = un.sum(axis=1, keepdims=True)
        if not np.isfinite(un).all() or np.any(denominator <= 0): raise FormalError("invalid Derm prior restoration")
        out[:, start:start+width] = un / denominator; start += width
    if start != scores.shape[1]: raise FormalError("restoration blocks do not cover scores")
    return out


def cbis_frozen_concept_fit_records(
        frozen_records: Sequence[Mapping[str, Any]], concept_names: Sequence[str],
        fit_rows: np.ndarray, predict_rows: np.ndarray, ids: np.ndarray,
        groups: np.ndarray, *, prediction_manifest_required: bool
        ) -> dict[str, Mapping[str, Any]]:
    """Read the producer's concept records and optional strict prediction tail."""
    concept_keys = {
        "concept", "fit_breasts", "fit_patients", "negative", "positive",
        "fit_unit_sha256", "invalid_targets_excluded_by_family_validity_mask",
        "fixed_probe_C", "parameter_count",
    }
    prediction_keys = {
        "prediction_role", "predict_breasts", "predict_patients",
        "predict_unit_sha256", "fit_predict_patient_disjoint",
        "patient_disjoint_required", "overlap_patient_count",
    }
    expected_count = len(concept_names) + int(prediction_manifest_required)
    if (not isinstance(frozen_records, (list, tuple))
            or len(frozen_records) != expected_count
            or any(not isinstance(item, Mapping) for item in frozen_records)):
        raise FormalError("CBIS frozen concept-fit record schema changed")
    concept_records = frozen_records[:len(concept_names)]
    if any(set(item) != concept_keys for item in concept_records):
        raise FormalError("CBIS frozen concept-fit record schema changed")
    names = [item["concept"] for item in concept_records]
    if (any(not isinstance(name, str) for name in names)
            or len(set(names)) != len(concept_names)
            or set(names) != set(concept_names)):
        raise FormalError("CBIS frozen concept-fit set changed")
    if prediction_manifest_required:
        prediction = frozen_records[-1]
        fit_groups = set(groups[fit_rows].astype(str))
        predict_groups = set(groups[predict_rows].astype(str))
        if (set(prediction) != prediction_keys
                or prediction["prediction_role"] != "CASE_EXCLUDED_PREDICTED_CONCEPT_EXPOSURE"
                or type(prediction["predict_breasts"]) is not int
                or prediction["predict_breasts"] != len(predict_rows)
                or type(prediction["predict_patients"]) is not int
                or prediction["predict_patients"] != len(predict_groups)
                or prediction["predict_unit_sha256"] != sha256_values(ids[predict_rows])
                or prediction["fit_predict_patient_disjoint"] is not True
                or prediction["patient_disjoint_required"] is not True
                or type(prediction["overlap_patient_count"]) is not int
                or prediction["overlap_patient_count"] != 0
                or fit_groups.intersection(predict_groups)):
            raise FormalError("CBIS frozen prediction-range lineage mismatch")
    return dict(zip(names, concept_records, strict=True))


def prior_lineage_records(role: str, fit_rows: np.ndarray, predict_rows: np.ndarray,
                          gold: np.ndarray, widths: list[int] | None,
                          ids: np.ndarray, groups: np.ndarray, valid: np.ndarray | None = None,
                          frozen_concept_fits: Sequence[Mapping[str, Any]] | None = None,
                          concept_names: Sequence[str] | None = None,
                          manifest_reference: str = "",
                          prediction_manifest_required: bool = False) -> list[dict[str, object]]:
    """Record every independently recomputed empirical concept prior."""
    fit_rows = np.asarray(fit_rows, int); predict_rows = np.asarray(predict_rows, int)
    records: list[dict[str, object]] = []
    if widths is None:
        if valid is None or valid.shape != gold.shape or concept_names is None:
            raise FormalError("CBIS prior lineage lacks concept-validity controls")
        if len(concept_names) != gold.shape[1] or frozen_concept_fits is None:
            raise FormalError("CBIS frozen concept-fit lineage is absent")
        fit_by_name = cbis_frozen_concept_fit_records(
            frozen_concept_fits, concept_names, fit_rows, predict_rows, ids, groups,
            prediction_manifest_required=prediction_manifest_required)
        blocks = []
        for column, name in enumerate(concept_names):
            concept_rows = fit_rows[valid[fit_rows, column]]
            frozen = fit_by_name[name]
            states = gold[concept_rows, column].astype(int)
            counts = np.bincount(states, minlength=2).astype(int)
            if (int(frozen.get("fit_breasts", -1)) != len(concept_rows)
                    or int(frozen.get("fit_patients", -1)) != len(set(groups[concept_rows]))
                    or int(frozen.get("negative", -1)) != int(counts[0])
                    or int(frozen.get("positive", -1)) != int(counts[1])
                    or frozen.get("fit_unit_sha256") != sha256_values(ids[concept_rows])):
                raise FormalError("CBIS frozen concept-fit lineage mismatch")
            blocks.append((column, states, 2, concept_rows, name, frozen))
    else:
        blocks, start = [], 0
        for concept, width in enumerate(widths):
            states = np.argmax(gold[fit_rows, start:start + width], axis=1)
            blocks.append((concept, states, width, fit_rows, None, None)); start += width
        if start != gold.shape[1]:
            raise FormalError("prior lineage widths do not cover gold features")
    for concept, states, width, concept_rows, name, frozen in blocks:
        counts = np.bincount(states, minlength=width).astype(int)
        prior = counts.astype(float) / counts.sum()
        records.append({
            "role": role, "concept_block": int(concept),
            "concept_name": name,
            "fit_row_indices": concept_rows.tolist(),
            "fit_case_scope_sha256": scope_sha(concept_rows, ids),
            "fit_group_scope_sha256": scope_sha(concept_rows, groups),
            "predict_case_scope_sha256": scope_sha(predict_rows, ids),
            "predict_group_scope_sha256": scope_sha(predict_rows, groups),
            "state_counts": counts.tolist(), "empirical_prior": prior.tolist(),
            "prior_recomputed_from_fit_rows": True,
            "predict_rows_excluded_from_prior_fit": not bool(set(concept_rows).intersection(predict_rows)),
            "frozen_generating_fit_manifest": frozen,
            "frozen_fit_manifest_verified": frozen is not None,
            "manifest_reference": manifest_reference,
        })
    return records


def load_cbis_control(path: Path, data: Mapping[str, np.ndarray], fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Bind the panel to the whitelisted cohort and recover concept-validity masks."""
    with np.load(path, allow_pickle=False) as source:
        needed = {"complete_concept_eligible", "concept_names", "unit_id", "patient_id",
                  "outcome_id", "outer_fold", "concept_multi_hot", "concept_family_valid"}
        if not needed.issubset(source.files):
            raise FormalError("CBIS cohort control fields are incomplete")
        eligible = source["complete_concept_eligible"].astype(bool)
        names = source["concept_names"].astype(str).tolist()
        indices = []
        for name in CBIS_CONCEPTS:
            matches = [i for i, observed in enumerate(names) if observed == name]
            if len(matches) != 1:
                raise FormalError("CBIS concept control is not uniquely identified")
            indices.append(matches[0])
        source_ids = source["unit_id"].astype(str)[eligible]
        source_groups = source["patient_id"].astype(str)[eligible]
        source_y = source["outcome_id"].astype(int)[eligible]
        source_fold = source["outer_fold"].astype(int)[eligible]
        source_gold = source["concept_multi_hot"].astype(float)[eligible][:, indices]
        family_valid = source["concept_family_valid"].astype(bool)[eligible]
        source_valid = np.column_stack((np.repeat(family_valid[:, [0]], 5, axis=1),
                                        np.repeat(family_valid[:, [1]], 5, axis=1)))
    if (not np.array_equal(source_ids, data["ids"].astype(str))
            or not np.array_equal(source_groups, data["groups"].astype(str))
            or not np.array_equal(source_y, data["y"].astype(int))
            or not np.array_equal(source_gold, data["gold"].astype(float))):
        raise FormalError("CBIS panel does not reproduce the whitelisted cohort")
    if not np.all(source_valid):
        raise FormalError("CBIS panel includes concept-invalid rows")
    if (set(np.flatnonzero(source_fold != fold).tolist()) != set(data["t"].astype(int).tolist())
            or set(np.flatnonzero(source_fold == fold).tolist()) != set(data["e"].astype(int).tolist())):
        raise FormalError("CBIS outer-fold control does not reproduce T/E")
    return source_valid, source_fold


def resolve_cbis_manifests(generation: Mapping[str, Any], data: Mapping[str, np.ndarray],
                           groups: np.ndarray, source_fold: np.ndarray,
                           outer_fold: int) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    manifests = generation.get("manifests")
    selected = generation.get("selection_manifests")
    if not isinstance(manifests, list) or len(manifests) != 5:
        raise FormalError("CBIS generation manifest count changed")
    h_manifests: list[Mapping[str, Any] | None] = [None] * 4
    final: Mapping[str, Any] | None = None
    for item in manifests:
        if not isinstance(item, Mapping):
            raise FormalError("CBIS generation manifest is not an object")
        if item.get("role") == "final_outer_train_probe":
            if final is not None:
                raise FormalError("duplicate CBIS final generation manifest")
            final = item
            continue
        if item.get("outer_fold") != outer_fold:
            raise FormalError("CBIS H generation outer-fold mismatch")
        heldout = item.get("diagnosis_selection_heldout_fold")
        matches = [h for h in range(4)
                   if set(data[f"h{h}"].astype(int).tolist())
                   == set(np.flatnonzero(source_fold == int(heldout)).tolist())]
        if len(matches) != 1 or h_manifests[matches[0]] is not None:
            raise FormalError("CBIS H generation manifest cannot be resolved")
        h_manifests[matches[0]] = item
    if final is None or any(item is None for item in h_manifests):
        raise FormalError("CBIS generation manifests are incomplete")
    resolved = [item for item in h_manifests if item is not None]
    if (not isinstance(selected, list) or len(selected) != 4
            or any(not isinstance(item, list) or len(item) != 1 for item in selected)
            or any(sum(block[0] == item for block in selected) != 1 for item in resolved)):
        raise FormalError("CBIS selection-manifest projection changed")
    t = data["t"].astype(int); e = data["e"].astype(int)
    if (manifests[4] != final
            or final.get("nested_probe_panels") not in (None, [])
            or final.get("outer_fold") != outer_fold
            or final.get("fit_patient_sha256") != sha256_values(groups[t])
            or final.get("validation_patient_sha256") != sha256_values(groups[e])
            or final.get("patient_disjoint") is not True):
        raise FormalError("CBIS final generation scope mismatch")
    return resolved, final


def _nested_subfolds(domain: str, rep: int, fold: int, h: int, frows: np.ndarray,
                     hrows: Sequence[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    # In CBIS, the other three archived H blocks are exactly the original
    # patient-fold G partition of F. Derm's frozen runner uses a deterministic
    # four-way case partition; contiguous array_split after the runner's seeded
    # permutation is reconstructed from the archived f ordering below.
    members = [np.intersect1d(frows, block, assume_unique=False) for block in hrows]
    members = [m for m in members if len(m)]
    if domain == "cbis" and len(members) == 3:
        return [(np.setdiff1d(frows, g), g) for g in members]
    if domain == "derm":
        local = case_kfold_splits(len(frows), 4, frozen_seed(2026090829, "strict_meta_inner", rep, fold, h, "split"))
        return [(frows[tr], frows[va]) for tr, va in local]
    raise FormalError("cannot reconstruct frozen nested subfolds")


def read_json(path: Path) -> Any:
    require_plain_file(path, str(path)); return json.loads(path.read_text())


def _archive_condition(rows: list[dict[str, str]], condition: str) -> np.ndarray:
    selected = [r for r in rows if r.get("condition") == condition]
    return np.asarray([float(r["p1"]) for r in selected], float)


def _validate_archive_predictions(rows: list[dict[str, str]], unit: tuple[str, int, int],
                                  e: np.ndarray, ids: np.ndarray, groups: np.ndarray,
                                  y: np.ndarray) -> None:
    """Validate identities and ordering before comparing archived probabilities."""
    domain, rep, fold = unit
    conditions = ("raw_gold_nested", "raw_gold_substitution", "raw_soft_nested")
    for condition in conditions:
        selected = [row for row in rows if row.get("condition") == condition]
        if len(selected) != len(e):
            raise FormalError("archive condition row count changed")
        for observed, index in zip(selected, e, strict=True):
            if (observed.get("domain") != domain
                    or int(observed.get("repetition", "-1")) != rep
                    or int(observed.get("outer_fold", "-1")) != fold
                    or observed.get("unit_id") != str(ids[index])
                    or observed.get("group_id") != str(groups[index])
                    or int(observed.get("label", "-1")) != int(y[index])):
                raise FormalError("archive prediction identity/order mismatch")


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle: return list(csv.DictReader(handle))


def validate_archive_selection_curve(
    archived: object,
    condition: str,
    expected_selected_c: float,
    computed: Mapping[str, object],
) -> None:
    """Validate the archived candidate curve, not a nonexistent singleton row."""
    error = {"code": "E_ARCHIVE_SELECTION_CONTRACT"}
    if not isinstance(archived, list):
        raise FormalError("archive selection object changed type", **error)
    rows = [
        row for row in archived
        if isinstance(row, Mapping) and row.get("condition") == condition
    ]
    if len(rows) != len(GRID):
        raise FormalError("archive selection candidate count changed", **error)
    try:
        c_values = [float(row["C"]) for row in rows]
        inner = np.asarray([float(row["inner_nll"]) for row in rows], dtype=float)
        fold_losses = [np.asarray(row["fold_losses"], dtype=float) for row in rows]
        candidates = computed["candidates"]
        selected_index = int(computed["selected_index"])
        computed_selected_c = float(computed["selected_C"])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise FormalError("archive selection fields changed", **error) from exc
    if (
        c_values != list(GRID)
        or inner.shape != (len(GRID),)
        or not np.isfinite(inner).all()
        or any(values.shape != (4,) or not np.isfinite(values).all() for values in fold_losses)
        or any(
            not np.isclose(inner[index], float(np.mean(fold_losses[index])), atol=1e-15, rtol=0)
            for index in range(len(GRID))
        )
    ):
        raise FormalError("archive selection curve structure changed", **error)
    archive_selected_index = int(np.argmin(inner))
    if (
        float(GRID[archive_selected_index]) != float(expected_selected_c)
        or selected_index != archive_selected_index
        or computed_selected_c != float(expected_selected_c)
        or not isinstance(candidates, list)
        or len(candidates) != len(GRID)
    ):
        raise FormalError("archive selection first-minimum changed", **error)
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise FormalError("computed selection candidate changed type", **error)
        try:
            candidate_c = float(candidate["C"])
            candidate_losses = np.asarray(candidate["fold_class_equal_NLL"], dtype=float)
            candidate_mean = float(candidate["mean_class_equal_NLL"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FormalError("computed selection fields changed", **error) from exc
        if (
            candidate_c != c_values[index]
            or candidate_losses.shape != (4,)
            or not np.allclose(candidate_losses, fold_losses[index], atol=1e-13, rtol=0)
            or not np.isclose(candidate_mean, inner[index], atol=1e-13, rtol=0)
        ):
            raise FormalError("computed selection curve does not reproduce archive", **error)


def validate_generation_partition(generation: Mapping[str, Any], domain: str,
                                  t: np.ndarray, e: np.ndarray,
                                  hrows: list[np.ndarray]) -> list[Any]:
    """Validate cache metadata and the unordered, disjoint inner partition.

    Cache rows use a local row-index lookup later; concatenating validation
    folds need not reproduce the sorted outer-training row order. ``seconds``
    is archived timing metadata, not a selection input.
    """
    expected_keys = {"selection_manifests", "probe_bundles", "seconds"}
    if domain == "cbis":
        expected_keys.add("manifests")
    if not isinstance(generation, dict) or set(generation) != expected_keys:
        raise FormalError("generation manifest structure changed")
    selections = generation["selection_manifests"]
    if not isinstance(selections, list) or len(selections) != 4:
        raise FormalError("generation selection manifest count changed")
    if generation["probe_bundles"] != (11 if domain == "cbis" else 21):
        raise FormalError("generation probe bundle count changed")
    if len(set(t.tolist())) != len(t) or len(set(e.tolist())) != len(e):
        raise FormalError("duplicate T/E row")
    if (set(t.tolist()) & set(e.tolist())
            or set(np.concatenate(hrows).tolist()) != set(t.tolist())
            or sum(map(len, hrows)) != len(t)):
        raise FormalError("T/E/H partition failure")
    if any(set(hr.tolist()) & set(e.tolist()) for hr in hrows):
        raise FormalError("E enters H")
    return selections


def compute(ticket: Mapping[str, Any], unit: tuple[str, int, int], paths: Mapping[str, Path]) -> dict[str, Any]:
    set_failure_stage("COMPUTE_ARCHIVE_LOAD")
    domain, rep, fold = unit; prefix = f"results/deadline_nested_20260912/{unit_name(unit)}"
    panel_path = paths[f"{prefix}/panel.npz"]
    with np.load(panel_path, allow_pickle=False) as archive:
        required = ("y", "ids", "groups", "gold", "widths", "t", "e", "raw", "val") + tuple(f"{k}{h}" for h in range(4) for k in ("f", "h", "x", "z"))
        if set(archive.files) != set(required): raise FormalError("panel fields changed")
        data = {k: archive[k].copy() for k in required}
    generation = read_json(paths[f"{prefix}/generation.json"])
    archived_predictions = _csv_rows(paths[f"{prefix}/predictions.csv"])
    archived_selection = read_json(paths[f"{prefix}/selection.json"])
    archived_heads = read_json(paths[f"{prefix}/heads.json"])
    archived_complete = read_json(paths[f"{prefix}/COMPLETE.json"])
    set_failure_stage("COMPUTE_STRUCTURE_CONTRACT")
    y = data["y"].astype(int); ids = data["ids"].astype(str); groups = data["groups"].astype(str); gold = data["gold"].astype(float)
    t = data["t"].astype(int); e = data["e"].astype(int); raw = data["raw"].astype(float); val = data["val"].astype(float)
    widths = data["widths"].astype(int).tolist() or None; eps = 1e-6 if domain == "cbis" else 1e-12; max_iter = 10000 if domain == "cbis" else 2000
    hrows = [data[f"h{h}"].astype(int) for h in range(4)]
    selection_manifests = validate_generation_partition(generation, domain, t, e, hrows)
    if archived_complete.get("domain") != domain or archived_complete.get("repetition") != rep or archived_complete.get("outer_fold") != fold:
        raise FormalError("archive COMPLETE unit mismatch")
    if domain == "cbis" and (set(groups[t]).intersection(groups[e])
                              or any(set(groups[hr]).intersection(groups[e]) for hr in hrows)):
        raise FormalError("CBIS patient leakage")
    cbis_valid: np.ndarray | None = None
    cbis_fold: np.ndarray | None = None
    cbis_manifests: list[Mapping[str, Any]] | None = None
    cbis_final: Mapping[str, Any] | None = None
    set_failure_stage("COMPUTE_GENERATION_LINEAGE")
    if domain == "cbis":
        control_path = paths["data/CBIS_DDSM/paper_b_cbis_mass_train_breast_cohort_v1_20260910.npz"]
        cbis_valid, cbis_fold = load_cbis_control(control_path, data, fold)
        cbis_manifests, cbis_final = resolve_cbis_manifests(generation, data, groups, cbis_fold, fold)
    _validate_archive_predictions(archived_predictions, unit, e, ids, groups, y)

    set_failure_stage("COMPUTE_NESTED_FITS")
    counter = {"head_fits": 0, "head_seconds": 0.0, "head_by_role": {}, "affine_fits": 0, "affine_seconds": 0.0, "prior_calls": 0, "prior_records": 0}
    raw_blocks, restored_blocks, calibration_rows, prior_records, reference_selections = [], [], [], [], []
    raw_ref_oof = np.full(len(t), np.nan); restored_ref_oof = np.full(len(t), np.nan)
    oof_write_count = np.zeros(len(t), int); tpos = {int(row): i for i, row in enumerate(t)}
    all_h = hrows
    for h in range(4):
        frows = data[f"f{h}"].astype(int); hr = hrows[h]; xh = data[f"x{h}"].astype(float); zh = data[f"z{h}"].astype(float)
        if set(frows.tolist()) != set(t.tolist()) - set(hr.tolist()) or set(frows) & set(hr): raise FormalError("F/H partition failure")
        if set(groups[frows]).intersection(groups[hr]) and domain == "cbis": raise FormalError("CBIS F/H patient leakage")
        manifest = selection_manifests[h]
        if domain == "cbis":
            assert cbis_manifests is not None and cbis_fold is not None
            panel_manifest = cbis_manifests[h]
            nested = panel_manifest.get("nested_probe_panels")
            if not isinstance(nested, list) or len(nested) != 3:
                raise FormalError("CBIS nested generation lineage missing")
            g_splits = []
            for inner in nested:
                if not isinstance(inner, Mapping):
                    raise FormalError("CBIS nested generation record changed")
                gfold = int(inner.get("probe_heldout_fold", -1))
                va = np.flatnonzero(cbis_fold == gfold)
                if not set(va.tolist()).issubset(set(frows.tolist())):
                    raise FormalError("CBIS G fold is outside F")
                g_splits.append((np.setdiff1d(frows, va), va))
        else:
            g_splits = _nested_subfolds(domain, rep, fold, h, frows, [r for j, r in enumerate(all_h) if j != h])
        g_validation = np.concatenate([g for _, g in g_splits])
        if (len(g_splits) not in (3, 4) or len(set(g_validation.tolist())) != len(frows)
                or set(g_validation.tolist()) != set(frows.tolist())): raise FormalError("G partition failure")
        for tr, va in g_splits:
            if set(tr.tolist()) & set(va.tolist()) or set(va.tolist()) & set(e.tolist()): raise FormalError("G exclusion failure")
            if domain == "cbis" and set(groups[tr]).intersection(groups[va]): raise FormalError("CBIS G patient leakage")
        if domain == "derm":
            if (not isinstance(manifest, dict)
                    or list(map(str, manifest.get("fit_case_ids", []))) != ids[frows].tolist()
                    or list(map(str, manifest.get("heldout_case_ids", []))) != ids[hr].tolist()
                    or not isinstance(manifest.get("inner"), list)
                    or len(manifest["inner"]) != 4):
                raise FormalError("Derm generation manifest scope mismatch")
            for inner, (tr, va) in zip(manifest["inner"], g_splits, strict=True):
                if (inner.get("fit_case_id_sha256") != sha256_values(ids[tr])
                        or inner.get("validation_case_id_sha256") != sha256_values(ids[va])
                        or inner.get("fit_validation_disjoint") is not True):
                    raise FormalError("Derm inner generation lineage mismatch")
        else:
            if (panel_manifest.get("fit_patient_sha256") != sha256_values(groups[frows])
                    or panel_manifest.get("heldout_patient_sha256") != sha256_values(groups[hr])
                    or panel_manifest.get("fit_heldout_patient_disjoint") is not True):
                raise FormalError("CBIS generation fit scope mismatch")
            for inner, (tr, va) in zip(nested, g_splits, strict=True):
                if (inner.get("probe_fit_patient_sha256") != sha256_values(groups[tr])
                        or inner.get("probe_validation_patient_sha256") != sha256_values(groups[va])
                        or inner.get("diagnosis_validation_excluded_from_probe_fit") is not True
                        or inner.get("probe_validation_excluded_from_probe_fit") is not True):
                    raise FormalError("CBIS inner generation lineage mismatch")
        ref_blocks = [(tr, va, gold[tr], gold[va]) for tr, va in g_splits]
        c_ref, ref_sel = select_c(ref_blocks, y, max_iter, counter, f"reference_H{h}_selection")
        ref_head = fit_head(gold[frows], y[frows], c_ref, max_iter, counter, f"reference_H{h}_final")
        raw_ref = ref_head.logits(zh)
        restored_x = np.empty_like(xh)
        for g, (tr, va) in enumerate(g_splits):
            local = np.flatnonzero(np.isin(frows, va)); restored_x[local] = restore(xh[local], gold, tr, widths, eps, cbis_valid)
            counter["prior_calls"] += 1
            frozen_fits = nested[g].get("concept_fits") if domain == "cbis" else None
            new_records = prior_lineage_records(
                f"H{h}_G{g}_OOF", tr, va, gold, widths, ids, groups, cbis_valid,
                frozen_fits, CBIS_CONCEPTS if domain == "cbis" else None,
                f"selection_manifests[{h}].nested_probe_panels[{g}]" if domain == "cbis" else "",
                prediction_manifest_required=domain == "cbis")
            counter["prior_records"] += len(new_records); prior_records.extend(new_records)
        restored_z = restore(zh, gold, frows, widths, eps, cbis_valid); counter["prior_calls"] += 1; counter["prior_records"] += (gold.shape[1] if widths is None else len(widths))
        h_fits = panel_manifest.get("full_meta_fit_concept_fits") if domain == "cbis" else None
        prior_records.extend(prior_lineage_records(
            f"H{h}_heldout", frows, hr, gold, widths, ids, groups, cbis_valid,
            h_fits, CBIS_CONCEPTS if domain == "cbis" else None,
            f"selection_manifests[{h}].full_meta_fit_concept_fits" if domain == "cbis" else ""))
        restored_ref = ref_head.logits(restored_z)
        positions = np.asarray([tpos[int(row)] for row in hr]); raw_ref_oof[positions] = raw_ref; restored_ref_oof[positions] = restored_ref
        oof_write_count[positions] += 1
        raw_blocks.append((frows, hr, xh, zh)); restored_blocks.append((frows, hr, restored_x, restored_z))
        reference_selections.append({"H": h, **ref_sel})
        for j, row in enumerate(hr):
            calibration_rows.append({"domain": domain, "repetition": rep, "outer_fold": fold, "row_index": int(row), "case_id": ids[row], "group_id": groups[row], "y": int(y[row]), "H": h, "reference_C_H": c_ref, "raw_base_logit": raw_ref[j], "prior_restored_base_logit": restored_ref[j], "probe_fit_scope_sha256": scope_sha(frows, ids), "reference_fit_scope_sha256": scope_sha(frows, ids)})

    if not (np.all(oof_write_count == 1) and np.isfinite(raw_ref_oof).all() and np.isfinite(restored_ref_oof).all()):
        raise FormalError("stitched calibration OOF coverage is not exactly once")
    set_failure_stage("COMPUTE_OUTER_FITS")
    c_gold, sel_gold = select_c([(tr, va, gold[tr], gold[va]) for tr, va, _, _ in raw_blocks], y, max_iter, counter, "outer_reference_selection")
    c_raw, sel_raw = select_c(raw_blocks, y, max_iter, counter, "outer_raw_matched_selection")
    c_restored, sel_restored = select_c(restored_blocks, y, max_iter, counter, "outer_restored_matched_selection")
    ref_final = fit_head(gold[t], y[t], c_gold, max_iter, counter, "outer_reference_final")
    matched_raw = fit_head(raw, y[t], c_raw, max_iter, counter, "outer_raw_matched_final")
    restored_raw = restore(raw, gold, t, widths, eps, cbis_valid); restored_val = restore(val, gold, t, widths, eps, cbis_valid)
    counter["prior_calls"] += 1
    assert domain != "cbis" or cbis_final is not None
    final_fits = cbis_final.get("concept_fits") if domain == "cbis" else None
    outer_records = prior_lineage_records(
        "outer_E", t, e, gold, widths, ids, groups, cbis_valid, final_fits,
        CBIS_CONCEPTS if domain == "cbis" else None,
        "manifests.final_outer_train_probe.concept_fits" if domain == "cbis" else "")
    counter["prior_records"] += len(outer_records); prior_records.extend(outer_records)
    matched_restored = fit_head(restored_raw, y[t], c_restored, max_iter, counter, "outer_restored_matched_final")
    a_raw, b_raw, cal_raw = fit_affine(raw_ref_oof, y[t], counter, "C")
    a_rest, b_rest, cal_rest = fit_affine(restored_ref_oof, y[t], counter, "CP")
    logits_s = ref_final.logits(val); logits_sp = ref_final.logits(restored_val)
    probabilities = {
        "R": ref_final.probabilities(gold[e]), "S": sigmoid(logits_s), "C": sigmoid(a_raw * logits_s + b_raw),
        "M": matched_raw.probabilities(val), "SP": sigmoid(logits_sp), "CP": sigmoid(a_rest * logits_sp + b_rest),
        "MP": matched_restored.probabilities(restored_val),
    }
    if not all(v.shape == (len(e),) and np.isfinite(v).all() and np.all((v >= 0) & (v <= 1)) for v in probabilities.values()): raise FormalError("invalid/alignment probability failure")

    # Frozen archive reproduction is a mandatory audit, not an alternative input.
    set_failure_stage("COMPUTE_ARCHIVE_PREDICTION_REPRODUCTION")
    archive_by_condition = {name: _archive_condition(archived_predictions, name) for name in ("raw_gold_nested", "raw_gold_substitution", "raw_soft_nested")}
    if not (np.allclose(probabilities["R"], archive_by_condition["raw_gold_nested"], atol=1e-10, rtol=1e-10) and np.allclose(probabilities["S"], archive_by_condition["raw_gold_substitution"], atol=1e-10, rtol=1e-10) and np.allclose(probabilities["M"], archive_by_condition["raw_soft_nested"], atol=1e-10, rtol=1e-10)):
        raise FormalError("R/S/M do not reproduce frozen archive")
    set_failure_stage("COMPUTE_ARCHIVE_SELECTION_REPRODUCTION")
    archived_c = float(archived_heads["raw_soft_nested"]["C"])
    if c_raw != archived_c: raise FormalError("M selected C does not reproduce archive")
    validate_archive_selection_curve(
        archived_selection, "raw_soft_nested", c_raw, sel_raw
    )

    set_failure_stage("COMPUTE_COST_CONTRACT")
    cost = EXPECTED_COST[domain]
    if counter["head_fits"] != cost["head_fits"] or counter["affine_fits"] != 2 or counter["prior_calls"] != cost["prior_restoration_calls"] or counter["prior_records"] != cost["prior_concept_records"]:
        raise FormalError(f"static compute contract mismatch: {counter}")
    return {"y": y, "ids": ids, "groups": groups, "t": t, "e": e, "probabilities": probabilities,
            "logits": {"S": logits_s, "C": a_raw*logits_s+b_raw, "SP": logits_sp, "CP": a_rest*logits_sp+b_rest},
            "calibration_rows": calibration_rows, "prior_records": prior_records,
            "selections": {"reference_H": reference_selections, "R_S_SP": sel_gold, "M": sel_raw, "MP": sel_restored},
            "calibrators": {"C": cal_raw, "CP": cal_rest}, "heads": {"R_S_SP": ref_final.record(), "M": matched_raw.record(), "MP": matched_restored.record()},
            "counter": counter, "generation_sha256": sha256_file(paths[f"{prefix}/generation.json"]), "archive_selection_sha256": sha256_file(paths[f"{prefix}/selection.json"]),
            "archive_reproduction": True, "selection_reproduction": True,
            "generation_loaded": isinstance(generation, dict), "archive_selection_loaded": isinstance(archived_selection, list),
            "audit_observations": {"H_blocks": 4, "H_exactly_once": True, "G_partition_exact": True,
                                   "E_excluded": True, "CBIS_group_disjoint": domain != "cbis" or True,
                                   "calibration_oof_exactly_once": True, "archive_tolerance_atol": 1e-10,
                                   "archive_tolerance_rtol": 1e-10, "positive_affine_slopes": [a_raw, a_rest],
                                   "no_bootstrap": True, "units_run": 1,
                                   "formal_ticket_from_stdin": True, "hypothesis_judgment_performed": False}}


def payloads(ticket: Mapping[str, Any], unit: tuple[str, int, int], result: Mapping[str, Any]) -> tuple[dict[str, list[dict[str, object]]], dict[str, dict[str, object]], str]:
    domain, rep, fold = unit; y = result["y"]; ids = result["ids"]; groups = result["groups"]; e = result["e"]; probs = result["probabilities"]; logits = result["logits"]
    prediction_rows = []
    for j, row in enumerate(e):
        prediction_rows.append({"domain": domain, "repetition": rep, "outer_fold": fold, "row_index": int(row), "case_id": ids[row], "group_id": groups[row], "y": int(y[row]), **{f"p_{a}": float(probs[a][j]) for a in ARMS}, "logit_S": float(logits["S"][j]), "logit_C": float(logits["C"][j]), "logit_SP": float(logits["SP"][j]), "logit_CP": float(logits["CP"][j])})
    case_rows, metric_rows = [], []
    for arm in ARMS:
        yy = y[e]; pp = probs[arm]; losses = -(yy*np.log(np.clip(pp,1e-12,1))+(1-yy)*np.log(np.clip(1-pp,1e-12,1)))
        weights = np.where(yy == 0, .5/np.sum(yy==0), .5/np.sum(yy==1))
        for j, row in enumerate(e):
            case_rows.append({"domain": domain, "repetition": rep, "outer_fold": fold, "row_index": int(row), "case_id": ids[row], "group_id": groups[row], "y": int(yy[j]), "arm": arm, "negative_log_likelihood": float(losses[j]), "squared_probability_error": float((pp[j]-yy[j])**2), "class_equal_nll_contribution": float(losses[j]*weights[j])})
        metric_rows.append({"domain": domain, "repetition": rep, "outer_fold": fold, "arm": arm, "class_equal_NLL": class_equal_nll(yy,pp), "AUC": float(roc_auc_score(yy,pp)), "natural_frequency_NLL": float(np.mean(losses)), "Brier": float(np.mean((pp-yy)**2)), "n_cases": len(e), "n_groups": len(set(groups[e]))})
    metric = {r["arm"]: r for r in metric_rows}
    pairs = {"C_minus_M":("C","M"), "S_minus_SP":("S","SP"), "S_minus_C":("S","C"), "CP_minus_MP":("CP","MP"), "M_minus_MP":("M","MP")}
    paired = [{"domain":domain,"repetition":rep,"outer_fold":fold,"contrast":name,"class_equal_NLL_X_minus_Y":float(metric[x]["class_equal_NLL"]-metric[z]["class_equal_NLL"]),"AUC_X_minus_Y":float(metric[x]["AUC"]-metric[z]["AUC"]),"favourable_sign_NLL":CONTRAST_SIGNS[name],"pilot_descriptive_only":False} for name,(x,z) in pairs.items()]
    base = {"unit": unit_object(unit)}
    jsons = {
        "selections.json": {**base, "selection_rule":"mean inner class-equal NLL; first minimum/smallest C", "C_grid":list(GRID), "selections":result["selections"], "final_heads":result["heads"]},
        "calibrators.json": {**base, "calibrators":result["calibrators"], "fit_scope":"stitched T out-of-fold reference-head logits only", "evaluation_scope":"E excluded"},
        "prior_lineage.json": {**base, "method":"fold-local empirical-prior restoration", "records":result["prior_records"], "outer_e_fit_scope_sha256":scope_sha(result["t"],ids)},
        "lineage.json": {**base, "ticket_sha256":ticket["ticket_sha256"], "anchors":{k:ticket[k] for k in ("protocol_sha256","p0_manifest_sha256","orchestrator_sha256","engine_sha256","spec_sha256","authorization_sha256","engine_audit_sha256")}, "access_boundary":{"input_whitelist_exact":True,"input_whitelist_sha256":ticket["input_whitelist_sha256"],"official_test_access":False,"official_test_discovery_or_listing":False,"unit_discovery":False,"pilot_or_formal_result_discovery":False,"pilot_external_audit_read_by_engine":False}, "archive_generation_sha256":result["generation_sha256"], "archive_selection_sha256":result["archive_selection_sha256"]},
        "audit.json": {**base, "checks":{k:True for k in F_CHECKS}, "all_formal_prepublication_audits_pass":True, "hypothesis_judgment":"NOT_PERMITTED_PER_UNIT", "publication_audit":"F20_RESERVED_FOR_ORCHESTRATOR"},
        "fit_cost.json": {**base, "head_fits":{"total_head_fits":result["counter"]["head_fits"],"seconds":result["counter"]["head_seconds"],"by_role":result["counter"]["head_by_role"]}, "affine_fits":{"fits":result["counter"]["affine_fits"],"seconds":result["counter"]["affine_seconds"]}, "prior_restoration":{"calls":result["counter"]["prior_calls"],"concept_records":result["counter"]["prior_records"]}, "probe_generation":{"fits":0}, "bootstrap_draws":0, "units_run":1},
    }
    tables = {"predictions.csv":prediction_rows, "calibration_oof.csv":result["calibration_rows"], "per_case_effects.csv":case_rows, "metrics.csv":metric_rows, "paired_effects.csv":paired}
    log = "\n".join((f"unit={unit_name(unit)}", "mode=FORMAL_PREPUBLICATION_ONE_UNIT", "cpu_only=true", "threads=1", "bootstrap_draws=0", "hypothesis_judgment=NOT_PERMITTED_PER_UNIT", "pilot_external_audit=ORCHESTRATOR_TRUST_BOUNDARY_NOT_READ", "official_test_access=false", "payload_count=12")) + "\n"
    return tables, jsons, log


def write_payload(stage: Path, tables: Mapping[str, Sequence[Mapping[str, object]]], jsons: Mapping[str, Mapping[str, object]], log: str) -> None:
    for name in CSV_SCHEMAS: write_new_csv(stage / name, CSV_SCHEMAS[name], tables[name])
    for name in JSON_PAYLOADS: write_new_json(stage / name, jsons[name])
    write_new_bytes(stage / "run.log", log.encode())
    fd = os.open(stage, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def main(argv: Sequence[str] | None = None) -> int:
    set_failure_stage("ARGUMENT_CONTRACT")
    parser = argparse.ArgumentParser(add_help=False); parser.add_argument("--formal-ticket-stdin", action="store_true")
    args, unknown = parser.parse_known_args(argv)
    if unknown or not args.formal_ticket_stdin or tuple(sys.argv[1:] if argv is None else argv) != ("--formal-ticket-stdin",): raise FormalError("only --formal-ticket-stdin is permitted")
    set_failure_stage("THREAD_ENV_CONTRACT")
    if any(os.environ.get(k) != v for k,v in _THREAD_ENV.items()): raise FormalError("CPU serial environment is not exact")
    set_failure_stage("TICKET_READ_PARSE")
    raw = sys.stdin.buffer.read(2_000_001)
    if not raw or len(raw) > 2_000_000: raise FormalError("ticket stdin size invalid")
    parsed_ticket = json.loads(raw)
    set_failure_stage("TICKET_VALIDATE")
    ticket, unit, stage = validate_ticket(parsed_ticket)
    set_failure_stage("WHITELIST_PRE")
    paths = validate_whitelist(ticket, unit)
    with threadpool_limits(limits=1): result = compute(ticket, unit, paths)
    # Rehash after computation before any publication-stage payload is written.
    set_failure_stage("WHITELIST_POST")
    validate_whitelist(ticket, unit)
    set_failure_stage("PAYLOAD_BUILD")
    tables, jsons, log = payloads(ticket, unit, result)
    set_failure_stage("PAYLOAD_WRITE")
    write_payload(stage, tables, jsons, log)
    set_failure_stage("SUCCESS_RECEIPT")
    receipt = {"status":"PAYLOAD_COMPLETE","unit":unit_name(unit),"ticket_sha256":ticket["ticket_sha256"],"engine_sha256":ticket["engine_sha256"],"protocol_sha256":ticket["protocol_sha256"],"input_whitelist_sha256":ticket["input_whitelist_sha256"],"payload_files_sha256":sha256_json(list(PAYLOAD_FILES)),"official_test_accessed":False,"hypothesis_judgment_performed":False}
    sys.stdout.write(json.dumps(receipt, separators=(",",":"), ensure_ascii=False) + "\n"); sys.stdout.flush(); return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Emit one fixed-schema, non-data-bearing receipt.  Exception text,
        # tracebacks, paths, identifiers, hashes, and observed values stay out.
        sys.stderr.write(
            json.dumps(
                sanitized_failure_receipt(exc),
                sort_keys=True,
                separators=(",", ":"),
            ) + "\n"
        )
        sys.stderr.flush()
        raise SystemExit(2)
