#!/usr/bin/env python3
"""Fail-closed aggregation for the frozen Paper-B P1 development run.

The production entry point has no data, unit, seed, draw-count, or output-path
selector.  It reads the literal 30-unit scope only after a separately reviewed
source receipt and authorization have been hash-pinned.  No official-test path
is represented in this program.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import stat
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# Set the process boundary before NumPy is imported.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np


SELF_PATH = Path(os.path.abspath(__file__))
ROOT = SELF_PATH.parent.parent

PROTOCOL_REL = "paper_b_readiness/P1_PROTOCOL.json"
SPEC_REL = "paper_b_readiness/P1_AGGREGATE_SPEC_V7_20260916.json"
SOURCE_AUDIT_REL = "paper_b_readiness/P1_AGGREGATE_SOURCE_AUDIT_V7_20260916.json"
AUTHORIZATION_REL = "paper_b_readiness/P1_AGGREGATE_AUTHORIZATION_V7_20260916.json"
UNITS_PARENT_REL = "paper_b_readiness/p1_complete/units"
FINAL_REL = "paper_b_readiness/p1_complete/aggregate"
LOCK_REL = "paper_b_readiness/p1_complete/.aggregate.lock"

PROTOCOL_SHA256 = "bdaca53493521a400f6307088d887eafa035f7eed4a094d522d05ee2ea5a1c1b"
# Filled only after the source-only specification is frozen.
SPEC_SHA256 = "1db0635bca0926249f45c7f9687e78d274d08ce8053d9e095796bda24152c5da"
UNITS_SHA256 = "76c7c9e2fbd8e9fe7318e7fa3d5a18aa349d7ccb90af8e6a4423c9c62dcd1e2b"
BOOTSTRAP_SEED = 2026091517
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_MAX_PROPOSALS = 100_000
NUMERIC_ATOL = 1e-10

ARMS = ("R", "S", "C", "M", "SP", "CP", "MP")
ARM_METRICS = ("class_equal_NLL", "AUC", "natural_frequency_NLL", "Brier")
CONTRASTS = (
    "C_minus_M",
    "S_minus_SP",
    "S_minus_C",
    "CP_minus_MP",
    "M_minus_MP",
)
CONTRAST_ARMS = {
    "C_minus_M": ("C", "M"),
    "S_minus_SP": ("S", "SP"),
    "S_minus_C": ("S", "C"),
    "CP_minus_MP": ("CP", "MP"),
    "M_minus_MP": ("M", "MP"),
}
CONTRAST_SIGNS = {
    "C_minus_M": "negative_favours_C",
    "S_minus_SP": "positive_favours_SP",
    "S_minus_C": "positive_favours_C",
    "CP_minus_MP": "negative_favours_CP",
    "M_minus_MP": "positive_favours_MP",
}

FORMAL_AUDIT_IDS = tuple(f"F{i:02d}" for i in range(1, 20))
FORMAL_PUBLICATION_AUDIT_ID = "F20"
AGGREGATE_AUDIT_IDS = tuple(f"G{i:02d}" for i in range(1, 19))
AGGREGATE_PUBLICATION_AUDIT_ID = "G19"

UNITS = (
    ("cbis", 0, 0), ("cbis", 0, 1), ("cbis", 0, 2),
    ("cbis", 0, 3), ("cbis", 0, 4),
    ("derm", 0, 0), ("derm", 0, 1), ("derm", 0, 2),
    ("derm", 0, 3), ("derm", 0, 4),
    ("derm", 1, 0), ("derm", 1, 1), ("derm", 1, 2),
    ("derm", 1, 3), ("derm", 1, 4),
    ("derm", 2, 0), ("derm", 2, 1), ("derm", 2, 2),
    ("derm", 2, 3), ("derm", 2, 4),
    ("derm", 3, 0), ("derm", 3, 1), ("derm", 3, 2),
    ("derm", 3, 3), ("derm", 3, 4),
    ("derm", 4, 0), ("derm", 4, 1), ("derm", 4, 2),
    ("derm", 4, 3), ("derm", 4, 4),
)

CSV_SCHEMAS = {
    "predictions.csv": (
        "domain", "repetition", "outer_fold", "row_index", "case_id",
        "group_id", "y", "p_R", "p_S", "p_C", "p_M", "p_SP", "p_CP",
        "p_MP", "logit_S", "logit_C", "logit_SP", "logit_CP",
    ),
    "calibration_oof.csv": (
        "domain", "repetition", "outer_fold", "row_index", "case_id",
        "group_id", "y", "H", "reference_C_H", "raw_base_logit",
        "prior_restored_base_logit", "probe_fit_scope_sha256",
        "reference_fit_scope_sha256",
    ),
    "per_case_effects.csv": (
        "domain", "repetition", "outer_fold", "row_index", "case_id",
        "group_id", "y", "arm", "negative_log_likelihood",
        "squared_probability_error", "class_equal_nll_contribution",
    ),
    "metrics.csv": (
        "domain", "repetition", "outer_fold", "arm", "class_equal_NLL",
        "AUC", "natural_frequency_NLL", "Brier", "n_cases", "n_groups",
    ),
    "paired_effects.csv": (
        "domain", "repetition", "outer_fold", "contrast",
        "class_equal_NLL_X_minus_Y", "AUC_X_minus_Y",
        "favourable_sign_NLL", "pilot_descriptive_only",
    ),
}
MERGED_NAMES = {
    name: f"all_{name}" for name in CSV_SCHEMAS
}
JSON_PAYLOADS = (
    "selections.json", "calibrators.json", "prior_lineage.json",
    "lineage.json", "audit.json", "fit_cost.json",
)
PAYLOAD_FILES = tuple(CSV_SCHEMAS) + JSON_PAYLOADS + ("run.log",)
UNIT_FILES = tuple(CSV_SCHEMAS) + JSON_PAYLOADS + (
    "execution_manifest.json", "run.log", "COMPLETE.json",
)
AGGREGATE_EFFECT_SCHEMA = (
    "domain", "effect_type", "name", "metric", "estimate", "ci_lower",
    "ci_upper", "n_bootstrap_valid", "n_clusters", "n_cases",
    "n_repetitions", "direction",
)
BOOTSTRAP_EFFECT_SCHEMA = (
    "domain", "draw", "proposal", "effect_type", "name", "metric", "value",
)
AGGREGATE_PAYLOAD_FILES = (
    "all_predictions.csv", "all_calibration_oof.csv",
    "all_per_case_effects.csv", "all_metrics.csv", "all_paired_effects.csv",
    "aggregate_effects.csv", "bootstrap_effects.csv", "aggregate_audit.json",
)
AGGREGATE_FILES = AGGREGATE_PAYLOAD_FILES + (
    "execution_manifest.json", "COMPLETE.json",
)

SOURCE_AUDIT_KEYS = (
    "status", "scope", "protocol_sha256", "aggregate_script_sha256",
    "aggregate_spec_sha256", "units_sha256", "source_review_complete",
    "synthetic_contract_tests_pass", "exact_unit_validation_reviewed",
    "case_first_estimators_reviewed", "paired_cluster_bootstrap_reviewed",
    "atomic_publication_reviewed", "p1_outcomes_accessed",
    "official_test_accessed", "reviewer", "completed_at_utc",
)
AUTHORIZATION_KEYS = (
    "status", "authorized", "protocol_sha256", "aggregate_script_sha256",
    "aggregate_spec_sha256", "units_sha256", "source_audit_path",
    "source_audit_sha256", "formal_units_complete_attested",
    "aggregate_overlay_approved", "issued_at_utc",
)

FORMAL_MANIFEST_PATHS = {
    "protocol_path": "paper_b_readiness/P1_PROTOCOL.json",
    "orchestrator_path": "scripts/run_p1_formal_orchestrator_v7_20260916.py",
    "engine_path": "scripts/run_p1_formal_unit_v5_20260916.py",
    "spec_path": "paper_b_readiness/P1_FORMAL_ORCHESTRATION_SPEC_V7_20260916.json",
    "authorization_path": "paper_b_readiness/P1_FORMAL_AUTHORIZATION_V7_20260916.json",
    "engine_audit_path": "paper_b_readiness/P1_FORMAL_ENGINE_AUDIT_V7_20260916.json",
}

# Mechanical source-chain successor only.  These result-blind source/spec pins
# identify the reviewed V7 orchestrator and V5 engine; no live gate or outcome
# file is inspected to obtain them.  Authorization/audit hashes remain dynamic
# but must still be valid and identical across the exact 30 formal units.
FORMAL_SOURCE_SHA256 = {
    "orchestrator_sha256": "89feb3433063bba330011fffa88d81aca86259328ead62092debe3bb448e39f0",
    "engine_sha256": "41d4311ea93eb622b2e08fd9098c941a18daeaae8264c085a41b309a0741750f",
    "spec_sha256": "80baf0c8a4bace22de7932ddb752a77ce5227d9da6df9e34a329076741809afd",
}
UNIT_MANIFEST_KEYS = (
    "unit", "protocol_path", "protocol_sha256", "p0_manifest_sha256",
    "orchestrator_path", "orchestrator_sha256", "engine_path",
    "engine_sha256", "spec_path", "spec_sha256", "authorization_path",
    "authorization_sha256", "engine_audit_path", "engine_audit_sha256",
    "units_sha256", "fixed_input_whitelist", "fixed_input_whitelist_sha256",
    "artifacts", "artifact_count_excluding_manifest_and_COMPLETE",
    "complete_excluded_because_written_last", "formal_prepublication_audit_ids",
    "formal_publication_audit_id", "scientific_interpretation",
)
UNIT_COMPLETE_KEYS = (
    "domain", "repetition", "outer_fold", "status", "protocol_sha256",
    "execution_manifest_sha256", "formal_prepublication_audit_ids",
    "formal_publication_audit_id", "all_formal_publication_audits_pass",
    "completed_at_utc", "interpretation",
)
AGGREGATE_AUDIT_KEYS = (
    "status", "scope", "protocol_sha256", "aggregate_spec_sha256",
    "aggregate_script_sha256", "source_audit_sha256", "authorization_sha256",
    "units_sha256", "unit_manifest_anchors", "formal_unit_common_anchors",
    "checks", "check_meanings", "all_aggregate_prepublication_audits_pass",
    "hypothesis_judgment", "bootstrap", "synthetic_test_mode",
    "official_test_access", "official_test_discovery_or_listing",
)
AGGREGATE_MANIFEST_KEYS = (
    "scope", "protocol_path", "protocol_sha256", "aggregate_script_path",
    "aggregate_script_sha256", "aggregate_spec_path", "aggregate_spec_sha256",
    "source_audit_path", "source_audit_sha256", "authorization_path",
    "authorization_sha256", "units_sha256", "unit_execution_manifests",
    "artifacts", "artifact_count_excluding_manifest_and_COMPLETE",
    "complete_excluded_because_written_last", "aggregate_prepublication_audit_ids",
    "aggregate_publication_audit_id", "scientific_interpretation",
)
AGGREGATE_COMPLETE_KEYS = (
    "status", "scope", "protocol_sha256", "execution_manifest_sha256",
    "aggregate_prepublication_audit_ids", "aggregate_publication_audit_id",
    "all_aggregate_publication_audits_pass", "completed_at_utc",
    "hypothesis_judgment",
)


class AggregateError(RuntimeError):
    """A fail-closed input, statistical, gate, or publication error."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return sha256_bytes(stable_json_bytes(value))


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def unit_object(unit: tuple[str, int, int]) -> dict[str, object]:
    return {"domain": unit[0], "repetition": unit[1], "outer_fold": unit[2]}


def unit_name(unit: tuple[str, int, int]) -> str:
    return f"{unit[0]}_r{unit[1]}_f{unit[2]}"


def lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise AggregateError(f"cannot inspect path: {path}") from exc


def require_plain_directory(path: Path, label: str) -> None:
    metadata = lstat_or_none(path)
    if metadata is None or stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise AggregateError(f"{label} is absent, a symlink, or not a directory: {path}")


def require_regular_file(path: Path, label: str) -> None:
    metadata = lstat_or_none(path)
    if metadata is None or stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise AggregateError(f"{label} is absent, a symlink, or not a regular file: {path}")


def exact_names(path: Path) -> tuple[str, ...]:
    require_plain_directory(path, "fixed directory")
    return tuple(sorted(entry.name for entry in path.iterdir()))


def load_json(path: Path, label: str) -> Any:
    require_regular_file(path, label)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AggregateError(f"{label} is not valid UTF-8 JSON") from exc


def write_new_bytes(path: Path, payload: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)


def write_new_json(path: Path, value: object) -> None:
    write_new_bytes(path, stable_json_bytes(value))


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def require_exact_keys(value: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    expected = set(keys)
    observed = set(value)
    if observed != expected:
        raise AggregateError(
            f"{label} keys differ; missing={sorted(expected-observed)}, "
            f"unexpected={sorted(observed-expected)}"
        )


def parse_int(value: str, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise AggregateError(f"{label} is not an integer") from exc


def parse_float(value: str, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AggregateError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise AggregateError(f"{label} is not finite")
    return result


def read_csv_exact(path: Path, schema: Sequence[str]) -> list[dict[str, str]]:
    require_regular_file(path, path.name)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(schema):
            raise AggregateError(f"{path.name} does not have the frozen schema")
        rows = list(reader)
    if not rows or any(set(row) != set(schema) for row in rows):
        raise AggregateError(f"{path.name} is empty or malformed")
    return rows


def write_csv(path: Path, schema: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="", closefd=False) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(schema), lineterminator="\n")
            writer.writeheader()
            for row in rows:
                if set(row) != set(schema):
                    raise AggregateError(f"writer received a row outside schema for {path.name}")
                writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)


def float_text(value: float) -> str:
    if not math.isfinite(value):
        raise AggregateError("attempted to publish a non-finite numeric value")
    return format(float(value), ".17g")


def binary_nll(y: int, p: float) -> float:
    if not 0.0 <= p <= 1.0:
        raise AggregateError("probability lies outside [0,1]")
    if y == 1:
        if p == 0.0:
            raise AggregateError("positive case has infinite negative log likelihood")
        return -math.log(p)
    if p == 1.0:
        raise AggregateError("negative case has infinite negative log likelihood")
    return -math.log1p(-p)


def weighted_auc(y: np.ndarray, score: np.ndarray, weight: np.ndarray) -> float:
    if y.ndim != 1 or score.shape != y.shape or weight.shape != y.shape:
        raise AggregateError("AUC arrays are not aligned")
    if np.any(~np.isfinite(score)) or np.any(~np.isfinite(weight)) or np.any(weight < 0):
        raise AggregateError("AUC inputs are non-finite or negatively weighted")
    positive = y == 1
    negative = y == 0
    pos_total = float(np.sum(weight[positive]))
    neg_total = float(np.sum(weight[negative]))
    if pos_total <= 0.0 or neg_total <= 0.0:
        raise AggregateError("AUC is undefined because one class is absent")
    order = np.argsort(score, kind="mergesort")
    ys = y[order]
    ss = score[order]
    ws = weight[order]
    numerator = 0.0
    negative_below = 0.0
    start = 0
    while start < len(ss):
        stop = start + 1
        while stop < len(ss) and ss[stop] == ss[start]:
            stop += 1
        tie_pos = float(np.sum(ws[start:stop][ys[start:stop] == 1]))
        tie_neg = float(np.sum(ws[start:stop][ys[start:stop] == 0]))
        numerator += tie_pos * (negative_below + 0.5 * tie_neg)
        negative_below += tie_neg
        start = stop
    return numerator / (pos_total * neg_total)


def class_equal_mean(y: np.ndarray, values: np.ndarray, weight: np.ndarray) -> float:
    parts = []
    for label in (0, 1):
        mask = y == label
        denominator = float(np.sum(weight[mask]))
        if denominator <= 0.0:
            raise AggregateError("class-equal mean is undefined because one class is absent")
        parts.append(float(np.sum(values[mask] * weight[mask])) / denominator)
    return 0.5 * (parts[0] + parts[1])


def weighted_mean(values: np.ndarray, weight: np.ndarray) -> float:
    denominator = float(np.sum(weight))
    if denominator <= 0.0:
        raise AggregateError("weighted mean has zero mass")
    return float(np.sum(values * weight)) / denominator


def close(observed: float, expected: float, label: str, atol: float = NUMERIC_ATOL) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=atol):
        raise AggregateError(f"{label} mismatch: observed={observed!r}, expected={expected!r}")


def validate_gate(root: Path) -> dict[str, str]:
    protocol_path = root / PROTOCOL_REL
    spec_path = root / SPEC_REL
    audit_path = root / SOURCE_AUDIT_REL
    authorization_path = root / AUTHORIZATION_REL
    for path, expected, label in (
        (protocol_path, PROTOCOL_SHA256, "frozen P1 protocol"),
        (spec_path, SPEC_SHA256, "aggregate specification"),
    ):
        require_regular_file(path, label)
        if sha256_file(path) != expected:
            raise AggregateError(f"{label} SHA-256 mismatch")
    script_sha = sha256_file(SELF_PATH)
    common = {
        "protocol_sha256": PROTOCOL_SHA256,
        "aggregate_script_sha256": script_sha,
        "aggregate_spec_sha256": SPEC_SHA256,
        "units_sha256": UNITS_SHA256,
    }
    audit = load_json(audit_path, "independent aggregate source audit")
    if not isinstance(audit, dict):
        raise AggregateError("aggregate source audit is not an object")
    require_exact_keys(audit, SOURCE_AUDIT_KEYS, "aggregate source audit")
    for key, value in {
        "status": "PASS",
        "scope": "P1_COMPLETE_AGGREGATE_PRE_OUTCOME",
        **common,
    }.items():
        if audit.get(key) != value:
            raise AggregateError(f"aggregate source audit does not pin {key}")
    for key in (
        "source_review_complete", "synthetic_contract_tests_pass",
        "exact_unit_validation_reviewed", "case_first_estimators_reviewed",
        "paired_cluster_bootstrap_reviewed", "atomic_publication_reviewed",
    ):
        if audit.get(key) is not True:
            raise AggregateError(f"aggregate source audit has not passed {key}")
    if audit.get("p1_outcomes_accessed") is not False or audit.get("official_test_accessed") is not False:
        raise AggregateError("pre-outcome source review crossed an outcome/test boundary")
    audit_sha = sha256_file(audit_path)
    authorization = load_json(authorization_path, "aggregate authorization")
    if not isinstance(authorization, dict):
        raise AggregateError("aggregate authorization is not an object")
    require_exact_keys(authorization, AUTHORIZATION_KEYS, "aggregate authorization")
    expected_auth = {
        "status": "AUTHORIZED_PRE_OUTCOME",
        "authorized": True,
        **common,
        "source_audit_path": SOURCE_AUDIT_REL,
        "source_audit_sha256": audit_sha,
        "formal_units_complete_attested": True,
        "aggregate_overlay_approved": True,
    }
    for key, value in expected_auth.items():
        if authorization.get(key) != value:
            raise AggregateError(f"aggregate authorization does not pin {key}")
    return {
        **common,
        "source_audit_sha256": audit_sha,
        "authorization_sha256": sha256_file(authorization_path),
    }


def validate_unit(
    unit_path: Path,
    unit: tuple[str, int, int],
) -> tuple[dict[str, list[dict[str, str]]], dict[str, str]]:
    if exact_names(unit_path) != tuple(sorted(UNIT_FILES)):
        raise AggregateError(f"{unit_name(unit)} does not contain exactly 14 frozen files")
    for name in UNIT_FILES:
        require_regular_file(unit_path / name, f"{unit_name(unit)}/{name}")

    manifest = load_json(unit_path / "execution_manifest.json", "unit execution manifest")
    complete = load_json(unit_path / "COMPLETE.json", "unit COMPLETE")
    if not isinstance(manifest, dict) or not isinstance(complete, dict):
        raise AggregateError("unit publication metadata is not object-valued")
    require_exact_keys(manifest, UNIT_MANIFEST_KEYS, "unit execution manifest")
    require_exact_keys(complete, UNIT_COMPLETE_KEYS, "unit COMPLETE")
    expected_unit = unit_object(unit)
    if manifest.get("unit") != expected_unit:
        raise AggregateError("unit manifest identity mismatch")
    if any(complete.get(key) != value for key, value in expected_unit.items()):
        raise AggregateError("unit COMPLETE identity mismatch")
    if complete.get("status") != "COMPLETE":
        raise AggregateError("unit COMPLETE status is not COMPLETE")
    if complete.get("protocol_sha256") != PROTOCOL_SHA256 or manifest.get("protocol_sha256") != PROTOCOL_SHA256:
        raise AggregateError("unit does not pin the frozen P1 protocol")
    if complete.get("execution_manifest_sha256") != sha256_file(unit_path / "execution_manifest.json"):
        raise AggregateError("unit COMPLETE does not anchor its execution manifest")
    if complete.get("formal_prepublication_audit_ids") != list(FORMAL_AUDIT_IDS):
        raise AggregateError("unit COMPLETE F01--F19 identifiers changed")
    if complete.get("formal_publication_audit_id") != FORMAL_PUBLICATION_AUDIT_ID:
        raise AggregateError("unit COMPLETE F20 identifier changed")
    if complete.get("all_formal_publication_audits_pass") is not True:
        raise AggregateError("unit publication audit did not pass")
    if manifest.get("formal_prepublication_audit_ids") != list(FORMAL_AUDIT_IDS):
        raise AggregateError("unit manifest F01--F19 identifiers changed")
    if manifest.get("formal_publication_audit_id") != FORMAL_PUBLICATION_AUDIT_ID:
        raise AggregateError("unit manifest F20 identifier changed")
    if manifest.get("units_sha256") != UNITS_SHA256:
        raise AggregateError("unit manifest formal-scope digest changed")
    for key, value in FORMAL_MANIFEST_PATHS.items():
        if manifest.get(key) != value:
            raise AggregateError(f"unit manifest source path changed: {key}")
    for key, value in FORMAL_SOURCE_SHA256.items():
        if manifest.get(key) != value:
            raise AggregateError(f"unit manifest reviewed source hash changed: {key}")
    if manifest.get("scientific_interpretation") != (
        "formal development unit only; no per-unit hypothesis judgment"
    ):
        raise AggregateError("unit manifest contains an unexpected interpretation")
    if complete.get("interpretation") != (
        "formal P1 development unit; aggregate judgment only after all units"
    ):
        raise AggregateError("unit COMPLETE contains an unexpected interpretation")
    if not isinstance(complete.get("completed_at_utc"), str) or not complete["completed_at_utc"]:
        raise AggregateError("unit COMPLETE lacks its completion timestamp")
    if manifest.get("artifact_count_excluding_manifest_and_COMPLETE") != len(PAYLOAD_FILES):
        raise AggregateError("unit manifest artifact count is not 12")
    if manifest.get("complete_excluded_because_written_last") is not True:
        raise AggregateError("unit manifest does not attest COMPLETE-last publication")
    records = manifest.get("artifacts")
    if not isinstance(records, list) or len(records) != len(PAYLOAD_FILES):
        raise AggregateError("unit manifest artifact list is not the exact 12 payloads")
    if [record.get("path") if isinstance(record, dict) else None for record in records] != list(PAYLOAD_FILES):
        raise AggregateError("unit manifest artifact order/set changed")
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise AggregateError("unit manifest contains a malformed artifact record")
        artifact = unit_path / record["path"]
        if not isinstance(record["bytes"], int) or artifact.stat().st_size != record["bytes"]:
            raise AggregateError("unit artifact size differs from manifest")
        if not is_sha256(record["sha256"]) or sha256_file(artifact) != record["sha256"]:
            raise AggregateError("unit artifact hash differs from manifest")
    whitelist = manifest.get("fixed_input_whitelist")
    if (
        not isinstance(whitelist, list)
        or not whitelist
        or not all(isinstance(item, dict) for item in whitelist)
        or manifest.get("fixed_input_whitelist_sha256") != sha256_json(whitelist)
    ):
        raise AggregateError("unit fixed-input whitelist hash is invalid")

    common_anchor_names = (
        "p0_manifest_sha256", "orchestrator_sha256", "engine_sha256",
        "spec_sha256", "authorization_sha256", "engine_audit_sha256",
    )
    anchors: dict[str, str] = {}
    for name in common_anchor_names:
        value = manifest.get(name)
        if not is_sha256(value):
            raise AggregateError(f"unit manifest has an invalid {name}")
        anchors[name] = value

    audit = load_json(unit_path / "audit.json", "unit audit")
    if not isinstance(audit, dict) or audit.get("unit") != expected_unit:
        raise AggregateError("unit audit identity mismatch")
    checks = audit.get("checks")
    if not isinstance(checks, dict) or tuple(checks) != FORMAL_AUDIT_IDS:
        raise AggregateError("unit audit does not contain exactly ordered F01--F19")
    if not all(value is True for value in checks.values()):
        raise AggregateError("unit audit contains a failed F01--F19 check")
    if audit.get("all_formal_prepublication_audits_pass") is not True:
        raise AggregateError("unit prepublication gate is not passed")
    if "all_formal_publication_audits_pass" in audit:
        raise AggregateError("unit engine audit prematurely claims F20")
    if audit.get("hypothesis_judgment") not in (None, "NOT_PERMITTED_PER_UNIT"):
        raise AggregateError("unit audit contains a per-unit scientific judgment")

    json_payloads = {}
    for name in JSON_PAYLOADS:
        value = load_json(unit_path / name, name)
        if not isinstance(value, dict) or value.get("unit") != expected_unit:
            raise AggregateError(f"{name} unit identity mismatch")
        json_payloads[name] = value
    lineage_boundary = json_payloads["lineage.json"].get("access_boundary")
    if not isinstance(lineage_boundary, dict):
        raise AggregateError("unit lineage lacks the access boundary")
    if lineage_boundary.get("official_test_access") is not False:
        raise AggregateError("unit lineage reports official-test access")
    if lineage_boundary.get("official_test_discovery_or_listing") is not False:
        raise AggregateError("unit lineage reports official-test discovery/listing")
    if lineage_boundary.get("input_whitelist_exact") is not True:
        raise AggregateError("unit lineage does not attest the exact input whitelist")

    tables = {name: read_csv_exact(unit_path / name, schema) for name, schema in CSV_SCHEMAS.items()}
    for name, rows in tables.items():
        for row in rows:
            if (
                row["domain"] != unit[0]
                or parse_int(row["repetition"], "repetition") != unit[1]
                or parse_int(row["outer_fold"], "outer_fold") != unit[2]
            ):
                raise AggregateError(f"{name} contains a row from another unit")

    predictions = tables["predictions.csv"]
    keys = []
    prediction_by_key: dict[tuple[int, str, str, int], dict[str, str]] = {}
    for row in predictions:
        y = parse_int(row["y"], "diagnosis label")
        if y not in (0, 1):
            raise AggregateError("prediction contains a non-binary label")
        key = (parse_int(row["row_index"], "row_index"), row["case_id"], row["group_id"], y)
        if key in prediction_by_key or any(existing[1] == row["case_id"] for existing in keys):
            raise AggregateError("prediction unit has duplicate row or case identity")
        keys.append(key)
        prediction_by_key[key] = row
        for arm in ARMS:
            p = parse_float(row[f"p_{arm}"], f"p_{arm}")
            if not 0.0 <= p <= 1.0:
                raise AggregateError("prediction probability lies outside [0,1]")
        for name in ("logit_S", "logit_C", "logit_SP", "logit_CP"):
            parse_float(row[name], name)
    if {key[3] for key in keys} != {0, 1}:
        raise AggregateError("unit outer-evaluation predictions do not contain both classes")

    calibration_keys: set[tuple[int, str, str, int]] = set()
    for row in tables["calibration_oof.csv"]:
        y = parse_int(row["y"], "calibration diagnosis label")
        if y not in (0, 1):
            raise AggregateError("calibration OOF table contains a non-binary label")
        key = (parse_int(row["row_index"], "row_index"), row["case_id"], row["group_id"], y)
        if key in calibration_keys:
            raise AggregateError("calibration OOF table contains a duplicate row identity")
        calibration_keys.add(key)
        # The frozen producer emits h from range(4), including H=0.
        if parse_int(row["H"], "H") not in (0, 1, 2, 3):
            raise AggregateError("calibration OOF table contains H outside zero-based folds 0..3")
        if parse_float(row["reference_C_H"], "reference_C_H") <= 0.0:
            raise AggregateError("calibration OOF table contains non-positive reference_C_H")
        parse_float(row["raw_base_logit"], "raw_base_logit")
        parse_float(row["prior_restored_base_logit"], "prior_restored_base_logit")
        for hash_name in ("probe_fit_scope_sha256", "reference_fit_scope_sha256"):
            if not is_sha256(row[hash_name]):
                raise AggregateError(f"calibration OOF table contains an invalid {hash_name}")

    per_case = tables["per_case_effects.csv"]
    if len(per_case) != len(predictions) * len(ARMS):
        raise AggregateError("per-case table is not seven rows per prediction")
    seen_arm_keys: dict[str, list[tuple[int, str, str, int]]] = defaultdict(list)
    class_counts = Counter(key[3] for key in keys)
    for row in per_case:
        arm = row["arm"]
        if arm not in ARMS:
            raise AggregateError("per-case table contains an unknown arm")
        key = (parse_int(row["row_index"], "row_index"), row["case_id"], row["group_id"], parse_int(row["y"], "y"))
        seen_arm_keys[arm].append(key)
        prediction = prediction_by_key.get(key)
        if prediction is None:
            raise AggregateError("per-case row is not aligned to predictions")
        p = parse_float(prediction[f"p_{arm}"], f"p_{arm}")
        loss = binary_nll(key[3], p)
        brier = (p - key[3]) ** 2
        contribution = loss / (2.0 * class_counts[key[3]])
        close(parse_float(row["negative_log_likelihood"], "per-case NLL"), loss, "per-case NLL")
        close(parse_float(row["squared_probability_error"], "per-case Brier"), brier, "per-case Brier")
        close(parse_float(row["class_equal_nll_contribution"], "class-equal contribution"), contribution, "class-equal contribution")
    if set(seen_arm_keys) != set(ARMS) or any(seen_arm_keys[arm] != keys for arm in ARMS):
        raise AggregateError("per-case rows are not exactly arm-aligned")

    metrics_rows = tables["metrics.csv"]
    if len(metrics_rows) != len(ARMS) or [row["arm"] for row in metrics_rows] != list(ARMS):
        raise AggregateError("metrics table is not in exact seven-arm order")
    y_array = np.asarray([key[3] for key in keys], dtype=np.int8)
    for row in metrics_rows:
        arm = row["arm"]
        p = np.asarray([parse_float(prediction_by_key[key][f"p_{arm}"], f"p_{arm}") for key in keys])
        losses = np.asarray([binary_nll(int(y), float(prob)) for y, prob in zip(y_array, p)])
        brier = (p - y_array) ** 2
        expected_metrics = {
            "class_equal_NLL": class_equal_mean(y_array, losses, np.ones(len(y_array))),
            "AUC": weighted_auc(y_array, p, np.ones(len(y_array))),
            "natural_frequency_NLL": float(np.mean(losses)),
            "Brier": float(np.mean(brier)),
        }
        for metric, expected in expected_metrics.items():
            close(parse_float(row[metric], metric), expected, f"unit {arm} {metric}")
        if parse_int(row["n_cases"], "n_cases") != len(keys):
            raise AggregateError("unit metrics n_cases mismatch")
        if parse_int(row["n_groups"], "n_groups") != len({key[2] for key in keys}):
            raise AggregateError("unit metrics n_groups mismatch")

    paired_rows = tables["paired_effects.csv"]
    if len(paired_rows) != len(CONTRASTS) or [row["contrast"] for row in paired_rows] != list(CONTRASTS):
        raise AggregateError("paired effects are not in exact five-contrast order")
    metric_index = {row["arm"]: row for row in metrics_rows}
    for row in paired_rows:
        name = row["contrast"]
        if row["favourable_sign_NLL"] != CONTRAST_SIGNS[name]:
            raise AggregateError("paired-effect direction changed")
        if row["pilot_descriptive_only"].strip().lower() not in ("false", "0"):
            raise AggregateError("formal paired effect is marked pilot-only")
        x_arm, y_arm = CONTRAST_ARMS[name]
        expected_nll = parse_float(metric_index[x_arm]["class_equal_NLL"], "metric") - parse_float(metric_index[y_arm]["class_equal_NLL"], "metric")
        expected_auc = parse_float(metric_index[x_arm]["AUC"], "metric") - parse_float(metric_index[y_arm]["AUC"], "metric")
        close(parse_float(row["class_equal_NLL_X_minus_Y"], "paired NLL"), expected_nll, "paired NLL")
        close(parse_float(row["AUC_X_minus_Y"], "paired AUC"), expected_auc, "paired AUC")
    return tables, anchors


def point_estimators(
    domain_rows: list[dict[str, str]],
    domain: str,
) -> tuple[dict[tuple[str, str, str], float], dict[str, Any]]:
    repetitions = (0,) if domain == "cbis" else (0, 1, 2, 3, 4)
    by_rep: dict[int, list[dict[str, str]]] = {rep: [] for rep in repetitions}
    case_meta: dict[str, tuple[str, int]] = {}
    for row in domain_rows:
        rep = parse_int(row["repetition"], "repetition")
        if rep not in by_rep:
            raise AggregateError("prediction has a repetition outside frozen scope")
        case_id = row["case_id"]
        meta = (row["group_id"], parse_int(row["y"], "y"))
        old = case_meta.setdefault(case_id, meta)
        if old != meta:
            raise AggregateError("case group or label changes across repetitions")
        by_rep[rep].append(row)
    expected_cases: set[str] | None = None
    for rep in repetitions:
        case_ids = [row["case_id"] for row in by_rep[rep]]
        if len(case_ids) != len(set(case_ids)):
            raise AggregateError("case appears in more than one outer fold within a repetition")
        observed = set(case_ids)
        if expected_cases is None:
            expected_cases = observed
        elif observed != expected_cases:
            raise AggregateError("repetitions do not contain the same case set")
        if {parse_int(row["outer_fold"], "outer_fold") for row in by_rep[rep]} != {0, 1, 2, 3, 4}:
            raise AggregateError("repetition does not cover all five outer folds")
        if {parse_int(row["y"], "y") for row in by_rep[rep]} != {0, 1}:
            raise AggregateError("repetition does not contain both outcome classes")
    assert expected_cases is not None
    case_ids = sorted(expected_cases)
    case_index = {case_id: i for i, case_id in enumerate(case_ids)}
    y_case = np.asarray([case_meta[case_id][1] for case_id in case_ids], dtype=np.int8)
    group_case = np.asarray([case_meta[case_id][0] for case_id in case_ids], dtype=object)
    points: dict[tuple[str, str, str], float] = {}
    rep_probabilities: dict[str, dict[int, np.ndarray]] = {arm: {} for arm in ARMS}
    case_loss: dict[str, np.ndarray] = {}
    case_brier: dict[str, np.ndarray] = {}
    for arm in ARMS:
        loss_matrix = np.empty((len(repetitions), len(case_ids)), dtype=float)
        brier_matrix = np.empty_like(loss_matrix)
        auc_values = []
        for rep_position, rep in enumerate(repetitions):
            ordered = sorted(by_rep[rep], key=lambda row: case_index[row["case_id"]])
            if [row["case_id"] for row in ordered] != case_ids:
                raise AggregateError("case ordering reconstruction failed")
            p = np.asarray([parse_float(row[f"p_{arm}"], f"p_{arm}") for row in ordered])
            rep_probabilities[arm][rep] = p
            losses = np.asarray([binary_nll(int(y), float(prob)) for y, prob in zip(y_case, p)])
            loss_matrix[rep_position] = losses
            brier_matrix[rep_position] = (p - y_case) ** 2
            auc_values.append(weighted_auc(y_case, p, np.ones(len(y_case))))
        mean_loss = np.mean(loss_matrix, axis=0)
        mean_brier = np.mean(brier_matrix, axis=0)
        case_loss[arm] = mean_loss
        case_brier[arm] = mean_brier
        points[("arm", arm, "class_equal_NLL")] = class_equal_mean(y_case, mean_loss, np.ones(len(y_case)))
        points[("arm", arm, "natural_frequency_NLL")] = float(np.mean(mean_loss))
        points[("arm", arm, "Brier")] = float(np.mean(mean_brier))
        points[("arm", arm, "AUC")] = float(np.mean(auc_values))
    for name in CONTRASTS:
        x_arm, y_arm = CONTRAST_ARMS[name]
        points[("contrast", name, "class_equal_NLL")] = points[("arm", x_arm, "class_equal_NLL")] - points[("arm", y_arm, "class_equal_NLL")]
        points[("contrast", name, "AUC")] = points[("arm", x_arm, "AUC")] - points[("arm", y_arm, "AUC")]
    clusters = group_case if domain == "cbis" else np.asarray(case_ids, dtype=object)
    cluster_names = sorted(set(str(value) for value in clusters))
    cluster_index = {name: i for i, name in enumerate(cluster_names)}
    case_cluster_index = np.asarray([cluster_index[str(value)] for value in clusters], dtype=int)
    context = {
        "repetitions": repetitions,
        "case_ids": case_ids,
        "y": y_case,
        "cluster_names": cluster_names,
        "case_cluster_index": case_cluster_index,
        "case_loss": case_loss,
        "case_brier": case_brier,
        "rep_probabilities": rep_probabilities,
    }
    return points, context


def effect_keys() -> tuple[tuple[str, str, str], ...]:
    """Return the one frozen row order used by every aggregate artifact."""
    return tuple(
        [("arm", arm, metric) for arm in ARMS for metric in ARM_METRICS]
        + [
            ("contrast", name, metric)
            for name in CONTRASTS
            for metric in ("class_equal_NLL", "AUC")
        ]
    )


def effect_direction(key: tuple[str, str, str]) -> str:
    effect_type, name, metric = key
    if effect_type == "arm":
        return "higher_better" if metric == "AUC" else "lower_better"
    if metric == "AUC":
        return "positive_favours_X;negative_favours_Y"
    return CONTRAST_SIGNS[name]


def bootstrap_draws(
    domain: str,
    context: Mapping[str, Any],
    rng: np.random.Generator,
    draws: int,
    max_proposals: int,
) -> Iterable[
    tuple[int, int, tuple[tuple[tuple[str, str, str], float], ...]]
]:
    """Yield accepted draws from one whole-cluster RNG stream.

    A single case-weight vector is constructed for a proposal and reused for
    every arm and contrast.  The yielded effect order is frozen by
    :func:`effect_keys` and is replayed during post-write validation.
    """
    y = context["y"]
    repetitions = context["repetitions"]
    n_clusters = len(context["cluster_names"])
    case_cluster_index = context["case_cluster_index"]
    accepted = 0
    proposals = 0
    while accepted < draws and proposals < max_proposals:
        proposals += 1
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        cluster_multiplicity = np.bincount(
            sampled, minlength=n_clusters
        ).astype(float)
        weight = cluster_multiplicity[case_cluster_index]
        # The case set is stable across repetitions.  The explicit loop keeps
        # the protocol's missing-class requirement visible and fail-closed.
        if any(
            float(np.sum(weight[y == label])) <= 0.0
            for _rep in repetitions
            for label in (0, 1)
        ):
            continue
        arm_values: dict[tuple[str, str], float] = {}
        ordered_values: list[tuple[tuple[str, str, str], float]] = []
        for arm in ARMS:
            values = {
                "class_equal_NLL": class_equal_mean(
                    y, context["case_loss"][arm], weight
                ),
                "AUC": float(np.mean([
                    weighted_auc(
                        y, context["rep_probabilities"][arm][rep], weight
                    )
                    for rep in repetitions
                ])),
                "natural_frequency_NLL": weighted_mean(
                    context["case_loss"][arm], weight
                ),
                "Brier": weighted_mean(context["case_brier"][arm], weight),
            }
            for metric in ARM_METRICS:
                value = values[metric]
                arm_values[(arm, metric)] = value
                ordered_values.append((("arm", arm, metric), value))
        for name in CONTRASTS:
            x_arm, y_arm = CONTRAST_ARMS[name]
            for metric in ("class_equal_NLL", "AUC"):
                value = arm_values[(x_arm, metric)] - arm_values[(y_arm, metric)]
                ordered_values.append((("contrast", name, metric), value))
        if tuple(key for key, _value in ordered_values) != effect_keys():
            raise AggregateError("internal bootstrap effect order changed")
        yield accepted, proposals, tuple(ordered_values)
        accepted += 1
    if accepted != draws:
        raise AggregateError(
            f"{domain} produced only {accepted} valid bootstrap draws in "
            f"{proposals} proposals"
        )


def bootstrap_domain(
    domain: str,
    context: Mapping[str, Any],
    rng: np.random.Generator,
    draws: int,
    max_proposals: int,
    writer: csv.DictWriter,
) -> tuple[dict[tuple[str, str, str], list[float]], int]:
    distributions: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    proposals = 0
    for draw, proposal, ordered_values in bootstrap_draws(
        domain, context, rng, draws, max_proposals
    ):
        proposals = proposal
        for key, value in ordered_values:
            effect_type, name, metric = key
            distributions[key].append(value)
            writer.writerow({
                "domain": domain,
                "draw": draw,
                "proposal": proposal,
                "effect_type": effect_type,
                "name": name,
                "metric": metric,
                "value": float_text(value),
            })
    return distributions, proposals


def artifact_records(stage: Path) -> list[dict[str, object]]:
    return [
        {"path": name, "bytes": (stage / name).stat().st_size, "sha256": sha256_file(stage / name)}
        for name in AGGREGATE_PAYLOAD_FILES
    ]


def validate_aggregate_stage(
    path: Path,
    *,
    draws: int,
    max_proposals: int,
    gate_anchors: Mapping[str, str],
    synthetic_test: bool,
    common_unit_anchors: Mapping[str, str],
    unit_manifest_hashes: Sequence[Mapping[str, object]],
    merged_expected: Mapping[str, Sequence[Mapping[str, str]]],
) -> None:
    """Independently read back and recompute the staged aggregate package."""
    if exact_names(path) != tuple(sorted(AGGREGATE_FILES)):
        raise AggregateError("aggregate package is not the exact ten-file set")
    for name in AGGREGATE_FILES:
        require_regular_file(path / name, f"aggregate/{name}")

    merged: dict[str, list[dict[str, str]]] = {}
    expected_units = set(UNITS)
    for source_name, schema in CSV_SCHEMAS.items():
        merged_name = MERGED_NAMES[source_name]
        rows = read_csv_exact(path / merged_name, schema)
        if rows != list(merged_expected[source_name]):
            raise AggregateError(f"{merged_name} is not the exact ordered unit concatenation")
        observed_units = {
            (
                row["domain"],
                parse_int(row["repetition"], "repetition"),
                parse_int(row["outer_fold"], "outer_fold"),
            )
            for row in rows
        }
        if observed_units != expected_units:
            raise AggregateError(f"{merged_name} does not cover the exact 30-unit scope")
        merged[source_name] = rows

    predictions_by_domain = {
        domain: [
            row for row in merged["predictions.csv"] if row["domain"] == domain
        ]
        for domain in ("cbis", "derm")
    }
    points: dict[str, dict[tuple[str, str, str], float]] = {}
    contexts: dict[str, dict[str, Any]] = {}
    for domain in ("cbis", "derm"):
        points[domain], contexts[domain] = point_estimators(
            predictions_by_domain[domain], domain
        )

    # Replay the fixed random stream and compare every published bootstrap row.
    distributions: dict[
        str, dict[tuple[str, str, str], list[float]]
    ] = {
        domain: defaultdict(list) for domain in ("cbis", "derm")
    }
    proposals_by_domain: dict[str, int] = {}
    bootstrap_path = path / "bootstrap_effects.csv"
    with bootstrap_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(BOOTSTRAP_EFFECT_SCHEMA):
            raise AggregateError("bootstrap_effects.csv does not have the frozen schema")
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        for domain in ("cbis", "derm"):
            last_proposal = 0
            for draw, proposal, ordered_values in bootstrap_draws(
                domain, contexts[domain], rng, draws, max_proposals
            ):
                last_proposal = proposal
                for key, expected_value in ordered_values:
                    try:
                        row = next(reader)
                    except StopIteration as exc:
                        raise AggregateError(
                            "bootstrap_effects.csv ended before the frozen draw scope"
                        ) from exc
                    if set(row) != set(BOOTSTRAP_EFFECT_SCHEMA):
                        raise AggregateError("bootstrap_effects.csv contains a malformed row")
                    effect_type, name, metric = key
                    expected_identity = {
                        "domain": domain,
                        "draw": draw,
                        "proposal": proposal,
                        "effect_type": effect_type,
                        "name": name,
                        "metric": metric,
                    }
                    for field, expected in expected_identity.items():
                        observed: object
                        if field in ("draw", "proposal"):
                            observed = parse_int(row[field], field)
                        else:
                            observed = row[field]
                        if observed != expected:
                            raise AggregateError(
                                f"bootstrap row identity changed at {domain} draw {draw}"
                            )
                    observed_value = parse_float(row["value"], "bootstrap value")
                    close(
                        observed_value,
                        expected_value,
                        f"bootstrap {domain} draw {draw} {key}",
                    )
                    distributions[domain][key].append(observed_value)
            proposals_by_domain[domain] = last_proposal
        try:
            extra = next(reader)
        except StopIteration:
            extra = None
        if extra is not None:
            raise AggregateError("bootstrap_effects.csv has rows beyond the frozen scope")

    aggregate_rows = read_csv_exact(
        path / "aggregate_effects.csv", AGGREGATE_EFFECT_SCHEMA
    )
    ordered_effects = effect_keys()
    expected_row_count = 2 * len(ordered_effects)
    if len(aggregate_rows) != expected_row_count:
        raise AggregateError("aggregate_effects.csv does not contain exactly 76 rows")
    row_position = 0
    for domain in ("cbis", "derm"):
        context = contexts[domain]
        for key in ordered_effects:
            row = aggregate_rows[row_position]
            row_position += 1
            effect_type, name, metric = key
            if (
                row["domain"], row["effect_type"], row["name"], row["metric"]
            ) != (domain, effect_type, name, metric):
                raise AggregateError("aggregate-effects row order or identity changed")
            observed_estimate = parse_float(row["estimate"], "aggregate estimate")
            close(
                observed_estimate,
                points[domain][key],
                f"aggregate point {domain} {key}",
            )
            values = np.asarray(distributions[domain][key], dtype=float)
            if len(values) != draws or np.any(~np.isfinite(values)):
                raise AggregateError("aggregate bootstrap distribution is incomplete")
            expected_lower, expected_upper = np.quantile(
                values, (0.025, 0.975), method="linear"
            )
            observed_lower = parse_float(row["ci_lower"], "CI lower")
            observed_upper = parse_float(row["ci_upper"], "CI upper")
            close(observed_lower, float(expected_lower), f"CI lower {domain} {key}")
            close(observed_upper, float(expected_upper), f"CI upper {domain} {key}")
            if observed_lower > observed_upper:
                raise AggregateError("aggregate confidence interval is reversed")
            expected_counts = {
                "n_bootstrap_valid": draws,
                "n_clusters": len(context["cluster_names"]),
                "n_cases": len(context["case_ids"]),
                "n_repetitions": len(context["repetitions"]),
            }
            for field, expected in expected_counts.items():
                if parse_int(row[field], field) != expected:
                    raise AggregateError(f"aggregate count changed: {field}")
            if row["direction"] != effect_direction(key):
                raise AggregateError("aggregate effect direction changed")

    audit = load_json(path / "aggregate_audit.json", "aggregate audit")
    if not isinstance(audit, dict):
        raise AggregateError("aggregate audit is not an object")
    require_exact_keys(audit, AGGREGATE_AUDIT_KEYS, "aggregate audit")
    expected_audit_values = {
        "status": "PASS",
        "scope": "P1_COMPLETE_AGGREGATE",
        "protocol_sha256": PROTOCOL_SHA256,
        "aggregate_spec_sha256": gate_anchors["aggregate_spec_sha256"],
        "aggregate_script_sha256": gate_anchors["aggregate_script_sha256"],
        "source_audit_sha256": gate_anchors["source_audit_sha256"],
        "authorization_sha256": gate_anchors["authorization_sha256"],
        "units_sha256": UNITS_SHA256,
        "unit_manifest_anchors": list(unit_manifest_hashes),
        "formal_unit_common_anchors": dict(common_unit_anchors),
        "all_aggregate_prepublication_audits_pass": True,
        "hypothesis_judgment": "NOT_PERMITTED_IN_AGGREGATION_LAYER",
        "synthetic_test_mode": synthetic_test,
        "official_test_access": False,
        "official_test_discovery_or_listing": False,
    }
    for key, expected in expected_audit_values.items():
        if audit.get(key) != expected:
            raise AggregateError(f"aggregate audit does not pin {key}")
    checks = audit.get("checks")
    meanings = audit.get("check_meanings")
    if (
        not isinstance(checks, dict)
        or tuple(checks) != AGGREGATE_AUDIT_IDS
        or any(value is not True for value in checks.values())
    ):
        raise AggregateError("aggregate audit does not contain ordered G01--G18 all true")
    if (
        not isinstance(meanings, dict)
        or tuple(meanings) != AGGREGATE_AUDIT_IDS
        or any(not isinstance(value, str) or not value for value in meanings.values())
    ):
        raise AggregateError("aggregate audit meanings do not cover ordered G01--G18")
    bootstrap_audit = audit.get("bootstrap")
    expected_bootstrap_keys = (
        "seed", "rng", "requested_valid_draws_per_domain",
        "maximum_proposals_per_domain", "valid_draws", "proposals", "interval",
    )
    if not isinstance(bootstrap_audit, dict):
        raise AggregateError("aggregate audit bootstrap block is not an object")
    require_exact_keys(bootstrap_audit, expected_bootstrap_keys, "bootstrap audit")
    expected_bootstrap_values = {
        "seed": BOOTSTRAP_SEED,
        "rng": "numpy.random.Generator(PCG64), one stream in fixed cbis-then-derm order",
        "requested_valid_draws_per_domain": draws,
        "maximum_proposals_per_domain": max_proposals,
        "valid_draws": {domain: draws for domain in ("cbis", "derm")},
        "proposals": proposals_by_domain,
        "interval": "two-sided 95% percentile; numpy quantile method=linear",
    }
    if bootstrap_audit != expected_bootstrap_values:
        raise AggregateError("aggregate bootstrap audit differs from replayed execution")

    manifest = load_json(path / "execution_manifest.json", "aggregate manifest")
    if not isinstance(manifest, dict):
        raise AggregateError("aggregate manifest is not an object")
    require_exact_keys(manifest, AGGREGATE_MANIFEST_KEYS, "aggregate manifest")
    expected_script_path = (
        str(SELF_PATH.relative_to(ROOT))
        if not synthetic_test
        else "synthetic-test/imported-source"
    )
    expected_manifest_values = {
        "scope": "P1_COMPLETE_AGGREGATE",
        "protocol_path": PROTOCOL_REL,
        "protocol_sha256": PROTOCOL_SHA256,
        "aggregate_script_path": expected_script_path,
        "aggregate_script_sha256": gate_anchors["aggregate_script_sha256"],
        "aggregate_spec_path": SPEC_REL,
        "aggregate_spec_sha256": gate_anchors["aggregate_spec_sha256"],
        "source_audit_path": SOURCE_AUDIT_REL,
        "source_audit_sha256": gate_anchors["source_audit_sha256"],
        "authorization_path": AUTHORIZATION_REL,
        "authorization_sha256": gate_anchors["authorization_sha256"],
        "units_sha256": UNITS_SHA256,
        "unit_execution_manifests": list(unit_manifest_hashes),
        "artifacts": artifact_records(path),
        "artifact_count_excluding_manifest_and_COMPLETE": len(AGGREGATE_PAYLOAD_FILES),
        "complete_excluded_because_written_last": True,
        "aggregate_prepublication_audit_ids": list(AGGREGATE_AUDIT_IDS),
        "aggregate_publication_audit_id": AGGREGATE_PUBLICATION_AUDIT_ID,
        "scientific_interpretation": (
            "not performed here; aggregate hypothesis judgment is a separate P1 decision stage"
        ),
    }
    for key, expected in expected_manifest_values.items():
        if manifest.get(key) != expected:
            raise AggregateError(f"aggregate manifest does not pin {key}")

    complete = load_json(path / "COMPLETE.json", "aggregate COMPLETE")
    if not isinstance(complete, dict):
        raise AggregateError("aggregate COMPLETE is not an object")
    require_exact_keys(complete, AGGREGATE_COMPLETE_KEYS, "aggregate COMPLETE")
    expected_complete_values = {
        "status": "COMPLETE",
        "scope": "P1_COMPLETE_AGGREGATE",
        "protocol_sha256": PROTOCOL_SHA256,
        "execution_manifest_sha256": sha256_file(path / "execution_manifest.json"),
        "aggregate_prepublication_audit_ids": list(AGGREGATE_AUDIT_IDS),
        "aggregate_publication_audit_id": AGGREGATE_PUBLICATION_AUDIT_ID,
        "all_aggregate_publication_audits_pass": True,
        "hypothesis_judgment": "NOT_PERMITTED_IN_AGGREGATION_LAYER",
    }
    for key, expected in expected_complete_values.items():
        if complete.get(key) != expected:
            raise AggregateError(f"aggregate COMPLETE does not pin {key}")
    if not isinstance(complete.get("completed_at_utc"), str) or not complete["completed_at_utc"]:
        raise AggregateError("aggregate COMPLETE lacks a completion timestamp")


def validate_published_hash_chain(path: Path) -> None:
    """Recheck immutable identities after the atomic directory rename."""
    if exact_names(path) != tuple(sorted(AGGREGATE_FILES)):
        raise AggregateError("published aggregate file set changed after rename")
    for name in AGGREGATE_FILES:
        require_regular_file(path / name, f"published aggregate/{name}")
    manifest = load_json(path / "execution_manifest.json", "published manifest")
    complete = load_json(path / "COMPLETE.json", "published COMPLETE")
    if not isinstance(manifest, dict) or not isinstance(complete, dict):
        raise AggregateError("published hash-chain metadata is malformed")
    require_exact_keys(manifest, AGGREGATE_MANIFEST_KEYS, "published manifest")
    require_exact_keys(complete, AGGREGATE_COMPLETE_KEYS, "published COMPLETE")
    if manifest.get("artifacts") != artifact_records(path):
        raise AggregateError("published aggregate payload hashes changed after rename")
    if complete.get("execution_manifest_sha256") != sha256_file(
        path / "execution_manifest.json"
    ):
        raise AggregateError("published COMPLETE no longer anchors its manifest")


def build_aggregate(
    root: Path,
    *,
    draws: int,
    max_proposals: int,
    gate_anchors: Mapping[str, str],
    synthetic_test: bool,
) -> Path:
    units_parent = root / UNITS_PARENT_REL
    final = root / FINAL_REL
    lock = root / LOCK_REL
    require_plain_directory(units_parent, "fixed formal-unit parent")
    require_plain_directory(final.parent, "aggregate publication parent")
    if lstat_or_none(final) is not None:
        raise AggregateError("aggregate final target already exists")
    lock_fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    stage = final.parent / f".aggregate.staging-{uuid.uuid4().hex}"
    try:
        os.write(lock_fd, stable_json_bytes({"pid": os.getpid(), "created_at_utc": utc_now()}))
        os.fsync(lock_fd)
        os.close(lock_fd)
        lock_fd = -1
        os.mkdir(stage, 0o700)
        merged: dict[str, list[dict[str, str]]] = {name: [] for name in CSV_SCHEMAS}
        common_unit_anchors: dict[str, str] | None = None
        unit_manifest_hashes = []
        for unit in UNITS:
            unit_path = units_parent / unit_name(unit)
            tables, anchors = validate_unit(unit_path, unit)
            if common_unit_anchors is None:
                common_unit_anchors = anchors
            elif anchors != common_unit_anchors:
                raise AggregateError("formal-unit source/gate anchors are not identical")
            for name in CSV_SCHEMAS:
                merged[name].extend(tables[name])
            unit_manifest_hashes.append({
                **unit_object(unit),
                "execution_manifest_sha256": sha256_file(unit_path / "execution_manifest.json"),
                "COMPLETE_sha256": sha256_file(unit_path / "COMPLETE.json"),
            })
        assert common_unit_anchors is not None

        for name, schema in CSV_SCHEMAS.items():
            write_csv(stage / MERGED_NAMES[name], schema, merged[name])

        prediction_rows = merged["predictions.csv"]
        by_domain = {
            domain: [row for row in prediction_rows if row["domain"] == domain]
            for domain in ("cbis", "derm")
        }
        points: dict[str, dict[tuple[str, str, str], float]] = {}
        contexts: dict[str, dict[str, Any]] = {}
        for domain in ("cbis", "derm"):
            points[domain], contexts[domain] = point_estimators(by_domain[domain], domain)

        bootstrap_path = stage / "bootstrap_effects.csv"
        bootstrap_fd = os.open(bootstrap_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        distributions_by_domain: dict[str, dict[tuple[str, str, str], list[float]]] = {}
        proposals_by_domain: dict[str, int] = {}
        try:
            with os.fdopen(bootstrap_fd, "w", encoding="utf-8", newline="", closefd=False) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(BOOTSTRAP_EFFECT_SCHEMA), lineterminator="\n")
                writer.writeheader()
                rng = np.random.default_rng(BOOTSTRAP_SEED)
                for domain in ("cbis", "derm"):
                    distributions, proposals = bootstrap_domain(
                        domain, contexts[domain], rng, draws, max_proposals, writer
                    )
                    distributions_by_domain[domain] = distributions
                    proposals_by_domain[domain] = proposals
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            os.close(bootstrap_fd)

        effect_rows = []
        for domain in ("cbis", "derm"):
            context = contexts[domain]
            for effect_type, names, metrics in (
                ("arm", ARMS, ARM_METRICS),
                ("contrast", CONTRASTS, ("class_equal_NLL", "AUC")),
            ):
                for name in names:
                    for metric in metrics:
                        key = (effect_type, name, metric)
                        values = np.asarray(distributions_by_domain[domain][key], dtype=float)
                        if len(values) != draws or np.any(~np.isfinite(values)):
                            raise AggregateError("bootstrap distribution is incomplete or non-finite")
                        lower, upper = np.quantile(values, (0.025, 0.975), method="linear")
                        effect_rows.append({
                            "domain": domain,
                            "effect_type": effect_type,
                            "name": name,
                            "metric": metric,
                            "estimate": float_text(points[domain][key]),
                            "ci_lower": float_text(float(lower)),
                            "ci_upper": float_text(float(upper)),
                            "n_bootstrap_valid": draws,
                            "n_clusters": len(context["cluster_names"]),
                            "n_cases": len(context["case_ids"]),
                            "n_repetitions": len(context["repetitions"]),
                            "direction": effect_direction(key),
                        })
        write_csv(stage / "aggregate_effects.csv", AGGREGATE_EFFECT_SCHEMA, effect_rows)

        audit = {
            "status": "PASS",
            "scope": "P1_COMPLETE_AGGREGATE",
            "protocol_sha256": PROTOCOL_SHA256,
            "aggregate_spec_sha256": gate_anchors["aggregate_spec_sha256"],
            "aggregate_script_sha256": gate_anchors["aggregate_script_sha256"],
            "source_audit_sha256": gate_anchors["source_audit_sha256"],
            "authorization_sha256": gate_anchors["authorization_sha256"],
            "units_sha256": UNITS_SHA256,
            "unit_manifest_anchors": unit_manifest_hashes,
            "formal_unit_common_anchors": common_unit_anchors,
            "checks": {audit_id: True for audit_id in AGGREGATE_AUDIT_IDS},
            "check_meanings": {
                "G01": "live protocol and aggregate gate anchors matched the frozen hashes",
                "G02": "the literal 30-unit scope was used with no selector or discovery",
                "G03": "each unit contained exactly 14 non-symlink regular files",
                "G04": "each unit manifest hash-covered exactly 12 payload artifacts",
                "G05": "each COMPLETE marker anchored its manifest and closed F20",
                "G06": "every unit carried ordered F01--F19 all true",
                "G07": "no unit carried a scientific hypothesis judgment",
                "G08": "all formal paired rows were marked non-pilot with frozen directions",
                "G09": "all five CSV inputs had exact schemas and unit identities",
                "G10": "all seven arms had aligned finite case-level probabilities",
                "G11": "per-case losses/contributions recomputed within tolerance",
                "G12": "unit metrics and paired effects recomputed within tolerance",
                "G13": "outer folds partitioned each repetition exactly once by case",
                "G14": "case labels/groups and case sets were stable across repetitions",
                "G15": "NLL/Brier used case-first repetition averaging and AUC used repetition-first averaging",
                "G16": "paired whole-cluster draws shared one multiset across all arms/contrasts",
                "G17": "missing-class proposals were rejected from one fixed RNG stream within the proposal cap",
                "G18": "no official-test path was accessed, discovered, or represented",
            },
            "all_aggregate_prepublication_audits_pass": True,
            "hypothesis_judgment": "NOT_PERMITTED_IN_AGGREGATION_LAYER",
            "bootstrap": {
                "seed": BOOTSTRAP_SEED,
                "rng": "numpy.random.Generator(PCG64), one stream in fixed cbis-then-derm order",
                "requested_valid_draws_per_domain": draws,
                "maximum_proposals_per_domain": max_proposals,
                "valid_draws": {domain: draws for domain in ("cbis", "derm")},
                "proposals": proposals_by_domain,
                "interval": "two-sided 95% percentile; numpy quantile method=linear",
            },
            "synthetic_test_mode": synthetic_test,
            "official_test_access": False,
            "official_test_discovery_or_listing": False,
        }
        write_new_json(stage / "aggregate_audit.json", audit)

        if exact_names(stage) != tuple(sorted(AGGREGATE_PAYLOAD_FILES)):
            raise AggregateError("aggregate stage is not the exact eight-payload set")
        records = artifact_records(stage)
        manifest = {
            "scope": "P1_COMPLETE_AGGREGATE",
            "protocol_path": PROTOCOL_REL,
            "protocol_sha256": PROTOCOL_SHA256,
            "aggregate_script_path": str(SELF_PATH.relative_to(ROOT)) if root == ROOT else "synthetic-test/imported-source",
            "aggregate_script_sha256": gate_anchors["aggregate_script_sha256"],
            "aggregate_spec_path": SPEC_REL,
            "aggregate_spec_sha256": gate_anchors["aggregate_spec_sha256"],
            "source_audit_path": SOURCE_AUDIT_REL,
            "source_audit_sha256": gate_anchors["source_audit_sha256"],
            "authorization_path": AUTHORIZATION_REL,
            "authorization_sha256": gate_anchors["authorization_sha256"],
            "units_sha256": UNITS_SHA256,
            "unit_execution_manifests": unit_manifest_hashes,
            "artifacts": records,
            "artifact_count_excluding_manifest_and_COMPLETE": len(records),
            "complete_excluded_because_written_last": True,
            "aggregate_prepublication_audit_ids": list(AGGREGATE_AUDIT_IDS),
            "aggregate_publication_audit_id": AGGREGATE_PUBLICATION_AUDIT_ID,
            "scientific_interpretation": "not performed here; aggregate hypothesis judgment is a separate P1 decision stage",
        }
        write_new_json(stage / "execution_manifest.json", manifest)
        if exact_names(stage) != tuple(sorted(AGGREGATE_FILES[:-1])):
            raise AggregateError("aggregate stage changed before COMPLETE")
        manifest_sha = sha256_file(stage / "execution_manifest.json")
        complete = {
            "status": "COMPLETE",
            "scope": "P1_COMPLETE_AGGREGATE",
            "protocol_sha256": PROTOCOL_SHA256,
            "execution_manifest_sha256": manifest_sha,
            "aggregate_prepublication_audit_ids": list(AGGREGATE_AUDIT_IDS),
            "aggregate_publication_audit_id": AGGREGATE_PUBLICATION_AUDIT_ID,
            "all_aggregate_publication_audits_pass": True,
            "completed_at_utc": utc_now(),
            "hypothesis_judgment": "NOT_PERMITTED_IN_AGGREGATION_LAYER",
        }
        write_new_json(stage / "COMPLETE.json", complete)
        if exact_names(stage) != tuple(sorted(AGGREGATE_FILES)):
            raise AggregateError("aggregate stage does not contain exactly ten required files")
        validate_aggregate_stage(
            stage,
            draws=draws,
            max_proposals=max_proposals,
            gate_anchors=gate_anchors,
            synthetic_test=synthetic_test,
            common_unit_anchors=common_unit_anchors,
            unit_manifest_hashes=unit_manifest_hashes,
            merged_expected=merged,
        )
        fsync_directory(stage)
        if lstat_or_none(final) is not None:
            raise AggregateError("aggregate final target appeared before atomic publication")
        os.rename(stage, final)
        require_plain_directory(final, "published aggregate")
        validate_published_hash_chain(final)
        fsync_directory(final.parent)
        return final
    finally:
        if lock_fd >= 0:
            os.close(lock_fd)
        if lstat_or_none(lock) is not None:
            metadata = os.lstat(lock)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise AggregateError("aggregate lock was replaced before release")
            os.unlink(lock)
            fsync_directory(lock.parent)


def main() -> int:
    if sys.argv[1:]:
        raise AggregateError("production aggregation accepts no command-line selectors")
    require_regular_file(SELF_PATH, "aggregate source")
    gate = validate_gate(ROOT)
    final = build_aggregate(
        ROOT,
        draws=BOOTSTRAP_DRAWS,
        max_proposals=BOOTSTRAP_MAX_PROPOSALS,
        gate_anchors=gate,
        synthetic_test=False,
    )
    print(json.dumps({"status": "COMPLETE", "path": str(final)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AggregateError as exc:
        print(f"P1 aggregate aborted: {exc}", file=sys.stderr)
        raise SystemExit(2)
