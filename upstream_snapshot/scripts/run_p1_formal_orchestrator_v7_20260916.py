#!/usr/bin/env python3
"""Unauthorised V7 candidate orchestration for the frozen Paper-B P1 scope.

This source-only preparation layer cannot choose a domain, repetition, fold,
input path, or output path. Formal execution remains disabled until a separate
unit engine, specification, independent audit, and pre-outcome authorization
receipt pass their fixed hashes.  In contrast with V2, a failed engine can
return only an exact, non-data-bearing stage/error-code receipt; raw child
stderr is never propagated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


# Keep the source and repository roots lexical.  Resolving either path here would
# erase exactly the symlink ancestry that the V2 publication boundary must reject.
SELF_PATH = Path(os.path.abspath(__file__))
ROOT = SELF_PATH.parent.parent
PROTOCOL_REL = "paper_b_readiness/P1_PROTOCOL.json"
P0_REL = "paper_b_readiness/P0_HASH_MANIFEST.json"
SPEC_REL = "paper_b_readiness/P1_FORMAL_ORCHESTRATION_SPEC_V7_20260916.json"
ENGINE_REL = "scripts/run_p1_formal_unit_v5_20260916.py"
AUTHORIZATION_REL = (
    "paper_b_readiness/P1_FORMAL_AUTHORIZATION_V7_20260916.json"
)
ENGINE_AUDIT_REL = (
    "paper_b_readiness/P1_FORMAL_ENGINE_AUDIT_V7_20260916.json"
)
FINAL_PARENT_REL = "paper_b_readiness/p1_complete/units"
LOCK_PARENT_REL = "paper_b_readiness/p1_complete"
LOCK_REL = "paper_b_readiness/p1_complete/.formal-run.lock"

PROTOCOL_SHA256 = "bdaca53493521a400f6307088d887eafa035f7eed4a094d522d05ee2ea5a1c1b"
P0_SHA256 = "d00b9d304fd87d98319a1b85315c38accbb83f758abe97b62c02a2fdb91cfdcd"
PILOT_REL = "scripts/run_p1_calibration_prior_20260915.py"
PILOT_SHA256 = "519d1e2631c46395139e401e4dcb3ec7dcecdf27f44fae191b186463030edbac"
PILOT_EXTERNAL_AUDIT_REL = (
    "paper_b_readiness/P1_EXTERNAL_AUDIT_V4_RECEIPT_20260916.json"
)
PILOT_EXTERNAL_AUDIT_SHA256 = (
    "587d6d098338acd47e079265d612efc73e054f292345d8685702adb3f5eb3549"
)

ENGINE_AUDIT_KEYS = (
    "status", "scope", "protocol_sha256", "p0_manifest_sha256",
    "orchestrator_sha256", "engine_sha256", "spec_sha256", "units_sha256",
    "pilot_external_audit_path", "pilot_external_audit_sha256",
    "source_review_complete", "synthetic_contract_tests_pass",
    "cpu_serial_enforced", "whitelist_contract_pass",
    "atomic_resume_contract_pass", "pilot_gate_verified",
    "formal_audit_mapping_reviewed", "p1_outcomes_accessed",
    "official_test_accessed", "reviewer", "completed_at_utc",
)

AUTHORIZATION_KEYS = (
    "status", "authorized", "protocol_sha256", "p0_manifest_sha256",
    "orchestrator_sha256", "engine_sha256", "spec_sha256", "units_sha256",
    "engine_audit_path", "engine_audit_sha256", "pilot_external_audit_path",
    "pilot_external_audit_sha256", "pilot_gate_status", "pilot_unit",
    "pilot_protocol_a03_a19_attested", "formal_overlay_approved",
    "issued_at_utc",
)

ARMS = ("R", "S", "C", "M", "SP", "CP", "MP")
CONTRASTS = (
    "C_minus_M",
    "S_minus_SP",
    "S_minus_C",
    "CP_minus_MP",
    "M_minus_MP",
)
CONTRAST_SIGNS = {
    "C_minus_M": "negative_favours_C",
    "S_minus_SP": "positive_favours_SP",
    "S_minus_C": "positive_favours_C",
    "CP_minus_MP": "negative_favours_CP",
    "M_minus_MP": "positive_favours_MP",
}

# F01--F19 are the separately reviewed formal-mode computation/lineage audits.
# They deliberately do not reuse the frozen pilot protocol's A01--A19 labels:
# A03 and A19 are pilot-gate facts, whereas a 30-unit formal run is a later,
# explicitly authorized mode. F20 is the publication audit and can only become
# true after this orchestrator has added the manifest and COMPLETE marker and
# atomically renamed the unit directory. Keeping F20 out of the unit-engine
# receipt prevents a pre-publication process from claiming a future event.
FORMAL_PREPUBLICATION_AUDIT_IDS = (
    "F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09",
    "F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17", "F18",
    "F19",
)
FORMAL_PUBLICATION_AUDIT_ID = "F20"

# This tuple is the complete, ordered formal scope. It is intentionally literal:
# no directory discovery, wildcard, range selector, or user-provided unit exists.
UNITS = (
    ("cbis", 0, 0),
    ("cbis", 0, 1),
    ("cbis", 0, 2),
    ("cbis", 0, 3),
    ("cbis", 0, 4),
    ("derm", 0, 0),
    ("derm", 0, 1),
    ("derm", 0, 2),
    ("derm", 0, 3),
    ("derm", 0, 4),
    ("derm", 1, 0),
    ("derm", 1, 1),
    ("derm", 1, 2),
    ("derm", 1, 3),
    ("derm", 1, 4),
    ("derm", 2, 0),
    ("derm", 2, 1),
    ("derm", 2, 2),
    ("derm", 2, 3),
    ("derm", 2, 4),
    ("derm", 3, 0),
    ("derm", 3, 1),
    ("derm", 3, 2),
    ("derm", 3, 3),
    ("derm", 3, 4),
    ("derm", 4, 0),
    ("derm", 4, 1),
    ("derm", 4, 2),
    ("derm", 4, 3),
    ("derm", 4, 4),
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
JSON_PAYLOADS = (
    "selections.json",
    "calibrators.json",
    "prior_lineage.json",
    "lineage.json",
    "audit.json",
    "fit_cost.json",
)
PAYLOAD_FILES = tuple(CSV_SCHEMAS) + JSON_PAYLOADS + ("run.log",)
PUBLISHED_FILES = (
    tuple(CSV_SCHEMAS)
    + JSON_PAYLOADS
    + ("execution_manifest.json", "run.log", "COMPLETE.json")
)
ARCHIVE_FILES = (
    "panel.npz",
    "generation.json",
    "predictions.csv",
    "selection.json",
    "heads.json",
    "COMPLETE.json",
)

DOMAIN_CONTROLS = {
    "cbis": (
        "results/deadline_nested_20260912/protocol.json",
        "scripts/run_deadline_nested_20260912.py",
        "src/paper_b_cbis_p3.py",
        "configs/paper_b_cbis_p3_minimal_protocol_final_v1_20260910.json",
        "data/CBIS_DDSM/paper_b_cbis_mass_train_breast_cohort_v1_20260910.npz",
        "data_manifests/paper_b_cbis_mass_train_breast_cohort_v1_20260910.csv",
    ),
    "derm": (
        "results/deadline_nested_20260912/protocol.json",
        "scripts/run_deadline_nested_20260912.py",
        "src/paper2_mpjc/pilot.py",
        "configs/cf_mpjc_d1_derm7pt.json",
        "data/Derm7pt/derm7pt_biomedclip_dev609_only_v1_20260908.npz",
        "data/Derm7pt/derm7pt_dev609_only_v1_20260908.csv",
    ),
}

EXPECTED_COST = {
    "cbis": {
        "head_fits": 151,
        "affine_fits": 2,
        "prior_restoration_calls": 17,
        "prior_concept_records": 170,
    },
    "derm": {
        "head_fits": 175,
        "affine_fits": 2,
        "prior_restoration_calls": 21,
        "prior_concept_records": 147,
    },
}

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
}

ENGINE_FAILURE_KEYS = frozenset(
    {
        "status",
        "stage",
        "error_code",
        "official_test_accessed",
        "hypothesis_judgment_performed",
    }
)
ENGINE_FAILURE_STAGES = frozenset(
    {
        "BOOT",
        "ARGUMENT_CONTRACT",
        "THREAD_ENV_CONTRACT",
        "TICKET_READ_PARSE",
        "TICKET_VALIDATE",
        "WHITELIST_PRE",
        "COMPUTE_ARCHIVE_LOAD",
        "COMPUTE_STRUCTURE_CONTRACT",
        "COMPUTE_GENERATION_LINEAGE",
        "COMPUTE_NESTED_FITS",
        "COMPUTE_OUTER_FITS",
        "COMPUTE_ARCHIVE_PREDICTION_REPRODUCTION",
        "COMPUTE_ARCHIVE_SELECTION_REPRODUCTION",
        "COMPUTE_COST_CONTRACT",
        "WHITELIST_POST",
        "PAYLOAD_BUILD",
        "PAYLOAD_WRITE",
        "SUCCESS_RECEIPT",
    }
)
ENGINE_FAILURE_ERROR_CODES = frozenset(
    {
        "E_FORMAL_CONTRACT",
        "E_ARCHIVE_SELECTION_CONTRACT",
        "E_UNEXPECTED_EXCEPTION",
    }
)


class FormalError(RuntimeError):
    """A fail-closed contract or publication error."""


def parse_sanitized_engine_failure(stderr: str) -> tuple[str, str] | None:
    """Accept only the exact V2 engine failure receipt; never return raw text."""
    if len(stderr.encode("utf-8")) > 1024:
        return None
    lines = stderr.splitlines()
    if len(lines) != 1:
        return None
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate failure-receipt key")
            value[key] = item
        return value

    try:
        value = json.loads(lines[0], object_pairs_hook=reject_duplicate_keys)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    if not isinstance(value, dict) or frozenset(value) != ENGINE_FAILURE_KEYS:
        return None
    if (
        value.get("status") != "PAYLOAD_FAILED_CLOSED"
        or value.get("stage") not in ENGINE_FAILURE_STAGES
        or value.get("error_code") not in ENGINE_FAILURE_ERROR_CODES
        or value.get("official_test_accessed") is not False
        or value.get("hypothesis_judgment_performed") is not False
    ):
        return None
    return str(value["stage"]), str(value["error_code"])


def lstat_or_none(path: Path) -> os.stat_result | None:
    """Return lexical metadata without following the final path component."""
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise FormalError(f"cannot lstat protected path: {path}") from exc


def require_plain_directory(path: Path, label: str) -> None:
    metadata = lstat_or_none(path)
    if metadata is None:
        raise FormalError(f"{label} is absent: {path}")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise FormalError(f"{label} is not a non-symlink directory: {path}")


def ensure_trusted_directory_chain(
    target: Path,
    *,
    root: Path,
    create_missing: bool,
) -> None:
    """Validate ROOT -> target lexically, creating at most one level at a time.

    ``Path.resolve`` and recursive ``mkdir`` are deliberately forbidden here:
    either would hide or traverse a substituted symlink.  This is a fail-closed
    user-space ancestry check, repeated immediately before every protected
    lock, staging, resume, rename, and release operation.
    """
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise FormalError(f"protected path escapes lexical root: {target}") from exc
    require_plain_directory(root, "formal lexical root")
    current = root
    for component in relative.parts:
        if component in ("", ".", ".."):
            raise FormalError(f"unsafe component in protected path: {target}")
        current = current / component
        metadata = lstat_or_none(current)
        if metadata is None:
            if not create_missing:
                raise FormalError(f"protected directory is absent: {current}")
            try:
                os.mkdir(current, 0o700)
            except OSError as exc:
                raise FormalError(
                    f"cannot create protected directory one level at a time: {current}"
                ) from exc
            metadata = lstat_or_none(current)
            if metadata is None:
                raise FormalError(f"new protected directory disappeared: {current}")
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise FormalError(
                f"protected ancestry contains a symlink or non-directory: {current}"
            )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def write_new_bytes(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)


def write_new_json(path: Path, value: object) -> None:
    write_new_bytes(path, stable_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(stable_json_bytes(value)).hexdigest()


def unit_name(unit: tuple[str, int, int]) -> str:
    return f"{unit[0]}_r{unit[1]}_f{unit[2]}"


def unit_object(unit: tuple[str, int, int]) -> dict[str, object]:
    return {
        "domain": unit[0],
        "repetition": unit[1],
        "outer_fold": unit[2],
    }


def units_digest() -> str:
    return sha256_json([unit_object(unit) for unit in UNITS])


def assert_regular_file(path: Path, label: str) -> None:
    metadata = lstat_or_none(path)
    if (
        metadata is None
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise FormalError(f"{label} is absent, non-regular, or a symlink: {path}")


def load_json(path: Path, label: str) -> Any:
    assert_regular_file(path, label)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FormalError(f"{label} is not valid UTF-8 JSON: {path}") from exc


def require_exact_keys(value: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    expected = set(keys)
    observed = set(value)
    if observed != expected:
        raise FormalError(
            f"{label} keys differ; missing={sorted(expected-observed)}, "
            f"unexpected={sorted(observed-expected)}"
        )


def verify_hash(path: Path, expected: str, label: str) -> None:
    assert_regular_file(path, label)
    observed = sha256_file(path)
    if observed != expected:
        raise FormalError(f"{label} SHA-256 mismatch")


def verify_live_base_anchors() -> tuple[dict[str, Any], dict[str, Any]]:
    verify_hash(ROOT / PROTOCOL_REL, PROTOCOL_SHA256, "frozen P1 protocol")
    verify_hash(ROOT / P0_REL, P0_SHA256, "P0 manifest")
    verify_hash(ROOT / PILOT_REL, PILOT_SHA256, "unchanged pilot runner")
    protocol = load_json(ROOT / PROTOCOL_REL, "frozen P1 protocol")
    p0 = load_json(ROOT / P0_REL, "P0 manifest")
    if protocol.get("status") != "FROZEN_PRE_OUTCOME":
        raise FormalError("frozen protocol status changed")
    planned = protocol.get("scope", {}).get("planned_complete_units")
    expected_planned = {
        "cbis": {"repetitions": 1, "outer_folds": 5, "grouping": "patient"},
        "derm": {"repetitions": 5, "outer_folds": 5, "grouping": "case"},
    }
    if planned != expected_planned:
        raise FormalError("frozen protocol no longer specifies the exact 30-unit scope")
    schemas = protocol.get("output_contract", {}).get("schemas", {})
    for name, expected in CSV_SCHEMAS.items():
        if schemas.get(name) != list(expected):
            raise FormalError(f"frozen schema changed for {name}")
    required = protocol.get("output_contract", {}).get(
        "complete_run", {}
    ).get("unit_required")
    if required != list(PUBLISHED_FILES):
        raise FormalError("frozen 14-file complete-unit contract changed")
    return protocol, p0


def p0_index(p0: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    critical = p0.get("critical_assets")
    frozen = p0.get("frozen_result_files")
    if not isinstance(critical, dict) or not isinstance(frozen, list):
        raise FormalError("P0 manifest has an unexpected structure")
    for path, record in critical.items():
        if not isinstance(record, dict) or "sha256" not in record:
            raise FormalError(f"P0 critical record malformed: {path}")
        index[path] = {"path": path, **record}
    for record in frozen:
        if not isinstance(record, dict) or set(("path", "sha256")) - set(record):
            raise FormalError("P0 frozen-result record malformed")
        old = index.get(str(record["path"]))
        if old is not None and old["sha256"] != record["sha256"]:
            raise FormalError(f"P0 contains conflicting hashes: {record['path']}")
        index[str(record["path"])] = dict(record)
    return index


def protocol_fixed_index(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    fixed = protocol.get("fixed_inputs")
    if not isinstance(fixed, dict):
        raise FormalError("frozen protocol has no fixed_inputs mapping")
    index: dict[str, dict[str, Any]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            path = value.get("path")
            digest = value.get("sha256")
            if path is not None or digest is not None:
                if not isinstance(path, str) or not isinstance(digest, str):
                    raise FormalError("frozen protocol contains a malformed fixed input")
                old = index.get(path)
                if old is not None and old["sha256"] != digest:
                    raise FormalError(
                        f"frozen protocol contains conflicting hashes: {path}"
                    )
                index[path] = {"path": path, **value}
                return
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(fixed)
    return index


def fixed_input_records(
    unit: tuple[str, int, int],
    protocol: Mapping[str, Any],
    p0: Mapping[str, Any],
) -> list[dict[str, Any]]:
    domain = unit[0]
    archive_root = f"results/deadline_nested_20260912/{unit_name(unit)}"
    paths = list(DOMAIN_CONTROLS[domain])
    paths.extend(f"{archive_root}/{name}" for name in ARCHIVE_FILES)
    p0_records = p0_index(p0)
    protocol_records = protocol_fixed_index(protocol)
    records: list[dict[str, Any]] = []
    for rel in paths:
        p0_record = p0_records.get(rel)
        protocol_record = protocol_records.get(rel)
        if (
            p0_record is not None
            and protocol_record is not None
            and p0_record["sha256"] != protocol_record["sha256"]
        ):
            raise FormalError(f"P0/protocol fixed-input hash conflict: {rel}")
        record = p0_record if p0_record is not None else protocol_record
        if record is None:
            raise FormalError(f"fixed input is not hash-pinned by P0 or protocol: {rel}")
        path = ROOT / rel
        assert_regular_file(path, f"fixed input {rel}")
        observed = sha256_file(path)
        if observed != record["sha256"]:
            raise FormalError(f"fixed input differs from its frozen hash: {rel}")
        records.append(
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": observed,
                "hash_source": (
                    "P0_HASH_MANIFEST"
                    if p0_record is not None
                    else "P1_PROTOCOL.fixed_inputs"
                ),
                "role": "complete_run_archive" if rel.startswith(archive_root) else "control",
            }
        )
    if len(records) != 12 or len({item["path"] for item in records}) != 12:
        raise FormalError("unit whitelist is not the exact six controls plus six archive files")
    return records


def verify_pilot_external_audit_at(
    root: Path,
    relative_path: str,
    expected_sha256: str,
) -> None:
    receipt_path = root / relative_path
    assert_regular_file(receipt_path, "independent V4 pilot audit receipt")
    if sha256_file(receipt_path) != expected_sha256:
        raise FormalError("independent V4 pilot audit receipt SHA-256 mismatch")


def validate_engine_audit_document(
    receipt: Mapping[str, Any],
    expected: Mapping[str, str],
) -> None:
    require_exact_keys(receipt, ENGINE_AUDIT_KEYS, "independent engine audit receipt")
    expected_values: dict[str, Any] = {
        "status": "PASS",
        "scope": "P1_FORMAL_UNIT_ENGINE_PRE_OUTCOME",
        **expected,
    }
    for key, value in expected_values.items():
        if receipt.get(key) != value:
            raise FormalError(f"engine audit receipt does not pin current {key}")
    required_true = (
        "source_review_complete",
        "synthetic_contract_tests_pass",
        "cpu_serial_enforced",
        "whitelist_contract_pass",
        "atomic_resume_contract_pass",
        "pilot_gate_verified",
        "formal_audit_mapping_reviewed",
    )
    if not all(receipt.get(key) is True for key in required_true):
        raise FormalError("engine audit receipt has an unpassed required check")
    if receipt.get("p1_outcomes_accessed") is not False:
        raise FormalError("pre-outcome engine audit reports P1 outcome access")
    if receipt.get("official_test_accessed") is not False:
        raise FormalError("pre-outcome engine audit reports official-test access")


def validate_authorization_document(
    authorization: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    require_exact_keys(authorization, AUTHORIZATION_KEYS, "authorization")
    expected_values: dict[str, Any] = {
        "status": "AUTHORIZED_PRE_OUTCOME",
        "authorized": True,
        "pilot_gate_status": "INDEPENDENTLY_AUDITED_PASS",
        "pilot_unit": {"domain": "cbis", "repetition": 0, "outer_fold": 0},
        "pilot_protocol_a03_a19_attested": True,
        "formal_overlay_approved": True,
        **expected,
    }
    for key, value in expected_values.items():
        if authorization.get(key) != value:
            raise FormalError(f"authorization does not pin current {key}")


def verify_authorization() -> dict[str, Any]:
    engine_path = ROOT / ENGINE_REL
    spec_path = ROOT / SPEC_REL
    auth_path = ROOT / AUTHORIZATION_REL
    receipt_path = ROOT / ENGINE_AUDIT_REL

    # V2 accepts neither gate document until the exact externally audited pilot
    # receipt has itself been verified as a non-symlink regular file by hash.
    verify_pilot_external_audit_at(
        ROOT,
        PILOT_EXTERNAL_AUDIT_REL,
        PILOT_EXTERNAL_AUDIT_SHA256,
    )
    for path, label in (
        (engine_path, "formal unit engine"),
        (spec_path, "formal orchestration specification"),
        (auth_path, "pre-outcome formal authorization"),
        (receipt_path, "independent engine audit receipt"),
    ):
        assert_regular_file(path, label)

    common_expected = {
        "protocol_sha256": PROTOCOL_SHA256,
        "p0_manifest_sha256": P0_SHA256,
        "orchestrator_sha256": sha256_file(SELF_PATH),
        "engine_sha256": sha256_file(engine_path),
        "spec_sha256": sha256_file(spec_path),
        "units_sha256": units_digest(),
        "pilot_external_audit_path": PILOT_EXTERNAL_AUDIT_REL,
        "pilot_external_audit_sha256": PILOT_EXTERNAL_AUDIT_SHA256,
    }
    receipt = load_json(receipt_path, "independent engine audit receipt")
    validate_engine_audit_document(receipt, common_expected)

    authorization_expected: dict[str, Any] = {
        **common_expected,
        "engine_audit_path": ENGINE_AUDIT_REL,
        "engine_audit_sha256": sha256_file(receipt_path),
    }
    authorization = load_json(auth_path, "pre-outcome formal authorization")
    validate_authorization_document(authorization, authorization_expected)
    return {
        **authorization,
        "_authorization_sha256": sha256_file(auth_path),
        "_engine_sha256": common_expected["engine_sha256"],
        "_spec_sha256": common_expected["spec_sha256"],
        "_audit_sha256": authorization_expected["engine_audit_sha256"],
    }


def exact_names(path: Path) -> tuple[str, ...]:
    return tuple(sorted(item.name for item in path.iterdir()))


def read_csv_exact(path: Path, expected: Sequence[str]) -> list[dict[str, str]]:
    assert_regular_file(path, path.name)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(expected):
            raise FormalError(f"{path.name} has a non-frozen column schema")
        rows = list(reader)
    if not rows:
        raise FormalError(f"{path.name} is empty")
    if any(set(row) != set(expected) for row in rows):
        raise FormalError(f"{path.name} contains a malformed row")
    return rows


def parse_int(value: str, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise FormalError(f"{label} is not an integer") from exc
    return parsed


def parse_float(value: str, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise FormalError(f"{label} is not numeric") from exc
    if not math.isfinite(parsed):
        raise FormalError(f"{label} is not finite")
    return parsed


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def row_unit_exact(row: Mapping[str, str], unit: tuple[str, int, int]) -> bool:
    return (
        row.get("domain") == unit[0]
        and parse_int(row.get("repetition", ""), "repetition") == unit[1]
        and parse_int(row.get("outer_fold", ""), "outer_fold") == unit[2]
    )


def require_json_unit(value: Mapping[str, Any], unit: tuple[str, int, int], label: str) -> None:
    observed = value.get("unit")
    if observed != unit_object(unit):
        raise FormalError(f"{label} does not carry the exact formal unit identity")


def validate_payload(
    stage: Path,
    unit: tuple[str, int, int],
    *,
    publication_metadata_allowed: bool = False,
) -> dict[str, Any]:
    expected_names = PUBLISHED_FILES if publication_metadata_allowed else PAYLOAD_FILES
    if exact_names(stage) != tuple(sorted(expected_names)):
        scope = "14 published files" if publication_metadata_allowed else "12 payload files"
        raise FormalError(f"directory does not contain exactly the frozen {scope}")
    for name in PAYLOAD_FILES:
        assert_regular_file(stage / name, f"payload {name}")

    tables = {
        name: read_csv_exact(stage / name, schema)
        for name, schema in CSV_SCHEMAS.items()
    }
    for name, rows in tables.items():
        if not all(row_unit_exact(row, unit) for row in rows):
            raise FormalError(f"{name} contains a row from another unit")

    predictions = tables["predictions.csv"]
    prediction_keys = [
        (
            parse_int(row["row_index"], "prediction row_index"),
            row["case_id"], row["group_id"], parse_int(row["y"], "prediction y"),
        )
        for row in predictions
    ]
    if len(set(key[0] for key in prediction_keys)) != len(prediction_keys):
        raise FormalError("predictions.csv has duplicate row_index values")
    for row in predictions:
        if parse_int(row["y"], "prediction y") not in (0, 1):
            raise FormalError("predictions.csv contains a non-binary diagnosis label")
        for arm in ARMS:
            probability = parse_float(row[f"p_{arm}"], f"p_{arm}")
            if not 0.0 <= probability <= 1.0:
                raise FormalError(f"p_{arm} falls outside [0,1]")
        for column in ("logit_S", "logit_C", "logit_SP", "logit_CP"):
            parse_float(row[column], column)

    calibration = tables["calibration_oof.csv"]
    calibration_indices = [parse_int(row["row_index"], "calibration row_index") for row in calibration]
    if len(set(calibration_indices)) != len(calibration_indices):
        raise FormalError("calibration_oof.csv has duplicate row_index values")
    for row in calibration:
        if parse_int(row["y"], "calibration y") not in (0, 1):
            raise FormalError("calibration_oof.csv contains a non-binary label")
        parse_int(row["H"], "H")
        parse_float(row["reference_C_H"], "reference_C_H")
        parse_float(row["raw_base_logit"], "raw_base_logit")
        parse_float(row["prior_restored_base_logit"], "prior_restored_base_logit")
        for column in ("probe_fit_scope_sha256", "reference_fit_scope_sha256"):
            if not is_sha256(row[column]):
                raise FormalError(f"{column} is not a SHA-256 value")

    per_case = tables["per_case_effects.csv"]
    if len(per_case) != len(predictions) * len(ARMS):
        raise FormalError("per_case_effects.csv is not seven rows per evaluation case")
    by_arm: dict[str, list[tuple[int, str, str, int]]] = {}
    for row in per_case:
        arm = row["arm"]
        if arm not in ARMS:
            raise FormalError("per_case_effects.csv contains an unknown arm")
        key = (
            parse_int(row["row_index"], "per-case row_index"),
            row["case_id"], row["group_id"], parse_int(row["y"], "per-case y"),
        )
        by_arm.setdefault(arm, []).append(key)
        for column in (
            "negative_log_likelihood",
            "squared_probability_error",
            "class_equal_nll_contribution",
        ):
            parse_float(row[column], column)
    if set(by_arm) != set(ARMS):
        raise FormalError("per_case_effects.csv does not contain all seven arms")
    if any(by_arm[arm] != prediction_keys for arm in ARMS):
        raise FormalError("arm case rows are not aligned exactly to predictions.csv")

    metrics = tables["metrics.csv"]
    if len(metrics) != len(ARMS) or {row["arm"] for row in metrics} != set(ARMS):
        raise FormalError("metrics.csv is not exactly one row per arm")
    for row in metrics:
        for column in ("class_equal_NLL", "AUC", "natural_frequency_NLL", "Brier"):
            parse_float(row[column], column)
        if parse_int(row["n_cases"], "n_cases") != len(predictions):
            raise FormalError("metrics.csv n_cases differs from predictions.csv")
        n_groups = parse_int(row["n_groups"], "n_groups")
        if not 1 <= n_groups <= len(predictions):
            raise FormalError("metrics.csv has an invalid n_groups")

    paired = tables["paired_effects.csv"]
    if len(paired) != len(CONTRASTS) or {row["contrast"] for row in paired} != set(CONTRASTS):
        raise FormalError("paired_effects.csv is not exactly the frozen five contrasts")
    for row in paired:
        name = row["contrast"]
        if row["favourable_sign_NLL"] != CONTRAST_SIGNS[name]:
            raise FormalError(f"paired-effect direction changed for {name}")
        if row["pilot_descriptive_only"].strip().lower() not in ("false", "0"):
            raise FormalError("formal paired effects are incorrectly labelled as pilot-only")
        parse_float(row["class_equal_NLL_X_minus_Y"], "paired NLL")
        parse_float(row["AUC_X_minus_Y"], "paired AUC")

    json_values: dict[str, Mapping[str, Any]] = {}
    for name in JSON_PAYLOADS:
        value = load_json(stage / name, name)
        if not isinstance(value, dict):
            raise FormalError(f"{name} is not a JSON object")
        require_json_unit(value, unit, name)
        json_values[name] = value

    audit = json_values["audit.json"]
    checks = audit.get("checks")
    if not isinstance(checks, dict) or tuple(checks) != FORMAL_PREPUBLICATION_AUDIT_IDS:
        raise FormalError("audit.json does not contain exactly ordered F01--F19")
    if not all(value is True for value in checks.values()):
        raise FormalError("audit.json has a failed F01--F19 check")
    if audit.get("all_formal_prepublication_audits_pass") is not True:
        raise FormalError("audit.json does not pass the pre-publication audit gate")
    if "all_formal_publication_audits_pass" in audit:
        raise FormalError("unit engine prematurely claimed the F20 publication audit")
    if audit.get("hypothesis_judgment") not in (None, "NOT_PERMITTED_PER_UNIT"):
        raise FormalError("a formal unit made a scientific hypothesis judgment")

    lineage = json_values["lineage.json"]
    boundary = lineage.get("access_boundary")
    if not isinstance(boundary, dict):
        raise FormalError("lineage.json lacks an access boundary")
    if boundary.get("official_test_access") is not False:
        raise FormalError("lineage reports official-test access")
    if boundary.get("official_test_discovery_or_listing") is not False:
        raise FormalError("lineage reports official-test discovery")
    if boundary.get("input_whitelist_exact") is not True:
        raise FormalError("lineage does not attest the exact input whitelist")

    cost = json_values["fit_cost.json"]
    expected_cost = EXPECTED_COST[unit[0]]
    observed_cost = {
        "head_fits": cost.get("head_fits", {}).get("total_head_fits"),
        "affine_fits": cost.get("affine_fits", {}).get("fits"),
        "prior_restoration_calls": cost.get("prior_restoration", {}).get("calls"),
        "prior_concept_records": cost.get("prior_restoration", {}).get("concept_records"),
    }
    if observed_cost != expected_cost:
        raise FormalError(
            f"fit_cost.json differs from the frozen static unit cost: {observed_cost}"
        )
    if cost.get("probe_generation", {}).get("fits") != 0:
        raise FormalError("fit_cost.json reports an upstream concept fit")
    if cost.get("bootstrap_draws") != 0 or cost.get("units_run") != 1:
        raise FormalError("fit_cost.json reports bootstrap or multi-unit work")

    log_path = stage / "run.log"
    if log_path.stat().st_size == 0 or b"\x00" in log_path.read_bytes():
        raise FormalError("run.log is empty or contains NUL bytes")
    return {
        "n_predictions": len(predictions),
        "n_calibration": len(calibration),
        "input_whitelist_sha256": lineage.get("access_boundary", {}).get(
            "input_whitelist_sha256"
        ),
    }


def artifact_records(stage: Path) -> list[dict[str, Any]]:
    records = []
    for name in PAYLOAD_FILES:
        path = stage / name
        records.append(
            {"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return records


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_stage(
    stage: Path,
    final: Path,
    unit: tuple[str, int, int],
    anchors: Mapping[str, str],
    inputs: Sequence[Mapping[str, Any]],
    *,
    trusted_root: Path = ROOT,
) -> None:
    ensure_trusted_directory_chain(
        stage.parent,
        root=trusted_root,
        create_missing=False,
    )
    require_plain_directory(stage, "staging directory")
    if final.parent != stage.parent:
        raise FormalError("staging and final directories do not share one parent")
    if lstat_or_none(final) is not None:
        raise FormalError(f"final target appeared before publication: {final}")
    records = artifact_records(stage)
    manifest = {
        "unit": unit_object(unit),
        "protocol_path": PROTOCOL_REL,
        "protocol_sha256": anchors["protocol_sha256"],
        "p0_manifest_sha256": anchors["p0_manifest_sha256"],
        "orchestrator_path": str(SELF_PATH.relative_to(ROOT)),
        "orchestrator_sha256": anchors["orchestrator_sha256"],
        "engine_path": ENGINE_REL,
        "engine_sha256": anchors["engine_sha256"],
        "spec_path": SPEC_REL,
        "spec_sha256": anchors["spec_sha256"],
        "authorization_path": AUTHORIZATION_REL,
        "authorization_sha256": anchors["authorization_sha256"],
        "engine_audit_path": ENGINE_AUDIT_REL,
        "engine_audit_sha256": anchors["engine_audit_sha256"],
        "units_sha256": units_digest(),
        "fixed_input_whitelist": list(inputs),
        "fixed_input_whitelist_sha256": sha256_json(list(inputs)),
        "artifacts": records,
        "artifact_count_excluding_manifest_and_COMPLETE": len(records),
        "complete_excluded_because_written_last": True,
        "formal_prepublication_audit_ids": list(FORMAL_PREPUBLICATION_AUDIT_IDS),
        "formal_publication_audit_id": FORMAL_PUBLICATION_AUDIT_ID,
        "scientific_interpretation": "formal development unit only; no per-unit hypothesis judgment",
    }
    write_new_json(stage / "execution_manifest.json", manifest)
    if exact_names(stage) != tuple(sorted(PUBLISHED_FILES[:-1])):
        raise FormalError("stage is not exact immediately before COMPLETE")
    manifest_hash = sha256_file(stage / "execution_manifest.json")
    completion = {
        **unit_object(unit),
        "status": "COMPLETE",
        "protocol_sha256": anchors["protocol_sha256"],
        "execution_manifest_sha256": manifest_hash,
        "formal_prepublication_audit_ids": list(FORMAL_PREPUBLICATION_AUDIT_IDS),
        "formal_publication_audit_id": FORMAL_PUBLICATION_AUDIT_ID,
        "all_formal_publication_audits_pass": True,
        "completed_at_utc": utc_now(),
        "interpretation": "formal P1 development unit; aggregate judgment only after all units",
    }
    write_new_json(stage / "COMPLETE.json", completion)
    if exact_names(stage) != tuple(sorted(PUBLISHED_FILES)):
        raise FormalError("completed stage does not have the exact 14-file contract")
    fsync_directory(stage)
    # Revalidate every protected component and both endpoints immediately before
    # the atomic rename. Lexical lstat checks make a substituted parent symlink
    # fail closed instead of redirecting publication outside the trusted tree.
    ensure_trusted_directory_chain(
        final.parent,
        root=trusted_root,
        create_missing=False,
    )
    require_plain_directory(stage, "staging directory before rename")
    if lstat_or_none(final) is not None:
        raise FormalError(f"final target appeared before publication: {final}")
    os.rename(stage, final)
    require_plain_directory(final, "published unit after rename")
    fsync_directory(final.parent)


def validate_published(
    final: Path,
    unit: tuple[str, int, int],
    expected_anchors: Mapping[str, str] | None = None,
    expected_inputs: Sequence[Mapping[str, Any]] | None = None,
    *,
    trusted_root: Path = ROOT,
) -> dict[str, Any]:
    ensure_trusted_directory_chain(
        final.parent,
        root=trusted_root,
        create_missing=False,
    )
    require_plain_directory(final, "published unit")
    if exact_names(final) != tuple(sorted(PUBLISHED_FILES)):
        raise FormalError(f"published unit has a non-exact artifact set: {final}")
    for name in PUBLISHED_FILES:
        assert_regular_file(final / name, f"published {name}")
    manifest = load_json(final / "execution_manifest.json", "execution manifest")
    complete = load_json(final / "COMPLETE.json", "COMPLETE")
    require_json_unit(manifest, unit, "execution manifest")
    if complete.get("status") != "COMPLETE":
        raise FormalError("COMPLETE status is not COMPLETE")
    for key, value in unit_object(unit).items():
        if complete.get(key) != value:
            raise FormalError("COMPLETE unit identity mismatch")
    if complete.get("execution_manifest_sha256") != sha256_file(
        final / "execution_manifest.json"
    ):
        raise FormalError("COMPLETE does not anchor execution_manifest.json")
    if complete.get("protocol_sha256") != manifest.get("protocol_sha256"):
        raise FormalError("COMPLETE and execution manifest disagree on protocol")
    expected_paths = {
        "protocol_path": PROTOCOL_REL,
        "orchestrator_path": str(SELF_PATH.relative_to(ROOT)),
        "engine_path": ENGINE_REL,
        "spec_path": SPEC_REL,
        "authorization_path": AUTHORIZATION_REL,
        "engine_audit_path": ENGINE_AUDIT_REL,
    }
    for key, value in expected_paths.items():
        if manifest.get(key) != value:
            raise FormalError(f"published manifest path changed: {key}")
    if manifest.get("units_sha256") != units_digest():
        raise FormalError("published manifest formal-scope digest changed")
    whitelist = manifest.get("fixed_input_whitelist")
    if not isinstance(whitelist, list):
        raise FormalError("execution manifest fixed-input whitelist is not a list")
    if manifest.get("fixed_input_whitelist_sha256") != sha256_json(whitelist):
        raise FormalError("execution manifest fixed-input whitelist digest mismatch")
    if expected_inputs is not None and whitelist != list(expected_inputs):
        raise FormalError("published fixed-input whitelist differs from current anchors")
    records = manifest.get("artifacts")
    if not isinstance(records, list) or len(records) != len(PAYLOAD_FILES):
        raise FormalError("execution manifest artifact count is not 12")
    if manifest.get("artifact_count_excluding_manifest_and_COMPLETE") != len(PAYLOAD_FILES):
        raise FormalError("execution manifest declared artifact count changed")
    if manifest.get("complete_excluded_because_written_last") is not True:
        raise FormalError("execution manifest does not attest COMPLETE-last publication")
    if manifest.get("formal_prepublication_audit_ids") != list(
        FORMAL_PREPUBLICATION_AUDIT_IDS
    ):
        raise FormalError("execution manifest pre-publication audit IDs changed")
    if manifest.get("formal_publication_audit_id") != FORMAL_PUBLICATION_AUDIT_ID:
        raise FormalError("execution manifest publication audit ID changed")
    if complete.get("formal_prepublication_audit_ids") != list(
        FORMAL_PREPUBLICATION_AUDIT_IDS
    ):
        raise FormalError("published COMPLETE pre-publication audit IDs changed")
    if complete.get("formal_publication_audit_id") != FORMAL_PUBLICATION_AUDIT_ID:
        raise FormalError("published COMPLETE publication audit ID changed")
    if complete.get("all_formal_publication_audits_pass") is not True:
        raise FormalError("published COMPLETE does not close the F01--F20 gate")
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise FormalError("execution manifest contains a malformed artifact record")
    if [record["path"] for record in records] != list(PAYLOAD_FILES):
        raise FormalError("execution manifest artifact order/set changed")
    for record in records:
        if not isinstance(record["bytes"], int) or record["bytes"] < 0:
            raise FormalError("execution manifest contains an invalid artifact size")
        if not is_sha256(record["sha256"]):
            raise FormalError("execution manifest contains an invalid artifact hash")
        path = final / str(record["path"])
        if path.stat().st_size != record.get("bytes"):
            raise FormalError(f"published artifact size mismatch: {path.name}")
        if sha256_file(path) != record.get("sha256"):
            raise FormalError(f"published artifact hash mismatch: {path.name}")
    if expected_anchors is not None:
        key_map = {
            "protocol_sha256": "protocol_sha256",
            "p0_manifest_sha256": "p0_manifest_sha256",
            "orchestrator_sha256": "orchestrator_sha256",
            "engine_sha256": "engine_sha256",
            "spec_sha256": "spec_sha256",
            "authorization_sha256": "authorization_sha256",
            "engine_audit_sha256": "engine_audit_sha256",
        }
        for manifest_key, anchor_key in key_map.items():
            if manifest.get(manifest_key) != expected_anchors[anchor_key]:
                raise FormalError(f"published manifest anchor changed: {manifest_key}")
        if complete.get("protocol_sha256") != expected_anchors["protocol_sha256"]:
            raise FormalError("published COMPLETE protocol anchor changed")
    validation = validate_payload(
        final,
        unit,
        publication_metadata_allowed=True,
    )
    return {
        "status": "PASS",
        "unit": unit_name(unit),
        "artifact_count": len(PUBLISHED_FILES),
        **validation,
    }

def make_ticket(
    unit: tuple[str, int, int],
    stage: Path,
    inputs: Sequence[Mapping[str, Any]],
    anchors: Mapping[str, str],
) -> dict[str, Any]:
    ticket = {
        "contract": "P1_FORMAL_UNIT_TICKET_V1",
        "unit": unit_object(unit),
        "ordered_scope_index": UNITS.index(unit),
        "ordered_scope_size": len(UNITS),
        "ordered_scope_sha256": units_digest(),
        "stage_directory": str(stage),
        "input_whitelist": list(inputs),
        "input_whitelist_sha256": sha256_json(list(inputs)),
        "required_payload_files": list(PAYLOAD_FILES),
        "csv_schemas": {name: list(schema) for name, schema in CSV_SCHEMAS.items()},
        "arms": list(ARMS),
        "contrasts": list(CONTRASTS),
        "protocol_sha256": anchors["protocol_sha256"],
        "p0_manifest_sha256": anchors["p0_manifest_sha256"],
        "orchestrator_sha256": anchors["orchestrator_sha256"],
        "engine_sha256": anchors["engine_sha256"],
        "spec_sha256": anchors["spec_sha256"],
        "authorization_sha256": anchors["authorization_sha256"],
        "engine_audit_sha256": anchors["engine_audit_sha256"],
        "execution_rules": {
            "development_only": True,
            "cpu_only": True,
            "single_thread": True,
            "one_unit_only": True,
            "no_bootstrap": True,
            "no_hypothesis_judgment": True,
            "no_path_discovery": True,
            "no_official_test_access_or_discovery": True,
        },
    }
    ticket["ticket_sha256"] = sha256_json(ticket)
    return ticket


def invoke_engine(
    ticket: Mapping[str, Any], unit: tuple[str, int, int], anchors: Mapping[str, str]
) -> None:
    env = dict(os.environ)
    env.update(THREAD_ENV)
    command = [sys.executable, str(ROOT / ENGINE_REL), "--formal-ticket-stdin"]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        input=json.dumps(ticket, sort_keys=True, separators=(",", ":")) + "\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        sanitized = parse_sanitized_engine_failure(completed.stderr)
        if sanitized is None or completed.stdout:
            raise FormalError(
                "formal unit engine failed closed without an admissible "
                "sanitized receipt; staging retained"
            )
        stage, error_code = sanitized
        raise FormalError(
            f"formal unit engine failed closed at {stage} "
            f"({error_code}); staging retained"
        )
    if completed.stderr:
        raise FormalError(f"formal unit engine wrote stderr for {unit_name(unit)}")
    lines = completed.stdout.splitlines()
    if len(lines) != 1:
        raise FormalError("formal unit engine did not emit exactly one receipt line")
    try:
        receipt = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise FormalError("formal unit engine receipt is not JSON") from exc
    expected = {
        "status": "PAYLOAD_COMPLETE",
        "unit": unit_name(unit),
        "ticket_sha256": ticket["ticket_sha256"],
        "engine_sha256": anchors["engine_sha256"],
        "protocol_sha256": anchors["protocol_sha256"],
        "input_whitelist_sha256": ticket["input_whitelist_sha256"],
        "payload_files_sha256": sha256_json(list(PAYLOAD_FILES)),
        "official_test_accessed": False,
        "hypothesis_judgment_performed": False,
    }
    if receipt != expected:
        raise FormalError("formal unit engine receipt differs from the exact contract")


def acquire_lock(
    anchors: Mapping[str, str],
    *,
    root: Path = ROOT,
    lock_rel: str = LOCK_REL,
) -> tuple[Path, str, tuple[int, int], str]:
    lock_path = root / lock_rel
    ensure_trusted_directory_chain(
        lock_path.parent,
        root=root,
        create_missing=True,
    )
    if lstat_or_none(lock_path) is not None:
        raise FormalError(
            f"formal-run lock exists and is never reclaimed automatically: {lock_path}"
        )
    token = uuid.uuid4().hex
    payload = {
        "token": token,
        "pid": os.getpid(),
        "started_at_utc": utc_now(),
        "orchestrator_sha256": anchors["orchestrator_sha256"],
        "units_sha256": units_digest(),
        "policy": "never auto-reclaim; inspect interrupted staging and remove manually",
    }
    try:
        ensure_trusted_directory_chain(
            lock_path.parent,
            root=root,
            create_missing=False,
        )
        write_new_json(lock_path, payload)
    except FileExistsError as exc:
        raise FormalError(
            f"formal-run lock exists and is never reclaimed automatically: {lock_path}"
        ) from exc
    metadata = lstat_or_none(lock_path)
    if (
        metadata is None
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise FormalError("new formal-run lock is absent, non-regular, or a symlink")
    identity = (metadata.st_dev, metadata.st_ino)
    return lock_path, token, identity, sha256_file(lock_path)


def release_own_lock(
    lock_path: Path,
    token: str,
    identity: tuple[int, int],
    expected_sha256: str,
    *,
    root: Path = ROOT,
) -> None:
    ensure_trusted_directory_chain(
        lock_path.parent,
        root=root,
        create_missing=False,
    )
    metadata = lstat_or_none(lock_path)
    if (
        metadata is None
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
    ):
        raise FormalError("refusing to remove a missing, symlinked, or non-regular lock")
    if (metadata.st_dev, metadata.st_ino) != identity:
        raise FormalError("refusing to remove a replaced formal-run lock")
    if sha256_file(lock_path) != expected_sha256:
        raise FormalError("refusing to remove a mutated formal-run lock")
    value = load_json(lock_path, "formal-run lock")
    if not isinstance(value, dict):
        raise FormalError("refusing to remove a malformed formal-run lock")
    if value.get("token") != token or value.get("pid") != os.getpid():
        raise FormalError("refusing to remove a formal-run lock not owned by this process")
    ensure_trusted_directory_chain(
        lock_path.parent,
        root=root,
        create_missing=False,
    )
    metadata = lstat_or_none(lock_path)
    if (
        metadata is None
        or stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or (metadata.st_dev, metadata.st_ino) != identity
    ):
        raise FormalError("formal-run lock was replaced before release")
    if sha256_file(lock_path) != expected_sha256:
        raise FormalError("formal-run lock was mutated before release")
    try:
        os.unlink(lock_path)
    except OSError as exc:
        raise FormalError(f"cannot release formal-run lock: {lock_path}") from exc
    fsync_directory(lock_path.parent)


def release_lock_preserving_primary(
    lock_path: Path,
    token: str,
    identity: tuple[int, int],
    expected_sha256: str,
    primary_exception: BaseException | None,
    *,
    root: Path = ROOT,
) -> None:
    """Release a lock without replacing an already active primary exception."""
    try:
        release_own_lock(
            lock_path,
            token,
            identity,
            expected_sha256,
            root=root,
        )
    except BaseException as release_error:
        if primary_exception is None:
            raise
        primary_exception.add_note(
            "formal-run lock release also failed: "
            f"{type(release_error).__name__}: {release_error}"
        )


def execute_frozen_scope(resume_verified: bool) -> None:
    protocol, p0 = verify_live_base_anchors()
    authorization = verify_authorization()
    anchors = {
        "protocol_sha256": PROTOCOL_SHA256,
        "p0_manifest_sha256": P0_SHA256,
        "orchestrator_sha256": sha256_file(SELF_PATH),
        "engine_sha256": authorization["_engine_sha256"],
        "spec_sha256": authorization["_spec_sha256"],
        "authorization_sha256": authorization["_authorization_sha256"],
        "engine_audit_sha256": authorization["_audit_sha256"],
    }
    final_parent = ROOT / FINAL_PARENT_REL
    ensure_trusted_directory_chain(
        ROOT / LOCK_PARENT_REL,
        root=ROOT,
        create_missing=True,
    )
    lock_path, token, lock_identity, lock_sha256 = acquire_lock(anchors, root=ROOT)
    primary_exception: BaseException | None = None
    try:
        ensure_trusted_directory_chain(
            final_parent,
            root=ROOT,
            create_missing=True,
        )
        for unit in UNITS:
            name = unit_name(unit)
            final = final_parent / name
            # Resolve and rehash the exact twelve permitted inputs even when a
            # completed unit is being resumed. Resume is therefore verification,
            # never directory-presence based skipping.
            inputs = fixed_input_records(unit, protocol, p0)
            ensure_trusted_directory_chain(
                final.parent,
                root=ROOT,
                create_missing=False,
            )
            final_metadata = lstat_or_none(final)
            if final_metadata is not None:
                if stat.S_ISLNK(final_metadata.st_mode) or not stat.S_ISDIR(
                    final_metadata.st_mode
                ):
                    raise FormalError(
                        f"final unit target is a symlink or non-directory: {final}"
                    )
                if not resume_verified:
                    raise FormalError(
                        f"final unit already exists; use only --resume-verified "
                        f"after independent authorization to verify and skip it: {name}"
                    )
                validate_published(
                    final,
                    unit,
                    anchors,
                    inputs,
                    trusted_root=ROOT,
                )
                continue

            stage = final_parent / f".{name}.staging-{os.getpid()}-{uuid.uuid4().hex}"
            ensure_trusted_directory_chain(
                stage.parent,
                root=ROOT,
                create_missing=False,
            )
            if lstat_or_none(stage) is not None:
                raise FormalError(f"unique staging path already exists: {stage}")
            try:
                os.mkdir(stage, 0o700)
            except OSError as exc:
                raise FormalError(f"cannot create unique staging directory: {stage}") from exc
            require_plain_directory(stage, "new staging directory")
            try:
                ticket = make_ticket(unit, stage, inputs, anchors)
                invoke_engine(ticket, unit, anchors)
                payload_summary = validate_payload(stage, unit)
                if payload_summary["input_whitelist_sha256"] != ticket[
                    "input_whitelist_sha256"
                ]:
                    raise FormalError("lineage whitelist digest differs from the ticket")
                # Rehash every fixed input and every authorization anchor after
                # computation. A changed source can never be published.
                verify_live_base_anchors()
                current = verify_authorization()
                if current["_authorization_sha256"] != anchors["authorization_sha256"]:
                    raise FormalError("authorization changed during unit computation")
                if current["_engine_sha256"] != anchors["engine_sha256"]:
                    raise FormalError("engine changed during unit computation")
                for record in inputs:
                    verify_hash(
                        ROOT / str(record["path"]),
                        str(record["sha256"]),
                        f"post-computation fixed input {record['path']}",
                    )
                publish_stage(
                    stage,
                    final,
                    unit,
                    anchors,
                    inputs,
                    trusted_root=ROOT,
                )
                validate_published(
                    final,
                    unit,
                    anchors,
                    inputs,
                    trusted_root=ROOT,
                )
            except Exception as exc:
                # Deliberately retain the unique staging directory. It is never
                # treated as complete and is never discovered on resume.
                raise FormalError(
                    f"{name} failed closed; forensic staging retained at {stage}; "
                    f"cause={type(exc).__name__}: {exc}"
                ) from exc
    except BaseException as exc:
        primary_exception = exc
        raise
    finally:
        release_lock_preserving_primary(
            lock_path,
            token,
            lock_identity,
            lock_sha256,
            primary_exception,
            root=ROOT,
        )
    print(
        json.dumps(
            {
                "status": "ALL_UNITS_PUBLISHED_AND_VERIFIED",
                "unit_count": len(UNITS),
                "units_sha256": units_digest(),
                "aggregate_not_run": True,
                "hypothesis_judgment_not_performed": True,
            },
            sort_keys=True,
        )
    )


def synthetic_row(schema: Sequence[str], unit: tuple[str, int, int]) -> dict[str, str]:
    row = {column: "0" for column in schema}
    values = {
        "domain": unit[0],
        "repetition": str(unit[1]),
        "outer_fold": str(unit[2]),
        "row_index": "11",
        "case_id": "synthetic-case",
        "group_id": "synthetic-group",
        "y": "1",
    }
    row.update({key: value for key, value in values.items() if key in row})
    return row


def write_csv(path: Path, schema: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="", closefd=False) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(schema), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(fd)


def make_synthetic_payload(stage: Path, unit: tuple[str, int, int]) -> None:
    prediction = synthetic_row(CSV_SCHEMAS["predictions.csv"], unit)
    for arm in ARMS:
        prediction[f"p_{arm}"] = "0.5"
    for column in ("logit_S", "logit_C", "logit_SP", "logit_CP"):
        prediction[column] = "0"
    write_csv(stage / "predictions.csv", CSV_SCHEMAS["predictions.csv"], [prediction])

    calibration = synthetic_row(CSV_SCHEMAS["calibration_oof.csv"], unit)
    calibration.update(
        {
            "H": "0",
            "reference_C_H": "1",
            "raw_base_logit": "0",
            "prior_restored_base_logit": "0",
            "probe_fit_scope_sha256": "a" * 64,
            "reference_fit_scope_sha256": "b" * 64,
        }
    )
    write_csv(
        stage / "calibration_oof.csv",
        CSV_SCHEMAS["calibration_oof.csv"],
        [calibration],
    )

    per_case_rows = []
    for arm in ARMS:
        row = synthetic_row(CSV_SCHEMAS["per_case_effects.csv"], unit)
        row.update(
            {
                "arm": arm,
                "negative_log_likelihood": "0.6931471805599453",
                "squared_probability_error": "0.25",
                "class_equal_nll_contribution": "0.6931471805599453",
            }
        )
        per_case_rows.append(row)
    write_csv(
        stage / "per_case_effects.csv",
        CSV_SCHEMAS["per_case_effects.csv"],
        per_case_rows,
    )

    metric_rows = []
    for arm in ARMS:
        row = synthetic_row(CSV_SCHEMAS["metrics.csv"], unit)
        row.update(
            {
                "arm": arm,
                "class_equal_NLL": "0.6931471805599453",
                "AUC": "0.5",
                "natural_frequency_NLL": "0.6931471805599453",
                "Brier": "0.25",
                "n_cases": "1",
                "n_groups": "1",
            }
        )
        metric_rows.append(row)
    write_csv(stage / "metrics.csv", CSV_SCHEMAS["metrics.csv"], metric_rows)

    paired_rows = []
    for contrast in CONTRASTS:
        row = synthetic_row(CSV_SCHEMAS["paired_effects.csv"], unit)
        row.update(
            {
                "contrast": contrast,
                "class_equal_NLL_X_minus_Y": "0",
                "AUC_X_minus_Y": "0",
                "favourable_sign_NLL": CONTRAST_SIGNS[contrast],
                "pilot_descriptive_only": "False",
            }
        )
        paired_rows.append(row)
    write_csv(
        stage / "paired_effects.csv",
        CSV_SCHEMAS["paired_effects.csv"],
        paired_rows,
    )

    generic = {"unit": unit_object(unit)}
    write_new_json(stage / "selections.json", {**generic, "synthetic": True})
    write_new_json(stage / "calibrators.json", {**generic, "synthetic": True})
    write_new_json(stage / "prior_lineage.json", {**generic, "synthetic": True})
    write_new_json(
        stage / "lineage.json",
        {
            **generic,
            "access_boundary": {
                "official_test_access": False,
                "official_test_discovery_or_listing": False,
                "input_whitelist_exact": True,
                "input_whitelist_sha256": sha256_json([]),
            },
        },
    )
    write_new_json(
        stage / "audit.json",
        {
            **generic,
            "checks": {
                audit_id: True for audit_id in FORMAL_PREPUBLICATION_AUDIT_IDS
            },
            "all_formal_prepublication_audits_pass": True,
            "hypothesis_judgment": "NOT_PERMITTED_PER_UNIT",
        },
    )
    cost = EXPECTED_COST[unit[0]]
    write_new_json(
        stage / "fit_cost.json",
        {
            **generic,
            "head_fits": {"total_head_fits": cost["head_fits"]},
            "affine_fits": {"fits": cost["affine_fits"]},
            "prior_restoration": {
                "calls": cost["prior_restoration_calls"],
                "concept_records": cost["prior_concept_records"],
            },
            "probe_generation": {"fits": 0},
            "bootstrap_draws": 0,
            "units_run": 1,
        },
    )
    write_new_bytes(stage / "run.log", b"synthetic source-only contract test\n")


def run_self_test() -> None:
    # The self-test is hermetic: it does not read ROOT, the P0 manifest, any P1
    # directory, or any dataset. It tests source contracts only in a temp dir.
    unit = ("cbis", 0, 0)
    anchors = {
        "protocol_sha256": "1" * 64,
        "p0_manifest_sha256": "2" * 64,
        "orchestrator_sha256": "3" * 64,
        "engine_sha256": "4" * 64,
        "spec_sha256": "5" * 64,
        "authorization_sha256": "6" * 64,
        "engine_audit_sha256": "7" * 64,
    }
    checks: dict[str, bool] = {}

    def rejects(action: Any) -> bool:
        try:
            action()
        except FormalError:
            return True
        return False

    def clone(value: Mapping[str, Any]) -> dict[str, Any]:
        return json.loads(json.dumps(value))

    with tempfile.TemporaryDirectory(prefix="p1-formal-source-test-") as temp:
        sandbox = Path(temp)
        synthetic_root = sandbox / "repository"
        os.mkdir(synthetic_root, 0o700)
        parent = synthetic_root / FINAL_PARENT_REL
        ensure_trusted_directory_chain(
            parent,
            root=synthetic_root,
            create_missing=True,
        )
        stage = parent / ".cbis_r0_f0.staging"
        final = parent / "cbis_r0_f0"
        os.mkdir(stage, 0o700)
        make_synthetic_payload(stage, unit)
        summary = validate_payload(stage, unit)
        checks["synthetic_payload_contract"] = summary["n_predictions"] == 1
        checks["complete_absent_before_publication"] = (
            lstat_or_none(stage / "COMPLETE.json") is None
        )
        publish_stage(
            stage,
            final,
            unit,
            anchors,
            [],
            trusted_root=synthetic_root,
        )
        checks["atomic_directory_publication"] = (
            lstat_or_none(stage) is None
            and lstat_or_none(final) is not None
        )
        checks["resume_full_verification"] = (
            validate_published(
                final,
                unit,
                anchors,
                [],
                trusted_root=synthetic_root,
            )["status"]
            == "PASS"
        )
        metrics_path = final / "metrics.csv"
        original_metrics = metrics_path.read_bytes()
        metrics_path.write_bytes(original_metrics + b"\n")
        checks["artifact_mutation_rejected"] = rejects(
            lambda: validate_published(
                final,
                unit,
                anchors,
                [],
                trusted_root=synthetic_root,
            )
        )
        metrics_path.write_bytes(original_metrics)
        extra = final / "UNEXPECTED"
        extra.write_bytes(b"x")
        checks["unexpected_file_rejected"] = rejects(
            lambda: validate_published(
                final,
                unit,
                anchors,
                [],
                trusted_root=synthetic_root,
            )
        )

        derm_stage = parent / ".derm_r4_f4.staging"
        os.mkdir(derm_stage, 0o700)
        make_synthetic_payload(derm_stage, ("derm", 4, 4))
        derm_summary = validate_payload(derm_stage, ("derm", 4, 4))
        checks["derm_synthetic_cost_contract"] = derm_summary["n_predictions"] == 1

        # Reject symlinks at ROOT and at every protected ancestry level.
        real_root = sandbox / "real-root"
        root_link = sandbox / "root-link"
        os.mkdir(real_root, 0o700)
        os.symlink(real_root, root_link, target_is_directory=True)
        checks["symlink_root_rejected"] = rejects(
            lambda: ensure_trusted_directory_chain(
                root_link / FINAL_PARENT_REL,
                root=root_link,
                create_missing=True,
            )
        )
        protected_parts = ("paper_b_readiness", "p1_complete", "units")
        for index, component in enumerate(protected_parts):
            chain_root = sandbox / f"chain-root-{index}"
            os.mkdir(chain_root, 0o700)
            current = chain_root
            for prior in protected_parts[:index]:
                current = current / prior
                os.mkdir(current, 0o700)
            redirect = sandbox / f"redirect-{index}"
            os.mkdir(redirect, 0o700)
            os.symlink(redirect, current / component, target_is_directory=True)
            checks[f"symlink_component_{index}_rejected"] = rejects(
                lambda chain_root=chain_root: ensure_trusted_directory_chain(
                    chain_root.joinpath(*protected_parts),
                    root=chain_root,
                    create_missing=True,
                )
            )
        file_root = sandbox / "file-component-root"
        os.mkdir(file_root, 0o700)
        write_new_bytes(file_root / "paper_b_readiness", b"not a directory\n")
        checks["regular_file_component_rejected"] = rejects(
            lambda: ensure_trusted_directory_chain(
                file_root / FINAL_PARENT_REL,
                root=file_root,
                create_missing=True,
            )
        )

        # Exercise lock acquisition, non-reclamation, ownership, and exception
        # precedence without touching the live Paper-B lock location.
        lock_root = sandbox / "lock-root-clean"
        os.mkdir(lock_root, 0o700)
        lock_path, lock_token, lock_identity, lock_sha256 = acquire_lock(
            anchors,
            root=lock_root,
        )
        release_own_lock(
            lock_path,
            lock_token,
            lock_identity,
            lock_sha256,
            root=lock_root,
        )
        checks["lock_clean_acquire_release"] = lstat_or_none(lock_path) is None

        existing_root = sandbox / "lock-root-existing"
        os.mkdir(existing_root, 0o700)
        (
            existing_path,
            existing_token,
            existing_identity,
            existing_sha256,
        ) = acquire_lock(anchors, root=existing_root)
        checks["existing_lock_not_reclaimed"] = (
            rejects(lambda: acquire_lock(anchors, root=existing_root))
            and lstat_or_none(existing_path) is not None
        )
        release_own_lock(
            existing_path,
            existing_token,
            existing_identity,
            existing_sha256,
            root=existing_root,
        )

        ownership_root = sandbox / "lock-root-ownership"
        os.mkdir(ownership_root, 0o700)
        (
            ownership_path,
            ownership_token,
            ownership_identity,
            ownership_sha256,
        ) = acquire_lock(anchors, root=ownership_root)
        checks["mutated_lock_token_rejected"] = (
            rejects(
                lambda: release_own_lock(
                    ownership_path,
                    "not-the-owner-token",
                    ownership_identity,
                    ownership_sha256,
                    root=ownership_root,
                )
            )
            and lstat_or_none(ownership_path) is not None
        )
        release_own_lock(
            ownership_path,
            ownership_token,
            ownership_identity,
            ownership_sha256,
            root=ownership_root,
        )

        replaced_root = sandbox / "lock-root-replaced"
        os.mkdir(replaced_root, 0o700)
        (
            replaced_path,
            replaced_token,
            replaced_identity,
            replaced_sha256,
        ) = acquire_lock(anchors, root=replaced_root)
        replaced_bytes = replaced_path.read_bytes()
        retained_original = replaced_path.with_name("retained-original.lock")
        os.rename(replaced_path, retained_original)
        write_new_bytes(replaced_path, replaced_bytes)
        checks["identical_content_replaced_lock_rejected"] = (
            rejects(
                lambda: release_own_lock(
                    replaced_path,
                    replaced_token,
                    replaced_identity,
                    replaced_sha256,
                    root=replaced_root,
                )
            )
            and lstat_or_none(replaced_path) is not None
            and lstat_or_none(retained_original) is not None
        )

        mutated_root = sandbox / "lock-root-mutated"
        os.mkdir(mutated_root, 0o700)
        (
            mutated_path,
            mutated_token,
            mutated_identity,
            mutated_sha256,
        ) = acquire_lock(anchors, root=mutated_root)
        with mutated_path.open("ab") as handle:
            handle.write(b" ")
            handle.flush()
            os.fsync(handle.fileno())
        checks["same_inode_mutated_lock_rejected"] = (
            rejects(
                lambda: release_own_lock(
                    mutated_path,
                    mutated_token,
                    mutated_identity,
                    mutated_sha256,
                    root=mutated_root,
                )
            )
            and lstat_or_none(mutated_path) is not None
        )

        missing_root = sandbox / "lock-root-missing"
        os.mkdir(missing_root, 0o700)
        (
            missing_path,
            missing_token,
            missing_identity,
            missing_sha256,
        ) = acquire_lock(anchors, root=missing_root)
        retained_missing = missing_path.with_name("retained-missing.lock")
        os.rename(missing_path, retained_missing)
        checks["missing_lock_rejected"] = (
            rejects(
                lambda: release_own_lock(
                    missing_path,
                    missing_token,
                    missing_identity,
                    missing_sha256,
                    root=missing_root,
                )
            )
            and lstat_or_none(retained_missing) is not None
        )

        symlink_lock_root = sandbox / "lock-root-symlink"
        os.mkdir(symlink_lock_root, 0o700)
        (
            symlink_lock_path,
            symlink_lock_token,
            symlink_lock_identity,
            symlink_lock_sha256,
        ) = acquire_lock(anchors, root=symlink_lock_root)
        retained_symlink_target = symlink_lock_path.with_name("retained-target.lock")
        os.rename(symlink_lock_path, retained_symlink_target)
        os.symlink(retained_symlink_target, symlink_lock_path)
        checks["symlinked_lock_rejected"] = (
            rejects(
                lambda: release_own_lock(
                    symlink_lock_path,
                    symlink_lock_token,
                    symlink_lock_identity,
                    symlink_lock_sha256,
                    root=symlink_lock_root,
                )
            )
            and stat.S_ISLNK(os.lstat(symlink_lock_path).st_mode)
        )

        malformed_root = sandbox / "lock-root-malformed"
        os.mkdir(malformed_root, 0o700)
        (
            malformed_path,
            malformed_token,
            malformed_identity,
            malformed_sha256,
        ) = acquire_lock(anchors, root=malformed_root)
        malformed_path.write_bytes(b"not-json\n")
        checks["malformed_lock_rejected"] = (
            rejects(
                lambda: release_own_lock(
                    malformed_path,
                    malformed_token,
                    malformed_identity,
                    malformed_sha256,
                    root=malformed_root,
                )
            )
            and lstat_or_none(malformed_path) is not None
        )

        sole_root = sandbox / "lock-root-sole-release"
        os.mkdir(sole_root, 0o700)
        sole_path, sole_token, sole_identity, sole_sha256 = acquire_lock(
            anchors,
            root=sole_root,
        )
        checks["sole_release_failure_raised"] = rejects(
            lambda: release_lock_preserving_primary(
                sole_path,
                sole_token + "-wrong",
                sole_identity,
                sole_sha256,
                None,
                root=sole_root,
            )
        )

        combined_root = sandbox / "lock-root-combined"
        os.mkdir(combined_root, 0o700)
        (
            combined_path,
            combined_token,
            combined_identity,
            combined_sha256,
        ) = acquire_lock(anchors, root=combined_root)
        primary = RuntimeError("synthetic primary failure")
        release_lock_preserving_primary(
            combined_path,
            combined_token + "-wrong",
            combined_identity,
            combined_sha256,
            primary,
            root=combined_root,
        )
        notes = getattr(primary, "__notes__", [])
        checks["combined_failure_preserves_primary_with_note"] = (
            str(primary) == "synthetic primary failure"
            and len(notes) == 1
            and "lock release also failed" in notes[0]
            and lstat_or_none(combined_path) is not None
        )

        # The external V4 receipt is a path-and-hash prerequisite, not a status
        # string accepted on trust. Test exact acceptance then one-byte tamper.
        gate_root = sandbox / "gate-root"
        os.mkdir(gate_root, 0o700)
        synthetic_receipt_rel = "receipts/v4.json"
        synthetic_receipt_path = gate_root / synthetic_receipt_rel
        ensure_trusted_directory_chain(
            synthetic_receipt_path.parent,
            root=gate_root,
            create_missing=True,
        )
        write_new_json(synthetic_receipt_path, {"status": "SYNTHETIC_PASS"})
        synthetic_receipt_sha = sha256_file(synthetic_receipt_path)
        verify_pilot_external_audit_at(
            gate_root,
            synthetic_receipt_rel,
            synthetic_receipt_sha,
        )
        checks["v4_receipt_exact_hash_accepted"] = True
        with synthetic_receipt_path.open("ab") as handle:
            handle.write(b" ")
            handle.flush()
            os.fsync(handle.fileno())
        checks["v4_receipt_tamper_rejected"] = rejects(
            lambda: verify_pilot_external_audit_at(
                gate_root,
                synthetic_receipt_rel,
                synthetic_receipt_sha,
            )
        )

        common_expected = {
            "protocol_sha256": "1" * 64,
            "p0_manifest_sha256": "2" * 64,
            "orchestrator_sha256": "3" * 64,
            "engine_sha256": "4" * 64,
            "spec_sha256": "5" * 64,
            "units_sha256": "6" * 64,
            "pilot_external_audit_path": PILOT_EXTERNAL_AUDIT_REL,
            "pilot_external_audit_sha256": PILOT_EXTERNAL_AUDIT_SHA256,
        }
        audit_document: dict[str, Any] = {
            "status": "PASS",
            "scope": "P1_FORMAL_UNIT_ENGINE_PRE_OUTCOME",
            **common_expected,
            "source_review_complete": True,
            "synthetic_contract_tests_pass": True,
            "cpu_serial_enforced": True,
            "whitelist_contract_pass": True,
            "atomic_resume_contract_pass": True,
            "pilot_gate_verified": True,
            "formal_audit_mapping_reviewed": True,
            "p1_outcomes_accessed": False,
            "official_test_accessed": False,
            "reviewer": "synthetic-independent-reviewer",
            "completed_at_utc": "2026-09-16T00:00:00+00:00",
        }
        validate_engine_audit_document(audit_document, common_expected)
        checks["engine_audit_exact_schema_accepted"] = True

        authorization_expected: dict[str, Any] = {
            **common_expected,
            "engine_audit_path": ENGINE_AUDIT_REL,
            "engine_audit_sha256": "7" * 64,
        }
        authorization_document: dict[str, Any] = {
            "status": "AUTHORIZED_PRE_OUTCOME",
            "authorized": True,
            **authorization_expected,
            "pilot_gate_status": "INDEPENDENTLY_AUDITED_PASS",
            "pilot_unit": {"domain": "cbis", "repetition": 0, "outer_fold": 0},
            "pilot_protocol_a03_a19_attested": True,
            "formal_overlay_approved": True,
            "issued_at_utc": "2026-09-16T00:00:00+00:00",
        }
        validate_authorization_document(
            authorization_document,
            authorization_expected,
        )
        checks["authorization_exact_schema_accepted"] = True

        for label, key, replacement in (
            ("wrong_v4_path", "pilot_external_audit_path", "wrong/v4.json"),
            ("wrong_v4_hash", "pilot_external_audit_sha256", "0" * 64),
            ("stale_orchestrator", "orchestrator_sha256", "8" * 64),
            ("stale_engine", "engine_sha256", "9" * 64),
            ("stale_spec", "spec_sha256", "a" * 64),
        ):
            changed_audit = clone(audit_document)
            changed_audit[key] = replacement
            checks[f"engine_audit_{label}_rejected"] = rejects(
                lambda changed_audit=changed_audit: validate_engine_audit_document(
                    changed_audit,
                    common_expected,
                )
            )
            changed_authorization = clone(authorization_document)
            changed_authorization[key] = replacement
            checks[f"authorization_{label}_rejected"] = rejects(
                lambda changed_authorization=changed_authorization: (
                    validate_authorization_document(
                        changed_authorization,
                        authorization_expected,
                    )
                )
            )

        missing_audit = clone(audit_document)
        missing_audit.pop("pilot_external_audit_sha256")
        checks["engine_audit_missing_key_rejected"] = rejects(
            lambda: validate_engine_audit_document(missing_audit, common_expected)
        )
        extra_audit = clone(audit_document)
        extra_audit["unexpected"] = True
        checks["engine_audit_extra_key_rejected"] = rejects(
            lambda: validate_engine_audit_document(extra_audit, common_expected)
        )
        renamed_audit = clone(audit_document)
        renamed_audit["pilot_audit_sha256"] = renamed_audit.pop(
            "pilot_external_audit_sha256"
        )
        checks["engine_audit_renamed_key_rejected"] = rejects(
            lambda: validate_engine_audit_document(renamed_audit, common_expected)
        )

        missing_authorization = clone(authorization_document)
        missing_authorization.pop("pilot_external_audit_path")
        checks["authorization_missing_key_rejected"] = rejects(
            lambda: validate_authorization_document(
                missing_authorization,
                authorization_expected,
            )
        )
        extra_authorization = clone(authorization_document)
        extra_authorization["unexpected"] = True
        checks["authorization_extra_key_rejected"] = rejects(
            lambda: validate_authorization_document(
                extra_authorization,
                authorization_expected,
            )
        )
        renamed_authorization = clone(authorization_document)
        renamed_authorization["pilot_audit_path"] = renamed_authorization.pop(
            "pilot_external_audit_path"
        )
        checks["authorization_renamed_key_rejected"] = rejects(
            lambda: validate_authorization_document(
                renamed_authorization,
                authorization_expected,
            )
        )

    expected_names = (
        tuple(unit_name(unit) for unit in UNITS[:5])
        + tuple(unit_name(unit) for unit in UNITS[5:])
    )
    checks["exact_static_unit_count"] = len(UNITS) == 30
    checks["exact_static_unit_order"] = expected_names == tuple(
        [f"cbis_r0_f{fold}" for fold in range(5)]
        + [
            f"derm_r{repetition}_f{fold}"
            for repetition in range(5)
            for fold in range(5)
        ]
    )
    checks["unit_digest_stable_in_process"] = len(units_digest()) == 64
    checks["audit_partition_exact"] = (
        len(FORMAL_PREPUBLICATION_AUDIT_IDS) == 19
        and FORMAL_PREPUBLICATION_AUDIT_IDS[0] == "F01"
        and FORMAL_PREPUBLICATION_AUDIT_IDS[-1] == "F19"
        and FORMAL_PUBLICATION_AUDIT_ID == "F20"
    )
    aggregate_cost = {
        key: sum(EXPECTED_COST[unit[0]][key] for unit in UNITS)
        for key in (
            "head_fits",
            "affine_fits",
            "prior_restoration_calls",
            "prior_concept_records",
        )
    }
    checks["static_aggregate_cost"] = aggregate_cost == {
        "head_fits": 5130,
        "affine_fits": 60,
        "prior_restoration_calls": 610,
        "prior_concept_records": 4525,
    }
    checks["cpu_only_serial_environment"] = THREAD_ENV == {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    if not all(checks.values()):
        raise FormalError(f"synthetic source-only self-test failed: {checks}")
    print(
        json.dumps(
            {
                "status": "PASS",
                "scope": "SOURCE_ONLY_SYNTHETIC",
                "project_inputs_read": False,
                "p1_outputs_read": False,
                "official_test_accessed": False,
                "checks": checks,
                "unit_count": len(UNITS),
                "units_sha256": units_digest(),
                "static_aggregate_cost": aggregate_cost,
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed Paper-B P1 formal development-scope orchestrator. "
            "There are deliberately no domain, repetition, fold, input, output, "
            "seed, model, or grid selectors."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--self-test",
        action="store_true",
        help="run a hermetic source-only synthetic contract test",
    )
    mode.add_argument(
        "--execute-frozen-development-scope",
        action="store_true",
        help=(
            "execute only the literal 30-unit frozen development scope; this "
            "fails closed unless separately audited and hash-authorized"
        ),
    )
    parser.add_argument(
        "--resume-verified",
        action="store_true",
        help=(
            "with formal execution only, fully verify and skip already published "
            "units; never trusts directory presence alone"
        ),
    )
    args = parser.parse_args()
    if args.resume_verified and not args.execute_frozen_development_scope:
        parser.error("--resume-verified requires --execute-frozen-development-scope")
    return args


def main() -> int:
    args = parse_args()
    try:
        if args.self_test:
            run_self_test()
        else:
            execute_frozen_scope(args.resume_verified)
    except FormalError as exc:
        print(f"FORMAL_FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
