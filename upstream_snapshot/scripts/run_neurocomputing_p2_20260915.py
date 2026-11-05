#!/usr/bin/env python3
"""Paper B P2 preparation runner.

The checked-in protocol is intentionally DRAFT_NOT_FROZEN.  The default command
performs a result-blind structural audit of the already-audited development
panels.  Scientific execution is guarded and deliberately not implemented in
this preparation revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = REPO / "paper_b_readiness/P2_PREP/P2_PROTOCOL_DRAFT.json"
OUTPUT_CONTRACT = REPO / "paper_b_readiness/P2_PREP/P2_OUTPUT_CONTRACT.json"
SOURCE_RECEIPT_OUTPUT = REPO / "paper_b_readiness/P2_PREP/SOURCE_RECEIPT.json"
PROTOCOL_MARKDOWN = REPO / "paper_b_readiness/P2_PREP/P2_PROTOCOL.md"
STRUCTURAL_TEST_SUITE = (
    REPO / "tests/test_neurocomputing_p2_structure_20260915.py"
)
DEFAULT_PANELS = REPO / "results/deadline_nested_20260912"
P0_MANIFEST = REPO / "paper_b_readiness/P0_HASH_MANIFEST.json"
P0_MANIFEST_SHA256 = "d00b9d304fd87d98319a1b85315c38accbb83f758abe97b62c02a2fdb91cfdcd"
P0_FROZEN_RESULT_AGGREGATE_SHA256 = (
    "50bdf69fca67c8dcc79a2a773d50a82f3d84158cf36317a4596340c7d88acab6"
)
SOURCE_PROTOCOL = DEFAULT_PANELS / "protocol.json"
SOURCE_PROTOCOL_SHA256 = "9b1092b78840b31da47cc1a85a67ae413b0e64d237a848ace7fe94add0f51d09"
DOMAIN_SOURCE_AGGREGATE_RECIPE = {
    "id": "paper_b_p2_domain_source_aggregate_v1",
    "encoding": "utf-8",
    "header_record": "{recipe_id}\\t{domain}\\n",
    "file_record": (
        "{domain}\\t{unit}\\t{repository_relative_path}\\t{bytes}\\t{sha256}\\n"
    ),
    "domain_order": ["cbis", "derm"],
    "unit_order_source": "source.ordered_units",
    "file_order_source": "source.exact_per_unit_files",
    "record_source": "P0_HASH_MANIFEST.json:frozen_result_files",
}
SOURCE_DOMAIN_AGGREGATES = {
    "cbis": "b7cf810ff71a54430fa1e1f65bf8a550d94bb19b940bb57a6793877ba30ec4a6",
    "derm": "87239870ac4c9450ca68b0cbf55efde14af2749ad890c4b5647fa4be3c67ccc8",
}

SIGNED_FAIL_AUDIT = (
    REPO / "paper_b_readiness/P2_PREP/P2_INDEPENDENT_RESULT_BLIND_AUDIT_20260916.json"
)
SIGNED_FAIL_AUDIT_SHA256 = (
    "d6246271f583f066b56bad7eecfb618689f7e2e5321b898a5cf1a19a78096365"
)
AMENDMENT_LEDGER = (
    REPO / "paper_b_readiness/P2_PREP/P2_PRE_FREEZE_AMENDMENTS_20260916.json"
)
FIXED_LR_CONFIGURATION = {
    "solver": "newton-cholesky",
    "class_weight": "balanced",
    "C_grid": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "fit_intercept": True,
    "max_iter": 20_000,
    "tol": 1e-12,
    "random_state": 0,
    "tie_rule": "smallest_C",
}
FIXED_EQUIVALENCE_TOLERANCE = 1e-5
FIXED_SYNTHETIC_GATE = {
    "random_seed": 2026091601,
    "sample_count": 480,
    "feature_count": 7,
    "temperatures": [0.25, 4.0],
    "C_grid": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "comparison_count": 12,
    "require_all_converged": True,
    "require_zero_warnings": True,
}
FIXED_RECIPE_REVISION_CONTROL = {
    "revision_count": 1,
    "maximum_pre_freeze_recipe_revisions": 1,
    "revision_reason": "documented_result_blind_numerical_equivalence_failure_only",
    "prior_recipe": "draft_v2_lbfgs",
    "amended_recipe": "draft_v3_newton_cholesky",
    "scientific_hypothesis_changed": False,
    "statistical_estimands_changed": False,
    "fit_count_changed": False,
    "v2_cost_ledger_status": "invalidated_by_solver_amendment",
    "v3_cost_ledger_status": (
        "reissued_same_fit_counts_awaiting_independent_audit"
    ),
    "additional_revision_permitted": False,
}
PRE_AMENDMENT_SNAPSHOT_SHA256 = {
    "runner": "75a4247cb0c8f51820ea554eed65f9b5d665f1d1a0275b139bc9db980bdceaaa",
    "tests": "94559a2bb71fc7c0f26de6e69b1e3b8b002d5a6fd0206a376e5c715996b353c8",
    "asset_audit": "201452bb80d4acad10c6279a19c79ab4a21e8bb857426cbb5b209ae785815684",
    "signed_fail_audit": SIGNED_FAIL_AUDIT_SHA256,
    "output_contract": "8b877839e177e8b1b774566155ab54206d99e14795df3959697a230dc02495af",
    "protocol_json": "2e80db1352f8b96c54a9b984a5caf1887ad5213d3e786d12265d3c075686aac6",
    "protocol_markdown": "8eba9dc7ed36011c75ade1ee0cc040076d456405a935a860b556e343a1619eec",
    "prior_math_audit": "0ab0765196efa1e7655587a778f1919d5c221255c36ec56f407d38a7f68e5c4c",
    "source_receipt": "355e2e27be89cff733283a102dda678275435f9b6bc7a2e11fd07a461a1f367e",
}

CBIS_UNITS = tuple(f"cbis_r0_f{fold}" for fold in range(5))
DERM_UNITS = tuple(
    f"derm_r{repetition}_f{fold}"
    for repetition in range(5)
    for fold in range(5)
)
EXPECTED_UNITS = CBIS_UNITS + DERM_UNITS
REQUIRED_FILES = (
    "panel.npz",
    "generation.json",
    "selection.json",
    "heads.json",
    "predictions.csv",
    "COMPLETE.json",
)
REQUIRED_ARRAYS = (
    "y", "ids", "groups", "gold", "widths", "t", "e", "raw", "val",
    "f0", "h0", "x0", "z0", "f1", "h1", "x1", "z1",
    "f2", "h2", "x2", "z2", "f3", "h3", "x3", "z3",
)
DOMAIN_UNITS = (
    ("cbis", CBIS_UNITS),
    ("derm", DERM_UNITS),
)


class ProtocolGateError(RuntimeError):
    """Raised before any scientific operation when the protocol is not frozen."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _typed_json_equal(left: Any, right: Any) -> bool:
    """Compare JSON-like values without Python's ``False == 0`` coercion."""
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if set(left) != set(right):
            return False
        return all(_typed_json_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) or isinstance(right, list):
        if not isinstance(left, list) or not isinstance(right, list):
            return False
        return len(left) == len(right) and all(
            _typed_json_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if type(left) is not type(right):
        return False
    return bool(left == right)


def _require_fixed_single_link_regular_file(
    path: Path, expected: Path, label: str
) -> None:
    """Reject path substitution, links, and nonregular bound artifacts."""
    if path != expected:
        raise ProtocolGateError(f"{label} path changed")
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise ProtocolGateError(f"{label} is missing") from exc
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ProtocolGateError(f"{label} must be a regular non-symlink file")
    if metadata.st_nlink != 1:
        raise ProtocolGateError(f"{label} hardlink count must equal one")


def load_protocol(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_amendment_ledger() -> dict[str, Any]:
    """Verify the one allowed result-blind pre-freeze amendment and snapshots."""
    _require_fixed_single_link_regular_file(
        AMENDMENT_LEDGER, AMENDMENT_LEDGER, "pre-freeze amendment ledger"
    )
    _require_fixed_single_link_regular_file(
        SIGNED_FAIL_AUDIT, SIGNED_FAIL_AUDIT, "signed triggering FAIL audit"
    )
    ledger = load_protocol(AMENDMENT_LEDGER)
    expected_scalars = {
        "ledger_id": "paper_b_p2_pre_freeze_amendments_20260916_v1",
        "status": (
            "CANDIDATE_AMENDED_NOT_FROZEN_AWAITING_FRESH_INDEPENDENT_AUDIT"
        ),
        "amendment_id": "P2-NUM-001",
        "trigger_audit_path": SIGNED_FAIL_AUDIT.relative_to(REPO).as_posix(),
        "trigger_audit_sha256": SIGNED_FAIL_AUDIT_SHA256,
        "scientific_results_read": False,
        "p1_outputs_read": False,
        "official_test_accessed": False,
        "scientific_execution_authorized": False,
        "freeze_authorized": False,
        "required_next_gate": "fresh_independent_result_blind_audit",
    }
    for key, expected in expected_scalars.items():
        if not _typed_json_equal(ledger.get(key), expected):
            raise ProtocolGateError(f"pre-freeze amendment ledger changed: {key}")
    if not _typed_json_equal(
        ledger.get("recipe_revision_control"), FIXED_RECIPE_REVISION_CONTROL
    ):
        raise ProtocolGateError("amendment-ledger recipe controls changed")
    if not _typed_json_equal(
        ledger.get("preserved_pre_amendment_snapshot_sha256"),
        PRE_AMENDMENT_SNAPSHOT_SHA256,
    ):
        raise ProtocolGateError("pre-amendment snapshot hashes changed")
    expected_change = {
        "field": "logistic_regression.solver",
        "from": "lbfgs",
        "to": "newton-cholesky",
        "reason": "signed_result_blind_synthetic_numerical_equivalence_failure",
        "acceptance_tolerance_changed": False,
        "acceptance_operator_changed": False,
        "scientific_hypothesis_changed": False,
        "statistical_estimands_changed": False,
        "fit_count_changed": False,
    }
    if not _typed_json_equal(ledger.get("sole_protocol_change"), expected_change):
        raise ProtocolGateError("pre-freeze amendment scope changed")
    expected_current = {
        "runner": sha256_file(Path(__file__).resolve()),
        "tests": sha256_file(STRUCTURAL_TEST_SUITE),
        "output_contract": sha256_file(OUTPUT_CONTRACT),
        "protocol_json": sha256_file(DEFAULT_PROTOCOL),
        "protocol_markdown": sha256_file(PROTOCOL_MARKDOWN),
        "source_receipt": sha256_file(SOURCE_RECEIPT_OUTPUT),
        "signed_fail_audit": sha256_file(SIGNED_FAIL_AUDIT),
    }
    if not _typed_json_equal(
        ledger.get("current_candidate_snapshot_sha256"), expected_current
    ):
        raise ProtocolGateError("amended candidate snapshot hashes changed")
    return {
        "amendment_id": ledger["amendment_id"],
        "ledger_path": AMENDMENT_LEDGER.relative_to(REPO).as_posix(),
        "ledger_sha256": sha256_file(AMENDMENT_LEDGER),
        "required_next_gate": ledger["required_next_gate"],
    }


def require_scientific_gate(
    protocol: Mapping[str, Any], supplied_sha256: str | None
) -> None:
    actual = canonical_json_sha256(protocol)
    conditions = {
        "status_is_frozen": protocol.get("status") == "FROZEN",
        "scientific_execution_authorized": bool(
            protocol.get("scientific_execution_authorized", False)
        ),
        "official_test_remains_forbidden": not bool(
            protocol.get("official_test_access_authorized", False)
        ),
        "matching_protocol_sha256": supplied_sha256 == actual,
    }
    if not all(conditions.values()):
        failed = ", ".join(key for key, value in conditions.items() if not value)
        raise ProtocolGateError(
            "P2 scientific execution refused before data/model fitting; failed: " + failed
        )


def stable_sigmoid(logit: np.ndarray) -> np.ndarray:
    logit = np.asarray(logit, dtype=np.float64)
    out = np.empty_like(logit)
    positive = logit >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exp_value = np.exp(logit[~positive])
    out[~positive] = exp_value / (1.0 + exp_value)
    return out


def clipped_logit(probability: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    probability = np.asarray(probability, dtype=np.float64)
    clipped = np.clip(probability, epsilon, 1.0 - epsilon)
    mask = clipped != probability
    return np.log(clipped) - np.log1p(-clipped), mask


def binary_temperature(
    probability: np.ndarray, tau: float, epsilon: float
) -> tuple[np.ndarray, np.ndarray]:
    if tau <= 0:
        raise ValueError("temperature must be positive")
    logit, mask = clipped_logit(probability, epsilon)
    return stable_sigmoid(logit / tau), mask


def block_temperature(
    probability: np.ndarray,
    widths: Sequence[int],
    tau: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply temperature independently to contiguous multiclass blocks."""
    probability = np.asarray(probability, dtype=np.float64)
    checked_widths = validate_block_widths(probability, widths)
    if tau <= 0:
        raise ValueError("temperature must be positive")
    if not (0.0 < epsilon < 1.0):
        raise ValueError("epsilon must lie strictly between zero and one")
    transformed = np.empty_like(probability)
    clipped_mask = np.zeros_like(probability, dtype=bool)
    start = 0
    for width in checked_widths:
        stop = start + width
        block = probability[:, start:stop]
        clipped = np.clip(block, epsilon, None)
        clipped_mask[:, start:stop] = clipped != block
        clipped /= clipped.sum(axis=1, keepdims=True)
        powered = np.exp(np.log(clipped) / tau)
        transformed[:, start:stop] = powered / powered.sum(axis=1, keepdims=True)
        start = stop
    return transformed, clipped_mask


def reference_log_odds(
    probability: np.ndarray, widths: Sequence[int], epsilon: float
) -> np.ndarray:
    """Use each frozen block's first category as the historical reference.

    This deliberately mirrors ``run_deadline_nested_20260912.py``: the first
    category is dropped and each remaining coordinate is ``log(p_j/p_0)``.
    Changing the reference category is an invertible reparameterization, but
    ordinary L2 regularization is not invariant to that coordinate change.
    """
    probability = np.asarray(probability, dtype=np.float64)
    checked_widths = validate_block_widths(probability, widths)
    if not (0.0 < epsilon < 1.0):
        raise ValueError("epsilon must lie strictly between zero and one")
    values: list[np.ndarray] = []
    start = 0
    for width in checked_widths:
        stop = start + width
        block = np.clip(probability[:, start:stop], epsilon, None)
        block /= block.sum(axis=1, keepdims=True)
        values.append(np.log(block[:, 1:]) - np.log(block[:, [0]]))
        start = stop
    return np.concatenate(values, axis=1)


def validate_block_widths(
    probability: np.ndarray, widths: Sequence[int]
) -> tuple[int, ...]:
    """Validate that frozen positive integer block widths exactly partition columns."""
    probability = np.asarray(probability)
    if probability.ndim != 2:
        raise ValueError("probability must be a two-dimensional matrix")
    checked: list[int] = []
    for width in widths:
        if isinstance(width, (bool, np.bool_)) or not isinstance(
            width, (int, np.integer)
        ):
            raise ValueError("every block width must be a positive integer")
        integer_width = int(width)
        if integer_width <= 0:
            raise ValueError("every block width must be a positive integer")
        checked.append(integer_width)
    if not checked or sum(checked) != probability.shape[1]:
        raise ValueError("widths must exactly partition the probability columns")
    return tuple(checked)


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, training: np.ndarray) -> "Standardizer":
        training = np.asarray(training, dtype=np.float64)
        mean = training.mean(axis=0)
        scale = training.std(axis=0, ddof=0)
        scale = np.where(scale <= 1e-12, 1.0, scale)
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.scale

    def unstandardize_linear_head(
        self, coefficient: np.ndarray, intercept: float
    ) -> tuple[np.ndarray, float]:
        raw_coefficient = np.asarray(coefficient, dtype=np.float64) / self.scale
        raw_intercept = float(intercept - np.dot(raw_coefficient, self.mean))
        return raw_coefficient, raw_intercept


def exact_temperature_parameter_map(
    coefficient: np.ndarray, intercept: float, c_value: float, tau: float
) -> tuple[np.ndarray, float, float]:
    if tau <= 0 or c_value <= 0:
        raise ValueError("tau and C must be positive")
    return (
        tau * np.asarray(coefficient, dtype=np.float64),
        float(intercept),
        tau * tau * float(c_value),
    )


def class_equal_binary_cross_entropy(
    y: np.ndarray, logit: np.ndarray
) -> float:
    y = np.asarray(y, dtype=np.int64)
    logit = np.asarray(logit, dtype=np.float64)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("class-equal loss requires both binary classes")
    per_case = np.logaddexp(0.0, logit) - y * logit
    return float(0.5 * (per_case[y == 0].mean() + per_case[y == 1].mean()))


def analytic_equivalence_check(
    values: np.ndarray,
    coefficient: np.ndarray,
    intercept: float,
    c_value: float,
    tau: float,
) -> dict[str, float]:
    mapped_w, mapped_b, mapped_c = exact_temperature_parameter_map(
        coefficient, intercept, c_value, tau
    )
    original_logit = np.asarray(values) @ np.asarray(coefficient) + intercept
    mapped_logit = (np.asarray(values) / tau) @ mapped_w + mapped_b
    return {
        "mapped_C": mapped_c,
        "max_abs_logit_difference": float(np.max(np.abs(original_logit - mapped_logit))),
        "max_abs_prediction_difference": float(
            np.max(np.abs(stable_sigmoid(original_logit) - stable_sigmoid(mapped_logit)))
        ),
    }


def validate_numerical_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed if the amended pre-freeze numerical recipe drifts."""
    for path, label in (
        (DEFAULT_PROTOCOL, "P2 protocol"),
        (OUTPUT_CONTRACT, "P2 output contract"),
        (PROTOCOL_MARKDOWN, "P2 protocol Markdown"),
        (STRUCTURAL_TEST_SUITE, "P2 structural/synthetic test suite"),
        (SOURCE_RECEIPT_OUTPUT, "P2 source receipt"),
        (SIGNED_FAIL_AUDIT, "signed triggering FAIL audit"),
    ):
        _require_fixed_single_link_regular_file(path, path, label)
    if not _typed_json_equal(protocol, load_protocol(DEFAULT_PROTOCOL)):
        raise ProtocolGateError(
            "numerical-gate input differs from fixed candidate protocol"
        )
    if protocol.get("protocol_id") != "paper_b_p2_20260916_draft_v3":
        raise ProtocolGateError("unexpected P2 protocol revision for numerical gate")
    if protocol.get("status") != "DRAFT_NOT_FROZEN":
        raise ProtocolGateError("numerical gate is defined only for the unfrozen candidate")
    if protocol.get("scientific_execution_authorized") is not False:
        raise ProtocolGateError("numerical gate cannot authorize scientific execution")
    if protocol.get("freeze_authorized") is not False:
        raise ProtocolGateError("numerical gate cannot authorize candidate freeze")
    if protocol.get("official_test_access_authorized") is not False:
        raise ProtocolGateError("numerical gate cannot authorize official-test access")
    if protocol.get("p1_access_authorized") is not False:
        raise ProtocolGateError("numerical gate cannot authorize P1 access")
    if not _typed_json_equal(
        protocol.get("logistic_regression"), FIXED_LR_CONFIGURATION
    ):
        raise ProtocolGateError("fixed logistic-regression recipe changed")
    if not _typed_json_equal(
        protocol.get("recipe_revision_control"), FIXED_RECIPE_REVISION_CONTROL
    ):
        raise ProtocolGateError("pre-freeze recipe-revision controls changed")

    exact = protocol.get("exact_equivalence")
    if not isinstance(exact, Mapping):
        raise ProtocolGateError("P2 protocol lacks exact-equivalence controls")
    expected_scalars = {
        "coefficient_map": "w_tau=tau*w",
        "intercept_map": "b_tau=b",
        "penalty_map": "C_tau=tau^2*C",
        "acceptance_metric": "class_equal_nll",
        "acceptance_operator": "strict_less_than",
        "max_logit_difference": FIXED_EQUIVALENCE_TOLERANCE,
        "max_prediction_difference": FIXED_EQUIVALENCE_TOLERANCE,
        "max_loss_difference": FIXED_EQUIVALENCE_TOLERANCE,
    }
    for key, expected in expected_scalars.items():
        if not _typed_json_equal(exact.get(key), expected):
            raise ProtocolGateError(f"exact-equivalence field changed: {key}")
    if not _typed_json_equal(exact.get("synthetic_gate"), FIXED_SYNTHETIC_GATE):
        raise ProtocolGateError("fixed synthetic numerical gate changed")

    amendments = protocol.get("pre_freeze_amendments")
    expected_amendment = {
        "amendment_id": "P2-NUM-001",
        "ledger_path": AMENDMENT_LEDGER.relative_to(REPO).as_posix(),
        "trigger_audit_path": SIGNED_FAIL_AUDIT.relative_to(REPO).as_posix(),
        "trigger_audit_sha256": SIGNED_FAIL_AUDIT_SHA256,
        "scientific_results_read": False,
        "p1_outputs_read": False,
        "official_test_accessed": False,
        "scientific_execution_authorized": False,
        "freeze_authorized": False,
        "solver_change": "lbfgs_to_newton-cholesky",
        "acceptance_tolerance_changed": False,
        "acceptance_operator_changed": False,
        "scientific_hypothesis_changed": False,
        "statistical_estimands_changed": False,
        "fit_count_changed": False,
        "required_next_gate": "fresh_independent_result_blind_audit",
    }
    if not _typed_json_equal(amendments, [expected_amendment]):
        raise ProtocolGateError("pre-freeze numerical amendment binding changed")
    if sha256_file(SIGNED_FAIL_AUDIT) != SIGNED_FAIL_AUDIT_SHA256:
        raise ProtocolGateError("signed triggering FAIL audit changed")

    contract_binding = protocol.get("output_contract_binding")
    expected_contract_path = OUTPUT_CONTRACT.relative_to(REPO).as_posix()
    if not isinstance(contract_binding, Mapping):
        raise ProtocolGateError("P2 protocol lacks output-contract binding")
    if contract_binding.get("path") != expected_contract_path:
        raise ProtocolGateError("output-contract path changed")
    if contract_binding.get("contract_id") != "paper_b_p2_output_contract_v3":
        raise ProtocolGateError("output-contract identifier changed")
    if contract_binding.get("sha256") != sha256_file(OUTPUT_CONTRACT):
        raise ProtocolGateError("output-contract SHA-256 changed")
    with OUTPUT_CONTRACT.open("r", encoding="utf-8") as handle:
        contract = json.load(handle)
    if contract.get("contract_id") != contract_binding["contract_id"]:
        raise ProtocolGateError("bound output-contract contents changed")
    if contract.get("status") != (
        "CANDIDATE_NOT_FROZEN_AWAITING_FRESH_INDEPENDENT_AUDIT"
    ):
        raise ProtocolGateError("output-contract candidate status changed")
    if contract.get("scientific_execution_authorized") is not False:
        raise ProtocolGateError("output contract cannot authorize execution")
    if contract.get("freeze_authorized") is not False:
        raise ProtocolGateError("output contract cannot authorize candidate freeze")
    if not _typed_json_equal(
        contract.get("pre_freeze_amendment"), expected_amendment
    ):
        raise ProtocolGateError("output-contract amendment binding changed")
    contract_invariants = contract.get("invariants")
    if not isinstance(contract_invariants, Mapping):
        raise ProtocolGateError("output contract lacks fail-closed invariants")
    if not _typed_json_equal(
        contract_invariants.get("logistic_regression_recipe"),
        FIXED_LR_CONFIGURATION,
    ):
        raise ProtocolGateError("output-contract solver recipe changed")
    expected_contract_gate = {
        "operator": "strict_less_than",
        "tolerance": FIXED_EQUIVALENCE_TOLERANCE,
        "comparison_count": 12,
        "fit_count": 18,
        "all_fits_converged": True,
        "warning_count": 0,
        "fallback_count": 0,
        "all_values_finite": True,
    }
    if not _typed_json_equal(
        contract_invariants.get("strict_numerical_gate"),
        expected_contract_gate,
    ):
        raise ProtocolGateError("output-contract numerical gate changed")
    if (
        contract_invariants.get("fresh_independent_result_blind_audit_required")
        is not True
    ):
        raise ProtocolGateError("output contract must require fresh independent audit")
    if contract_invariants.get("candidate_is_frozen") is not False:
        raise ProtocolGateError("output contract cannot mark candidate frozen")

    exact_contract = contract.get("aggregate_outputs", {}).get(
        "exact_equivalence.json"
    )
    if not isinstance(exact_contract, Mapping):
        raise ProtocolGateError("output contract lacks exact-equivalence schema")
    expected_acceptance = {
        "operator": "strict_less_than",
        "max_abs_logit_difference": FIXED_EQUIVALENCE_TOLERANCE,
        "max_abs_prediction_difference": FIXED_EQUIVALENCE_TOLERANCE,
        "max_abs_class_equal_nll_difference": FIXED_EQUIVALENCE_TOLERANCE,
        "nonfinite_is_failure": True,
        "warning_is_failure": True,
        "nonconvergence_is_failure": True,
        "solver_fallback_or_configuration_drift_is_failure": True,
    }
    if not _typed_json_equal(
        exact_contract.get("numerical_acceptance"), expected_acceptance
    ):
        raise ProtocolGateError("output-contract numerical acceptance changed")
    expected_synthetic_contract = {
        "comparison_count": FIXED_SYNTHETIC_GATE["comparison_count"],
        "fit_count": 18,
        "temperatures": FIXED_SYNTHETIC_GATE["temperatures"],
        "C_grid": FIXED_SYNTHETIC_GATE["C_grid"],
    }
    if not _typed_json_equal(
        exact_contract.get("synthetic_gate"), expected_synthetic_contract
    ):
        raise ProtocolGateError("output-contract synthetic-gate schema changed")

    documentation = protocol.get("documentation_binding")
    expected_markdown_path = PROTOCOL_MARKDOWN.relative_to(REPO).as_posix()
    if not isinstance(documentation, Mapping):
        raise ProtocolGateError("P2 protocol lacks Markdown binding")
    if documentation.get("path") != expected_markdown_path:
        raise ProtocolGateError("protocol Markdown path changed")
    if documentation.get("sha256") != sha256_file(PROTOCOL_MARKDOWN):
        raise ProtocolGateError("protocol Markdown SHA-256 changed")
    amendment = validate_amendment_ledger()
    return {
        "logistic_regression": dict(FIXED_LR_CONFIGURATION),
        "synthetic_gate": dict(FIXED_SYNTHETIC_GATE),
        "tolerance": FIXED_EQUIVALENCE_TOLERANCE,
        "contract_id": contract_binding["contract_id"],
        "contract_sha256": contract_binding["sha256"],
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "signed_fail_audit_sha256": SIGNED_FAIL_AUDIT_SHA256,
        **amendment,
    }


def _strict_equivalence_pass(
    metrics: Iterable[float], tolerance: float
) -> bool:
    """Apply the frozen finite, strict-less-than numerical comparator."""
    try:
        threshold = float(tolerance)
        values = tuple(float(value) for value in metrics)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(
        math.isfinite(threshold)
        and threshold > 0.0
        and values
        and all(math.isfinite(value) and abs(value) < threshold for value in values)
    )


def _fit_synthetic_logistic_regression(
    values: np.ndarray,
    labels: np.ndarray,
    c_value: float,
    *,
    fit_id: str,
    stage: str,
    base_c: float,
    temperature: float,
    mapped_c: float,
) -> tuple[Any, dict[str, Any]]:
    """Fit one synthetic control and fail closed on every numerical anomaly."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    configuration = FIXED_LR_CONFIGURATION
    design = np.asarray(values, dtype=np.float64)
    target = np.asarray(labels)
    context = (
        f"stage={stage}, C={float(base_c):.17g}, "
        f"tau={float(temperature):.17g}, mapped_C={float(mapped_c):.17g}"
    )
    labels_valid = bool(
        target.ndim == 1
        and target.size > 0
        and not np.issubdtype(target.dtype, np.bool_)
        and np.issubdtype(target.dtype, np.number)
        and np.all(np.isfinite(target))
        and np.all(target == np.floor(target))
        and set(np.unique(target)) == {0, 1}
    )
    if (
        design.ndim != 2
        or target.ndim != 1
        or design.shape[0] != target.shape[0]
        or design.shape[0] == 0
        or not np.all(np.isfinite(design))
        or not labels_valid
        or not math.isfinite(float(c_value))
        or float(c_value) <= 0.0
    ):
        raise ProtocolGateError("invalid synthetic fit inputs: " + context)
    target = target.astype(np.int64, copy=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = LogisticRegression(
            solver=configuration["solver"],
            class_weight=configuration["class_weight"],
            C=float(c_value),
            fit_intercept=configuration["fit_intercept"],
            max_iter=configuration["max_iter"],
            tol=configuration["tol"],
            random_state=configuration["random_state"],
        ).fit(design, target)
    reported_configuration = model.get_params(deep=False)
    expected_reported_configuration = {
        "solver": configuration["solver"],
        "class_weight": configuration["class_weight"],
        "C": float(c_value),
        "fit_intercept": configuration["fit_intercept"],
        "max_iter": configuration["max_iter"],
        "tol": configuration["tol"],
        "random_state": configuration["random_state"],
    }
    configuration_matches = _typed_json_equal(
        {
            key: reported_configuration.get(key)
            for key in expected_reported_configuration
        },
        expected_reported_configuration,
    )
    warning_records = [
        {
            "category": item.category.__name__,
            "message": str(item.message),
        }
        for item in caught
    ]
    raw_iterations = getattr(model, "n_iter_", None)
    iterations_valid = False
    iterations: list[int] = []
    if raw_iterations is not None:
        raw_array = np.asarray(raw_iterations)
        if (
            raw_array.size > 0
            and np.issubdtype(raw_array.dtype, np.number)
            and np.all(np.isfinite(raw_array))
            and np.all(raw_array == np.floor(raw_array))
        ):
            iterations = [int(value) for value in np.ravel(raw_array)]
            iterations_valid = bool(all(value > 0 for value in iterations))
    convergence_warning_count = sum(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )
    fallback_detected = bool(
        not configuration_matches
        or any(
            "fallback" in record["message"].lower()
            or "switch" in record["message"].lower()
            for record in warning_records
        )
    )
    coefficient = np.asarray(getattr(model, "coef_", np.array([np.nan])))
    intercept = np.asarray(getattr(model, "intercept_", np.array([np.nan])))
    coefficients_finite = bool(
        coefficient.size > 0
        and intercept.size > 0
        and np.all(np.isfinite(coefficient))
        and np.all(np.isfinite(intercept))
    )
    output_logit = np.asarray(model.decision_function(design), dtype=np.float64)
    output_probability = np.asarray(
        model.predict_proba(design), dtype=np.float64
    )
    outputs_finite = bool(
        output_logit.shape == (design.shape[0],)
        and output_probability.shape == (design.shape[0], 2)
        and np.all(np.isfinite(output_logit))
        and np.all(np.isfinite(output_probability))
    )
    converged = bool(
        iterations_valid
        and max(iterations) < configuration["max_iter"]
        and convergence_warning_count == 0
    )
    diagnostic = {
        "fit_id": fit_id,
        "stage": stage,
        "base_C": float(base_c),
        "temperature": float(temperature),
        "effective_C": float(c_value),
        "mapped_C": float(mapped_c),
        "requested_solver": configuration["solver"],
        "reported_solver": getattr(model, "solver", None),
        "reported_C": reported_configuration.get("C"),
        "configuration_matches": configuration_matches,
        "class_weight": configuration["class_weight"],
        "fit_intercept": configuration["fit_intercept"],
        "max_iter": configuration["max_iter"],
        "tol": configuration["tol"],
        "random_state": configuration["random_state"],
        "iterations": iterations,
        "iterations_valid": iterations_valid,
        "converged": converged,
        "warning_count": len(warning_records),
        "convergence_warning_count": convergence_warning_count,
        "warnings": warning_records,
        "fallback_detected": fallback_detected,
        "coefficients_finite": coefficients_finite,
        "outputs_finite": outputs_finite,
    }
    failures = []
    if warning_records:
        failures.append("warning_emitted")
    if not iterations_valid:
        failures.append("missing_or_malformed_n_iter")
    elif max(iterations) >= configuration["max_iter"]:
        failures.append("iteration_limit_reached")
    if not converged:
        failures.append("nonconverged")
    if fallback_detected:
        failures.append("configuration_drift_or_solver_fallback")
    if not coefficients_finite:
        failures.append("nonfinite_coefficients_or_intercept")
    if not outputs_finite:
        failures.append("nonfinite_logits_or_probabilities")
    if failures:
        raise ProtocolGateError(
            "synthetic logistic fit failed closed ("
            + ",".join(dict.fromkeys(failures))
            + "): "
            + context
        )
    return model, diagnostic


def synthetic_solver_equivalence_check(
    *,
    random_seed: int = 2026091601,
    sample_count: int = 480,
    feature_count: int = 7,
    c_grid: Sequence[float] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
    temperatures: Sequence[float] = (0.25, 4.0),
) -> dict[str, Any]:
    """Run the fixed result-blind numerical gate on synthetic data only."""
    import sklearn

    protocol = load_protocol(DEFAULT_PROTOCOL)
    numerical = validate_numerical_protocol(protocol)
    fixed_gate = numerical["synthetic_gate"]
    c_values = tuple(float(value) for value in c_grid)
    temperature_values = tuple(float(value) for value in temperatures)
    supplied_gate = {
        "random_seed": random_seed,
        "sample_count": sample_count,
        "feature_count": feature_count,
        "temperatures": list(temperature_values),
        "C_grid": list(c_values),
        "comparison_count": len(c_values) * len(temperature_values),
        "require_all_converged": True,
        "require_zero_warnings": True,
    }
    if not _typed_json_equal(supplied_gate, fixed_gate):
        raise ProtocolGateError("synthetic numerical-gate inputs changed")

    rng = np.random.default_rng(random_seed)
    values = rng.normal(size=(sample_count, feature_count))
    generating_coefficient = rng.normal(size=feature_count)
    generating_logit = values @ generating_coefficient - 0.37
    generating_probability = stable_sigmoid(generating_logit)
    labels = (rng.uniform(size=sample_count) < generating_probability).astype(int)
    if set(np.unique(labels)) != {0, 1}:
        raise RuntimeError("synthetic design unexpectedly lacks a binary class")

    maximum_logit_difference = 0.0
    maximum_prediction_difference = 0.0
    maximum_class_equal_nll_difference = 0.0
    maximum_coefficient_map_difference = 0.0
    maximum_intercept_map_difference = 0.0
    comparison_diagnostics: list[dict[str, Any]] = []
    fit_diagnostics: list[dict[str, Any]] = []
    all_converged = True
    tolerance = numerical["tolerance"]
    for c_index, c_value in enumerate(c_values):
        reference_fit_id = f"reference_C{c_index}"
        reference, reference_fit = _fit_synthetic_logistic_regression(
            values,
            labels,
            float(c_value),
            fit_id=reference_fit_id,
            stage="reference",
            base_c=float(c_value),
            temperature=1.0,
            mapped_c=float(c_value),
        )
        fit_diagnostics.append(reference_fit)
        reference_logit = reference.decision_function(values)
        reference_probability = reference.predict_proba(values)[:, 1]
        reference_nll = class_equal_binary_cross_entropy(labels, reference_logit)
        if not (
            np.all(np.isfinite(reference_logit))
            and np.all(np.isfinite(reference_probability))
            and math.isfinite(reference_nll)
        ):
            raise ProtocolGateError(
                "nonfinite reference fit statistic: "
                f"stage=reference, C={c_value:.17g}, tau=1, mapped_C={c_value:.17g}"
            )
        for temperature_index, tau in enumerate(temperature_values):
            mapped_c = float(c_value) * float(tau) ** 2
            scaled_fit_id = f"scaled_C{c_index}_T{temperature_index}"
            scaled, scaled_fit = _fit_synthetic_logistic_regression(
                values / float(tau),
                labels,
                mapped_c,
                fit_id=scaled_fit_id,
                stage="scaled",
                base_c=float(c_value),
                temperature=float(tau),
                mapped_c=mapped_c,
            )
            fit_diagnostics.append(scaled_fit)
            scaled_logit = scaled.decision_function(values / float(tau))
            scaled_probability = scaled.predict_proba(values / float(tau))[:, 1]
            scaled_nll = class_equal_binary_cross_entropy(labels, scaled_logit)
            logit_difference = float(
                np.max(np.abs(reference_logit - scaled_logit))
            )
            prediction_difference = float(
                np.max(np.abs(reference_probability - scaled_probability))
            )
            nll_difference = abs(reference_nll - scaled_nll)
            coefficient_map_difference = float(
                np.max(
                    np.abs(
                        scaled.coef_[0]
                        - float(tau) * reference.coef_[0]
                    )
                )
            )
            intercept_map_difference = float(
                np.max(np.abs(scaled.intercept_ - reference.intercept_))
            )
            metrics = (
                logit_difference,
                prediction_difference,
                nll_difference,
            )
            if not all(math.isfinite(value) for value in metrics):
                raise ProtocolGateError(
                    "nonfinite synthetic equivalence statistic: "
                    f"stage=comparison, C={c_value:.17g}, "
                    f"tau={tau:.17g}, mapped_C={mapped_c:.17g}"
                )
            maximum_logit_difference = max(
                maximum_logit_difference,
                logit_difference,
            )
            maximum_prediction_difference = max(
                maximum_prediction_difference,
                prediction_difference,
            )
            maximum_class_equal_nll_difference = max(
                maximum_class_equal_nll_difference,
                nll_difference,
            )
            maximum_coefficient_map_difference = max(
                maximum_coefficient_map_difference,
                coefficient_map_difference,
            )
            maximum_intercept_map_difference = max(
                maximum_intercept_map_difference,
                intercept_map_difference,
            )
            comparison_converged = bool(
                reference_fit["converged"] and scaled_fit["converged"]
            )
            comparison_passed = bool(
                comparison_converged
                and reference_fit["warning_count"] == 0
                and scaled_fit["warning_count"] == 0
                and _strict_equivalence_pass(metrics, tolerance)
            )
            all_converged = all_converged and comparison_converged
            comparison_diagnostics.append(
                {
                    "C": float(c_value),
                    "temperature": float(tau),
                    "mapped_C": mapped_c,
                    "solver": FIXED_LR_CONFIGURATION["solver"],
                    "reference_fit_id": reference_fit_id,
                    "scaled_fit_id": scaled_fit_id,
                    "max_abs_coefficient_map_difference": (
                        coefficient_map_difference
                    ),
                    "max_abs_intercept_map_difference": intercept_map_difference,
                    "max_abs_logit_difference": logit_difference,
                    "max_abs_prediction_difference": prediction_difference,
                    "max_abs_class_equal_nll_difference": nll_difference,
                    "acceptance_tolerance": tolerance,
                    "acceptance_operator": "strict_less_than",
                    "pass": comparison_passed,
                }
            )
    comparison_count = len(comparison_diagnostics)
    warning_count = sum(item["warning_count"] for item in fit_diagnostics)
    fallback_count = sum(item["fallback_detected"] for item in fit_diagnostics)
    all_finite = all(
        item["coefficients_finite"] and item["outputs_finite"]
        for item in fit_diagnostics
    )
    expected_fit_count = len(c_values) * (1 + len(temperature_values))
    passed = bool(
        comparison_count == fixed_gate["comparison_count"]
        and len(fit_diagnostics) == expected_fit_count
        and all_converged
        and warning_count == 0
        and fallback_count == 0
        and all_finite
        and all(item["pass"] for item in comparison_diagnostics)
    )
    return {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": numerical["protocol_sha256"],
        "output_contract_id": numerical["contract_id"],
        "output_contract_sha256": numerical["contract_sha256"],
        "amendment_id": numerical["amendment_id"],
        "amendment_ledger_sha256": numerical["ledger_sha256"],
        "signed_fail_audit_sha256": numerical["signed_fail_audit_sha256"],
        "solver_configuration": numerical["logistic_regression"],
        "sklearn_version": sklearn.__version__,
        "random_seed": random_seed,
        "sample_count": sample_count,
        "feature_count": feature_count,
        "positive_count": int(labels.sum()),
        "comparison_count": comparison_count,
        "fit_count": len(fit_diagnostics),
        "expected_fit_count": expected_fit_count,
        "acceptance_thresholds": {
            "max_abs_logit_difference": tolerance,
            "max_abs_prediction_difference": tolerance,
            "max_abs_class_equal_nll_difference": tolerance,
            "operator": "strict_less_than",
        },
        "convergence_summary": {
            "all_converged": all_converged,
            "warning_count": warning_count,
            "require_zero_warnings": True,
            "fallback_count": fallback_count,
            "all_finite": all_finite,
        },
        "fit_diagnostics": fit_diagnostics,
        "comparison_diagnostics": comparison_diagnostics,
        "max_abs_coefficient_map_difference": maximum_coefficient_map_difference,
        "max_abs_intercept_map_difference": maximum_intercept_map_difference,
        "max_abs_logit_difference": maximum_logit_difference,
        "max_abs_prediction_difference": maximum_prediction_difference,
        "max_abs_class_equal_nll_difference": maximum_class_equal_nll_difference,
        "pass": passed,
    }


@dataclass(frozen=True)
class DiagonalAffineProxy:
    slope: np.ndarray
    intercept: np.ndarray

    @classmethod
    def fit(cls, soft_training: np.ndarray, logodds_training: np.ndarray) -> "DiagonalAffineProxy":
        soft = np.asarray(soft_training, dtype=np.float64)
        logodds = np.asarray(logodds_training, dtype=np.float64)
        if soft.shape != logodds.shape:
            raise ValueError("CBIS affine proxy requires matching binary coordinates")
        centered = soft - soft.mean(axis=0)
        denominator = np.sum(centered * centered, axis=0)
        numerator = np.sum(centered * (logodds - logodds.mean(axis=0)), axis=0)
        slope = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-15,
        )
        intercept = logodds.mean(axis=0) - slope * soft.mean(axis=0)
        return cls(slope=slope, intercept=intercept)

    def predict(self, soft: np.ndarray) -> np.ndarray:
        return np.asarray(soft, dtype=np.float64) * self.slope + self.intercept


def affine_bound_check(
    exact_logodds: np.ndarray,
    affine_logodds: np.ndarray,
    coefficient: np.ndarray,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    residual = np.asarray(exact_logodds) - np.asarray(affine_logodds)
    coefficient = np.asarray(coefficient, dtype=np.float64)
    logit_difference = np.abs(residual @ coefficient)
    bound = np.linalg.norm(coefficient) * np.linalg.norm(residual, axis=1)
    return {
        "holds": bool(np.all(logit_difference <= bound + tolerance)),
        "max_violation": float(np.max(logit_difference - bound)),
        "logit_difference": logit_difference,
        "bound": bound,
    }


def mlp_specification(input_dimension: int, protocol: Mapping[str, Any]) -> dict[str, Any]:
    spec = protocol["mlp"]
    return {
        "layers": [input_dimension, 32, 1],
        "activation": "GELU",
        "dropout": 0.0,
        "optimizer": "Adam",
        "learning_rate": spec["learning_rate"],
        "epochs": spec["epochs"],
        "early_stopping": False,
        "batching": "full_batch",
        "selection_seed": spec["selection_seed"],
        "refit_seeds": list(spec["refit_seeds"]),
        "lambda_grid": list(spec["lambda_grid"]),
        "penalized_parameters": "weights_only",
    }


def build_blinded_fit_plan(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Return counts and arm identities only; it never opens prediction arrays."""
    core_arms = [
        {"tau": tau, "representation": representation, "head": head}
        for tau in protocol["capacity_matrix"]["main_temperatures"]
        for representation in protocol["capacity_matrix"]["representations"]
        for head in protocol["capacity_matrix"]["heads"]
    ]
    return {
        "status": "BLINDED_PLAN_ONLY",
        "development_units": len(EXPECTED_UNITS),
        "core_arms_per_unit": len(core_arms),
        "core_arms": core_arms,
        "pretrigger_fit_counts": dict(protocol["cost"]),
        "tau4_trigger": protocol["capacity_matrix"]["tau4_full_matrix_trigger"],
        "official_test_jobs": 0,
    }


def _shape_tuple(array: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in array.shape)


def _require_fixed_non_symlink_path(path: Path, expected: Path, label: str) -> None:
    """Reject path substitution and symlinks before opening a frozen input."""
    if path != expected:
        raise ProtocolGateError(f"{label} must use the fixed repository path")
    repository = REPO.resolve(strict=True)
    try:
        path.relative_to(REPO)
    except ValueError as error:
        raise ProtocolGateError(f"{label} is outside the repository") from error
    current = path
    while current != REPO:
        if current.is_symlink():
            raise ProtocolGateError(f"{label} contains a symlink: {current}")
        current = current.parent
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(repository)
    except ValueError as error:
        raise ProtocolGateError(f"{label} resolves outside the repository") from error


def _load_frozen_source_records() -> dict[str, dict[str, Any]]:
    _require_fixed_non_symlink_path(P0_MANIFEST, P0_MANIFEST, "P0 manifest")
    if sha256_file(P0_MANIFEST) != P0_MANIFEST_SHA256:
        raise ProtocolGateError("P0 manifest SHA-256 does not match the frozen value")
    manifest = load_protocol(P0_MANIFEST)
    if (
        manifest.get("frozen_result_aggregate_sha256")
        != P0_FROZEN_RESULT_AGGREGATE_SHA256
    ):
        raise ProtocolGateError("P0 frozen-result aggregate SHA-256 changed")
    records = manifest.get("frozen_result_files")
    if not isinstance(records, list):
        raise ProtocolGateError("P0 manifest lacks frozen_result_files")
    expected_count = manifest.get("frozen_result_file_count")
    if expected_count != len(records):
        raise ProtocolGateError("P0 frozen-result file count changed")
    frozen_records: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"bytes", "path", "sha256"}:
            raise ProtocolGateError("P0 frozen-result record schema changed")
        relative = str(record["path"])
        if relative in frozen_records:
            raise ProtocolGateError(f"duplicate P0 manifest path: {relative}")
        byte_count = record["bytes"]
        sha256 = record["sha256"]
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
            or not isinstance(sha256, str)
            or len(sha256) != 64
        ):
            raise ProtocolGateError(f"invalid P0 frozen-result record: {relative}")
        frozen_records[relative] = {"bytes": byte_count, "sha256": sha256}
    return frozen_records


def _domain_source_aggregate_sha256(
    domain: str, frozen_records: Mapping[str, Mapping[str, Any]]
) -> str:
    """Hash one domain's fixed ordered source-file inventory from the P0 manifest."""
    ordered_units_by_domain = dict(DOMAIN_UNITS)
    if domain not in ordered_units_by_domain:
        raise ProtocolGateError(f"unknown domain for source aggregate: {domain}")
    digest = hashlib.sha256()
    recipe_id = DOMAIN_SOURCE_AGGREGATE_RECIPE["id"]
    digest.update(f"{recipe_id}\t{domain}\n".encode("utf-8"))
    for unit in ordered_units_by_domain[domain]:
        for filename in REQUIRED_FILES:
            relative = (DEFAULT_PANELS / unit / filename).relative_to(REPO).as_posix()
            record = frozen_records.get(relative)
            if not isinstance(record, Mapping):
                raise ProtocolGateError(
                    f"domain source aggregate lacks frozen record: {relative}"
                )
            byte_count = record.get("bytes")
            sha256 = record.get("sha256")
            if (
                isinstance(byte_count, bool)
                or not isinstance(byte_count, int)
                or byte_count < 0
                or not isinstance(sha256, str)
                or len(sha256) != 64
                or sha256.lower() != sha256
                or any(character not in "0123456789abcdef" for character in sha256)
            ):
                raise ProtocolGateError(
                    f"invalid frozen record for domain source aggregate: {relative}"
                )
            digest.update(
                (
                    f"{domain}\t{unit}\t{relative}\t{byte_count}\t{sha256}\n"
                ).encode("utf-8")
            )
    return digest.hexdigest()


def _validate_protocol_source_contract(
    protocol: Mapping[str, Any],
    frozen_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    """Bind the draft protocol's source claims to a documented recomputation."""
    source = protocol.get("source")
    if not isinstance(source, Mapping):
        raise ProtocolGateError("P2 protocol lacks a source contract")
    expected_scalars = {
        "fixed_repository_relative_root": DEFAULT_PANELS.relative_to(REPO).as_posix(),
        "receipt_schema_version": (
            "paper_b_p2_content_addressed_source_receipt_v2"
        ),
        "protocol_path": SOURCE_PROTOCOL.relative_to(REPO).as_posix(),
        "protocol_sha256": SOURCE_PROTOCOL_SHA256,
        "p0_manifest_path": P0_MANIFEST.relative_to(REPO).as_posix(),
        "p0_manifest_sha256": P0_MANIFEST_SHA256,
        "p0_frozen_result_aggregate_sha256": (
            P0_FROZEN_RESULT_AGGREGATE_SHA256
        ),
    }
    for key, expected in expected_scalars.items():
        if source.get(key) != expected:
            raise ProtocolGateError(f"P2 protocol source field changed: {key}")
    if source.get("domain_source_aggregate_recipe") != DOMAIN_SOURCE_AGGREGATE_RECIPE:
        raise ProtocolGateError("P2 protocol domain-source aggregate recipe changed")
    if source.get("ordered_units") != {
        domain: list(units) for domain, units in DOMAIN_UNITS
    }:
        raise ProtocolGateError("P2 protocol ordered source units changed")
    if source.get("exact_per_unit_files") != list(REQUIRED_FILES):
        raise ProtocolGateError("P2 protocol exact source-file order changed")
    if source.get("exact_archive_key_order") != list(REQUIRED_ARRAYS):
        raise ProtocolGateError("P2 protocol archive-key order changed")

    computed = {
        domain: _domain_source_aggregate_sha256(domain, frozen_records)
        for domain, _ in DOMAIN_UNITS
    }
    if computed != SOURCE_DOMAIN_AGGREGATES:
        raise ProtocolGateError(
            "code-bound domain source aggregates do not match P0 recomputation"
        )
    protocol_aggregates = {
        "cbis": source.get("cbis_domain_source_aggregate_sha256"),
        "derm": source.get("derm_domain_source_aggregate_sha256"),
    }
    if protocol_aggregates != computed:
        raise ProtocolGateError(
            "P2 protocol domain source aggregates do not match P0 recomputation"
        )
    return computed


def _regular_unlinked_file_record(
    path: Path,
    expected: Path,
    label: str,
    frozen_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify one immutable input without following links or accepting hard links."""
    _require_fixed_non_symlink_path(path, expected, label)
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ProtocolGateError(f"{label} is not a regular file")
    if metadata.st_nlink != 1:
        raise ProtocolGateError(f"{label} has a non-unique hard-link count")
    relative = path.relative_to(REPO).as_posix()
    expected_record = frozen_records.get(relative)
    if expected_record is None:
        raise ProtocolGateError(f"{label} is absent from the frozen P0 manifest")
    observed_sha256 = sha256_file(path)
    observed_bytes = metadata.st_size
    if observed_bytes != expected_record["bytes"]:
        raise ProtocolGateError(f"{label} byte count does not match the frozen value")
    if observed_sha256 != expected_record["sha256"]:
        raise ProtocolGateError(f"{label} SHA-256 does not match the frozen value")
    return {
        "path": relative,
        "bytes": observed_bytes,
        "sha256": observed_sha256,
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "link_count": int(metadata.st_nlink),
    }


def _array_content_record(array: np.ndarray) -> dict[str, Any]:
    array = np.asarray(array)
    if array.dtype.hasobject:
        raise ProtocolGateError("object arrays are prohibited in frozen panels")
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": array.dtype.str,
        "shape": list(_shape_tuple(array)),
        "content_sha256": hashlib.sha256(
            contiguous.tobytes(order="C")
        ).hexdigest(),
    }


def _case_mapping_sha256(
    identifiers: np.ndarray, groups: np.ndarray, labels: np.ndarray
) -> str:
    identifiers = np.asarray(identifiers)
    groups = np.asarray(groups)
    labels = np.asarray(labels)
    if identifiers.ndim != 1 or groups.ndim != 1 or labels.ndim != 1:
        raise ProtocolGateError("case mapping arrays must be one-dimensional")
    if not (identifiers.shape == groups.shape == labels.shape):
        raise ProtocolGateError("case mapping arrays must have equal lengths")
    digest = hashlib.sha256()
    for identifier, group, label in zip(identifiers, groups, labels, strict=True):
        record = json.dumps(
            [str(identifier), str(group), int(label)],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(record).to_bytes(8, byteorder="big", signed=False))
        digest.update(record)
    return digest.hexdigest()


def _fixed_unit_directories(panel_root: Path) -> tuple[set[str], list[str]]:
    """Enumerate only the fixed development root and return its directories."""
    directories: set[str] = set()
    non_directory_entries: list[str] = []
    with os.scandir(panel_root) as iterator:
        for entry in iterator:
            if entry.is_symlink():
                raise ProtocolGateError(
                    f"development panel root contains a symlink: {entry.name}"
                )
            if entry.is_dir(follow_symlinks=False):
                directories.add(entry.name)
            else:
                non_directory_entries.append(entry.name)
    return directories, sorted(non_directory_entries)


def _source_receipt_required_fields() -> tuple[str, ...]:
    """Load the exact top-level SOURCE_RECEIPT schema from the fixed contract."""
    _require_fixed_non_symlink_path(
        OUTPUT_CONTRACT, OUTPUT_CONTRACT, "P2 output contract"
    )
    contract = load_protocol(OUTPUT_CONTRACT)
    try:
        fields = contract["aggregate_outputs"]["SOURCE_RECEIPT.json"][
            "required_fields"
        ]
    except (KeyError, TypeError) as error:
        raise ProtocolGateError(
            "P2 output contract lacks the SOURCE_RECEIPT required-fields schema"
        ) from error
    if (
        not isinstance(fields, list)
        or not fields
        or any(not isinstance(field, str) or not field for field in fields)
        or len(fields) != len(set(fields))
    ):
        raise ProtocolGateError("invalid SOURCE_RECEIPT required-fields schema")
    return tuple(fields)


def _content_addressed_source_components(
    units: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Construct deterministic ordered receipts without computing any outcome."""
    if set(units) != set(EXPECTED_UNITS):
        raise ProtocolGateError("source receipt requires exactly the 30 frozen units")

    ordered_identities: list[dict[str, Any]] = []
    file_receipts: list[dict[str, Any]] = []
    archive_array_receipts: list[dict[str, Any]] = []
    mapping_receipts: list[dict[str, str]] = []

    for domain, ordered_units in DOMAIN_UNITS:
        domain_identities: list[dict[str, str]] = []
        for unit in ordered_units:
            record = units[unit]
            if record.get("domain") != domain:
                raise ProtocolGateError(f"{unit}: domain identity changed")

            raw_files = record.get("files")
            if not isinstance(raw_files, list) or len(raw_files) != len(REQUIRED_FILES):
                raise ProtocolGateError(f"{unit}: source receipt requires exactly six files")
            receipt_files: list[dict[str, Any]] = []
            for expected_name, raw_file in zip(
                REQUIRED_FILES, raw_files, strict=True
            ):
                if not isinstance(raw_file, Mapping):
                    raise ProtocolGateError(f"{unit}/{expected_name}: invalid file record")
                expected_path = (
                    DEFAULT_PANELS / unit / expected_name
                ).relative_to(REPO).as_posix()
                if raw_file.get("path") != expected_path:
                    raise ProtocolGateError(f"{unit}/{expected_name}: file order/path changed")
                size = raw_file.get("bytes")
                sha256 = raw_file.get("sha256")
                if (
                    isinstance(size, bool)
                    or not isinstance(size, int)
                    or size < 0
                    or not isinstance(sha256, str)
                    or len(sha256) != 64
                ):
                    raise ProtocolGateError(f"{unit}/{expected_name}: invalid receipt")
                receipt_files.append(
                    {
                        "name": expected_name,
                        "path": expected_path,
                        "size": size,
                        "sha256": sha256,
                    }
                )
            file_group = {
                "domain": domain,
                "unit_identity": unit,
                "files": receipt_files,
            }
            file_receipts.append(file_group)

            raw_arrays = record.get("arrays")
            if not isinstance(raw_arrays, Mapping) or tuple(raw_arrays) != REQUIRED_ARRAYS:
                raise ProtocolGateError(
                    f"{unit}: archive array order/whitelist changed"
                )
            receipt_arrays: list[dict[str, Any]] = []
            for name in REQUIRED_ARRAYS:
                raw_array = raw_arrays[name]
                if not isinstance(raw_array, Mapping):
                    raise ProtocolGateError(f"{unit}/{name}: invalid array record")
                dtype = raw_array.get("dtype")
                shape = raw_array.get("shape")
                content_sha256 = raw_array.get("content_sha256")
                if (
                    not isinstance(dtype, str)
                    or not isinstance(shape, list)
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                        for value in shape
                    )
                    or not isinstance(content_sha256, str)
                    or len(content_sha256) != 64
                ):
                    raise ProtocolGateError(f"{unit}/{name}: invalid array receipt")
                receipt_arrays.append(
                    {
                        "name": name,
                        "dtype": dtype,
                        "shape": shape,
                        "content_sha256": content_sha256,
                    }
                )
            array_group = {
                "domain": domain,
                "unit_identity": unit,
                "archive_path": receipt_files[0]["path"],
                "arrays": receipt_arrays,
            }
            archive_array_receipts.append(array_group)

            mapping_sha256 = record.get("case_group_label_mapping_sha256")
            if not isinstance(mapping_sha256, str) or len(mapping_sha256) != 64:
                raise ProtocolGateError(f"{unit}: invalid case/group/label digest")
            mapping_receipts.append(
                {
                    "domain": domain,
                    "unit_identity": unit,
                    "case_group_label_mapping_sha256": mapping_sha256,
                }
            )
            unit_digest = canonical_json_sha256(
                {
                    "domain": domain,
                    "unit_identity": unit,
                    "files": receipt_files,
                    "archive": array_group,
                    "case_group_label_mapping_sha256": mapping_sha256,
                }
            )
            domain_identities.append(
                {"unit_identity": unit, "unit_digest": unit_digest}
            )
        ordered_identities.append(
            {"domain": domain, "ordered_units": domain_identities}
        )

    mapping_digest = canonical_json_sha256(
        {"ordered_case_group_label_mappings": mapping_receipts}
    )
    inventory_digest = canonical_json_sha256(
        {
            "ordered_domain_unit_identities_and_digests": ordered_identities,
            "file_receipts": file_receipts,
            "archive_array_receipts": archive_array_receipts,
            "case_group_label_mapping_digest": mapping_digest,
        }
    )
    return {
        "ordered_domain_unit_identities_and_digests": ordered_identities,
        "file_receipts": file_receipts,
        "archive_array_receipts": archive_array_receipts,
        "case_group_label_mapping_digest": mapping_digest,
        "ordered_inventory_digest": inventory_digest,
    }


def build_source_receipt(
    units: Mapping[str, Mapping[str, Any]],
    source_protocol_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact candidate SOURCE_RECEIPT artifact defined by the contract."""
    components = _content_addressed_source_components(units)
    producing_code = Path(__file__).resolve()
    _require_fixed_non_symlink_path(
        producing_code,
        REPO / "scripts/run_neurocomputing_p2_20260915.py",
        "P2 producing code",
    )
    source_protocol_path = SOURCE_PROTOCOL.relative_to(REPO).as_posix()
    if source_protocol_record.get("path") != source_protocol_path:
        raise ProtocolGateError("source protocol receipt path changed")

    receipt = {
        "schema_version": "paper_b_p2_content_addressed_source_receipt_v2",
        "status": "CANDIDATE_DEV_ONLY_AWAITING_INDEPENDENT_FREEZE",
        "producing_code_path": producing_code.relative_to(REPO).as_posix(),
        "producing_code_sha256": sha256_file(producing_code),
        "source_protocol_path": source_protocol_path,
        "source_protocol_size": source_protocol_record.get("bytes"),
        "source_protocol_sha256": source_protocol_record.get("sha256"),
        **components,
        "official_test_accessed": False,
        "official_test_path_represented": False,
        "official_test_discovery_or_listing_performed": False,
    }
    validate_source_receipt(receipt, units, source_protocol_record)
    return receipt


def validate_source_receipt(
    receipt: Mapping[str, Any],
    units: Mapping[str, Mapping[str, Any]],
    source_protocol_record: Mapping[str, Any],
) -> None:
    """Fail closed on schema, ordering, or content-addressing drift."""
    required_fields = _source_receipt_required_fields()
    if tuple(receipt) != required_fields or set(receipt) != set(required_fields):
        raise ProtocolGateError("SOURCE_RECEIPT top-level schema/order changed")
    if receipt.get("schema_version") != "paper_b_p2_content_addressed_source_receipt_v2":
        raise ProtocolGateError("SOURCE_RECEIPT schema version changed")
    if receipt.get("status") != "CANDIDATE_DEV_ONLY_AWAITING_INDEPENDENT_FREEZE":
        raise ProtocolGateError("SOURCE_RECEIPT must remain an unfrozen candidate")

    producing_code = REPO / str(receipt.get("producing_code_path"))
    if producing_code != REPO / "scripts/run_neurocomputing_p2_20260915.py":
        raise ProtocolGateError("SOURCE_RECEIPT producing-code path changed")
    _require_fixed_non_symlink_path(
        producing_code,
        REPO / "scripts/run_neurocomputing_p2_20260915.py",
        "P2 producing code",
    )
    if receipt.get("producing_code_sha256") != sha256_file(producing_code):
        raise ProtocolGateError("SOURCE_RECEIPT producing-code SHA-256 changed")
    if receipt.get("source_protocol_path") != SOURCE_PROTOCOL.relative_to(REPO).as_posix():
        raise ProtocolGateError("SOURCE_RECEIPT source-protocol path changed")
    if receipt.get("source_protocol_size") != source_protocol_record.get("bytes"):
        raise ProtocolGateError("SOURCE_RECEIPT source-protocol size changed")
    if receipt.get("source_protocol_sha256") != source_protocol_record.get("sha256"):
        raise ProtocolGateError("SOURCE_RECEIPT source-protocol SHA-256 changed")

    expected = _content_addressed_source_components(units)
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise ProtocolGateError(f"SOURCE_RECEIPT {key} changed")
    for flag in (
        "official_test_accessed",
        "official_test_path_represented",
        "official_test_discovery_or_listing_performed",
    ):
        if receipt.get(flag) is not False:
            raise ProtocolGateError(f"SOURCE_RECEIPT {flag} must be false")


def persist_source_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Atomically publish only the fixed candidate receipt path."""
    output = SOURCE_RECEIPT_OUTPUT
    parent = output.parent
    _require_fixed_non_symlink_path(parent, parent, "SOURCE_RECEIPT parent")
    if output.is_symlink():
        raise ProtocolGateError("SOURCE_RECEIPT output may not be a symlink")
    if output.exists():
        metadata = output.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ProtocolGateError("existing SOURCE_RECEIPT output is not a direct file")

    serialized = (
        json.dumps(dict(receipt), indent=2, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, output)
        temporary_path = None
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    metadata = output.lstat()
    return {
        "path": output.relative_to(REPO).as_posix(),
        "size": int(metadata.st_size),
        "sha256": sha256_file(output),
    }


def audit_panel_structure(panel_root: Path = DEFAULT_PANELS) -> dict[str, Any]:
    """Audit only the content-addressed 30-unit development source."""
    _require_fixed_non_symlink_path(panel_root, DEFAULT_PANELS, "development panel root")
    frozen_records = _load_frozen_source_records()
    units: dict[str, Any] = {}
    problems: list[str] = []
    computed_domain_source_aggregates: dict[str, str] | None = None
    try:
        _require_fixed_non_symlink_path(
            DEFAULT_PROTOCOL, DEFAULT_PROTOCOL, "P2 protocol"
        )
        p2_protocol = load_protocol(DEFAULT_PROTOCOL)
        computed_domain_source_aggregates = _validate_protocol_source_contract(
            p2_protocol, frozen_records
        )
    except (OSError, TypeError, ValueError, ProtocolGateError) as error:
        problems.append(str(error))
    observed_inodes: set[tuple[int, int]] = set()
    domain_mapping_digests: dict[str, set[str]] = {"cbis": set(), "derm": set()}
    try:
        actual_unit_directories, ignored_root_files = _fixed_unit_directories(panel_root)
    except (OSError, ProtocolGateError) as error:
        actual_unit_directories = set()
        ignored_root_files = []
        problems.append(str(error))
    expected_unit_directories = set(EXPECTED_UNITS)
    missing_unit_directories = sorted(expected_unit_directories - actual_unit_directories)
    unexpected_unit_directories = sorted(actual_unit_directories - expected_unit_directories)
    if missing_unit_directories:
        problems.append(f"missing unit directories: {missing_unit_directories}")
    if unexpected_unit_directories:
        problems.append(f"unexpected unit directories: {unexpected_unit_directories}")

    source_protocol_record: dict[str, Any] | None = None
    try:
        source_protocol_record = _regular_unlinked_file_record(
            SOURCE_PROTOCOL,
            DEFAULT_PANELS / "protocol.json",
            "source protocol",
            frozen_records,
        )
        if source_protocol_record["sha256"] != SOURCE_PROTOCOL_SHA256:
            raise ProtocolGateError("source protocol SHA-256 changed")
        observed_inodes.add(
            (source_protocol_record["device"], source_protocol_record["inode"])
        )
    except (OSError, ProtocolGateError) as error:
        problems.append(str(error))

    for unit in EXPECTED_UNITS:
        unit_dir = panel_root / unit
        if unit not in actual_unit_directories:
            continue
        try:
            _require_fixed_non_symlink_path(
                unit_dir, DEFAULT_PANELS / unit, f"unit {unit}"
            )
            with os.scandir(unit_dir) as iterator:
                entries = {entry.name: entry for entry in iterator}
            entry_names = set(entries)
            missing = sorted(set(REQUIRED_FILES) - entry_names)
            extra = sorted(entry_names - set(REQUIRED_FILES))
            if missing or extra:
                raise ProtocolGateError(
                    f"{unit}: exact file whitelist failed; missing={missing}, extra={extra}"
                )
            file_records: list[dict[str, Any]] = []
            for name in REQUIRED_FILES:
                entry = entries[name]
                if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                    raise ProtocolGateError(f"{unit}/{name} is not a regular direct file")
                input_record = _regular_unlinked_file_record(
                    unit_dir / name,
                    DEFAULT_PANELS / unit / name,
                    f"{unit}/{name}",
                    frozen_records,
                )
                inode_key = (input_record["device"], input_record["inode"])
                if inode_key in observed_inodes:
                    raise ProtocolGateError(
                        f"{unit}/{name} reuses an input inode already audited"
                    )
                observed_inodes.add(inode_key)
                file_records.append(input_record)
        except (OSError, ProtocolGateError) as error:
            problems.append(str(error))
            continue
        try:
            with np.load(unit_dir / "panel.npz", allow_pickle=False) as panel:
                if tuple(panel.files) != REQUIRED_ARRAYS:
                    raise ProtocolGateError(
                        f"{unit}: archive key order/whitelist differs from the frozen schema"
                    )
                arrays = {name: np.asarray(panel[name]) for name in REQUIRED_ARRAYS}
            array_records = {
                name: _array_content_record(arrays[name]) for name in REQUIRED_ARRAYS
            }
            shapes = {name: _shape_tuple(arrays[name]) for name in REQUIRED_ARRAYS}
            case_count = shapes["y"][0]
            train_count = shapes["t"][0]
            eval_count = shapes["e"][0]
            if train_count + eval_count != case_count:
                raise ProtocolGateError(f"{unit}: outer counts do not sum to all cases")
            if not (
                shapes["ids"] == (case_count,)
                and shapes["groups"] == (case_count,)
                and shapes["gold"][0] == case_count
            ):
                raise ProtocolGateError(f"{unit}: case metadata dimensions disagree")
            if shapes["raw"][0] != train_count or shapes["val"][0] != eval_count:
                raise ProtocolGateError(f"{unit}: outer score rows do not match split sizes")
            if shapes["raw"][1] != shapes["val"][1]:
                raise ProtocolGateError(f"{unit}: outer score dimensions disagree")
            if unit.startswith("cbis_"):
                if arrays["widths"].size != 0:
                    raise ProtocolGateError(f"{unit}: CBIS widths must be empty")
                domain = "cbis"
            else:
                validate_block_widths(arrays["raw"], arrays["widths"])
                domain = "derm"
            for fold in range(4):
                if shapes[f"x{fold}"][0] != shapes[f"f{fold}"][0]:
                    raise ProtocolGateError(
                        f"{unit}: inner train rows differ for fold {fold}"
                    )
                if shapes[f"z{fold}"][0] != shapes[f"h{fold}"][0]:
                    raise ProtocolGateError(
                        f"{unit}: inner validation rows differ for fold {fold}"
                    )
                if not (
                    shapes[f"x{fold}"][1]
                    == shapes[f"z{fold}"][1]
                    == shapes["raw"][1]
                ):
                    raise ProtocolGateError(
                        f"{unit}: inner score dimensions differ for fold {fold}"
                    )
            mapping_digest = _case_mapping_sha256(
                arrays["ids"], arrays["groups"], arrays["y"]
            )
            domain_mapping_digests[domain].add(mapping_digest)
            file_manifest = {
                "unit": unit,
                "files": [
                    {key: value for key, value in record.items() if key not in {"device", "inode"}}
                    for record in file_records
                ],
            }
            units[unit] = {
                "domain": domain,
                "case_count": case_count,
                "outer_train_count": train_count,
                "outer_eval_count": eval_count,
                "score_dimension": shapes["raw"][1],
                "inner_fold_count": 4,
                "panel_sha256": sha256_file(unit_dir / "panel.npz"),
                "unit_file_manifest_sha256": canonical_json_sha256(file_manifest),
                "case_group_label_mapping_sha256": mapping_digest,
                "files": file_manifest["files"],
                "arrays": array_records,
            }
        except (OSError, ValueError, ProtocolGateError) as error:
            problems.append(str(error))

    for domain, digests in domain_mapping_digests.items():
        if len(digests) != 1:
            problems.append(
                f"{domain}: case/group/label mapping is not stable across repetitions/folds"
            )

    source_receipt: dict[str, Any] | None = None
    if not problems and len(units) == len(EXPECTED_UNITS):
        if source_protocol_record is None:
            problems.append("source protocol receipt was not established")
        else:
            try:
                source_receipt = build_source_receipt(units, source_protocol_record)
            except (OSError, TypeError, ValueError, ProtocolGateError) as error:
                problems.append(str(error))

    passed = (
        not problems
        and len(units) == len(EXPECTED_UNITS)
        and source_receipt is not None
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "fixed_expected_unit_count": len(EXPECTED_UNITS),
        "audited_unit_count": len(units),
        "units": units,
        "problems": problems,
        "source_contract": source_receipt,
        "source_audit_invariants": {
            "protocol_source_contract_verified": (
                computed_domain_source_aggregates is not None
            ),
            "domain_source_aggregate_recipe": dict(
                DOMAIN_SOURCE_AGGREGATE_RECIPE
            ),
            "computed_domain_source_aggregate_sha256": (
                dict(computed_domain_source_aggregates)
                if computed_domain_source_aggregates is not None
                else None
            ),
            "exact_unit_directory_whitelist_verified": not missing_unit_directories
            and not unexpected_unit_directories,
            "exact_per_unit_file_whitelist_verified": not any(
                "exact file whitelist failed" in problem for problem in problems
            ),
            "archive_array_whitelist_and_content_digests_recorded": len(units)
            == len(EXPECTED_UNITS),
            "case_group_label_stability_verified": all(
                len(digests) == 1 for digests in domain_mapping_digests.values()
            ),
            "unconsumed_root_regular_files": ignored_root_files,
            "source_protocol_record": (
                {
                    key: value
                    for key, value in source_protocol_record.items()
                    if key not in {"device", "inode"}
                }
                if source_protocol_record is not None
                else None
            ),
            "arbitrary_source_selector_accepted": False,
            "symlinks_accepted": False,
            "hardlinks_accepted": False,
        },
        "outcomes_or_effect_directions_computed": False,
        "official_test_accessed": False,
        "official_test_path_represented": False,
        "official_test_discovery_or_listing_performed": False,
    }


def run_structure(*, persist_candidate_receipt: bool = False) -> dict[str, Any]:
    _require_fixed_non_symlink_path(DEFAULT_PROTOCOL, DEFAULT_PROTOCOL, "P2 protocol")
    protocol = load_protocol(DEFAULT_PROTOCOL)
    numerical_gate = synthetic_solver_equivalence_check()
    if numerical_gate["pass"] is not True:
        raise ProtocolGateError("fixed synthetic numerical-equivalence gate failed")
    audit = audit_panel_structure(DEFAULT_PANELS)
    receipt_artifact: dict[str, Any] | None = None
    if persist_candidate_receipt:
        if audit["status"] != "PASS" or audit["source_contract"] is None:
            raise ProtocolGateError(
                "candidate SOURCE_RECEIPT publication requires a passing source audit"
            )
        receipt_artifact = persist_source_receipt(audit["source_contract"])
    return {
        "mode": "structure",
        "protocol_status": protocol["status"],
        "protocol_file_sha256": sha256_file(DEFAULT_PROTOCOL),
        "protocol_canonical_sha256": canonical_json_sha256(protocol),
        "numerical_equivalence_gate": numerical_gate,
        "panel_audit": audit,
        "candidate_source_receipt_artifact": receipt_artifact,
        "blinded_fit_plan": build_blinded_fit_plan(protocol),
        "scientific_models_fitted": 0,
        "scientific_results_written": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("structure", "scientific"), default="structure")
    parser.add_argument("--confirm-protocol-sha256")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = load_protocol(DEFAULT_PROTOCOL)
    if args.mode == "scientific":
        require_scientific_gate(protocol, args.confirm_protocol_sha256)
        raise ProtocolGateError(
            "This preparation revision has no scientific orchestrator; freeze and review "
            "a later implementation before execution."
        )
    payload = run_structure(persist_candidate_receipt=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if (
        payload["panel_audit"]["status"] == "PASS"
        and payload["numerical_equivalence_gate"]["pass"] is True
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
