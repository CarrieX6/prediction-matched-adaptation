#!/usr/bin/env python3
"""Run the formally frozen, development-only Derm7pt CF-MPJC D1 screen.

The command refuses to start unless every byte recorded by the one-shot freeze
manifest still matches and both declared result paths are absent.  All model
outcomes and the frozen analysis are first produced in memory, then written to
unique staging directories and renamed into place only after artifact hashes
have been computed.  Existing official results are never overwritten.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from paper2_evidence_handoff.data import (
    EXPECTED_DEVELOPMENT_CLASS_COUNTS,
    load_derm7pt_development,
)
from paper2_mpjc.analysis import MPJCBootstrapConfig, analyze_mpjc_d1
from paper2_mpjc.pilot import (
    D1Config,
    MODEL_NAMES,
    run_mpjc_pilot,
    validate_official_test_exclusion,
)


DEFAULT_CONFIG = "configs/cf_mpjc_d1_derm7pt.json"
EXPECTED_CONFIG_STATUS = "READY_FOR_RESULT_BLIND_FREEZE_NOT_YET_FROZEN"
EXPECTED_MANIFEST_STATUS = "FROZEN_BEFORE_ANY_CF_MPJC_D1_OUTCOME"
EXPECTED_LOADER_DATASET_ID = "DERM7PT-TRAIN-VALID-MEL-EVIDENCE-HANDOFF-DEV609"
AUDIT_TOLERANCE = 1e-10
PROJECTION_IDENTITY_TOLERANCE = 64.0 * np.finfo(np.float64).eps


def _strict_json_preflight(value: object, *, field: str) -> None:
    """Reject the non-standard NaN/Infinity extensions accepted by json.dumps."""

    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} is not finite standard JSON") from error


def _safe_path(root: Path, raw: object, *, field: str) -> tuple[str, Path]:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{field} must be a non-empty repository-relative path")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{field} must not be absolute or contain '..'")
    normalized = relative.as_posix()
    absolute = (root / relative).resolve(strict=False)
    try:
        absolute.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field} escapes the repository root") from error
    return normalized, absolute


def _read_json(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{field} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_freeze(
    root: Path, config_relative: str, config: Mapping[str, Any]
) -> tuple[dict[str, Any], Path, Path]:
    if config.get("status") != EXPECTED_CONFIG_STATUS:
        raise ValueError(
            "D1 machine config is not in its result-blind freeze-ready state"
        )
    manifest_relative, manifest_path = _safe_path(
        root, config.get("formal_freeze_manifest"), field="formal_freeze_manifest"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"formal D1 freeze manifest is missing: {manifest_relative}"
        )
    manifest = _read_json(manifest_path, field="freeze manifest")
    if (
        manifest.get("status") != EXPECTED_MANIFEST_STATUS
        or manifest.get("manifest_written") is not True
        or manifest.get("dry_run") is not False
    ):
        raise ValueError("D1 freeze manifest is not a completed formal freeze")
    if manifest.get("protocol_id") != config.get("protocol_id"):
        raise ValueError("freeze manifest and config protocol IDs differ")
    if manifest.get("config_path") != config_relative:
        raise ValueError("freeze manifest points to a different machine config")

    groups = _mapping(manifest.get("input_files"), "freeze.input_files")
    observed_files = 0
    changed: dict[str, dict[str, str]] = {}
    for group_name, raw_entries in groups.items():
        entries = _mapping(raw_entries, f"freeze.input_files.{group_name}")
        for raw_relative, raw_record in entries.items():
            relative, path = _safe_path(
                root, raw_relative, field=f"freeze.input_files.{group_name}"
            )
            record = _mapping(raw_record, f"freeze record for {relative}")
            expected = record.get("sha256")
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError(f"freeze hash is malformed for {relative}")
            if not path.is_file():
                raise FileNotFoundError(f"frozen D1 input is missing: {relative}")
            actual = _sha256_file(path)
            if actual != expected:
                changed[relative] = {"expected": expected, "observed": actual}
            observed_files += 1
    if observed_files == 0:
        raise ValueError("freeze manifest contains no input files")
    if changed:
        raise ValueError(f"D1 frozen inputs changed: {changed}")

    outputs = _mapping(config.get("outputs"), "outputs")
    run_relative, run_path = _safe_path(
        root, outputs.get("run_directory"), field="outputs.run_directory"
    )
    analysis_relative, analysis_path = _safe_path(
        root,
        outputs.get("analysis_directory"),
        field="outputs.analysis_directory",
    )
    if run_path == analysis_path:
        raise ValueError("run and analysis directories must differ")
    declared_absent = set(manifest.get("result_paths_absent_at_check", []))
    if {run_relative, analysis_relative} != declared_absent:
        raise ValueError("freeze manifest does not cover exactly the two result paths")
    for relative, path in (
        (run_relative, run_path),
        (analysis_relative, analysis_path),
    ):
        if path.exists():
            raise FileExistsError(f"frozen D1 result path already exists: {relative}")
    return manifest, run_path, analysis_path


def _config_from_contract_mapping(config: Mapping[str, Any]) -> D1Config:
    splitting = _mapping(config.get("splitting"), "splitting")
    outer = _mapping(splitting.get("outer"), "splitting.outer")
    inner = _mapping(splitting.get("inner_probe"), "splitting.inner_probe")
    calibration_split = _mapping(splitting.get("calibration"), "splitting.calibration")
    probe = _mapping(config.get("concept_probe"), "concept_probe")
    calibration = _mapping(config.get("concept_calibration"), "concept_calibration")
    joint = _mapping(config.get("joint_model"), "joint_model")
    diagnosis = _mapping(config.get("diagnosis_interface"), "diagnosis_interface")
    controls = _mapping(config.get("controls"), "controls")
    bilinear = _mapping(
        controls.get("BILINEAR_SCORE_HEAD"), "controls.BILINEAR_SCORE_HEAD"
    )
    bounds = calibration.get("temperature_bounds")
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError("concept_calibration.temperature_bounds must have two values")
    if joint.get("ipf_tolerance_fit") != joint.get("ipf_tolerance_evaluation"):
        raise ValueError(
            "the pilot requires the frozen fit/evaluation IPF tolerances to match"
        )
    if probe.get("max_iter") != diagnosis.get("gold_task_model_max_iter"):
        raise ValueError(
            "the fixed probe and diagnosis iteration caps unexpectedly differ"
        )
    if (
        calibration_split.get("outer_validation_temperature_fit_score_source")
        != "top_level_strict_metaheldout_prior_corrected_probe_scores"
    ):
        raise ValueError("outer-validation temperature source contract changed")
    prior_correction = _mapping(
        probe.get("prior_correction"), "concept_probe.prior_correction"
    )
    probability_epsilon = float(prior_correction["epsilon"])
    if probability_epsilon != float(calibration["epsilon_for_log_scoring"]):
        raise ValueError("prior correction and calibration epsilon unexpectedly differ")
    output = D1Config(
        seed=int(splitting["base_seed"]),
        outer_folds=int(outer["folds"]),
        inner_folds=int(inner["folds"]),
        calibration_folds=int(calibration_split["folds"]),
        repetitions=int(splitting["repetitions"]),
        probe_c=float(probe["C"]),
        diagnosis_c=float(diagnosis["gold_task_model_C"]),
        max_iter=int(probe["max_iter"]),
        temperature_lower=float(bounds[0]),
        temperature_upper=float(bounds[1]),
        temperature_optimization_tolerance=float(calibration["optimization_tolerance"]),
        probability_epsilon=probability_epsilon,
        joint_l2=float(joint["l2"]),
        joint_max_iterations=int(joint["optimizer_max_iterations"]),
        joint_n_starts=int(joint["starts"]),
        joint_start_scale=float(joint["start_scale"]),
        joint_parameter_bound=float(joint["parameter_bound"]),
        ipf_tolerance=float(joint["ipf_tolerance_evaluation"]),
        ipf_max_iterations=int(joint["ipf_max_iterations"]),
        bilinear_l2=float(bilinear["l2"]),
        bilinear_starts=int(bilinear["starts"]),
        bilinear_seed=int(bilinear["seed"]),
        bilinear_max_iter=int(bilinear["max_iter"]),
    )
    if output.outer_folds != 5 or output.repetitions != 5:
        raise ValueError("formal D1 must remain repeated 5x5 outer KFold")
    if output.inner_folds != 4 or output.calibration_folds != 4:
        raise ValueError("formal D1 must retain both four-fold inner layers")
    if int(joint.get("rank", -1)) != 1:
        raise ValueError("formal D1 joint rank must remain one")
    return output


def _analysis_config(config: Mapping[str, Any]) -> MPJCBootstrapConfig:
    analysis = _mapping(config.get("analysis"), "analysis")
    return MPJCBootstrapConfig(
        draws=int(analysis["bootstrap_draws"]),
        seed=int(analysis["bootstrap_seed"]),
        noninferiority_nats=0.005,
        tail_fraction=0.10,
        concentration_top_fraction=0.10,
        concentration_max_share=0.80,
    )


def _finite_numeric(
    table: pd.DataFrame, columns: Sequence[str], *, field: str
) -> np.ndarray:
    try:
        value = table[list(columns)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not np.isfinite(value).all():
        raise ValueError(f"{field} contains a non-finite value")
    return value


def _validate_concept_prediction_runtime(
    concept_records: pd.DataFrame,
    predictions: pd.DataFrame,
    cohort: object,
    training: D1Config,
    contract: Mapping[str, Any],
) -> dict[str, object]:
    """Independently recompute every formal long-table probability transform."""

    expected_columns = contract.get("required_columns")
    if not isinstance(expected_columns, list) or not all(
        isinstance(value, str) for value in expected_columns
    ):
        raise ValueError("concept_prediction_contract.required_columns is invalid")
    if list(concept_records.columns) != expected_columns:
        raise ValueError(
            "concept_predictions exact column contract changed: "
            f"observed={list(concept_records.columns)!r}"
        )
    runtime_guard = _mapping(
        contract.get("runtime_guard"), "concept_prediction_contract.runtime_guard"
    )
    if float(runtime_guard.get("numeric_tolerance", -1.0)) != AUDIT_TOLERANCE:
        raise ValueError("concept prediction numeric audit tolerance changed")
    expected_census = int(contract.get("required_census", -1))
    widths = tuple(
        len(cohort.concept_states[name]) for name in cohort.concept_states
    )
    calculated_census = int(
        len(cohort.y) * training.repetitions * sum(widths)
    )
    if expected_census != calculated_census or len(concept_records) != expected_census:
        raise ValueError(
            "concept_predictions census differs from the frozen/calculated contract"
        )

    duplicate_key = contract.get("duplicate_key_prohibited")
    if not isinstance(duplicate_key, list) or not all(
        isinstance(value, str) for value in duplicate_key
    ):
        raise ValueError("concept prediction duplicate-key contract is invalid")
    if concept_records.duplicated(duplicate_key).any():
        raise ValueError("concept_predictions contains a duplicate formal key")
    if concept_records.isna().any().any():
        raise ValueError("concept_predictions contains a missing value")

    integer_columns = (
        "metadata_index",
        "diagnosis_label",
        "concept_label",
        "state",
        "repetition",
        "outer_fold",
    )
    integer_value = _finite_numeric(
        concept_records, integer_columns, field="concept prediction identities"
    )
    if not np.array_equal(integer_value, np.floor(integer_value)):
        raise ValueError("concept prediction identities must be integers")

    probability_columns = (
        "raw_weighted_probability",
        "probe_fit_prior",
        "prior_corrected_probability",
        "calibrated_probability",
    )
    probability = _finite_numeric(
        concept_records, probability_columns, field="concept probabilities"
    )
    if np.any(probability < 0.0) or np.any(probability > 1.0):
        raise ValueError("concept probability lies outside [0,1]")
    temperature = _finite_numeric(
        concept_records, ("fit_all_temperature",), field="concept temperatures"
    ).reshape(-1)
    if np.any(temperature < training.temperature_lower) or np.any(
        temperature > training.temperature_upper
    ):
        raise ValueError("concept temperature lies outside the frozen bounds")

    if set(concept_records["dataset_id"].astype(str)) != {cohort.dataset_id}:
        raise ValueError("concept_predictions dataset identity changed")
    if set(concept_records["case_id"].astype(str)) != set(
        np.asarray(cohort.case_id).astype(str)
    ):
        raise ValueError("concept_predictions case census changed")
    cohort_case_id = np.asarray(cohort.case_id).astype(str)
    if len(set(cohort_case_id)) != len(cohort_case_id):
        raise ValueError("cohort case IDs are not unique")
    cohort_position = {case_id: row for row, case_id in enumerate(cohort_case_id)}
    try:
        record_position = np.asarray(
            [cohort_position[str(value)] for value in concept_records["case_id"]],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError("concept_predictions contains a case outside the cohort") from error
    if not np.array_equal(
        concept_records["metadata_index"].to_numpy(dtype=np.int64),
        np.asarray(cohort.metadata_index, dtype=np.int64)[record_position],
    ):
        raise ValueError("concept_predictions metadata identity differs from cohort")
    if not np.array_equal(
        concept_records["diagnosis_label"].to_numpy(dtype=np.int64),
        np.asarray(cohort.y, dtype=np.int64)[record_position],
    ):
        raise ValueError("concept_predictions diagnosis labels differ from cohort")
    if set(concept_records["repetition"].astype(int)) != set(
        range(training.repetitions)
    ):
        raise ValueError("concept_predictions repetition census changed")
    if not set(concept_records["outer_fold"].astype(int)).issubset(
        set(range(training.outer_folds))
    ):
        raise ValueError("concept_predictions contains an invalid outer fold")

    prediction_identity_columns = (
        "dataset_id",
        "case_id",
        "metadata_index",
        "label",
        "repetition",
        "outer_fold",
    )
    missing_prediction_columns = set(prediction_identity_columns).difference(
        predictions.columns
    )
    if missing_prediction_columns:
        raise ValueError(
            f"predictions omit identity columns: {sorted(missing_prediction_columns)}"
        )
    identity_nunique = predictions.groupby(
        ["case_id", "repetition"], sort=False, dropna=False
    )[list(prediction_identity_columns)].nunique(dropna=False)
    if (identity_nunique != 1).any().any():
        raise ValueError("prediction models disagree on case/repetition identity")
    prediction_identity = predictions[list(prediction_identity_columns)].drop_duplicates(
        ["case_id", "repetition"]
    )
    expected_case_repetitions = len(cohort.y) * training.repetitions
    if len(prediction_identity) != expected_case_repetitions:
        raise ValueError("prediction case/repetition identity census is incomplete")
    concept_identity = concept_records[
        [
            "dataset_id",
            "case_id",
            "metadata_index",
            "diagnosis_label",
            "repetition",
            "outer_fold",
        ]
    ].drop_duplicates()
    if len(concept_identity) != expected_case_repetitions:
        raise ValueError("concept rows disagree within a case/repetition identity")
    aligned = concept_identity.merge(
        prediction_identity,
        on=["case_id", "repetition"],
        how="outer",
        validate="one_to_one",
        suffixes=("_concept", "_prediction"),
        indicator=True,
    )
    if not (aligned["_merge"] == "both").all():
        raise ValueError("concept and diagnosis case/repetition panels differ")
    for concept_column, prediction_column in (
        ("dataset_id_concept", "dataset_id_prediction"),
        ("metadata_index_concept", "metadata_index_prediction"),
        ("diagnosis_label", "label"),
        ("outer_fold_concept", "outer_fold_prediction"),
    ):
        if not np.array_equal(
            aligned[concept_column].to_numpy(),
            aligned[prediction_column].to_numpy(),
        ):
            raise ValueError(
                "concept and diagnosis identities disagree on " + concept_column
            )

    group_key = ["case_id", "repetition", "concept"]
    expected_group_count = int(
        len(cohort.y) * training.repetitions * len(widths)
    )
    if concept_records.groupby(group_key, sort=False).ngroups != expected_group_count:
        raise ValueError("concept_predictions concept-block census is incomplete")
    names = tuple(cohort.concept_states)
    if set(concept_records["concept"].astype(str)) != set(names):
        raise ValueError("concept_predictions concept names changed")
    for name, width in zip(names, widths, strict=True):
        block = concept_records.loc[concept_records["concept"] == name]
        block_position = np.asarray(
            [cohort_position[str(value)] for value in block["case_id"]],
            dtype=np.int64,
        )
        if not np.array_equal(
            block["concept_label"].to_numpy(dtype=np.int64),
            np.asarray(cohort.concepts[name], dtype=np.int64)[block_position],
        ):
            raise ValueError(f"concept block {name} labels differ from cohort")
        sizes = block.groupby(["case_id", "repetition"], sort=False).size()
        if len(sizes) != expected_case_repetitions or not (sizes == width).all():
            raise ValueError(f"concept block {name} has the wrong state census")
        states = block["state"].to_numpy(dtype=np.int64)
        labels = block["concept_label"].to_numpy(dtype=np.int64)
        if set(states) != set(range(width)) or np.any(labels < 0) or np.any(
            labels >= width
        ):
            raise ValueError(f"concept block {name} contains an invalid state/label")
        state_nunique = block.groupby(
            ["case_id", "repetition"], sort=False
        )["state"].nunique()
        label_nunique = block.groupby(
            ["case_id", "repetition"], sort=False
        )["concept_label"].nunique()
        if not (state_nunique == width).all() or not (label_nunique == 1).all():
            raise ValueError(f"concept block {name} is not a complete labeled simplex")

    block_sum = concept_records.groupby(group_key, sort=False)[
        list(probability_columns)
    ].sum()
    simplex_error = {
        column: float(np.max(np.abs(block_sum[column].to_numpy() - 1.0)))
        for column in probability_columns
    }
    if any(value > AUDIT_TOLERANCE for value in simplex_error.values()):
        raise ValueError(f"concept probability simplex audit failed: {simplex_error}")

    raw = concept_records["raw_weighted_probability"].to_numpy(dtype=np.float64)
    prior = concept_records["probe_fit_prior"].to_numpy(dtype=np.float64)
    numerator = np.maximum(raw, training.probability_epsilon) * prior
    denominator = pd.Series(numerator, index=concept_records.index).groupby(
        [concept_records[column] for column in group_key], sort=False
    ).transform("sum").to_numpy(dtype=np.float64)
    if np.any(denominator <= 0.0):
        raise ValueError("prior-correction recomputation has an empty block")
    expected_corrected = numerator / denominator
    corrected = concept_records["prior_corrected_probability"].to_numpy(
        dtype=np.float64
    )
    correction_error = float(np.max(np.abs(expected_corrected - corrected)))
    if correction_error > AUDIT_TOLERANCE:
        raise ValueError("concept prior-correction recomputation failed")

    group_temperature_count = concept_records.groupby(group_key, sort=False)[
        "fit_all_temperature"
    ].nunique()
    fold_temperature_count = concept_records.groupby(
        ["repetition", "outer_fold", "concept"], sort=False
    )["fit_all_temperature"].nunique()
    if not (group_temperature_count == 1).all() or not (
        fold_temperature_count == 1
    ).all():
        raise ValueError("fit-all temperature is inconsistent within a fitted fold")
    logits = np.log(
        np.clip(corrected, training.probability_epsilon, 1.0)
    ) / temperature
    groupers = [concept_records[column] for column in group_key]
    maximum = pd.Series(logits, index=concept_records.index).groupby(
        groupers, sort=False
    ).transform("max").to_numpy(dtype=np.float64)
    exponential = np.exp(logits - maximum)
    softmax_denominator = pd.Series(
        exponential, index=concept_records.index
    ).groupby(groupers, sort=False).transform("sum").to_numpy(dtype=np.float64)
    expected_calibrated = exponential / softmax_denominator
    calibrated = concept_records["calibrated_probability"].to_numpy(
        dtype=np.float64
    )
    calibration_error = float(np.max(np.abs(expected_calibrated - calibrated)))
    if calibration_error > AUDIT_TOLERANCE:
        raise ValueError("concept temperature-transform recomputation failed")

    return {
        "passed": True,
        "tolerance": float(AUDIT_TOLERANCE),
        "row_count": int(len(concept_records)),
        "expected_row_count": int(expected_census),
        "unique_key": list(duplicate_key),
        "concept_block_count": int(expected_group_count),
        "maximum_simplex_error": simplex_error,
        "maximum_prior_correction_recompute_error": correction_error,
        "maximum_temperature_recompute_error": calibration_error,
        "identity_matches_diagnosis_predictions": True,
    }


def _validate_fit_runtime(
    fit_manifests: Sequence[Mapping[str, object]],
    predictions: pd.DataFrame,
    joint_records: pd.DataFrame,
    training: D1Config,
    config: Mapping[str, Any],
) -> dict[str, object]:
    """Validate optimizer/projection records without using any outcome metric."""

    contract = _mapping(config.get("fit_manifest_contract"), "fit_manifest_contract")
    required_fields = contract.get("required_fields")
    if not isinstance(required_fields, list) or not all(
        isinstance(value, str) for value in required_fields
    ):
        raise ValueError("fit_manifest_contract.required_fields is invalid")
    expected_count = int(contract.get("required_census", -1))
    calculated_count = training.repetitions * training.outer_folds
    if expected_count != calculated_count or len(fit_manifests) != expected_count:
        raise ValueError("fit manifest census differs from the frozen contract")
    expected_pairs = set(
        predictions[["repetition", "outer_fold"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    formal_pairs = set(
        (repetition, fold)
        for repetition in range(training.repetitions)
        for fold in range(training.outer_folds)
    )
    observed_pairs: list[tuple[int, int]] = []
    projection_converged: list[bool] = []
    bilinear_converged: list[bool] = []
    joint_converged: list[bool] = []
    unprojected_converged: list[bool] = []
    projection_errors: list[float] = []
    projection_iterations: list[int] = []
    bilinear_iterations: list[int] = []
    controls = _mapping(config.get("controls"), "controls")
    bilinear_contract = _mapping(
        controls.get("BILINEAR_SCORE_HEAD"), "controls.BILINEAR_SCORE_HEAD"
    )
    expected_parameter_count = int(bilinear_contract["parameter_count"])
    joint_contract = _mapping(config.get("joint_model"), "joint_model")
    expected_joint_dimension = int(joint_contract["free_interaction_parameters"])
    solver_contract = _mapping(
        joint_contract.get("ipf_solver_contract"), "joint_model.ipf_solver_contract"
    )
    zero_solver = str(solver_contract["zero_phi_solver"])
    nonzero_solver = str(solver_contract["nonzero_phi_diagnostic_solver"])
    required_joint_columns = {
        "repetition",
        "outer_fold",
        "max_joint_marginal_error",
    }
    if not required_joint_columns.issubset(joint_records.columns):
        raise ValueError("joint records omit projection-audit columns")
    joint_numeric = _finite_numeric(
        joint_records,
        tuple(sorted(required_joint_columns)),
        field="joint record projection audit",
    )
    del joint_numeric

    for index, item in enumerate(fit_manifests):
        missing = set(required_fields).difference(item)
        if missing:
            raise ValueError(
                f"fit manifest {index} omits frozen fields: {sorted(missing)}"
            )
        repetition = int(item["repetition"])
        outer_fold = int(item["outer_fold"])
        observed_pairs.append((repetition, outer_fold))
        for field in (
            "joint_converged",
            "joint_projection_converged",
            "bilinear_converged",
            "strict_metaheldout_isolation",
            "unprojected_converged",
        ):
            if not isinstance(item.get(field), (bool, np.bool_)):
                raise ValueError(f"fit manifest {index} field {field} is not boolean")
        if item["strict_metaheldout_isolation"] is not True:
            raise ValueError("fit manifest does not assert strict metaheldout isolation")
        joint_converged.append(bool(item["joint_converged"]))
        projection_converged.append(bool(item["joint_projection_converged"]))
        bilinear_converged.append(bool(item["bilinear_converged"]))
        unprojected_converged.append(bool(item["unprojected_converged"]))

        numeric_fields = (
            "joint_objective",
            "joint_concept_nll",
            "joint_gradient_norm",
            "joint_projection_energy_span",
            "joint_projection_max_marginal_error",
            "max_joint_marginal_error",
            "unprojected_objective",
            "unprojected_concept_nll",
            "unprojected_gradient_norm",
            "max_unprojected_marginal_drift",
            "bilinear_objective",
        )
        numeric = np.asarray([item[field] for field in numeric_fields], dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError(f"fit manifest {index} contains non-finite diagnostics")
        error = float(item["joint_projection_max_marginal_error"])
        recomputed_error = float(item["max_joint_marginal_error"])
        if error > training.ipf_tolerance:
            raise ValueError("converged projection exceeds the frozen IPF tolerance")
        if recomputed_error > training.ipf_tolerance + PROJECTION_IDENTITY_TOLERANCE:
            raise ValueError("independently recomputed projection error exceeds tolerance")
        if abs(error - recomputed_error) > PROJECTION_IDENTITY_TOLERANCE:
            raise ValueError("projection and independently recomputed errors disagree")
        fold_joint = joint_records.loc[
            (joint_records["repetition"].astype(int) == repetition)
            & (joint_records["outer_fold"].astype(int) == outer_fold)
        ]
        if fold_joint.empty:
            raise ValueError("joint records omit a fitted outer fold")
        recorded_error = float(fold_joint["max_joint_marginal_error"].max())
        if recorded_error > training.ipf_tolerance + PROJECTION_IDENTITY_TOLERANCE:
            raise ValueError("joint-record projection error exceeds tolerance")
        if abs(recorded_error - recomputed_error) > PROJECTION_IDENTITY_TOLERANCE:
            raise ValueError("fit manifest and joint-record projection errors disagree")
        projection_errors.append(error)

        iteration = int(item["joint_projection_iterations"])
        stages = int(item["joint_projection_stages"])
        stage_iterations_raw = item["joint_projection_stage_iterations"]
        if not isinstance(stage_iterations_raw, list):
            raise ValueError("joint projection stage iterations must be a list")
        stage_iterations = [int(value) for value in stage_iterations_raw]
        if iteration < 0 or iteration > training.ipf_max_iterations:
            raise ValueError("joint projection iteration count is outside its budget")
        if any(value < 0 for value in stage_iterations) or sum(stage_iterations) != iteration:
            raise ValueError("joint projection stage iteration identity failed")
        if stages < 0 or len(stage_iterations) > stages:
            raise ValueError("joint projection stage census is invalid")
        if bool(item["joint_projection_converged"]) and len(stage_iterations) != stages:
            raise ValueError("converged joint projection omits a planned stage")
        solver = item["joint_projection_solver"]
        if solver not in (zero_solver, nonzero_solver):
            raise ValueError("joint projection solver identity changed")
        phi = np.asarray(item["joint_phi"], dtype=np.float64).reshape(-1)
        if phi.shape != (expected_joint_dimension,) or not np.isfinite(phi).all():
            raise ValueError("joint parameter vector has invalid dimension/values")
        expected_solver = zero_solver if np.count_nonzero(phi) == 0 else nonzero_solver
        if solver != expected_solver:
            raise ValueError("joint parameter and projection solver identities disagree")
        if solver == zero_solver and (
            stages != 0 or stage_iterations or float(item["joint_projection_energy_span"]) != 0.0
        ):
            raise ValueError("independent projection diagnostics are inconsistent")
        projection_iterations.append(iteration)

        bilinear_iteration = int(item["bilinear_iterations"])
        selected_start = int(item["bilinear_selected_start"])
        if bilinear_iteration < 0 or bilinear_iteration > training.bilinear_max_iter:
            raise ValueError("bilinear iteration count is outside its frozen budget")
        if selected_start < 0 or selected_start >= training.bilinear_starts:
            raise ValueError("bilinear selected-start identity is invalid")
        if int(item["bilinear_parameter_count"]) != expected_parameter_count:
            raise ValueError("bilinear parameter count changed")
        bilinear_iterations.append(bilinear_iteration)

        top_meta = item["outer_validation_temperature_top_level_meta_folds"]
        calibration = item["strict_meta_calibration_folds"]
        if not isinstance(top_meta, list) or len(top_meta) != training.calibration_folds:
            raise ValueError("top-level strict meta-fold audit census changed")
        if not isinstance(calibration, list) or len(calibration) != training.calibration_folds + 1:
            raise ValueError("strict meta calibration audit census changed")
        if calibration[:-1] != top_meta:
            raise ValueError("outer-temperature source and strict meta audits diverge")
        if {int(value.get("meta_fold", -1)) for value in top_meta} != set(
            range(training.calibration_folds)
        ):
            raise ValueError("strict meta-fold identities changed")
        for meta in top_meta:
            if meta.get("fit_heldout_disjoint") is not True:
                raise ValueError("a strict meta fold does not assert disjointness")
            inner = meta.get("inner_probe_folds_within_meta_fit")
            if not isinstance(inner, list) or len(inner) != training.inner_folds:
                raise ValueError("strict meta fold inner-probe census changed")
            if any(value.get("fit_validation_disjoint") is not True for value in inner):
                raise ValueError("an inner probe fold does not assert disjointness")
        outer_temperature = calibration[-1]
        if (
            outer_temperature.get("fit_all_for_outer_validation") is not True
            or outer_temperature.get("temperature_fit_score_source")
            != "top_level_strict_metaheldout_prior_corrected_probe_scores"
            or int(outer_temperature.get("top_level_metaheldout_fold_count", -1))
            != training.calibration_folds
        ):
            raise ValueError("outer-validation temperature source contract changed")

    if (
        len(set(observed_pairs)) != len(observed_pairs)
        or set(observed_pairs) != formal_pairs
        or expected_pairs != formal_pairs
    ):
        raise ValueError("fit manifests and predictions disagree on outer-fold census")
    maximum_projection_error = float(max(projection_errors))
    return {
        "passed": True,
        "manifest_count": int(len(fit_manifests)),
        "expected_manifest_count": int(expected_count),
        "joint_fit_converged_count": int(sum(joint_converged)),
        "unprojected_fit_converged_count": int(sum(unprojected_converged)),
        "projection_converged_count": int(sum(projection_converged)),
        "bilinear_converged_count": int(sum(bilinear_converged)),
        "all_projection_diagnostics_converged": bool(all(projection_converged)),
        "all_bilinear_fits_converged": bool(all(bilinear_converged)),
        "maximum_projection_marginal_error": maximum_projection_error,
        "maximum_projection_iterations": int(max(projection_iterations)),
        "maximum_bilinear_iterations": int(max(bilinear_iterations)),
    }


def _apply_gate_validity(
    analysis: dict[str, object],
    concept_audit: Mapping[str, object],
    fit_audit: Mapping[str, object],
) -> None:
    """Make strong-control/runtime validity logically prior to performance gates."""

    preconditions = {
        "all_bilinear_fits_converged": bool(
            fit_audit["all_bilinear_fits_converged"]
        ),
        "all_projection_diagnostics_converged": bool(
            fit_audit["all_projection_diagnostics_converged"]
        ),
        "concept_prediction_runtime_audit_passed": bool(concept_audit["passed"]),
        # A final preflight after this field is inserted verifies the assertion.
        "all_json_artifacts_standard_finite_json": True,
    }
    valid = bool(all(preconditions.values()))
    gate = analysis.get("gate")
    if not isinstance(gate, dict):
        raise ValueError("analysis omitted its gate object")
    gate["validity_preconditions"] = preconditions
    gate["gate_evaluation_valid"] = valid
    if not valid:
        gate["performance_signal_before_validity_check"] = gate.get(
            "derm7pt_initial_method_signal"
        )
        gate["derm7pt_initial_method_signal"] = "INVALID_GATE_EVALUATION"


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_csv(path: Path, table: pd.DataFrame) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        table.to_csv(stream, index=False)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_publish(
    root: Path,
    run_path: Path,
    analysis_path: Path,
    predictions: pd.DataFrame,
    joint_records: pd.DataFrame,
    concept_records: pd.DataFrame,
    fit_manifests: list[dict[str, object]],
    run_manifest: dict[str, object],
    analysis: dict[str, object],
) -> None:
    if run_path.exists() or analysis_path.exists():
        raise FileExistsError("a D1 result path appeared before atomic publication")
    if run_path.parent != analysis_path.parent:
        raise ValueError("atomic D1 publication requires a shared result parent")
    parent = run_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".cf_mpjc_d1_staging_", dir=parent))
    stage_run = staging_root / "run"
    stage_analysis = staging_root / "analysis"
    stage_run.mkdir()
    stage_analysis.mkdir()
    published_run = False
    published_analysis = False
    try:
        _write_csv(stage_run / "predictions.csv", predictions)
        _write_csv(stage_run / "joint_records.csv", joint_records)
        _write_csv(stage_run / "concept_predictions.csv", concept_records)
        _write_json(stage_run / "fit_manifests.json", fit_manifests)
        run_manifest["artifact_sha256"] = {
            name: _sha256_file(stage_run / name)
            for name in (
                "predictions.csv",
                "joint_records.csv",
                "concept_predictions.csv",
                "fit_manifests.json",
            )
        }
        _write_json(stage_analysis / "analysis.json", analysis)
        run_manifest["artifact_sha256"]["analysis.json"] = _sha256_file(
            stage_analysis / "analysis.json"
        )
        run_manifest["analysis_artifact"] = str(
            analysis_path.relative_to(root) / "analysis.json"
        )
        _write_json(stage_run / "run_manifest.json", run_manifest)
        # Recheck immediately before publishing; no existing directory is ever
        # removed or replaced by this command.
        if run_path.exists() or analysis_path.exists():
            raise FileExistsError("a D1 result path appeared during staging")
        stage_run.rename(run_path)
        published_run = True
        stage_analysis.rename(analysis_path)
        published_analysis = True
        staging_root.rmdir()
    except BaseException:
        # If the second of the two sibling renames fails, move the first one
        # back into its unique staging root.  This is a recovery rename, not a
        # deletion or overwrite, and keeps formal paths all-or-none.
        if published_run and not published_analysis and run_path.exists():
            run_path.rename(stage_run)
        if published_analysis and not published_run and analysis_path.exists():
            analysis_path.rename(stage_analysis)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen Derm7pt train+valid CF-MPJC D1 screen."
    )
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--config", type=Path, default=Path(DEFAULT_CONFIG))
    args = parser.parse_args()

    root = args.repo_root.resolve(strict=True)
    config_relative, config_path = _safe_path(
        root, args.config.as_posix(), field="config"
    )
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = _read_json(config_path, field="D1 config")
    serialization = _mapping(config.get("serialization_contract"), "serialization_contract")
    if serialization.get("allow_nan") is not False:
        raise ValueError("formal D1 JSON must prohibit NaN and Infinity")
    if tuple(config.get("models", ())) != MODEL_NAMES:
        raise ValueError("machine config and pilot model order differ")
    outputs = _mapping(config.get("outputs"), "outputs")
    if set(outputs.get("required_run_files", ())) != {
        "predictions.csv",
        "concept_predictions.csv",
        "joint_records.csv",
        "fit_manifests.json",
        "run_manifest.json",
    }:
        raise ValueError(
            "machine config required-run-file census differs from the runner"
        )
    if outputs.get("required_analysis_files") != ["analysis.json"]:
        raise ValueError(
            "machine config required-analysis-file census differs from the runner"
        )
    if outputs.get("overwrite") is not False:
        raise ValueError("formal D1 output overwrite policy must remain false")
    freeze_manifest, run_path, analysis_path = _verify_freeze(
        root, config_relative, config
    )
    dataset = _mapping(config.get("dataset"), "dataset")
    if dataset.get("dataset_id") != EXPECTED_LOADER_DATASET_ID:
        raise ValueError(
            "frozen machine config does not name the loader's dataset identity"
        )
    if dataset.get("formal_runner_loads_only_materialized_development_rows") is not True:
        raise ValueError("formal D1 requires a development-only materialized archive")
    _, feature_archive = _safe_path(
        root, dataset.get("feature_archive"), field="dataset.feature_archive"
    )
    _, metadata_csv = _safe_path(
        root, dataset.get("metadata"), field="dataset.metadata"
    )
    _, official_meta_directory = _safe_path(
        root,
        dataset.get("official_meta_directory"),
        field="dataset.official_meta_directory",
    )
    cohort = load_derm7pt_development(
        feature_archive,
        metadata_csv,
        official_meta_directory,
        require_development_only_archive=True,
    )
    if cohort.dataset_id != dataset.get("dataset_id"):
        raise ValueError("loaded cohort and frozen dataset IDs differ")
    if len(cohort.y) != int(dataset["expected_case_count"]):
        raise ValueError("loaded D1 case census differs from the frozen config")
    expected_counts = tuple(int(value) for value in EXPECTED_DEVELOPMENT_CLASS_COUNTS)
    frozen_counts = _mapping(
        dataset.get("expected_diagnosis_class_counts"),
        "dataset.expected_diagnosis_class_counts",
    )
    if tuple(int(frozen_counts[label]) for label in cohort.labels) != expected_counts:
        raise ValueError(
            "frozen diagnosis class census differs from the loader contract"
        )
    if (
        tuple(np.bincount(cohort.y, minlength=len(cohort.labels)).tolist())
        != expected_counts
    ):
        raise ValueError("loaded diagnosis class census changed")
    expected_names = tuple(str(value) for value in dataset["concept_order"])
    expected_widths = tuple(int(value) for value in dataset["concept_cardinalities"])
    if (
        tuple(cohort.concept_states) != expected_names
        or tuple(len(cohort.concept_states[name]) for name in expected_names)
        != expected_widths
    ):
        raise ValueError("loaded concept order or cardinalities changed")
    if int(dataset.get("concept_one_hot_dimension", -1)) != sum(expected_widths):
        raise ValueError("frozen concept one-hot dimension changed")
    if int(dataset.get("cartesian_state_count", -1)) != int(np.prod(expected_widths)):
        raise ValueError("frozen Cartesian state count changed")
    joint_contract = _mapping(config.get("joint_model"), "joint_model")
    free_interactions = sum(width - 1 for width in expected_widths)
    if int(joint_contract.get("free_interaction_parameters", -1)) != free_interactions:
        raise ValueError("frozen rank-one interaction parameter count changed")
    bilinear_contract = _mapping(
        _mapping(config.get("controls"), "controls").get("BILINEAR_SCORE_HEAD"),
        "controls.BILINEAR_SCORE_HEAD",
    )
    if int(bilinear_contract.get("parameter_count", -1)) != 1 + 2 * free_interactions:
        raise ValueError("frozen bilinear capacity-match parameter count changed")
    official_test_audit = validate_official_test_exclusion(
        cohort, official_meta_directory
    )

    training = _config_from_contract_mapping(config)
    result = run_mpjc_pilot(
        cohort,
        training,
        official_test_audit=official_test_audit,
        progress=lambda message: print(message, flush=True),
    )
    predictions = pd.DataFrame(result.pop("predictions"))
    joint_records = pd.DataFrame(result.pop("joint_records"))
    concept_records = pd.DataFrame(result.pop("concept_records"))
    fit_manifests = result.pop("fit_manifests")
    concept_audit = _validate_concept_prediction_runtime(
        concept_records,
        predictions,
        cohort,
        training,
        _mapping(
            config.get("concept_prediction_contract"),
            "concept_prediction_contract",
        ),
    )
    fit_audit = _validate_fit_runtime(
        fit_manifests,
        predictions,
        joint_records,
        training,
        config,
    )
    analysis = analyze_mpjc_d1(
        predictions,
        joint_records,
        fit_manifests,
        models=MODEL_NAMES,
        config=_analysis_config(config),
    )
    _apply_gate_validity(analysis, concept_audit, fit_audit)
    result.update(
        {
            "freeze_manifest": str(config["formal_freeze_manifest"]),
            "freeze_manifest_sha256": _sha256_file(
                root / str(config["formal_freeze_manifest"])
            ),
            "freeze_manifest_id": freeze_manifest.get("manifest_id"),
            "machine_config": config_relative,
            "machine_config_sha256": _sha256_file(config_path),
            "fit_manifest_count": int(len(fit_manifests)),
            "prediction_row_count": int(len(predictions)),
            "joint_record_count": int(len(joint_records)),
            "concept_record_count": int(len(concept_records)),
            "runtime_audit": {
                "concept_predictions": concept_audit,
                "fit_manifests": fit_audit,
                "standard_finite_json_preflight": True,
            },
            "analysis_gate": analysis["gate"],
        }
    )
    _strict_json_preflight(fit_manifests, field="fit manifests")
    _strict_json_preflight(analysis, field="analysis")
    _strict_json_preflight(result, field="run manifest")
    # Outcomes and analysis existed only in memory up to this point.  The two
    # formal paths are still required to be absent inside _atomic_publish.
    _atomic_publish(
        root,
        run_path,
        analysis_path,
        predictions,
        joint_records,
        concept_records,
        fit_manifests,
        result,
        analysis,
    )
    print(
        json.dumps(
            analysis["gate"], ensure_ascii=False, indent=2, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
