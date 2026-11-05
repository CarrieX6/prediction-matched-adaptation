"""Fail-closed Paper B CBIS P3 experiment machinery.

The dry-run path validates only the frozen Mass-Training cohort contract.  The
scientific path cannot begin until an atomically published embedding release,
a hash-complete experiment freeze manifest, and a root execution approval
receipt all agree.  No function in this module knows a Mass-Test path.
"""

from __future__ import annotations

import csv
import ctypes
import errno
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from threadpoolctl import threadpool_info, threadpool_limits


PROTOCOL_ID = "PAPER_B_CBIS_P3_MINIMAL_V1_20260910"
DRAFT_STATUS = "DRAFT_PREOUTCOME_WITH_EMBEDDING_BOUND"
FINAL_CONFIG_STATUS = "READY_FOR_ROOT_APPROVAL"
FREEZE_STATUS = "FROZEN_AWAITING_ROOT_APPROVAL"
APPROVAL_STATUS = "ROOT_APPROVED_FOR_SCIENTIFIC_OUTCOME_EXECUTION"
OUTCOME_LOG_LOSS_CLIP = 1e-12

PRIMARY_VARIANT = "PRIMARY_COMPLETE_CONCEPT"
SENSITIVITY_VARIANT = "PREDECLARED_MISSINGNESS_SENSITIVITY"

OOF_RAW_SOFT = "OOF_RAW_SOFT_ADDITIVE"
OOF_RAW_HARD = "OOF_RAW_HARD_ADDITIVE"
OOF_RAW_LOG_ODDS = "OOF_RAW_LOG_ODDS_ADDITIVE"
GOLD_ORACLE = "GOLD_C_ORACLE_ADDITIVE"
GOLD_TO_PREDICTED = "GOLD_C_TO_PREDICTED_SOFT_MISMATCH"
DIRECT_X = "DIRECT_X_LOGISTIC"
INSAMPLE_RAW_SOFT = "INSAMPLE_RAW_SOFT_ADDITIVE"

PRIMARY_MODELS = (
    OOF_RAW_SOFT,
    OOF_RAW_HARD,
    OOF_RAW_LOG_ODDS,
    GOLD_ORACLE,
    GOLD_TO_PREDICTED,
    DIRECT_X,
    INSAMPLE_RAW_SOFT,
)
SENSITIVITY_MODELS = (
    OOF_RAW_SOFT,
    OOF_RAW_HARD,
    OOF_RAW_LOG_ODDS,
    DIRECT_X,
    INSAMPLE_RAW_SOFT,
)

STABLE_CONCEPTS = (
    "shape_round",
    "shape_oval",
    "shape_lobulated",
    "shape_irregular",
    "shape_architectural_distortion",
    "margin_circumscribed",
    "margin_microlobulated",
    "margin_obscured",
    "margin_ill_defined",
    "margin_spiculated",
)

EXPOSURE_ROLES = (
    "OUTER_TRAIN_CASE_EXCLUDED_OOF",
    "OUTER_TRAIN_IN_SAMPLE_AUDIT",
    "OUTER_VALIDATION_CASE_EXCLUDED",
)

EXPECTED_COMPARISONS = (
    ("PRIMARY_RAW_SOFT_VS_HARD", PRIMARY_VARIANT, OOF_RAW_SOFT, OOF_RAW_HARD, "PRIMARY", "NLL(hard)-NLL(soft)"),
    ("SECONDARY_RAW_SOFT_VS_REFERENCE_LOG_ODDS", PRIMARY_VARIANT, OOF_RAW_SOFT, OOF_RAW_LOG_ODDS, "SECONDARY", "NLL(log-odds)-NLL(soft)"),
    ("ORACLE_GAP", PRIMARY_VARIANT, GOLD_ORACLE, OOF_RAW_SOFT, "ORACLE", "NLL(soft)-NLL(gold oracle)"),
    ("SAME_ORACLE_HEAD_GOLD_INPUT_TO_PREDICTED_INPUT_DROP", PRIMARY_VARIANT, GOLD_ORACLE, GOLD_TO_PREDICTED, "RECOVERABILITY_AND_INPUT_SHIFT_AUDIT", "NLL(gold-to-predicted)-NLL(gold oracle)"),
    ("DIAGNOSIS_HEAD_TRAINING_EXPOSURE_MISMATCH", PRIMARY_VARIANT, OOF_RAW_SOFT, GOLD_TO_PREDICTED, "MISMATCH", "NLL(gold-trained-to-predicted)-NLL(predicted-trained-to-predicted)"),
    ("DIRECT_X_CONTEXT", PRIMARY_VARIANT, OOF_RAW_SOFT, DIRECT_X, "CONTEXT_ONLY", "NLL(direct-X)-NLL(soft)"),
    ("CASE_EXCLUSION_EXPOSURE_AUDIT", PRIMARY_VARIANT, OOF_RAW_SOFT, INSAMPLE_RAW_SOFT, "EXPOSURE_AUDIT", "NLL(in-sample exposure)-NLL(case-excluded exposure)"),
    ("SENSITIVITY_RAW_SOFT_VS_HARD", SENSITIVITY_VARIANT, OOF_RAW_SOFT, OOF_RAW_HARD, "MISSINGNESS_SENSITIVITY", "NLL(hard)-NLL(soft)"),
    ("SENSITIVITY_RAW_SOFT_VS_REFERENCE_LOG_ODDS", SENSITIVITY_VARIANT, OOF_RAW_SOFT, OOF_RAW_LOG_ODDS, "MISSINGNESS_SENSITIVITY", "NLL(log-odds)-NLL(soft)"),
    ("SENSITIVITY_CASE_EXCLUSION_EXPOSURE_AUDIT", SENSITIVITY_VARIANT, OOF_RAW_SOFT, INSAMPLE_RAW_SOFT, "MISSINGNESS_SENSITIVITY_EXPOSURE_AUDIT", "NLL(in-sample exposure)-NLL(case-excluded exposure)"),
)

THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def observed_scientific_runtime() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable_resolved": str(Path(sys.executable).resolve(strict=True)),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scikit-learn", "scipy", "threadpoolctl")
        },
    }


def verify_scientific_runtime(config: Mapping[str, Any]) -> dict[str, object]:
    runtime = config.get("runtime", {})
    required = runtime.get("required")
    observed = observed_scientific_runtime()
    if observed != required:
        raise RuntimeError(
            f"scientific runtime differs from freeze: observed={observed!r}"
        )
    if tuple(runtime.get("environment_thread_variables_must_be_unset", ())) != THREAD_ENVIRONMENT_VARIABLES:
        raise ValueError("scientific thread-environment contract changed")
    contaminated = {
        name: os.environ[name]
        for name in THREAD_ENVIRONMENT_VARIABLES
        if name in os.environ
    }
    if contaminated:
        raise RuntimeError(
            f"scientific runtime requires unset thread environment: {contaminated}"
        )
    if runtime.get("numeric_library_thread_limit_during_fit_and_analysis") != 1:
        raise ValueError("scientific numeric-library thread limit changed")
    return observed


def scientific_threadpool_snapshot() -> list[dict[str, object]]:
    keys = (
        "user_api",
        "internal_api",
        "prefix",
        "version",
        "threading_layer",
        "architecture",
        "num_threads",
    )
    return sorted(
        [{key: pool.get(key) for key in keys} for pool in threadpool_info()],
        key=lambda row: (
            str(row["user_api"]),
            str(row["prefix"]),
            str(row["version"]),
        ),
    )


def assert_scientific_thread_limit(snapshot: Sequence[Mapping[str, object]]) -> None:
    if not snapshot or any(pool.get("num_threads") != 1 for pool in snapshot):
        raise RuntimeError(f"numeric-library thread limit is not exactly one: {snapshot!r}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_values(values: Iterable[object]) -> str:
    payload = "\n".join(sorted(str(value) for value in values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def seed_from(base: int, *parts: object) -> int:
    payload = "|".join([str(base), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") & 0x7FFFFFFF


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: str | Path, value: Mapping[str, object]) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def safe_path(root: Path, raw: object, *, must_exist: bool = True) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError("path must be a non-empty repository-relative string")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe repository path: {raw!r}")
    path = (root / relative).resolve(strict=must_exist)
    path.relative_to(root.resolve())
    return path


@dataclass(frozen=True)
class CbisCohort:
    variant: str
    unit_id: np.ndarray
    patient_id: np.ndarray
    outer_fold: np.ndarray
    y: np.ndarray
    concepts: np.ndarray
    concept_valid: np.ndarray
    concept_names: tuple[str, ...]
    view_present: np.ndarray
    image_id: np.ndarray
    local_dicom_path: np.ndarray
    x: np.ndarray | None = None

    def validate(self, *, expected_breasts: int, expected_patients: int) -> None:
        n = len(self.unit_id)
        if n != expected_breasts or len(set(self.patient_id.astype(str))) != expected_patients:
            raise ValueError("CBIS P3 cohort census changed")
        arrays = (
            self.patient_id,
            self.outer_fold,
            self.y,
            self.concepts,
            self.concept_valid,
            self.view_present,
            self.image_id,
            self.local_dicom_path,
        )
        if any(len(value) != n for value in arrays):
            raise ValueError("CBIS P3 cohort arrays have inconsistent lengths")
        if len(set(self.unit_id.astype(str))) != n:
            raise ValueError("CBIS P3 breast unit IDs are duplicated")
        if self.concepts.shape != (n, len(self.concept_names)):
            raise ValueError("CBIS P3 concept matrix shape changed")
        if self.concept_valid.shape != self.concepts.shape:
            raise ValueError("CBIS P3 concept validity shape changed")
        if self.view_present.shape != (n, 2) or self.image_id.shape != (n, 2):
            raise ValueError("CBIS P3 view layout changed")
        if not np.all(np.isin(self.outer_fold, np.arange(5))):
            raise ValueError("CBIS P3 outer fold is outside 0..4")
        if not np.all(np.isin(self.y, (0, 1))) or not np.all(np.isin(self.concepts, (0, 1))):
            raise ValueError("CBIS P3 binary labels changed")
        if not np.all(np.isin(self.concept_valid, (0, 1))):
            raise ValueError("CBIS P3 validity masks changed")
        patient_fold: dict[str, set[int]] = {}
        for patient, fold in zip(self.patient_id.astype(str), self.outer_fold, strict=True):
            patient_fold.setdefault(patient, set()).add(int(fold))
        if any(len(folds) != 1 for folds in patient_fold.values()):
            raise ValueError("a CBIS P3 patient crosses outer folds")
        if self.x is not None:
            if self.x.shape != (n, 512) or not np.isfinite(self.x).all():
                raise ValueError("CBIS P3 breast embeddings are malformed")

    def with_x(self, x: np.ndarray) -> "CbisCohort":
        return CbisCohort(
            variant=self.variant,
            unit_id=self.unit_id,
            patient_id=self.patient_id,
            outer_fold=self.outer_fold,
            y=self.y,
            concepts=self.concepts,
            concept_valid=self.concept_valid,
            concept_names=self.concept_names,
            view_present=self.view_present,
            image_id=self.image_id,
            local_dicom_path=self.local_dicom_path,
            x=np.asarray(x, dtype=np.float64),
        )


def load_cohort(
    archive_path: Path,
    stable_concepts: Sequence[str],
    *,
    complete_only: bool,
) -> CbisCohort:
    with np.load(archive_path, allow_pickle=False) as archive:
        required = {
            "unit_id",
            "patient_id",
            "outer_fold",
            "outcome_id",
            "complete_concept_eligible",
            "concept_family_valid",
            "concept_multi_hot",
            "concept_names",
            "view_present",
            "image_id",
            "local_dicom_path",
        }
        if not required.issubset(archive.files):
            raise ValueError(f"cohort archive lacks {sorted(required - set(archive.files))}")
        if (
            str(archive["schema_version"].item())
            != "paper-b-cbis-mass-train-breast-cohort-v1"
            or archive["outcome_names"].astype(str).tolist() != ["BENIGN", "MALIGNANT"]
            or archive["concept_family_names"].astype(str).tolist() != ["shape", "margin"]
            or archive["view_names"].astype(str).tolist() != ["CC", "MLO"]
        ):
            raise ValueError("frozen cohort archive semantic schema changed")
        all_names = archive["concept_names"].astype(str).tolist()
        if len(set(stable_concepts)) != len(stable_concepts):
            raise ValueError("stable concept names are duplicated")
        try:
            concept_columns = [all_names.index(str(name)) for name in stable_concepts]
        except ValueError as error:
            raise ValueError("a stable concept is absent from the cohort archive") from error
        complete = archive["complete_concept_eligible"].astype(bool)
        keep = complete if complete_only else np.ones(len(complete), dtype=bool)
        family_valid = archive["concept_family_valid"].astype(np.uint8)
        concept_valid = np.column_stack(
            [
                family_valid[:, 0] if str(name).startswith("shape_") else family_valid[:, 1]
                for name in stable_concepts
            ]
        )
        cohort = CbisCohort(
            variant=PRIMARY_VARIANT if complete_only else SENSITIVITY_VARIANT,
            unit_id=archive["unit_id"].astype(str)[keep],
            patient_id=archive["patient_id"].astype(str)[keep],
            outer_fold=archive["outer_fold"].astype(np.int8)[keep],
            y=archive["outcome_id"].astype(np.int8)[keep],
            concepts=archive["concept_multi_hot"].astype(np.int8)[keep][:, concept_columns],
            concept_valid=concept_valid.astype(np.uint8)[keep],
            concept_names=tuple(str(name) for name in stable_concepts),
            view_present=archive["view_present"].astype(np.uint8)[keep],
            image_id=archive["image_id"].astype(str)[keep],
            local_dicom_path=archive["local_dicom_path"].astype(str)[keep],
        )
    cohort.validate(
        expected_breasts=655 if complete_only else 662,
        expected_patients=633 if complete_only else 639,
    )
    if complete_only and not np.all(cohort.concept_valid == 1):
        raise ValueError("primary CBIS P3 cohort contains missing concepts")
    if not complete_only:
        missing = missingness_contract_audit(cohort)
        if missing["incomplete_breasts"] != 7 or missing["shape_missing_breasts"] != 1 or missing["margin_missing_breasts"] != 7:
            raise ValueError("CBIS P3 frozen missing-family census changed")
    return cohort


def missingness_contract_audit(cohort: CbisCohort) -> dict[str, object]:
    """Audit the sensitivity placeholder/mask rule without treating missing as negative."""

    shape_columns = [
        index for index, name in enumerate(cohort.concept_names) if name.startswith("shape_")
    ]
    margin_columns = [
        index for index, name in enumerate(cohort.concept_names) if name.startswith("margin_")
    ]
    if len(shape_columns) != 5 or len(margin_columns) != 5:
        raise ValueError("stable concept families must contain five shape and five margin nodes")
    shape_valid = cohort.concept_valid[:, shape_columns[0]].astype(bool)
    margin_valid = cohort.concept_valid[:, margin_columns[0]].astype(bool)
    if not np.all(cohort.concept_valid[:, shape_columns] == shape_valid[:, None]):
        raise ValueError("shape validity is not a family-level mask")
    if not np.all(cohort.concept_valid[:, margin_columns] == margin_valid[:, None]):
        raise ValueError("margin validity is not a family-level mask")
    if np.any(cohort.concepts[~shape_valid][:, shape_columns] != 0):
        raise ValueError("missing shape storage placeholder is not fixed zero")
    if np.any(cohort.concepts[~margin_valid][:, margin_columns] != 0):
        raise ValueError("missing margin storage placeholder is not fixed zero")
    return {
        "storage_placeholder": 0,
        "storage_placeholder_interpretation": "nonsemantic; validity mask mandatory",
        "invalid_targets_excluded_from_each_concept_probe": True,
        "invalid_gold_concepts_never_enter_diagnosis_heads": True,
        "shape_missing_breasts": int(np.sum(~shape_valid)),
        "margin_missing_breasts": int(np.sum(~margin_valid)),
        "both_missing_breasts": int(np.sum(~shape_valid & ~margin_valid)),
        "incomplete_breasts": int(np.sum(~(shape_valid & margin_valid))),
    }


def all_view_mapping(cohort: CbisCohort) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in range(len(cohort.unit_id)):
        for view in range(2):
            present = bool(cohort.view_present[row, view])
            image_id = str(cohort.image_id[row, view])
            path = str(cohort.local_dicom_path[row, view])
            if present:
                if not image_id or not path or image_id in mapping:
                    raise ValueError("cohort view keys are empty or duplicated")
                mapping[image_id] = path
            elif image_id or path:
                raise ValueError("absent cohort view retains an image key or path")
    return mapping


def _release_files(config: Mapping[str, Any], root: Path) -> dict[str, Path]:
    release = config.get("embedding_release")
    if not isinstance(release, Mapping):
        raise ValueError("embedding release mapping is missing")
    files = release.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("embedding release file map is missing")
    required = {
        "view_embeddings",
        "embedding_audit",
        "per_view_receipts",
        "output_manifest",
        "flattened_view_manifest",
    }
    if set(files) != required:
        raise ValueError("embedding release file roles changed")
    resolved: dict[str, Path] = {}
    for role, specification in files.items():
        if not isinstance(specification, Mapping):
            raise ValueError(f"embedding release role {role} is malformed")
        path = safe_path(root, specification.get("path"))
        expected = specification.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"embedding release role {role} lacks a SHA-256")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"embedding release hash mismatch for {role}")
        resolved[str(role)] = path
    return resolved


def _manifest_artifact_hashes(payload: Mapping[str, Any]) -> dict[str, str]:
    raw = payload.get("artifacts", payload.get("files"))
    if not isinstance(raw, Mapping):
        raise ValueError("embedding OUTPUT_MANIFEST has no artifacts")
    output: dict[str, str] = {}
    for name, value in raw.items():
        if isinstance(value, Mapping):
            digest = value.get("sha256")
        else:
            digest = value
        if isinstance(digest, str) and len(digest) == 64:
            output[str(name)] = digest
    return output


def validate_embedding_release(
    root: Path,
    config: Mapping[str, Any],
    full_cohort: CbisCohort,
) -> tuple[np.ndarray, dict[str, object]]:
    """Validate a published embedding release and aggregate views by breast."""

    files = _release_files(config, root)
    release = config["embedding_release"]
    output_manifest = read_json(files["output_manifest"])
    if (
        output_manifest.get("schema_version")
        != "paper-b-cbis-full-mammogram-embedding-output-v2"
        or output_manifest.get("status") != "PASS_ATOMIC_NO_OVERWRITE"
        or set(output_manifest.get("files", {}))
        != {"view_embeddings.npz", "embedding_audit.json", "per_view_receipts.csv"}
    ):
        raise ValueError("embedding OUTPUT_MANIFEST contract changed")
    output_hashes = _manifest_artifact_hashes(output_manifest)
    for role in ("view_embeddings", "embedding_audit", "per_view_receipts"):
        name = files[role].name
        expected = str(release["files"][role]["sha256"])
        if output_hashes.get(name) != expected:
            raise ValueError(f"embedding OUTPUT_MANIFEST does not bind {name}")

    audit = read_json(files["embedding_audit"])
    source_protocol = release.get("source_protocol")
    source_freeze = release.get("source_freeze_manifest")
    if not isinstance(source_protocol, Mapping) or not isinstance(source_freeze, Mapping):
        raise ValueError("embedding source protocol/freeze bindings are missing")
    source_protocol_path = safe_path(root, source_protocol.get("path"))
    source_freeze_path = safe_path(root, source_freeze.get("path"))
    if sha256_file(source_protocol_path) != source_protocol.get("sha256"):
        raise ValueError("embedding source protocol hash changed")
    if sha256_file(source_freeze_path) != source_freeze.get("sha256"):
        raise ValueError("embedding source freeze hash changed")
    exact_audit = {
        "schema_version": "paper-b-cbis-full-mammogram-embedding-audit-v2",
        "status": "PASS_FORMAL_X_ONLY_EXTRACTION",
        "official_test_assets_accessed": False,
        "scientific_outcomes_computed": False,
        "input_view_count": 1140,
        "output_view_count": 1140,
        "unique_input_key_count": 1140,
        "unique_output_key_count": 1140,
        "embedding_dtype": "float32",
        "embedding_shape": [1140, 512],
        "full_replay_view_count": 1140,
        "full_replay_max_abs_difference": 0.0,
        "all_finite": True,
    }
    if any(audit.get(key) != value for key, value in exact_audit.items()):
        raise ValueError("embedding audit exact pass contract changed")
    if (
        audit.get("model_name") != release.get("model_name")
        or audit.get("checkpoint_sha256") != release.get("checkpoint_sha256")
        or audit.get("full_replay_max_abs_tolerance") != 1e-6
        or float(audit.get("l2_norm_max_abs_error", np.inf)) > 5e-6
        or audit.get("ordered_source_digest_sha256")
        != "46098c77638401ca098d95a7d95fdf7bbcab018d79504a2b5d004fa198aee655"
        or audit.get("access_assertions")
        != {
            "formal_input_manifest_only_contains_key_path_hash_and_bytes": True,
            "legacy_cache_or_archive_used_as_formal_input": False,
            "mass_test_metadata_pixels_labels_or_counts_loaded": False,
            "pathology_concepts_folds_or_outcomes_loaded": False,
            "roi_pixels_or_masks_loaded": False,
        }
    ):
        raise ValueError("embedding audit replay/access/lineage contract changed")
    if audit.get("config_sha256") != source_protocol.get("sha256"):
        raise ValueError("embedding audit is not bound to its source protocol")
    if audit.get("freeze_manifest_sha256") != source_freeze.get("sha256"):
        raise ValueError("embedding audit is not bound to its source freeze")
    flattened_sha = str(release.get("flattened_view_manifest_sha256", ""))
    cohort_sha = str(release.get("cohort_manifest_sha256", ""))
    if len(flattened_sha) != 64 or len(cohort_sha) != 64:
        raise ValueError("embedding release lineage hashes are incomplete")
    if sha256_file(files["flattened_view_manifest"]) != flattened_sha:
        raise ValueError("flattened view manifest hash changed")
    if str(audit.get("input_manifest_sha256")) != flattened_sha:
        raise ValueError("embedding audit is not bound to the flattened view manifest")
    cohort_manifest = safe_path(root, config["cohort"]["manifest"])
    if sha256_file(cohort_manifest) != cohort_sha:
        raise ValueError("embedding release cohort manifest hash changed")

    expected_views = all_view_mapping(full_cohort)
    with files["flattened_view_manifest"].open(newline="", encoding="utf-8") as handle:
        flattened_rows = list(csv.DictReader(handle))
    flattened_columns = {
        "image_id",
        "local_dicom_path",
        "source_dicom_sha256",
        "source_dicom_bytes",
    }
    if len(flattened_rows) != 1140 or set(flattened_rows[0] if flattened_rows else ()) != flattened_columns:
        raise ValueError("flattened embedding input manifest schema changed")
    flattened_ids = [row["image_id"] for row in flattened_rows]
    if flattened_ids != sorted(flattened_ids) or len(set(flattened_ids)) != 1140:
        raise ValueError("flattened embedding input keys are not unique lexicographic order")
    flattened_by_id = {row["image_id"]: row for row in flattened_rows}
    if set(flattened_by_id) != set(expected_views):
        raise ValueError("flattened embedding input keys differ from frozen cohort")
    for key, expected_path in expected_views.items():
        source = flattened_by_id[key]
        if source["local_dicom_path"] != expected_path:
            raise ValueError("flattened embedding input path differs from frozen cohort")
        if len(source["source_dicom_sha256"]) != 64 or int(source["source_dicom_bytes"]) <= 0:
            raise ValueError("flattened embedding source receipt is malformed")
    with np.load(files["view_embeddings"], allow_pickle=False) as archive:
        required = {
            "image_id",
            "embeddings",
            "l2_norm",
            "local_dicom_path",
            "source_dicom_sha256",
            "model_name",
            "checkpoint_sha256",
        }
        if not required.issubset(archive.files):
            raise ValueError(f"embedding archive lacks {sorted(required - set(archive.files))}")
        image_id = archive["image_id"].astype(str)
        embedding = archive["embeddings"]
        l2_norm = archive["l2_norm"].astype(np.float64)
        local_path = archive["local_dicom_path"].astype(str)
        source_sha = archive["source_dicom_sha256"].astype(str)
        model_name = str(archive["model_name"].item())
        checkpoint_sha = str(archive["checkpoint_sha256"].item())
    if embedding.dtype != np.float32 or embedding.shape != (1140, 512):
        raise ValueError("embedding matrix must be float32[1140,512]")
    if image_id.shape != (1140,) or len(set(image_id.tolist())) != 1140:
        raise ValueError("embedding image keys are not 1140 unique rows")
    if image_id.tolist() != flattened_ids:
        raise ValueError("embedding archive key order differs from flattened manifest")
    if local_path.shape != (1140,) or source_sha.shape != (1140,) or l2_norm.shape != (1140,):
        raise ValueError("embedding lineage arrays have the wrong shape")
    if any(
        flattened_by_id[key]["local_dicom_path"] != path
        or flattened_by_id[key]["source_dicom_sha256"] != digest
        for key, path, digest in zip(image_id, local_path, source_sha, strict=True)
    ):
        raise ValueError("embedding archive path/hash lineage differs from flattened manifest")
    actual_norm = np.linalg.norm(embedding.astype(np.float64), axis=1)
    tolerance = float(release.get("l2_norm_tolerance"))
    if tolerance != 5e-6 or audit.get("l2_norm_tolerance") != tolerance:
        raise ValueError("embedding norm tolerance changed")
    if not np.isfinite(embedding).all() or not np.allclose(actual_norm, 1.0, atol=tolerance, rtol=0.0):
        raise ValueError("view embeddings are non-finite or not L2-normalized")
    if not np.allclose(l2_norm, actual_norm, atol=1e-6, rtol=0.0):
        raise ValueError("saved and recomputed embedding norms differ")
    if model_name != str(release.get("model_name")):
        raise ValueError("embedding model name changed")
    if checkpoint_sha != str(release.get("checkpoint_sha256")):
        raise ValueError("embedding checkpoint hash changed")

    with files["per_view_receipts"].open(newline="", encoding="utf-8") as handle:
        receipt_rows = list(csv.DictReader(handle))
    receipt_required = {"image_id", "local_dicom_path", "source_dicom_sha256"}
    if len(receipt_rows) != 1140 or not receipt_required.issubset(
        receipt_rows[0].keys() if receipt_rows else ()
    ):
        raise ValueError("per-view embedding receipts are incomplete")
    receipt_by_id = {row["image_id"]: row for row in receipt_rows}
    if [row["image_id"] for row in receipt_rows] != flattened_ids or len(receipt_by_id) != 1140:
        raise ValueError("per-view receipt key order differs from flattened manifest")
    archive_by_id = {key: index for index, key in enumerate(image_id.tolist())}
    for key, expected_path in expected_views.items():
        receipt = receipt_by_id[key]
        index = archive_by_id[key]
        source = flattened_by_id[key]
        if (
            receipt["local_dicom_path"] != expected_path
            or receipt["source_dicom_sha256"] != source["source_dicom_sha256"]
            or receipt.get("source_dicom_bytes") != source["source_dicom_bytes"]
            or source_sha[index] != source["source_dicom_sha256"]
            or float(receipt.get("replay_max_abs_difference", "nan")) != 0.0
            or abs(float(receipt.get("normalized_l2_norm", "nan")) - 1.0)
            > tolerance
        ):
            raise ValueError("per-view receipt differs from flattened source lineage")

    by_id = {key: embedding[index].astype(np.float64) for index, key in enumerate(image_id)}
    breast_x = np.empty((len(full_cohort.unit_id), 512), dtype=np.float64)
    for row in range(len(full_cohort.unit_id)):
        vectors = [
            by_id[str(full_cohort.image_id[row, view])]
            for view in range(2)
            if full_cohort.view_present[row, view]
        ]
        if len(vectors) not in (1, 2):
            raise ValueError("a breast has no available view embedding")
        breast_x[row] = np.mean(np.stack(vectors), axis=0)
    if not np.isfinite(breast_x).all():
        raise ValueError("breast-mean embeddings are non-finite")
    return breast_x, {
        "views": 1140,
        "embedding_dimension": 512,
        "model_name": model_name,
        "checkpoint_sha256": checkpoint_sha,
        "flattened_view_manifest_sha256": flattened_sha,
        "cohort_manifest_sha256": cohort_sha,
        "view_embeddings_sha256": sha256_file(files["view_embeddings"]),
        "aggregation": "arithmetic mean over available L2-normalized CC/MLO view vectors",
    }


def subset_x(full: CbisCohort, subset: CbisCohort, full_x: np.ndarray) -> np.ndarray:
    lookup = {unit: index for index, unit in enumerate(full.unit_id.astype(str))}
    return np.stack([full_x[lookup[unit]] for unit in subset.unit_id.astype(str)])


def fold_partition(cohort: CbisCohort, outer_fold: int) -> tuple[np.ndarray, np.ndarray]:
    fit = np.flatnonzero(cohort.outer_fold != int(outer_fold))
    validation = np.flatnonzero(cohort.outer_fold == int(outer_fold))
    if not len(fit) or not len(validation):
        raise ValueError("empty CBIS P3 outer partition")
    if set(cohort.patient_id[fit]).intersection(cohort.patient_id[validation]):
        raise ValueError("patient crosses CBIS P3 outer partition")
    return fit, validation


def class_equal_nll(y: np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    if (
        probability.shape != (len(y), 2)
        or not np.isfinite(probability).all()
        or not np.all(np.isin(y, (0, 1)))
        or np.any(probability < 0.0)
        or np.any(probability > 1.0)
        or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-10, rtol=0.0)
    ):
        raise ValueError("diagnosis probabilities are malformed")
    true_probability = np.clip(
        probability[np.arange(len(y)), y], OUTCOME_LOG_LOSS_CLIP, 1.0
    )
    values = [-np.log(true_probability[y == label]).mean() for label in (0, 1)]
    if any(not np.isfinite(value) for value in values):
        raise ValueError("both diagnosis classes are required for class-equal NLL")
    return float(np.mean(values))


class BalancedBinaryLogistic:
    """Binary L2 logistic regression with fail-closed convergence handling."""

    def __init__(self, c: float, max_iter: int, seed: int) -> None:
        if c <= 0.0 or max_iter < 1:
            raise ValueError("invalid logistic configuration")
        self.c = float(c)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.model: LogisticRegression | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BalancedBinaryLogistic":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all():
            raise ValueError("logistic training arrays are malformed")
        counts = np.bincount(y, minlength=2)
        if len(counts) != 2 or np.any(counts == 0):
            raise ValueError("both binary classes are required")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = LogisticRegression(
                C=self.c,
                penalty="l2",
                solver="lbfgs",
                class_weight="balanced",
                max_iter=self.max_iter,
                random_state=self.seed,
            ).fit(x, y)
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            raise RuntimeError("L-BFGS logistic fit did not converge")
        if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
            raise RuntimeError("L-BFGS logistic parameters are non-finite")
        self.model = model
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("logistic model has not been fit")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("logistic prediction array is malformed")
        probability = self.model.predict_proba(x)
        if (
            list(self.model.classes_) != [0, 1]
            or probability.shape != (len(x), 2)
            or not np.isfinite(probability).all()
            or np.any(probability < 0.0)
            or np.any(probability > 1.0)
            or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-10, rtol=0.0)
        ):
            raise ValueError("binary logistic class layout changed")
        return probability

    @property
    def parameter_count(self) -> int:
        if self.model is None:
            raise RuntimeError("logistic model has not been fit")
        return int(self.model.coef_.size + self.model.intercept_.size)


@dataclass(frozen=True)
class ConceptProbeBundle:
    cohort_variant: str
    concept_names: tuple[str, ...]
    fit_rows: np.ndarray
    fit_patient_ids: frozenset[str]
    models: tuple[BalancedBinaryLogistic, ...]
    manifests: tuple[dict[str, object], ...]


def fit_concept_probes(
    cohort: CbisCohort,
    fit_rows: np.ndarray,
    *,
    probe_c: float,
    max_iter: int,
    seed: int,
    minimum_state_support: int,
) -> ConceptProbeBundle:
    if cohort.x is None:
        raise ValueError("concept probes require breast embeddings")
    fit_rows = np.asarray(fit_rows, dtype=np.int64)
    if not len(fit_rows) or len(set(fit_rows.tolist())) != len(fit_rows):
        raise ValueError("concept probe fit rows are empty or duplicated")
    models: list[BalancedBinaryLogistic] = []
    manifests: list[dict[str, object]] = []
    for column, name in enumerate(cohort.concept_names):
        valid_fit = fit_rows[cohort.concept_valid[fit_rows, column].astype(bool)]
        labels = cohort.concepts[valid_fit, column]
        counts = np.bincount(labels, minlength=2)
        if np.any(counts < int(minimum_state_support)):
            raise ValueError(f"concept {name} lacks frozen minimum state support: {counts.tolist()}")
        model = BalancedBinaryLogistic(
            probe_c,
            max_iter,
            seed_from(seed, "concept", name, sha256_values(cohort.unit_id[valid_fit])),
        ).fit(cohort.x[valid_fit], labels)
        models.append(model)
        manifests.append(
            {
                "concept": name,
                "fit_breasts": int(len(valid_fit)),
                "fit_patients": int(len(set(cohort.patient_id[valid_fit]))),
                "negative": int(counts[0]),
                "positive": int(counts[1]),
                "fit_unit_sha256": sha256_values(cohort.unit_id[valid_fit]),
                "invalid_targets_excluded_by_family_validity_mask": True,
                "fixed_probe_C": float(probe_c),
                "parameter_count": model.parameter_count,
            }
        )
    return ConceptProbeBundle(
        cohort_variant=cohort.variant,
        concept_names=cohort.concept_names,
        fit_rows=fit_rows.copy(),
        fit_patient_ids=frozenset(cohort.patient_id[fit_rows].astype(str).tolist()),
        models=tuple(models),
        manifests=tuple(manifests),
    )


def predict_concept_probabilities(
    bundle: ConceptProbeBundle,
    cohort: CbisCohort,
    predict_rows: np.ndarray,
    *,
    prediction_role: str,
    require_patient_disjoint: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    if cohort.x is None:
        raise ValueError("concept probes require breast embeddings")
    if bundle.cohort_variant != cohort.variant or bundle.concept_names != cohort.concept_names:
        raise ValueError("concept probe bundle belongs to another cohort contract")
    predict_rows = np.asarray(predict_rows, dtype=np.int64)
    if not len(predict_rows) or len(set(predict_rows.tolist())) != len(predict_rows):
        raise ValueError("concept prediction rows are empty or duplicated")
    predict_patients = set(cohort.patient_id[predict_rows].astype(str).tolist())
    overlap = bundle.fit_patient_ids.intersection(predict_patients)
    if require_patient_disjoint and overlap:
        raise ValueError("case-excluded concept prediction patients overlap probe fit")
    if not require_patient_disjoint:
        if prediction_role not in {
            "IN_SAMPLE_AUDIT_TRAIN_EXPOSURE",
            "FINAL_IN_SAMPLE_AUDIT_TRAIN_EXPOSURE",
        }:
            raise ValueError("fit/predict overlap is only allowed for named in-sample audits")
        if not overlap:
            raise ValueError("named in-sample prediction unexpectedly has no fit overlap")
    probability = np.column_stack(
        [
            model.predict_proba(cohort.x[predict_rows])[:, 1]
            for model in bundle.models
        ]
    )
    if not np.isfinite(probability).all() or np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("concept probabilities are invalid")
    return probability, {
        "prediction_role": prediction_role,
        "predict_breasts": int(len(predict_rows)),
        "predict_patients": int(len(predict_patients)),
        "predict_unit_sha256": sha256_values(cohort.unit_id[predict_rows]),
        "fit_predict_patient_disjoint": not bool(overlap),
        "patient_disjoint_required": bool(require_patient_disjoint),
        "overlap_patient_count": int(len(overlap)),
    }


def fit_concept_probabilities(
    cohort: CbisCohort,
    fit_rows: np.ndarray,
    predict_rows: np.ndarray,
    *,
    probe_c: float,
    max_iter: int,
    seed: int,
    minimum_state_support: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Strict convenience wrapper for genuinely case-excluded predictions."""

    bundle = fit_concept_probes(
        cohort,
        fit_rows,
        probe_c=probe_c,
        max_iter=max_iter,
        seed=seed,
        minimum_state_support=minimum_state_support,
    )
    probability, prediction_manifest = predict_concept_probabilities(
        bundle,
        cohort,
        predict_rows,
        prediction_role="CASE_EXCLUDED_PREDICTED_CONCEPT_EXPOSURE",
        require_patient_disjoint=True,
    )
    return probability, [*bundle.manifests, prediction_manifest]


def raw_soft(value: np.ndarray, epsilon: float) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("raw concept probability is malformed")
    return np.clip(value, epsilon, 1.0 - epsilon)


def raw_hard(value: np.ndarray, epsilon: float) -> np.ndarray:
    return (raw_soft(value, epsilon) >= 0.5).astype(np.float64)


def reference_log_odds(value: np.ndarray, epsilon: float) -> np.ndarray:
    probability = raw_soft(value, epsilon)
    output = np.log(probability) - np.log1p(-probability)
    if not np.isfinite(output).all():
        raise ValueError("reference log odds are non-finite")
    return output


@dataclass(frozen=True)
class SelectionPanel:
    heldout_fold: int
    fit_rows: np.ndarray
    heldout_rows: np.ndarray
    nested_oof_fit_probability: np.ndarray
    insample_fit_probability: np.ndarray
    heldout_probability: np.ndarray
    manifests: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class OuterConceptPanel:
    outer_fold: int
    fit_rows: np.ndarray
    validation_rows: np.ndarray
    selection: tuple[SelectionPanel, ...]
    final_oof_probability: np.ndarray
    final_insample_probability: np.ndarray
    validation_probability: np.ndarray
    manifests: tuple[dict[str, object], ...]


def build_outer_concept_panel(
    cohort: CbisCohort,
    outer_fold: int,
    *,
    probe_c: float,
    max_iter: int,
    seed: int,
    minimum_state_support: int,
) -> OuterConceptPanel:
    fit_rows, validation_rows = fold_partition(cohort, outer_fold)
    meta_folds = sorted(set(cohort.outer_fold[fit_rows].astype(int).tolist()))
    if meta_folds != [fold for fold in range(5) if fold != outer_fold]:
        raise ValueError("CBIS P3 meta-fold layout changed")
    selection_panels: list[SelectionPanel] = []
    final_oof = np.full((len(fit_rows), len(cohort.concept_names)), np.nan)
    fit_position = {int(row): position for position, row in enumerate(fit_rows)}
    all_manifests: list[dict[str, object]] = []
    for heldout_fold in meta_folds:
        heldout = fit_rows[cohort.outer_fold[fit_rows] == heldout_fold]
        meta_fit = fit_rows[cohort.outer_fold[fit_rows] != heldout_fold]
        if set(cohort.patient_id[meta_fit]).intersection(cohort.patient_id[heldout]):
            raise ValueError("meta-heldout patients entered concept-probe fit")
        meta_bundle = fit_concept_probes(
            cohort,
            meta_fit,
            probe_c=probe_c,
            max_iter=max_iter,
            seed=seed,
            minimum_state_support=minimum_state_support,
        )
        insample_fit, insample_manifest = predict_concept_probabilities(
            meta_bundle,
            cohort,
            meta_fit,
            prediction_role="IN_SAMPLE_AUDIT_TRAIN_EXPOSURE",
            require_patient_disjoint=False,
        )
        heldout_probability, heldout_manifest = predict_concept_probabilities(
            meta_bundle,
            cohort,
            heldout,
            prediction_role="DIAGNOSIS_SELECTION_VALIDATION_EXPOSURE",
            require_patient_disjoint=True,
        )
        final_oof[[fit_position[int(row)] for row in heldout]] = heldout_probability

        nested_oof = np.full((len(meta_fit), len(cohort.concept_names)), np.nan)
        meta_position = {int(row): position for position, row in enumerate(meta_fit)}
        nested_manifests: list[dict[str, object]] = []
        nested_folds = sorted(set(cohort.outer_fold[meta_fit].astype(int).tolist()))
        if len(nested_folds) != 3:
            raise ValueError("nested selection must retain three patient folds")
        for probe_heldout_fold in nested_folds:
            probe_validation = meta_fit[
                cohort.outer_fold[meta_fit] == probe_heldout_fold
            ]
            probe_fit = meta_fit[cohort.outer_fold[meta_fit] != probe_heldout_fold]
            predicted, manifests = fit_concept_probabilities(
                cohort,
                probe_fit,
                probe_validation,
                probe_c=probe_c,
                max_iter=max_iter,
                seed=seed,
                minimum_state_support=minimum_state_support,
            )
            nested_oof[[meta_position[int(row)] for row in probe_validation]] = predicted
            nested_manifests.append(
                {
                    "diagnosis_selection_heldout_fold": int(heldout_fold),
                    "probe_heldout_fold": int(probe_heldout_fold),
                    "probe_fit_folds": sorted(
                        set(cohort.outer_fold[probe_fit].astype(int).tolist())
                    ),
                    "probe_fit_patient_sha256": sha256_values(cohort.patient_id[probe_fit]),
                    "probe_validation_patient_sha256": sha256_values(
                        cohort.patient_id[probe_validation]
                    ),
                    "diagnosis_validation_excluded_from_probe_fit": not bool(
                        set(cohort.patient_id[probe_fit]).intersection(
                            cohort.patient_id[heldout]
                        )
                    ),
                    "probe_validation_excluded_from_probe_fit": not bool(
                        set(cohort.patient_id[probe_fit]).intersection(
                            cohort.patient_id[probe_validation]
                        )
                    ),
                    "concept_fits": manifests,
                }
            )
        if not np.isfinite(nested_oof).all():
            raise AssertionError("nested OOF concept predictions are incomplete")
        panel_manifest = {
            "outer_fold": int(outer_fold),
            "diagnosis_selection_heldout_fold": int(heldout_fold),
            "fit_folds": sorted(set(cohort.outer_fold[meta_fit].astype(int).tolist())),
            "fit_patient_sha256": sha256_values(cohort.patient_id[meta_fit]),
            "heldout_patient_sha256": sha256_values(cohort.patient_id[heldout]),
            "fit_heldout_patient_disjoint": True,
            "full_meta_fit_concept_fits": list(meta_bundle.manifests),
            "in_sample_prediction": insample_manifest,
            "case_excluded_heldout_prediction": heldout_manifest,
            "nested_probe_panels": nested_manifests,
        }
        all_manifests.append(panel_manifest)
        selection_panels.append(
            SelectionPanel(
                heldout_fold=heldout_fold,
                fit_rows=meta_fit,
                heldout_rows=heldout,
                nested_oof_fit_probability=nested_oof,
                insample_fit_probability=insample_fit,
                heldout_probability=heldout_probability,
                manifests=(panel_manifest,),
            )
        )
    if not np.isfinite(final_oof).all():
        raise AssertionError("final OOF concept predictions are incomplete")
    final_bundle = fit_concept_probes(
        cohort,
        fit_rows,
        probe_c=probe_c,
        max_iter=max_iter,
        seed=seed,
        minimum_state_support=minimum_state_support,
    )
    final_insample, final_insample_manifest = predict_concept_probabilities(
        final_bundle,
        cohort,
        fit_rows,
        prediction_role="FINAL_IN_SAMPLE_AUDIT_TRAIN_EXPOSURE",
        require_patient_disjoint=False,
    )
    validation_probability, final_validation_manifest = predict_concept_probabilities(
        final_bundle,
        cohort,
        validation_rows,
        prediction_role="FINAL_OUTER_VALIDATION_EXPOSURE",
        require_patient_disjoint=True,
    )
    all_manifests.append(
        {
            "outer_fold": int(outer_fold),
            "role": "final_outer_train_probe",
            "fit_patient_sha256": sha256_values(cohort.patient_id[fit_rows]),
            "validation_patient_sha256": sha256_values(cohort.patient_id[validation_rows]),
            "patient_disjoint": True,
            "concept_fits": list(final_bundle.manifests),
            "in_sample_prediction": final_insample_manifest,
            "case_excluded_validation_prediction": final_validation_manifest,
        }
    )
    return OuterConceptPanel(
        outer_fold=outer_fold,
        fit_rows=fit_rows,
        validation_rows=validation_rows,
        selection=tuple(selection_panels),
        final_oof_probability=final_oof,
        final_insample_probability=final_insample,
        validation_probability=validation_probability,
        manifests=tuple(all_manifests),
    )


def _interface_transform(model: str, value: np.ndarray, epsilon: float) -> np.ndarray:
    if model in (OOF_RAW_SOFT, INSAMPLE_RAW_SOFT, GOLD_TO_PREDICTED):
        return raw_soft(value, epsilon)
    if model == OOF_RAW_HARD:
        return raw_hard(value, epsilon)
    if model == OOF_RAW_LOG_ODDS:
        return reference_log_odds(value, epsilon)
    raise ValueError(f"no predicted-concept transform for {model}")


def select_regularization(
    cohort: CbisCohort,
    panel: OuterConceptPanel,
    model: str,
    c_grid: Sequence[float],
    *,
    epsilon: float,
    max_iter: int,
    seed: int,
) -> tuple[float, list[dict[str, object]]]:
    if cohort.x is None:
        raise ValueError("diagnosis selection requires breast embeddings")
    records: list[dict[str, object]] = []
    outer_validation_patients = set(
        cohort.patient_id[panel.validation_rows].astype(str).tolist()
    )
    for c_value in c_grid:
        fold_values: list[float] = []
        for selection in panel.selection:
            fit_patients = set(cohort.patient_id[selection.fit_rows].astype(str).tolist())
            heldout_patients = set(
                cohort.patient_id[selection.heldout_rows].astype(str).tolist()
            )
            if (
                fit_patients.intersection(heldout_patients)
                or fit_patients.intersection(outer_validation_patients)
                or heldout_patients.intersection(outer_validation_patients)
                or set(selection.fit_rows).union(selection.heldout_rows)
                != set(panel.fit_rows)
            ):
                raise AssertionError("diagnosis C selection accessed an excluded patient/row")
            if model in (OOF_RAW_SOFT, OOF_RAW_HARD, OOF_RAW_LOG_ODDS):
                x_fit = _interface_transform(
                    model, selection.nested_oof_fit_probability, epsilon
                )
                x_validation = _interface_transform(
                    model, selection.heldout_probability, epsilon
                )
            elif model == INSAMPLE_RAW_SOFT:
                x_fit = raw_soft(selection.insample_fit_probability, epsilon)
                x_validation = raw_soft(selection.heldout_probability, epsilon)
            elif model == GOLD_ORACLE:
                x_fit = cohort.concepts[selection.fit_rows].astype(np.float64)
                x_validation = cohort.concepts[selection.heldout_rows].astype(np.float64)
            elif model == DIRECT_X:
                x_fit = cohort.x[selection.fit_rows]
                x_validation = cohort.x[selection.heldout_rows]
            else:
                raise ValueError(f"model is not independently selected: {model}")
            head = BalancedBinaryLogistic(
                float(c_value),
                max_iter,
                seed_from(
                    seed,
                    "diagnosis_selection",
                    cohort.variant,
                    panel.outer_fold,
                    selection.heldout_fold,
                    model,
                    c_value,
                ),
            ).fit(x_fit, cohort.y[selection.fit_rows])
            probability = head.predict_proba(x_validation)
            fold_values.append(class_equal_nll(cohort.y[selection.heldout_rows], probability))
        records.append(
            {
                "variant": cohort.variant,
                "outer_fold": int(panel.outer_fold),
                "model": model,
                "C": float(c_value),
                "mean_meta_fold_class_equal_nll": float(np.mean(fold_values)),
                "meta_fold_class_equal_nll": fold_values,
                "selection_folds": [
                    int(selection.heldout_fold) for selection in panel.selection
                ],
                "outer_validation_accessed": False,
                "feature_source": (
                    "DIRECT_BREAST_MEAN_X"
                    if model == DIRECT_X
                    else "GOLD_CONCEPTS"
                    if model == GOLD_ORACLE
                    else "IN_SAMPLE_BALANCED_PROBE_SCORES"
                    if model == INSAMPLE_RAW_SOFT
                    else "NESTED_CASE_EXCLUDED_BALANCED_PROBE_SCORES"
                ),
            }
        )
    best = min(
        records,
        key=lambda row: (
            float(row["mean_meta_fold_class_equal_nll"]),
            float(row["C"]),
        ),
    )
    return float(best["C"]), records


def _prediction_records(
    cohort: CbisCohort,
    rows: np.ndarray,
    probability: np.ndarray,
    model: str,
) -> list[dict[str, object]]:
    return [
        {
            "cohort_variant": cohort.variant,
            "unit_id": str(cohort.unit_id[row]),
            "patient_id": str(cohort.patient_id[row]),
            "outer_fold": int(cohort.outer_fold[row]),
            "label": int(cohort.y[row]),
            "model": model,
            "p0": float(probability[local, 0]),
            "p1": float(probability[local, 1]),
        }
        for local, row in enumerate(rows)
    ]


def _concept_exposure_records(
    cohort: CbisCohort,
    rows: np.ndarray,
    probability: np.ndarray,
    *,
    outer_fold: int,
    exposure_role: str,
    case_excluded: bool,
) -> list[dict[str, object]]:
    if probability.shape != (len(rows), len(cohort.concept_names)):
        raise ValueError("concept exposure matrix shape changed")
    return [
        {
            "cohort_variant": cohort.variant,
            "outer_fold": int(outer_fold),
            "exposure_role": exposure_role,
            "case_excluded": bool(case_excluded),
            "unit_id": str(cohort.unit_id[row]),
            "patient_id": str(cohort.patient_id[row]),
            "unit_fold": int(cohort.outer_fold[row]),
            "concept": name,
            "gold_label": int(cohort.concepts[row, column]),
            "gold_valid": int(cohort.concept_valid[row, column]),
            "balanced_probe_score": float(probability[local, column]),
        }
        for local, row in enumerate(rows)
        for column, name in enumerate(cohort.concept_names)
    ]


def fit_variant(
    cohort: CbisCohort,
    config: Mapping[str, Any],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    if cohort.x is None:
        raise ValueError("scientific fit requires X embeddings")
    probe = config["concept_probe"]
    head_config = config["diagnosis_head"]
    probe_c = float(probe["C"])
    max_iter = int(head_config["max_iter"])
    minimum_support = int(probe["minimum_state_support_per_fit"])
    c_grid = tuple(float(value) for value in head_config["c_grid"])
    epsilon = float(head_config["probability_clip"])
    base_seed = int(config["seed"])
    models = PRIMARY_MODELS if cohort.variant == PRIMARY_VARIANT else SENSITIVITY_MODELS
    predictions: list[dict[str, object]] = []
    selection_records: list[dict[str, object]] = []
    fit_manifests: list[dict[str, object]] = []
    concept_exposures: list[dict[str, object]] = []
    for outer_fold in range(5):
        panel = build_outer_concept_panel(
            cohort,
            outer_fold,
            probe_c=probe_c,
            max_iter=int(probe["max_iter"]),
            seed=base_seed,
            minimum_state_support=minimum_support,
        )
        selected: dict[str, float] = {}
        for model in models:
            if model == GOLD_TO_PREDICTED:
                continue
            selected[model], rows = select_regularization(
                cohort,
                panel,
                model,
                c_grid,
                epsilon=epsilon,
                max_iter=max_iter,
                seed=base_seed,
            )
            selection_records.extend(rows)

        fit_rows, validation_rows = panel.fit_rows, panel.validation_rows
        concept_exposures.extend(
            _concept_exposure_records(
                cohort,
                fit_rows,
                panel.final_oof_probability,
                outer_fold=outer_fold,
                exposure_role="OUTER_TRAIN_CASE_EXCLUDED_OOF",
                case_excluded=True,
            )
        )
        concept_exposures.extend(
            _concept_exposure_records(
                cohort,
                fit_rows,
                panel.final_insample_probability,
                outer_fold=outer_fold,
                exposure_role="OUTER_TRAIN_IN_SAMPLE_AUDIT",
                case_excluded=False,
            )
        )
        concept_exposures.extend(
            _concept_exposure_records(
                cohort,
                validation_rows,
                panel.validation_probability,
                outer_fold=outer_fold,
                exposure_role="OUTER_VALIDATION_CASE_EXCLUDED",
                case_excluded=True,
            )
        )
        fitted_heads: dict[str, BalancedBinaryLogistic] = {}
        for model in models:
            if model == GOLD_TO_PREDICTED:
                continue
            if model in (OOF_RAW_SOFT, OOF_RAW_HARD, OOF_RAW_LOG_ODDS):
                x_fit = _interface_transform(model, panel.final_oof_probability, epsilon)
                x_validation = _interface_transform(model, panel.validation_probability, epsilon)
            elif model == INSAMPLE_RAW_SOFT:
                x_fit = raw_soft(panel.final_insample_probability, epsilon)
                x_validation = raw_soft(panel.validation_probability, epsilon)
            elif model == GOLD_ORACLE:
                x_fit = cohort.concepts[fit_rows].astype(np.float64)
                x_validation = cohort.concepts[validation_rows].astype(np.float64)
            elif model == DIRECT_X:
                x_fit = cohort.x[fit_rows]
                x_validation = cohort.x[validation_rows]
            else:
                raise AssertionError(model)
            head = BalancedBinaryLogistic(
                selected[model],
                max_iter,
                seed_from(base_seed, "diagnosis_final", cohort.variant, outer_fold, model),
            ).fit(x_fit, cohort.y[fit_rows])
            expected_parameters = 513 if model == DIRECT_X else 11
            if head.parameter_count != expected_parameters:
                raise AssertionError(
                    f"{model} parameter count changed: {head.parameter_count}"
                )
            fitted_heads[model] = head
            predictions.extend(
                _prediction_records(
                    cohort, validation_rows, head.predict_proba(x_validation), model
                )
            )
        if GOLD_TO_PREDICTED in models:
            oracle_head = fitted_heads[GOLD_ORACLE]
            mismatch_probability = oracle_head.predict_proba(
                raw_soft(panel.validation_probability, epsilon)
            )
            predictions.extend(
                _prediction_records(
                    cohort, validation_rows, mismatch_probability, GOLD_TO_PREDICTED
                )
            )
        fit_manifests.append(
            {
                "cohort_variant": cohort.variant,
                "outer_fold": int(outer_fold),
                "fit_breasts": int(len(fit_rows)),
                "validation_breasts": int(len(validation_rows)),
                "fit_patients": int(len(set(cohort.patient_id[fit_rows]))),
                "validation_patients": int(len(set(cohort.patient_id[validation_rows]))),
                "fit_validation_patient_disjoint": True,
                "selected_C": selected,
                "parameter_count": {
                    model: fitted_heads[model].parameter_count for model in fitted_heads
                },
                "gold_to_predicted_shares_exact_oracle_head": GOLD_TO_PREDICTED in models,
                "concept_panels": list(panel.manifests),
            }
        )
    expected = len(cohort.unit_id) * len(models)
    keys = {
        (row["cohort_variant"], row["unit_id"], row["model"]) for row in predictions
    }
    if len(predictions) != expected or len(keys) != expected:
        raise AssertionError("CBIS P3 prediction census is incomplete or duplicated")
    exposure_keys = {
        (
            row["cohort_variant"],
            row["outer_fold"],
            row["exposure_role"],
            row["unit_id"],
            row["concept"],
        )
        for row in concept_exposures
    }
    if len(exposure_keys) != len(concept_exposures):
        raise AssertionError("CBIS P3 concept exposure audit rows are duplicated")
    return predictions, selection_records, fit_manifests, concept_exposures


def _prediction_arrays(rows: Sequence[Mapping[str, object]]) -> tuple[np.ndarray, ...]:
    unit = np.asarray([str(row["unit_id"]) for row in rows])
    patient = np.asarray([str(row["patient_id"]) for row in rows])
    y = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    probability = np.asarray(
        [[float(row["p0"]), float(row["p1"])] for row in rows], dtype=np.float64
    )
    if len(set(unit.tolist())) != len(unit):
        raise ValueError("prediction rows duplicate a breast unit")
    if (
        not np.isfinite(probability).all()
        or np.any(probability < 0.0)
        or np.any(probability > 1.0)
        or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-10, rtol=0.0)
    ):
        raise ValueError("prediction rows are not finite probability simplexes")
    return patient, y, probability


def class_equal_ece(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    predicted = (probability[:, 1] >= 0.5).astype(np.int64)
    confidence = probability[np.arange(len(y)), predicted]
    correct = (predicted == y).astype(np.float64)
    counts = np.bincount(y, minlength=2).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("ECE requires both diagnosis classes")
    weight = np.asarray([1.0 / (2.0 * counts[label]) for label in y])
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignment = np.minimum(np.searchsorted(edges, confidence, side="right") - 1, bins - 1)
    value = 0.0
    for index in range(bins):
        selected = assignment == index
        mass = weight[selected].sum()
        if mass:
            accuracy = np.average(correct[selected], weights=weight[selected])
            average_confidence = np.average(confidence[selected], weights=weight[selected])
            value += float(mass * abs(accuracy - average_confidence))
    return float(value)


def metric_vector(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability[:, 1] >= 0.5).astype(np.int64)
    return {
        "class_equal_nll": class_equal_nll(y, probability),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability[:, 1])),
        "class_equal_top_label_ece10": class_equal_ece(y, probability, bins=10),
    }


def patient_cluster_strata(patient_id: np.ndarray, y: np.ndarray) -> dict[str, list[str]]:
    patient_id = np.asarray(patient_id).astype(str)
    y = np.asarray(y, dtype=np.int64)
    if len(patient_id) != len(y) or not np.all(np.isin(y, (0, 1))):
        raise ValueError("bootstrap patient/label arrays are malformed")
    strata: dict[str, list[str]] = {}
    for patient in sorted(set(patient_id.tolist())):
        labels = y[patient_id == patient]
        signature = (
            f"breasts_{len(labels)}__benign_{int(np.sum(labels == 0))}"
            f"__malignant_{int(np.sum(labels == 1))}"
        )
        strata.setdefault(signature, []).append(patient)
    return strata


def patient_bootstrap_indices(
    patient_id: np.ndarray, y: np.ndarray, draws: int, seed: int
) -> Iterable[np.ndarray]:
    patient_id = np.asarray(patient_id).astype(str)
    y = np.asarray(y, dtype=np.int64)
    mapping = {
        patient: np.flatnonzero(patient_id == patient)
        for patient in sorted(set(patient_id.tolist()))
    }
    strata = patient_cluster_strata(patient_id, y)
    rng = np.random.default_rng(seed)
    for _ in range(draws):
        sampled: list[np.ndarray] = []
        for signature in sorted(strata):
            patients = np.asarray(strata[signature])
            selected = rng.choice(patients, size=len(patients), replace=True)
            sampled.extend(mapping[str(patient)] for patient in selected)
        yield np.concatenate(sampled)


def percentile_intervals(values: Sequence[float]) -> dict[str, float]:
    value = np.asarray(values, dtype=np.float64)
    if not len(value) or not np.isfinite(value).all():
        raise ValueError("bootstrap values are empty or non-finite")
    return {
        "ci90_lower": float(np.quantile(value, 0.05, method="linear")),
        "ci90_upper": float(np.quantile(value, 0.95, method="linear")),
        "ci95_lower": float(np.quantile(value, 0.025, method="linear")),
        "ci95_upper": float(np.quantile(value, 0.975, method="linear")),
    }


def summarize_model(
    rows: Sequence[Mapping[str, object]], draws: int, seed: int
) -> dict[str, object]:
    patient, y, probability = _prediction_arrays(rows)
    point = metric_vector(y, probability)
    samples: dict[str, list[float]] = {name: [] for name in point}
    for indices in patient_bootstrap_indices(patient, y, draws, seed):
        if len(np.unique(y[indices])) != 2:
            continue
        value = metric_vector(y[indices], probability[indices])
        for name, number in value.items():
            samples[name].append(number)
    return {
        **point,
        **{
            f"{name}_{bound}": number
            for name, values in samples.items()
            for bound, number in percentile_intervals(values).items()
        },
        "bootstrap_draws": int(min(len(values) for values in samples.values())),
        "bootstrap_cluster": "patient_id",
        "bootstrap_strata": "patient breast-count and benign/malignant-count signature",
        "interval_scope": "conditional_on_fixed_oof_prediction_panel_no_model_refitting",
    }


def _paired_comparison_inputs(
    candidate_rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidate = {str(row["unit_id"]): row for row in candidate_rows}
    baseline = {str(row["unit_id"]): row for row in baseline_rows}
    if (
        set(candidate) != set(baseline)
        or len(candidate) != len(candidate_rows)
        or len(baseline) != len(baseline_rows)
    ):
        raise ValueError("paired comparison unit keys differ or duplicate")
    units = sorted(candidate)
    patient = np.asarray([str(candidate[unit]["patient_id"]) for unit in units])
    y = np.asarray([int(candidate[unit]["label"]) for unit in units], dtype=np.int64)
    if any(
        int(baseline[unit]["label"]) != label
        or str(baseline[unit]["patient_id"]) != patient_id
        for unit, label, patient_id in zip(units, y, patient, strict=True)
    ):
        raise ValueError("paired comparison labels or patient IDs differ")
    candidate_probability = np.asarray(
        [[float(candidate[unit]["p0"]), float(candidate[unit]["p1"])] for unit in units]
    )
    baseline_probability = np.asarray(
        [[float(baseline[unit]["p0"]), float(baseline[unit]["p1"])] for unit in units]
    )
    for probability in (candidate_probability, baseline_probability):
        if (
            not np.isfinite(probability).all()
            or np.any(probability < 0.0)
            or np.any(probability > 1.0)
            or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-10, rtol=0.0)
        ):
            raise ValueError("paired comparison contains invalid probabilities")
    return patient, y, candidate_probability, baseline_probability


def _class_equal_log_score_gain(
    y: np.ndarray,
    candidate_probability: np.ndarray,
    baseline_probability: np.ndarray,
    indices: np.ndarray,
) -> float:
    selected_y = y[indices]
    if len(np.unique(selected_y)) != 2:
        raise ValueError("class-equal paired gain requires both diagnosis classes")
    candidate_true = np.clip(
        candidate_probability[indices, selected_y], OUTCOME_LOG_LOSS_CLIP, 1.0
    )
    baseline_true = np.clip(
        baseline_probability[indices, selected_y], OUTCOME_LOG_LOSS_CLIP, 1.0
    )
    per_breast = np.log(candidate_true) - np.log(baseline_true)
    return float(np.mean([per_breast[selected_y == label].mean() for label in (0, 1)]))


def paired_log_score_gain(
    candidate_rows: Sequence[Mapping[str, object]],
    baseline_rows: Sequence[Mapping[str, object]],
    draws: int,
    seed: int,
) -> dict[str, object]:
    patient, y, candidate_probability, baseline_probability = _paired_comparison_inputs(
        candidate_rows, baseline_rows
    )

    all_rows = np.arange(len(y), dtype=np.int64)
    values = [
        _class_equal_log_score_gain(
            y, candidate_probability, baseline_probability, indices
        )
        for indices in patient_bootstrap_indices(patient, y, draws, seed)
        if len(np.unique(y[indices])) == 2
    ]
    point = _class_equal_log_score_gain(
        y, candidate_probability, baseline_probability, all_rows
    )
    return {
        "class_equal_log_score_gain": point,
        "baseline_class_equal_nll_minus_candidate_class_equal_nll": point,
        **percentile_intervals(values),
        "positive_bootstrap_fraction": float(np.mean(np.asarray(values) > 0.0)),
        "bootstrap_draws": len(values),
        "bootstrap_cluster": "patient_id",
        "bootstrap_strata": "patient breast-count and benign/malignant-count signature",
        "positive_favors": "candidate",
    }


def concept_probe_diagnostics(
    exposures: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for variant in (PRIMARY_VARIANT, SENSITIVITY_VARIANT):
        expected_units = 655 if variant == PRIMARY_VARIANT else 662
        for concept in STABLE_CONCEPTS:
            rows = [
                row
                for row in exposures
                if row["cohort_variant"] == variant
                and row["exposure_role"] == "OUTER_VALIDATION_CASE_EXCLUDED"
                and row["concept"] == concept
            ]
            units = [str(row["unit_id"]) for row in rows]
            if len(rows) != expected_units or len(set(units)) != expected_units:
                raise ValueError("outer-validation concept exposure census changed")
            valid_rows = [row for row in rows if int(row["gold_valid"]) == 1]
            y = np.asarray([int(row["gold_label"]) for row in valid_rows], dtype=np.int64)
            score = np.asarray(
                [float(row["balanced_probe_score"]) for row in valid_rows],
                dtype=np.float64,
            )
            if (
                not len(score)
                or not np.isfinite(score).all()
                or np.any(score < 0.0)
                or np.any(score > 1.0)
                or len(np.unique(y)) != 2
            ):
                raise ValueError("concept diagnostic input is malformed")
            hard = (score >= 0.5).astype(np.int64)
            output.append(
                {
                    "cohort_variant": variant,
                    "concept": concept,
                    "valid_breasts": int(len(y)),
                    "negative_breasts": int(np.sum(y == 0)),
                    "positive_breasts": int(np.sum(y == 1)),
                    "hard_positive_prevalence": float(np.mean(hard)),
                    "balanced_accuracy_at_0_5": float(
                        balanced_accuracy_score(y, hard)
                    ),
                    "roc_auc": float(roc_auc_score(y, score)),
                    "score_semantics": "raw class-balanced probe score; not natural-prevalence calibrated",
                    "role": "DESCRIPTIVE_NON_GATING",
                }
            )
    return output


def analyze_predictions(
    predictions: Sequence[Mapping[str, object]],
    concept_exposures: Sequence[Mapping[str, object]],
    config: Mapping[str, Any],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    statistics = config["statistics"]
    draws = int(statistics["bootstrap_draws"])
    metric_rows: list[dict[str, object]] = []
    for variant in (PRIMARY_VARIANT, SENSITIVITY_VARIANT):
        models = PRIMARY_MODELS if variant == PRIMARY_VARIANT else SENSITIVITY_MODELS
        for model_index, model in enumerate(models):
            rows = [
                row
                for row in predictions
                if row["cohort_variant"] == variant and row["model"] == model
            ]
            metric_rows.append(
                {
                    "cohort_variant": variant,
                    "model": model,
                    **summarize_model(
                        rows,
                        draws,
                        seed_from(int(config["seed"]), "metric", variant, model_index),
                    ),
                }
            )
    comparison_rows: list[dict[str, object]] = []
    for index, specification in enumerate(statistics["comparisons"]):
        variant = str(specification["cohort_variant"])
        candidate_name = str(specification["candidate"])
        baseline_name = str(specification["baseline"])
        candidate = [
            row
            for row in predictions
            if row["cohort_variant"] == variant and row["model"] == candidate_name
        ]
        baseline = [
            row
            for row in predictions
            if row["cohort_variant"] == variant and row["model"] == baseline_name
        ]
        fold_gains: dict[str, float] = {}
        for fold in range(5):
            fold_candidate = [row for row in candidate if int(row["outer_fold"]) == fold]
            fold_baseline = [row for row in baseline if int(row["outer_fold"]) == fold]
            _, fold_y, fold_candidate_probability, fold_baseline_probability = (
                _paired_comparison_inputs(fold_candidate, fold_baseline)
            )
            fold_gains[f"outer_fold_{fold}_class_equal_log_score_gain"] = (
                _class_equal_log_score_gain(
                    fold_y,
                    fold_candidate_probability,
                    fold_baseline_probability,
                    np.arange(len(fold_y), dtype=np.int64),
                )
            )
        comparison_rows.append(
            {
                "comparison_id": str(specification["id"]),
                "cohort_variant": variant,
                "candidate": candidate_name,
                "baseline": baseline_name,
                "role": str(specification["role"]),
                "registered_effect": str(specification["effect"]),
                **paired_log_score_gain(
                    candidate,
                    baseline,
                    draws,
                    seed_from(int(config["seed"]), "comparison", index),
                ),
                **fold_gains,
                "outer_fold_points_role": "descriptive_not_independent_tests",
                "interval_scope": "conditional_on_fixed_oof_prediction_panel_no_model_refitting",
            }
        )
    return metric_rows, comparison_rows, concept_probe_diagnostics(concept_exposures)


def verify_execution_gate(
    root: Path, config_path: Path, config: Mapping[str, Any]
) -> dict[str, object]:
    """Verify every release/freeze/approval binding before any scientific fit."""

    if config.get("status") != FINAL_CONFIG_STATUS:
        raise PermissionError("CBIS P3 config is not finalized for root approval")
    validate_protocol_contract(config, expected_status=FINAL_CONFIG_STATUS)
    freeze_path = safe_path(root, config.get("freeze_manifest"))
    freeze = read_json(freeze_path)
    if (
        freeze.get("schema_version") != "paper-b-cbis-p3-minimal-freeze-v1"
        or freeze.get("protocol_id") != PROTOCOL_ID
        or freeze.get("status") != FREEZE_STATUS
        or freeze.get("official_test_assets_accessed") is not False
        or freeze.get("scientific_models_fitted") != 0
        or freeze.get("scientific_outcomes_computed") is not False
    ):
        raise PermissionError("CBIS P3 experiment freeze is invalid")
    frozen_files = freeze.get("files")
    if not isinstance(frozen_files, Mapping) or not frozen_files:
        raise ValueError("CBIS P3 freeze manifest has no files")
    for raw, expected in frozen_files.items():
        path = safe_path(root, raw)
        if sha256_file(path) != expected:
            raise ValueError(f"frozen CBIS P3 input changed: {raw}")
    relative_config = config_path.resolve().relative_to(root.resolve()).as_posix()
    if frozen_files.get(relative_config) != sha256_file(config_path):
        raise ValueError("CBIS P3 freeze does not bind the active config")
    approval_path = safe_path(root, config.get("root_approval_receipt"))
    approval = read_json(approval_path)
    if (
        approval.get("protocol_id") != PROTOCOL_ID
        or approval.get("status") != APPROVAL_STATUS
        or approval.get("approved") is not True
    ):
        raise PermissionError("root has not approved CBIS P3 outcome execution")
    if approval.get("freeze_manifest") != freeze_path.relative_to(root).as_posix():
        raise ValueError("root approval names a different freeze manifest")
    if approval.get("freeze_manifest_sha256") != sha256_file(freeze_path):
        raise ValueError("root approval does not bind the current freeze manifest")
    runtime = verify_scientific_runtime(config)
    return {
        "freeze_manifest": freeze_path.relative_to(root).as_posix(),
        "freeze_manifest_sha256": sha256_file(freeze_path),
        "root_approval_receipt": approval_path.relative_to(root).as_posix(),
        "root_approval_receipt_sha256": sha256_file(approval_path),
        "observed_runtime": runtime,
    }


def validate_protocol_contract(
    config: Mapping[str, Any], *, expected_status: str
) -> None:
    """Fail closed on every predeclared scientific degree of freedom."""

    if (
        config.get("schema_version") != "paper-b-cbis-p3-minimal-protocol-v1"
        or config.get("protocol_id") != PROTOCOL_ID
        or config.get("status") != expected_status
    ):
        raise ValueError("CBIS P3 protocol identity/status changed")
    access = config.get("access_boundary", {})
    if (
        config.get("official_test_rows_allowed") != 0
        or access.get("no_joint_training") is not True
        or access.get("no_d8") is not True
        or access.get("no_llm") is not True
        or "Mass-Test" not in str(access.get("forbidden"))
    ):
        raise PermissionError("CBIS P3 access or method boundary changed")
    if tuple(config.get("stable_concepts", ())) != STABLE_CONCEPTS:
        raise ValueError("CBIS P3 must retain the exact ten stable concepts in order")

    cohort = config.get("cohort", {})
    if (
        cohort.get("archive")
        != "data/CBIS_DDSM/paper_b_cbis_mass_train_breast_cohort_v1_20260910.npz"
        or cohort.get("archive_sha256")
        != "4d868f37a5daf2fa33f4b833d01da4c1476403b5d6fb157478451b9ce80f7725"
        or cohort.get("manifest")
        != "data_manifests/paper_b_cbis_mass_train_breast_cohort_v1_20260910.csv"
        or cohort.get("manifest_sha256")
        != "70ef2b8603c9d7823c787a4d7ea7187b59edc235eae0a41ae5c0cac807a4f2cf"
        or cohort.get("primary")
        != {"rule": "complete_concept_eligible == 1", "breasts": 655, "patients": 633}
        or cohort.get("missingness_sensitivity")
        != {"rule": "all frozen eligible breasts", "breasts": 662, "patients": 639}
        or cohort.get("split_unit") != "patient_id"
    ):
        raise ValueError("CBIS P3 frozen cohort contract changed")

    release = config.get("embedding_release", {})
    expected_release_hashes = {
        "view_embeddings": "d4a4ce8164d2e3bcb7a84c69c726e7b1c2f007ce3f1ea7811b882702b152e254",
        "embedding_audit": "a975c4b20744acd28a63c95aed243fbcc025a223e4e420671a1c55bf93af713e",
        "per_view_receipts": "749e6aa2ba05f180ee5e99e24803d2a24fc86a1fde88fe3d31845954963e787d",
        "output_manifest": "89223c292ddc4f6a6ba4704f655b881473ce76f61e9a624f60738a9601013773",
        "flattened_view_manifest": "008d9688fddc580380b18c987dd855f438c6569a2f5a0b45587a796ac69b5ba7",
    }
    files = release.get("files", {})
    if (
        release.get("model_name")
        != "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        or release.get("checkpoint_sha256")
        != "52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be"
        or release.get("cohort_manifest_sha256") != cohort.get("manifest_sha256")
        or release.get("flattened_view_manifest_sha256")
        != expected_release_hashes["flattened_view_manifest"]
        or release.get("l2_norm_tolerance") != 5e-6
        or release.get("view_mask_input") is not False
        or set(files) != set(expected_release_hashes)
        or any(
            not isinstance(files.get(role), Mapping)
            or files[role].get("sha256") != digest
            for role, digest in expected_release_hashes.items()
        )
    ):
        raise ValueError("CBIS P3 embedding release binding changed")

    probe = config.get("concept_probe", {})
    if (
        probe.get("model")
        != "ten independent binary L2 logistic regressions with intercept"
        or probe.get("solver") != "lbfgs"
        or probe.get("penalty") != "l2"
        or probe.get("class_weight") != "balanced"
        or probe.get("C") != 1.0
        or probe.get("C_selection") != "none"
        or probe.get("max_iter") != 10000
        or probe.get("minimum_state_support_per_fit") != 10
        or "not a calibrated natural-prevalence" not in str(probe.get("output_semantics"))
        or "before scientific outcomes" not in str(probe.get("C_rationale"))
        or "excluding O, H, and P" not in str(probe.get("selection_isolation"))
        or "OOF_RAW_SOFT_ADDITIVE" not in str(probe.get("case_exclusion"))
        or "INSAMPLE_RAW_SOFT_ADDITIVE" not in str(probe.get("case_exclusion"))
    ):
        raise ValueError("CBIS P3 fixed concept-probe contract changed")

    missing = config.get("missingness_sensitivity", {})
    if (
        missing.get("storage_imputation_value") != 0
        or "nonsemantic" not in str(missing.get("storage_imputation_semantics"))
        or "exclude" not in str(missing.get("validity_mask_rule"))
        or "forbidden" not in str(missing.get("invalid_gold_rule"))
        or [
            missing.get("expected_shape_missing_breasts"),
            missing.get("expected_margin_missing_breasts"),
            missing.get("expected_both_missing_breasts"),
            missing.get("expected_incomplete_breasts"),
        ]
        != [1, 7, 1, 7]
    ):
        raise ValueError("CBIS P3 missing-family placeholder/mask contract changed")

    head = config.get("diagnosis_head", {})
    if (
        head.get("model") != "additive binary L2 logistic regression with intercept"
        or head.get("solver") != "lbfgs"
        or head.get("penalty") != "l2"
        or head.get("class_weight") != "balanced"
        or head.get("c_grid") != [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        or head.get("selection_loss")
        != "mean across four patient-grouped inner folds of breast-level class-equal NLL"
        or head.get("selection_tie_break") != "smallest C"
        or "select C independently" not in str(head.get("per_model_selection"))
        or "exact selected gold-oracle head" not in str(head.get("per_model_selection"))
        or head.get("max_iter") != 10000
        or head.get("probability_clip") != 1e-6
        or head.get("hard_threshold") != 0.5
        or head.get("hard_tie_rule") != "score >= 0.5 maps to 1"
        or head.get("diagnosis_decision_threshold") != 0.5
        or head.get("diagnosis_decision_tie_rule")
        != "p1 >= 0.5 maps to MALIGNANT=1"
        or head.get("reference_log_odds")
        != "log(clip(score))-log(1-clip(score)) with the same 1e-6 clip"
        or head.get("primary_additive_parameter_count") != 11
        or "outer-validation is inaccessible" not in str(head.get("direct_x_selection"))
    ):
        raise ValueError("CBIS P3 diagnosis-head/transform contract changed")
    models = config.get("models", {})
    if (
        tuple(models.get("primary_complete_concept", ())) != PRIMARY_MODELS
        or tuple(models.get("missingness_sensitivity", ())) != SENSITIVITY_MODELS
        or GOLD_ORACLE in SENSITIVITY_MODELS
        or GOLD_TO_PREDICTED in SENSITIVITY_MODELS
    ):
        raise ValueError("CBIS P3 registered model panel changed")

    estimand = config.get("estimand", {})
    if (
        estimand.get("primary_id") != "PRIMARY_RAW_SOFT_VS_HARD"
        or estimand.get("effect") != "NLL(hard additive)-NLL(raw-soft additive)"
        or estimand.get("positive_direction")
        != "raw soft has lower class-equal NLL than hard"
        or "breast-weighted class-equal" not in str(estimand.get("endpoint"))
        or "clipped at 1e-12" not in str(estimand.get("endpoint"))
        or "both breasts together" not in str(estimand.get("bootstrap_cluster"))
        or "never split a bilateral cluster" not in str(estimand.get("bootstrap_strata"))
        or "models are not refit" not in str(estimand.get("uncertainty_scope"))
        or estimand.get("cross_domain_pooling")
        != "No pooling, meta-analysis, or combined estimate with Derm7pt."
    ):
        raise ValueError("CBIS P3 primary sign/aggregation/cluster estimand changed")
    statistics = config.get("statistics", {})
    comparisons = tuple(
        (
            row.get("id"),
            row.get("cohort_variant"),
            row.get("candidate"),
            row.get("baseline"),
            row.get("role"),
            row.get("effect"),
        )
        for row in statistics.get("comparisons", ())
        if isinstance(row, Mapping)
    )
    if (
        statistics.get("bootstrap_draws") != 10000
        or statistics.get("confidence_intervals") != [0.9, 0.95]
        or statistics.get("outcome_log_loss_clip") != OUTCOME_LOG_LOSS_CLIP
        or statistics.get("percentile_quantile_method") != "linear"
        or statistics.get("metrics")
        != [
            "class_equal_nll",
            "balanced_accuracy",
            "macro_f1",
            "roc_auc",
            "class_equal_top_label_ece10",
        ]
        or comparisons != EXPECTED_COMPARISONS
        or statistics.get("multiplicity")
        != "Only PRIMARY_RAW_SOFT_VS_HARD is primary; all other intervals are descriptive secondary, oracle, context, mismatch, exposure, or sensitivity audits."
    ):
        raise ValueError("CBIS P3 statistics/comparison registry changed")

    bilinear = config.get("bilinear_secondary", {})
    if bilinear != {
        "included": False,
        "additive_parameters": 11,
        "standard_rank_one_binary_concept_parameters": 21,
        "reason": "For ten binary concepts, the standard rank-one bilinear form has 1+2*10=21 trainable parameters versus 1+10=11 for the additive head; it is not capacity matched, and no outcome-blind safe 11-parameter interaction restriction is justified.",
    }:
        raise ValueError("unmatched interaction head entered CBIS P3")
    exposures = config.get("auditable_exposures", {})
    if (
        exposures.get("save") is not True
        or tuple(exposures.get("roles", ())) != EXPOSURE_ROLES
        or exposures.get("columns")
        != [
            "cohort_variant",
            "outer_fold",
            "exposure_role",
            "case_excluded",
            "unit_id",
            "patient_id",
            "unit_fold",
            "concept",
            "gold_label",
            "gold_valid",
            "balanced_probe_score",
        ]
        or "cannot gate methods" not in str(exposures.get("diagnostics"))
    ):
        raise ValueError("CBIS P3 auditable exposure contract changed")
    outputs = config.get("outputs", {})
    if (
        outputs.get("overwrite") is not False
        or "renameat2(RENAME_NOREPLACE)" not in str(outputs.get("publication"))
        or outputs.get("directory") == outputs.get("failure_directory")
    ):
        raise ValueError("CBIS P3 atomic output contract changed")
    if config.get("runtime") != {
        "required": {
            "python": "3.11.9",
            "python_implementation": "CPython",
            "python_executable_resolved": "/home/xzf/.conda/envs/upenn_iclr2027_cu121/bin/python3.11",
            "platform": "Linux-5.15.0-113-generic-x86_64-with-glibc2.31",
            "packages": {
                "numpy": "1.26.4",
                "scikit-learn": "1.5.0",
                "scipy": "1.14.0",
                "threadpoolctl": "3.5.0",
            },
        },
        "numeric_library_thread_limit_during_fit_and_analysis": 1,
        "environment_thread_variables_must_be_unset": list(
            THREAD_ENVIRONMENT_VARIABLES
        ),
    }:
        raise ValueError("CBIS P3 scientific runtime contract changed")
    claim_boundary = config.get("claim_boundary", {})
    if claim_boundary != {
        "role": "heavily method-exposed second development domain and directional stress test",
        "forbidden_claims": [
            "independent replication",
            "fresh confirmation",
            "external validation",
            "clinical utility",
            "clinical safety",
            "natural-prevalence calibrated concept probabilities",
            "causal, mechanistic, or concept-faithfulness identification",
            "a pooled or meta-analytic estimate with Derm7pt",
        ],
    }:
        raise ValueError("CBIS P3 claim boundary changed")


def validate_draft_config(config: Mapping[str, Any]) -> None:
    validate_protocol_contract(config, expected_status=DRAFT_STATUS)


def dry_run_audit(root: Path, config_path: Path) -> dict[str, object]:
    config = read_json(config_path)
    validate_draft_config(config)
    cohort_spec = config["cohort"]
    archive_path = safe_path(root, cohort_spec["archive"])
    if sha256_file(archive_path) != cohort_spec["archive_sha256"]:
        raise ValueError("frozen cohort archive hash changed")
    primary = load_cohort(archive_path, config["stable_concepts"], complete_only=True)
    sensitivity = load_cohort(
        archive_path, config["stable_concepts"], complete_only=False
    )

    deepest_support: dict[str, dict[str, int]] = {}
    split_panels = 0
    for cohort in (primary, sensitivity):
        minimum = {name: [10**9, 10**9] for name in cohort.concept_names}
        for outer_fold in range(5):
            outer_fit, outer_validation = fold_partition(cohort, outer_fold)
            for heldout_fold in sorted(set(cohort.outer_fold[outer_fit].astype(int))):
                meta_fit = outer_fit[cohort.outer_fold[outer_fit] != heldout_fold]
                heldout = outer_fit[cohort.outer_fold[outer_fit] == heldout_fold]
                assert not set(cohort.patient_id[meta_fit]).intersection(
                    cohort.patient_id[heldout]
                )
                for probe_heldout_fold in sorted(
                    set(cohort.outer_fold[meta_fit].astype(int))
                ):
                    probe_fit = meta_fit[
                        cohort.outer_fold[meta_fit] != probe_heldout_fold
                    ]
                    probe_validation = meta_fit[
                        cohort.outer_fold[meta_fit] == probe_heldout_fold
                    ]
                    if set(cohort.patient_id[probe_fit]).intersection(
                        set(cohort.patient_id[probe_validation])
                        | set(cohort.patient_id[heldout])
                        | set(cohort.patient_id[outer_validation])
                    ):
                        raise AssertionError("nested dry-run patient isolation failed")
                    for column, name in enumerate(cohort.concept_names):
                        valid = probe_fit[
                            cohort.concept_valid[probe_fit, column].astype(bool)
                        ]
                        counts = np.bincount(
                            cohort.concepts[valid, column], minlength=2
                        )
                        minimum[name][0] = min(minimum[name][0], int(counts[0]))
                        minimum[name][1] = min(minimum[name][1], int(counts[1]))
                    split_panels += 1
        deepest_support[cohort.variant] = {
            f"{name}_minimum_negative": counts[0]
            for name, counts in minimum.items()
        } | {
            f"{name}_minimum_positive": counts[1]
            for name, counts in minimum.items()
        }
    required_support = int(config["concept_probe"]["minimum_state_support_per_fit"])
    if min(value for mapping in deepest_support.values() for value in mapping.values()) < required_support:
        raise ValueError("a stable concept fails the frozen deepest-fit support rule")

    fold_counts = {}
    patient_cluster_audit = {}
    for cohort in (primary, sensitivity):
        fold_counts[cohort.variant] = {
            f"fold_{fold}_benign": int(
                np.sum((cohort.outer_fold == fold) & (cohort.y == 0))
            )
            for fold in range(5)
        } | {
            f"fold_{fold}_malignant": int(
                np.sum((cohort.outer_fold == fold) & (cohort.y == 1))
            )
            for fold in range(5)
        }
        strata = patient_cluster_strata(cohort.patient_id, cohort.y)
        bilateral_patients = {
            patient
            for patient in set(cohort.patient_id.astype(str).tolist())
            if int(np.sum(cohort.patient_id.astype(str) == patient)) == 2
        }
        discordant_bilateral = {
            patient
            for patient in bilateral_patients
            if len(np.unique(cohort.y[cohort.patient_id.astype(str) == patient])) == 2
        }
        patient_cluster_audit[cohort.variant] = {
            "strata_patient_counts": {
                signature: len(patients) for signature, patients in sorted(strata.items())
            },
            "bilateral_patients": len(bilateral_patients),
            "discordant_bilateral_patients": len(discordant_bilateral),
            "bilateral_cluster_split_allowed": False,
        }
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "DRY_RUN_SCHEMA_PASS_EMBEDDING_NOT_LOADED_AWAITING_FREEZE_AND_ROOT_APPROVAL",
        "config": config_path.relative_to(root).as_posix(),
        "config_sha256": sha256_file(config_path),
        "official_test_assets_accessed": False,
        "embedding_loaded": False,
        "embedding_release_hashes_bound": {
            role: specification["sha256"]
            for role, specification in config["embedding_release"]["files"].items()
        },
        "scientific_models_fitted": 0,
        "scientific_prediction_rows": 0,
        "scientific_metrics_computed": 0,
        "cohort": {
            PRIMARY_VARIANT: {
                "breasts": len(primary.unit_id),
                "patients": len(set(primary.patient_id)),
                "models": list(PRIMARY_MODELS),
                "expected_prediction_rows": len(primary.unit_id) * len(PRIMARY_MODELS),
            },
            SENSITIVITY_VARIANT: {
                "breasts": len(sensitivity.unit_id),
                "patients": len(set(sensitivity.patient_id)),
                "models": list(SENSITIVITY_MODELS),
                "expected_prediction_rows": len(sensitivity.unit_id)
                * len(SENSITIVITY_MODELS),
            },
        },
        "fold_counts": fold_counts,
        "patient_cluster_bootstrap_contract": patient_cluster_audit,
        "missingness_contract": missingness_contract_audit(sensitivity),
        "deepest_nested_probe_fit_support": deepest_support,
        "nested_partition_panels_checked": split_panels,
        "registered_comparisons": config["statistics"]["comparisons"],
        "bilinear_secondary": config["bilinear_secondary"],
        "planned_output_schema": {
            "predictions.csv": [
                "cohort_variant",
                "unit_id",
                "patient_id",
                "outer_fold",
                "label",
                "model",
                "p0",
                "p1",
            ],
            "selected_regularization.csv": [
                "variant",
                "outer_fold",
                "model",
                "C",
                "mean_meta_fold_class_equal_nll",
                "meta_fold_class_equal_nll",
                "selection_folds",
                "outer_validation_accessed",
                "feature_source",
            ],
            "fit_manifests.json": "patient-isolation hashes, concept support, convergence, selected C",
            "concept_exposures.csv": config["auditable_exposures"]["columns"],
            "metrics.csv": "class-equal NLL, BA, macro-F1, AUC, class-equal ECE10 and patient-bootstrap 90/95% CIs",
            "comparisons.csv": "patient-clustered paired class-equal log-score gains with 90/95% CIs and five descriptive outer-fold points",
            "concept_probe_diagnostics.csv": "valid support, hard-positive prevalence, BA@0.5, and AUC from saved outer-validation exposures",
            "OUTPUT_MANIFEST.json": "single top-level hash manifest for all run/ and analysis/ artifacts",
        },
    }


def rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Linux atomic publication that refuses even an empty existing target."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace directory publication is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    if error_number in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
        raise RuntimeError(
            "filesystem does not support atomic no-replace directory publication"
        ) from OSError(error_number, os.strerror(error_number), destination)
    raise OSError(error_number, os.strerror(error_number), destination)


def prepare_atomic_output(
    root: Path, output_relative: str, failure_relative: str
) -> tuple[Path, Path, Path, Path, Path]:
    output_path = safe_path(root, output_relative, must_exist=False)
    failure_path = safe_path(root, failure_relative, must_exist=False)
    if output_path.parent != failure_path.parent:
        raise ValueError("scientific output and failure receipt must share a parent")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stale = sorted(output_path.parent.glob(".paper_b_cbis_p3_scientific_*"))
    if output_path.exists() or failure_path.exists() or stale:
        raise FileExistsError(
            "scientific output, failure receipt, or outcome-exposed staging already exists"
        )

    # Prove filesystem support before a scientific model can be fit.
    probe_source = Path(
        tempfile.mkdtemp(prefix=".paper_b_cbis_p3_publish_probe_", dir=output_path.parent)
    )
    probe_destination = probe_source.with_name(probe_source.name + "_published")
    rename_directory_no_replace(probe_source, probe_destination)
    probe_destination.rmdir()

    staging = Path(
        tempfile.mkdtemp(prefix=".paper_b_cbis_p3_scientific_", dir=output_path.parent)
    )
    stage_run = staging / "run"
    stage_analysis = staging / "analysis"
    stage_run.mkdir()
    stage_analysis.mkdir()
    return output_path, failure_path, staging, stage_run, stage_analysis


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def artifact_manifest(directory: Path, names: Sequence[str]) -> dict[str, object]:
    return {
        "artifacts": {
            name: {
                "bytes": int((directory / name).stat().st_size),
                "sha256": sha256_file(directory / name),
            }
            for name in names
        }
    }


def run_scientific_experiment(root: Path, config_path: Path) -> Path:
    config = read_json(config_path)
    gate = verify_execution_gate(root, config_path, config)
    if config.get("official_test_rows_allowed") != 0:
        raise PermissionError("CBIS P3 official-test boundary changed")
    cohort_spec = config["cohort"]
    archive_path = safe_path(root, cohort_spec["archive"])
    manifest_path = safe_path(root, cohort_spec["manifest"])
    if sha256_file(archive_path) != cohort_spec["archive_sha256"]:
        raise ValueError("CBIS P3 cohort archive changed after freeze")
    if sha256_file(manifest_path) != cohort_spec["manifest_sha256"]:
        raise ValueError("CBIS P3 cohort manifest changed after freeze")
    full = load_cohort(archive_path, config["stable_concepts"], complete_only=False)
    full_x, embedding_audit = validate_embedding_release(root, config, full)
    primary = load_cohort(archive_path, config["stable_concepts"], complete_only=True)
    primary = primary.with_x(subset_x(full, primary, full_x))
    sensitivity = full.with_x(full_x)
    output = config["outputs"]
    if output.get("overwrite") is not False:
        raise ValueError("CBIS P3 scientific outputs must be no-clobber")
    output_path, failure_path, staging, stage_run, stage_analysis = prepare_atomic_output(
        root, str(output["directory"]), str(output["failure_directory"])
    )
    threadpools_before = scientific_threadpool_snapshot()
    threadpools_during: list[dict[str, object]] | None = None
    try:
        with threadpool_limits(limits=1):
            threadpools_during = scientific_threadpool_snapshot()
            assert_scientific_thread_limit(threadpools_during)
            predictions: list[dict[str, object]] = []
            selections: list[dict[str, object]] = []
            fit_manifests: list[dict[str, object]] = []
            concept_exposures: list[dict[str, object]] = []
            for cohort in (primary, sensitivity):
                (
                    cohort_predictions,
                    cohort_selections,
                    cohort_manifests,
                    cohort_exposures,
                ) = fit_variant(cohort, config)
                predictions.extend(cohort_predictions)
                selections.extend(cohort_selections)
                fit_manifests.extend(cohort_manifests)
                concept_exposures.extend(cohort_exposures)
            metric_rows, comparison_rows, concept_diagnostics = analyze_predictions(
                predictions, concept_exposures, config
            )

        write_csv(stage_run / "predictions.csv", predictions)
        write_csv(stage_run / "selected_regularization.csv", selections)
        write_csv(stage_run / "concept_exposures.csv", concept_exposures)
        write_json(stage_run / "fit_manifests.json", {"folds": fit_manifests})
        write_csv(stage_analysis / "metrics.csv", metric_rows)
        write_csv(stage_analysis / "comparisons.csv", comparison_rows)
        write_csv(
            stage_analysis / "concept_probe_diagnostics.csv", concept_diagnostics
        )
        analysis = {
            "protocol_id": PROTOCOL_ID,
            "evidence_role": "HEAVILY_EXPOSED_SECOND_DEVELOPMENT_DOMAIN",
            "official_test_rows_used": 0,
            "prediction_rows": len(predictions),
            "concept_exposure_rows": len(concept_exposures),
            "embedding_release": embedding_audit,
            "execution_gate": gate,
            "numeric_threadpools_before_limit": threadpools_before,
            "numeric_threadpools_during_fit_and_analysis": threadpools_during,
            "metrics": metric_rows,
            "comparisons": comparison_rows,
            "concept_probe_diagnostics": concept_diagnostics,
            "uncertainty_scope": config["estimand"]["uncertainty_scope"],
            "cross_domain_pooling": config["estimand"]["cross_domain_pooling"],
            "claim_boundary": config["claim_boundary"],
        }
        write_json(stage_analysis / "analysis.json", analysis)
        run_manifest = {
            "schema_version": "paper-b-cbis-p3-minimal-scientific-run-v1",
            "status": "PASS_ATOMIC_NO_OVERWRITE",
            "protocol_id": PROTOCOL_ID,
            "config": config_path.relative_to(root).as_posix(),
            "config_sha256": sha256_file(config_path),
            "execution_gate": gate,
            "official_test_rows_used": 0,
            "embedding_release": embedding_audit,
            "numeric_threadpools_before_limit": threadpools_before,
            "numeric_threadpools_during_fit_and_analysis": threadpools_during,
            "prediction_rows": len(predictions),
            "concept_exposure_rows": len(concept_exposures),
        }
        write_json(stage_run / "run_manifest.json", run_manifest)
        artifact_names = [
            "run/predictions.csv",
            "run/selected_regularization.csv",
            "run/concept_exposures.csv",
            "run/fit_manifests.json",
            "run/run_manifest.json",
            "analysis/metrics.csv",
            "analysis/comparisons.csv",
            "analysis/concept_probe_diagnostics.csv",
            "analysis/analysis.json",
        ]
        top_manifest = artifact_manifest(staging, artifact_names)
        top_manifest.update(
            {
                "schema_version": "paper-b-cbis-p3-minimal-output-v1",
                "status": "PASS_ATOMIC_NO_OVERWRITE",
                "protocol_id": PROTOCOL_ID,
            }
        )
        write_json(staging / "OUTPUT_MANIFEST.json", top_manifest)
        rename_directory_no_replace(staging, output_path)
        return output_path
    except BaseException as error:
        # Once fitting begins, any failure is outcome-exposed.  Publish the entire
        # staging directory as a no-replace failure receipt and block silent reruns.
        write_json(
            staging / "FAILURE.json",
            {
                "schema_version": "paper-b-cbis-p3-minimal-scientific-failure-v1",
                "status": "OUTCOME_EXPOSED_FAILURE_REQUIRES_NEW_ROOT_DECISION",
                "protocol_id": PROTOCOL_ID,
                "config_sha256": sha256_file(config_path),
                "freeze_manifest_sha256": gate["freeze_manifest_sha256"],
                "official_test_rows_used": 0,
                "observed_runtime": gate["observed_runtime"],
                "numeric_threadpools_before_limit": threadpools_before,
                "numeric_threadpools_during_fit_and_analysis": threadpools_during,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "staging_may_contain_partial_scientific_outcomes": True,
                "automatic_rerun_allowed": False,
            },
        )
        try:
            rename_directory_no_replace(staging, failure_path)
        except BaseException as publication_error:
            raise RuntimeError(
                f"scientific run failed and failure publication failed; preserve staging {staging}"
            ) from publication_error
        raise RuntimeError(
            f"scientific run failed; outcome-exposed receipt published at {failure_path}"
        ) from error


__all__ = [
    "APPROVAL_STATUS",
    "DRAFT_STATUS",
    "FINAL_CONFIG_STATUS",
    "FREEZE_STATUS",
    "GOLD_ORACLE",
    "GOLD_TO_PREDICTED",
    "INSAMPLE_RAW_SOFT",
    "OOF_RAW_HARD",
    "OOF_RAW_LOG_ODDS",
    "OOF_RAW_SOFT",
    "PRIMARY_MODELS",
    "PRIMARY_VARIANT",
    "PROTOCOL_ID",
    "SENSITIVITY_MODELS",
    "SENSITIVITY_VARIANT",
    "CbisCohort",
    "all_view_mapping",
    "build_outer_concept_panel",
    "class_equal_nll",
    "dry_run_audit",
    "fit_concept_probabilities",
    "fold_partition",
    "load_cohort",
    "paired_log_score_gain",
    "raw_hard",
    "raw_soft",
    "reference_log_odds",
    "run_scientific_experiment",
    "sha256_file",
    "validate_embedding_release",
    "verify_execution_gate",
    "write_json",
]
