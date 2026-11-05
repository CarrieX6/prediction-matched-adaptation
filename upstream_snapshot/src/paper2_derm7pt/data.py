"""Strict Derm7pt source audit and the frozen X/Y/Z cohort adapter."""
from __future__ import annotations

from collections import defaultdict
from hashlib import sha256 as _sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

from iclr_gate1a.data import GateCohort


DERM7PT_LABELS = ("BCC", "NEV", "MEL", "MISC", "SK")
DERM7PT_DIAGNOSIS_GROUPS = {
    "basal cell carcinoma": "BCC",
    "nevus": "NEV",
    "blue nevus": "NEV",
    "clark nevus": "NEV",
    "combined nevus": "NEV",
    "congenital nevus": "NEV",
    "dermal nevus": "NEV",
    "recurrent nevus": "NEV",
    "reed or spitz nevus": "NEV",
    "melanoma": "MEL",
    "melanoma (in situ)": "MEL",
    "melanoma (less than 0.76 mm)": "MEL",
    "melanoma (0.76 to 1.5 mm)": "MEL",
    "melanoma (more than 1.5 mm)": "MEL",
    "melanoma metastasis": "MEL",
    "df/lt/mls/misc": "MISC",
    "dermatofibroma": "MISC",
    "lentigo": "MISC",
    "melanosis": "MISC",
    "miscellaneous": "MISC",
    "vascular lesion": "MISC",
    "seborrheic keratosis": "SK",
}
EXPECTED_CASES = 1011
EXPECTED_SPLIT_COUNTS = {"train": 413, "valid": 203, "test": 395}
EXPECTED_IDENTICAL_MODALITY_CASES = (564, 799, 823, 826, 832, 833, 838, 843, 850)
EXPECTED_ELIGIBLE_SPLIT_COUNTS = {"train": 411, "valid": 198, "test": 393}
EXPECTED_ELIGIBLE_CLASS_COUNTS = {
    "train": (19, 256, 90, 30, 16),
    "valid": (7, 100, 59, 22, 10),
    "test": (16, 219, 100, 39, 19),
}
REQUIRED_METADATA = {
    "clinic",
    "derm",
    "diagnosis",
    "sex",
    "location",
    "elevation",
}


def sha256_file(path: str | Path) -> str:
    digest = _sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _member(names: set[str], prefix: str, relative: str) -> str:
    relative = PurePosixPath(relative).as_posix().lstrip("/")
    candidates = (f"{prefix}images/{relative}", f"{prefix}{relative}")
    matches = [candidate for candidate in candidates if candidate in names]
    if len(matches) != 1:
        raise ValueError(f"expected one official image for {relative!r}, found {matches}")
    return matches[0]


def _load_zip_tables(archive: zipfile.ZipFile) -> tuple[pd.DataFrame, dict[str, np.ndarray], str]:
    names = archive.namelist()
    metadata_names = [name for name in names if name.endswith("meta/meta.csv")]
    if len(metadata_names) != 1:
        raise ValueError(f"expected one meta/meta.csv, found {metadata_names}")
    metadata_name = metadata_names[0]
    prefix = metadata_name[: -len("meta/meta.csv")]
    metadata = pd.read_csv(archive.open(metadata_name), dtype=str).fillna("")
    partitions: dict[str, np.ndarray] = {}
    for split in ("train", "valid", "test"):
        split_name = f"{prefix}meta/{split}_indexes.csv"
        table = pd.read_csv(archive.open(split_name))
        if list(table.columns) != ["indexes"]:
            raise ValueError(f"{split_name} must contain only the indexes column")
        partitions[split] = table["indexes"].to_numpy(dtype=np.int64)
    return metadata, partitions, prefix


def _labels(metadata: pd.DataFrame) -> np.ndarray:
    normalized = metadata["diagnosis"].astype(str).str.strip().str.lower()
    unknown = sorted(set(normalized) - set(DERM7PT_DIAGNOSIS_GROUPS))
    if unknown:
        raise ValueError(f"diagnoses absent from the frozen grouping: {unknown}")
    grouped = normalized.map(DERM7PT_DIAGNOSIS_GROUPS)
    label_to_id = {label: index for index, label in enumerate(DERM7PT_LABELS)}
    return grouped.map(label_to_id).to_numpy(dtype=np.int64)


def _validate_partitions(partitions: dict[str, np.ndarray], n_cases: int) -> None:
    for split, indexes in partitions.items():
        if len(indexes) != EXPECTED_SPLIT_COUNTS[split]:
            raise ValueError(f"unexpected {split} split size: {len(indexes)}")
        if len(set(indexes.tolist())) != len(indexes):
            raise ValueError(f"{split} contains duplicate indexes")
        if np.any((indexes < 0) | (indexes >= n_cases)):
            raise ValueError(f"{split} contains out-of-range indexes")
    combined = np.concatenate(list(partitions.values()))
    if len(set(combined.tolist())) != len(combined):
        raise ValueError("official partitions overlap")
    if set(combined.tolist()) != set(range(n_cases)):
        raise ValueError("official partitions do not cover all metadata rows")


def audit_release(source: str | Path, *, verify_images: bool = True) -> dict[str, object]:
    """Audit the credentialed archive without exporting row-level metadata."""
    source = Path(source)
    with zipfile.ZipFile(source) as archive:
        if archive.testzip() is not None:
            raise ValueError("Derm7pt archive has a CRC failure")
        names_list = archive.namelist()
        if len(names_list) != len(set(names_list)):
            raise ValueError("Derm7pt archive contains duplicate member names")
        names = set(names_list)
        metadata, partitions, prefix = _load_zip_tables(archive)
        if len(metadata) != EXPECTED_CASES:
            raise ValueError(f"expected {EXPECTED_CASES} cases, found {len(metadata)}")
        missing = sorted(REQUIRED_METADATA - set(metadata.columns))
        if missing:
            raise ValueError(f"metadata lacks required columns: {missing}")
        _validate_partitions(partitions, len(metadata))
        labels = _labels(metadata)
        clinic = [_member(names, prefix, value) for value in metadata["clinic"]]
        derm = [_member(names, prefix, value) for value in metadata["derm"]]
        referenced = sorted(set(clinic + derm))
        content_hash: dict[str, str] = {}
        for name in referenced:
            payload = archive.read(name)
            content_hash[name] = _sha256(payload).hexdigest()
            if verify_images:
                with Image.open(BytesIO(payload)) as image:
                    image.verify()

        uses: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for index, (clinic_name, derm_name) in enumerate(zip(clinic, derm, strict=True)):
            uses[content_hash[clinic_name]].append((index, "clinic"))
            uses[content_hash[derm_name]].append((index, "derm"))
        cross_case = {
            digest: records
            for digest, records in uses.items()
            if len({index for index, _ in records}) > 1
        }
        if cross_case:
            raise ValueError(f"image content is reused across {len(cross_case)} case groups")
        identical = tuple(
            index
            for index, (clinic_name, derm_name) in enumerate(zip(clinic, derm, strict=True))
            if content_hash[clinic_name] == content_hash[derm_name]
        )
        if identical != EXPECTED_IDENTICAL_MODALITY_CASES:
            raise ValueError(
                "identical clinical/dermoscopic cases differ from the frozen source audit: "
                f"{identical}"
            )
        excluded = set(identical)
        eligible_counts: dict[str, int] = {}
        class_counts: dict[str, list[int]] = {}
        for split, indexes in partitions.items():
            eligible = np.asarray([index for index in indexes if int(index) not in excluded])
            eligible_counts[split] = int(len(eligible))
            class_counts[split] = np.bincount(
                labels[eligible], minlength=len(DERM7PT_LABELS)
            ).astype(int).tolist()
            if eligible_counts[split] != EXPECTED_ELIGIBLE_SPLIT_COUNTS[split]:
                raise ValueError(f"eligible {split} count changed")
            if tuple(class_counts[split]) != EXPECTED_ELIGIBLE_CLASS_COUNTS[split]:
                raise ValueError(f"eligible {split} class census changed")

        metadata_name = f"{prefix}meta/meta.csv"
        split_sha = {
            split: _sha256(archive.read(f"{prefix}meta/{split}_indexes.csv")).hexdigest()
            for split in partitions
        }
        return {
            "schema_version": "1.0.0",
            "audit_id": "DERM7PT_CANONICAL_SOURCE_AUDIT_20260904",
            "status": "PASS_WITH_RESULT_BLIND_IDENTICAL_MODALITY_EXCLUSIONS",
            "archive_bytes": source.stat().st_size,
            "archive_sha256": sha256_file(source),
            "metadata_sha256": _sha256(archive.read(metadata_name)).hexdigest(),
            "official_split_sha256": split_sha,
            "raw_cases": len(metadata),
            "raw_split_counts": {key: len(value) for key, value in partitions.items()},
            "available_image_files": sum(
                PurePosixPath(name).suffix.lower() in {".jpg", ".jpeg", ".png"}
                for name in names
            ),
            "distinct_referenced_image_files": len(referenced),
            "cross_case_duplicate_content_groups": 0,
            "identical_clinical_dermoscopic_cases": list(identical),
            "exclusion_rule": (
                "exclude a whole case iff the clinical and dermoscopic references have "
                "identical image bytes, because no additional modality Z exists"
            ),
            "eligible_cases": len(metadata) - len(identical),
            "eligible_split_counts": eligible_counts,
            "labels": list(DERM7PT_LABELS),
            "eligible_class_counts_by_split": class_counts,
            "official_partitions_cross_case_content_leakage": 0,
            "patient_identifier_status": "not_released",
            "independent_unit": "official case index after identical-modality exclusion",
            "image_decode_audit": "all distinct referenced images verified" if verify_images else "not_run",
            "license": "CC BY-NC-ND 4.0; local non-commercial academic use; no redistribution",
        }


def load_gate_cohort(feature_archive: str | Path, metadata_csv: str | Path) -> GateCohort:
    """Load eligible clinical-image X and paired dermoscopic-image Z embeddings."""
    feature_archive = Path(feature_archive)
    metadata_csv = Path(metadata_csv)
    with np.load(feature_archive, allow_pickle=False) as archive:
        required = {
            "metadata_index",
            "clinic_path",
            "derm_path",
            "clinical_embeddings",
            "derm_embeddings",
            "metadata_sha256",
            "excluded_metadata_indices",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"feature archive lacks keys: {missing}")
        metadata_index = archive["metadata_index"].astype(np.int64)
        clinic_path = archive["clinic_path"].astype(str)
        derm_path = archive["derm_path"].astype(str)
        clinical = archive["clinical_embeddings"].astype(np.float32)
        derm = archive["derm_embeddings"].astype(np.float32)
        metadata_hash = str(archive["metadata_sha256"].item())
        excluded = tuple(archive["excluded_metadata_indices"].astype(int).tolist())
    if excluded != EXPECTED_IDENTICAL_MODALITY_CASES:
        raise ValueError("feature archive exclusion set changed")
    if sha256_file(metadata_csv) != metadata_hash:
        raise ValueError("feature archive is not aligned to meta.csv")
    metadata = pd.read_csv(metadata_csv, dtype=str).fillna("")
    if len(metadata) != EXPECTED_CASES:
        raise ValueError("metadata case count changed")
    expected_index = np.asarray(
        [index for index in range(len(metadata)) if index not in set(excluded)],
        dtype=np.int64,
    )
    if not np.array_equal(metadata_index, expected_index):
        raise ValueError("feature metadata index differs from the frozen eligible census")
    table = metadata.iloc[metadata_index]
    if not np.array_equal(clinic_path, table["clinic"].astype(str).to_numpy()):
        raise ValueError("clinical embeddings are not aligned to meta.csv")
    if not np.array_equal(derm_path, table["derm"].astype(str).to_numpy()):
        raise ValueError("dermoscopic embeddings are not aligned to meta.csv")
    if clinical.shape != derm.shape or clinical.ndim != 2 or len(clinical) != len(table):
        raise ValueError("paired embeddings must be aligned 2D matrices")
    if not np.isfinite(clinical).all() or not np.isfinite(derm).all():
        raise ValueError("feature archive contains non-finite embeddings")
    clinical /= np.clip(np.linalg.norm(clinical, axis=1, keepdims=True), 1e-8, None)
    derm /= np.clip(np.linalg.norm(derm, axis=1, keepdims=True), 1e-8, None)
    all_labels = _labels(metadata)
    y = all_labels[metadata_index]
    exam_id = np.asarray([f"derm7pt:{index:04d}" for index in metadata_index], dtype=str)
    cohort = GateCohort(
        name="DERM7PT-OFFICIAL-HOLDOUT-5CLASS-PAIR1002",
        exam_id=exam_id,
        group_id=exam_id.copy(),
        y=y,
        labels=DERM7PT_LABELS,
        x=clinical,
        z=derm,
        z_kind="numeric",
        z_feature_names=tuple(f"derm_embedding_{index:04d}" for index in range(derm.shape[1])),
    )
    cohort.validate()
    return cohort


def load_official_positions(metadata_directory: str | Path, cohort: GateCohort) -> dict[str, np.ndarray]:
    """Map official metadata indexes to positions in the eligible cohort."""
    metadata_directory = Path(metadata_directory)
    position = {int(value.split(":", 1)[1]): row for row, value in enumerate(cohort.exam_id)}
    excluded = set(EXPECTED_IDENTICAL_MODALITY_CASES)
    partitions: dict[str, np.ndarray] = {}
    raw: dict[str, np.ndarray] = {}
    for split in ("train", "valid", "test"):
        table = pd.read_csv(metadata_directory / f"{split}_indexes.csv")
        if list(table.columns) != ["indexes"]:
            raise ValueError(f"{split}_indexes.csv schema changed")
        indexes = table["indexes"].to_numpy(dtype=np.int64)
        raw[split] = indexes
        eligible = [int(index) for index in indexes if int(index) not in excluded]
        partitions[split] = np.asarray([position[index] for index in eligible], dtype=np.int64)
        if len(partitions[split]) != EXPECTED_ELIGIBLE_SPLIT_COUNTS[split]:
            raise ValueError(f"eligible {split} size changed")
    _validate_partitions(raw, EXPECTED_CASES)
    combined = np.concatenate(list(partitions.values()))
    if len(set(combined.tolist())) != len(combined) or set(combined) != set(range(len(cohort.y))):
        raise ValueError("eligible official partitions do not cover the cohort exactly once")
    return partitions

