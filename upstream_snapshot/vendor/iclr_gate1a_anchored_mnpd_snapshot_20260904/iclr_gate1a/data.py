"""Dataset contracts with raw privileged fields retained for fold-local preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PAD_NUMERIC_FIELDS = ("age", "fitspatrick", "diameter_1", "diameter_2")
PAD_CATEGORICAL_FIELDS = (
    "gender",
    "region",
    "skin_cancer_history",
    "cancer_history",
    "itch",
    "grew",
    "hurt",
    "changed",
    "bleed",
    "elevation",
    "smoke",
    "drink",
    "pesticide",
)

VINDR_GEOMETRY_FIELDS = (
    "box_count",
    "box_area_mean",
    "box_area_max",
    "box_area_min",
    "box_area_sum",
    "box_center_y_mean",
    "box_center_y_sd",
    "box_center_x_mean",
    "box_center_x_sd",
    "box_aspect_ratio_mean",
    "box_center_y_range",
)

VINDR_SINGLE_LESION_LABELS = (
    "Disc space narrowing",
    "Foraminal stenosis",
    "Osteophytes",
    "Other lesions",
    "Spondylolysthesis",
    "Surgical implant",
    "Vertebral collapse",
)


@dataclass(frozen=True)
class GateCohort:
    """Row-aligned image/exam data and an explicitly declared grouping unit."""

    name: str
    exam_id: np.ndarray
    group_id: np.ndarray
    y: np.ndarray
    labels: tuple[str, ...]
    x: np.ndarray
    z: np.ndarray | pd.DataFrame
    z_kind: str
    z_feature_names: tuple[str, ...]

    def validate(self) -> None:
        n = len(self.y)
        for field in ("exam_id", "group_id", "x", "z"):
            if len(getattr(self, field)) != n:
                raise ValueError(f"{field} is not row aligned")
        if self.x.ndim != 2 or not np.isfinite(self.x).all():
            raise ValueError("x must be a finite 2D matrix")
        if len(set(self.exam_id.astype(str))) != n:
            raise ValueError("exam_id must be unique")
        if np.any(self.group_id.astype(str) == ""):
            raise ValueError("group_id contains empty values")
        if set(np.unique(self.y)) - set(range(len(self.labels))):
            raise ValueError("labels must be contiguous integers")
        if self.z_kind == "numeric":
            z = np.asarray(self.z)
            if z.ndim != 2:
                raise ValueError("numeric z must be a 2D matrix")
        elif self.z_kind != "pad_mixed":
            raise ValueError(f"unknown z_kind={self.z_kind!r}")


def _normalize_embeddings(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-8, None)


def _restrict_binary(
    cohort: GateCohort,
    label_pair: tuple[str, str],
) -> GateCohort:
    """Keep only groups having one unambiguous label from the requested pair."""
    try:
        old_ids = [cohort.labels.index(label) for label in label_pair]
    except ValueError as error:
        raise ValueError(f"{label_pair} is not contained in {cohort.labels}") from error
    eligible_label = np.isin(cohort.y, old_ids)
    group_text = cohort.group_id.astype(str)
    keep_groups: list[str] = []
    for group in np.unique(group_text[eligible_label]):
        rows = np.flatnonzero(group_text == group)
        labels = set(cohort.y[rows].tolist())
        if len(labels) == 1 and labels.issubset(old_ids):
            keep_groups.append(group)
    keep = np.isin(group_text, keep_groups)
    old_to_new = {old: new for new, old in enumerate(old_ids)}
    y = np.array([old_to_new[int(value)] for value in cohort.y[keep]], dtype=np.int64)
    z = cohort.z.loc[keep].reset_index(drop=True) if isinstance(cohort.z, pd.DataFrame) else cohort.z[keep]
    restricted = GateCohort(
        name=cohort.name,
        exam_id=cohort.exam_id[keep],
        group_id=cohort.group_id[keep],
        y=y,
        labels=label_pair,
        x=cohort.x[keep],
        z=z,
        z_kind=cohort.z_kind,
        z_feature_names=cohort.z_feature_names,
    )
    restricted.validate()
    return restricted


def load_pad(
    embedding_archive: str | Path,
    metadata_csv: str | Path,
    label_pair: tuple[str, str] = ("ACK", "BCC"),
) -> GateCohort:
    archive = np.load(Path(embedding_archive), allow_pickle=True)
    image_id = archive["image_id"].astype(str)
    metadata = pd.read_csv(metadata_csv, dtype=str)
    if metadata["img_id"].astype(str).duplicated().any():
        raise ValueError("PAD metadata contains duplicate img_id")
    metadata = metadata.set_index(metadata["img_id"].astype(str), drop=False)
    missing = sorted(set(image_id) - set(metadata.index))
    if missing:
        raise ValueError(f"{len(missing)} PAD embeddings lack metadata")
    fields = PAD_NUMERIC_FIELDS + PAD_CATEGORICAL_FIELDS
    raw_z = metadata.loc[image_id, fields].reset_index(drop=True)
    cohort = GateCohort(
        name="PAD-UFES-20",
        exam_id=image_id,
        group_id=archive["patient_id"].astype(str),
        y=archive["label_id"].astype(np.int64),
        labels=tuple(archive["labels_order"].astype(str).tolist()),
        x=_normalize_embeddings(archive["embeddings"]),
        z=raw_z,
        z_kind="pad_mixed",
        z_feature_names=fields,
    )
    cohort.validate()
    return _restrict_binary(cohort, label_pair)


def load_pad_multiclass(
    embedding_archive: str | Path,
    metadata_csv: str | Path,
) -> GateCohort:
    """Load PAD and retain only patient groups having one unambiguous class."""
    archive = np.load(Path(embedding_archive), allow_pickle=True)
    image_id = archive["image_id"].astype(str)
    metadata = pd.read_csv(metadata_csv, dtype=str)
    if metadata["img_id"].astype(str).duplicated().any():
        raise ValueError("PAD metadata contains duplicate img_id")
    metadata = metadata.set_index(metadata["img_id"].astype(str), drop=False)
    missing = sorted(set(image_id) - set(metadata.index))
    if missing:
        raise ValueError(f"{len(missing)} PAD embeddings lack metadata")
    fields = PAD_NUMERIC_FIELDS + PAD_CATEGORICAL_FIELDS
    raw_z = metadata.loc[image_id, fields].reset_index(drop=True)
    y = archive["label_id"].astype(np.int64)
    group_id = archive["patient_id"].astype(str)
    keep_groups = []
    for group in np.unique(group_id):
        rows = np.flatnonzero(group_id == group)
        if len(np.unique(y[rows])) == 1:
            keep_groups.append(group)
    keep = np.isin(group_id, keep_groups)
    cohort = GateCohort(
        name="PAD-UFES-20-MULTICLASS",
        exam_id=image_id[keep],
        group_id=group_id[keep],
        y=y[keep],
        labels=tuple(archive["labels_order"].astype(str).tolist()),
        x=_normalize_embeddings(archive["embeddings"])[keep],
        z=raw_z.loc[keep].reset_index(drop=True),
        z_kind="pad_mixed",
        z_feature_names=fields,
    )
    cohort.validate()
    return cohort


def load_busi(
    feature_archive: str | Path,
    label_pair: tuple[str, str] = ("benign", "malignant"),
) -> GateCohort:
    archive = np.load(Path(feature_archive), allow_pickle=True)
    names = tuple(archive["geometry_names"].astype(str).tolist())
    cohort = GateCohort(
        name="BUSI-WHU",
        exam_id=archive["image_id"].astype(str),
        group_id=archive["patient_id"].astype(str),
        y=archive["label_id"].astype(np.int64),
        labels=tuple(archive["labels_order"].astype(str).tolist()),
        x=_normalize_embeddings(archive["embeddings"]),
        z=archive["mask_geometry"].astype(np.float32),
        z_kind="numeric",
        z_feature_names=names,
    )
    cohort.validate()
    return _restrict_binary(cohort, label_pair)


def load_vindr(
    feature_archive: str | Path,
    label_pair: tuple[str, str] = ("osteophytes", "disc_space_narrowing"),
) -> GateCohort:
    """Load a row-aligned VinDr archive without inferring features from old results.

    Required keys are deliberately explicit. Alternate historic key names are not silently
    accepted because doing so could change the deployment/privileged information contract.
    """
    archive = np.load(Path(feature_archive), allow_pickle=True)
    required = {
        "exam_id",
        "patient_id",
        "label_id",
        "labels_order",
        "embeddings",
        "box_geometry",
        "geometry_names",
    }
    missing = sorted(required - set(archive.files))
    if missing:
        raise ValueError(f"VinDr feature archive lacks required keys: {missing}")
    cohort = GateCohort(
        name="VinDr-SpineXR",
        exam_id=archive["exam_id"].astype(str),
        group_id=archive["patient_id"].astype(str),
        y=archive["label_id"].astype(np.int64),
        labels=tuple(archive["labels_order"].astype(str).tolist()),
        x=_normalize_embeddings(archive["embeddings"]),
        z=archive["box_geometry"].astype(np.float32),
        z_kind="numeric",
        z_feature_names=tuple(archive["geometry_names"].astype(str).tolist()),
    )
    cohort.validate()
    return _restrict_binary(cohort, label_pair)


def load_vindr_single_lesion_multiclass(
    feature_archive: str | Path,
    train_annotations: str | Path,
    test_annotations: str | Path,
) -> GateCohort:
    """Load the auditable seven-class VinDr single-lesion cohort.

    ``Y`` is retained only when an image has exactly one non-background lesion
    type. Studies containing different retained labels across views are then
    removed so study-disjoint aggregation has one well-defined outcome. ``Z``
    is the category-free 11-dimensional box geometry cached with the image
    embedding; lesion names are never included in ``Z``.

    The explicit legacy schema is accepted only in this function because the
    annotation CSVs are re-read and used to reconstruct and validate every
    retained label. This prevents the historic "rarest lesion present" target
    from silently entering the validation cohort.
    """
    archive = np.load(Path(feature_archive), allow_pickle=True)
    required = {
        "embeddings", "geom", "study_id", "image_id", "split",
    }
    missing = sorted(required - set(archive.files))
    if missing:
        raise ValueError(f"VinDr legacy feature archive lacks required keys: {missing}")
    image_id = archive["image_id"].astype(str)
    study_id = archive["study_id"].astype(str)
    split = archive["split"].astype(str)
    if len(set(image_id)) != len(image_id):
        raise ValueError("VinDr feature archive contains duplicate image_id")
    frames = []
    for expected_split, path in (
        ("train", train_annotations), ("test", test_annotations),
    ):
        frame = pd.read_csv(path, dtype={"image_id": str, "lesion_type": str})
        needed = {"image_id", "lesion_type"}
        if not needed.issubset(frame.columns):
            raise ValueError(f"VinDr {expected_split} annotations lack {sorted(needed - set(frame.columns))}")
        frame = frame.loc[:, ["image_id", "lesion_type"]].copy()
        frame["split"] = expected_split
        frames.append(frame)
    annotations = pd.concat(frames, ignore_index=True)
    split_count = annotations.groupby("image_id")["split"].nunique()
    if (split_count > 1).any():
        raise ValueError("VinDr image_id occurs in both official splits")
    annotated_split = annotations.groupby("image_id")["split"].first().to_dict()
    missing_annotation = sorted(set(image_id) - set(annotated_split))
    if missing_annotation:
        raise ValueError(f"{len(missing_annotation)} VinDr features lack annotations")
    if any(annotated_split[value] != observed for value, observed in zip(image_id, split)):
        raise ValueError("VinDr cached split disagrees with official annotation files")
    lesion_sets = annotations.groupby("image_id")["lesion_type"].agg(
        lambda values: frozenset(str(value) for value in values)
    )
    single_label = {
        key: next(iter(values))
        for key, values in lesion_sets.items()
        if len(values) == 1 and "No finding" not in values
    }
    eligible = np.array([value in single_label for value in image_id], dtype=bool)
    raw_label = np.array([single_label.get(value, "") for value in image_id], dtype=str)
    unexpected = sorted(set(raw_label[eligible]) - set(VINDR_SINGLE_LESION_LABELS))
    if unexpected:
        raise ValueError(f"unexpected VinDr lesion labels: {unexpected}")
    keep_studies = []
    for group in np.unique(study_id[eligible]):
        labels = np.unique(raw_label[eligible & (study_id == group)])
        if len(labels) == 1:
            keep_studies.append(group)
    keep = eligible & np.isin(study_id, keep_studies)
    label_to_id = {label: index for index, label in enumerate(VINDR_SINGLE_LESION_LABELS)}
    y = np.array([label_to_id[value] for value in raw_label[keep]], dtype=np.int64)
    z = archive["geom"].astype(np.float32)[keep]
    if z.ndim != 2 or z.shape[1] != len(VINDR_GEOMETRY_FIELDS):
        raise ValueError("VinDr geometry must have the registered 11 columns")
    cohort = GateCohort(
        name="VinDr-SpineXR-SINGLE-LESION-MULTICLASS",
        exam_id=image_id[keep],
        group_id=study_id[keep],
        y=y,
        labels=VINDR_SINGLE_LESION_LABELS,
        x=_normalize_embeddings(archive["embeddings"])[keep],
        z=z,
        z_kind="numeric",
        z_feature_names=VINDR_GEOMETRY_FIELDS,
    )
    cohort.validate()
    return cohort


def load_isic2017_multiclass(feature_archive: str | Path) -> GateCohort:
    """Load ISIC 2017, excluding groups with conflicting image diagnoses."""
    archive = np.load(Path(feature_archive), allow_pickle=True)
    required = {
        "image_id", "group_id", "label_id", "labels_order", "embeddings",
        "mask_geometry", "geometry_names",
    }
    missing = sorted(required - set(archive.files))
    if missing:
        raise ValueError(f"ISIC 2017 archive lacks required keys: {missing}")
    labels = tuple(archive["labels_order"].astype(str).tolist())
    expected = ("nevus", "melanoma", "seborrheic_keratosis")
    if labels != expected:
        raise ValueError(f"unexpected ISIC 2017 class order: {labels}")
    group_id = archive["group_id"].astype(str)
    y = archive["label_id"].astype(np.int64)
    keep_groups = [
        group
        for group in np.unique(group_id)
        if len(np.unique(y[group_id == group])) == 1
    ]
    keep = np.isin(group_id, keep_groups)
    cohort = GateCohort(
        name="ISIC-2017-TRAIN-MULTICLASS",
        exam_id=archive["image_id"].astype(str)[keep],
        group_id=group_id[keep],
        y=y[keep],
        labels=labels,
        x=_normalize_embeddings(archive["embeddings"])[keep],
        z=archive["mask_geometry"].astype(np.float32)[keep],
        z_kind="numeric",
        z_feature_names=tuple(archive["geometry_names"].astype(str).tolist()),
    )
    cohort.validate()
    if cohort.z.shape[1] != 10 or not np.isfinite(cohort.z).all():
        raise ValueError("ISIC 2017 mask geometry must be finite with 10 columns")
    return cohort


def load_cbis_mass_shape_multiclass(feature_archive: str | Path) -> GateCohort:
    """Load the frozen patient-grouped CBIS mass-morphology cohort."""
    archive = np.load(Path(feature_archive), allow_pickle=True)
    labels = tuple(archive["labels_order"].astype(str).tolist())
    expected = ("regular", "irregular", "architectural_distortion")
    if labels != expected:
        raise ValueError(f"unexpected CBIS mass-shape class order: {labels}")
    cohort = GateCohort(
        name="CBIS-DDSM-MASS-SHAPE-MULTICLASS",
        exam_id=archive["image_id"].astype(str),
        group_id=archive["group_id"].astype(str),
        y=archive["label_id"].astype(np.int64), labels=labels,
        x=_normalize_embeddings(archive["embeddings"]),
        z=archive["mask_geometry"].astype(np.float32), z_kind="numeric",
        z_feature_names=tuple(archive["geometry_names"].astype(str).tolist()),
    )
    cohort.validate()
    if cohort.z.shape[1] != 10 or not np.isfinite(cohort.z).all():
        raise ValueError("CBIS mask geometry must be finite with 10 columns")
    return cohort


def load_isic2018_task1_nonoverlap_multiclass(
    feature_archive: str | Path,
) -> GateCohort:
    """Load the frozen image-disjoint ISIC 2018 Task 1 diagnosis cohort."""
    archive = np.load(Path(feature_archive), allow_pickle=True)
    labels = tuple(archive["labels_order"].astype(str).tolist())
    expected = ("nevus", "melanoma", "seborrheic_keratosis")
    if labels != expected:
        raise ValueError(f"unexpected ISIC 2018 class order: {labels}")
    cohort = GateCohort(
        name="ISIC-2018-TASK1-NONOVERLAP-MULTICLASS",
        exam_id=archive["image_id"].astype(str),
        group_id=archive["group_id"].astype(str),
        y=archive["label_id"].astype(np.int64), labels=labels,
        x=_normalize_embeddings(archive["embeddings"]),
        z=archive["mask_geometry"].astype(np.float32), z_kind="numeric",
        z_feature_names=tuple(archive["geometry_names"].astype(str).tolist()),
    )
    cohort.validate()
    if cohort.z.shape[1] != 10 or not np.isfinite(cohort.z).all():
        raise ValueError("ISIC 2018 mask geometry must be finite with 10 columns")
    return cohort


def load_isic2018_heldout_multiclass(feature_archive: str | Path) -> GateCohort:
    """Load the registered ISIC 2018 validation/test diagnosis cohort."""
    archive = np.load(Path(feature_archive), allow_pickle=True)
    labels = tuple(archive["labels_order"].astype(str).tolist())
    expected = ("nevus", "melanoma", "seborrheic_keratosis")
    if labels != expected:
        raise ValueError(f"unexpected ISIC 2018 held-out class order: {labels}")
    cohort = GateCohort(
        name="ISIC-2018-HELDOUT-MULTICLASS",
        exam_id=archive["image_id"].astype(str),
        group_id=archive["group_id"].astype(str),
        y=archive["label_id"].astype(np.int64), labels=labels,
        x=_normalize_embeddings(archive["embeddings"]),
        z=archive["mask_geometry"].astype(np.float32), z_kind="numeric",
        z_feature_names=tuple(archive["geometry_names"].astype(str).tolist()),
    )
    cohort.validate()
    if cohort.z.shape[1] != 10 or not np.isfinite(cohort.z).all():
        raise ValueError("ISIC 2018 held-out mask geometry must be finite with 10 columns")
    return cohort
