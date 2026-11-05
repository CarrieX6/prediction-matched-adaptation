"""Outcome-blind, ROI-blind CBIS full-mammogram embedding utilities.

This module deliberately accepts only a flattened ``image_id``/DICOM-path
manifest.  It reuses the already audited image-only DICOM reader and
Mammo-CLIP breast-field crop from :mod:`p2_stagewise.dicom_io`, but it never
loads lesion masks, ROI images, pathology, concepts, folds, or outcomes.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as functional

from .dicom_io import _read_single_full_image, breast_crop_indices


MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
EXPECTED_CHECKPOINT_SHA256 = (
    "52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be"
)
EXPECTED_VIEW_COUNT = 1140
INPUT_MANIFEST_COLUMNS = (
    "image_id",
    "local_dicom_path",
    "source_dicom_sha256",
    "source_dicom_bytes",
)
ALLOWED_SOURCE_PREFIX = "data/cbis_ddsm_mass_train/full/"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
INTERMEDIATE_SIZE = (1520, 912)
ENCODER_SIZE = (224, 224)


@dataclass(frozen=True)
class SourceAsset:
    image_id: str
    local_dicom_path: str
    source_dicom_sha256: str
    source_dicom_bytes: int


@dataclass(frozen=True)
class PreprocessReceipt:
    image_id: str
    local_dicom_path: str
    source_dicom_sha256: str
    source_dicom_bytes: int
    photometric_interpretation: str
    original_height: int
    original_width: int
    crop_y0: int
    crop_x0: int
    crop_y1_exclusive: int
    crop_x1_exclusive: int
    cropped_height: int
    cropped_width: int
    normalized_l2_norm: float
    replay_max_abs_difference: float


class BiomedCLIPVisual(torch.nn.Module):
    """Vision-only reconstruction of the frozen local BiomedCLIP checkpoint."""

    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        state = torch.load(checkpoint, map_location="cpu")
        if not isinstance(state, Mapping):
            raise ValueError("BiomedCLIP checkpoint is not a state mapping")
        trunk_state = {
            key.removeprefix("visual.trunk."): value
            for key, value in state.items()
            if isinstance(key, str) and key.startswith("visual.trunk.")
        }
        self.trunk = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=0
        )
        missing, unexpected = self.trunk.load_state_dict(trunk_state, strict=False)
        if missing or unexpected:
            raise ValueError(
                f"BiomedCLIP visual trunk mismatch: missing={missing}, "
                f"unexpected={unexpected}"
            )
        projection = state.get("visual.head.proj.weight")
        if projection is None or tuple(projection.shape) != (512, 768):
            raise ValueError("BiomedCLIP visual projection is missing or malformed")
        self.projection = torch.nn.Linear(768, 512, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(projection)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(self.trunk(image))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    Path(path).write_bytes(canonical_json_bytes(payload))


def write_deterministic_npz(
    path: str | Path, arrays: Mapping[str, np.ndarray]
) -> None:
    """Write a compressed NPZ with fixed member order and ZIP metadata."""

    destination = Path(path)
    with zipfile.ZipFile(destination, mode="x") as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer, np.asarray(arrays[name]), allow_pickle=False
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(
                info,
                buffer.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )


def _safe_project_source(project_root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe source path: {relative!r}")
    normalized = pure.as_posix()
    if not normalized.startswith(ALLOWED_SOURCE_PREFIX):
        raise ValueError(f"source is outside the frozen Mass-Training root: {relative}")
    if "Mass-Test" in normalized or "mass_test" in normalized.lower():
        raise ValueError("official Mass-Test source is forbidden")
    path = (project_root / normalized).resolve(strict=True)
    allowed = (project_root / ALLOWED_SOURCE_PREFIX).resolve(strict=True)
    if not path.is_relative_to(allowed) or not path.is_file():
        raise ValueError(f"source path escaped or is not a file: {relative}")
    return path


def flatten_cohort_sources(cohort_csv: Path, project_root: Path) -> list[tuple[str, str]]:
    """Read only view-presence, key, and path fields from the frozen cohort CSV."""

    allowed_columns = [
        "cc_present",
        "cc_image_id",
        "cc_local_dicom_path",
        "mlo_present",
        "mlo_image_id",
        "mlo_local_dicom_path",
    ]
    frame = pd.read_csv(cohort_csv, dtype=str, keep_default_na=False, usecols=allowed_columns)
    if tuple(frame.columns) != tuple(allowed_columns):
        raise ValueError("cohort source columns changed")
    records: list[tuple[str, str]] = []
    for row in frame.itertuples(index=False):
        for prefix in ("cc", "mlo"):
            present = getattr(row, f"{prefix}_present")
            image_id = getattr(row, f"{prefix}_image_id")
            relative = getattr(row, f"{prefix}_local_dicom_path")
            if present not in {"0", "1"}:
                raise ValueError(f"invalid {prefix} presence value")
            if present == "0":
                if image_id or relative:
                    raise ValueError(f"absent {prefix} view has a key or path")
                continue
            if not image_id or not relative:
                raise ValueError(f"present {prefix} view lacks a key or path")
            expected_suffix = "_CC" if prefix == "cc" else "_MLO"
            if not image_id.endswith(expected_suffix):
                raise ValueError(f"view key disagrees with {prefix}: {image_id}")
            _safe_project_source(project_root, relative)
            records.append((image_id, PurePosixPath(relative).as_posix()))
    records.sort()
    if len(records) != EXPECTED_VIEW_COUNT:
        raise ValueError(f"expected {EXPECTED_VIEW_COUNT} source views, got {len(records)}")
    image_ids = [record[0] for record in records]
    paths = [record[1] for record in records]
    if len(set(image_ids)) != len(image_ids) or len(set(paths)) != len(paths):
        raise ValueError("frozen source keys or DICOM paths are duplicated")
    return records


def load_input_manifest(path: Path, project_root: Path) -> list[SourceAsset]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != INPUT_MANIFEST_COLUMNS:
            raise ValueError("embedding input-manifest columns changed")
        rows = list(reader)
    if len(rows) != EXPECTED_VIEW_COUNT:
        raise ValueError(f"expected {EXPECTED_VIEW_COUNT} input assets, got {len(rows)}")
    assets: list[SourceAsset] = []
    for row in rows:
        image_id = row["image_id"]
        relative = PurePosixPath(row["local_dicom_path"]).as_posix()
        digest = row["source_dicom_sha256"]
        byte_text = row["source_dicom_bytes"]
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"malformed source hash for {image_id}")
        try:
            byte_count = int(byte_text)
        except ValueError as error:
            raise ValueError(f"malformed source byte count for {image_id}") from error
        source = _safe_project_source(project_root, relative)
        if source.stat().st_size != byte_count:
            raise ValueError(f"source byte count changed for {image_id}")
        assets.append(SourceAsset(image_id, relative, digest, byte_count))
    if [asset.image_id for asset in assets] != sorted(asset.image_id for asset in assets):
        raise ValueError("embedding input manifest is not sorted by image_id")
    if len({asset.image_id for asset in assets}) != len(assets):
        raise ValueError("embedding input manifest has duplicate image IDs")
    if len({asset.local_dicom_path for asset in assets}) != len(assets):
        raise ValueError("embedding input manifest has duplicate DICOM paths")
    return assets


def ordered_source_digest(assets: Iterable[SourceAsset]) -> str:
    digest = hashlib.sha256()
    for asset in assets:
        digest.update(
            (
                f"{asset.image_id}\t{asset.local_dicom_path}\t"
                f"{asset.source_dicom_sha256}\t{asset.source_dicom_bytes}\n"
            ).encode("utf-8")
        )
    return digest.hexdigest()


def preprocess_pixels(
    pixels: np.ndarray, photometric_interpretation: str
) -> tuple[torch.Tensor, dict[str, int | str]]:
    """Apply the frozen image-only mammogram transform.

    Orientation is the native stored raster orientation: there is no
    laterality-dependent flip, rotation, or transpose.  This matches the
    established local full-mammogram path and avoids introducing identifiers
    into the image transform.
    """

    raw = np.asarray(pixels)
    if raw.ndim != 2:
        raise ValueError("expected one 2-D full mammogram")
    if raw.shape[0] <= 10 or raw.shape[1] <= 10:
        raise ValueError("full mammogram is too small for the frozen border trim")
    original_height, original_width = map(int, raw.shape)
    raw = raw[5:-5, 5:-5]
    if photometric_interpretation == "MONOCHROME1":
        raw = raw.max() - raw
    elif photometric_interpretation != "MONOCHROME2":
        raise ValueError(
            f"unsupported PhotometricInterpretation: {photometric_interpretation}"
        )
    raw_float = raw.astype(np.float32)
    minimum = float(raw_float.min())
    dynamic_range = float(raw_float.max() - minimum)
    if not math.isfinite(dynamic_range) or dynamic_range <= 0.0:
        raise ValueError("full mammogram has invalid or zero dynamic range")
    image_uint8 = np.rint((raw_float - minimum) / dynamic_range * 255.0).astype(
        np.uint8
    )
    rows, columns = breast_crop_indices(image_uint8)
    if (
        rows.size == 0
        or columns.size == 0
        or (rows.size > 1 and not np.all(np.diff(rows) == 1))
        or (columns.size > 1 and not np.all(np.diff(columns) == 1))
    ):
        raise ValueError("breast-field crop indices are empty or non-contiguous")
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(columns[0]), int(columns[-1]) + 1
    cropped = image_uint8[y0:y1, x0:x1]
    if cropped.size == 0 or not np.isfinite(cropped).all():
        raise ValueError("breast-field crop is empty or non-finite")

    image = torch.from_numpy(cropped.copy()).float()[None, None] / 255.0
    image = functional.interpolate(
        image, size=INTERMEDIATE_SIZE, mode="bilinear", align_corners=False
    )
    image = functional.interpolate(
        image, size=ENCODER_SIZE, mode="bilinear", align_corners=False
    )[0].repeat(3, 1, 1)
    mean = torch.tensor(CLIP_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(CLIP_STD, dtype=torch.float32)[:, None, None]
    normalized = (image - mean) / std
    if tuple(normalized.shape) != (3, 224, 224) or not torch.isfinite(normalized).all():
        raise ValueError("frozen encoder input is malformed")
    receipt: dict[str, int | str] = {
        "photometric_interpretation": photometric_interpretation,
        "original_height": original_height,
        "original_width": original_width,
        "crop_y0": y0 + 5,
        "crop_x0": x0 + 5,
        "crop_y1_exclusive": y1 + 5,
        "crop_x1_exclusive": x1 + 5,
        "cropped_height": y1 - y0,
        "cropped_width": x1 - x0,
    }
    return normalized.contiguous(), receipt


def prepare_source(
    asset: SourceAsset, project_root: Path, *, verify_hash: bool = True
) -> tuple[torch.Tensor, dict[str, int | str]]:
    source = _safe_project_source(project_root, asset.local_dicom_path)
    if source.stat().st_size != asset.source_dicom_bytes:
        raise ValueError(f"source byte count changed for {asset.image_id}")
    if verify_hash and sha256_file(source) != asset.source_dicom_sha256:
        raise ValueError(f"source DICOM hash changed for {asset.image_id}")
    pixels, photometric = _read_single_full_image(source.parent)
    return preprocess_pixels(pixels, photometric)


def validate_embedding_arrays(
    assets: Sequence[SourceAsset],
    image_ids: np.ndarray,
    embeddings: np.ndarray,
    l2_norms: np.ndarray,
    *,
    norm_tolerance: float,
) -> None:
    expected_ids = np.asarray([asset.image_id for asset in assets], dtype=str)
    if image_ids.shape != (EXPECTED_VIEW_COUNT,) or not np.array_equal(
        image_ids.astype(str), expected_ids
    ):
        raise ValueError("output keys do not exactly equal the frozen 1140 input keys")
    if embeddings.shape != (EXPECTED_VIEW_COUNT, 512) or embeddings.dtype != np.float32:
        raise ValueError("embedding archive must be float32 with shape (1140, 512)")
    if l2_norms.shape != (EXPECTED_VIEW_COUNT,) or l2_norms.dtype != np.float32:
        raise ValueError("per-view L2 norms must be float32 with shape (1140,)")
    if not np.isfinite(embeddings).all() or not np.isfinite(l2_norms).all():
        raise ValueError("embedding archive contains NaN or infinity")
    observed = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    if np.max(np.abs(observed - l2_norms.astype(np.float64))) > 2e-6:
        raise ValueError("saved per-view norms disagree with the embeddings")
    if np.max(np.abs(observed - 1.0)) > norm_tolerance:
        raise ValueError("one or more embeddings violate the frozen L2-norm tolerance")

