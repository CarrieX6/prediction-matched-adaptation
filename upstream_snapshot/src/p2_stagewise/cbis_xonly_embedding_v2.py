"""V2 CBIS X-only preprocessing fixed to the audited 384x256 local path.

V1 was frozen but never executed.  A post-freeze, pre-extraction audit showed
that the materialized local CBIS cache used 384x256 rather than the source
function's 1520x912 default.  This module changes only that intermediate size;
the DICOM reading, image-only breast crop, encoder normalization, and all
source/output guards remain inherited from the V1 utilities.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from .cbis_xonly_embedding import (  # re-exported below where appropriate
    ALLOWED_SOURCE_PREFIX,
    CLIP_MEAN,
    CLIP_STD,
    ENCODER_SIZE,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_VIEW_COUNT,
    INPUT_MANIFEST_COLUMNS,
    MODEL_NAME,
    BiomedCLIPVisual,
    PreprocessReceipt,
    SourceAsset,
    canonical_json_bytes,
    flatten_cohort_sources,
    load_input_manifest,
    ordered_source_digest,
    sha256_file,
    validate_embedding_arrays,
    write_deterministic_npz,
    write_json,
)
from .dicom_io import _read_single_full_image, breast_crop_indices


AUDITED_INTERMEDIATE_SIZE = (384, 256)


def _safe_project_source(project_root: Path, relative: str) -> Path:
    """Delegate to the V1 path guard without exposing it as public API."""

    from .cbis_xonly_embedding import _safe_project_source as guard

    return guard(project_root, relative)


def preprocess_pixels_v2(
    pixels: np.ndarray, photometric_interpretation: str
) -> tuple[torch.Tensor, dict[str, int | str]]:
    """Apply the image-only 384x256 local CBIS path before 224x224 encoding."""

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
        image,
        size=AUDITED_INTERMEDIATE_SIZE,
        mode="bilinear",
        align_corners=False,
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


def prepare_source_v2(
    asset: SourceAsset, project_root: Path, *, verify_hash: bool = True
) -> tuple[torch.Tensor, dict[str, int | str]]:
    source = _safe_project_source(project_root, asset.local_dicom_path)
    if source.stat().st_size != asset.source_dicom_bytes:
        raise ValueError(f"source byte count changed for {asset.image_id}")
    if verify_hash and sha256_file(source) != asset.source_dicom_sha256:
        raise ValueError(f"source DICOM hash changed for {asset.image_id}")
    pixels, photometric = _read_single_full_image(source.parent)
    return preprocess_pixels_v2(pixels, photometric)


__all__ = [
    "AUDITED_INTERMEDIATE_SIZE",
    "BiomedCLIPVisual",
    "EXPECTED_CHECKPOINT_SHA256",
    "EXPECTED_VIEW_COUNT",
    "MODEL_NAME",
    "PreprocessReceipt",
    "SourceAsset",
    "load_input_manifest",
    "ordered_source_digest",
    "prepare_source_v2",
    "preprocess_pixels_v2",
    "sha256_file",
    "validate_embedding_arrays",
    "write_deterministic_npz",
    "write_json",
]

