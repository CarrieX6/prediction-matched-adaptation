"""Audited CBIS-DDSM loading with joint image/mask geometry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
import torch
import torch.nn.functional as functional


MAMMO_CLIP_MEAN = 0.3089279
MAMMO_CLIP_STD = 0.25053555408335154


@dataclass(frozen=True)
class PreparedView:
    image: torch.Tensor
    mask: torch.Tensor
    original_shape: tuple[int, int]
    crop_yxyx: tuple[int, int, int, int]
    mask_rescaled_to_full: bool = False


def series_uid_from_csv_path(path: str) -> str:
    parts = Path(path.strip()).parts
    if len(parts) < 2:
        raise ValueError(f"Cannot parse series UID from CBIS path: {path!r}")
    return parts[-2]


def _read_single_full_image(series_dir: Path) -> tuple[np.ndarray, str]:
    files = sorted(series_dir.glob("*.dcm"))
    if len(files) != 1:
        raise ValueError(f"Expected one full mammogram DICOM in {series_dir}, got {len(files)}")
    dataset = pydicom.dcmread(files[0])
    pixels = dataset.pixel_array
    if pixels.ndim != 2:
        raise ValueError(f"Expected a 2-D mammogram in {files[0]}")
    return pixels, str(dataset.PhotometricInterpretation)


def _read_roi_mask(series_dir: Path) -> np.ndarray:
    masks: list[np.ndarray] = []
    for path in sorted(series_dir.glob("*.dcm")):
        dataset = pydicom.dcmread(path)
        if str(getattr(dataset, "SeriesDescription", "")).lower() == "roi mask images":
            masks.append(dataset.pixel_array > 0)
    if len(masks) != 1:
        raise ValueError(f"Expected one ROI mask DICOM in {series_dir}, got {len(masks)}")
    return masks[0]


def union_masks(
    mask_series_dirs: list[Path],
    expected_shape: tuple[int, int],
    *,
    allow_uniform_rescale: bool = False,
    maximum_aspect_ratio_error: float = 0.001,
) -> tuple[np.ndarray, bool]:
    if not mask_series_dirs:
        raise ValueError("At least one lesion-mask series is required")
    union = np.zeros(expected_shape, dtype=bool)
    any_rescaled = False
    for series_dir in mask_series_dirs:
        mask = _read_roi_mask(series_dir)
        if mask.shape != expected_shape:
            mask_aspect = mask.shape[0] / mask.shape[1]
            expected_aspect = expected_shape[0] / expected_shape[1]
            aspect_error = abs(mask_aspect / expected_aspect - 1.0)
            if not allow_uniform_rescale or aspect_error > maximum_aspect_ratio_error:
                raise ValueError(
                    f"ROI/full-image shape mismatch for {series_dir}: "
                    f"{mask.shape} != {expected_shape}; aspect_error={aspect_error:.8f}"
                )
            mask = (
                functional.interpolate(
                    torch.from_numpy(mask.copy()).float()[None, None],
                    size=expected_shape,
                    mode="nearest",
                )[0, 0]
                .bool()
                .numpy()
            )
            any_rescaled = True
        union |= mask
    if not union.any():
        raise ValueError("Union ROI mask is empty")
    return union, any_rescaled


def _continuing_ones(binary: np.ndarray) -> np.ndarray:
    left = np.arange(len(binary))
    left[binary > 0] = 0
    left = np.maximum.accumulate(left)
    reverse = binary[::-1]
    right = np.arange(len(reverse))
    right[reverse > 0] = 0
    right = np.maximum.accumulate(right)
    right = len(reverse) - 1 - right[::-1]
    return right - left - 1


def breast_crop_indices(image_uint8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the Mammo-CLIP ExtractBreast row/column indices."""

    detection = np.where(image_uint8 <= 40, 0, image_uint8)
    height = detection.shape[0]
    y_low = height // 2 - int(height * 0.4)
    y_high = height // 2 + int(height * 0.4)
    columns = detection[y_low:y_high].std(axis=0) != 0
    column_runs = _continuing_ones(columns)
    column_indices = np.where(column_runs == column_runs.max())[0]
    if column_indices.size == 0:
        raise ValueError("Mammo-CLIP breast crop found no columns")

    detection = detection[:, column_indices]
    width = detection.shape[1]
    x_low = width // 2 - int(width * 0.4)
    x_high = width // 2 + int(width * 0.4)
    rows = detection[:, x_low:x_high].std(axis=1) != 0
    row_runs = _continuing_ones(rows)
    row_indices = np.where(row_runs == row_runs.max())[0]
    if row_indices.size == 0:
        raise ValueError("Mammo-CLIP breast crop found no rows")
    return row_indices, column_indices


def prepare_view(
    full_series_dir: Path,
    mask_series_dirs: list[Path],
    *,
    output_size: tuple[int, int] = (1520, 912),
    allow_uniform_mask_rescale: bool = False,
) -> PreparedView:
    """Load and jointly crop/resize a full mammogram and union lesion mask."""

    raw, photometric = _read_single_full_image(full_series_dir)
    mask, mask_rescaled_to_full = union_masks(
        mask_series_dirs,
        raw.shape,
        allow_uniform_rescale=allow_uniform_mask_rescale,
    )
    original_shape = tuple(int(value) for value in raw.shape)

    raw = raw[5:-5, 5:-5]
    mask = mask[5:-5, 5:-5]
    if photometric == "MONOCHROME1":
        raw = raw.max() - raw
    elif photometric != "MONOCHROME2":
        raise ValueError(f"Unsupported PhotometricInterpretation: {photometric}")

    raw_float = raw.astype(np.float32)
    dynamic_range = float(raw_float.max() - raw_float.min())
    if dynamic_range <= 0:
        raise ValueError("Mammogram has zero dynamic range")
    image_uint8 = np.rint((raw_float - raw_float.min()) / dynamic_range * 255.0).astype(np.uint8)
    rows, columns = breast_crop_indices(image_uint8)
    image_crop = image_uint8[np.ix_(rows, columns)]
    mask_crop = mask[np.ix_(rows, columns)]
    if not mask_crop.any():
        raise ValueError("Breast crop removed the lesion mask")

    image_tensor = torch.from_numpy(image_crop.copy()).float()[None, None] / 255.0
    mask_tensor = torch.from_numpy(mask_crop.copy()).float()[None, None]
    image_tensor = functional.interpolate(
        image_tensor, size=output_size, mode="bilinear", align_corners=False
    )[0]
    mask_tensor = functional.interpolate(mask_tensor, size=output_size, mode="nearest")[0]
    image_tensor = image_tensor.repeat(3, 1, 1)
    image_tensor = (image_tensor - MAMMO_CLIP_MEAN) / MAMMO_CLIP_STD

    crop_yxyx = (
        int(rows.min()) + 5,
        int(columns.min()) + 5,
        int(rows.max()) + 5,
        int(columns.max()) + 5,
    )
    return PreparedView(
        image=image_tensor,
        mask=mask_tensor,
        original_shape=original_shape,
        crop_yxyx=crop_yxyx,
        mask_rescaled_to_full=mask_rescaled_to_full,
    )
