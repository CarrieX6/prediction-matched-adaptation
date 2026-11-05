#!/usr/bin/env python3
"""Extract OpenAI CLIP ViT-B/16 features with the locked Paper B pixels."""

from __future__ import annotations

import hashlib
import json
import csv
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

# Load the audited preprocessing submodule without executing the unrelated
# p2_stagewise package initializer, which imports a local torchvision build.
package = types.ModuleType("p2_stagewise")
package.__path__ = [str(ROOT / "src/p2_stagewise")]
sys.modules.setdefault("p2_stagewise", package)

import paper_b_cbis_p3 as cb  # noqa: E402
from p2_stagewise.dicom_io import _read_single_full_image, breast_crop_indices  # noqa: E402


CHECKPOINT = Path("/home/xzf/.cache/clip/ViT-B-16.pt")
MEAN = np.asarray((0.48145466, 0.4578275, 0.40821073), dtype=np.float32)
STD = np.asarray((0.26862954, 0.26130258, 0.27577711), dtype=np.float32)


@dataclass(frozen=True)
class SourceAsset:
    image_id: str
    local_dicom_path: str
    source_dicom_sha256: str
    source_dicom_bytes: int


def load_input_manifest(path: Path, project_root: Path) -> list[SourceAsset]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assets = [SourceAsset(row["image_id"], row["local_dicom_path"],
                          row["source_dicom_sha256"], int(row["source_dicom_bytes"]))
              for row in rows]
    if len(assets) != 1140 or [a.image_id for a in assets] != sorted(a.image_id for a in assets):
        raise ValueError("CBIS input manifest differs from the locked 1140-view manifest")
    for asset in assets:
        source = project_root / asset.local_dicom_path
        if not source.is_file() or source.stat().st_size != asset.source_dicom_bytes:
            raise ValueError(f"CBIS source differs for {asset.image_id}")
    return assets


def prepare_source_v2(asset: SourceAsset, project_root: Path, *, verify_hash: bool = True):
    source = project_root / asset.local_dicom_path
    if verify_hash and sha256_file(source) != asset.source_dicom_sha256:
        raise ValueError(f"CBIS source hash differs for {asset.image_id}")
    raw, photometric = _read_single_full_image(source.parent)
    raw = np.asarray(raw)[5:-5, 5:-5]
    if photometric == "MONOCHROME1":
        raw = raw.max() - raw
    elif photometric != "MONOCHROME2":
        raise ValueError(f"unsupported photometric interpretation: {photometric}")
    values = raw.astype(np.float32)
    minimum = float(values.min())
    dynamic_range = float(values.max() - minimum)
    if not math.isfinite(dynamic_range) or dynamic_range <= 0:
        raise ValueError("invalid CBIS image dynamic range")
    image_uint8 = np.rint((values - minimum) / dynamic_range * 255.0).astype(np.uint8)
    rows, columns = breast_crop_indices(image_uint8)
    cropped = image_uint8[rows[0]:rows[-1] + 1, columns[0]:columns[-1] + 1]
    tensor = torch.from_numpy(cropped.copy()).float()[None, None] / 255.0
    tensor = functional.interpolate(tensor, size=(384, 256), mode="bilinear", align_corners=False)
    tensor = functional.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)[0].repeat(3, 1, 1)
    mean = torch.from_numpy(MEAN)[:, None, None]
    std = torch.from_numpy(STD)[:, None, None]
    return ((tensor - mean) / std).contiguous(), {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def encode(model: torch.jit.ScriptModule, batch: list[torch.Tensor]) -> np.ndarray:
    pixels = torch.stack(batch).to("cuda", dtype=torch.float16)
    with torch.inference_mode():
        values = model.encode_image(pixels).float()
        values /= values.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return values.cpu().numpy().astype(np.float32, copy=False)


def derm_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - MEAN[None, None, :]) / STD[None, None, :]
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def extract_derm(model: torch.jit.ScriptModule, batch_size: int = 64) -> dict[str, object]:
    csv_path = ROOT / "data/Derm7pt/derm7pt_dev609_only_v1_20260908.csv"
    frame = pd.read_csv(csv_path)
    if len(frame) != 609 or frame["metadata_index"].nunique() != 609:
        raise ValueError("Derm7pt development cohort differs from the locked 609 cases")
    image_root = ROOT / "data/Derm7pt/release_v0"
    paths = []
    for relative in frame["derm"].astype(str):
        candidates = (image_root / "images" / relative, image_root / relative)
        matches = [path for path in candidates if path.is_file()]
        if len(matches) != 1:
            raise FileNotFoundError(f"Derm7pt image path is ambiguous: {relative}")
        paths.append(matches[0])
    features = np.empty((609, 512), dtype=np.float32)
    for start in range(0, 609, batch_size):
        stop = min(start + batch_size, 609)
        features[start:stop] = encode(model, [derm_tensor(p) for p in paths[start:stop]])
        print(f"Derm7pt {stop}/609", flush=True)
    norms = np.linalg.norm(features.astype(np.float64), axis=1)
    np.savez_compressed(
        OUT / "derm7pt_clip_vit_b16_features.npz",
        metadata_index=frame["metadata_index"].to_numpy(dtype=np.int64),
        derm_path=frame["derm"].astype(str).to_numpy(),
        derm_embeddings=features,
        cohort_role=np.asarray("DERM7PT_OFFICIAL_TRAIN_VALID_DEV609_ONLY"),
        l2_norm=norms.astype(np.float32),
        model_name=np.asarray("OpenAI CLIP ViT-B/16"),
        checkpoint_sha256=np.asarray(sha256_file(CHECKPOINT)),
    )
    return {"cases": 609, "dimension": 512, "max_norm_error": float(np.max(np.abs(norms - 1)))}


def extract_cbis(model: torch.jit.ScriptModule, batch_size: int = 20) -> dict[str, object]:
    config = json.loads((ROOT / "configs/paper_b_cbis_mass_train_embedding_protocol_v2_20260910.json").read_text())
    assets = load_input_manifest(ROOT / config["inputs"]["input_manifest"], ROOT)
    if len(assets) != 1140:
        raise ValueError("CBIS view count differs from 1140")
    image_ids = np.asarray([asset.image_id for asset in assets], dtype=str)
    view_cache = OUT / "cbis_clip_vit_b16_view_features.npz"
    if view_cache.is_file():
        with np.load(view_cache, allow_pickle=False) as cached:
            if not np.array_equal(cached["image_id"].astype(str), image_ids):
                raise ValueError("cached CBIS view order differs from the locked manifest")
            if str(cached["checkpoint_sha256"].item()) != sha256_file(CHECKPOINT):
                raise ValueError("cached CBIS checkpoint differs")
            features = cached["embeddings"].astype(np.float32)
        print("CBIS-DDSM 1140/1140 (validated cache)", flush=True)
    else:
        features = np.empty((1140, 512), dtype=np.float32)
        for start in range(0, 1140, batch_size):
            stop = min(start + batch_size, 1140)
            tensors = [prepare_source_v2(asset, ROOT, verify_hash=False)[0] for asset in assets[start:stop]]
            features[start:stop] = encode(model, tensors)
            print(f"CBIS-DDSM {stop}/1140", flush=True)
        norms = np.linalg.norm(features.astype(np.float64), axis=1)
        np.savez_compressed(
            view_cache,
            image_id=image_ids,
            embeddings=features,
            l2_norm=norms.astype(np.float32),
            local_dicom_path=np.asarray([asset.local_dicom_path for asset in assets], dtype=str),
            model_name=np.asarray("OpenAI CLIP ViT-B/16"),
            checkpoint_sha256=np.asarray(sha256_file(CHECKPOINT)),
        )
    norms = np.linalg.norm(features.astype(np.float64), axis=1)
    cohort_path = ROOT / "data/CBIS_DDSM/paper_b_cbis_mass_train_breast_cohort_v1_20260910.npz"
    with np.load(cohort_path, allow_pickle=False) as cohort:
        unit_id = cohort["unit_id"].astype(str)
        patient_id = cohort["patient_id"].astype(str)
        cohort_image_id = cohort["image_id"].astype(str)
        view_present = cohort["view_present"].astype(bool)
        complete_mask = cohort["complete_concept_eligible"].astype(bool)
    by_id = {key: features[i].astype(np.float64) for i, key in enumerate(image_ids)}
    breast = np.empty((len(unit_id), 512), dtype=np.float64)
    for row in range(len(unit_id)):
        vectors = [by_id[str(cohort_image_id[row, view])] for view in range(2) if view_present[row, view]]
        if len(vectors) not in (1, 2):
            raise ValueError("CBIS breast has no view")
        breast[row] = np.mean(np.stack(vectors), axis=0)
    complete_x = breast[complete_mask].astype(np.float32)
    if complete_x.shape != (655, 512):
        raise ValueError("CBIS complete-case cohort differs from 655 breasts")
    np.savez_compressed(
        OUT / "cbis_clip_vit_b16_breast_features.npz",
        unit_id=unit_id[complete_mask],
        patient_id=patient_id[complete_mask],
        embeddings=complete_x,
        model_name=np.asarray("OpenAI CLIP ViT-B/16"),
        aggregation=np.asarray("arithmetic mean over available L2-normalized CC/MLO view vectors"),
    )
    return {"views": 1140, "breasts": 655, "dimension": 512,
            "max_view_norm_error": float(np.max(np.abs(norms - 1)))}


def main() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    torch.manual_seed(20261010)
    model = torch.jit.load(str(CHECKPOINT), map_location="cuda").eval()
    smoke = encode(model, [torch.zeros((3, 224, 224), dtype=torch.float32)])
    if smoke.shape != (1, 512) or abs(float(np.linalg.norm(smoke[0])) - 1.0) > 5e-5:
        raise RuntimeError("CLIP smoke test failed")
    audit = {
        "status": "PASS",
        "model_name": "OpenAI CLIP ViT-B/16",
        "checkpoint_path": str(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "smoke_shape": list(smoke.shape),
        "smoke_norm": float(np.linalg.norm(smoke[0])),
        "preprocessing": "unchanged Paper B pixels: 224x224 and CLIP mean/std; only encoder changed",
        "cbis": extract_cbis(model),
        "derm": extract_derm(model),
    }
    (OUT / "FEATURE_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
