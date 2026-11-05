#!/usr/bin/env python3
"""Extract paired frozen BiomedCLIP features after source and approval checks."""
from __future__ import annotations

import argparse
import json
import zlib
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from paper2_derm7pt.data import (
    EXPECTED_IDENTICAL_MODALITY_CASES,
    sha256_file,
)


MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
EXPECTED_CHECKPOINT_SHA256 = "52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be"


class DermImageDataset(Dataset):
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB")), index


class BiomedCLIPVisual(torch.nn.Module):
    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        state = torch.load(checkpoint, map_location="cpu")
        trunk_state = {
            key.removeprefix("visual.trunk."): value
            for key, value in state.items()
            if key.startswith("visual.trunk.")
        }
        self.trunk = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
        missing, unexpected = self.trunk.load_state_dict(trunk_state, strict=False)
        if missing or unexpected:
            raise ValueError(f"BiomedCLIP visual trunk mismatch: {missing=} {unexpected=}")
        projection = state.get("visual.head.proj.weight")
        if projection is None or tuple(projection.shape) != (512, 768):
            raise ValueError("BiomedCLIP visual projection is missing or malformed")
        self.projection = torch.nn.Linear(768, 512, bias=False)
        self.projection.weight.data.copy_(projection)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(self.trunk(image))


def _resolve(root: Path, relative: str) -> Path:
    candidates = (root / "images" / relative, root / relative)
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one official image for {relative!r}: {matches}")
    return matches[0]


def _member(names: set[str], relative: str) -> str:
    relative = PurePosixPath(relative).as_posix().lstrip("/")
    matches = [name for name in (f"release_v0/images/{relative}", f"release_v0/{relative}") if name in names]
    if len(matches) != 1:
        raise ValueError(f"archive member mismatch for {relative!r}: {matches}")
    return matches[0]


def _verify_extraction(paths: list[Path], relatives: list[str], source_zip: Path) -> None:
    with zipfile.ZipFile(source_zip) as archive:
        names = set(archive.namelist())
        for path, relative in zip(paths, relatives, strict=True):
            info = archive.getinfo(_member(names, relative))
            crc = 0
            size = 0
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    crc = zlib.crc32(block, crc)
                    size += len(block)
            if size != info.file_size or crc & 0xFFFFFFFF != info.CRC:
                raise ValueError(f"extracted image differs from archive: {relative}")


def _encode(model: BiomedCLIPVisual, paths: list[Path], device: torch.device, batch_size: int) -> np.ndarray:
    loader = DataLoader(DermImageDataset(paths), batch_size=batch_size, shuffle=False, num_workers=4)
    embeddings = np.zeros((len(paths), 512), dtype=np.float32)
    with torch.no_grad():
        for batch, (images, index) in enumerate(loader):
            feature = model(images.to(device)).float()
            feature /= feature.norm(dim=1, keepdim=True).clamp_min(1e-8)
            embeddings[index.numpy()] = feature.cpu().numpy()
            if batch % 10 == 0:
                print(f"encoded {min((batch + 1) * batch_size, len(paths))}/{len(paths)}", flush=True)
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-zip", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--candidate-protocol", type=Path, required=True)
    parser.add_argument("--author-approval", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    root = args.repo_root.resolve(strict=True)
    approval = json.loads(args.author_approval.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate_protocol.read_text(encoding="utf-8"))
    audit = json.loads(args.source_audit.read_text(encoding="utf-8"))
    if approval.get("author_decision") != "APPROVED_DERM7PT_FRESH_TO_A_MNPD":
        raise ValueError("feature extraction lacks the recorded author approval")
    if approval.get("approved_candidate_protocol_sha256") != sha256_file(args.candidate_protocol):
        raise ValueError("author approval does not bind the candidate protocol")
    if candidate.get("status") != "APPROVED_PREFEATURE_RESULT_BLIND_NOT_EXECUTABLE_FOR_OUTCOME":
        raise ValueError("candidate protocol status changed")
    if audit.get("status") != "PASS_WITH_RESULT_BLIND_IDENTICAL_MODALITY_EXCLUSIONS":
        raise ValueError("source audit did not pass")
    if sha256_file(args.source_zip) != candidate["dataset"]["archive_sha256"]:
        raise ValueError("official archive hash changed")
    if sha256_file(args.source_audit) != candidate["dataset"]["source_audit_sha256"]:
        raise ValueError("source audit hash changed")
    if sha256_file(args.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("BiomedCLIP checkpoint hash changed")
    if args.output.resolve() != (root / candidate["dataset"]["feature_archive"]).resolve():
        raise ValueError("feature output path differs from the candidate protocol")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite feature archive: {args.output}")

    metadata_path = args.data_root / "meta" / "meta.csv"
    metadata = pd.read_csv(metadata_path, dtype=str).fillna("")
    if sha256_file(metadata_path) != audit["metadata_sha256"]:
        raise ValueError("extracted metadata differs from the audited archive")
    excluded = tuple(map(int, audit["identical_clinical_dermoscopic_cases"]))
    if excluded != EXPECTED_IDENTICAL_MODALITY_CASES:
        raise ValueError("source exclusion set changed")
    eligible = np.asarray([index for index in range(len(metadata)) if index not in set(excluded)], dtype=np.int64)
    clinic_relative = metadata.iloc[eligible]["clinic"].astype(str).tolist()
    derm_relative = metadata.iloc[eligible]["derm"].astype(str).tolist()
    clinic_paths = [_resolve(args.data_root, value) for value in clinic_relative]
    derm_paths = [_resolve(args.data_root, value) for value in derm_relative]
    _verify_extraction(clinic_paths + derm_paths, clinic_relative + derm_relative, args.source_zip)

    device = torch.device(args.device)
    model = BiomedCLIPVisual(args.checkpoint).float().eval().to(device)
    print("encoding clinical deployment images", flush=True)
    clinical = _encode(model, clinic_paths, device, args.batch_size)
    print("encoding paired dermoscopic privileged images", flush=True)
    derm = _encode(model, derm_paths, device, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        metadata_index=eligible,
        clinic_path=np.asarray(clinic_relative, dtype=str),
        derm_path=np.asarray(derm_relative, dtype=str),
        clinical_embeddings=clinical,
        derm_embeddings=derm,
        model_name=np.asarray(MODEL_NAME),
        checkpoint_sha256=np.asarray(EXPECTED_CHECKPOINT_SHA256),
        metadata_sha256=np.asarray(audit["metadata_sha256"]),
        excluded_metadata_indices=np.asarray(excluded, dtype=np.int64),
    )
    print(f"saved {len(eligible)} eligible paired cases to {args.output}", flush=True)


if __name__ == "__main__":
    main()

