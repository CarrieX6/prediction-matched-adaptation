#!/usr/bin/env python3
"""Extract the frozen Paper B CBIS full-mammogram BiomedCLIP archive on CPU."""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p2_stagewise.cbis_xonly_embedding import (  # noqa: E402
    BiomedCLIPVisual,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_VIEW_COUNT,
    MODEL_NAME,
    PreprocessReceipt,
    load_input_manifest,
    ordered_source_digest,
    prepare_source,
    sha256_file,
    validate_embedding_arrays,
    write_deterministic_npz,
    write_json,
)


DEFAULT_CONFIG = ROOT / "configs/paper_b_cbis_mass_train_embedding_protocol_v1_20260910.json"


def runtime_versions() -> dict[str, object]:
    packages = {}
    for distribution, key in (
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("pydicom", "pydicom"),
        ("Pillow", "pillow"),
        ("timm", "timm"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
    ):
        packages[key] = importlib.metadata.version(distribution)
    return {
        "invocation": ".venv/bin/python",
        "python_executable_resolved": str(Path(sys.executable).resolve(strict=True)),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "device": "cpu",
        "cuda_visible_devices": "",
        "torch_num_threads": 8,
        "torch_num_interop_threads": 1,
        "data_loader_workers": 0,
        "batch_size": 16,
    }


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
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


def _verify_freeze(config_path: Path) -> tuple[dict[str, object], list, Path]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest_path = ROOT / config["freeze"]["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN_BEFORE_EXTRACTION_NO_SCIENTIFIC_OUTCOME":
        raise ValueError("embedding extraction lacks a valid pre-outcome freeze")
    if manifest.get("official_test_assets_accessed") is not False:
        raise ValueError("official-test access boundary changed")
    if manifest.get("config_sha256") != sha256_file(config_path):
        raise ValueError("embedding protocol changed after freeze")
    if config.get("runtime") != runtime_versions() or manifest.get("runtime") != runtime_versions():
        raise ValueError("active runtime differs from the frozen extraction runtime")
    for relative, record in manifest["project_files"].items():
        path = ROOT / relative
        if sha256_file(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"frozen project file changed: {relative}")
    checkpoint = Path(config["model"]["checkpoint_path"]).resolve(strict=True)
    if (
        sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256
        or manifest["external_checkpoint"]["sha256"] != EXPECTED_CHECKPOINT_SHA256
    ):
        raise ValueError("BiomedCLIP checkpoint changed after freeze")
    input_path = ROOT / config["inputs"]["input_manifest"]
    assets = load_input_manifest(input_path, ROOT)
    if sha256_file(input_path) != config["inputs"]["input_manifest_sha256"]:
        raise ValueError("embedding input manifest changed after freeze")
    if ordered_source_digest(assets) != config["inputs"]["ordered_source_digest_sha256"]:
        raise ValueError("ordered source identity changed after freeze")
    return config, assets, checkpoint


def _encode_all(
    model: BiomedCLIPVisual,
    assets: list,
    *,
    batch_size: int,
    verify_source_hashes: bool,
) -> tuple[np.ndarray, list[dict[str, int | str]]]:
    embeddings = np.zeros((len(assets), 512), dtype=np.float32)
    receipts: list[dict[str, int | str]] = []
    with torch.inference_mode():
        for start in range(0, len(assets), batch_size):
            stop = min(start + batch_size, len(assets))
            tensors = []
            for asset in assets[start:stop]:
                tensor, receipt = prepare_source(
                    asset, ROOT, verify_hash=verify_source_hashes
                )
                tensors.append(tensor)
                receipts.append(receipt)
            batch = torch.stack(tensors)
            features = model(batch).float()
            norms = features.norm(dim=1, keepdim=True)
            if not torch.isfinite(features).all() or not torch.isfinite(norms).all():
                raise ValueError("BiomedCLIP produced a non-finite feature")
            features = features / norms.clamp_min(1e-8)
            embeddings[start:stop] = features.cpu().numpy().astype(np.float32, copy=False)
            print(f"encoded {stop}/{len(assets)}", flush=True)
    return embeddings, receipts


def _write_receipts(
    path: Path,
    assets: list,
    preprocess: list[dict[str, int | str]],
    l2_norms: np.ndarray,
    replay_differences: np.ndarray,
) -> None:
    fields = tuple(PreprocessReceipt.__dataclass_fields__)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for asset, details, norm, replay in zip(
            assets, preprocess, l2_norms, replay_differences, strict=True
        ):
            writer.writerow(
                {
                    "image_id": asset.image_id,
                    "local_dicom_path": asset.local_dicom_path,
                    "source_dicom_sha256": asset.source_dicom_sha256,
                    "source_dicom_bytes": asset.source_dicom_bytes,
                    **details,
                    "normalized_l2_norm": format(float(norm), ".9g"),
                    "replay_max_abs_difference": format(float(replay), ".9g"),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("formal extraction requires CUDA_VISIBLE_DEVICES='' (CPU only)")
    config_path = args.config.resolve(strict=True)
    if config_path != DEFAULT_CONFIG.resolve(strict=True):
        raise ValueError("only the canonical frozen embedding protocol may run")
    config, assets, checkpoint = _verify_freeze(config_path)
    output_dir = ROOT / config["outputs"]["directory"]
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite embedding output: {output_dir}")
    if len(assets) != EXPECTED_VIEW_COUNT:
        raise ValueError("formal extraction requires exactly 1140 inputs")

    torch.manual_seed(int(config["determinism"]["seed"]))
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(config["runtime"]["torch_num_threads"]))
    torch.set_num_interop_threads(int(config["runtime"]["torch_num_interop_threads"]))
    model = BiomedCLIPVisual(checkpoint).float().eval().cpu()
    batch_size = int(config["runtime"]["batch_size"])

    print("formal CPU extraction pass", flush=True)
    embeddings, preprocess_receipts = _encode_all(
        model, assets, batch_size=batch_size, verify_source_hashes=True
    )
    print("full deterministic CPU replay pass", flush=True)
    replay_embeddings, replay_receipts = _encode_all(
        model, assets, batch_size=batch_size, verify_source_hashes=False
    )
    if preprocess_receipts != replay_receipts:
        raise ValueError("preprocessing receipts changed during deterministic replay")
    replay_difference = np.max(
        np.abs(embeddings.astype(np.float64) - replay_embeddings.astype(np.float64)), axis=1
    )
    replay_tolerance = float(config["determinism"]["full_replay_max_abs_tolerance"])
    if float(replay_difference.max()) > replay_tolerance:
        raise ValueError("full deterministic replay exceeded the frozen tolerance")
    l2_norms = np.linalg.norm(embeddings.astype(np.float64), axis=1).astype(np.float32)
    norm_tolerance = float(config["determinism"]["l2_norm_tolerance"])
    image_ids = np.asarray([asset.image_id for asset in assets], dtype=str)
    validate_embedding_arrays(
        assets, image_ids, embeddings, l2_norms, norm_tolerance=norm_tolerance
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent))
    try:
        archive_path = stage / config["outputs"]["archive_name"]
        receipt_path = stage / config["outputs"]["receipt_name"]
        audit_path = stage / config["outputs"]["audit_name"]
        write_deterministic_npz(
            archive_path,
            {
                "checkpoint_sha256": np.asarray(EXPECTED_CHECKPOINT_SHA256),
                "embeddings": embeddings,
                "image_id": image_ids,
                "l2_norm": l2_norms,
                "local_dicom_path": np.asarray(
                    [asset.local_dicom_path for asset in assets], dtype=str
                ),
                "model_name": np.asarray(MODEL_NAME),
                "source_dicom_sha256": np.asarray(
                    [asset.source_dicom_sha256 for asset in assets], dtype=str
                ),
            },
        )
        _write_receipts(
            receipt_path,
            assets,
            preprocess_receipts,
            l2_norms,
            replay_difference,
        )
        audit = {
            "schema_version": "paper-b-cbis-full-mammogram-embedding-audit-v1",
            "status": "PASS_METHOD_COMPARISON_READY",
            "date": "2026-09-10",
            "official_test_assets_accessed": False,
            "scientific_outcomes_computed": False,
            "input_view_count": len(assets),
            "output_view_count": len(image_ids),
            "unique_input_key_count": len({asset.image_id for asset in assets}),
            "unique_output_key_count": len(set(image_ids.tolist())),
            "embedding_shape": list(embeddings.shape),
            "embedding_dtype": str(embeddings.dtype),
            "all_finite": bool(np.isfinite(embeddings).all()),
            "l2_norm_min": float(l2_norms.min()),
            "l2_norm_max": float(l2_norms.max()),
            "l2_norm_max_abs_error": float(np.max(np.abs(l2_norms - 1.0))),
            "l2_norm_tolerance": norm_tolerance,
            "full_replay_view_count": len(assets),
            "full_replay_max_abs_difference": float(replay_difference.max()),
            "full_replay_max_abs_tolerance": replay_tolerance,
            "ordered_source_digest_sha256": ordered_source_digest(assets),
            "input_manifest_sha256": config["inputs"]["input_manifest_sha256"],
            "config_sha256": sha256_file(config_path),
            "freeze_manifest_sha256": sha256_file(ROOT / config["freeze"]["manifest"]),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "model_name": MODEL_NAME,
            "runtime": runtime_versions(),
            "preprocessing": config["preprocessing"],
            "access_assertions": {
                "input_manifest_only_contains_key_path_hash_and_bytes": True,
                "roi_pixels_or_masks_loaded": False,
                "pathology_concepts_folds_or_outcomes_loaded": False,
                "mass_test_metadata_pixels_labels_or_counts_loaded": False,
            },
        }
        write_json(audit_path, audit)
        output_names = [archive_path.name, receipt_path.name, audit_path.name]
        output_manifest = {
            "schema_version": "paper-b-cbis-full-mammogram-embedding-output-v1",
            "status": "PASS_ATOMIC_NO_OVERWRITE",
            "files": {
                name: {"sha256": sha256_file(stage / name), "bytes": (stage / name).stat().st_size}
                for name in sorted(output_names)
            },
        }
        write_json(stage / "OUTPUT_MANIFEST.json", output_manifest)

        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "checkpoint_sha256",
                "embeddings",
                "image_id",
                "l2_norm",
                "local_dicom_path",
                "model_name",
                "source_dicom_sha256",
            }:
                raise ValueError("staged embedding archive schema changed")
            validate_embedding_arrays(
                assets,
                archive["image_id"],
                archive["embeddings"],
                archive["l2_norm"],
                norm_tolerance=norm_tolerance,
            )
        reloaded_manifest = json.loads(
            (stage / "OUTPUT_MANIFEST.json").read_text(encoding="utf-8")
        )
        for name, record in reloaded_manifest["files"].items():
            if sha256_file(stage / name) != record["sha256"]:
                raise ValueError(f"staged output hash changed: {name}")
        _rename_directory_no_replace(stage, output_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(f"published outcome-blind embeddings atomically to {output_dir}")


if __name__ == "__main__":
    main()

