#!/usr/bin/env python3
"""Finalize the corrected V2 Paper B CBIS embedding extraction freeze."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p2_stagewise.cbis_xonly_embedding_v2 import (  # noqa: E402
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_VIEW_COUNT,
    load_input_manifest,
    ordered_source_digest,
    sha256_file,
    write_json,
)


DEFAULT_CONFIG = ROOT / "configs/paper_b_cbis_mass_train_embedding_protocol_v2_20260910.json"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    if config_path != DEFAULT_CONFIG.resolve(strict=True):
        raise ValueError("only the canonical V2 embedding protocol can be frozen")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "EXTRACTION_FROZEN_NO_SCIENTIFIC_OUTCOME":
        raise ValueError("V2 embedding protocol is not pre-extraction frozen")
    if config.get("supersedes_unexecuted_protocol") != (
        "configs/paper_b_cbis_mass_train_embedding_protocol_v1_20260910.json"
    ):
        raise ValueError("V2 does not identify the unexecuted V1 protocol")
    if config.get("official_test_assets_accessed") is not False:
        raise ValueError("official-test access boundary changed")
    if config.get("runtime") != runtime_versions():
        raise ValueError("active numeric runtime differs from the V2 protocol")

    inputs = config["inputs"]
    input_path = ROOT / inputs["input_manifest"]
    assets = load_input_manifest(input_path, ROOT)
    if (
        len(assets) != EXPECTED_VIEW_COUNT
        or len({asset.image_id for asset in assets}) != inputs["unique_key_count"]
        or len({asset.local_dicom_path for asset in assets})
        != inputs["unique_dicom_path_count"]
        or sum(asset.source_dicom_bytes for asset in assets)
        != inputs["total_source_dicom_bytes"]
    ):
        raise ValueError("V2 protocol source census changed")
    if sha256_file(input_path) != inputs["input_manifest_sha256"]:
        raise ValueError("input manifest hash changed")
    if ordered_source_digest(assets) != inputs["ordered_source_digest_sha256"]:
        raise ValueError("ordered source digest changed")
    for path_key, hash_key in (
        ("cohort_contract", "cohort_contract_sha256"),
        ("cohort_manifest", "cohort_manifest_sha256"),
        ("legacy_preprocess_audit", "legacy_preprocess_audit_sha256"),
    ):
        source = ROOT / inputs[path_key]
        if sha256_file(source) != inputs[hash_key]:
            raise ValueError(f"V2 protocol input hash changed: {path_key}")

    checkpoint = Path(config["model"]["checkpoint_path"]).resolve(strict=True)
    if (
        sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256
        or config["model"]["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256
        or checkpoint.stat().st_size != config["model"]["checkpoint_bytes"]
    ):
        raise ValueError("BiomedCLIP checkpoint identity changed")
    output = ROOT / config["outputs"]["directory"]
    if output.exists():
        raise FileExistsError("V2 output already exists; freeze is not pre-extraction")
    v1_output = ROOT / config["correction"]["v1_output_directory"]
    if v1_output.exists():
        raise FileExistsError("V1 output exists; it can no longer be called unexecuted")

    overlap_reference = Path(
        config["technical_overlap_audit"]["reference_archive"]
    ).resolve(strict=True)
    if (
        sha256_file(overlap_reference)
        != config["technical_overlap_audit"]["reference_sha256"]
        or overlap_reference.stat().st_size
        != config["technical_overlap_audit"]["reference_bytes"]
    ):
        raise ValueError("technical-only legacy overlap reference changed")

    frozen_paths = list(config["freeze"]["project_files"])
    if len(frozen_paths) != len(set(frozen_paths)):
        raise ValueError("duplicate project path in V2 freeze")
    files: dict[str, dict[str, object]] = {}
    for relative in frozen_paths:
        path = (ROOT / relative).resolve(strict=True)
        if not path.is_relative_to(ROOT.resolve(strict=True)) or not path.is_file():
            raise ValueError(f"invalid V2 project freeze path: {relative}")
        files[relative] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}

    manifest_path = ROOT / config["freeze"]["manifest"]
    if manifest_path.exists():
        raise FileExistsError("refusing to overwrite the V2 embedding freeze manifest")
    manifest = {
        "schema_version": "paper-b-cbis-full-mammogram-embedding-freeze-v2",
        "status": "FROZEN_BEFORE_EXTRACTION_NO_SCIENTIFIC_OUTCOME",
        "date": "2026-09-10",
        "supersedes_unexecuted_v1": True,
        "v1_output_exists_at_v2_freeze": False,
        "official_test_assets_accessed": False,
        "scientific_outcomes_computed": False,
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(config_path),
        "project_files": files,
        "source_dicoms": {
            "count": len(assets),
            "total_bytes": sum(asset.source_dicom_bytes for asset in assets),
            "input_manifest_sha256": sha256_file(input_path),
            "ordered_source_digest_sha256": ordered_source_digest(assets),
            "individual_sha256_frozen_in": input_path.relative_to(ROOT).as_posix(),
        },
        "external_checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
        },
        "external_technical_overlap_reference": {
            "path": str(overlap_reference),
            "sha256": sha256_file(overlap_reference),
            "bytes": overlap_reference.stat().st_size,
            "formal_extraction_input_role": "none",
            "permitted_arrays_after_formal_publication": ["image_id", "embeddings"],
        },
        "runtime": runtime_versions(),
        "preprocessing_contract": config["preprocessing"],
        "determinism_contract": config["determinism"],
        "output_contract": config["outputs"],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, manifest)
    print(
        f"wrote pre-extraction V2 freeze {manifest_path}; "
        f"sha256={sha256_file(manifest_path)}"
    )


if __name__ == "__main__":
    main()
