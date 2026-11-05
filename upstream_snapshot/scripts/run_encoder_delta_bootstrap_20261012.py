"""Prepare aligned BiomedCLIP/CLIP predictions and run paired bootstraps."""
from __future__ import annotations

from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_b_readiness/ENCODER_DELTA_20261012"
BMC_PATH = ROOT / "paper_b_readiness/CFLEX_CONTROL_20260917/aggregate/all_predictions.csv"
CLIP_PATH = ROOT / "paper_b_readiness/ENCODER_ROBUSTNESS_CLIP_20261010/analysis1_predictions.csv"
BOOTSTRAP_SCRIPT = OUT / "encoder_delta_bootstrap.py"


def build(domain: str) -> Path:
    bmc = pd.read_csv(BMC_PATH).query("domain == @domain").copy()
    clip = pd.read_csv(CLIP_PATH).query("domain == @domain").copy()
    keys = ["repetition", "case_id"]
    bmc = bmc.sort_values(keys)
    clip = clip.sort_values(keys)
    np.testing.assert_array_equal(bmc[keys].to_numpy(), clip[keys].to_numpy())
    np.testing.assert_array_equal(bmc.y.to_numpy(dtype=int), clip.label.to_numpy(dtype=int))
    np.testing.assert_array_equal(bmc.group_id.astype(str).to_numpy(), clip.group_id.astype(str).to_numpy())

    case_ids = np.sort(bmc.case_id.astype(str).unique())
    repetitions = np.sort(bmc.repetition.unique())
    y_by_case = bmc.groupby("case_id").y.first().reindex(case_ids)
    group_by_case = bmc.groupby("case_id").group_id.first().astype(str).reindex(case_ids)
    assert bmc.groupby("case_id").y.nunique().eq(1).all()
    assert bmc.groupby("case_id").group_id.nunique().eq(1).all()

    def matrix(frame: pd.DataFrame, column: str) -> np.ndarray:
        pivot = frame.pivot(index="repetition", columns="case_id", values=column)
        pivot = pivot.reindex(index=repetitions, columns=case_ids)
        assert not pivot.isna().any().any()
        return pivot.to_numpy(dtype=float)

    path = OUT / f"{domain}_aligned_predictions.npz"
    np.savez_compressed(
        path,
        y=y_by_case.to_numpy(dtype=int),
        group=group_by_case.to_numpy(dtype=str),
        case_id=case_ids,
        bmc_cflex=matrix(bmc, "p_Cflex"),
        bmc_m=matrix(bmc, "p_M"),
        clip_cflex=matrix(clip, "Cflex"),
        clip_m=matrix(clip, "M"),
    )
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not BOOTSTRAP_SCRIPT.is_file():
        raise FileNotFoundError(f"Bundled bootstrap helper is missing: {BOOTSTRAP_SCRIPT}")
    outputs = {}
    for domain, seed in [("cbis", 20261021), ("derm", 20261022)]:
        input_path = build(domain)
        output_path = OUT / f"{domain}_encoder_delta.json"
        subprocess.run([
            sys.executable, str(BOOTSTRAP_SCRIPT), str(input_path),
            "--n-boot", "10000", "--seed", str(seed), "--out", str(output_path),
        ], check=True)
        outputs[domain] = json.loads(output_path.read_text())
    (OUT / "PROTOCOL.md").write_text(
        "# Paired encoder-effect comparison\n\n"
        "BiomedCLIP and CLIP predictions are aligned by cohort, repetition, and case. "
        "Each bootstrap draw resamples the same patient clusters for both encoders. "
        "The reported encoder contrast is the CLIP Cflex-to-M gain minus the BiomedCLIP "
        "Cflex-to-M gain. CBIS-DDSM uses seed 20261021 and Derm7pt uses seed 20261022, "
        "with 10,000 draws per cohort.\n"
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
