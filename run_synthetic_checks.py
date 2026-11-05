#!/usr/bin/env python3
"""Explicit synthetic-only allowlist, never clinical runner auto-discovery."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "upstream_snapshot"
ALLOWLIST = (
    ("Cflex mathematical/source adapter", ("-m", "pytest", "-q", "checks/test_b_cflex_independent_20260917.py")),
    ("P1 unit contracts", ("scripts/test_run_p1_formal_unit_v7_20260916.py",)),
    ("P1 aggregation contracts", ("tests/test_p1_complete_aggregate_synthetic_v7_20260916.py",)),
    ("P2 generated-data contracts (54; four full-project binding checks excluded)", ("../tests/run_p2_generated_contracts.py",)),
    ("P3 source-only upstream contracts", ("tests/test_paper_b_p3_u1_synthetic_20260916.py",)),
    ("P3 DEV runner contracts", ("tests/test_paper_b_p3_dev_synthetic_20260916.py",)),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="run only the named generated-data tests")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "NOT_RUN", "allowlist": [name for name, _ in ALLOWLIST]}))
        return 0
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(str(SNAPSHOT / directory) for directory in ("src", "scripts", "tests"))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        env[key] = "1"
    results = []
    for name, command in ALLOWLIST:
        print(f"SYNTHETIC ONLY: {name}", flush=True)
        # Disable pytest's cache writer; no broad discovery is ever invoked.
        if command[:2] == ("-m", "pytest"):
            command = (*command[:2], "-p", "no:cacheprovider", *command[2:])
        completed = subprocess.run([sys.executable, "-B", *command], cwd=SNAPSHOT, env=env, check=False)
        results.append({"name": name, "exit_code": completed.returncode})
    print(json.dumps({"clinical_reproduction": "NOT_RUN", "clinical_inputs_opened": False, "synthetic_results": results}, indent=2))
    return int(any(row["exit_code"] != 0 for row in results))


if __name__ == "__main__":
    sys.exit(main())
