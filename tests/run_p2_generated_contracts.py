#!/usr/bin/env python3
"""Run reviewed generated-data tests, not historical project-source binding checks.

The unchanged P2 test source has four checks that hash its full historical project
closure, including a P0 manifest containing unrelated A provenance. That manifest
is deliberately not distributed. Exclusions here are a package boundary, not a
passing assertion about the frozen project binding or clinical reproduction.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "upstream_snapshot"
EXCLUDED = {
    "test_missing_approval_measure_cannot_load_or_fit",
    "test_binding_complete_and_candidate_not_authorized",
    "test_binding_source_tamper_fails",
    "test_execute_missing_approvals_fails_before_source_load",
}


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def main() -> int:
    for folder in ("scripts", "src", "tests"):
        sys.path.insert(0, str(SNAPSHOT / folder))
    source = SNAPSHOT / "tests/test_neurocomputing_p2_scientific_v3_20260916.py"
    spec = importlib.util.spec_from_file_location("reviewed_p2_generated_tests", source)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    cases = list(flatten(unittest.defaultTestLoader.loadTestsFromModule(module)))
    selected = [case for case in cases if case._testMethodName not in EXCLUDED]
    excluded = sorted(case._testMethodName for case in cases if case._testMethodName in EXCLUDED)
    if set(excluded) != EXCLUDED or len(cases) != 58 or len(selected) != 54:
        raise RuntimeError("reviewed P2 test inventory changed; do not broaden the allowlist")
    print(json.dumps({"full_project_binding_checks": "NOT_RUN", "excluded": excluded,
                      "reason": "historical full-project source closure is not included",
                      "generated_data_checks": len(selected), "clinical_inputs_opened": False}), flush=True)
    result = unittest.TextTestRunner(verbosity=1).run(unittest.TestSuite(selected))
    return int(not result.wasSuccessful())


if __name__ == "__main__":
    sys.exit(main())
