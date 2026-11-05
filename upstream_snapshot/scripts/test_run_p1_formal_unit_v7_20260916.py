#!/usr/bin/env python3
"""Hermetic, result-blind tests for the unauthorised P1 repair candidates.

The tests use only invented selection curves, row partitions, manifests and
deliberately invalid ticket text. They do not enumerate or open project inputs, result payloads, pilot
outputs, P2 assets, or official-test assets, and they cannot execute a unit.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


HERE = Path(__file__).resolve().parent
ENGINE_PATH = HERE / "run_p1_formal_unit_v5_20260916.py"
ORCHESTRATOR_PATH = HERE / "run_p1_formal_orchestrator_v7_20260916.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("test module loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


engine = load_module("p1_formal_unit_v5_candidate", ENGINE_PATH)
orchestrator = load_module("p1_formal_orchestrator_v7_candidate", ORCHESTRATOR_PATH)


def synthetic_curve():
    means = (0.60, 0.50, 0.40, 0.40, 0.45, 0.55)
    archived = []
    candidates = []
    for c_value, mean in zip(engine.GRID, means, strict=True):
        fold_losses = [mean, mean, mean, mean]
        archived.append(
            {
                "condition": "raw_soft_nested",
                "C": c_value,
                "inner_nll": mean,
                "fold_losses": fold_losses,
            }
        )
        candidates.append(
            {
                "C": c_value,
                "fold_class_equal_NLL": fold_losses,
                "mean_class_equal_NLL": mean,
            }
        )
    computed = {
        "C_grid": list(engine.GRID),
        "candidates": candidates,
        "selected_index": 2,
        "selected_C": engine.GRID[2],
        "tie_break": "first_minimum_smallest_C",
    }
    return archived, computed


class GenerationPartitionContractTests(unittest.TestCase):
    def fixture(self, domain):
        generation = {"selection_manifests": [{}, {}, {}, {}],
                      "probe_bundles": 11 if domain == "cbis" else 21,
                      "seconds": 2.5}
        if domain == "cbis":
            generation["manifests"] = []
        t = engine.np.arange(8)
        e = engine.np.array([8, 9])
        hrows = [engine.np.array(x) for x in ([2, 6], [0, 4], [3, 7], [1, 5])]
        return generation, t, e, hrows

    def test_timing_metadata_and_permuted_valid_fold_partition_are_accepted(self):
        for domain in ("cbis", "derm"):
            generation, t, e, hrows = self.fixture(domain)
            self.assertFalse(engine.np.array_equal(engine.np.concatenate(hrows), t))
            self.assertIs(engine.validate_generation_partition(generation, domain, t, e, hrows),
                          generation["selection_manifests"])

    def test_partition_and_metadata_mutations_fail_closed(self):
        for domain in ("cbis", "derm"):
            for mutation in ("overlap", "duplicate", "missing", "extra", "metadata",
                             "missing_seconds", "wrong_probe", "wrong_selection_count"):
                with self.subTest(domain=domain, mutation=mutation):
                    generation, t, e, hrows = self.fixture(domain)
                    if mutation == "overlap": e = engine.np.array([0, 9])
                    elif mutation == "duplicate": hrows[0] = engine.np.array([2, 6, 6])
                    elif mutation == "missing": hrows[0] = engine.np.array([2])
                    elif mutation == "extra": hrows[0] = engine.np.array([2, 10])
                    elif mutation == "metadata": generation["unknown"] = 1
                    elif mutation == "missing_seconds": generation.pop("seconds")
                    elif mutation == "wrong_probe": generation["probe_bundles"] = 0
                    else: generation["selection_manifests"] = [{}]
                    with self.assertRaises(engine.FormalError):
                        engine.validate_generation_partition(generation, domain, t, e, hrows)


class CbisFinalManifestContractTests(unittest.TestCase):
    def fixture(self):
        groups = engine.np.array([f"invented-patient-{i}" for i in range(10)])
        source_fold = engine.np.repeat(engine.np.arange(5), 2)
        data = {"t": engine.np.arange(2, 10), "e": engine.np.array([0, 1])}
        # The H order need not equal source fold order.
        for h, fold in enumerate((3, 1, 4, 2)):
            data[f"h{h}"] = engine.np.flatnonzero(source_fold == fold)
        panels = [{"outer_fold": 0, "diagnosis_selection_heldout_fold": fold,
                   "fit_heldout_patient_disjoint": True} for fold in range(1, 5)]
        final = {"outer_fold": 0, "role": "final_outer_train_probe",
                 "fit_patient_sha256": engine.sha256_values(groups[data["t"]]),
                 "validation_patient_sha256": engine.sha256_values(groups[data["e"]]),
                 "patient_disjoint": True}
        generation = {"manifests": [*panels, final],
                      "selection_manifests": [[panel] for panel in panels]}
        return generation, data, groups, source_fold, final

    def test_producer_final_patient_disjoint_schema_is_accepted(self):
        generation, data, groups, source_fold, final = self.fixture()
        resolved, actual_final = engine.resolve_cbis_manifests(
            generation, data, groups, source_fold, 0)
        self.assertIs(actual_final, final)
        self.assertNotIn("fit_heldout_patient_disjoint", final)
        self.assertEqual([m["diagnosis_selection_heldout_fold"] for m in resolved],
                         [3, 1, 4, 2])

    def test_final_disjointness_and_scope_mutations_fail_closed(self):
        for mutation in ("missing", "false", "integer", "string", "legacy_alias",
                         "wrong_fit_scope", "wrong_validation_scope", "wrong_outer_fold",
                         "nested_panel"):
            with self.subTest(mutation=mutation):
                generation, data, groups, source_fold, final = self.fixture()
                if mutation == "missing": final.pop("patient_disjoint")
                elif mutation == "false": final["patient_disjoint"] = False
                elif mutation == "integer": final["patient_disjoint"] = 1
                elif mutation == "string": final["patient_disjoint"] = "true"
                elif mutation == "legacy_alias":
                    final.pop("patient_disjoint")
                    final["fit_heldout_patient_disjoint"] = True
                elif mutation == "wrong_fit_scope": final["fit_patient_sha256"] = "0" * 64
                elif mutation == "wrong_validation_scope":
                    final["validation_patient_sha256"] = "0" * 64
                elif mutation == "wrong_outer_fold": final["outer_fold"] = 1
                else: final["nested_probe_panels"] = [{}]
                with self.assertRaisesRegex(engine.FormalError,
                                            "CBIS final generation scope mismatch"):
                    engine.resolve_cbis_manifests(generation, data, groups, source_fold, 0)


class ArchiveSelectionContractTests(unittest.TestCase):
    def assert_contract_rejected(self, archived, computed):
        with self.assertRaises(engine.FormalError) as caught:
            engine.validate_archive_selection_curve(
                archived, "raw_soft_nested", engine.GRID[2], computed
            )
        self.assertEqual(caught.exception.code, "E_ARCHIVE_SELECTION_CONTRACT")

    def test_six_candidate_curve_is_the_valid_archive_contract(self):
        archived, computed = synthetic_curve()
        engine.validate_archive_selection_curve(
            archived, "raw_soft_nested", engine.GRID[2], computed
        )

    def test_historical_singleton_assumption_fails_with_fixed_code(self):
        archived, computed = synthetic_curve()
        self.assert_contract_rejected(archived[:1], computed)

    def test_computed_curve_mismatch_fails_closed(self):
        archived, computed = synthetic_curve()
        computed["candidates"][4]["fold_class_equal_NLL"][1] += 0.01
        self.assert_contract_rejected(archived, computed)

    def test_first_minimum_tie_break_is_enforced(self):
        archived, computed = synthetic_curve()
        computed["selected_index"] = 3
        computed["selected_C"] = engine.GRID[3]
        with self.assertRaises(engine.FormalError):
            engine.validate_archive_selection_curve(
                archived, "raw_soft_nested", engine.GRID[2], computed
            )

    def test_archived_curve_boundary_mutations_fail_closed(self):
        mutations = {
            "wrong_grid_order": lambda rows: rows.__setitem__(
                slice(0, 2), [rows[1], rows[0]]
            ),
            "missing_candidate": lambda rows: rows.pop(),
            "wrong_fold_count": lambda rows: rows[1].__setitem__(
                "fold_losses", rows[1]["fold_losses"][:3]
            ),
            "nonfinite_fold_loss": lambda rows: rows[1]["fold_losses"].__setitem__(
                0, float("nan")
            ),
            "nonfinite_mean": lambda rows: rows[1].__setitem__(
                "inner_nll", float("inf")
            ),
            "mean_not_fold_mean": lambda rows: rows[1].__setitem__(
                "inner_nll", rows[1]["inner_nll"] + 0.01
            ),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label):
                archived, computed = synthetic_curve()
                mutation(archived)
                self.assert_contract_rejected(archived, computed)

    def test_computed_curve_boundary_mutations_fail_closed(self):
        def wrong_c(value):
            value["candidates"][1]["C"] = engine.GRID[2]

        def wrong_fold_count(value):
            value["candidates"][1]["fold_class_equal_NLL"] = [0.5] * 3

        def nonfinite_loss(value):
            value["candidates"][1]["fold_class_equal_NLL"][0] = float("nan")

        def wrong_mean(value):
            value["candidates"][1]["mean_class_equal_NLL"] += 0.01

        mutations = {
            "wrong_candidate_C": wrong_c,
            "wrong_fold_count": wrong_fold_count,
            "nonfinite_loss": nonfinite_loss,
            "wrong_candidate_mean": wrong_mean,
            "missing_candidate": lambda value: value["candidates"].pop(),
            "wrong_selected_C": lambda value: value.__setitem__(
                "selected_C", engine.GRID[3]
            ),
        }
        for label, mutation in mutations.items():
            with self.subTest(label=label):
                archived, computed = synthetic_curve()
                mutation(computed)
                self.assert_contract_rejected(archived, computed)


class SanitizedFailureReceiptTests(unittest.TestCase):
    def test_receipt_has_exact_non_data_bearing_schema(self):
        engine.set_failure_stage("COMPUTE_ARCHIVE_SELECTION_REPRODUCTION")
        receipt = engine.sanitized_failure_receipt(
            engine.FormalError(
                "invented secret must not escape",
                code="E_ARCHIVE_SELECTION_CONTRACT",
            )
        )
        self.assertEqual(
            receipt,
            {
                "status": "PAYLOAD_FAILED_CLOSED",
                "stage": "COMPUTE_ARCHIVE_SELECTION_REPRODUCTION",
                "error_code": "E_ARCHIVE_SELECTION_CONTRACT",
                "official_test_accessed": False,
                "hypothesis_judgment_performed": False,
            },
        )
        self.assertNotIn("secret", json.dumps(receipt))

    def test_unexpected_exception_maps_to_fixed_code_without_exception_text(self):
        engine.set_failure_stage("BOOT")
        receipt = engine.sanitized_failure_receipt(
            ValueError("SYNTHETIC_EXCEPTION_TEXT_MUST_NOT_ESCAPE")
        )
        self.assertEqual(receipt["error_code"], "E_UNEXPECTED_EXCEPTION")
        self.assertNotIn("SYNTHETIC", json.dumps(receipt))

    def test_engine_and_orchestrator_allowlists_are_exactly_identical(self):
        self.assertEqual(
            frozenset(engine.FAILURE_STAGES), orchestrator.ENGINE_FAILURE_STAGES
        )
        self.assertEqual(
            frozenset(engine.FAILURE_ERROR_CODES),
            orchestrator.ENGINE_FAILURE_ERROR_CODES,
        )

    def test_invalid_ticket_cannot_execute_and_emits_only_sanitized_receipt(self):
        secret = "DO_NOT_ECHO_THIS_SYNTHETIC_SECRET"
        completed = subprocess.run(
            [sys.executable, str(ENGINE_PATH), "--formal-ticket-stdin"],
            input=json.dumps({"synthetic_secret": secret}) + "\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, **engine._THREAD_ENV},
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, "")
        self.assertNotIn(secret, completed.stderr)
        receipt = json.loads(completed.stderr)
        self.assertEqual(receipt["stage"], "TICKET_VALIDATE")
        self.assertEqual(receipt["error_code"], "E_FORMAL_CONTRACT")
        self.assertEqual(
            set(receipt),
            {
                "status",
                "stage",
                "error_code",
                "official_test_accessed",
                "hypothesis_judgment_performed",
            },
        )

    def test_orchestrator_accepts_only_exact_allowlisted_failure_receipt(self):
        valid = json.dumps(
            {
                "status": "PAYLOAD_FAILED_CLOSED",
                "stage": "COMPUTE_ARCHIVE_SELECTION_REPRODUCTION",
                "error_code": "E_ARCHIVE_SELECTION_CONTRACT",
                "official_test_accessed": False,
                "hypothesis_judgment_performed": False,
            }
        )
        self.assertEqual(
            orchestrator.parse_sanitized_engine_failure(valid),
            (
                "COMPUTE_ARCHIVE_SELECTION_REPRODUCTION",
                "E_ARCHIVE_SELECTION_CONTRACT",
            ),
        )
        extra = json.loads(valid)
        extra["raw_error"] = "must be rejected"
        self.assertIsNone(
            orchestrator.parse_sanitized_engine_failure(json.dumps(extra))
        )
        bad_stage = json.loads(valid)
        bad_stage["stage"] = "DATA_VALUE_OR_PATH"
        self.assertIsNone(
            orchestrator.parse_sanitized_engine_failure(json.dumps(bad_stage))
        )
        self.assertIsNone(
            orchestrator.parse_sanitized_engine_failure(valid + "\n" + valid)
        )

    def test_orchestrator_rejects_all_malformed_receipt_boundaries(self):
        baseline = {
            "status": "PAYLOAD_FAILED_CLOSED",
            "stage": "COMPUTE_ARCHIVE_SELECTION_REPRODUCTION",
            "error_code": "E_ARCHIVE_SELECTION_CONTRACT",
            "official_test_accessed": False,
            "hypothesis_judgment_performed": False,
        }
        malformed = {
            "empty": "",
            "not_json": "not-json",
            "non_object": "[]",
            "unknown_status": json.dumps({**baseline, "status": "FAILED"}),
            "unknown_stage": json.dumps({**baseline, "stage": "SECRET_STAGE"}),
            "unknown_code": json.dumps({**baseline, "error_code": "E_SECRET"}),
            "official_access_true": json.dumps(
                {**baseline, "official_test_accessed": True}
            ),
            "judgment_true": json.dumps(
                {**baseline, "hypothesis_judgment_performed": True}
            ),
            "numeric_false": json.dumps(
                {**baseline, "official_test_accessed": 0}
            ),
            "extra_key": json.dumps({**baseline, "detail": "secret"}),
            "multi_line": json.dumps(baseline) + "\n" + json.dumps(baseline),
            "duplicate_key": json.dumps(baseline)[:-1]
            + ',"status":"PAYLOAD_FAILED_CLOSED"}',
            "oversized": "x" * 1025,
        }
        for label, text in malformed.items():
            with self.subTest(label=label):
                self.assertIsNone(orchestrator.parse_sanitized_engine_failure(text))

    def test_invoke_engine_discards_raw_stderr_and_rejects_stdout_on_failure(self):
        valid_receipt = json.dumps(
            {
                "status": "PAYLOAD_FAILED_CLOSED",
                "stage": "COMPUTE_ARCHIVE_SELECTION_REPRODUCTION",
                "error_code": "E_ARCHIVE_SELECTION_CONTRACT",
                "official_test_accessed": False,
                "hypothesis_judgment_performed": False,
            }
        )
        ticket = {"ticket_sha256": "synthetic-ticket"}
        unit = ("cbis", 0, 0)

        completed = SimpleNamespace(returncode=2, stdout="", stderr=valid_receipt)
        with mock.patch.object(orchestrator.subprocess, "run", return_value=completed):
            with self.assertRaises(orchestrator.FormalError) as caught:
                orchestrator.invoke_engine(ticket, unit, {})
        message = str(caught.exception)
        self.assertIn("COMPUTE_ARCHIVE_SELECTION_REPRODUCTION", message)
        self.assertIn("E_ARCHIVE_SELECTION_CONTRACT", message)

        secret = "SYNTHETIC_RAW_STDERR_MUST_NOT_ESCAPE"
        for label, completed in {
            "raw_stderr": SimpleNamespace(
                returncode=2, stdout="", stderr=secret
            ),
            "multiline": SimpleNamespace(
                returncode=2,
                stdout="",
                stderr=valid_receipt + "\n" + secret,
            ),
            "failure_stdout": SimpleNamespace(
                returncode=2, stdout="synthetic-output", stderr=valid_receipt
            ),
        }.items():
            with self.subTest(label=label):
                with mock.patch.object(
                    orchestrator.subprocess, "run", return_value=completed
                ):
                    with self.assertRaises(orchestrator.FormalError) as caught:
                        orchestrator.invoke_engine(ticket, unit, {})
                self.assertNotIn(secret, str(caught.exception))
                self.assertNotIn("synthetic-output", str(caught.exception))


class CbisMixedConceptFitContractTests(unittest.TestCase):
    def fixture(self, *, prediction_tail=True):
        np = engine.np
        names = list(engine.CBIS_CONCEPTS)
        ids = np.array(["invented-fit-0", "invented-fit-1", "invented-predict-0"])
        groups = np.array(["invented-patient-0", "invented-patient-1", "invented-patient-2"])
        fit_rows, predict_rows = np.array([0, 1]), np.array([2])
        gold = np.tile(np.array([[0], [1], [0]]), (1, len(names)))
        valid = np.ones(gold.shape, dtype=bool)
        records = [{
            "concept": name, "fit_breasts": 2, "fit_patients": 2,
            "negative": 1, "positive": 1,
            "fit_unit_sha256": engine.sha256_values(ids[fit_rows]),
            "invalid_targets_excluded_by_family_validity_mask": True,
            "fixed_probe_C": 1.0, "parameter_count": 2,
        } for name in names]
        if prediction_tail:
            records.append({
                "prediction_role": "CASE_EXCLUDED_PREDICTED_CONCEPT_EXPOSURE",
                "predict_breasts": 1, "predict_patients": 1,
                "predict_unit_sha256": engine.sha256_values(ids[predict_rows]),
                "fit_predict_patient_disjoint": True,
                "patient_disjoint_required": True, "overlap_patient_count": 0,
            })
        return records, names, fit_rows, predict_rows, gold, valid, ids, groups

    def read(self, fixture, *, require_prediction=True):
        records, names, fit_rows, predict_rows, gold, valid, ids, groups = fixture
        return engine.prior_lineage_records(
            "invented_H0_G0_OOF", fit_rows, predict_rows, gold, None, ids, groups,
            valid, records, names, "invented_generation_manifest",
            prediction_manifest_required=require_prediction)

    def test_nested_ten_concept_records_plus_prediction_tail_are_accepted_without_fits(self):
        fixture = self.fixture()
        with mock.patch.object(engine, "fit_head", side_effect=AssertionError("no fits")) as fit:
            result = self.read(fixture)
        fit.assert_not_called()
        self.assertEqual(len(result), len(engine.CBIS_CONCEPTS))
        self.assertTrue(all(item["frozen_fit_manifest_verified"] for item in result))
        self.assertEqual([item["concept_name"] for item in result], list(engine.CBIS_CONCEPTS))

    def test_plain_h_and_final_records_remain_strictly_ten_concepts(self):
        with mock.patch.object(engine, "fit_head", side_effect=AssertionError("no fits")) as fit:
            self.assertEqual(len(self.read(self.fixture(prediction_tail=False),
                                          require_prediction=False)), 10)
            with self.assertRaises(engine.FormalError):
                self.read(self.fixture(), require_prediction=False)
            with self.assertRaises(engine.FormalError):
                self.read(self.fixture(prediction_tail=False))
        fit.assert_not_called()

    def test_mixed_record_schema_and_prediction_range_mutations_fail_closed(self):
        mutations = {
            "prediction_missing": lambda x: x[0].pop(),
            "prediction_duplicate": lambda x: x[0].append(dict(x[0][-1])),
            "concept_duplicate": lambda x: x[0].__setitem__(1, dict(x[0][0])),
            "concept_unknown": lambda x: x[0][0].__setitem__("concept", "unknown"),
            "concept_nonmapping": lambda x: x[0].__setitem__(0, "not a mapping"),
            "concept_extra_key": lambda x: x[0][0].__setitem__("unexpected", True),
            "concept_missing_key": lambda x: x[0][0].pop("fit_unit_sha256"),
            "concept_bad_fit_hash": lambda x: x[0][0].__setitem__("fit_unit_sha256", "0" * 64),
            "concept_bad_count": lambda x: x[0][0].__setitem__("positive", 99),
            "prediction_extra_key": lambda x: x[0][-1].__setitem__("unexpected", True),
            "prediction_missing_key": lambda x: x[0][-1].pop("prediction_role"),
            "prediction_role": lambda x: x[0][-1].__setitem__("prediction_role", "IN_SAMPLE_AUDIT_TRAIN_EXPOSURE"),
            "prediction_rows": lambda x: x[0][-1].__setitem__("predict_breasts", 2),
            "prediction_rows_bool": lambda x: x[0][-1].__setitem__("predict_breasts", True),
            "prediction_patients": lambda x: x[0][-1].__setitem__("predict_patients", 2),
            "prediction_patients_bool": lambda x: x[0][-1].__setitem__("predict_patients", True),
            "prediction_hash": lambda x: x[0][-1].__setitem__("predict_unit_sha256", "0" * 64),
            "prediction_disjoint": lambda x: x[0][-1].__setitem__("fit_predict_patient_disjoint", False),
            "prediction_disjoint_integer": lambda x: x[0][-1].__setitem__("fit_predict_patient_disjoint", 1),
            "prediction_required": lambda x: x[0][-1].__setitem__("patient_disjoint_required", False),
            "prediction_required_integer": lambda x: x[0][-1].__setitem__("patient_disjoint_required", 1),
            "prediction_overlap": lambda x: x[0][-1].__setitem__("overlap_patient_count", 1),
            "prediction_overlap_bool": lambda x: x[0][-1].__setitem__("overlap_patient_count", False),
            "actual_patient_overlap": lambda x: x[7].__setitem__(2, x[7][0]),
            "prediction_wrong_position": lambda x: x[0].insert(0, x[0].pop()),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label):
                fixture = self.fixture()
                mutate(fixture)
                with mock.patch.object(engine, "fit_head", side_effect=AssertionError("no fits")) as fit:
                    with self.assertRaises(engine.FormalError):
                        self.read(fixture)
                fit.assert_not_called()


class VersionedTrustChainTests(unittest.TestCase):
    def test_v7_uses_exact_distinct_versioned_gate_paths(self):
        expected = {
            "spec": "paper_b_readiness/P1_FORMAL_ORCHESTRATION_SPEC_V7_20260916.json",
            "audit": "paper_b_readiness/P1_FORMAL_ENGINE_AUDIT_V7_20260916.json",
            "authorization": "paper_b_readiness/P1_FORMAL_AUTHORIZATION_V7_20260916.json",
        }
        self.assertEqual(orchestrator.SPEC_REL, expected["spec"])
        self.assertEqual(orchestrator.ENGINE_AUDIT_REL, expected["audit"])
        self.assertEqual(orchestrator.AUTHORIZATION_REL, expected["authorization"])
        # Gate documents are deliberately created later in the trust-chain
        # lifecycle.  Test the pure path contract, not live project absence.
        self.assertEqual(len(set(expected.values())), 3)
        self.assertTrue(all(Path(relative).parent == Path("paper_b_readiness")
                            for relative in expected.values()))

    def test_v7_never_reuses_legacy_gate_paths(self):
        legacy = {
            "paper_b_readiness/P1_FORMAL_ENGINE_AUDIT.json",
            "paper_b_readiness/P1_FORMAL_AUTHORIZATION.json",
        }
        self.assertNotIn(orchestrator.ENGINE_AUDIT_REL, legacy)
        self.assertNotIn(orchestrator.AUTHORIZATION_REL, legacy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
