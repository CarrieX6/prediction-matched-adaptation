#!/usr/bin/env python3
"""Purely synthetic contract tests for the frozen P1 aggregate layer.

This test creates its own temporary 30-unit tree.  It does not inspect, list,
or import any pilot/formal outcome directory and it contains no official-test
path.  The reduced draw count is a test-only argument to the non-production
builder; the production entry point remains fixed at 10,000 draws.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "scripts" / "run_p1_complete_aggregate_v7_20260916.py"
MODULE_SPEC = importlib.util.spec_from_file_location("p1_aggregate_synthetic_subject", SOURCE)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError("cannot import aggregate source under test")
aggregate = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(aggregate)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


COMMON_UNIT_ANCHORS = {
    "p0_manifest_sha256": digest("synthetic-p0-manifest"),
    **aggregate.FORMAL_SOURCE_SHA256,
    "authorization_sha256": digest("synthetic-formal-authorization"),
    "engine_audit_sha256": digest("synthetic-formal-engine-audit"),
}

GATE_ANCHORS = {
    "protocol_sha256": aggregate.PROTOCOL_SHA256,
    "aggregate_script_sha256": aggregate.sha256_file(SOURCE),
    "aggregate_spec_sha256": aggregate.SPEC_SHA256,
    "units_sha256": aggregate.UNITS_SHA256,
    "source_audit_sha256": digest("synthetic-aggregate-source-audit"),
    "authorization_sha256": digest("synthetic-aggregate-authorization"),
}


def probability(case_position: int, arm_position: int, repetition: int) -> float:
    interaction = ((case_position + 1) * (arm_position + 2) + 3 * repetition) % 7
    z = -1.15 + 0.105 * case_position + 0.035 * arm_position + 0.075 * (interaction - 3)
    return 1.0 / (1.0 + math.exp(-z))


def prediction_rows(unit: tuple[str, int, int]) -> list[dict[str, object]]:
    domain, repetition, outer_fold = unit
    rows: list[dict[str, object]] = []
    for local_position in range(4):
        y = 0 if local_position < 2 else 1
        case_position = outer_fold * 4 + local_position
        case_id = f"{domain}_case_{case_position:02d}"
        if domain == "cbis":
            group_id = f"cbis_patient_f{outer_fold}_y{y}"
        else:
            group_id = f"derm_group_{case_position:02d}"
        probabilities = {
            arm: probability(case_position, arm_position, repetition)
            for arm_position, arm in enumerate(aggregate.ARMS)
        }
        row: dict[str, object] = {
            "domain": domain,
            "repetition": repetition,
            "outer_fold": outer_fold,
            "row_index": case_position,
            "case_id": case_id,
            "group_id": group_id,
            "y": y,
        }
        row.update({f"p_{arm}": aggregate.float_text(value) for arm, value in probabilities.items()})
        for arm in ("S", "C", "SP", "CP"):
            p = probabilities[arm]
            row[f"logit_{arm}"] = aggregate.float_text(math.log(p / (1.0 - p)))
        rows.append(row)
    return rows


def calibration_rows(unit: tuple[str, int, int], predictions: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in predictions:
        p = float(row["p_S"])
        logit = math.log(p / (1.0 - p))
        rows.append({
            "domain": unit[0],
            "repetition": unit[1],
            "outer_fold": unit[2],
            "row_index": row["row_index"],
            "case_id": row["case_id"],
            "group_id": row["group_id"],
            "y": row["y"],
            "H": 2,
            "reference_C_H": "1.5",
            "raw_base_logit": aggregate.float_text(logit),
            "prior_restored_base_logit": aggregate.float_text(logit + 0.01),
            "probe_fit_scope_sha256": digest(f"probe-{unit}"),
            "reference_fit_scope_sha256": digest(f"reference-{unit}"),
        })
    return rows


def per_case_rows(unit: tuple[str, int, int], predictions: list[dict[str, object]]) -> list[dict[str, object]]:
    counts = {0: 0, 1: 0}
    for row in predictions:
        counts[int(row["y"])] += 1
    rows: list[dict[str, object]] = []
    for arm in aggregate.ARMS:
        for prediction in predictions:
            y = int(prediction["y"])
            p = float(prediction[f"p_{arm}"])
            loss = aggregate.binary_nll(y, p)
            rows.append({
                "domain": unit[0],
                "repetition": unit[1],
                "outer_fold": unit[2],
                "row_index": prediction["row_index"],
                "case_id": prediction["case_id"],
                "group_id": prediction["group_id"],
                "y": y,
                "arm": arm,
                "negative_log_likelihood": aggregate.float_text(loss),
                "squared_probability_error": aggregate.float_text((p - y) ** 2),
                "class_equal_nll_contribution": aggregate.float_text(
                    loss / (2.0 * counts[y])
                ),
            })
    return rows


def metric_rows(unit: tuple[str, int, int], predictions: list[dict[str, object]]) -> list[dict[str, object]]:
    y = np.asarray([int(row["y"]) for row in predictions], dtype=np.int8)
    weight = np.ones(len(y), dtype=float)
    rows: list[dict[str, object]] = []
    for arm in aggregate.ARMS:
        p = np.asarray([float(row[f"p_{arm}"]) for row in predictions], dtype=float)
        losses = np.asarray([
            aggregate.binary_nll(int(label), float(probability_value))
            for label, probability_value in zip(y, p)
        ])
        rows.append({
            "domain": unit[0],
            "repetition": unit[1],
            "outer_fold": unit[2],
            "arm": arm,
            "class_equal_NLL": aggregate.float_text(aggregate.class_equal_mean(y, losses, weight)),
            "AUC": aggregate.float_text(aggregate.weighted_auc(y, p, weight)),
            "natural_frequency_NLL": aggregate.float_text(float(np.mean(losses))),
            "Brier": aggregate.float_text(float(np.mean((p - y) ** 2))),
            "n_cases": len(predictions),
            "n_groups": len({str(row["group_id"]) for row in predictions}),
        })
    return rows


def paired_rows(unit: tuple[str, int, int], metrics: list[dict[str, object]]) -> list[dict[str, object]]:
    index = {str(row["arm"]): row for row in metrics}
    rows: list[dict[str, object]] = []
    for name in aggregate.CONTRASTS:
        x_arm, y_arm = aggregate.CONTRAST_ARMS[name]
        rows.append({
            "domain": unit[0],
            "repetition": unit[1],
            "outer_fold": unit[2],
            "contrast": name,
            "class_equal_NLL_X_minus_Y": aggregate.float_text(
                float(index[x_arm]["class_equal_NLL"])
                - float(index[y_arm]["class_equal_NLL"])
            ),
            "AUC_X_minus_Y": aggregate.float_text(
                float(index[x_arm]["AUC"]) - float(index[y_arm]["AUC"])
            ),
            "favourable_sign_NLL": aggregate.CONTRAST_SIGNS[name],
            "pilot_descriptive_only": "false",
        })
    return rows


def write_unit(unit_path: Path, unit: tuple[str, int, int]) -> None:
    unit_path.mkdir(mode=0o700)
    predictions = prediction_rows(unit)
    metrics = metric_rows(unit, predictions)
    tables = {
        "predictions.csv": predictions,
        "calibration_oof.csv": calibration_rows(unit, predictions),
        "per_case_effects.csv": per_case_rows(unit, predictions),
        "metrics.csv": metrics,
        "paired_effects.csv": paired_rows(unit, metrics),
    }
    for name, schema in aggregate.CSV_SCHEMAS.items():
        aggregate.write_csv(unit_path / name, schema, tables[name])

    identity = aggregate.unit_object(unit)
    generic_payloads = {
        "selections.json": {"unit": identity, "synthetic": True},
        "calibrators.json": {"unit": identity, "synthetic": True},
        "prior_lineage.json": {"unit": identity, "synthetic": True},
        "lineage.json": {
            "unit": identity,
            "access_boundary": {
                "official_test_access": False,
                "official_test_discovery_or_listing": False,
                "input_whitelist_exact": True,
            },
        },
        "audit.json": {
            "unit": identity,
            "checks": {audit_id: True for audit_id in aggregate.FORMAL_AUDIT_IDS},
            "all_formal_prepublication_audits_pass": True,
            "hypothesis_judgment": "NOT_PERMITTED_PER_UNIT",
        },
        "fit_cost.json": {"unit": identity, "synthetic": True},
    }
    for name in aggregate.JSON_PAYLOADS:
        aggregate.write_new_json(unit_path / name, generic_payloads[name])
    aggregate.write_new_bytes(unit_path / "run.log", b"synthetic formal-unit contract fixture\n")

    whitelist = [{
        "role": "synthetic_contract_fixture",
        "path": f"synthetic-memory://{aggregate.unit_name(unit)}",
        "sha256": digest(f"whitelist-{aggregate.unit_name(unit)}"),
    }]
    artifacts = [{
        "path": name,
        "bytes": (unit_path / name).stat().st_size,
        "sha256": aggregate.sha256_file(unit_path / name),
    } for name in aggregate.PAYLOAD_FILES]
    manifest: dict[str, object] = {
        "unit": identity,
        "protocol_sha256": aggregate.PROTOCOL_SHA256,
        **aggregate.FORMAL_MANIFEST_PATHS,
        **COMMON_UNIT_ANCHORS,
        "units_sha256": aggregate.UNITS_SHA256,
        "fixed_input_whitelist": whitelist,
        "fixed_input_whitelist_sha256": aggregate.sha256_json(whitelist),
        "artifacts": artifacts,
        "artifact_count_excluding_manifest_and_COMPLETE": len(aggregate.PAYLOAD_FILES),
        "complete_excluded_because_written_last": True,
        "formal_prepublication_audit_ids": list(aggregate.FORMAL_AUDIT_IDS),
        "formal_publication_audit_id": aggregate.FORMAL_PUBLICATION_AUDIT_ID,
        "scientific_interpretation": "formal development unit only; no per-unit hypothesis judgment",
    }
    aggregate.write_new_json(unit_path / "execution_manifest.json", manifest)
    complete = {
        **identity,
        "status": "COMPLETE",
        "protocol_sha256": aggregate.PROTOCOL_SHA256,
        "execution_manifest_sha256": aggregate.sha256_file(unit_path / "execution_manifest.json"),
        "formal_prepublication_audit_ids": list(aggregate.FORMAL_AUDIT_IDS),
        "formal_publication_audit_id": aggregate.FORMAL_PUBLICATION_AUDIT_ID,
        "all_formal_publication_audits_pass": True,
        "completed_at_utc": "2026-09-16T00:00:00+00:00",
        "interpretation": "formal P1 development unit; aggregate judgment only after all units",
    }
    aggregate.write_new_json(unit_path / "COMPLETE.json", complete)


def populate_root(root: Path) -> None:
    units = root / aggregate.UNITS_PARENT_REL
    units.mkdir(parents=True, mode=0o700)
    for unit in aggregate.UNITS:
        write_unit(units / aggregate.unit_name(unit), unit)


def rewrite_csv(path: Path, schema: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    path.unlink()
    aggregate.write_csv(path, schema, rows)


def reseal_unit(unit_path: Path) -> None:
    manifest_path = unit_path / "execution_manifest.json"
    complete_path = unit_path / "COMPLETE.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"] = [{
        "path": name,
        "bytes": (unit_path / name).stat().st_size,
        "sha256": aggregate.sha256_file(unit_path / name),
    } for name in aggregate.PAYLOAD_FILES]
    manifest_path.unlink()
    aggregate.write_new_json(manifest_path, manifest)
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    complete["execution_manifest_sha256"] = aggregate.sha256_file(manifest_path)
    complete_path.unlink()
    aggregate.write_new_json(complete_path, complete)


class SyntheticAggregateContract(unittest.TestCase):
    def test_v7_source_chain_is_bound_to_v7_orchestrator_v5_engine(self) -> None:
        spec = json.loads((REPOSITORY / aggregate.SPEC_REL).read_text(encoding="utf-8"))
        self.assertEqual(aggregate.sha256_file(REPOSITORY / aggregate.SPEC_REL), aggregate.SPEC_SHA256)
        self.assertEqual(spec["status"], "SOURCE_ONLY_CANDIDATE_PRE_OUTCOME")
        self.assertEqual(spec["implementation"]["path"], str(SOURCE.relative_to(REPOSITORY)))
        self.assertEqual(spec["synthetic_contract_tests"]["path"], str(Path(__file__).resolve().relative_to(REPOSITORY)))
        self.assertEqual(spec["synthetic_contract_tests"]["sha256"], aggregate.sha256_file(Path(__file__).resolve()))
        self.assertEqual(spec["calibration_H_index_contract"]["allowed_ids"], [0, 1, 2, 3])
        self.assertEqual(spec["versioned_integration"]["formal_manifest_paths"], aggregate.FORMAL_MANIFEST_PATHS)
        self.assertEqual(spec["versioned_integration"]["formal_source_sha256"], aggregate.FORMAL_SOURCE_SHA256)
        self.assertEqual(aggregate.FORMAL_MANIFEST_PATHS["orchestrator_path"], "scripts/run_p1_formal_orchestrator_v7_20260916.py")
        self.assertEqual(aggregate.FORMAL_MANIFEST_PATHS["engine_path"], "scripts/run_p1_formal_unit_v5_20260916.py")
        self.assertEqual(aggregate.FORMAL_MANIFEST_PATHS["spec_path"], "paper_b_readiness/P1_FORMAL_ORCHESTRATION_SPEC_V7_20260916.json")
        for key, path_key in (("orchestrator_sha256", "orchestrator_path"), ("engine_sha256", "engine_path"), ("spec_sha256", "spec_path")):
            self.assertEqual(aggregate.sha256_file(REPOSITORY / aggregate.FORMAL_MANIFEST_PATHS[path_key]), aggregate.FORMAL_SOURCE_SHA256[key])

    def test_zero_based_calibration_h_ids_are_legal_and_malformed_ids_fail(self) -> None:
        # This complete, hash-sealed unit is wholly invented in a temporary tree.
        with tempfile.TemporaryDirectory(prefix="p1-aggregate-zero-based-h-") as directory:
            unit_path = Path(directory) / "cbis_r0_f0"
            unit = ("cbis", 0, 0)
            write_unit(unit_path, unit)
            calibration_path = unit_path / "calibration_oof.csv"
            rows = aggregate.read_csv_exact(calibration_path, aggregate.CSV_SCHEMAS["calibration_oof.csv"])
            for h, row in enumerate(rows):
                row["H"] = str(h)
            rewrite_csv(calibration_path, aggregate.CSV_SCHEMAS["calibration_oof.csv"], rows)
            reseal_unit(unit_path)
            tables, anchors = aggregate.validate_unit(unit_path, unit)
            self.assertEqual([row["H"] for row in tables["calibration_oof.csv"]], ["0", "1", "2", "3"])
            self.assertEqual(anchors, COMMON_UNIT_ANCHORS)
            for invalid_h in ("-1", "4", "1.5", "nan", ""):
                with self.subTest(invalid_h=invalid_h):
                    rows[0]["H"] = invalid_h
                    rewrite_csv(calibration_path, aggregate.CSV_SCHEMAS["calibration_oof.csv"], rows)
                    reseal_unit(unit_path)
                    with self.assertRaises(aggregate.AggregateError):
                        aggregate.validate_unit(unit_path, unit)
            # H=0 is not a regularization value: reference C must remain positive.
            rows[0]["H"] = "0"
            rows[0]["reference_C_H"] = "0"
            rewrite_csv(calibration_path, aggregate.CSV_SCHEMAS["calibration_oof.csv"], rows)
            reseal_unit(unit_path)
            with self.assertRaisesRegex(aggregate.AggregateError, "non-positive reference_C_H"):
                aggregate.validate_unit(unit_path, unit)

    def test_wrong_formal_source_hash_is_rejected_before_tables_are_read(self) -> None:
        for anchor in aggregate.FORMAL_SOURCE_SHA256:
            with self.subTest(anchor=anchor), tempfile.TemporaryDirectory(prefix="p1-aggregate-chain-rejection-") as directory:
                unit_path = Path(directory) / "cbis_r0_f0"
                unit = ("cbis", 0, 0)
                write_unit(unit_path, unit)
                manifest_path = unit_path / "execution_manifest.json"
                manifest = aggregate.load_json(manifest_path, "synthetic manifest")
                manifest[anchor] = digest("wrong-source-" + anchor)
                manifest_path.unlink()
                aggregate.write_new_json(manifest_path, manifest)
                complete_path = unit_path / "COMPLETE.json"
                complete = aggregate.load_json(complete_path, "synthetic COMPLETE")
                complete["execution_manifest_sha256"] = aggregate.sha256_file(manifest_path)
                complete_path.unlink()
                aggregate.write_new_json(complete_path, complete)
                with self.assertRaisesRegex(aggregate.AggregateError, "reviewed source hash changed"):
                    aggregate.validate_unit(unit_path, unit)

    def test_point_estimators_are_case_first_and_auc_is_repetition_first(self) -> None:
        rows: list[dict[str, object]] = []
        probabilities_by_repetition: list[np.ndarray] = []
        labels = np.asarray([0, 1] * 5, dtype=np.int8)
        for repetition in range(5):
            probabilities = np.asarray([
                0.05 + 0.9 * (((case_position * (repetition + 2) + 3 * repetition) % 10) / 9.0)
                for case_position in range(10)
            ])
            probabilities_by_repetition.append(probabilities)
            for case_position, (label, probability_value) in enumerate(
                zip(labels, probabilities)
            ):
                row: dict[str, object] = {
                    "domain": "derm7pt",
                    "repetition": repetition,
                    "outer_fold": case_position // 2,
                    "case_id": f"case_{case_position:02d}",
                    "group_id": f"case_{case_position:02d}",
                    "y": int(label),
                }
                for arm_position, arm in enumerate(aggregate.ARMS):
                    adjusted = min(0.99, max(0.01, probability_value + 0.002 * arm_position))
                    row[f"p_{arm}"] = aggregate.float_text(adjusted)
                rows.append(row)

        points, _context = aggregate.point_estimators(rows, "derm7pt")
        loss_matrix = np.asarray([
            [
                aggregate.binary_nll(int(label), float(probability_value))
                for label, probability_value in zip(labels, probabilities)
            ]
            for probabilities in probabilities_by_repetition
        ])
        case_first_loss = np.mean(loss_matrix, axis=0)
        expected_class_equal = aggregate.class_equal_mean(
            labels, case_first_loss, np.ones(len(labels))
        )
        expected_natural = float(np.mean(case_first_loss))
        expected_auc = float(np.mean([
            aggregate.weighted_auc(labels, probabilities, np.ones(len(labels)))
            for probabilities in probabilities_by_repetition
        ]))
        self.assertAlmostEqual(
            points[("arm", "R", "class_equal_NLL")], expected_class_equal, places=12
        )
        self.assertAlmostEqual(
            points[("arm", "R", "natural_frequency_NLL")], expected_natural, places=12
        )
        self.assertAlmostEqual(points[("arm", "R", "AUC")], expected_auc, places=12)

        mean_probability = np.mean(probabilities_by_repetition, axis=0)
        loss_after_probability_average = np.mean([
            aggregate.binary_nll(int(label), float(probability_value))
            for label, probability_value in zip(labels, mean_probability)
        ])
        auc_after_probability_average = aggregate.weighted_auc(
            labels, mean_probability, np.ones(len(labels))
        )
        self.assertGreater(abs(expected_natural - loss_after_probability_average), 1e-4)
        self.assertGreater(abs(expected_auc - auc_after_probability_average), 1e-4)

    def test_one_cluster_multiset_is_shared_by_all_arms_and_contrasts(self) -> None:
        labels = np.asarray([0, 0, 1, 0, 1, 1], dtype=np.int8)
        cluster_index = np.asarray([0, 0, 1, 2, 3, 3], dtype=int)
        repetitions = (0, 1)
        case_loss = {
            arm: np.asarray([
                0.11 + 0.03 * arm_position + 0.017 * case_position
                for case_position in range(len(labels))
            ])
            for arm_position, arm in enumerate(aggregate.ARMS)
        }
        case_brier = {
            arm: np.asarray([
                0.02 + 0.01 * arm_position + 0.009 * case_position
                for case_position in range(len(labels))
            ])
            for arm_position, arm in enumerate(aggregate.ARMS)
        }
        rep_probabilities = {
            arm: {
                repetition: np.asarray([
                    0.08 + 0.11 * case_position + 0.006 * arm_position
                    + 0.003 * repetition
                    for case_position in range(len(labels))
                ])
                for repetition in repetitions
            }
            for arm_position, arm in enumerate(aggregate.ARMS)
        }
        context = {
            "y": labels,
            "repetitions": repetitions,
            "cluster_names": ["g0", "g1", "g2", "g3"],
            "case_cluster_index": cluster_index,
            "case_loss": case_loss,
            "case_brier": case_brier,
            "rep_probabilities": rep_probabilities,
        }
        seed = 2817
        _accepted, proposal_number, ordered = next(aggregate.bootstrap_draws(
            "synthetic", context, np.random.default_rng(seed), 1, 100
        ))

        replay_rng = np.random.default_rng(seed)
        weight = None
        for _proposal in range(proposal_number):
            sampled = replay_rng.integers(0, 4, size=4)
            multiplicity = np.bincount(sampled, minlength=4).astype(float)
            candidate_weight = multiplicity[cluster_index]
            if all(float(np.sum(candidate_weight[labels == value])) > 0.0 for value in (0, 1)):
                weight = candidate_weight
        self.assertIsNotNone(weight)
        assert weight is not None
        self.assertEqual(weight[0], weight[1])
        self.assertEqual(weight[4], weight[5])

        observed = dict(ordered)
        for arm in aggregate.ARMS:
            self.assertAlmostEqual(
                observed[("arm", arm, "class_equal_NLL")],
                aggregate.class_equal_mean(labels, case_loss[arm], weight),
                places=12,
            )
            self.assertAlmostEqual(
                observed[("arm", arm, "natural_frequency_NLL")],
                aggregate.weighted_mean(case_loss[arm], weight),
                places=12,
            )
            self.assertAlmostEqual(
                observed[("arm", arm, "Brier")],
                aggregate.weighted_mean(case_brier[arm], weight),
                places=12,
            )
            self.assertAlmostEqual(
                observed[("arm", arm, "AUC")],
                float(np.mean([
                    aggregate.weighted_auc(labels, rep_probabilities[arm][rep], weight)
                    for rep in repetitions
                ])),
                places=12,
            )
        for name, (x_arm, y_arm) in aggregate.CONTRAST_ARMS.items():
            for metric in ("class_equal_NLL", "AUC"):
                self.assertAlmostEqual(
                    observed[("contrast", name, metric)],
                    observed[("arm", x_arm, metric)] - observed[("arm", y_arm, metric)],
                    places=12,
                )

    def test_complete_30_unit_package_is_deterministic(self) -> None:
        payload_hash_sets = []
        for _repeat in range(2):
            with tempfile.TemporaryDirectory(prefix="p1-aggregate-synthetic-") as temporary:
                root = Path(temporary)
                populate_root(root)
                final = aggregate.build_aggregate(
                    root,
                    draws=32,
                    max_proposals=1_000,
                    gate_anchors=GATE_ANCHORS,
                    synthetic_test=True,
                )
                self.assertEqual(
                    aggregate.exact_names(final),
                    tuple(sorted(aggregate.AGGREGATE_FILES)),
                )
                payload_hash_sets.append({
                    name: aggregate.sha256_file(final / name)
                    for name in (
                        "all_predictions.csv",
                        "all_calibration_oof.csv",
                        "all_per_case_effects.csv",
                        "all_metrics.csv",
                        "all_paired_effects.csv",
                        "aggregate_effects.csv",
                        "bootstrap_effects.csv",
                    )
                })
        self.assertEqual(payload_hash_sets[0], payload_hash_sets[1])

    def test_formal_paired_flag_cannot_be_pilot(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p1-aggregate-pilot-flag-") as temporary:
            root = Path(temporary)
            populate_root(root)
            unit = aggregate.UNITS[0]
            unit_path = root / aggregate.UNITS_PARENT_REL / aggregate.unit_name(unit)
            paired_path = unit_path / "paired_effects.csv"
            with paired_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["pilot_descriptive_only"] = "true"
            rewrite_csv(paired_path, aggregate.CSV_SCHEMAS["paired_effects.csv"], rows)
            reseal_unit(unit_path)
            with self.assertRaisesRegex(aggregate.AggregateError, "pilot-only"):
                aggregate.validate_unit(unit_path, unit)

    def test_exact_14_file_contract_rejects_extra_file(self) -> None:
        with tempfile.TemporaryDirectory(prefix="p1-aggregate-extra-file-") as temporary:
            root = Path(temporary)
            populate_root(root)
            unit = aggregate.UNITS[0]
            unit_path = root / aggregate.UNITS_PARENT_REL / aggregate.unit_name(unit)
            (unit_path / "unexpected.txt").write_text("synthetic\n", encoding="utf-8")
            with self.assertRaisesRegex(aggregate.AggregateError, "exactly 14"):
                aggregate.validate_unit(unit_path, unit)

    def test_missing_class_proposals_fail_at_cap(self) -> None:
        context = {
            "y": np.asarray([0, 0], dtype=np.int8),
            "repetitions": (0,),
            "cluster_names": ["a", "b"],
            "case_cluster_index": np.asarray([0, 1], dtype=int),
        }
        with self.assertRaisesRegex(aggregate.AggregateError, "only 0 valid"):
            list(aggregate.bootstrap_draws(
                "synthetic", context, np.random.default_rng(1), 1, 3
            ))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticAggregateContract)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "fixture_scope": "synthetic-only; exact 30-unit contract",
        "formal_or_pilot_outcomes_accessed": False,
        "official_test_accessed": False,
    }, sort_keys=True))
    raise SystemExit(0 if result.wasSuccessful() else 1)
