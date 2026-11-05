"""Result-blind tests: every fit here uses generated arrays, never real outcomes."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_neurocomputing_p2_scientific_v3_20260916 as p2
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paper2_mpjc.pilot import case_kfold_splits, _seed


def panel(domain="cbis"):
    rng = np.random.default_rng(719)
    n = 36
    y = np.tile([0, 1], n // 2)
    t, e = np.arange(24), np.arange(24, n)
    widths = np.array([2, 3]) if domain == "derm" else np.array([], dtype=int)
    scores = 0.1 + 0.8 * rng.random((n, 5 if domain == "derm" else 3))
    if domain == "derm":
        for lo, hi in ((0, 2), (2, 5)):
            scores[:, lo:hi] /= scores[:, lo:hi].sum(axis=1, keepdims=True)
    p = {"y": y, "t": t, "e": e, "ids": np.array([f"synthetic_{i:03}" for i in range(n)]), "groups": np.array([f"synthetic_group_{i:03}" for i in range(n)]), "widths": widths, "raw": scores[t], "val": scores[e]}
    manifests = []
    for i in range(4):
        h = t[i * 6:(i + 1) * 6]
        f = np.setdiff1d(t, h)
        p.update({f"f{i}": f, f"h{i}": h, f"x{i}": scores[f], f"z{i}": scores[h]})
        if domain == "cbis":
            nested = []
            for j in range(4):
                if j == i:
                    continue
                g = t[j * 6:(j + 1) * 6]
                fit = np.setdiff1d(f, g)
                nested.append({"diagnosis_selection_heldout_fold": i + 1, "probe_heldout_fold": j + 1, "probe_fit_folds": sorted(set(range(1, 5)) - {i + 1, j + 1}), "probe_fit_patient_sha256": p2.ids_hash(p["groups"][fit]), "probe_validation_patient_sha256": p2.ids_hash(p["groups"][g]), "diagnosis_validation_excluded_from_probe_fit": True, "probe_validation_excluded_from_probe_fit": True})
            manifests.append([{"outer_fold": 0, "diagnosis_selection_heldout_fold": i + 1, "fit_folds": sorted(set(range(1, 5)) - {i + 1}), "fit_patient_sha256": p2.ids_hash(p["groups"][f]), "heldout_patient_sha256": p2.ids_hash(p["groups"][h]), "fit_heldout_patient_disjoint": True, "nested_probe_panels": nested}])
        else:
            nested = [{"inner_fold": j, "fit_validation_disjoint": True, "split_inputs": "case_indices_only_no_C_no_Y", "fit_count": len(fit), "validation_count": len(g), "fit_case_id_sha256": p2.ids_hash(p["ids"][fit]), "validation_case_id_sha256": p2.ids_hash(p["ids"][g])} for j, (fit, g) in enumerate(p2.derm_g_partitions(f, 0, 0, i))]
            manifests.append({"fit_case_ids": p["ids"][f].tolist(), "heldout_case_ids": p["ids"][h].tolist(), "inner": nested})
    generation = {"selection_manifests": manifests}
    if domain == "cbis":
        generation["manifests"] = [block[0] for block in manifests] + [{"outer_fold": 0, "role": "final_outer_train_probe", "fit_patient_sha256": p2.ids_hash(p["groups"][t]), "validation_patient_sha256": p2.ids_hash(p["groups"][e]), "patient_disjoint": True}]
    return p, generation


def synthetic_lineage(p, g, domain, source_record):
    """Only generated identities; never open the P0 cohort for test injection."""
    source_fold = np.zeros(len(p["ids"]), dtype=int)
    for i in range(4):
        source_fold[p[f"h{i}"]] = i + 1
    return p2.lineage(p, g, domain, source_record, unit=f"{domain}_r0_f0", cbis_source_fold=source_fold if domain == "cbis" else None)

class ZeroHead:
    def __init__(self, d):
        self.coef_ = np.zeros((1, d))
        self.intercept_ = np.zeros(1)

    def decision_function(self, x):
        return np.zeros(len(x))


def fake_fit(x, y, head, hp, seed, fit_id, stage, tau):
    diagnostic = {"head": head, "fit_id": fit_id, "stage": stage, "optimization_status": "synthetic_mock", "epochs_completed": 0 if head == "lr" else 200}
    record = {"head": head, "seed": seed, "coefficient": np.zeros(x.shape[1]).tolist(), "intercept": 0.0}
    return ZeroHead(x.shape[1]), diagnostic, record


def prediction_rows(domain="cbis", tied=False):
    rows = []
    reps = [0] if domain == "cbis" else list(range(5))
    for tau in p2.TEMPS:
        for rep in ("soft", "logodds"):
            for head in ("lr", "mlp"):
                for seed in ((0,) if head == "lr" else p2.SEEDS):
                    for repetition in reps:
                        for i in range(12):
                            q = (i % 3 - 1) if tied else (i - 5.5) * .2
                            q += 0 if tied else .01 * seed + .05 * repetition + (.3 if rep == "soft" else -.1) * tau
                            rows.append({"case_id": f"synthetic_{i}", "patient_group": f"g{i // 2}", "label": i % 2, "temperature": tau, "representation": rep, "head": head, "refit_seed": seed, "repetition": repetition, "logit": q})
    return rows


class ScientificP2Test(unittest.TestCase):
    def test_bounded_fit_counts(self):
        protocol = p2.json_read(p2.PROTOCOL)
        self.assertEqual(protocol["fit_counts"]["scientific_total"], 30 * (4 * (4 * 6 + 1) + 4 * (4 * 3 + 3)) + 6 + 2 * (4 * 6 * 3 + 4 * 6 + 1))
        self.assertEqual(protocol["core"]["temperatures"], [1., .25])
        self.assertEqual(protocol["temperature"]["tau4"], "exact_controls_only_no_triggered_expansion")

    def test_binding_complete_and_candidate_not_authorized(self):
        b = p2.validate_binding()
        self.assertFalse(b["scientific_execution_authorized"])
        self.assertEqual(b["status"], "CANDIDATE_NOT_FROZEN")

    def test_execute_missing_approvals_fails_before_source_load(self):
        with patch.object(p2.reviewed, "audit_panel_structure", side_effect=AssertionError("must not load real panels")):
            with self.assertRaises(p2.GateError):
                p2.execute()

    def test_component_audit_does_not_authorize_new_runner(self):
        b = {"status": "CANDIDATE_NOT_FROZEN"}
        fake = {"source_binding_sha256": "b", "scope": "development_only_no_official_test_no_P1", "official_test_access_authorized": False, "status": "PASS_COMPONENT_ONLY"}
        with patch.object(p2, "validate_binding", return_value=b), patch.object(p2.reviewed, "sha256_file", return_value="b"), patch.object(p2, "json_read", return_value=fake):
            with self.assertRaises(p2.GateError):
                p2.require_execution_gate()

    def test_binding_source_tamper_fails(self):
        original = p2.json_read
        def changed(path):
            obj = original(path)
            if path == p2.BINDING:
                obj["files"] = dict(obj["files"])
                obj["files"]["scripts/run_neurocomputing_p2_scientific_v3_20260916.py"] = "0" * 64
            return obj
        with patch.object(p2, "json_read", side_effect=changed):
            with self.assertRaises(p2.GateError):
                p2.validate_binding()

    def test_train_local_scaling_not_validation_statistics(self):
        x = np.array([[1., 7.], [3., 7.]])
        z = np.array([[100., -5.]])
        a, b, receipt = p2.scale_data(x, z)
        np.testing.assert_equal(a, [[-1., 0.], [1., 0.]])
        np.testing.assert_equal(b, [[98., -12.]])
        self.assertEqual(receipt["fit_count"], 2)
        self.assertEqual(receipt["zero_variance_coordinate_count"], 1)

    def test_cb_binary_logodds_temperature_exact(self):
        s = np.array([[0., .45, 1.], [.2, .6, .8]])
        base, clipped = p2.coordinates(s, "cbis", np.array([]), 1., "logodds")
        self.assertEqual(clipped.sum(), 2)
        for tau in (.25, 4.):
            other, _ = p2.coordinates(s, "cbis", np.array([]), tau, "logodds")
            np.testing.assert_equal(other, base / tau)
        np.testing.assert_equal(p2.coordinates(s, "cbis", np.array([]), 1., "soft")[0], s)

    def test_derm_reference_logodds_temperature_exact(self):
        s = np.array([[.7, .3, .2, .3, .5], [.4, .6, .1, .3, .6]])
        base, _ = p2.coordinates(s, "derm", np.array([2, 3]), 1., "logodds")
        self.assertEqual(base.shape, (2, 3))
        np.testing.assert_allclose(base[:, 0], np.log(s[:, 1] / s[:, 0]))
        for tau in (.25, 4.):
            np.testing.assert_equal(p2.coordinates(s, "derm", np.array([2, 3]), tau, "logodds")[0], base / tau)

    def test_invalid_score_fails_closed(self):
        for s in (np.array([[np.nan]]), np.array([[1.01]]), np.array([.5])):
            with self.assertRaises(p2.GateError):
                p2.coordinates(s, "cbis", np.array([]), 1., "soft")

    def test_cb_complete_nested_lineage(self):
        p, g = panel()
        lin = synthetic_lineage(p, g, "cbis", {"panel_sha256": "synthetic"})
        self.assertEqual(len(lin["inner"]), 4)
        self.assertTrue(all(x["upstream_inner_validation_excluded"] for x in lin["inner"]))

    def test_derm_complete_nested_lineage(self):
        p, g = panel("derm")
        self.assertEqual(len(synthetic_lineage(p, g, "derm", {})["inner"]), 4)

    def test_cb_upstream_validation_leak_rejected(self):
        p, g = panel()
        g["selection_manifests"][0][0]["nested_probe_panels"][0]["diagnosis_validation_excluded_from_probe_fit"] = False
        with self.assertRaises(p2.GateError):
            synthetic_lineage(p, g, "cbis", {})

    def test_cb_corrupt_G_fit_hash_rejected(self):
        p, g = panel()
        g["selection_manifests"][0][0]["nested_probe_panels"][0]["probe_fit_patient_sha256"] = "0" * 64
        with self.assertRaises(p2.GateError):
            synthetic_lineage(p, g, "cbis", {})

    def test_cb_corrupt_G_validation_hash_rejected(self):
        p, g = panel()
        g["selection_manifests"][0][0]["nested_probe_panels"][0]["probe_validation_patient_sha256"] = "0" * 64
        with self.assertRaises(p2.GateError):
            synthetic_lineage(p, g, "cbis", {})

    def test_cb_duplicate_G_partition_rejected(self):
        p, g = panel()
        nested = g["selection_manifests"][0][0]["nested_probe_panels"]
        nested[1] = dict(nested[0])
        with self.assertRaises(p2.GateError):
            synthetic_lineage(p, g, "cbis", {})

    def test_cb_G_heldout_identity_rejected(self):
        p, g = panel()
        g["selection_manifests"][0][0]["nested_probe_panels"][0]["diagnosis_selection_heldout_fold"] = 4
        with self.assertRaises(p2.GateError):
            synthetic_lineage(p, g, "cbis", {})

    def test_cb_final_projection_rejected(self):
        p, g = panel()
        g["manifests"][1] = dict(g["manifests"][0])
        with self.assertRaises(p2.GateError):
            synthetic_lineage(p, g, "cbis", {})

    def test_cb_final_scope_hash_rejected(self):
        p, g = panel()
        g["manifests"][4]["validation_patient_sha256"] = "0" * 64
        with self.assertRaises(p2.GateError):
            synthetic_lineage(p, g, "cbis", {})

    def test_cb_final_role_rejected(self):
        p, g = panel()
        g["manifests"][4]["role"] = "inner_probe"
        with self.assertRaises(p2.GateError):
            synthetic_lineage(p, g, "cbis", {})

    def test_derm_G_fit_hash_rejected(self):
        p, g = panel("derm")
        g["selection_manifests"][0]["inner"][0]["fit_case_id_sha256"] = "0" * 64
        with self.assertRaises(p2.GateError):
            synthetic_lineage(p, g, "derm", {})

    def test_derm_G_validation_hash_rejected(self):
        p, g = panel("derm")
        g["selection_manifests"][0]["inner"][0]["validation_case_id_sha256"] = "0" * 64
        with self.assertRaises(p2.GateError):
            synthetic_lineage(p, g, "derm", {})

    def test_derm_G_inconsistent_individual_counts_rejected(self):
        p, g = panel("derm")
        q = g["selection_manifests"][0]["inner"][0]
        q["fit_count"] += 1
        q["validation_count"] -= 1
        with self.assertRaises(p2.GateError):
            synthetic_lineage(p, g, "derm", {})

    def test_derm_G_index_rejected(self):
        p, g = panel("derm")
        g["selection_manifests"][0]["inner"][0]["inner_fold"] = 2
        with self.assertRaises(p2.GateError):
            synthetic_lineage(p, g, "derm", {})

    def test_derm_seed_is_exact_archived_producer_recipe(self):
        f = np.arange(3, 41, 2)
        for repetition, outer_fold, h in ((0, 0, 0), (4, 3, 2)):
            seed = _seed(2026090829, "strict_meta_inner", repetition, outer_fold, h, "split")
            expected = [(f[tr], f[va]) for tr, va in case_kfold_splits(len(f), 4, seed)]
            actual = p2.derm_g_partitions(f, repetition, outer_fold, h)
            for (et, ev), (at, av) in zip(expected, actual):
                np.testing.assert_array_equal(at, et)
                np.testing.assert_array_equal(av, ev)

    def test_derm_wrong_repetition_cannot_reuse_G_manifest(self):
        p, g = panel("derm")
        with self.assertRaises(p2.GateError):
            p2.lineage(p, g, "derm", {}, unit="derm_r1_f0")

    def test_patient_digest_preserves_archived_multiplicity(self):
        self.assertNotEqual(p2.ids_hash(["p", "p", "q"]), p2.ids_hash(["p", "q"]))

    def test_derm_generation_row_mismatch_rejected(self):
        p, g = panel("derm")
        g["selection_manifests"][0]["fit_case_ids"][0] = "wrong_case"
        with self.assertRaises(p2.GateError):
            synthetic_lineage(p, g, "derm", {})

    def test_patient_overlap_rejected(self):
        p, g = panel()
        p["groups"][p["e"][0]] = p["groups"][p["t"][0]]
        with self.assertRaises(p2.GateError):
            synthetic_lineage(p, g, "cbis", {})

    def test_partition_invalid_indices_rejected(self):
        for a, b in ((np.array([0, 0]), np.array([1])), (np.array([0]), np.array([0, 1])), (np.array([0.]), np.array([1]))):
            with self.assertRaises(p2.GateError):
                p2.validate_partition(a, b, np.array([0, 1]), "synthetic")

    def test_lr_tie_smallest_C_and_four_common_folds(self):
        p, _ = panel()
        with patch.object(p2, "fit_head", side_effect=fake_fit):
            arm = p2.run_arm(p, "cbis_r0_f1", 1., "soft", "lr", {})
        self.assertEqual(arm["selected"], .001)
        self.assertEqual(len(arm["diagnostics"]), 25)
        self.assertEqual({x["inner_fold"] for x in arm["selection"]}, set(range(4)))

    def test_mlp_tie_largest_lambda_and_individual_seed_predictions(self):
        p, _ = panel()
        with patch.object(p2, "fit_head", side_effect=fake_fit):
            arm = p2.run_arm(p, "cbis_r0_f1", 1., "soft", "mlp", {})
        self.assertEqual(arm["selected"], .01)
        self.assertEqual(len(arm["diagnostics"]), 15)
        self.assertEqual({x["refit_seed"] for x in arm["predictions"]}, {17, 29, 43})
        self.assertEqual(len(arm["predictions"]), 3 * len(p["e"]))
        self.assertTrue(all(x["selection_seed"] == 17 for x in arm["selection"]))

    def test_changing_outer_evaluation_labels_does_not_change_fitted_head(self):
        p, _ = panel()
        with patch.object(p2, "fit_head", side_effect=fake_fit):
            first = p2.run_arm(p, "cbis_r0_f1", 1., "soft", "lr", {})
            p["y"][p["e"]] = 1 - p["y"][p["e"]]
            second = p2.run_arm(p, "cbis_r0_f1", 1., "soft", "lr", {})
        self.assertEqual(first["fitted"], second["fitted"])
        self.assertEqual(first["losses"], second["losses"])

    def test_actual_synthetic_lr_convergence_diagnostics(self):
        rng = np.random.default_rng(49)
        x, y = rng.normal(size=(48, 3)), np.tile([0, 1], 24)
        _, d, r = p2.fit_head(x, y, "lr", 1., 0, "synthetic", "synthetic_test", 1.)
        self.assertEqual(d["solver"], "newton-cholesky")
        self.assertEqual(d["warning_count"], 0)
        self.assertEqual(d["optimization_status"], "converged")
        self.assertEqual(r["seed"], 0)

    def test_actual_synthetic_mlp_fixed_recipe_determinism(self):
        rng = np.random.default_rng(57)
        x, y = rng.normal(size=(24, 3)), np.tile([0, 1], 12)
        a, d, r = p2.fit_head(x, y, "mlp", .001, 17, "synthetic", "synthetic_test", 1.)
        b, _, rr = p2.fit_head(x, y, "mlp", .001, 17, "synthetic", "synthetic_test", 1.)
        np.testing.assert_equal(a.decision_function(x), b.decision_function(x))
        self.assertEqual(r, rr)
        self.assertEqual(d["epochs_completed"], 200)
        self.assertTrue(d["weights_only_penalty"])
        self.assertNotEqual(d["optimization_status"], "converged")

    def test_nonfinite_training_data_rejected(self):
        with self.assertRaises(p2.GateError):
            p2.fit_head(np.array([[np.nan], [1.]]), np.array([0, 1]), "lr", 1., 0, "synthetic", "synthetic_test", 1.)

    def test_numeric_tolerance_not_relaxed(self):
        with self.assertRaises(p2.GateError):
            p2.comparison(np.array([0, 1]), np.zeros(2), np.array([1.00001e-5, 0]), "synthetic_boundary")
        self.assertTrue(p2.comparison(np.array([0, 1]), np.zeros(2), np.zeros(2), "synthetic_equal")["pass"])

    def test_complete_last_atomic_resume_and_hash_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            self.assertFalse(p2.resume_valid(directory, "binding", "source", {"x.json"}, 1))
            p2.atomic_json(directory / "x.json", {"synthetic": True})
            self.assertFalse(p2.resume_valid(directory, "binding", "source", {"x.json"}, 1))
            p2.complete_bundle(directory, ["x.json"], "binding", "source", 1, .1)
            self.assertTrue(p2.resume_valid(directory, "binding", "source", {"x.json"}, 1))
            p2.atomic_json(directory / "x.json", {"synthetic": False})
            with self.assertRaises(p2.GateError):
                p2.resume_valid(directory, "binding", "source", {"x.json"}, 1)

    def test_completed_wrong_fit_count_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            p2.atomic_json(directory / "x.json", {})
            p2.complete_bundle(directory, ["x.json"], "binding", "source", 1, .1)
            with self.assertRaises(p2.GateError):
                p2.resume_valid(directory, "binding", "source", {"x.json"}, 2)

    def test_atomic_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.json", Path(d) / "b.json"
            p2.atomic_json(a, {})
            b.symlink_to(a)
            with self.assertRaises(p2.GateError):
                p2.atomic_json(b, {})

    def test_mocked_full_unit_contract_and_160_fit_ledger(self):
        p, g = panel()
        provenance = synthetic_lineage(p, g, "cbis", {"panel_sha256": "synthetic"})
        with tempfile.TemporaryDirectory() as d, patch.object(p2, "fit_head", side_effect=fake_fit):
            directory = Path(d)
            p2.fit_unit(p, "cbis_r0_f1", provenance, directory, "synthetic_binding")
            complete = p2.json_read(directory / "COMPLETE.json")
            self.assertEqual(complete["fit_count"], 160)
            contract = p2.json_read(p2.PREP / "P2_OUTPUT_CONTRACT.json")
            self.assertEqual(set(contract["unit_files"]), {x.name for x in directory.iterdir()})
            predictions = p2.read_csv(directory / "predictions.csv")
            self.assertEqual(len(predictions), len(p["e"]) * 16)
            self.assertEqual({r["refit_seed"] for r in predictions if r["head"] == "mlp"}, {"17", "29", "43"})

    def test_full_actual_synthetic_exact_controls(self):
        p, _ = panel()
        arms = {(tau, "logodds", "lr"): p2.run_arm(p, "cbis_r0_f0", tau, "logodds", "lr", {}) for tau in (1., .25)}
        result, diagnostics, _ = p2.exact_controls(p, "cbis_r0_f0", arms, {})
        self.assertTrue(result["pass"])
        self.assertEqual(len(diagnostics), 97)
        self.assertTrue(all(c["pass"] for c in result["comparisons"]))

    def test_loss_average_not_probability_ensemble(self):
        rows = prediction_rows("derm")
        e, y, _, _ = p2.prepare_estimators(rows, "derm")
        arm = e[(1., "soft", "mlp", 17)]
        expected = np.mean(np.logaddexp(0, arm["logits"]) - y * arm["logits"], axis=0)
        np.testing.assert_equal(arm["loss"], expected)
        ensemble_p = p2.reviewed.stable_sigmoid(arm["logits"]).mean(axis=0)
        ensemble_loss = -(y * np.log(ensemble_p) + (1 - y) * np.log1p(-ensemble_p))
        self.assertGreater(np.max(np.abs(expected - ensemble_loss)), 1e-5)

    def test_stitch_duplicate_case_rejected(self):
        rows = prediction_rows()
        rows.append(dict(rows[0]))
        with self.assertRaises(p2.GateError):
            p2.prepare_estimators(rows, "cbis")

    def test_stitch_missing_seed_rejected(self):
        rows = [r for r in prediction_rows() if r["refit_seed"] != 43]
        with self.assertRaises(p2.GateError):
            p2.prepare_estimators(rows, "cbis")

    def test_stitch_group_mapping_change_rejected(self):
        rows = prediction_rows()
        rows[1]["patient_group"] = "wrong_group"
        with self.assertRaises(p2.GateError):
            p2.prepare_estimators(rows, "cbis")

    def test_cached_weighted_auc_equals_sklearn_with_ties_and_zero_weights(self):
        e, y, _, _ = p2.prepare_estimators(prediction_rows(tied=True), "cbis")
        rng = np.random.default_rng(37)
        for _ in range(12):
            weights = rng.integers(0, 5, len(y))
            for estimator in e.values():
                actual = p2.metrics(estimator, y, weights)["auc"]
                expected = np.mean([roc_auc_score(y, z, sample_weight=weights) for z in estimator["logits"]])
                self.assertAlmostEqual(actual, expected, places=14)

    def test_direct_representation_and_capacity_and_temperature_interactions(self):
        arms = {(tau, rep, head, seed): {metric: 0.1 * tau + (0.2 if rep == "soft" else 0.5) * (2 if head == "mlp" else 1) for metric in ("class_equal_nll", "auc", "natural_nll", "brier")} for tau in p2.TEMPS for rep in ("soft", "logodds") for head in ("lr", "mlp") for seed in ((0,) if head == "lr" else p2.SEEDS)}
        result = p2.effects(arms)
        self.assertAlmostEqual(result[("class_equal_nll", "representation_D", 1., "lr", "0")], .3)
        self.assertAlmostEqual(result[("auc", "representation_D", 1., "lr", "0")], -.3)
        self.assertAlmostEqual(result[("class_equal_nll", "capacity_I", 1., "mlp_minus_lr", "seed_mean")], .3)
        self.assertAlmostEqual(result[("class_equal_nll", "temperature_interaction", "0.25_minus_1", "mlp", "seed_mean")], 0)

    def test_bootstrap_reproducible_paired_fixed_cluster_stream(self):
        rows = prediction_rows()
        a = p2.aggregate_domain(rows, "cbis", np.random.default_rng(2026091517), replicates=13, maximum=100)
        b = p2.aggregate_domain(rows, "cbis", np.random.default_rng(2026091517), replicates=13, maximum=100)
        self.assertEqual(a, b)
        self.assertEqual(a["accepted"], 13)
        self.assertEqual(len(a["paired_effects"]), 92)
        for index in range(1, 14):
            self.assertEqual(len({r["cluster_multiplicity_digest"] for r in a["draws"] if r["accepted_index"] == index}), 1)
        self.assertTrue(all(r["resolution_interpretation"] == "descriptive_not_equivalence_or_clinical_margin" for r in a["paired_effects"]))

    def test_draw_sink_streams_without_accumulating_large_dictionary_list(self):
        captured = []
        a = p2.aggregate_domain(prediction_rows(), "cbis", np.random.default_rng(52), replicates=3, maximum=100, draw_sink=captured.append)
        self.assertEqual(a["draws"], [])
        self.assertEqual(len(captured), 3 * 92)



def cost_fake_fit(x, y, head, hp, seed, fit_id, stage, tau):
    model, diagnostic, record = fake_fit(x, y, head, hp, seed, fit_id, stage, tau)
    diagnostic.update({"lambda": hp, "seed": seed, "temperature": tau,
                      "optimization_status": "fixed_epoch_numerically_complete",
                      "optimizer": "Adam", "learning_rate": .001, "batching": "full_batch",
                      "warning_count": 0, "gradient_finite": True, "parameters_finite": True,
                      "weights_only_penalty": True, "elapsed_seconds": .25,
                      "first_objective": .7, "last_objective": .6})
    return model, diagnostic, record


class RepresentativeCostP2Test(unittest.TestCase):
    def fixture(self):
        p, g = panel()
        audit = {"status": "PASS", "source_contract": {"synthetic": True}, "units": {}}
        gate = {"source_binding_sha256": "synthetic-binding", "freeze_sha256": "f",
                "audit_sha256": "a", "authorization_sha256": "z",
                "binding": {"archived_source_receipt_sha256": p2.digest(audit["source_contract"])}}
        provenance = {"synthetic": True}
        return p, gate, audit, provenance

    def test_missing_approval_measure_cannot_load_or_fit(self):
        with patch.object(p2.reviewed, "audit_panel_structure", side_effect=AssertionError("no arrays")), patch.object(p2, "fit_head", side_effect=AssertionError("no fits")):
            with self.assertRaises(p2.GateError):
                p2.measure_representative_cost()

    def test_pilot_exact_six_fits_and_train_only_scalers(self):
        p, _, _, _ = self.fixture()
        with patch.object(p2, "fit_head", side_effect=cost_fake_fit) as fit:
            rows, scalers = p2.run_representative_pilot(p)
        self.assertEqual(fit.call_count, 6)
        p2.validate_pilot_diagnostics(rows)
        for call in fit.call_args_list:
            x, y, head, hp, seed, fit_id, stage, tau = call.args
            np.testing.assert_allclose(x.mean(axis=0), 0, atol=1e-12)
            np.testing.assert_array_equal(y, p["y"][p["f0"]])
            self.assertEqual((head, seed, stage, tau), ("mlp", 17, "fixed_recipe_pilot", 1.0))
        self.assertEqual(len(scalers), 2)
        self.assertEqual(scalers[0]["fit_case_id_sha256"], p2.ids_hash(p["ids"][p["f0"]]))

    def test_validation_and_outer_labels_cannot_change_pilot_inputs(self):
        p, _, _, _ = self.fixture()
        with patch.object(p2, "fit_head", side_effect=cost_fake_fit) as fit:
            p2.run_representative_pilot(p)
            before = [(c.args[0].copy(), c.args[1].copy()) for c in fit.call_args_list]
            fit.reset_mock()
            p["y"][p["h0"]] = 1 - p["y"][p["h0"]]
            p["y"][p["e"]] = 1 - p["y"][p["e"]]
            p2.run_representative_pilot(p)
            after = [(c.args[0], c.args[1]) for c in fit.call_args_list]
        for (x, y), (xx, yy) in zip(before, after):
            np.testing.assert_array_equal(x, xx)
            np.testing.assert_array_equal(y, yy)

    def test_measure_atomic_complete_last_then_resume_zero_new_fits(self):
        p, gate, audit, provenance = self.fixture()
        events = []
        original_json, original_csv = p2.atomic_json, p2.atomic_csv
        def write_json(path, value):
            events.append(path.name)
            original_json(path, value)
        def write_csv(path, value):
            events.append(path.name)
            original_csv(path, value)
        with tempfile.TemporaryDirectory() as tmp, patch.object(p2, "REPRESENTATIVE_COST", Path(tmp)), patch.object(p2, "require_execution_gate", return_value=gate), patch.object(p2.reviewed, "audit_panel_structure", return_value=audit), patch.object(p2.reviewed, "synthetic_solver_equivalence_check", return_value={"pass": True}), patch.object(p2, "load_representative_panel", return_value=(p, provenance)), patch.object(p2, "fit_head", side_effect=cost_fake_fit) as fit, patch.object(p2, "atomic_json", side_effect=write_json), patch.object(p2, "atomic_csv", side_effect=write_csv):
            receipt = p2.measure_representative_cost()
            self.assertEqual(fit.call_count, 6)
            self.assertEqual(events[-1], "COMPLETE.json")
            self.assertEqual(receipt["remaining_full_execute_new_fits"], 4994)
            fit.reset_mock()
            self.assertEqual(p2.measure_representative_cost(), receipt)
            fit.assert_not_called()
            marker = p2.json_read(Path(tmp) / "COMPLETE.json")
            self.assertEqual(marker["fit_count"], 6)
            self.assertEqual(set(marker["files"]), p2.COST_FILES)

    def test_cost_receipt_tampering_and_incomplete_bundle_fail(self):
        p, gate, audit, provenance = self.fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(p2, "REPRESENTATIVE_COST", Path(tmp)), patch.object(p2, "fit_head", side_effect=cost_fake_fit):
            rows, scalers = p2.run_representative_pilot(p)
            receipt = p2.representative_cost_receipt(gate, audit, provenance, rows, scalers, 2.)
            p2.atomic_json(Path(tmp) / "COST_RECEIPT.json", receipt)
            with self.assertRaises(p2.GateError):
                p2.require_representative_cost(gate, audit, provenance)
            p2.atomic_csv(Path(tmp) / "fit_diagnostics.csv", rows)
            p2.atomic_json(Path(tmp) / "lineage.json", provenance)
            p2.complete_bundle(Path(tmp), sorted(p2.COST_FILES), gate["source_binding_sha256"], p2.digest(provenance), 6, 2.)
            receipt["seed"] = 29
            p2.atomic_json(Path(tmp) / "COST_RECEIPT.json", receipt)
            with self.assertRaises(p2.GateError):
                p2.require_representative_cost(gate, audit, provenance)
            p2.complete_bundle(Path(tmp), sorted(p2.COST_FILES), gate["source_binding_sha256"], p2.digest(provenance), 6, 2.)
            with self.assertRaises(p2.GateError):
                p2.require_representative_cost(gate, audit, provenance)

    def test_pilot_wrong_count_recipe_or_nonfinite_cost_fails(self):
        p, _, _, _ = self.fixture()
        with patch.object(p2, "fit_head", side_effect=cost_fake_fit):
            rows, _ = p2.run_representative_pilot(p)
        for key, value in (("seed", 29), ("elapsed_seconds", float("nan")), ("elapsed_seconds", 0), ("epochs_completed", 199), ("gradient_finite", False), ("lambda", 100)):
            changed = [dict(row) for row in rows]
            changed[0][key] = value
            with self.assertRaises(p2.GateError):
                p2.validate_pilot_diagnostics(changed)
        with self.assertRaises(p2.GateError):
            p2.validate_pilot_diagnostics(rows[:-1])

    def test_full_execute_missing_cost_fails_before_scientific_fit(self):
        p, gate, audit, provenance = self.fixture()
        with tempfile.TemporaryDirectory() as tmp, patch.object(p2, "REPRESENTATIVE_COST", Path(tmp)), patch.object(p2, "require_execution_gate", return_value=gate), patch.object(p2.reviewed, "audit_panel_structure", return_value=audit), patch.object(p2.reviewed, "synthetic_solver_equivalence_check", return_value={"pass": True}), patch.object(p2, "load_representative_panel", return_value=(p, provenance)), patch.object(p2, "fit_unit", side_effect=AssertionError("no science")), patch.object(p2, "fit_head", side_effect=AssertionError("no science")):
            with self.assertRaises(p2.GateError):
                p2.execute()

    def test_cli_cost_dispatch_exits_without_full_execute_or_structure(self):
        with patch.object(sys, "argv", ["p2", "--measure-representative-cost"]), patch.object(p2, "measure_representative_cost", return_value={"synthetic": True}) as measure, patch.object(p2, "execute", side_effect=AssertionError("no full execute")), patch.object(p2, "structure_measurement", side_effect=AssertionError("no structure")):
            self.assertEqual(p2.main(), 0)
            measure.assert_called_once()

    def test_versioned_contract_preserves_budget_and_requires_cost(self):
        protocol = p2.json_read(p2.PROTOCOL)
        old = p2.json_read(p2.ROOT / "paper_b_readiness/P2_PREP/SCIENTIFIC_V2/P2_PROTOCOL.json")
        for key in ("core", "lr", "mlp", "temperature", "selection", "exact", "pilot", "affine", "statistics", "fit_counts", "lineage"):
            self.assertEqual(protocol[key], old[key])
        self.assertEqual(protocol["representative_cost"]["unique_total_fits"], 5000)
        contract = p2.json_read(p2.PREP / "P2_OUTPUT_CONTRACT.json")
        self.assertEqual(set(contract["representative_cost_files"]) - {"COMPLETE.json"}, p2.COST_FILES)


if __name__ == "__main__":
    unittest.main()

