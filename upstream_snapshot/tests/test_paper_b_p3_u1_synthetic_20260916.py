"""P3 source preparation tests. Generated arrays only: no project loader/cache."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from paper_b_p3_u1 import (
    C_GRID, ConceptData, DiagnosticSplit, NestedPlan, U1Generator, U1Recipe,
    build_u1_nested_panel, build_u1_representative_inner, class_equal_nll, concept_diagnostics,
    upstream_fit_forecast, validate_nested_plan,
)


def synthetic(n=180, widths=(2, 3), grouped=False):
    rng = np.random.default_rng(62891)
    labels = np.column_stack([np.arange(n) % width for width in widths]).astype(int)
    x = rng.normal(size=(n, 3))
    x[:, 0] += labels[:, 0] * .15
    ids = np.asarray([f"SYNTHETIC-{i:04}" for i in range(n)])
    groups = np.asarray([f"GROUP-{i // (2 if grouped else 1):04}" for i in range(n)])
    return ConceptData(x, ids, groups, labels, np.ones_like(labels, dtype=bool),
                       tuple(f"synthetic_concept_{i}" for i in range(len(widths))), widths)


def recipe(domain="Derm7pt"):
    return U1Recipe(domain, 2026090829, 1000, 10 if domain == "CBIS-DDSM" else 0,
                    "a" * 64)


def plan_for(data, *, cbis=False):
    folds = tuple(np.array_split(np.arange(len(data.ids)), 5))
    e, hs = folds[0], folds[1:]
    t = np.concatenate(hs)
    splits = []
    for h in hs:
        d = np.setdiff1d(t, h)
        gs = (tuple(v for v in hs if not np.array_equal(v, h)) if cbis
              else tuple(np.array_split(d, 4)))
        splits.append(DiagnosticSplit(d, h, gs))
    return NestedPlan("SYNTHETIC-only-plan", "b" * 64, t, e, tuple(splits), hs)


class U1SyntheticTests(unittest.TestCase):
    def test_representative_is_one_inner_without_outer_predictions(self):
        data = synthetic()
        plan = plan_for(data)
        generator = U1Generator(data, recipe())
        result = build_u1_representative_inner(generator, plan)
        self.assertFalse(result["outer_predictions_generated"])
        self.assertFalse(result["final_refit_generated"])
        self.assertNotIn("val", result)
        self.assertEqual(generator.cost_summary()["unique_fit_bundles"], 5)
        self.assertEqual(generator.cost_summary()["fit_count"], 5*2*19)
        outer_ids = set(data.ids[plan.outer_evaluation_rows])
        for call in result["lineage"]:
            self.assertFalse(set(call["target_ids"]) & outer_ids)
        self.assertEqual(len(result["source_plan_content_sha256"]), 64)
        with self.assertRaises(ValueError):
            build_u1_representative_inner(generator, plan, inner_index=99)

    def test_class_equal_nll_is_class_mean_not_natural_mean(self):
        y = np.array([0, 0, 0, 1])
        p = np.array([[.9, .1], [.9, .1], [.9, .1], [.8, .2]])
        self.assertAlmostEqual(class_equal_nll(y, p), (-np.log(.9) - np.log(.2)) / 2)
        self.assertNotAlmostEqual(class_equal_nll(y, p), (-3*np.log(.9)-np.log(.2))/4)

    def test_frozen_recipe_rejects_grid_or_fold_expansion(self):
        with self.assertRaises(ValueError):
            replace(recipe(), c_grid=(1.,)).validate()
        with self.assertRaises(ValueError):
            replace(recipe(), selection_folds=4).validate()
        with self.assertRaises(ValueError):
            replace(recipe("CBIS-DDSM"), minimum_state_support=0).validate()

    def test_every_candidate_ancestor_stays_in_F_and_excludes_Q(self):
        data = synthetic()
        g = U1Generator(data, recipe())
        result = g.fit_predict(np.arange(150), np.arange(150, 180),
                               forbidden_rows=np.arange(150, 180), context="synthetic")
        forbidden = set(data.groups[150:])
        for selection in result["lineage"]["selections"]:
            self.assertEqual([v["c"] for v in selection["candidates"]], list(C_GRID))
            best = min(selection["candidates"], key=lambda v: (v["class_equal_oof_nll"], v["c"]))
            self.assertEqual(selection["selected_c"], best["c"])
            for candidate in selection["candidates"]:
                self.assertEqual(len(candidate["ancestors"]), 3)
                for a in candidate["ancestors"]:
                    self.assertFalse(set(a["fit_groups"]) & set(a["validation_groups"]))
                    self.assertFalse((set(a["fit_groups"]) | set(a["validation_groups"])) & forbidden)
        self.assertFalse(result["lineage"]["diagnosis_label_access"])
        self.assertEqual(g.cost_summary()["fit_count"], 2*19)

    def test_query_labels_never_read_or_used_in_selection(self):
        data = synthetic()
        g = U1Generator(data, recipe())
        args = dict(forbidden_rows=np.arange(150, 180), context="synthetic")
        first = g.fit_predict(np.arange(150), np.arange(150, 180), **args)
        data.labels[150:] = 999999  # invalid if inspected; Q must be prediction-only
        second = g.fit_predict(np.arange(150), np.arange(150, 180), **args)
        self.assertTrue(second["lineage"]["cache_reused"])
        for a, b in zip(first["blocks"], second["blocks"]):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(g.cost_summary()["fit_count"], 38)

    def test_group_firewall_and_forbidden_membership(self):
        data = synthetic(grouped=True)
        g = U1Generator(data, recipe())
        with self.assertRaises(ValueError):
            g.fit_predict(np.arange(151), np.arange(151, 180),
                          forbidden_rows=np.arange(151, 180), context="synthetic")
        with self.assertRaises(ValueError):
            g.fit_predict(np.arange(150), np.arange(150, 180),
                          forbidden_rows=[], context="synthetic")
        self.assertFalse(g.fit_cost)

    def test_split_is_label_blind_and_group_whole(self):
        data = synthetic(grouped=True)
        g = U1Generator(data, recipe())
        before = g._group_splits(np.arange(150))
        data.labels[:] = data.labels[::-1]
        after = g._group_splits(np.arange(150))
        for (a, b), (c, d) in zip(before, after):
            np.testing.assert_array_equal(a, c)
            np.testing.assert_array_equal(b, d)
            self.assertFalse(set(data.groups[a]) & set(data.groups[b]))

    def test_cache_changes_when_F_features_change(self):
        data = synthetic()
        g = U1Generator(data, recipe())
        args = dict(forbidden_rows=np.arange(150, 180), context="synthetic")
        first = g.fit_predict(np.arange(150), np.arange(150, 180), **args)
        data.x[0, 0] += .75
        second = g.fit_predict(np.arange(150), np.arange(150, 180), **args)
        self.assertNotEqual(first["lineage"]["bundle_key"], second["lineage"]["bundle_key"])
        self.assertEqual(g.cost_summary()["fit_count"], 76)

    def test_returned_lineage_cannot_corrupt_cached_ancestors(self):
        data = synthetic()
        g = U1Generator(data, recipe())
        args = dict(forbidden_rows=np.arange(150, 180), context="synthetic")
        first = g.fit_predict(np.arange(150), np.arange(150, 180), **args)
        first["lineage"]["selections"][0]["refit_groups"].append("GROUP-0179")
        second = g.fit_predict(np.arange(150), np.arange(150, 180), **args)
        self.assertNotIn("GROUP-0179", second["lineage"]["selections"][0]["refit_groups"])

    def test_derm_constant_and_absent_state_policy_matches_U0(self):
        data = synthetic()
        g = U1Generator(data, recipe())
        model = g._fit(np.flatnonzero(data.labels[:, 1] == 0), 1, 1.,
                       phase="synthetic_constant_policy", fold=None, bundle_key="synthetic")
        p = model.predict(data.x[:3])
        np.testing.assert_array_equal(p, np.array([[1., 0., 0.]]*3))
        self.assertTrue(g.fit_cost[0]["constant_probe"])
        self.assertEqual(g.fit_cost[0]["optimizer_iterations"], 0)

    def test_CBIS_support_is_checked_before_any_fit(self):
        data = synthetic(n=60, widths=(2,))
        data.labels[:50, 0] = 0
        data.labels[40:50, 0] = 1  # parent F supports 10, CV subfits cannot
        g = U1Generator(data, recipe("CBIS-DDSM"))
        with self.assertRaises(ValueError):
            g.fit_predict(np.arange(50), np.arange(50, 60),
                          forbidden_rows=np.arange(50, 60), context="synthetic")
        self.assertFalse(g.fit_cost)
        self.assertFalse(g.cache)

    def test_missing_F_declared_state_does_not_silently_change_metric(self):
        data = synthetic()
        data.labels[:150, 1] = data.labels[:150, 1] % 2
        g = U1Generator(data, recipe())
        with self.assertRaises(ValueError):
            g.fit_predict(np.arange(150), np.arange(150, 180),
                          forbidden_rows=np.arange(150, 180), context="synthetic")
        self.assertFalse(g.fit_cost)

    def test_nested_plan_error_fails_before_fits(self):
        data = synthetic()
        p = plan_for(data)
        bad = replace(p, final_oof_queries=(p.outer_fit_rows[:-1],))
        g = U1Generator(data, recipe())
        with self.assertRaises(ValueError):
            build_u1_nested_panel(g, bad)
        self.assertFalse(g.fit_cost)
        with self.assertRaises(ValueError):
            validate_nested_plan(data, replace(p, source_plan_sha256="missing"))

    def test_actual_nested_generator_excludes_all_H_E_G_ancestors(self):
        data = synthetic()
        g = U1Generator(data, recipe())
        panel = build_u1_nested_panel(g, plan_for(data))
        self.assertEqual(panel["raw"].shape, (144, 5))
        self.assertEqual(panel["val"].shape, (36, 5))
        self.assertEqual(len(panel["sel"]), 4)
        self.assertEqual(panel["cost_summary"]["unique_fit_bundles"], 21)
        self.assertEqual(panel["cost_summary"]["fit_count"], 21*2*19)
        self.assertEqual(panel["cost_summary"]["cache_reuse_calls"], 4)
        for record in panel["lineage"]:
            forbidden = set(record["forbidden_groups"])
            for s in record["selections"]:
                self.assertFalse(set(s["select_groups"]) & forbidden)
                for c in s["candidates"]:
                    for a in c["ancestors"]:
                        self.assertFalse((set(a["fit_groups"]) | set(a["validation_groups"])) & forbidden)
        self.assertEqual(panel["scientific_status"], "SOURCE_CANDIDATE_NOT_P3_COMPLETE")

    def test_CBIS_identical_fit_set_reuse_has_11_unique_bundles(self):
        data = synthetic(n=300, widths=(2, 2))
        g = U1Generator(data, recipe("CBIS-DDSM"))
        panel = build_u1_nested_panel(g, plan_for(data, cbis=True))
        self.assertEqual(panel["raw"].shape, (240, 2))
        self.assertEqual(panel["cost_summary"]["unique_fit_bundles"], 11)
        self.assertEqual(panel["cost_summary"]["fit_count"], 11*2*19)
        self.assertEqual(panel["cost_summary"]["cache_reuse_calls"], 10)

    def test_descriptive_diagnostics_reports_undefined_not_fake_macro(self):
        data = synthetic(n=6, widths=(2, 3))
        blocks = [np.full((6, 2), .5), np.full((6, 3), 1/3)]
        report = concept_diagnostics(data.labels, data.valid, blocks, data.names)
        self.assertAlmostEqual(report["macro_class_equal_nll"], (np.log(2)+np.log(3))/2)
        data.labels[:, 1] = 0
        report = concept_diagnostics(data.labels, data.valid, blocks, data.names)
        self.assertIsNone(report["macro_class_equal_nll"])
        self.assertIsNone(report["macro_auc"])
        self.assertEqual(report["per_concept"][1]["missing_states"], [1, 2])

    def test_source_derived_cost_counts_no_project_timing_claim(self):
        value = upstream_fit_forecast()
        self.assertEqual(value["cbis_u1_fit_count"], 10450)
        self.assertEqual(value["derm_u1_fit_count"], 69825)
        self.assertEqual(value["total_u1_fit_count"], 80275)
        self.assertFalse(value["timing_measured_on_project"])


if __name__ == "__main__":
    unittest.main()
