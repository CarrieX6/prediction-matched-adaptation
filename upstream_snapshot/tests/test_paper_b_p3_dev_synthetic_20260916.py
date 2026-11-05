"""Generated arrays and temporary files only; never loads project assets."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import run_neurocomputing_p3_dev_20260916 as runner
import paper_b_p3_u1 as u1
from test_paper_b_p3_u1_synthetic_20260916 import synthetic, recipe, plan_for


def paired_rows(domain="cbis", control="SP"):
    rows = []
    reps = (0,) if domain == "cbis" else range(5)
    arms = [(a, s) for a in runner.CORE for s in ((0,) if a.endswith("/lr") else runner.p2.SEEDS)] + [("S", 0), (control, 0)]
    for upstream in ("U0", "U1"):
        for arm, seed in arms:
            for rep in reps:
                for i in range(12):
                    y = i % 2
                    logit = (2 * y - 1) * (.2 + .07 * i) + (.1 if upstream == "U1" else 0) * (2*y-1)
                    rows.append(dict(domain=domain, upstream=upstream, arm=arm, refit_seed=seed,
                        repetition=rep, case_id=f"SYNTHETIC-{i}", patient_group=f"PATIENT-{i//2}", label=y, logit=logit))
    return rows


class P3DevSyntheticTests(unittest.TestCase):
    def test_default_never_constructs_project_loader_or_fit(self):
        with patch.object(runner, "source_check", return_value={"synthetic": True}), patch.object(runner, "DevInputs", side_effect=AssertionError("no loader")), patch("sys.argv", ["p3"]), contextlib.redirect_stdout(io.StringIO()) as out:
            runner.main()
        self.assertEqual(json.loads(out.getvalue())["status"], "SOURCE_ONLY_CHECK_PASS_NOT_EXECUTED")

    def test_execution_requires_explicit_external_gate(self):
        for argv in (["p3", "--execute"], ["p3", "--representative", "--execute"]):
            with patch.object(runner, "source_check", return_value={}), patch.object(runner, "DevInputs", side_effect=AssertionError("no loader")), patch("sys.argv", argv), self.assertRaises(runner.P3Error):
                runner.main()

    def test_allowlist_rejects_official_test_and_arbitrary_units_before_io(self):
        obj = object.__new__(runner.DevInputs)
        for name, unit in (("test", None), ("panel", "cbis_r9_f0"), ("generation", "derm_r5_f0")):
            with self.assertRaises(runner.P3Error):
                obj.get(name, unit=unit)

    def test_mean_views_preserves_original_recipe(self):
        image_ids = np.array([["CC", "MLO"], ["CC", "missing"]])
        x = runner.average_views(image_ids, np.array([[1,1],[1,0]]), np.array(["CC","MLO"]), np.eye(2))
        np.testing.assert_array_equal(x, [[.5,.5],[1,0]])
        self.assertNotAlmostEqual(np.linalg.norm(x[0]), 1)
        with self.assertRaises(runner.P3Error):
            runner.average_views(image_ids, np.ones((2,2)), np.array(["CC","MLO"]), np.eye(2))

    def test_fixed_recipe_does_not_depend_on_open_history(self):
        data = synthetic()
        inputs = object.__new__(runner.DevInputs)
        inputs.records = {v: dict(path=v, sha256="a"*64, bytes=1) for v in runner.DEV_ASSETS.values()}
        inputs.opened = []
        first = runner.new_generator({"data":data}, "derm", inputs).recipe
        inputs.opened.append(dict(path="unrelated_generation", sha256="b"*64))
        second = runner.new_generator({"data":data}, "derm", inputs).recipe
        self.assertEqual(first, second)
        self.assertEqual(first.c_grid, u1.C_GRID)

    def test_exact_cb_is_one_legal_inner_760_logical_fits(self):
        data = synthetic(n=300, widths=(2,)*10)
        plan = plan_for(data, cbis=True)
        result = u1.build_u1_representative_inner(u1.U1Generator(data, recipe("CBIS-DDSM")), plan, inner_index=0)
        self.assertEqual(result["cost_summary"]["fit_count"],760)
        self.assertEqual(result["cost_summary"]["unique_fit_bundles"],4)
        self.assertFalse(result["outer_predictions_generated"])
        self.assertFalse(result["final_refit_generated"])

    def test_cb_plan_reconstruction_is_original_folds(self):
        data = synthetic(n=300, widths=(2,)*10)
        cohort = dict(data=data, original_fold=np.repeat(np.arange(5),60))
        plan = runner.make_plan(cohort,"cbis_r0_f0","a"*64)
        np.testing.assert_array_equal(plan.outer_evaluation_rows,np.arange(60))
        self.assertEqual(len(plan.diagnostic_splits),4)
        self.assertTrue(all(len(s.oof_queries)==3 for s in plan.diagnostic_splits))

    def test_atomic_publication_verification_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runner,"ROOT",Path(temp)):
            final = Path(temp)/"results"/"unit"
            with runner.staged(final,"b"*64,"synthetic") as stage:
                runner.write_json(stage/"payload.json",{"generated":True})
                self.assertFalse(final.exists())
            runner.verify_complete(final,"b"*64)
            with self.assertRaises(runner.P3Error), runner.staged(final,"b"*64,"synthetic"):
                pass
            with (final/"payload.json").open("a") as stream:
                stream.write(" ")
            with self.assertRaises(runner.P3Error):runner.verify_complete(final,"b"*64)

    def test_failure_quarantine_new_sibling_retry(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runner,"ROOT",Path(temp)):
            final=Path(temp)/"results"/"unit"
            with self.assertRaises(ValueError), runner.staged(final,"b"*64,"synthetic") as stage:
                runner.write_json(stage/"partial.json",{"generated":True})
                raise ValueError("synthetic interruption")
            failures=list(final.parent.glob("unit.staging-*"))
            self.assertEqual(len(failures),1)
            self.assertTrue((failures[0]/"FAILED.json").exists())
            self.assertFalse(final.exists())
            with runner.staged(final,"b"*64,"synthetic") as stage:
                runner.write_json(stage/"new.json",{"retry":True})
            self.assertEqual(set(runner.verify_complete(final,"b"*64)["payload"]),{"new.json"})

    def test_symlink_result_parent_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as other, patch.object(runner,"ROOT",Path(temp)):
            (Path(temp)/"results").symlink_to(other,target_is_directory=True)
            with self.assertRaises(runner.P3Error), runner.staged(Path(temp)/"results"/"new"/"unit","b"*64,"synthetic"):
                pass
            self.assertFalse((Path(other)/"new").exists())

    def test_shared_paired_ci_gain_and_case_cluster_contract(self):
        report=runner.paired_ci(paired_rows(),"cbis","SP",replicates=32,maximum=1000)
        self.assertEqual(report["cluster_count"],6)
        self.assertEqual(len(report["draws"]),32)
        gain=next(v for v in report["paired_effects"] if v["metric"]=="class_equal_nll" and v["contrast"]=="upstream_gain")
        self.assertGreater(gain["estimate"],0)
        self.assertTrue(all(len(d["cluster_multiplicity"])==6 for d in report["draws"]))
        duplicate=paired_rows();duplicate.append(dict(duplicate[0]))
        with self.assertRaises(runner.P3Error):runner.prepare_estimators(duplicate,"cbis","SP")

    def test_derm_repetitions_remain_paired_case_clusters(self):
        report=runner.paired_ci(paired_rows("derm"),"derm","SP",replicates=8,maximum=1000)
        self.assertEqual(report["cluster_count"],12)
        self.assertIn("seeds stay together",report["inference"])

    def test_reference_s_and_all_legal_controls_synthetic(self):
        data=synthetic(n=150,widths=(2,)*2)
        plan=plan_for(data,cbis=True)
        y=data.labels[:,0]
        co=dict(data=data,y=y,gold=data.labels.astype(float),widths=np.array([],dtype=int))
        panel=runner.metadata_panel(co,plan)
        p=.15+.7*co["gold"]
        panel.update(raw=p[plan.outer_fit_rows],val=p[plan.outer_evaluation_rows])
        for i,s in enumerate(plan.diagnostic_splits):panel[f"x{i}"],panel[f"z{i}"]=p[s.fit_rows],p[s.validation_rows]
        for control in runner.CONTROLS:
            predictions,record=runner.reference_and_control(panel,plan,control,"cbis_r0_f0","U1")
            self.assertEqual(set(r["arm"] for r in predictions),{"S",control})
            self.assertTrue(all(np.isfinite(r["logit"]) for r in predictions))
            self.assertEqual(record["cost"]["head_by_role"]["S/final"],1)


if __name__ == "__main__":
    unittest.main()
