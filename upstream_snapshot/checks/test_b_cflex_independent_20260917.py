"""Independent synthetic-only mathematical gate for the 2026-09-17 control.

No project data, manuscript, held-out data or saved predictions are opened by
these tests. The oracle is deliberately independent of the implementation.
The production module is imported after its source-only inspection; all its
scientific-I/O entry points are mocked or prohibited in the adapter tests.
"""
from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def independent_restricted_cubic(z: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Four nonconstant columns; a separate fitted intercept is required.

    Five distinct knots yield z and three constrained truncated-cubic terms.
    Dividing cubic terms by the squared boundary span is a fixed rescaling,
    not an additional degree of freedom or a fitted parameter.
    """
    z = np.asarray(z, dtype=float).reshape(-1)
    knots = np.asarray(knots, dtype=float).reshape(-1)
    if knots.shape != (5,) or not np.all(np.isfinite(knots)):
        raise ValueError("exactly five finite knots are required")
    if not np.all(np.diff(knots) > 0) or not np.all(np.isfinite(z)):
        raise ValueError("knots must be distinct and inputs finite")
    k4, k5 = knots[-2:]
    span2 = (k5 - knots[0]) ** 2
    columns = [z]
    for kj in knots[:3]:
        h = (
            np.maximum(z - kj, 0.0) ** 3
            - (k5 - kj) / (k5 - k4) * np.maximum(z - k4, 0.0) ** 3
            + (k4 - kj) / (k5 - k4) * np.maximum(z - k5, 0.0) ** 3
        ) / span2
        columns.append(h)
    return np.column_stack(columns)


def test_four_nonconstant_degrees_plus_separate_intercept() -> None:
    knots = np.array([-2.0, -0.9, 0.1, 0.8, 1.7])
    x = np.linspace(-3.0, 3.0, 121)
    basis = independent_restricted_cubic(x, knots)
    assert basis.shape == (121, 4)
    assert np.linalg.matrix_rank(np.column_stack([np.ones(len(x)), basis])) == 5


def test_natural_linear_tails() -> None:
    knots = np.array([-2.0, -0.9, 0.1, 0.8, 1.7])
    for tail in (np.linspace(-4.0, -2.1, 31), np.linspace(1.8, 4.0, 31)):
        basis = independent_restricted_cubic(tail, knots)
        assert np.max(np.abs(np.diff(basis, n=2, axis=0))) < 1e-12


def test_second_derivative_continuity() -> None:
    knots = np.array([-2.0, -0.9, 0.1, 0.8, 1.7])
    step = 1e-4
    for knot in knots:
        points = knot + step * np.arange(-3, 4)
        basis = independent_restricted_cubic(points, knots)
        # Adjacent one-sided second differences converge to the same value.
        second = np.diff(basis, n=2, axis=0) / step**2
        assert np.max(np.abs(second[1] - second[3])) < 2e-4


def test_duplicate_knots_are_not_silently_a_lower_df_model() -> None:
    try:
        independent_restricted_cubic(np.array([0.0]), np.array([0.0, 0.0, 1.0, 2.0, 3.0]))
    except ValueError:
        return
    raise AssertionError("a nominal 4df model must not silently lose rank")


def test_unconstrained_scalar_map_need_not_preserve_auc_ranking() -> None:
    # A free spline logistic head permits a negative linear coefficient even
    # without using its nonlinear columns.  "Scalar calibration" therefore
    # does not imply positive monotonic or ranking-preserving calibration.
    z = np.array([-1.0, 1.0])
    basis = independent_restricted_cubic(z, np.array([-2.0, -0.9, 0.1, 0.8, 1.7]))
    mapped_logit = basis @ np.array([-1.0, 0.0, 0.0, 0.0])
    assert z[0] < z[1] and mapped_logit[0] > mapped_logit[1]


@pytest.fixture(scope="module")
def production():
    """Import the reviewed source, without invoking its main or data helpers."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_b_cflex_control_20260917.py"
    name = "independent_b_cflex_source_under_review"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_production_basis_matches_independent_formula(production) -> None:
    knots = np.array([-2.0, -0.9, 0.1, 0.8, 1.7])
    values = np.linspace(-4.0, 4.0, 241)
    np.testing.assert_allclose(
        production.RestrictedCubicBasis.raw(values, knots),
        independent_restricted_cubic(values, knots), rtol=1e-13, atol=1e-13,
    )


def test_production_knots_and_scaler_are_training_only(production) -> None:
    values = np.linspace(-3.0, 2.0, 131) + np.sin(np.arange(131)) * 0.01
    basis = production.RestrictedCubicBasis.fit(values)
    expected_knots = np.quantile(values, (0.05, 0.275, 0.5, 0.725, 0.95), method="linear")
    raw = independent_restricted_cubic(values, expected_knots)
    np.testing.assert_allclose(basis.knots, expected_knots)
    np.testing.assert_allclose(basis.mean, raw.mean(axis=0))
    np.testing.assert_allclose(basis.sd, raw.std(axis=0, ddof=0))
    before = basis.record()
    transformed = basis.transform(np.array([-100.0, 100.0]))
    assert transformed.shape == (2, 4) and np.isfinite(transformed).all()
    assert basis.record() == before
    assert before["nonintercept_basis_columns"] == 4
    assert before["train_rank_including_separate_intercept"] == 5


def test_production_degenerate_knots_fail_closed(production) -> None:
    with pytest.raises(production.ControlError):
        production.RestrictedCubicBasis.fit(np.zeros(12))
    with pytest.raises(production.ControlError):
        production.RestrictedCubicBasis.raw(np.array([0.0]), np.array([0.0, 0.0, 1.0, 2.0, 3.0]))


@pytest.mark.parametrize("g_count", [3, 4])
def test_deep_reference_c_cv_excludes_current_g(production, g_count) -> None:
    d = np.arange(g_count * 6)
    blocks = np.array_split(d, g_count)
    splits = [(np.setdiff1d(d, g), g) for g in blocks]
    for current_g, (k, excluded_g) in enumerate(splits):
        j_splits = production.reference_j_splits(splits, current_g)
        assert len(j_splits) == g_count - 1
        seen = []
        for train, validation in j_splits:
            assert set(train).isdisjoint(excluded_g)
            assert set(validation).isdisjoint(excluded_g)
            assert set(train).isdisjoint(validation)
            assert set(train) | set(validation) == set(k)
            seen.extend(validation.tolist())
        assert sorted(seen) == sorted(k.tolist())


def test_patient_exclusion_is_stricter_than_row_exclusion(production) -> None:
    groups = np.array(["same_patient", "same_patient", "third", "fourth"])
    production.assert_scope_disjoint(np.array([0]), np.array([1]), np.array([2]), groups, False)
    with pytest.raises(production.ControlError, match="patient"):
        production.assert_scope_disjoint(np.array([0]), np.array([1]), np.array([2]), groups, True)
    with pytest.raises(production.ControlError, match="row"):
        production.assert_scope_disjoint(np.array([0]), np.array([1]), np.array([0]), groups, False)


@pytest.mark.parametrize("domain", ["cbis", "derm"])
@pytest.mark.parametrize("corrupt_upstream_scope", [False, True])
def test_upstream_h_and_g_scope_manifests_are_bound(
    production, domain, corrupt_upstream_scope,
) -> None:
    """Synthetic cached scopes, not real assets or a reconstructed predictor."""
    t, e = np.arange(48), np.arange(48, 60)
    ids = np.array([f"synthetic_case_{i}" for i in range(60)])
    groups = np.array([f"synthetic_patient_{i}" for i in range(60)])
    hrows = np.array_split(t, 4)
    data = {"t": t, "e": e, "ids": ids, "groups": groups, "gold": np.zeros((60, 2))}
    source_folds = np.concatenate([np.full(12, h) for h in range(5)])
    manifests, splits_by_h = [], []

    def synthetic_scope_hash(values):
        return "|".join(np.asarray(values).astype(str).tolist())

    for h, heldout in enumerate(hrows):
        d = np.setdiff1d(t, heldout)
        data.update({f"f{h}": d, f"h{h}": heldout, f"x{h}": np.zeros((len(d), 2))})
        blocks = ([np.flatnonzero(source_folds == j) for j in range(4) if j != h]
                  if domain == "cbis" else np.array_split(d, 4))
        splits = [(np.setdiff1d(d, g), g) for g in blocks]
        splits_by_h.append(splits)
        if domain == "cbis":
            manifests.append({
                "fit_patient_sha256": synthetic_scope_hash(groups[d]),
                "heldout_patient_sha256": synthetic_scope_hash(groups[heldout]),
                "fit_heldout_patient_disjoint": True,
                "nested_probe_panels": [{
                    "probe_heldout_fold": int(source_folds[g[0]]),
                    "probe_fit_patient_sha256": synthetic_scope_hash(groups[k]),
                    "probe_validation_patient_sha256": synthetic_scope_hash(groups[g]),
                    "diagnosis_validation_excluded_from_probe_fit": True,
                    "probe_validation_excluded_from_probe_fit": True,
                } for k, g in splits],
            })
        else:
            manifests.append({
                "fit_case_ids": ids[d].tolist(), "heldout_case_ids": ids[heldout].tolist(),
                "inner": [{"fit_case_id_sha256": synthetic_scope_hash(ids[k]),
                           "validation_case_id_sha256": synthetic_scope_hash(ids[g]),
                           "fit_validation_disjoint": True} for k, g in splits],
            })
    if corrupt_upstream_scope:
        if domain == "cbis":
            manifests[0]["nested_probe_panels"][0]["probe_fit_patient_sha256"] = "wrong_scope"
        else:
            manifests[0]["inner"][0]["fit_case_id_sha256"] = "wrong_scope"
    helper = SimpleNamespace(
        validate_generation_partition=lambda *args: manifests,
        sha256_values=synthetic_scope_hash,
        _nested_subfolds=lambda domain, rep, outer, h, d, other: splits_by_h[h],
    )
    if corrupt_upstream_scope:
        with pytest.raises(production.ControlError, match="G upstream generating scope"):
            production.build_g_splits((domain, 0, 0), data, {}, helper, source_folds, manifests)
    else:
        actual = production.build_g_splits((domain, 0, 0), data, {}, helper, source_folds, manifests)
        assert len(actual) == 4
        for h, splits in enumerate(actual):
            for k, g in splits:
                assert set(k).isdisjoint(hrows[h]) and set(g).isdisjoint(hrows[h])
                assert set(k).isdisjoint(e) and set(g).isdisjoint(e)


def test_exact_unit_whitelist_and_fit_counts(production) -> None:
    expected = tuple(("cbis", 0, fold) for fold in range(5)) + tuple(
        ("derm", repetition, fold) for repetition in range(5) for fold in range(5)
    )
    assert production.expected_units() == expected
    assert len(set(expected)) == 30
    # Each H has G reference refits, each with six candidates on G-1 blocks.
    assert production.REFERENCE_FITS["cbis"] == 4 * 3 * (6 * 2 + 1)
    assert production.REFERENCE_FITS["derm"] == 4 * 4 * (6 * 3 + 1)
    assert production.frozen_budgets()["reference_fits_full"] == 5 * 156 + 25 * 304
    assert production.frozen_budgets()["spline_fits_full"] == 30 * (6 * 4 + 1)
    paths = production.expected_input_relpaths()
    assert len(paths) == 273 and len(set(paths)) == 273
    assert all("test" not in path.lower() for path in paths)


def test_cost_outer_prediction_guard_precedes_any_io(production, monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("cost mode attempted outer I/O")
    monkeypatch.setattr(production, "read_csv", forbidden)
    with pytest.raises(production.ControlError, match="prohibited"):
        production.predict_outer(("cbis", 0, 0), {}, {}, SimpleNamespace(), "cost")


def test_cost_main_dispatch_never_calls_outer_predictor(production, monkeypatch, tmp_path) -> None:
    calls = []
    receipts = {}
    protocol = {"script_sha256": "synthetic_source", "protocol_id": "synthetic_protocol"}
    monkeypatch.setattr(production, "CONTROL_ROOT", tmp_path / "synthetic_control")
    production.CONTROL_ROOT.mkdir()
    monkeypatch.setattr(production, "verify_protocol", lambda _: protocol)
    monkeypatch.setattr(production, "import_v5", lambda: SimpleNamespace())
    monkeypatch.setattr(production, "sha256_file", lambda _: "synthetic_artifact")
    monkeypatch.setattr(production, "write_new_json", lambda path, value: receipts.__setitem__(str(path), value))
    monkeypatch.setattr(production, "write_new_csv", lambda *args: None)

    def synthetic_training(unit, frozen, budget, helpers):
        calls.append(unit)
        artifacts = {key: {} for key in ("models", "basis", "scope_definitions", "fit_scopes", "selections")}
        artifacts["training_logits"] = []
        artifacts["run_record"] = {"E_prediction_performed": False, "E_metrics_performed": False}
        return artifacts, {"head": SimpleNamespace()}

    def forbidden(*args, **kwargs):
        raise AssertionError("cost main dispatched an outer predictor")

    monkeypatch.setattr(production, "compute_training", synthetic_training)
    monkeypatch.setattr(production, "predict_outer", forbidden)
    assert production.main(["--mode", "cost", "--protocol-sha256", "synthetic"]) == 0
    assert calls == [("cbis", 0, 0), ("derm", 0, 0)]
    complete = receipts[str(production.CONTROL_ROOT / "cost" / "RUN_COMPLETE.json")]
    assert complete["E_prediction_performed"] is False
    assert complete["E_metrics_performed"] is False
    assert complete["official_test_accessed"] is False


@pytest.fixture(scope="module")
def aggregation():
    path = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_b_cflex_control_20260917.py"
    spec = importlib.util.spec_from_file_location("independent_b_cflex_aggregation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pairwise_weighted_auc(y, scores, weights):
    """Independent quadratic positive-negative pair oracle, including ties."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    numerator = sum(
        weights[i] * weights[j] * (float(scores[i] > scores[j]) + 0.5 * float(scores[i] == scores[j]))
        for i in pos for j in neg
    )
    return numerator / (weights[pos].sum() * weights[neg].sum())


def test_aggregate_weighted_auc_matches_independent_pairwise_oracle(aggregation):
    rng = np.random.default_rng(771906)
    for _ in range(100):
        y = np.array([0, 1] * 15)
        scores = rng.choice([0.0, 0.2, 0.6, 0.9, 1.0], len(y))
        weights = rng.integers(0, 5, len(y)).astype(float)
        weights[:2] = 1.0
        actual = aggregation.weighted_auc(aggregation.auc_cache(y, scores), weights)
        assert abs(actual - pairwise_weighted_auc(y, scores, weights)) < 5e-16


def test_aggregate_class_equal_nll_does_not_use_natural_frequency(aggregation):
    y, loss, w = np.array([0, 0, 0, 1]), np.array([1., 2., 4., 10.]), np.array([1., 2., 3., 4.])
    oracle = 0.5 * ((1. + 4. + 12.) / 6. + 10.)
    assert aggregation.ceq(y, loss, w) == oracle
    assert aggregation.ceq(y, loss, w) != np.average(loss, weights=w)


def synthetic_panel(aggregation, domain):
    import pandas as pd
    n, reps = (655, 1) if domain == "cbis" else (609, 5)
    rng = np.random.default_rng(69217)
    y = np.arange(n) % 2
    records = []
    for r in range(reps):
        for i in range(n):
            row = {"rep": r, "fold": i % 5, "case_id": f"synthetic_{i:04}",
                   "group_id": f"synthetic_patient_{i // 2:04}", "y": y[i]}
            row.update({f"p_{arm}": rng.uniform(0.01, 0.99) for arm in aggregation.ARMS})
            records.append(row)
    return pd.DataFrame(records)


def test_aggregate_repetitions_average_loss_and_auc_not_probabilities(aggregation):
    panel = synthetic_panel(aggregation, "derm")
    ctx = aggregation.context(panel, "derm")
    y, losses, _, _, _, points, cases = ctx
    ps = np.array([panel[panel.rep == r].sort_values("case_id").p_M for r in range(5)])
    oracle_loss = (-y * np.log(ps) - (1-y) * np.log1p(-ps)).mean(axis=0)
    np.testing.assert_allclose(losses["M"], oracle_loss)
    oracle_auc = np.mean([pairwise_weighted_auc(y, p, np.ones(len(y))) for p in ps])
    assert abs(points["M_AUC"] - oracle_auc) < 5e-16
    averaged_probability_loss = -y*np.log(ps.mean(axis=0))-(1-y)*np.log1p(-ps.mean(axis=0))
    assert np.max(np.abs(losses["M"] - averaged_probability_loss)) > 0.01
    np.testing.assert_allclose(cases.vector_over_flex_NLL_gain, losses["Cflex"]-losses["M"])
    assert points["vector_over_flex_AUC_gain"] == points["M_AUC"]-points["Cflex_AUC"]


def test_aggregate_domain_cluster_unit_is_patient_or_case(aggregation):
    cbis = aggregation.context(synthetic_panel(aggregation, "cbis"), "cbis")
    derm = aggregation.context(synthetic_panel(aggregation, "derm"), "derm")
    assert cbis[4] == 328 and cbis[3][0] == cbis[3][1]
    assert derm[4] == 609 and derm[3][0] != derm[3][1]


def test_aggregate_bootstrap_pairing_rejects_one_class_and_has_exact_draws(aggregation):
    y = np.array([0, 1])
    loss = {arm: np.array([i + 0.2, i + 0.5]) for i, arm in enumerate(aggregation.ARMS)}
    caches = {arm: [aggregation.auc_cache(y, np.array([0.2, 0.8]))] for arm in aggregation.ARMS}
    ctx = y, loss, caches, np.array([0, 1]), 2, {}, None

    class FixedSyntheticRng:
        calls = 0
        def integers(self, low, high, size):
            assert (low, high, size) == (0, 2, 2)
            self.calls += 1
            return np.array([0, 0]) if self.calls == 1 else np.array([0, 1])

    draws, proposals = aggregation.bootstrap(ctx, FixedSyntheticRng())
    assert len(draws) == 10000 and proposals == 10001
    assert draws.proposal.iloc[0] == 2
    np.testing.assert_allclose(draws.vector_over_flex_NLL_gain, -1.0)
    np.testing.assert_allclose(draws.vector_over_flex_AUC_gain, 0.0)


def test_frozen_protocol_matches_reviewed_sources_and_exact_whitelist(production, aggregation):
    root = Path(__file__).resolve().parents[1]
    path = root / "paper_b_readiness/CFLEX_CONTROL_20260917/PROTOCOL.json"
    content = path.read_bytes()
    protocol = json.loads(content)
    assert hashlib.sha256(content).hexdigest() == "0875aaffc80804daab8e04d9a97c675d9880b4af3efecb8e6749d75c8e7f118d"
    assert protocol["status"] == "FROZEN_PRE_EXECUTION"
    for key, filename in (("script_sha256", "run_b_cflex_control_20260917.py"),
                          ("aggregate_script_sha256", "aggregate_b_cflex_control_20260917.py"),
                          ("helper_sha256", "run_p1_formal_unit_v5_20260916.py")):
        assert hashlib.sha256((root / "scripts" / filename).read_bytes()).hexdigest() == protocol[key]
    assert protocol["settings"] == production.frozen_settings()
    assert protocol["budgets"] == production.frozen_budgets()
    assert protocol["ordered_units"] == list(aggregation.UNITS)
    assert set(protocol["input_sha256"]) == set(production.expected_input_relpaths())
    assert len(protocol["input_sha256"]) == 273
    assert all(len(h) == 64 and int(h, 16) >= 0 for h in protocol["input_sha256"].values())
    assert protocol["aggregation"]["primary_contrast"] == "L_class_equal(Cflex)-L_class_equal(M)"
    assert protocol["resource_decision"]["delta_nats"] == 0.005
    assert protocol["access_boundary"]["official_reserved_Test_access"] is False
