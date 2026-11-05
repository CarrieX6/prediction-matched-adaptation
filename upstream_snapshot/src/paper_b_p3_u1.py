"""Source-only P3/U1 concept generator; no project loader or diagnosis target.

The caller must supply audited, development-only frozen arrays and an externally
frozen legal split plan. Importing this module never opens project data. This API
does not authorize execution, choose a P1 control, score diagnosis, or implement
P4/official-test access. U1 is six-C, three-group-fold selection inside EACH F.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
from hashlib import sha256
import json
import resource
import time
import warnings
from typing import Any, Sequence

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
VERSION = "PAPER_B_P3_U1_SOURCE_CANDIDATE_V1_20260916"


def _hash(value: Any) -> str:
    def numpy_json(item: Any) -> Any:
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"unsupported canonical JSON type: {type(item).__name__}")
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False, default=numpy_json).encode()).hexdigest()


def _seed(base: int, *parts: Any) -> int:
    return int.from_bytes(sha256("|".join(map(str, (base, *parts))).encode()).digest()[:4],
                          "little") & 0x7fffffff


def _ids(data: "ConceptData", rows: np.ndarray) -> list[str]:
    return sorted(data.ids[rows].astype(str).tolist())


def _groups(data: "ConceptData", rows: np.ndarray) -> list[str]:
    return sorted(set(data.groups[rows].astype(str)))


def _rows(data: "ConceptData", rows: Sequence[int], name: str,
          *, empty_ok: bool = False) -> np.ndarray:
    raw = np.asarray(rows)
    if raw.ndim != 1 or (raw.size and raw.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be one-dimensional integer row indices")
    value = np.sort(raw.astype(np.int64))
    if (not empty_ok and not len(value)) or len(np.unique(value)) != len(value):
        raise ValueError(f"{name} is empty or duplicated")
    if np.any(value < 0) or np.any(value >= len(data.ids)):
        raise ValueError(f"{name} is out of bounds")
    return value


@dataclass(frozen=True)
class U1Recipe:
    # Explicit source-preserving domain choices, NOT outcome-selected settings.
    domain: str
    seed: int
    max_iter: int
    minimum_state_support: int
    feature_contract_sha256: str
    c_grid: tuple[float, ...] = C_GRID
    selection_folds: int = 3

    def validate(self) -> None:
        if self.domain not in ("CBIS-DDSM", "Derm7pt"):
            raise ValueError("only the two frozen domains are allowed")
        if tuple(self.c_grid) != C_GRID or self.selection_folds != 3:
            raise ValueError("U1 may not expand/change the frozen six-C/three-fold grid")
        if self.max_iter <= 0 or self.minimum_state_support < 0:
            raise ValueError("invalid optimizer/support contract")
        if self.domain == "CBIS-DDSM" and self.minimum_state_support != 10:
            raise ValueError("CBIS retains U0's per-state minimum support 10")
        if self.domain == "Derm7pt" and self.minimum_state_support != 0:
            raise ValueError("Derm retains U0's absent-state/constant-probe policy")
        if len(self.feature_contract_sha256) != 64 or any(
                c not in "0123456789abcdef" for c in self.feature_contract_sha256):
            raise ValueError("an external feature-contract SHA-256 is required")


@dataclass(frozen=True)
class ConceptData:
    x: np.ndarray
    ids: np.ndarray
    groups: np.ndarray
    labels: np.ndarray
    valid: np.ndarray
    names: tuple[str, ...]
    widths: tuple[int, ...]

    def validate_structure(self) -> None:
        n = len(self.ids)
        if self.ids.shape != (n,) or self.groups.shape != (n,):
            raise ValueError("IDs/groups must be vectors")
        if len(set(self.ids.astype(str))) != n:
            raise ValueError("case/unit IDs must be unique")
        if any(not str(v) for v in self.ids) or any(not str(v) for v in self.groups):
            raise ValueError("empty case/group IDs")
        if self.x.ndim != 2 or len(self.x) != n or self.x.dtype.kind not in "fi":
            raise ValueError("frozen feature array is malformed")
        shape = (n, len(self.names))
        if self.labels.shape != shape or self.valid.shape != shape:
            raise ValueError("concept labels/validity shape mismatch")
        if self.labels.dtype.kind not in "iu" or self.valid.dtype.kind != "b":
            raise ValueError("integer concept labels and boolean validity are required")
        if len(self.widths) != len(self.names) or not self.names or any(w < 2 for w in self.widths):
            raise ValueError("concept block layout is invalid")
        if len(set(self.names)) != len(self.names):
            raise ValueError("concept names must be unique")
        # No target label values are inspected here. Only current F labels may
        # enter upstream fitting, model selection, support checks or priors.


def class_equal_nll(labels: np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    if (p.ndim != 2 or len(y) != len(p) or not len(y) or not np.isfinite(p).all()
            or np.any(p < 0) or np.any(p > 1)
            or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-10)):
        raise ValueError("NLL inputs are malformed")
    if np.any(y < 0) or np.any(y >= p.shape[1]):
        raise ValueError("NLL target outside concept state range")
    counts = np.bincount(y, minlength=p.shape[1])
    if np.any(counts == 0):
        raise ValueError("class-equal concept NLL requires all declared states in F")
    loss = -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.0))
    return float(np.mean([loss[y == state].mean() for state in range(p.shape[1])]))


@dataclass
class _Probe:
    width: int
    model: Any
    constant: int | None
    optimizer_iterations: int

    def predict(self, x: np.ndarray) -> np.ndarray:
        p = np.zeros((len(x), self.width), dtype=np.float64)
        if self.constant is not None:
            p[:, self.constant] = 1.0
        else:
            p[:, self.model.classes_.astype(int)] = self.model.predict_proba(x)
        if not np.isfinite(p).all() or np.any(p < 0) or np.any(p.sum(1) <= 0):
            raise ValueError("invalid concept probability")
        return p / p.sum(1, keepdims=True)


class U1Generator:
    """Reusable in-memory U1 runner. All fit ancestors remain wholly inside F.

    Cache keys include current F data + full recipe. Reuse only avoids fitting
    identical recipes on identical fit sets; every query gets a new firewall
    check and lineage record. There is deliberately no diagnosis-label input.
    """

    def __init__(self, data: ConceptData, recipe: U1Recipe):
        data.validate_structure()
        recipe.validate()
        self.data, self.recipe = data, recipe
        self.recipe_sha256 = _hash(asdict(recipe))
        self.cache: dict[str, dict[str, Any]] = {}
        self.fit_cost: list[dict[str, Any]] = []
        self.prediction_cost: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.started = time.perf_counter()

    def _group_splits(self, f: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        groups = np.asarray(_groups(self.data, f))
        if len(groups) < 3:
            raise ValueError("current F has fewer than three groups; no grid/split fallback")
        splitter = KFold(n_splits=3, shuffle=True,
                         random_state=_seed(self.recipe.seed, "U1_CV", _hash(_ids(self.data, f))))
        output = []
        for a, b in splitter.split(groups):
            fit = f[np.isin(self.data.groups[f].astype(str), groups[a])]
            val = f[np.isin(self.data.groups[f].astype(str), groups[b])]
            if set(_groups(self.data, fit)) & set(_groups(self.data, val)):
                raise AssertionError("U1 group selection split overlaps")
            output.append((fit, val))
        if sorted(np.concatenate([b for _, b in output]).tolist()) != f.tolist():
            raise AssertionError("U1 selection validation folds do not partition F")
        return output

    def _fit(self, rows: np.ndarray, column: int, c: float, *, phase: str,
             fold: int | None, bundle_key: str) -> _Probe:
        d, r = self.data, self.recipe
        valid = rows[d.valid[rows, column]]
        y, width = d.labels[valid, column], int(d.widths[column])
        if not len(valid) or np.any(y < 0) or np.any(y >= width):
            raise ValueError(f"{d.names[column]}: malformed/absent F-only concept labels")
        x = np.asarray(d.x[valid], dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError("nonfinite training features")
        counts = np.bincount(y, minlength=width)
        if np.any(counts < r.minimum_state_support):
            raise ValueError(f"{d.names[column]}: frozen support fails inside F: {counts.tolist()}")
        seed = _seed(r.seed, "U1_fit", d.names[column], _hash(_ids(d, valid)), c)
        observed = np.unique(y)
        model = None
        constant = int(observed[0]) if len(observed) == 1 else None
        started = time.perf_counter()
        rss_before = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        if constant is None:
            # Same linear, balanced, L2/lbfgs family as source U0. No scaler,
            # calibration or diagnosis labels are inserted upstream.
            weight = np.array([len(y) / (len(observed) * counts[state]) for state in y])
            model = LogisticRegression(C=c, solver="lbfgs", penalty="l2",
                                       max_iter=r.max_iter, random_state=seed)
            with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(x, y, sample_weight=weight)
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                raise RuntimeError(f"{d.names[column]} C={c}: nonconvergence; no silent retry")
        iterations = 0 if model is None else int(np.max(model.n_iter_))
        record = dict(bundle_key=bundle_key, concept=d.names[column], phase=phase,
                      fold=fold, c=float(c), seed=seed, fit_ids=_ids(d, valid),
                      fit_groups=_groups(d, valid), fit_id_sha256=_hash(_ids(d, valid)),
                      fit_group_sha256=_hash(_groups(d, valid)), state_counts=counts.tolist(),
                      empirical_prior=(counts / counts.sum()).tolist(),
                      fit_seconds=time.perf_counter() - started,
                      optimizer_iterations=iterations,
                      optimizer_updates_exact=None, fit_count=1,
                      constant_probe=constant is not None,
                      rss_before_process_highwater_bytes=rss_before,
                      rss_after_process_highwater_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                      memory_measurement="process high-water RSS, not per-fit incremental peak",
                      preprocessing_fit_ids=[], calibration_fit_ids=[],
                      prior_used_in_predictions=False)
        self.fit_cost.append(record)
        return _Probe(width, model, constant, iterations)

    def fit_predict(self, fit_rows: Sequence[int], query_rows: Sequence[int], *,
                    forbidden_rows: Sequence[int], context: str) -> dict[str, Any]:
        d, r = self.data, self.recipe
        f, q = _rows(d, fit_rows, "F"), _rows(d, query_rows, "Q")
        forbidden = _rows(d, forbidden_rows, "forbidden", empty_ok=True)
        fg, qg, bg = set(_groups(d, f)), set(_groups(d, q)), set(_groups(d, forbidden))
        if fg & (qg | bg):
            raise ValueError("F intersects query/forbidden patient or case groups")
        if not qg <= bg:
            raise ValueError("forbidden set must include every query group")
        if r.domain == "CBIS-DDSM" and any(w != 2 for w in d.widths):
            raise ValueError("CBIS retains binary concept blocks")
        # Cache fingerprint reads ONLY current F labels/features/validity. Q
        # labels are never inspected and cannot change selection/cache identity.
        fingerprint = sha256()
        for array in (np.asarray(d.x[f], dtype=np.float64), d.labels[f], d.valid[f]):
            fingerprint.update(np.ascontiguousarray(array).tobytes())
        key = _hash(dict(recipe=self.recipe_sha256, fit_ids=_ids(d, f),
                         groups=_groups(d, f), fit_data_sha256=fingerprint.hexdigest(),
                         names=d.names, widths=d.widths))
        reused = key in self.cache
        if not reused:
            # Validate all F concept states BEFORE incurring a partial fit queue.
            for column, width in enumerate(d.widths):
                y = d.labels[f[d.valid[f, column]], column]
                if not len(y) or np.any(y < 0) or np.any(y >= width):
                    raise ValueError("F-only concept labels are absent/out of range")
                if np.any(np.bincount(y, minlength=width) == 0):
                    raise ValueError("current F lacks a declared state for class-equal selection")
            splits = self._group_splits(f)
            # CBIS's support floor applies inside each actual selection fit,
            # not just the parent F. Fail BEFORE fitting any candidate.
            for column, width in enumerate(d.widths):
                for a in [f] + [fit for fit, _ in splits]:
                    y = d.labels[a[d.valid[a, column]], column]
                    if not len(y) or np.any(np.bincount(y, minlength=width) < r.minimum_state_support):
                        raise ValueError(f"{d.names[column]}: frozen support fails in U1 selection fit")
            models, selections = [], []
            cost_start = len(self.fit_cost)
            for column, width in enumerate(d.widths):
                candidates = []
                for c in r.c_grid:
                    oof = np.full((len(f), width), np.nan)
                    candidate_ancestors = []
                    for fold, (a, b) in enumerate(splits):
                        model = self._fit(a, column, c, phase="C_selection", fold=fold, bundle_key=key)
                        local = np.searchsorted(f, b)
                        oof[local] = model.predict(np.asarray(d.x[b], dtype=np.float64))
                        valid_a = a[d.valid[a, column]]
                        candidate_ancestors.append(dict(fold=fold, fit_ids=_ids(d, valid_a),
                            fit_groups=_groups(d, valid_a), validation_ids=_ids(d, b),
                            validation_groups=_groups(d, b), group_disjoint=True))
                    valid_f = d.valid[f, column]
                    score = class_equal_nll(d.labels[f[valid_f], column], oof[valid_f])
                    candidates.append(dict(c=float(c), class_equal_oof_nll=score,
                                           ancestors=candidate_ancestors))
                chosen = min(candidates, key=lambda item: (item["class_equal_oof_nll"], item["c"]))
                model = self._fit(f, column, chosen["c"], phase="selected_refit", fold=None, bundle_key=key)
                models.append(model)
                selections.append(dict(concept=d.names[column], selected_c=chosen["c"],
                    selection_metric="pooled F-only OOF class-equal concept NLL",
                    tie_break="smallest C at exact full-precision tie", candidates=candidates,
                    select_ids=_ids(d, f[d.valid[f, column]]), select_groups=_groups(d, f),
                    refit_ids=_ids(d, f[d.valid[f, column]]), refit_groups=_groups(d, f)))
            self.cache[key] = dict(models=models, selections=selections,
                                   fit_records=self.fit_cost[cost_start:])
        bundle = self.cache[key]
        started = time.perf_counter()
        xq = np.asarray(d.x[q], dtype=np.float64)
        if not np.isfinite(xq).all():
            raise ValueError("nonfinite query features")
        blocks = [model.predict(xq) for model in bundle["models"]]
        lineage = dict(version=VERSION, context=context, recipe_sha256=self.recipe_sha256,
            feature_contract_sha256=r.feature_contract_sha256, bundle_key=key,
            target_ids=_ids(d, q), target_groups=_groups(d, q),
            forbidden_ids=_ids(d, forbidden), forbidden_groups=_groups(d, forbidden),
            fit_ids=_ids(d, f), fit_groups=_groups(d, f), selections=copy.deepcopy(bundle["selections"]),
            model_family="balanced L2 LogisticRegression/lbfgs; domain U0 constant/absent-state policy",
            sklearn_version=sklearn.__version__,
            calibrate_ids=[], scale_ids=[], diagnosis_label_access=False,
            cache_reused=reused, query_labels_accessed=False,
            all_ancestor_groups_disjoint_from_forbidden=True)
        # Independent call-level check over EVERY candidate/refit ancestor.
        for selection in lineage["selections"]:
            for group_list in (selection["select_groups"], selection["refit_groups"]):
                if set(group_list) & bg:
                    raise AssertionError("selection/refit ancestor enters forbidden groups")
            for candidate in selection["candidates"]:
                for ancestor in candidate["ancestors"]:
                    if (set(ancestor["fit_groups"]) | set(ancestor["validation_groups"])) & bg:
                        raise AssertionError("upstream C-selection ancestor enters forbidden groups")
        self.calls.append(lineage)
        self.prediction_cost.append(dict(context=context, query_count=len(q), bundle_key=key,
            cache_reused=reused, seconds=time.perf_counter() - started, fit_count=0))
        return dict(query_rows=q, blocks=tuple(blocks), lineage=lineage)

    def cost_summary(self) -> dict[str, Any]:
        return dict(unique_fit_bundles=len(self.cache), fit_count=len(self.fit_cost),
            logical_probe_fit_count=len(self.fit_cost),
            constant_probe_count=sum(v["constant_probe"] for v in self.fit_cost),
            actual_optimizer_fit_count=sum(not v["constant_probe"] for v in self.fit_cost),
            selection_fit_count=sum(v["phase"] == "C_selection" for v in self.fit_cost),
            refit_count=sum(v["phase"] == "selected_refit" for v in self.fit_cost),
            optimizer_iterations=sum(v["optimizer_iterations"] for v in self.fit_cost),
            optimizer_updates_exact=None,
            fit_seconds=sum(v["fit_seconds"] for v in self.fit_cost),
            prediction_seconds=sum(v["seconds"] for v in self.prediction_cost),
            total_elapsed_seconds=time.perf_counter() - self.started,
            process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            cache_reuse_calls=sum(v["cache_reused"] for v in self.calls),
            note="synthetic timing is NOT a real project cost forecast")


@dataclass(frozen=True)
class DiagnosticSplit:
    fit_rows: Sequence[int]  # D = T minus H
    validation_rows: Sequence[int]  # H
    oof_queries: tuple[Sequence[int], ...]  # G blocks partition D


@dataclass(frozen=True)
class NestedPlan:
    plan_id: str
    source_plan_sha256: str
    outer_fit_rows: Sequence[int]
    outer_evaluation_rows: Sequence[int]
    diagnostic_splits: tuple[DiagnosticSplit, ...]
    final_oof_queries: tuple[Sequence[int], ...]


def _assert_partition(data: ConceptData, universe: np.ndarray,
                      pieces: Sequence[Sequence[int]], name: str) -> list[np.ndarray]:
    rows = [_rows(data, v, name) for v in pieces]
    combined = np.concatenate(rows) if rows else np.array([], dtype=int)
    if sorted(combined.tolist()) != universe.tolist():
        raise ValueError(f"{name} must be an exact, unique row partition")
    seen: set[str] = set()
    for part in rows:
        groups = set(_groups(data, part))
        if groups & seen:
            raise ValueError(f"{name} splits a patient/case group")
        seen |= groups
    return rows


def validate_nested_plan(data: ConceptData, plan: NestedPlan) -> dict[str, Any]:
    if len(plan.source_plan_sha256) != 64 or any(
            v not in "0123456789abcdef" for v in plan.source_plan_sha256):
        raise ValueError("externally audited source-plan SHA-256 is required")
    t = _rows(data, plan.outer_fit_rows, "T")
    e = _rows(data, plan.outer_evaluation_rows, "E")
    if set(_groups(data, t)) & set(_groups(data, e)):
        raise ValueError("outer T/E groups overlap")
    hs = _assert_partition(data, t, [s.validation_rows for s in plan.diagnostic_splits], "H")
    for split, h in zip(plan.diagnostic_splits, hs, strict=True):
        d = _rows(data, split.fit_rows, "D")
        if d.tolist() != np.setdiff1d(t, h).tolist():
            raise ValueError("diagnostic D must equal T minus H")
        _assert_partition(data, d, split.oof_queries, "G within D")
    final_queries = _assert_partition(data, t, plan.final_oof_queries, "final OOF within T")
    return dict(t=t, e=e, h=hs, final_queries=final_queries)


def _readout_matrix(blocks: Sequence[np.ndarray], domain: str) -> np.ndarray:
    if domain == "CBIS-DDSM":
        return np.column_stack([block[:, 1] for block in blocks])
    return np.concatenate(blocks, axis=1)


def build_u1_representative_inner(generator: U1Generator, plan: NestedPlan,
                                  *, inner_index: int = 0) -> dict[str, Any]:
    """One predeclared legal H unit for cost/support feasibility, NOT P3 results.

    This API never generates an outer-E prediction or final diagnostic panel.
    Index zero is a source-preparation default, not an outcome-dependent choice;
    the caller must freeze the selected outer/inner IDs before project execution.
    It performs real optimization only on the in-memory arrays supplied by that
    caller, so import/tests alone provide no authorization or project measurement.
    """
    d = generator.data
    checked = validate_nested_plan(d, plan)
    if not isinstance(inner_index, int) or not 0 <= inner_index < len(plan.diagnostic_splits):
        raise ValueError("representative inner index is outside the frozen plan")
    split = plan.diagnostic_splits[inner_index]
    f, h, e = _rows(d, split.fit_rows, "D"), checked["h"][inner_index], checked["e"]
    dimension = len(d.names) if generator.recipe.domain == "CBIS-DDSM" else sum(d.widths)
    nested = np.full((len(f), dimension), np.nan)
    start = len(generator.calls)
    for index, query in enumerate(split.oof_queries):
        g = _rows(d, query, "G")
        result = generator.fit_predict(np.setdiff1d(f, g), g,
            forbidden_rows=np.unique(np.concatenate((e, h, g))),
            context=f"{plan.plan_id}/representative/H{inner_index}/G{index}")
        nested[np.searchsorted(f, result["query_rows"])] = _readout_matrix(
            result["blocks"], generator.recipe.domain)
    validation = generator.fit_predict(f, h, forbidden_rows=np.union1d(e, h),
        context=f"{plan.plan_id}/representative/H{inner_index}/validation")
    if not np.isfinite(nested).all():
        raise AssertionError("representative inner fit panel has missing values")
    return dict(fit_rows=f, heldout_rows=h, xi=nested,
        zi=_readout_matrix(validation["blocks"], generator.recipe.domain),
        plan_id=plan.plan_id, source_plan_sha256=plan.source_plan_sha256,
        source_plan_content_sha256=_hash(asdict(plan)), inner_index=inner_index,
        lineage=generator.calls[start:], fit_cost=generator.fit_cost,
        prediction_cost=generator.prediction_cost, cost_summary=generator.cost_summary(),
        outer_predictions_generated=False, final_refit_generated=False,
        scientific_status="REPRESENTATIVE_COST_SUPPORT_ONLY_NOT_P3_COMPLETE")


def build_u1_nested_panel(generator: U1Generator, plan: NestedPlan) -> dict[str, Any]:
    """Generate real U1 nested scores from caller-supplied arrays/plan.

    This is an executable generator, not a fully wired/authorized P3 science
    orchestrator. No downstream y/C/control choice or diagnosis scoring occurs.
    Exact source-plan equality against U0 must be independently audited first.
    """
    d = generator.data
    checked = validate_nested_plan(d, plan)  # all structural checks before fits
    t, e = checked["t"], checked["e"]
    dimension = len(d.names) if generator.recipe.domain == "CBIS-DDSM" else sum(d.widths)
    selections = []
    call_start = len(generator.calls)
    for index, split in enumerate(plan.diagnostic_splits):
        f = _rows(d, split.fit_rows, "D")
        h = checked["h"][index]
        nested = np.full((len(f), dimension), np.nan)
        for g_index, query in enumerate(split.oof_queries):
            g = _rows(d, query, "G")
            fit = np.setdiff1d(f, g)
            forbidden = np.unique(np.concatenate((e, h, g)))
            result = generator.fit_predict(fit, g, forbidden_rows=forbidden,
                       context=f"{plan.plan_id}/H{index}/G{g_index}")
            nested[np.searchsorted(f, result["query_rows"])] = _readout_matrix(
                result["blocks"], generator.recipe.domain)
        h_result = generator.fit_predict(f, h, forbidden_rows=np.union1d(e, h),
                         context=f"{plan.plan_id}/H{index}/validation")
        if not np.isfinite(nested).all():
            raise AssertionError("nested diagnostic fit panel has missing values")
        selections.append(dict(fit_rows=f, heldout_rows=h, xi=nested,
            zi=_readout_matrix(h_result["blocks"], generator.recipe.domain)))
    # Explicit final OOF regeneration, with reuse ONLY for identical F+recipe.
    # Nothing is assembled from in-sample concept scores.
    raw = np.full((len(t), dimension), np.nan)
    for index, g in enumerate(checked["final_queries"]):
        result = generator.fit_predict(np.setdiff1d(t, g), g,
            forbidden_rows=np.union1d(e, g), context=f"{plan.plan_id}/final_OOF/G{index}")
        raw[np.searchsorted(t, result["query_rows"])] = _readout_matrix(result["blocks"], generator.recipe.domain)
    eval_result = generator.fit_predict(t, e, forbidden_rows=e, context=f"{plan.plan_id}/outer_evaluation")
    if not np.isfinite(raw).all():
        raise AssertionError("final OOF panel has missing values")
    return dict(t=t, e=e, sel=selections, raw=raw,
        val=_readout_matrix(eval_result["blocks"], generator.recipe.domain),
        plan_id=plan.plan_id, source_plan_sha256=plan.source_plan_sha256,
        source_plan_content_sha256=_hash(asdict(plan)),
        lineage=generator.calls[call_start:], fit_cost=generator.fit_cost,
        prediction_cost=generator.prediction_cost, cost_summary=generator.cost_summary(),
        provenance="P3 U1 nested generation only; no diagnosis outcomes selected/scored",
        scientific_status="SOURCE_CANDIDATE_NOT_P3_COMPLETE")


def concept_diagnostics(labels: np.ndarray, valid: np.ndarray,
                        blocks: Sequence[np.ndarray], names: Sequence[str]) -> dict[str, Any]:
    """Post-prediction descriptive diagnostics; not a fitting/selection API."""
    y, mask = np.asarray(labels), np.asarray(valid, dtype=bool)
    if y.shape != mask.shape or y.shape[1] != len(blocks) or len(names) != len(blocks):
        raise ValueError("diagnostic layout mismatch")
    reports, errors = [], []
    for column, (name, p) in enumerate(zip(names, blocks, strict=True)):
        use = mask[:, column]
        target, probability = y[use, column], np.asarray(p)[use]
        width = probability.shape[1]
        if (probability.ndim != 2 or len(probability) != len(target)
                or np.any(target < 0) or np.any(target >= width)
                or not np.isfinite(probability).all() or np.any(probability < 0)
                or not np.allclose(probability.sum(1), 1, rtol=0, atol=1e-10)):
            raise ValueError("invalid post-prediction diagnostic probabilities/labels")
        counts = np.bincount(target, minlength=width)
        complete_states = len(target) > 0 and np.all(counts > 0)
        ce = class_equal_nll(target, probability) if complete_states else None
        nll = (float(-np.log(np.clip(probability[np.arange(len(target)), target], 1e-15, 1)).mean())
               if len(target) else None)
        aucs = []
        for state in range(probability.shape[1]):
            indicator = (target == state).astype(int)
            aucs.append(float(roc_auc_score(indicator, probability[:, state]))
                        if len(np.unique(indicator)) == 2 else None)
        error = np.full(len(y), np.nan)
        error[use] = (np.argmax(probability, axis=1) != target).astype(float)
        errors.append(error)
        reports.append(dict(concept=name, natural_frequency_nll=nll, class_equal_nll=ce,
            macro_ovr_auc=float(np.mean(aucs)) if all(v is not None for v in aucs) else None,
            per_state_auc=aucs, state_counts=counts.tolist(),
            missing_states=np.flatnonzero(counts == 0).tolist(),
            score_quantiles_by_state=(np.quantile(probability, [0, .05, .25, .5, .75, .95, 1], axis=0).tolist()
                                      if len(target) else None),
            quantile_levels=[0, .05, .25, .5, .75, .95, 1], valid_count=int(use.sum())))
    correlations = [[None for _ in errors] for _ in errors]
    for i, a in enumerate(errors):
        for j, b in enumerate(errors):
            use = np.isfinite(a) & np.isfinite(b)
            if use.sum() >= 2 and np.std(a[use]) > 0 and np.std(b[use]) > 0:
                correlations[i][j] = float(np.corrcoef(a[use], b[use])[0, 1])
    def macro(key: str) -> float | None:
        values = [v[key] for v in reports]
        return float(np.mean(values)) if all(v is not None for v in values) else None
    return dict(per_concept=reports, error_correlation=correlations,
                macro_natural_frequency_nll=macro("natural_frequency_nll"),
                macro_class_equal_nll=macro("class_equal_nll"),
                macro_auc=macro("macro_ovr_auc"),
                undefined_policy="null full-layout macro if any required concept/state metric is undefined; no silent observed-state replacement",
                role="post-prediction descriptive only; no U1/control/route selection")


def upstream_fit_forecast(*, cbis_outer_units: int = 5, derm_outer_units: int = 25) -> dict[str, Any]:
    """Source-derived counts, contingent on frozen plans/support; no timing claims."""
    cbis = cbis_outer_units * 11 * 10 * 19
    derm = derm_outer_units * 21 * 7 * 19
    return dict(cbis_unique_bundles_per_outer=11, cbis_concept_blocks=10,
                derm_bundles_per_outer=21, derm_concept_blocks=7,
                fits_per_block_per_bundle=19, cbis_u1_fit_count=cbis,
                derm_u1_fit_count=derm, total_u1_fit_count=cbis + derm,
                timing_measured_on_project=False,
                caveat="plan/support and identical-fit recipe reuse must be verified before execution")
