#!/usr/bin/env python3
"""P3 development execution wrapper; source checks are the non-fitting default.

External root-issued approval is required BEFORE whitelisted arrays are opened.
The representative is exactly CBIS r0/f0/H0 (760 logical probe fits), NOT P3.
Full mode reuses P0-pinned U0 panels, generates U1, fits the frozen P2 tau=1
heads, S and one externally frozen P1 control, and reports paired uncertainty.
Neither mode can address an official test path or add U2/grid/seed/arm choices.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import time
import uuid
from typing import Any

THREADS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
INHERITED_THREADS = {k: os.environ.get(k) for k in THREADS}
for _name in THREADS:
    os.environ.setdefault(_name, "1")
import numpy as np
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import paper_b_cbis_p3 as cb
import paper_b_p3_u1 as u1
from paper2_evidence_handoff.data import CONCEPT_STATES, CONCEPT_VALUE_MAP, DIAGNOSIS_GROUPS
import run_neurocomputing_p2_scientific_v2_20260916 as p2
import run_p1_formal_unit_v5_20260916 as p1

VERSION = "P3_DEV_WRAPPER_SOURCE_V1_20260916"
PREP = ROOT / "paper_b_readiness/P3_PREP/EXECUTION_V1"
BINDING = PREP / "SOURCE_BINDING.json"
OUTPUT = ROOT / "results/neurocomputing_p3_dev_v1_20260916"
P0 = ROOT / "paper_b_readiness/P0_HASH_MANIFEST.json"
P0_SHA = "d00b9d304fd87d98319a1b85315c38accbb83f758abe97b62c02a2fdb91cfdcd"
DEV_ASSETS = {
    "cbis_cohort": "data/CBIS_DDSM/paper_b_cbis_mass_train_breast_cohort_v1_20260910.npz",
    "cbis_views": "results/paper_b_cbis_mass_train_biomedclip_full_mammogram_v2_20260910/view_embeddings.npz",
    "derm_features": "data/Derm7pt/derm7pt_biomedclip_dev609_only_v1_20260908.npz",
    "derm_metadata": "data/Derm7pt/derm7pt_dev609_only_v1_20260908.csv",
}
UNITS = tuple([f"cbis_r0_f{f}" for f in range(5)] + [f"derm_r{r}_f{f}" for r in range(5) for f in range(5)])
CORE = ("soft/lr", "logodds/lr", "soft/mlp", "logodds/mlp")
CONTROLS = ("C", "SP", "CP", "MP")

class P3Error(RuntimeError):
    pass

def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v.item()).encode()

def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()

def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def plain(path: Path, base: Path | None = None) -> Path:
    base = ROOT if base is None else base
    path = Path(os.path.abspath(path))
    try:
        parts = path.relative_to(base).parts
    except ValueError as exc:
        raise P3Error("bound path escapes its fixed root") from exc
    current = base
    for component in parts:
        if component in ("", ".", ".."):
            raise P3Error("unsafe path component")
        current /= component
        info = os.lstat(current)
        if stat.S_ISLNK(info.st_mode):
            raise P3Error("symlink in bound path ancestry")
    info = os.stat(path)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise P3Error("bound input is not a unique plain file")
    return path

def json_read(path: Path) -> Any:
    return json.loads(plain(path).read_text())

def write_json(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n"); stream.flush(); os.fsync(stream.fileno())

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({k for row in rows for k in row})
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (canonical(v).decode() if isinstance(v, (dict, list, tuple)) else v) for k, v in row.items()})
        stream.flush(); os.fsync(stream.fileno())

def source_check() -> dict[str, Any]:
    binding = json_read(BINDING)
    if binding["version"] != VERSION or binding["p0_sha256"] != P0_SHA:
        raise P3Error("source binding/version changed")
    if file_sha(plain(P0)) != P0_SHA:
        raise P3Error("P0 manifest changed")
    for name, expected in binding["source_files"].items():
        if file_sha(plain(ROOT / name)) != expected:
            raise P3Error(f"frozen source changed: {name}")
    return binding

def validate_gate(path: Path, mode: str, binding: dict[str, Any]) -> dict[str, Any]:
    gate = json_read(path)
    expected = {"contract", "mode", "source_binding_sha256", "source_audit", "root_authorization", "pre_outcome_frozen", "official_test_access_allowed", "ordered_units", "representative_inner_index", "control_decision", "cost_acceptance"}
    if set(gate) != expected or gate["contract"] != "P3_ROOT_EXECUTION_GATE_V1":
        raise P3Error("external gate schema changed")
    if gate["mode"] != mode or gate["source_binding_sha256"] != digest(binding):
        raise P3Error("gate scope/source mismatch")
    if gate["pre_outcome_frozen"] is not True or gate["official_test_access_allowed"] is not False:
        raise P3Error("pre-outcome/development-only firewall failed")
    for role in ("source_audit", "root_authorization"):
        record = gate[role]
        if set(record) != {"path", "sha256"} or file_sha(plain(ROOT / record["path"])) != record["sha256"]:
            raise P3Error(f"missing external {role} pin")
        receipt = json_read(ROOT / record["path"])
        if receipt.get("source_binding_sha256") != digest(binding):
            raise P3Error(f"external {role} pins another source")
        if role == "source_audit" and receipt.get("status") != "PASS":
            raise P3Error("source audit has not passed")
        if role == "root_authorization" and (receipt.get("authorized") is not True or receipt.get("mode") != mode):
            raise P3Error("root did not authorize this execution scope")
    if mode == "representative":
        if gate["ordered_units"] != ["cbis_r0_f0"] or gate["representative_inner_index"] != 0 or gate["control_decision"] is not None or gate["cost_acceptance"] is not None:
            raise P3Error("representative must halt after exactly one legal H0 generation")
    elif mode == "full-dev":
        if gate["ordered_units"] != list(UNITS) or gate["representative_inner_index"] is not None:
            raise P3Error("full-dev ordered scope changed")
        decision = gate["control_decision"]
        if set(decision) != {"path", "sha256"} or file_sha(plain(ROOT / decision["path"])) != decision["sha256"]:
            raise P3Error("missing frozen root P1 control decision")
        control = json_read(ROOT / decision["path"])
        if set(control) != {"contract", "control", "p1_accepted_evidence", "frozen_before_p3_outcomes", "same_control_both_upstreams"} or control["contract"] != "P3_SHARED_P1_CONTROL_V1" or control["control"] not in CONTROLS or control["frozen_before_p3_outcomes"] is not True or control["same_control_both_upstreams"] is not True:
            raise P3Error("P1 control is not a shared pre-outcome root decision")
        evidence = control["p1_accepted_evidence"]
        if set(evidence) != {"path", "sha256"} or file_sha(plain(ROOT / evidence["path"])) != evidence["sha256"]:
            raise P3Error("P1 accepted-evidence pin missing")
        acceptance = gate["cost_acceptance"]
        if set(acceptance) != {"representative_directory", "complete_sha256", "accepted", "source_binding_sha256"} or acceptance["accepted"] is not True or acceptance["source_binding_sha256"] != digest(binding):
            raise P3Error("full queue requires explicit acceptance of measured cost")
        rep_dir = ROOT / acceptance["representative_directory"]
        if rep_dir != OUTPUT / "representative_cbis_r0_f0_h0":
            raise P3Error("cost acceptance must bind the predeclared representative directory")
        complete = verify_complete(rep_dir, digest(binding))
        if file_sha(rep_dir / "COMPLETE.json") != acceptance["complete_sha256"] or complete["role"] != "representative" or complete["logical_fit_count"] != 760:
            raise P3Error("accepted representative cost receipt mismatch")
        gate = dict(gate, accepted_control=control["control"])
    else:
        raise P3Error("unknown gate scope")
    return gate

class DevInputs:
    """Exact P0 allowlist; other manifest entries are never opened/hashed."""
    def __init__(self):
        self.manifest = json_read(P0)
        self.records = dict(self.manifest["critical_assets"])
        self.records.update({r["path"]: r for r in self.manifest["frozen_result_files"]})
        self.opened: list[dict[str, Any]] = []

    def get(self, name: str, *, unit: str | None = None) -> Path:
        if name in DEV_ASSETS:
            relative = DEV_ASSETS[name]
        elif name in ("panel", "generation") and unit in UNITS:
            relative = f"results/deadline_nested_20260912/{unit}/{name}.{'npz' if name == 'panel' else 'json'}"
        else:
            raise P3Error("path is outside the fixed development whitelist")
        record = self.records[relative]
        path = plain(ROOT / relative)
        if path.stat().st_size != record["bytes"] or file_sha(path) != record["sha256"]:
            raise P3Error(f"P0-bound dev asset changed: {relative}")
        self.opened.append({"path": relative, "sha256": record["sha256"], "bytes": record["bytes"]})
        return path

def average_views(image_ids: np.ndarray, present: np.ndarray, ids: np.ndarray, embedding: np.ndarray) -> np.ndarray:
    ids = ids.astype(str)
    if len(set(ids)) != len(ids) or embedding.ndim != 2 or len(ids) != len(embedding) or not np.isfinite(embedding).all() or not np.allclose(np.linalg.norm(embedding, axis=1), 1, atol=5e-6, rtol=0):
        raise P3Error("invalid unique L2-normalized frozen view embeddings")
    lookup = {value: i for i, value in enumerate(ids)}
    output = []
    for row, mask in zip(image_ids.astype(str), present.astype(bool), strict=True):
        if not np.any(mask) or any(value not in lookup for value in row[mask]):
            raise P3Error("a present breast view lacks its pinned embedding")
        output.append(np.mean([embedding[lookup[value]] for value in row[mask]], axis=0))
    return np.asarray(output, dtype=np.float64)  # No breast-mean renormalization.

def load_domain(inputs: DevInputs, domain: str) -> dict[str, Any]:
    if domain == "cbis":
        co = cb.load_cohort(inputs.get("cbis_cohort"), cb.STABLE_CONCEPTS, complete_only=True)
        with np.load(inputs.get("cbis_views"), allow_pickle=False) as archive:
            if str(archive["checkpoint_sha256"].item()) != "52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be":
                raise P3Error("frozen BiomedCLIP checkpoint changed")
            x = average_views(co.image_id, co.view_present, archive["image_id"], archive["embeddings"])
        if x.shape != (655, 512):
            raise P3Error("CBIS primary feature/cohort shape changed")
        data = u1.ConceptData(x, co.unit_id.astype(str), co.patient_id.astype(str), co.concepts.astype(int), co.concept_valid.astype(bool), tuple(co.concept_names), (2,) * 10)
        return dict(data=data, y=co.y.astype(int), gold=co.concepts.astype(float), widths=np.array([], dtype=int), original_fold=co.outer_fold.astype(int))
    if domain != "derm":
        raise P3Error("only frozen CBIS and Derm development cohorts are permitted")
    import pandas as pd
    with np.load(inputs.get("derm_features"), allow_pickle=False) as archive:
        if str(archive["cohort_role"].item()) != "DERM7PT_OFFICIAL_TRAIN_VALID_DEV609_ONLY":
            raise P3Error("Derm official test or other cohort prohibited")
        indices, x = archive["metadata_index"].astype(int), archive["derm_embeddings"].astype(float)
    md = pd.read_csv(inputs.get("derm_metadata")).set_index("metadata_index").loc[indices]
    labels = np.column_stack([[states.index(CONCEPT_VALUE_MAP[name][str(v).strip().lower()]) for v in md[name]] for name, states in CONCEPT_STATES.items()]).astype(int)
    y = np.array([int(DIAGNOSIS_GROUPS[str(v).strip().lower()] == "MEL") for v in md.diagnosis])
    if len(y) != 609 or np.bincount(y).tolist() != [460, 149] or not np.isfinite(x).all() or np.any(np.linalg.norm(x, axis=1) <= 0):
        raise P3Error("Derm609 development cohort changed")
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    ids = np.array([f"derm7pt:{i:04d}" for i in indices])
    widths = np.array([len(states) for states in CONCEPT_STATES.values()], dtype=int)
    data = u1.ConceptData(x, ids, ids.copy(), labels, np.ones_like(labels, dtype=bool), tuple(CONCEPT_STATES), tuple(widths))
    gold = np.concatenate([np.eye(width)[labels[:, j]] for j, width in enumerate(widths)], axis=1)
    return dict(data=data, y=y, gold=gold, widths=widths, original_fold=None)

def make_plan(cohort: dict[str, Any], unit: str, generation_sha: str) -> u1.NestedPlan:
    identity = p2.unit_identity(unit)
    domain, rep, fold = identity["domain"], identity["repetition"], identity["outer_fold"]
    data = cohort["data"]
    if domain == "cbis":
        original = cohort["original_fold"]
        t, e = np.flatnonzero(original != fold), np.flatnonzero(original == fold)
        splits = []
        for heldout in sorted(set(original[t].tolist())):
            h = np.flatnonzero(original == heldout)
            d = np.setdiff1d(t, h)
            queries = tuple(np.flatnonzero(original == g) for g in sorted(set(original[d].tolist())))
            splits.append(u1.DiagnosticSplit(d, h, queries))
    else:
        t, e = p1.case_kfold_splits(len(data.ids), 5, u1._seed(2026090829, "outer", rep))[fold]
        splits = []
        for index, (a, b) in enumerate(p1.case_kfold_splits(len(t), 4, u1._seed(2026090829, "strict_meta", rep, fold))):
            d, h = t[a], t[b]
            queries = tuple(d[q] for _, q in p1.case_kfold_splits(len(d), 4, u1._seed(2026090829, "strict_meta_inner", rep, fold, index, "split")))
            splits.append(u1.DiagnosticSplit(d, h, queries))
    plan = u1.NestedPlan(unit, generation_sha, t, e, tuple(splits), tuple(s.validation_rows for s in splits))
    u1.validate_nested_plan(data, plan)
    return plan

def metadata_panel(cohort: dict[str, Any], plan: u1.NestedPlan) -> dict[str, np.ndarray]:
    data = cohort["data"]
    out = dict(y=cohort["y"], ids=data.ids, groups=data.groups, gold=cohort["gold"], widths=cohort["widths"], t=plan.outer_fit_rows, e=plan.outer_evaluation_rows)
    for i, split in enumerate(plan.diagnostic_splits):
        out[f"f{i}"], out[f"h{i}"] = split.fit_rows, split.validation_rows
    return out

def legal_plan(inputs: DevInputs, cohort: dict[str, Any], unit: str) -> tuple[u1.NestedPlan, dict[str, Any], dict[str, Any]]:
    path = inputs.get("generation", unit=unit)
    generation = json_read(path)
    plan = make_plan(cohort, unit, file_sha(path))
    record = {"generation_path": str(path.relative_to(ROOT)), "generation_sha256": file_sha(path), "panel_sha256": None}
    provenance = p2.lineage(metadata_panel(cohort, plan), generation, p2.unit_identity(unit)["domain"], record, unit=unit, cbis_source_fold=cohort["original_fold"])
    provenance["verification"] = "production reconstruction from P0-whitelisted frozen cohort/seed metadata; no test-only input injection"
    return plan, generation, provenance

def verify_complete(directory: Path, binding_sha: str) -> dict[str, Any]:
    receipt = json_read(directory / "COMPLETE.json")
    if receipt["source_binding_sha256"] != binding_sha:
        raise P3Error("completed bundle belongs to another frozen source")
    names = {p.name for p in directory.iterdir()}
    if names != set(receipt["payload"]) | {"COMPLETE.json"}:
        raise P3Error("complete bundle has missing/extra payload")
    for name, pin in receipt["payload"].items():
        path = plain(directory / name)
        if path.stat().st_size != pin["bytes"] or file_sha(path) != pin["sha256"]:
            raise P3Error("complete bundle payload changed")
    return receipt

@contextlib.contextmanager
def staged(final: Path, binding_sha: str, role: str, extra: dict[str, Any] | None = None):
    # Plain ancestry, exclusive lock and final directory prevent unsafe resume.
    current = ROOT
    try:
        parts = final.parent.relative_to(ROOT).parts
    except ValueError as exc:
        raise P3Error("result path escapes the fixed root") from exc
    for part in parts:
        current /= part
        if not current.exists() and not current.is_symlink():
            current.mkdir()
        if current.is_symlink() or not current.is_dir():
            raise P3Error("unsafe result ancestry")
    if final.exists() or final.is_symlink():
        raise P3Error("final bundle exists; only explicitly verified full-unit reuse is allowed")
    lock = final.with_name(final.name + ".lock")
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    stage = final.with_name(final.name + ".staging-" + uuid.uuid4().hex)
    stage.mkdir(mode=0o700)
    try:
        yield stage
        payload = {p.name: {"sha256": file_sha(plain(p)), "bytes": p.stat().st_size} for p in sorted(stage.iterdir())}
        receipt = dict(contract="P3_ATOMIC_COMPLETE_V1", role=role, source_binding_sha256=binding_sha, payload=payload, **(extra or {}))
        write_json(stage / "COMPLETE.json", receipt)
        verify_complete(stage, binding_sha)
        if final.exists():
            raise P3Error("final bundle appeared while lock held")
        os.rename(stage, final)
        directory_fd = os.open(final.parent, os.O_RDONLY)
        try: os.fsync(directory_fd)
        finally: os.close(directory_fd)
    except BaseException as exc:
        if stage.exists() and not (stage / "FAILED.json").exists():
            remaining = {p.name: {"sha256": file_sha(p), "bytes": p.stat().st_size} for p in stage.iterdir() if p.is_file() and not p.is_symlink()}
            write_json(stage / "FAILED.json", dict(contract="P3_QUARANTINED_FAILURE_V1", source_binding_sha256=binding_sha, exception_type=type(exc).__name__, payload=remaining, retry="new sibling staging only; no reuse of failed payload"))
        raise
    finally:
        os.close(fd)
        lock.unlink()

def new_generator(cohort: dict[str, Any], domain: str, inputs: DevInputs) -> u1.U1Generator:
    keys = ("cbis_cohort", "cbis_views") if domain == "cbis" else ("derm_features", "derm_metadata")
    feature_contract = {
        "domain": domain,
        "p0_assets": {DEV_ASSETS[k]: inputs.records[DEV_ASSETS[k]] for k in keys},
        "preprocessing": "arithmetic mean of present L2-normalized frozen views; no breast-mean renormalization" if domain == "cbis" else "L2 normalize frozen dermoscopic embedding per case",
    }
    recipe = u1.U1Recipe("CBIS-DDSM" if domain == "cbis" else "Derm7pt", 20260910 if domain == "cbis" else 2026090829, 10000 if domain == "cbis" else 2000, 10 if domain == "cbis" else 0, digest(feature_contract))
    return u1.U1Generator(cohort["data"], recipe)

def representative(inputs: DevInputs, binding: dict[str, Any], gate: dict[str, Any]) -> Path:
    final = OUTPUT / "representative_cbis_r0_f0_h0"
    started = time.perf_counter()
    with staged(final, digest(binding), "representative", {"logical_fit_count": 760}) as stage:
        cohort = load_domain(inputs, "cbis")
        plan, generation, provenance = legal_plan(inputs, cohort, "cbis_r0_f0")
        result = u1.build_u1_representative_inner(new_generator(cohort, "cbis", inputs), plan, inner_index=0)
        if result["cost_summary"]["fit_count"] != 760 or result["outer_predictions_generated"] or result["final_refit_generated"]:
            raise P3Error("representative fit/scope count changed")
        np.savez_compressed(stage / "representative_panel.npz", f=result["fit_rows"], h=result["heldout_rows"], x=result["xi"], z=result["zi"])
        for name, value in (("lineage.json", result["lineage"]), ("source_plan.json", asdict(plan)), ("u0_structural_ancestry.json", provenance), ("opened_dev_assets.json", inputs.opened), ("execution_gate.json", gate)):
            write_json(stage / name, value)
        write_csv(stage / "fit_cost.csv", result["fit_cost"])
        write_csv(stage / "prediction_cost.csv", result["prediction_cost"])
        write_json(stage / "cost_summary.json", dict(result["cost_summary"], wrapper_elapsed_seconds=time.perf_counter()-started, forecast=u1.upstream_fit_forecast(), forecast_note="source-derived logical count; representative does not establish full support/runtime, no automatic full launch"))
        write_json(stage / "status.json", {"status": "REPRESENTATIVE_COST_SUPPORT_ONLY_NOT_P3_COMPLETE", "outer_predictions_generated": False, "final_refit_generated": False})
    return final

def load_u0(inputs: DevInputs, cohort: dict[str, Any], plan: u1.NestedPlan, unit: str) -> dict[str, np.ndarray]:
    path = inputs.get("panel", unit=unit)
    with np.load(path, allow_pickle=False) as archive:
        panel = {k: archive[k].copy() for k in archive.files}
    for name, expected in metadata_panel(cohort, plan).items():
        if name not in panel or not np.array_equal(panel[name], expected):
            raise P3Error(f"P0 U0 metadata reconstruction mismatch: {unit}/{name}")
    if panel["raw"].shape[0] != len(panel["t"]) or panel["val"].shape[0] != len(panel["e"]):
        raise P3Error("U0 final score row shape changed")
    for i in range(4):
        if panel[f"x{i}"].shape[0] != len(panel[f"f{i}"]) or panel[f"z{i}"].shape[0] != len(panel[f"h{i}"]):
            raise P3Error("U0 nested score row shape changed")
    return panel

def reference_and_control(panel: dict[str, np.ndarray], plan: u1.NestedPlan, control: str, unit: str, upstream: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    domain = p2.unit_identity(unit)["domain"]
    y, gold, t, e = panel["y"], panel["gold"], panel["t"], panel["e"]
    widths = panel["widths"].tolist() or None
    valid = np.ones_like(gold, dtype=bool) if widths is None else None
    maximum, eps = (10000, 1e-6) if domain == "cbis" else (2000, 1e-12)
    counter = {"head_fits": 0, "head_seconds": 0., "head_by_role": {}, "affine_fits": 0, "affine_seconds": 0.}
    blocks = [(s.fit_rows, s.validation_rows, gold[s.fit_rows], gold[s.validation_rows]) for s in plan.diagnostic_splits]
    c, selection = p1.select_c(blocks, y, maximum, counter, "S/selection")
    head = p1.fit_head(gold[t], y[t], c, maximum, counter, "S/final")
    s_logit = head.logits(panel["val"])
    out = {"S": s_logit}
    records = {"S_selected_C": c, "S_selection": selection, "S_fitted_head": head.record(), "prior_lineage": [], "control": control}
    def restored(scores, fit, query, role):
        fit, query = np.asarray(fit), np.asarray(query)
        # Frozen P1 recipe uses T priors on final OOF inputs; all T precedes E.
        prior = []
        if widths is None:
            prior = [np.bincount(gold[fit, j].astype(int), minlength=2).tolist() for j in range(gold.shape[1])]
        else:
            start = 0
            for width in widths:
                prior.append(np.bincount(gold[fit, start:start+width].argmax(1), minlength=width).tolist()); start += width
        records["prior_lineage"].append(dict(role=role, fit_case_ids=panel["ids"][fit].tolist(), query_case_ids=panel["ids"][query].tolist(), state_counts=prior, final_oof_prior_rule="P1 training-T empirical prior, not claimed G-excluded prior for this training transform" if role == "final_OOF" else None))
        return p1.restore(scores, gold, fit, widths, eps, valid)
    if control in ("SP", "CP", "MP"):
        restored_val = restored(panel["val"], t, e, "T_predict_E")
        restored_raw = restored(panel["raw"], t, t, "final_OOF")
        sp_logit = head.logits(restored_val)
        if control in ("SP", "CP"):
            out[control] = sp_logit
    if control in ("C", "CP"):
        oof = np.full(len(t), np.nan)
        for i, split in enumerate(plan.diagnostic_splits):
            d, h = split.fit_rows, split.validation_rows
            nested = [(np.setdiff1d(d, g), g, gold[np.setdiff1d(d,g)], gold[g]) for g in split.oof_queries]
            ci, si = p1.select_c(nested, y, maximum, counter, f"{control}/reference/H{i}/selection")
            hi = p1.fit_head(gold[d], y[d], ci, maximum, counter, f"{control}/reference/H{i}/final")
            scores = panel[f"z{i}"] if control == "C" else restored(panel[f"z{i}"], d, h, f"H{i}_validation")
            oof[np.searchsorted(t, h)] = hi.logits(scores)
            records[f"reference_H{i}"] = dict(selection=si, head=hi.record(), fit_case_ids=panel["ids"][d].tolist(), validation_case_ids=panel["ids"][h].tolist())
        a, b, affine = p1.fit_affine(oof, y[t], counter, control)
        records["affine"] = dict(affine, fit_case_ids=panel["ids"][t].tolist(), oof_logits=oof.tolist())
        out[control] = a * (s_logit if control == "C" else sp_logit) + b
    elif control == "MP":
        matched = []
        for i, split in enumerate(plan.diagnostic_splits):
            d, h = split.fit_rows, split.validation_rows
            rx = np.full_like(panel[f"x{i}"], np.nan)
            for j, g in enumerate(split.oof_queries):
                pos = np.searchsorted(d, g)
                rx[pos] = restored(panel[f"x{i}"][pos], np.setdiff1d(d, g), g, f"H{i}/G{j}")
            rz = restored(panel[f"z{i}"], d, h, f"H{i}_validation")
            matched.append((d, h, rx, rz))
        cm, sm = p1.select_c(matched, y, maximum, counter, "MP/selection")
        hm = p1.fit_head(restored_raw, y[t], cm, maximum, counter, "MP/final")
        out[control] = hm.logits(restored_val)
        records["matched_control"] = dict(selection=sm, head=hm.record())
    predictions = []
    for arm, logits in out.items():
        for row, logit in zip(e, logits, strict=True):
            predictions.append(dict(p2.unit_identity(unit), upstream=upstream, arm=arm, refit_seed=0, case_id=str(panel["ids"][row]), patient_group=str(panel["groups"][row]), label=int(y[row]), logit=float(logit), probability=float(p1.sigmoid(np.array([logit]))[0])))
    records["cost"] = counter
    return predictions, records

def score_blocks(scores: np.ndarray, widths: np.ndarray) -> list[np.ndarray]:
    if not len(widths):
        return [np.column_stack((1-scores[:,j], scores[:,j])) for j in range(scores.shape[1])]
    ends = np.cumsum(widths)
    return list(np.split(scores, ends[:-1], axis=1))

def full_unit(inputs: DevInputs, cohort: dict[str, Any], unit: str, binding: dict[str, Any], gate: dict[str, Any]) -> Path:
    final = OUTPUT / "full-dev" / unit
    if final.exists():
        complete = verify_complete(final, digest(binding))
        if complete.get("execution_gate_sha256") != digest(gate) or complete.get("unit") != unit:
            raise P3Error("completed unit gate/control scope differs; unsafe resume refused")
        return final
    started = time.perf_counter()
    with staged(final, digest(binding), "full-dev-unit", {"unit": unit, "execution_gate_sha256": digest(gate)}) as stage:
        plan, generation, provenance = legal_plan(inputs, cohort, unit)
        p0_panel = load_u0(inputs, cohort, plan, unit)
        generated = u1.build_u1_nested_panel(new_generator(cohort, p2.unit_identity(unit)["domain"], inputs), plan)
        panel = metadata_panel(cohort, plan)
        panel.update(raw=generated["raw"], val=generated["val"])
        for i, split in enumerate(generated["sel"]):
            panel[f"x{i}"], panel[f"z{i}"] = split["xi"], split["zi"]
        np.savez_compressed(stage / "u1_panel.npz", **panel)
        predictions, selection, diagnostics, fitted, references, upstream_reports = [], [], [], [], {}, {}
        for upstream, current in (("U0", p0_panel), ("U1", panel)):
            ancestry = dict(provenance, upstream=upstream, source_record={"panel_sha256": file_sha(inputs.get("panel", unit=unit)) if upstream == "U0" else file_sha(stage / "u1_panel.npz")}, u1_recipe=asdict(new_generator(cohort, p2.unit_identity(unit)["domain"], inputs).recipe) if upstream == "U1" else None)
            for representation in ("soft", "logodds"):
                for head_name in ("lr", "mlp"):
                    result = p2.run_arm(current, unit, 1., representation, head_name, ancestry)
                    for row in result["predictions"]:
                        row.update(upstream=upstream, arm=f"{representation}/{head_name}")
                    predictions.extend(result["predictions"])
                    for key, target in (("selection",selection),("diagnostics",diagnostics),("fitted",fitted)):
                        target.extend(dict(r, upstream=upstream) for r in result[key])
            ref_predictions, record = reference_and_control(current, plan, gate["accepted_control"], unit, upstream)
            predictions.extend(ref_predictions); references[upstream] = record
            e = current["e"]
            upstream_reports[upstream] = u1.concept_diagnostics(cohort["data"].labels[e], cohort["data"].valid[e], score_blocks(current["val"], current["widths"]), cohort["data"].names)
        write_csv(stage / "predictions.csv", predictions)
        write_csv(stage / "selection.csv", selection)
        write_csv(stage / "downstream_fit_cost.csv", diagnostics)
        write_csv(stage / "upstream_fit_cost.csv", generated["fit_cost"])
        write_csv(stage / "upstream_prediction_cost.csv", generated["prediction_cost"])
        for name, value in (("fitted_heads.json",fitted),("reference_control_heads.json",references),("u1_lineage.json",generated["lineage"]),("u0_structural_lineage.json",provenance),("source_plan.json",asdict(plan)),("upstream_diagnostics.json",upstream_reports),("opened_dev_assets.json",inputs.opened),("execution_gate.json",gate)):
            write_json(stage/name,value)
        write_json(stage / "cost_summary.json", {"upstream":generated["cost_summary"], "core_downstream_fit_count":len(diagnostics), "reference_control":{u:r["cost"] for u,r in references.items()}, "wrapper_elapsed_seconds":time.perf_counter()-started, "not_full_factorial":True})
    return final

def prepare_estimators(rows: list[dict[str, Any]], domain: str, control: str) -> tuple[dict[Any, Any], np.ndarray, list[str], list[str]]:
    index, mapping = {}, {}
    reps = (0,) if domain == "cbis" else tuple(range(5))
    arm_seeds = [(arm,seed) for arm in CORE for seed in ((0,) if arm.endswith("/lr") else p2.SEEDS)] + [("S",0),(control,0)]
    for row in rows:
        case = str(row["case_id"])
        pair = (int(row["label"]), str(row["patient_group"]))
        if case in mapping and mapping[case] != pair:
            raise P3Error("case label/group changed")
        mapping[case] = pair
        key = (row["upstream"],row["arm"],int(row["refit_seed"]),int(row["repetition"]),case)
        if key in index:
            raise P3Error("duplicate outer prediction")
        index[key] = float(row["logit"])
    cases = sorted(mapping)
    expected = {(u,a,s,r,c) for u in ("U0","U1") for a,s in arm_seeds for r in reps for c in cases}
    if set(index) != expected:
        raise P3Error("incomplete upstream/arm/seed/repetition outer panel")
    labels = np.array([mapping[c][0] for c in cases])
    if set(labels.tolist()) != {0,1}:
        raise P3Error("paired domain lacks both diagnosis classes")
    estimators = {}
    for u in ("U0","U1"):
        for arm,seed in arm_seeds:
            logits = np.array([[index[u,arm,seed,r,c] for c in cases] for r in reps])
            if not np.isfinite(logits).all():
                raise P3Error("nonfinite outer logits")
            orders=[]
            for z in logits:
                order=np.argsort(z,kind="stable"); starts=np.r_[0,np.flatnonzero(np.diff(z[order])!=0)+1]; orders.append((order,starts))
            probability = p1.sigmoid(logits)
            estimators[u,arm,str(seed)] = {"loss":np.mean(np.logaddexp(0,logits)-labels*logits,axis=0),"brier":np.mean((probability-labels)**2,axis=0),"auc_orders":orders}
    return estimators,labels,cases,[mapping[c][1] if domain=="cbis" else c for c in cases]

def point_effects(values: dict[Any, dict[str,float]], control: str) -> dict[Any,float]:
    values = dict(values)
    for u in ("U0","U1"):
        for arm in ("soft/mlp","logodds/mlp"):
            values[u,arm,"seed_mean"]={m:float(np.mean([values[u,arm,str(s)][m] for s in p2.SEEDS])) for m in values[u,arm,str(p2.SEEDS[0])]}  # Mean METRICS, not an ensemble.
    effects={}
    for metric in ("class_equal_nll","natural_nll","auc","brier"):
        gain=lambda baseline,candidate: candidate[metric]-baseline[metric] if metric=="auc" else baseline[metric]-candidate[metric]
        for arm in (*CORE,"S",control):
            seeds=("0",) if not arm.endswith("/mlp") else tuple(map(str,p2.SEEDS))+("seed_mean",)
            for seed in seeds:
                effects[metric,"upstream_gain",arm,seed]=gain(values["U0",arm,seed],values["U1",arm,seed])
        for head in ("lr","mlp"):
            seeds=("0",) if head=="lr" else tuple(map(str,p2.SEEDS))+("seed_mean",)
            for seed in seeds:
                for u in ("U0","U1"):
                    effects[metric,f"{u}_soft_over_logodds",head,seed]=gain(values[u,f"logodds/{head}",seed],values[u,f"soft/{head}",seed])
                effects[metric,"upstream_x_representation",head,seed]=effects[metric,"U1_soft_over_logodds",head,seed]-effects[metric,"U0_soft_over_logodds",head,seed]
                for representation in ("soft","logodds"):
                    arm=f"{representation}/{head}"
                    effects[metric,"upstream_x_adaptation",arm,seed]=gain(values["U1","S","0"],values["U1",arm,seed])-gain(values["U0","S","0"],values["U0",arm,seed])
    return effects

def paired_ci(rows: list[dict[str,Any]], domain: str, control: str, *, replicates: int=10000, maximum: int=100000) -> dict[str,Any]:
    """Helper-only draw limits are for synthetics; production CLI has no override."""
    estimators,labels,cases,groups=prepare_estimators(rows,domain,control)
    clusters=sorted(set(groups)); lookup={g:i for i,g in enumerate(clusters)}; group_index=np.array([lookup[g] for g in groups])
    evaluate=lambda w:{k:p2.metrics(v,labels,w) for k,v in estimators.items()}
    arm_points=evaluate(np.ones(len(cases))); point=point_effects(arm_points,control); keys=list(point)
    rng=np.random.default_rng(2026091517); draws=[]; vectors=[]; proposal=0
    while len(vectors)<replicates and proposal<maximum:
        proposal+=1; multiplicity=np.bincount(rng.integers(len(clusters),size=len(clusters)),minlength=len(clusters)); weights=multiplicity[group_index]
        if any(weights[labels==k].sum()==0 for k in (0,1)):continue
        current=point_effects(evaluate(weights),control); vectors.append([current[k] for k in keys])
        draws.append(dict(accepted_index=len(vectors),proposal_index=proposal,cluster_multiplicity=multiplicity.tolist(),cluster_multiplicity_sha256=digest(multiplicity.tolist()),effects=[current[k] for k in keys]))
    if len(vectors)!=replicates:raise P3Error("insufficient valid shared bootstrap draws; no changed seed/cluster scheme")
    bounds=np.quantile(np.array(vectors),[.025,.975],axis=0,method="linear")
    summaries=[]
    for j,key in enumerate(keys):
        metric,contrast,arm,seed=key; lower,upper=map(float,bounds[:,j])
        summaries.append(dict(domain=domain,metric=metric,contrast=contrast,arm_or_head=arm,refit_seed_or_seed_mean=seed,estimate=point[key],ci_lower=lower,ci_upper=upper,bootstrap_replicates=replicates,bootstrap_seed=2026091517,interpretation="direction_supported_under_fitted_dev_protocol" if lower>0 else "opposite_direction_supported_under_fitted_dev_protocol" if upper<0 else "inconclusive_not_equivalence",direction="NLL_Brier_gain=baseline-candidate;AUC_gain=candidate-baseline;interactions=direct_difference_of_gains"))
    return dict(arm_metrics=[dict(upstream=k[0],arm=k[1],seed=k[2],**v) for k,v in arm_points.items()],paired_effects=summaries,draws=draws,estimand_order=[list(k) for k in keys],cluster_order=clusters,case_count=len(cases),cluster_count=len(clusters),proposals=proposal,scientific_status="P3_DEVELOPMENT_COMPARISON_NOT_UNTOUCHED_CONFIRMATION",inference="conditional on fitted panels; repetitions and seeds stay together; no probability ensemble")

def full_dev(inputs: DevInputs, binding: dict[str,Any], gate: dict[str,Any]) -> Path:
    cohorts={}
    for unit in UNITS:
        domain=p2.unit_identity(unit)["domain"]
        if domain not in cohorts:cohorts[domain]=load_domain(inputs,domain)
        full_unit(inputs,cohorts[domain],unit,binding,gate)
    final=OUTPUT/"full-dev-aggregate"
    with staged(final,digest(binding),"full-dev-aggregate",{"execution_gate_sha256":digest(gate),"ordered_units":list(UNITS)}) as stage:
        all_rows=[]; unit_pins={}
        for unit in UNITS:
            directory=OUTPUT/"full-dev"/unit; verify_complete(directory,digest(binding));unit_pins[unit]=file_sha(directory/"COMPLETE.json")
            with (directory/"predictions.csv").open(newline="") as stream:all_rows.extend(csv.DictReader(stream))
        summaries=[]
        for domain in ("cbis","derm"):
            report=paired_ci([r for r in all_rows if r["domain"]==domain],domain,gate["accepted_control"])
            draws=report.pop("draws");write_json(stage/f"{domain}_bootstrap_draws.json",draws);write_json(stage/f"{domain}_analysis.json",report);summaries.extend(report["paired_effects"])
        write_csv(stage/"paired_effects.csv",summaries);write_json(stage/"unit_complete_pins.json",unit_pins);write_json(stage/"execution_gate.json",gate)
        write_json(stage/"hypothesis_judgments.json",{"status":"P3_DEV_RESULTS_REQUIRE_ROOT_RESEARCH_POSITIONING_JUDGMENT","completion_is_not_positive_results":True,"all_effects_reported":True,"u2_started":False,"untouched_confirmation":False,"intervals_crossing_zero":"inconclusive, not equivalence","upstream_quality_diagnostics":"reported separately from downstream outcomes; no automatic U2/control rerouting"})
    return final

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group();modes.add_argument("--representative",action="store_true");modes.add_argument("--full-dev",action="store_true")
    parser.add_argument("--execute",action="store_true");parser.add_argument("--gate",type=Path)
    args=parser.parse_args();binding=source_check()
    if not args.execute:
        if args.gate is not None:raise P3Error("gate loading is reserved for explicit execution")
        print(json.dumps({"status":"SOURCE_ONLY_CHECK_PASS_NOT_EXECUTED","source_binding_sha256":digest(binding),"representative":"cbis_r0_f0/H0,760 logical fits","full_logical_probe_forecast":80275,"external_gates_required":True}));return
    if not (args.representative or args.full_dev) or args.gate is None:raise P3Error("explicit mode and external gate required")
    if any(v not in (None,"1") for v in INHERITED_THREADS.values()):raise P3Error("inherited thread configuration differs; not silently corrected by imports")
    p2.require_cpu_one();mode="representative" if args.representative else "full-dev";gate=validate_gate(args.gate,mode,binding)
    with threadpool_limits(limits=1):
        result=representative(DevInputs(),binding,gate) if mode=="representative" else full_dev(DevInputs(),binding,gate)
    print(json.dumps({"status":"COMPLETE","mode":mode,"directory":str(result),"not_official_test_confirmation":True}))

if __name__=="__main__":
    main()
