#!/usr/bin/env python3
"""Bounded P2 development-only orchestrator. Default mode cannot fit real data.

The reviewed 20260915 component is imported, not modified. The external approval
chain must precede panel loading/fitting in --execute. Helpers also accept purely
synthetic arrays for independent tests; no test can open a reserved dataset.
"""
from __future__ import annotations

import os
THREAD_NAMES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
for _name in THREAD_NAMES:
    os.environ.setdefault(_name, "1")

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import warnings
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_neurocomputing_p2_20260915 as reviewed

ROOT = reviewed.REPO
PREP = ROOT / "paper_b_readiness/P2_PREP/SCIENTIFIC_V2"
PROTOCOL = PREP / "P2_PROTOCOL.json"
BINDING = PREP / "SOURCE_BINDING.json"
OUTPUT = ROOT / "results/neurocomputing_p2_scientific_v2_20260916"
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
LAMBDA_GRID = (0.0001, 0.001, 0.01)
SEEDS = (17, 29, 43)
TEMPS = (1.0, 0.25)
REPRESENTATIVES = ("cbis_r0_f0", "derm_r0_f0")
CBIS_CONTROL = ROOT / "data/CBIS_DDSM/paper_b_cbis_mass_train_breast_cohort_v1_20260910.npz"
CBIS_CONTROL_SHA256 = "4d868f37a5daf2fa33f4b833d01da4c1476403b5d6fb157478451b9ce80f7725"
DERM_ARCHIVED_SEED = 2026090829
GateError = reviewed.ProtocolGateError


def digest(value: Any) -> str:
    return reviewed.canonical_json_sha256(value)


def ids_hash(values: Any) -> str:
    return hashlib.sha256("\n".join(sorted(str(x) for x in values)).encode()).hexdigest()


def json_read(path: Path) -> Any:
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise GateError(f"missing or unsafe bound file: {path.name}")
    return json.loads(path.read_text())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.parent.is_symlink():
        raise GateError("unsafe output path")
    stage = path.with_name(path.name + ".pending")
    if stage.is_symlink():
        raise GateError("unsafe staging path")
    with stage.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(stage, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({k for row in rows for k in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_name(path.name + ".pending")
    if path.is_symlink() or stage.is_symlink() or path.parent.is_symlink():
        raise GateError("unsafe csv output")
    with stage.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v for k, v in row.items()})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(stage, path)


def require_cpu_one() -> None:
    if any(os.environ.get(name) != "1" for name in THREAD_NAMES):
        raise GateError("all four numeric thread controls must equal1")
    if sys.platform != "linux":
        raise GateError("candidate reviewed only for Linux CPU")


def validate_binding() -> dict[str, Any]:
    require_cpu_one()
    binding = json_read(BINDING)
    if binding.get("status") != "CANDIDATE_NOT_FROZEN" or binding.get("scientific_execution_authorized") is not False:
        raise GateError("candidate binding must not self-authorize")
    mandatory = {
        Path(__file__).relative_to(ROOT).as_posix(),
        "tests/test_neurocomputing_p2_scientific_v2_20260916.py",
        "paper_b_readiness/P2_PREP/SCIENTIFIC_V2/P2_PROTOCOL.json",
        "paper_b_readiness/P2_PREP/SCIENTIFIC_V2/P2_PROTOCOL.md",
        "paper_b_readiness/P2_PREP/SCIENTIFIC_V2/P2_OUTPUT_CONTRACT.json",
        "paper_b_readiness/P2_PREP/SCIENTIFIC_V2/P2_AMENDMENT_LEDGER.json",
        "scripts/run_neurocomputing_p2_20260915.py",
        "paper_b_readiness/P2_PREP/P2_INDEPENDENT_RESULT_BLIND_AUDIT_20260916.json",
        "paper_b_readiness/P2_PREP/P2_INDEPENDENT_RESULT_BLIND_AUDIT_V2_20260916.json",
        "scripts/run_deadline_nested_20260912.py",
        "src/paper_b_cbis_p3.py",
        "src/paper2_mpjc/pilot.py",
        "scripts/run_p1_formal_unit_v5_20260916.py",
        reviewed.P0_MANIFEST.relative_to(ROOT).as_posix(),
    }
    if not mandatory.issubset(binding.get("files", {})):
        raise GateError("source binding omits required scientific/component evidence")
    for relative, expected in binding["files"].items():
        path = ROOT / relative
        if not path.resolve().is_relative_to(ROOT.resolve()) or path.is_symlink():
            raise GateError("unsafe source binding path")
        json_read(path) if path.suffix == ".json" else None
        if reviewed.sha256_file(path) != expected:
            raise GateError(f"bound source changed: {relative}")
    p = json_read(PROTOCOL)
    old = json_read(reviewed.DEFAULT_PROTOCOL)
    if p["lr"] != old["logistic_regression"]:
        raise GateError("LR recipe differs from reviewed component")
    mlp = dict(p["mlp"])
    if mlp.pop("dtype") != "float64" or mlp != {key: old["mlp"].get(key) for key in mlp}:
        raise GateError("MLP recipe differs from reviewed component")
    if p["execution"]["python_executable"] != str(Path(sys.executable).resolve()) or "LD_LIBRARY_PATH" in os.environ or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise GateError("fixed CPU runtime requires pinned Python, unset LD_LIBRARY_PATH and empty CUDA_VISIBLE_DEVICES")
    if p["status"] != "CANDIDATE_NOT_FROZEN" or p["scientific_execution_authorized"] is not False or p["official_test_access_authorized"] is not False:
        raise GateError("candidate protocol is not an authorization document")
    if p["source"] != "results/deadline_nested_20260912" or p["output"] != OUTPUT.relative_to(ROOT).as_posix():
        raise GateError("fixed dev source/output changed")
    if p["core"]["temperatures"] != list(TEMPS) or p["fit_counts"]["scientific_total"] != 5000:
        raise GateError("bounded core changed")
    if p.get("lineage") != {"cbis_control": CBIS_CONTROL.relative_to(ROOT).as_posix(), "cbis_control_sha256": CBIS_CONTROL_SHA256, "derm_archived_seed": DERM_ARCHIVED_SEED, "nested_G": "exact_producer_fit_validation_hashes_counts_and_partition", "cbis_final": "exact_manifest_projection_and_P0_T_E_patient_hashes", "derm_final": "archived_producer_T_E_scope_no_separate_final_manifest"}:
        raise GateError("bound producer-specific lineage recipe changed")
    ledger = json_read(PREP / "P2_AMENDMENT_LEDGER.json")
    if ledger.get("status") != "CANDIDATE_NOT_FROZEN" or ledger.get("result_blind") is not True or ledger.get("scientific_execution_authorized") is not False:
        raise GateError("amendment ledger is not a result-blind preparation record")
    for relative, sha in ledger["preserved_snapshot_sha256"].items():
        if binding["files"].get(relative) != sha:
            raise GateError("preserved reviewed snapshot changed")
    if p["reviewed_component_sha256"] != binding["files"]["scripts/run_neurocomputing_p2_20260915.py"] or p["reviewed_component_audit_sha256"] != binding["files"]["paper_b_readiness/P2_PREP/P2_INDEPENDENT_RESULT_BLIND_AUDIT_V2_20260916.json"]:
        raise GateError("protocol/component audit binding differs")
    current_environment = environment()
    if binding["environment"] != current_environment:
        raise GateError("numeric environment changed; requires new independent review")
    reviewed.validate_amendment_ledger()
    return binding


def environment() -> dict[str, Any]:
    return {"python": sys.version.split()[0], "python_executable": str(Path(sys.executable).resolve()), **{name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "torch", "threadpoolctl")}, "threads": {name: os.environ[name] for name in THREAD_NAMES}, "device": "cpu", "LD_LIBRARY_PATH": "unset" if "LD_LIBRARY_PATH" not in os.environ else "set", "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}


def require_execution_gate() -> dict[str, Any]:
    """Read only preparation approval documents; fail before loading real arrays."""
    binding = validate_binding()
    binding_sha = reviewed.sha256_file(BINDING)
    freeze_path = PREP / "P2_FREEZE.json"
    audit_path = PREP / "P2_INDEPENDENT_AUDIT.json"
    authorization_path = PREP / "P2_EXECUTION_AUTHORIZATION.json"
    freeze, audit, auth = (json_read(p) for p in (freeze_path, audit_path, authorization_path))
    for doc in (freeze, audit, auth):
        if doc.get("source_binding_sha256") != binding_sha or doc.get("scope") != "development_only_no_official_test_no_P1":
            raise GateError("approval document is not bound to this dev-only source")
        if doc.get("official_test_access_authorized") is not False:
            raise GateError("no official-test authority accepted by this executable")
    if freeze.get("status") != "FROZEN" or audit.get("status") != "PASS_FULL_SCIENTIFIC_ORCHESTRATOR":
        raise GateError("full-orchestrator freeze/audit required; component PASS insufficient")
    if audit.get("freeze_sha256") != reviewed.sha256_file(freeze_path):
        raise GateError("audit does not bind frozen candidate")
    if auth.get("freeze_sha256") != reviewed.sha256_file(freeze_path) or auth.get("audit_sha256") != reviewed.sha256_file(audit_path):
        raise GateError("authorization approval chain changed")
    if auth.get("scientific_execution_authorized") is not True or not auth.get("authorized_by"):
        raise GateError("explicit external development execution authorization required")
    return {"source_binding_sha256": binding_sha, "freeze_sha256": reviewed.sha256_file(freeze_path), "audit_sha256": reviewed.sha256_file(audit_path), "authorization_sha256": reviewed.sha256_file(authorization_path), "binding": binding}


def validate_partition(a: np.ndarray, b: np.ndarray, total: np.ndarray, name: str) -> None:
    for x in (a, b, total):
        if x.ndim != 1 or not np.issubdtype(x.dtype, np.integer) or len(set(map(int, x))) != len(x):
            raise GateError(f"invalid unique integer partition: {name}")
    if np.intersect1d(a, b).size or set(map(int, np.r_[a, b])) != set(map(int, total)):
        raise GateError(f"incomplete or overlapping partition: {name}")


def frozen_seed(base: int, *parts: object) -> int:
    """Exact archived case-only seed recipe; neither scores nor labels are inputs."""
    payload = "|".join([str(base), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little") & 0x7FFFFFFF


def derm_g_partitions(f: np.ndarray, repetition: int, fold: int, h: int) -> list[tuple[np.ndarray, np.ndarray]]:
    seed = frozen_seed(DERM_ARCHIVED_SEED, "strict_meta_inner", repetition, fold, h, "split")
    splitter = KFold(n_splits=4, shuffle=True, random_state=seed)
    return [(f[tr], f[va]) for tr, va in splitter.split(np.arange(len(f), dtype=np.int64))]


def load_cbis_fold_control(panel: dict[str, np.ndarray], outer_fold: int) -> np.ndarray:
    """Read only P0-bound development identity/eligibility/fold control fields."""
    if reviewed.sha256_file(reviewed.P0_MANIFEST) != reviewed.P0_MANIFEST_SHA256:
        raise GateError("P0 control manifest changed")
    record = json_read(reviewed.P0_MANIFEST).get("critical_assets", {}).get(CBIS_CONTROL.relative_to(ROOT).as_posix(), {})
    if record.get("sha256") != CBIS_CONTROL_SHA256 or reviewed.sha256_file(CBIS_CONTROL) != CBIS_CONTROL_SHA256 or CBIS_CONTROL.stat().st_size != record.get("bytes"):
        raise GateError("P0-bound CBIS development control changed")
    if CBIS_CONTROL.is_symlink() or CBIS_CONTROL.stat().st_nlink != 1:
        raise GateError("unsafe development control")
    with np.load(CBIS_CONTROL, allow_pickle=False) as source:
        required = {"complete_concept_eligible", "unit_id", "patient_id", "outer_fold"}
        if not required.issubset(source.files):
            raise GateError("development identity/fold control missing")
        eligible = source["complete_concept_eligible"].astype(bool)
        source_ids = source["unit_id"].astype(str)[eligible]
        groups = source["patient_id"].astype(str)[eligible]
        folds = source["outer_fold"].astype(int)[eligible]
    if not np.array_equal(source_ids, panel["ids"].astype(str)) or not np.array_equal(groups, panel["groups"].astype(str)):
        raise GateError("CBIS panel identity differs from P0-bound development control")
    if set(np.flatnonzero(folds != outer_fold)) != set(panel["t"]) or set(np.flatnonzero(folds == outer_fold)) != set(panel["e"]):
        raise GateError("CBIS outer partition differs from P0 development control")
    return folds


def lineage(panel: dict[str, np.ndarray], generation: dict[str, Any], domain: str, source_record: dict[str, Any], *, unit: str, cbis_source_fold: np.ndarray | None = None) -> dict[str, Any]:
    # cbis_source_fold is an explicit injection for pure synthetic tests only;
    # scientific/structural orchestration always reconstructs P0 control itself.
    identity = unit_identity(unit)
    if identity["domain"] != domain:
        raise GateError("unit/domain lineage mismatch")
    outer_fold = identity["outer_fold"]
    n = len(panel["y"])
    if set(map(int, np.unique(panel["y"]))) != {0, 1} or len(set(map(str, panel["ids"]))) != n:
        raise GateError("invalid binary case mapping")
    validate_partition(panel["t"], panel["e"], np.arange(n), "outer")
    if domain == "cbis" and set(map(str, panel["groups"][panel["t"]])).intersection(map(str, panel["groups"][panel["e"]])):
        raise GateError("CBIS outer patient overlap")
    source_fold = None
    if domain == "cbis":
        source_fold = load_cbis_fold_control(panel, outer_fold) if cbis_source_fold is None else np.asarray(cbis_source_fold)
        if source_fold.shape != (n,) or not np.issubdtype(source_fold.dtype, np.integer) or set(np.unique(source_fold)) != set(range(5)):
            raise GateError("invalid CBIS original-fold control")
        if set(np.flatnonzero(source_fold != outer_fold)) != set(panel["t"]) or set(np.flatnonzero(source_fold == outer_fold)) != set(panel["e"]):
            raise GateError("CBIS control does not reproduce outer rows")
    selections = generation.get("selection_manifests")
    if not isinstance(selections, list) or len(selections) != 4:
        raise GateError("missing complete-nested generation manifests")
    inner = []
    for i in range(4):
        f, h = panel[f"f{i}"], panel[f"h{i}"]
        validate_partition(f, h, panel["t"], f"inner{i}")
        if domain == "cbis":
            if set(map(str, panel["groups"][f])).intersection(map(str, panel["groups"][h])):
                raise GateError("CBIS inner patient overlap")
            m = selections[i]
            if not isinstance(m, list) or len(m) != 1:
                raise GateError("CBIS selection manifest wrapper changed")
            m = m[0]
            heldout_fold = m.get("diagnosis_selection_heldout_fold")
            if m.get("outer_fold") != outer_fold or heldout_fold not in set(range(5)) - {outer_fold} or set(np.flatnonzero(source_fold == heldout_fold)) != set(h) or m.get("fit_folds") != sorted(map(int, np.unique(source_fold[f]))):
                raise GateError("CBIS H scope does not reproduce original patient folds")
            if m.get("fit_patient_sha256") != ids_hash(panel["groups"][f]) or m.get("heldout_patient_sha256") != ids_hash(panel["groups"][h]) or m.get("fit_heldout_patient_disjoint") is not True:
                raise GateError("CBIS generation lineage does not match inner rows")
            nested = m.get("nested_probe_panels", [])
            if len(nested) != 3 or any(q.get("diagnosis_validation_excluded_from_probe_fit") is not True or q.get("probe_validation_excluded_from_probe_fit") is not True for q in nested):
                raise GateError("CBIS upstream fitting did not exclude inner validation")
            g_partitions = []
            for q in nested:
                g_fold = q.get("probe_heldout_fold")
                if g_fold not in set(map(int, np.unique(source_fold[f]))):
                    raise GateError("CBIS G fold is outside F")
                g = np.flatnonzero(source_fold == g_fold)
                fit = np.setdiff1d(f, g)
                validate_partition(fit, g, f, "CBIS G")
                if q.get("diagnosis_selection_heldout_fold") != heldout_fold or q.get("probe_fit_folds") != sorted(map(int, np.unique(source_fold[fit]))) or q.get("probe_fit_patient_sha256") != ids_hash(panel["groups"][fit]) or q.get("probe_validation_patient_sha256") != ids_hash(panel["groups"][g]):
                    raise GateError("CBIS G exact fit/validation lineage changed")
                if set(map(str, panel["groups"][fit])).intersection(map(str, panel["groups"][g])):
                    raise GateError("CBIS G patient overlap")
                g_partitions.append((fit, g))
        else:
            m = selections[i]
            if list(map(str, m.get("fit_case_ids", []))) != list(map(str, panel["ids"][f])) or list(map(str, m.get("heldout_case_ids", []))) != list(map(str, panel["ids"][h])):
                raise GateError("Derm generation lineage does not match inner rows")
            nested = m.get("inner", [])
            if len(nested) != 4 or any(q.get("fit_validation_disjoint") is not True or q.get("split_inputs") != "case_indices_only_no_C_no_Y" or q.get("fit_count", 0) + q.get("validation_count", 0) != len(f) for q in nested):
                raise GateError("Derm nested training-score exclusion absent")
            g_partitions = derm_g_partitions(f, identity["repetition"], outer_fold, i)
            for g_index, (q, (fit, g)) in enumerate(zip(nested, g_partitions)):
                validate_partition(fit, g, f, "Derm G")
                if q.get("inner_fold") != g_index or q.get("fit_count") != len(fit) or q.get("validation_count") != len(g) or q.get("fit_case_id_sha256") != ids_hash(panel["ids"][fit]) or q.get("validation_case_id_sha256") != ids_hash(panel["ids"][g]):
                    raise GateError("Derm G exact deterministic fit/validation lineage changed")
        if not np.array_equal(np.sort(np.concatenate([g for _, g in g_partitions])), np.sort(f)):
            raise GateError("nested G validation must partition F exactly once")
        inner.append({"fold": i, "fit_case_id_sha256": ids_hash(panel["ids"][f]), "validation_case_id_sha256": ids_hash(panel["ids"][h]), "fit_indices_sha256": digest(f.tolist()), "validation_indices_sha256": digest(h.tolist()), "upstream_inner_validation_excluded": True, "generation_manifest_sha256": digest(m), "nested_G": [{"fit_case_id_sha256": ids_hash(panel["ids"][fit]), "validation_case_id_sha256": ids_hash(panel["ids"][g]), "fit_count": len(fit), "validation_count": len(g)} for fit, g in g_partitions]})
    if not np.array_equal(np.sort(np.concatenate([panel[f"h{i}"] for i in range(4)])), np.sort(panel["t"])):
        raise GateError("inner validation rows must partition outer training")
    final_scope = {"verification": "Derm archived producer final T-to-E scope; no separate final manifest was emitted", "new_manifest_required": False}
    if domain == "cbis":
        manifests = generation.get("manifests")
        resolved = [block[0] for block in selections]
        if not isinstance(manifests, list) or len(manifests) != 5 or any(sum(item == selected for item in manifests[:4]) != 1 for selected in resolved):
            raise GateError("CBIS selection-manifest projection changed")
        final = manifests[4]
        if not isinstance(final, dict) or final.get("role") != "final_outer_train_probe" or final.get("outer_fold") != outer_fold or final.get("nested_probe_panels") not in (None, []) or final.get("fit_patient_sha256") != ids_hash(panel["groups"][panel["t"]]) or final.get("validation_patient_sha256") != ids_hash(panel["groups"][panel["e"]]) or final.get("patient_disjoint") is not True:
            raise GateError("CBIS final T-to-E generation scope changed")
        final_scope = {"generation_manifest_sha256": digest(final), "verification": "P0-bound original patient folds and exact final T-to-E patient hashes", "control_sha256": CBIS_CONTROL_SHA256}
    return {"source_record": source_record, "generation_sha256": digest(generation), "outer_fit_case_id_sha256": ids_hash(panel["ids"][panel["t"]]), "outer_evaluation_case_id_sha256": ids_hash(panel["ids"][panel["e"]]), "inner": inner, "final_generation_scope": final_scope, "upstream": "archived_complete_nested_no_new_fits", "official_test_accessed": False, "p1_outputs_read": False}


def coordinates(scores: np.ndarray, domain: str, widths: np.ndarray, tau: float, representation: str) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 2 or not np.all(np.isfinite(s)) or np.any(s < 0) or np.any(s > 1) or tau not in (1.0, 0.25, 4.0):
        raise GateError("invalid frozen scores/temperature")
    eps = 1e-6 if domain == "cbis" else 1e-12
    if domain == "cbis":
        z, clipped = reviewed.clipped_logit(s, eps)
        if representation == "logodds":
            return z / tau, clipped
        if representation == "soft":
            return (s.copy() if tau == 1.0 else reviewed.stable_sigmoid(z / tau)), clipped
    else:
        sizes = reviewed.validate_block_widths(s, widths)
        z = reviewed.reference_log_odds(s, sizes, eps)
        clipped = s < eps
        if representation == "logodds":
            return z / tau, clipped
        if representation == "soft":
            if tau == 1.0:
                return s.copy(), clipped
            return reviewed.block_temperature(s, sizes, tau, eps)[0], clipped
    raise GateError("only separate soft/logodds pipelines are supported")


def scale_data(training: np.ndarray, validation: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    scaler = reviewed.Standardizer.fit(training)
    record = {"mean": scaler.mean.tolist(), "scale": scaler.scale.tolist(), "fit_count": len(training), "zero_variance_coordinate_count": int(np.sum(np.std(training, axis=0) <= 1e-12)), "fit_array_sha256": reviewed._array_content_record(training)["content_sha256"]}
    return scaler.transform(training), scaler.transform(validation), record


class TorchHead:
    def __init__(self, net: Any):
        self.net = net

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            return self.net(torch.as_tensor(x, dtype=torch.float64)).flatten().numpy()


def fit_head(x: np.ndarray, y: np.ndarray, head: str, hp: float, seed: int, fit_id: str, stage: str, tau: float) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    if set(map(int, np.unique(y))) != {0, 1} or not np.all(np.isfinite(x)):
        raise GateError("finite training features and both labels required")
    if head == "lr":
        model, diagnostic = reviewed._fit_synthetic_logistic_regression(x, y, hp, fit_id=fit_id, stage=stage, base_c=hp, temperature=tau, mapped_c=hp)
        diagnostic.update({"head": "lr", "solver": diagnostic["reported_solver"], "optimization_status": "converged", "epochs_completed": 0})
        return model, diagnostic, {"head": "lr", "coefficient": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]), "C": hp, "seed": seed}
    if head != "mlp":
        raise GateError("unknown diagnostic head")
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 32), torch.nn.GELU(), torch.nn.Linear(32, 1)).double()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    tx, ty = torch.as_tensor(x, dtype=torch.float64), torch.as_tensor(y, dtype=torch.float64)
    weights = [p for name, p in net.named_parameters() if name.endswith("weight")]
    first, last = None, None
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for epoch in range(200):
            optimizer.zero_grad()
            logits = net(tx).flatten()
            loss_i = torch.nn.functional.binary_cross_entropy_with_logits(logits, ty, reduction="none")
            loss = 0.5 * (loss_i[ty == 0].mean() + loss_i[ty == 1].mean()) + hp / 2 * sum((p * p).sum() for p in weights)
            if not torch.isfinite(loss):
                raise GateError(f"MLP nonfinite objective: {fit_id}")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in net.parameters()):
                raise GateError(f"MLP nonfinite gradient: {fit_id}")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in net.parameters()):
                raise GateError(f"MLP nonfinite weights: {fit_id}")
            if first is None:
                first = float(loss.detach())
            last = float(loss.detach())
    if caught:
        raise GateError(f"MLP warning emitted: {fit_id}: {caught[0].message}")
    model = TorchHead(net.eval())
    if not np.all(np.isfinite(model.decision_function(x))):
        raise GateError("MLP nonfinite training outputs")
    record = {"head": "mlp", "seed": seed, "lambda": hp, "state_dict": {k: v.detach().numpy().tolist() for k, v in net.state_dict().items()}}
    diagnostic = {"fit_id": fit_id, "stage": stage, "head": "mlp", "temperature": tau, "lambda": hp, "seed": seed, "optimizer": "Adam", "learning_rate": 0.001, "batching": "full_batch", "epochs_completed": 200, "optimization_status": "fixed_epoch_numerically_complete", "gradient_finite": True, "parameters_finite": True, "warning_count": 0, "weights_only_penalty": True, "first_objective": first, "last_objective": last, "elapsed_seconds": time.perf_counter() - started}
    return model, diagnostic, record


def unit_identity(unit: str) -> dict[str, Any]:
    domain, rep, fold = unit.split("_")
    return {"domain": domain, "repetition": int(rep[1:]), "outer_fold": int(fold[1:]), "outer_unit": unit}


def run_arm(panel: dict[str, np.ndarray], unit: str, tau: float, rep: str, head: str, provenance: dict[str, Any]) -> dict[str, Any]:
    identity = unit_identity(unit)
    domain = identity["domain"]
    grid = C_GRID if head == "lr" else LAMBDA_GRID
    selection, diagnostics, fold_losses = [], [], {hp: [] for hp in grid}
    standardized_panels = []
    for i in range(4):
        a, _ = coordinates(panel[f"x{i}"], domain, panel["widths"], tau, rep)
        b, _ = coordinates(panel[f"z{i}"], domain, panel["widths"], tau, rep)
        x, z, scaler = scale_data(a, b)
        standardized_panels.append({"train": x, "validation": z})
        for hp in grid:
            model, diag, _ = fit_head(x, panel["y"][panel[f"f{i}"]], head, hp, 0 if head == "lr" else 17, f"{unit}/{tau}/{rep}/{head}/inner{i}/{hp}", "inner_selection", tau)
            logits = model.decision_function(z)
            loss = reviewed.class_equal_binary_cross_entropy(panel["y"][panel[f"h{i}"]], logits)
            if not np.isfinite(loss):
                raise GateError("nonfinite validation objective")
            fold_losses[hp].append(loss)
            diagnostics.append(diag)
            selection.append({**identity, "temperature": tau, "representation": rep, "head": head, "candidate_hyperparameter": hp, "inner_fold": i, "selection_seed": 0 if head == "lr" else 17, "class_equal_nll": loss, "selected": False, "tie_rule": "smallest_C" if head == "lr" else "largest_lambda", "optimization_status": diag["optimization_status"], "epochs_completed": diag["epochs_completed"], "standardizer_fit_case_count": len(x), "zero_variance_coordinate_count": scaler["zero_variance_coordinate_count"], "scaler_sha256": digest(scaler), "fit_case_id_sha256": ids_hash(panel["ids"][panel[f"f{i}"]]), "validation_case_id_sha256": ids_hash(panel["ids"][panel[f"h{i}"]])})
    means = {hp: float(np.mean(v)) for hp, v in fold_losses.items()}
    selected = min(grid, key=lambda hp: (means[hp], hp if head == "lr" else -hp))
    for row in selection:
        row["selected"] = row["candidate_hyperparameter"] == selected
        row["candidate_mean_nll"] = means[row["candidate_hyperparameter"]]
    a, _ = coordinates(panel["raw"], domain, panel["widths"], tau, rep)
    b, clipped = coordinates(panel["val"], domain, panel["widths"], tau, rep)
    x, z, scaler = scale_data(a, b)
    predictions, fitted = [], []
    for seed in ((0,) if head == "lr" else SEEDS):
        model, diag, record = fit_head(x, panel["y"][panel["t"]], head, selected, seed, f"{unit}/{tau}/{rep}/{head}/refit/{seed}", "outer_training_refit", tau)
        diagnostics.append(diag)
        record.update({"scaler": scaler, "representation": rep, "temperature": tau, "training_case_id_sha256": ids_hash(panel["ids"][panel["t"]]), "lineage_sha256": digest(provenance)})
        fitted.append(record)
        logits = model.decision_function(z)
        probs = reviewed.stable_sigmoid(logits)
        if not np.all(np.isfinite(logits)) or not np.all(np.isfinite(probs)):
            raise GateError("nonfinite outer development predictions")
        for local, global_row in enumerate(panel["e"]):
            predictions.append({**identity, "case_id": str(panel["ids"][global_row]), "patient_group": str(panel["groups"][global_row]), "label": int(panel["y"][global_row]), "representation": rep, "temperature": tau, "head": head, "selected_hyperparameter": selected, "refit_seed": seed, "probability": float(probs[local]), "logit": float(logits[local]), "was_clipped": bool(np.any(clipped[local])), "probability_clip_epsilon": 1e-6 if domain == "cbis" else 1e-12, "reference_category_rule": "binary_logit" if domain == "cbis" else "first_category_each_block", "panel_sha256": provenance.get("source_record", {}).get("panel_sha256", "synthetic"), "lineage_sha256": digest(provenance), "scaler_sha256": digest(scaler), "head_sha256": digest(record)})
    return {"predictions": predictions, "selection": selection, "diagnostics": diagnostics, "fitted": fitted, "selected": selected, "losses": means, "inner_arrays": standardized_panels, "final_arrays": {"train": x, "validation": z}}


def comparison(y: np.ndarray, a: np.ndarray, b: np.ndarray, name: str) -> dict[str, Any]:
    metrics = {"max_abs_logit_difference": float(np.max(np.abs(a - b))), "max_abs_prediction_difference": float(np.max(np.abs(reviewed.stable_sigmoid(a) - reviewed.stable_sigmoid(b)))), "max_abs_class_equal_nll_difference": abs(reviewed.class_equal_binary_cross_entropy(y, a) - reviewed.class_equal_binary_cross_entropy(y, b))}
    if not all(np.isfinite(v) and v < 1e-5 for v in metrics.values()):
        raise GateError(f"strict numerical equivalence failed: {name}: {metrics}")
    return {"comparison": name, **metrics, "strict_tolerance": 1e-5, "pass": True}


def exact_controls(panel: dict[str, np.ndarray], unit: str, arms: dict[tuple[float, str, str], dict[str, Any]], provenance: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    checks, diagnostics, extra_fitted = [], [], []
    domain = unit_identity(unit)["domain"]
    for i in range(4):
        x, _ = coordinates(panel[f"x{i}"], domain, panel["widths"], 1.0, "logodds")
        z, _ = coordinates(panel[f"z{i}"], domain, panel["widths"], 1.0, "logodds")
        y, v = panel["y"][panel[f"f{i}"]], panel["y"][panel[f"h{i}"]]
        for c in C_GRID:
            base, diag, _ = fit_head(x, y, "lr", c, 0, f"{unit}/exact/inner{i}/base/{c}", "raw_exact_reference", 1.0)
            diagnostics.append(diag)
            for tau in (0.25, 4.0):
                mapped_c = c * tau * tau
                model, diag, _ = fit_head(x / tau, y, "lr", mapped_c, 0, f"{unit}/exact/inner{i}/{tau}/{c}", "raw_exact_refit", tau)
                diag.update({"base_C": c, "mapped_C": mapped_c})
                diagnostics.append(diag)
                ref = base.decision_function(z)
                analytic = (z / tau) @ (base.coef_[0] * tau) + base.intercept_[0]
                checks.append(comparison(v, ref, analytic, f"inner{i}/C{c}/tau{tau}/analytic"))
                checks.append(comparison(v, ref, model.decision_function(z / tau), f"inner{i}/C{c}/tau{tau}/fixed_refit"))
    tau4 = run_arm(panel, unit, 4.0, "logodds", "lr", provenance)
    diagnostics.extend(tau4["diagnostics"])
    extra_fitted.extend(tau4["fitted"])
    base = arms[(1.0, "logodds", "lr")]
    for tau, current in ((0.25, arms[(0.25, "logodds", "lr")]), (4.0, tau4)):
        for kind in ("train", "validation"):
            delta = float(np.max(np.abs(base["final_arrays"][kind] - current["final_arrays"][kind])))
            if not np.isfinite(delta) or delta >= 1e-5:
                raise GateError("final training-local standardized coordinates differ")
            checks.append({"comparison": f"standardized_tau{tau}/final/{kind}", "max_abs_coordinate_difference": delta, "strict_tolerance": 1e-5, "pass": True})
        for i in range(4):
            for kind in ("train", "validation"):
                delta = float(np.max(np.abs(base["inner_arrays"][i][kind] - current["inner_arrays"][i][kind])))
                if not np.isfinite(delta) or delta >= 1e-5:
                    raise GateError("standardized coordinate equivalence failed")
                checks.append({"comparison": f"standardized_tau{tau}/inner{i}/{kind}", "max_abs_coordinate_difference": delta, "pass": True})
        for c in C_GRID:
            for i in range(4):
                loss_a = next(r["class_equal_nll"] for r in base["selection"] if r["inner_fold"] == i and r["candidate_hyperparameter"] == c)
                loss_b = next(r["class_equal_nll"] for r in current["selection"] if r["inner_fold"] == i and r["candidate_hyperparameter"] == c)
                if not np.isfinite(loss_a - loss_b) or abs(loss_a - loss_b) >= 1e-5:
                    raise GateError("standardized same-C validation equivalence failed")
                checks.append({"comparison": f"standardized_tau{tau}/inner{i}/C{c}/NLL", "max_abs_class_equal_nll_difference": abs(loss_a - loss_b), "pass": True})
        if base["selected"] != current["selected"]:
            raise GateError("standardized selected C differs; no forced tie substitution")
        a = np.array([r["logit"] for r in base["predictions"]])
        b = np.array([r["logit"] for r in current["predictions"]])
        checks.append(comparison(panel["y"][panel["e"]], a, b, f"standardized_tau{tau}/selected_head"))
    return {"unit": unit, "strict_tolerance": 1e-5, "controls_fit_count": len(diagnostics), "comparisons": checks, "tau4_selection": tau4["selection"], "tau4_predictions": tau4["predictions"], "pass": True}, diagnostics, extra_fitted


def affine_diagnostics(panel: dict[str, np.ndarray], unit: str, tau: float, arms: dict[tuple[float, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for i in range(5):
        train_key, eval_key = (f"x{i}", f"z{i}") if i < 4 else ("raw", "val")
        s, _ = coordinates(panel[train_key], "cbis", panel["widths"], tau, "soft")
        l, _ = coordinates(panel[train_key], "cbis", panel["widths"], tau, "logodds")
        se, _ = coordinates(panel[eval_key], "cbis", panel["widths"], tau, "soft")
        le, _ = coordinates(panel[eval_key], "cbis", panel["widths"], tau, "logodds")
        proxy = reviewed.DiagonalAffineProxy.fit(s, l)
        residual = le - proxy.predict(se)
        rows.append({"unit": unit, "temperature": tau, "panel": i, "fit_case_count": len(s), "fit_scope": "unlabelled_training_only", "slope": proxy.slope.tolist(), "intercept": proxy.intercept.tolist(), "residual_rmse": float(np.sqrt(np.mean(residual ** 2))), "residual_max_abs": float(np.max(np.abs(residual))), "residual_case_norm_mean": float(np.mean(np.linalg.norm(residual, axis=1)))})
        if i == 4:
            head = arms[(tau, "logodds", "lr")]["fitted"][0]
            scaler = head["scaler"]
            coef = np.array(head["coefficient"]) / np.array(scaler["scale"])
            intercept = head["intercept"] - coef @ np.array(scaler["mean"])
            exact = le @ coef + intercept
            approximate = proxy.predict(se) @ coef + intercept
            bound = reviewed.affine_bound_check(le, proxy.predict(se), coef)
            if not bound["holds"]:
                raise GateError("affine Cauchy-Schwarz bound failed")
            soft = np.array([r["logit"] for r in arms[(tau, "soft", "lr")]["predictions"]])
            y = panel["y"][panel["e"]]
            L = lambda q: reviewed.class_equal_binary_cross_entropy(y, q)
            decomposition = (L(exact) - L(approximate)) + (L(approximate) - L(soft))
            if abs(decomposition - (L(exact) - L(soft))) >= 1e-12:
                raise GateError("affine loss identity failed")
            for j, global_row in enumerate(panel["e"]):
                rows.append({"unit": unit, "temperature": tau, "panel": "final_case", "case_id": str(panel["ids"][global_row]), "fixed_head_logit": float(exact[j]), "affine_fixed_head_logit": float(approximate[j]), "logit_abs_difference": float(bound["logit_difference"][j]), "norm_bound": float(bound["bound"][j]), "sigmoid_lipschitz_bound": float(bound["bound"][j] / 4), "head_refit": False, "label_fit": False, "L_log_minus_L_affine": L(exact) - L(approximate), "L_affine_minus_L_soft_refit": L(approximate) - L(soft), "L_log_minus_L_soft_refit": L(exact) - L(soft), "interpretation": "numerical_decomposition_not_isolated_cause"})
    return rows


def complete_bundle(directory: Path, filenames: list[str], binding_sha: str, source_digest: str, fit_count: int, elapsed: float) -> None:
    atomic_json(directory / "COMPLETE.json", {"status": "COMPLETE", "source_binding_sha256": binding_sha, "source_digest": source_digest, "files": {name: reviewed.sha256_file(directory / name) for name in filenames}, "fit_count": fit_count, "elapsed_seconds": elapsed, "official_test_accessed": False})


def resume_valid(directory: Path, binding_sha: str, source_digest: str, expected_files: set[str], expected_fits: int | None = None) -> bool:
    path = directory / "COMPLETE.json"
    if not path.exists():
        return False
    m = json_read(path)
    if m.get("status") != "COMPLETE" or m.get("source_binding_sha256") != binding_sha or m.get("source_digest") != source_digest or set(m.get("files", {})) != expected_files or m.get("official_test_accessed") is not False:
        raise GateError("completed bundle identity differs; do not overwrite")
    if expected_fits is not None and m.get("fit_count") != expected_fits:
        raise GateError("completed bundle fit ledger differs")
    for name, sha in m["files"].items():
        if Path(name).name != name or (directory / name).is_symlink() or reviewed.sha256_file(directory / name) != sha:
            raise GateError("completed bundle data hash differs")
    return True


def fit_unit(panel: dict[str, np.ndarray], unit: str, provenance: dict[str, Any], directory: Path, binding_sha: str) -> None:
    started = time.perf_counter()
    identity = unit_identity(unit)
    arms, diagnostics, predictions, selection, fitted, affine = {}, [], [], [], [], []
    if unit == "cbis_r0_f0":
        for rep in ("soft", "logodds"):
            a, _ = coordinates(panel["x0"], "cbis", panel["widths"], 1.0, rep)
            b, _ = coordinates(panel["z0"], "cbis", panel["widths"], 1.0, rep)
            x, _, _ = scale_data(a, b)
            for hp in LAMBDA_GRID:
                _, diag, _ = fit_head(x, panel["y"][panel["f0"]], "mlp", hp, 17, f"{unit}/pilot/{rep}/{hp}", "fixed_recipe_pilot", 1.0)
                diagnostics.append(diag)
    for tau in TEMPS:
        for rep in ("soft", "logodds"):
            for head in ("lr", "mlp"):
                result = run_arm(panel, unit, tau, rep, head, provenance)
                arms[(tau, rep, head)] = result
                for key, target in (("predictions", predictions), ("selection", selection), ("diagnostics", diagnostics), ("fitted", fitted)):
                    target.extend(result[key])
        if identity["domain"] == "cbis":
            affine.extend(affine_diagnostics(panel, unit, tau, arms))
    exact = {"unit": unit, "representative": False, "pass": True, "fit_count": 0}
    if unit in REPRESENTATIVES:
        exact, diag, extra = exact_controls(panel, unit, arms, provenance)
        diagnostics.extend(diag)
        fitted.extend(extra)
    expected_count = 160 + (6 if unit == "cbis_r0_f0" else 0) + (97 if unit in REPRESENTATIVES else 0)
    if len(diagnostics) != expected_count:
        raise GateError("unit scientific fit count mismatch")
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in (("predictions.csv", predictions), ("selection.csv", selection), ("fit_diagnostics.csv", diagnostics), ("affine.csv", affine)):
        atomic_csv(directory / name, value)
    for name, value in (("fitted_heads.json", fitted), ("lineage.json", provenance), ("exact_equivalence.json", exact)):
        atomic_json(directory / name, value)
    filenames = ["predictions.csv", "selection.csv", "fit_diagnostics.csv", "affine.csv", "fitted_heads.json", "lineage.json", "exact_equivalence.json"]
    complete_bundle(directory, filenames, binding_sha, digest(provenance), len(diagnostics), time.perf_counter() - started)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def prepare_estimators(rows: list[dict[str, Any]], domain: str) -> tuple[dict[Any, Any], np.ndarray, list[str], list[str]]:
    expected_reps = [0] if domain == "cbis" else list(range(5))
    arm_keys = [(tau, rep, head, seed) for tau in TEMPS for rep in ("soft", "logodds") for head in ("lr", "mlp") for seed in ((0,) if head == "lr" else SEEDS)]
    index = {}
    mapping = {}
    for row in rows:
        case = str(row["case_id"])
        label, group = int(row["label"]), str(row["patient_group"])
        if case in mapping and mapping[case] != (label, group):
            raise GateError("case/group/label mapping changed during stitching")
        mapping[case] = (label, group)
        key = (float(row["temperature"]), row["representation"], row["head"], int(row["refit_seed"]), int(row["repetition"]), case)
        if key in index:
            raise GateError("duplicate outer case prediction")
        index[key] = float(row["logit"])
    cases = sorted(mapping)
    labels = np.array([mapping[c][0] for c in cases])
    groups = [mapping[c][1] if domain == "cbis" else c for c in cases]
    expected_keys = {(*arm, r, c) for arm in arm_keys for r in expected_reps for c in cases}
    if set(index) != expected_keys or set(np.unique(labels)) != {0, 1}:
        raise GateError("incomplete stitched arm/repetition/seed panels")
    estimators = {}
    for arm in arm_keys:
        logits = np.array([[index[(*arm, r, c)] for c in cases] for r in expected_reps])
        if not np.all(np.isfinite(logits)):
            raise GateError("nonfinite stitched logits")
        p = reviewed.stable_sigmoid(logits)
        auc_orders = []
        for repetition_logits in logits:
            order = np.argsort(repetition_logits, kind="stable")
            starts = np.r_[0, np.flatnonzero(np.diff(repetition_logits[order]) != 0) + 1]
            auc_orders.append((order, starts))
        estimators[arm] = {"logits": logits, "auc_orders": auc_orders, "loss": np.mean(np.logaddexp(0, logits) - labels * logits, axis=0), "brier": np.mean((p - labels) ** 2, axis=0)}
    return estimators, labels, cases, groups


def metrics(estimator: dict[str, Any], labels: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    means = [np.average(estimator["loss"][labels == k], weights=weights[labels == k]) for k in (0, 1)]
    aucs = []
    # Fixed scores permit exact cached weighted pair-count AUC, including ties.
    # This avoids millions of repeated sklearn sort operations, not a new estimand.
    for order, starts in estimator["auc_orders"]:
        positive = np.add.reduceat(weights[order] * labels[order], starts)
        negative = np.add.reduceat(weights[order] * (1 - labels[order]), starts)
        aucs.append(float(np.sum(positive * (np.cumsum(negative) - negative / 2)) / (positive.sum() * negative.sum())))
    return {"class_equal_nll": float(np.mean(means)), "natural_nll": float(np.average(estimator["loss"], weights=weights)), "brier": float(np.average(estimator["brier"], weights=weights)), "auc": float(np.mean(aucs))}


def effects(arm_metrics: dict[Any, dict[str, float]]) -> dict[Any, float]:
    out = {}
    for metric in ("class_equal_nll", "auc", "natural_nll", "brier"):
        direction = -1 if metric == "auc" else 1
        for tau in TEMPS:
            lr = direction * (arm_metrics[(tau, "logodds", "lr", 0)][metric] - arm_metrics[(tau, "soft", "lr", 0)][metric])
            out[(metric, "representation_D", tau, "lr", "0")] = lr
            values = []
            for seed in SEEDS:
                value = direction * (arm_metrics[(tau, "logodds", "mlp", seed)][metric] - arm_metrics[(tau, "soft", "mlp", seed)][metric])
                values.append(value)
                out[(metric, "representation_D", tau, "mlp", str(seed))] = value
                out[(metric, "capacity_I", tau, "mlp_minus_lr", str(seed))] = value - lr
            out[(metric, "representation_D", tau, "mlp", "seed_mean")] = float(np.mean(values))
            out[(metric, "capacity_I", tau, "mlp_minus_lr", "seed_mean")] = float(np.mean(values)) - lr
        for head, seeds in (("lr", ("0",)), ("mlp", tuple(map(str, SEEDS)) + ("seed_mean",))):
            for seed in seeds:
                out[(metric, "temperature_interaction", "0.25_minus_1", head, seed)] = out[(metric, "representation_D", 0.25, head, seed)] - out[(metric, "representation_D", 1.0, head, seed)]
    return out


def aggregate_domain(rows: list[dict[str, Any]], domain: str, rng: np.random.Generator, replicates: int = 10000, maximum: int = 100000, draw_sink: Any = None) -> dict[str, Any]:
    """replicates/maximum are helper-only knobs for synthetic tests, not CLI options."""
    estimators, labels, cases, groups = prepare_estimators(rows, domain)
    clusters = sorted(set(groups))
    group_index = np.array([clusters.index(group) for group in groups])
    evaluate = lambda w: {arm: metrics(e, labels, w) for arm, e in estimators.items()}
    point_arms = evaluate(np.ones(len(cases)))
    point = effects(point_arms)
    order = list(point)
    values = np.empty((replicates, len(order)))
    draws = []
    accepted, proposal = 0, 0
    while accepted < replicates and proposal < maximum:
        proposal += 1
        multiplicity = np.bincount(rng.integers(len(clusters), size=len(clusters)), minlength=len(clusters))
        weights = multiplicity[group_index]
        if any(weights[labels == k].sum() == 0 for k in (0, 1)):
            continue
        current = effects(evaluate(weights))
        cluster_digest = digest(multiplicity.tolist())
        for j, key in enumerate(order):
            values[accepted, j] = current[key]
            metric, contrast, tau, head, seed = key
            row = {"domain": domain, "accepted_index": accepted + 1, "proposal_index": proposal, "cluster_multiplicity_digest": cluster_digest, "estimand_order": j, "metric": metric, "contrast": contrast, "temperature": tau, "head": head, "refit_seed_or_seed_mean": seed, "effect": current[key]}
            if draw_sink is None:
                draws.append(row)
            else:
                draw_sink(row)
        accepted += 1
    if accepted != replicates:
        raise GateError("bootstrap did not reach exact accepted proposal count")
    summary, arm_rows = [], []
    for arm, metric_values in point_arms.items():
        tau, rep, head, seed = arm
        for metric, value in metric_values.items():
            arm_rows.append({"domain": domain, "metric": metric, "representation": rep, "temperature": tau, "head": head, "refit_seed": seed, "estimate": value, "case_count": len(cases), "cluster_count": len(clusters), "repetition_rule": "casewise_mean_losses_or_mean_repetition_auc", "seed_rule": "individual_seed_not_probability_ensemble"})
    for tau in TEMPS:
        for rep in ("soft", "logodds"):
            for metric in ("class_equal_nll", "auc", "natural_nll", "brier"):
                arm_rows.append({"domain": domain, "metric": metric, "representation": rep, "temperature": tau, "head": "mlp", "refit_seed": "seed_mean", "estimate": float(np.mean([point_arms[(tau, rep, "mlp", seed)][metric] for seed in SEEDS])), "case_count": len(cases), "cluster_count": len(clusters), "repetition_rule": "casewise_mean_losses_or_mean_repetition_auc", "seed_rule": "arithmetic_mean_of_three_seed_metrics_not_probabilities"})
    for j, key in enumerate(order):
        metric, contrast, tau, head, seed = key
        lower, upper = np.quantile(values[:, j], [0.025, 0.975], method="linear")
        summary.append({"domain": domain, "metric": metric, "contrast": contrast, "temperature": tau, "head": head, "refit_seed_or_seed_mean": seed, "point_estimate": point[key], "ci_lower": float(lower), "ci_upper": float(upper), "bootstrap_replicates": replicates, "bootstrap_seed": 2026091517, "cluster_variable": "patient_group" if domain == "cbis" else "case_id", "case_count": len(cases), "cluster_count": len(clusters), "direction_convention": "NLL_Brier_gain=logodds-soft;AUC_gain=soft-logodds;interactions_are_direct_differences", "repetition_rule": "casewise_mean_losses_or_mean_repetition_auc", "mlp_seed_handling": "matching_seed_effects_then_arithmetic_mean", "resource_resolution": 0.01 if metric == "auc" else (0.005 if "nll" in metric else "not_defined"), "resolution_interpretation": "descriptive_not_equivalence_or_clinical_margin", "interval_interpretation": "crossing_zero_is_inconclusive_not_equivalence"})
    return {"arm_metrics": arm_rows, "paired_effects": summary, "draws": draws, "accepted": accepted, "proposals": proposal}


def structure_measurement() -> dict[str, Any]:
    binding = validate_binding()
    started = time.perf_counter()
    audit = reviewed.audit_panel_structure()
    if audit["status"] != "PASS" or digest(audit["source_contract"]) != binding["archived_source_receipt_sha256"]:
        raise GateError("archived development source receipt differs")
    source_seconds = time.perf_counter() - started
    started = time.perf_counter()
    synthetic = reviewed.synthetic_solver_equivalence_check()
    if not synthetic["pass"]:
        raise GateError("reviewed synthetic equivalence failed")
    synthetic_seconds = time.perf_counter() - started
    rng = np.random.default_rng(2026091602)
    x = rng.normal(size=(480, 19))
    y = np.tile([0, 1], 240)
    started = time.perf_counter()
    fit_head(x, y, "mlp", 0.001, 17, "synthetic_cost_only", "synthetic_cost_only", 1.0)
    mlp_seconds = time.perf_counter() - started
    representative = []
    for unit in REPRESENTATIVES:
        started = time.perf_counter()
        with np.load(reviewed.DEFAULT_PANELS / unit / "panel.npz", allow_pickle=False) as p:
            panels = {k: np.asarray(p[k]) for k in reviewed.REQUIRED_ARRAYS}
        lin = lineage(panels, json_read(reviewed.DEFAULT_PANELS / unit / "generation.json"), unit_identity(unit)["domain"], audit["units"][unit], unit=unit)
        representative.append({"unit": unit, "load_and_lineage_seconds": time.perf_counter() - started, "training_shape": list(panels["raw"].shape), "evaluation_shape": list(panels["val"].shape), "lineage_sha256": digest(lin), "scientific_fits": 0})
    forecast = 3194 * synthetic_seconds / 18 + 1806 * mlp_seconds
    return {"status": "STRUCTURAL_SYNTHETIC_ONLY_NOT_EXECUTION_APPROVAL", "source_binding_sha256": reviewed.sha256_file(BINDING), "environment": environment(), "source_audit_seconds": source_seconds, "synthetic_LR_18fits_seconds": synthetic_seconds, "synthetic_MLP_200epochs_seconds": mlp_seconds, "representative_structural_measurement": representative, "synthetic_only_forecast_seconds": forecast, "forecast_2x4x_seconds": [forecast * 2, forecast * 4], "forecast_limit": "synthetic_CPU_forecast_not_real_representative_fit_cost; actual_recipe_pilot_after_freeze_audit_authorization", "fit_counts": json_read(PROTOCOL)["fit_counts"], "synthetic_fits": 19, "scientific_fits": 0, "scientific_outcomes_read": False, "official_test_accessed": False, "p1_outputs_read": False, "synthetic_equivalence": synthetic}


def execute() -> None:
    started = time.perf_counter()
    gate = require_execution_gate()
    audit = reviewed.audit_panel_structure()
    if audit["status"] != "PASS" or digest(audit["source_contract"]) != gate["binding"]["archived_source_receipt_sha256"]:
        raise GateError("development source contract changed")
    synthetic = reviewed.synthetic_solver_equivalence_check()
    if not synthetic["pass"]:
        raise GateError("numeric source gate failed")
    contract = json_read(PREP / "P2_OUTPUT_CONTRACT.json")
    unit_files = set(contract["unit_files"]) - {"COMPLETE.json"}
    for unit in reviewed.EXPECTED_UNITS:
        with np.load(reviewed.DEFAULT_PANELS / unit / "panel.npz", allow_pickle=False) as archive:
            panel = {name: np.asarray(archive[name]) for name in reviewed.REQUIRED_ARRAYS}
        provenance = lineage(panel, json_read(reviewed.DEFAULT_PANELS / unit / "generation.json"), unit_identity(unit)["domain"], audit["units"][unit], unit=unit)
        expected_fits = 160 + (6 if unit == "cbis_r0_f0" else 0) + (97 if unit in REPRESENTATIVES else 0)
        if not resume_valid(OUTPUT / unit, gate["source_binding_sha256"], digest(provenance), unit_files, expected_fits):
            fit_unit(panel, unit, provenance, OUTPUT / unit, gate["source_binding_sha256"])
        print(json.dumps({"event": "unit_complete", "unit": unit, "fit_count": expected_fits}), flush=True)
    aggregate = OUTPUT / "aggregate"
    files = set(contract["aggregate_files"]) - {"COMPLETE.json"}
    source_digest = digest(audit["source_contract"])
    if resume_valid(aggregate, gate["source_binding_sha256"], source_digest, files, 5000):
        return
    rng = np.random.Generator(np.random.PCG64(2026091517))
    arms, paired, exact = [], [], []
    bootstrap_log = []
    aggregate.mkdir(parents=True, exist_ok=True)
    draw_path = aggregate / "bootstrap_draws.csv"
    draw_stage = draw_path.with_name(draw_path.name + ".pending")
    if draw_path.is_symlink() or draw_stage.is_symlink() or aggregate.is_symlink():
        raise GateError("unsafe aggregate draw output")
    # Draw rows stream to disk; only the 10000-by-92 effect matrix is retained.
    draw_keys = ["domain", "accepted_index", "proposal_index", "cluster_multiplicity_digest", "estimand_order", "metric", "contrast", "temperature", "head", "refit_seed_or_seed_mean", "effect"]
    with draw_stage.open("w", newline="") as draw_handle:
        writer = csv.DictWriter(draw_handle, fieldnames=draw_keys)
        writer.writeheader()
        for domain in ("cbis", "derm"):
            domain_rows = []
            for unit in reviewed.EXPECTED_UNITS:
                if unit_identity(unit)["domain"] == domain:
                    domain_rows.extend(read_csv(OUTPUT / unit / "predictions.csv"))
                    exact.append(json_read(OUTPUT / unit / "exact_equivalence.json"))
            result = aggregate_domain(domain_rows, domain, rng, draw_sink=writer.writerow)
            arms.extend(result["arm_metrics"])
            paired.extend(result["paired_effects"])
            bootstrap_log.append({"domain": domain, "accepted": result["accepted"], "proposals": result["proposals"]})
        draw_handle.flush()
        os.fsync(draw_handle.fileno())
    os.replace(draw_stage, draw_path)
    atomic_csv(aggregate / "arm_metrics.csv", arms)
    atomic_csv(aggregate / "paired_effects.csv", paired)
    atomic_csv(aggregate / "seedwise_effects.csv", [{k: v for k, v in r.items() if k not in ("ci_lower", "ci_upper")} for r in paired])
    atomic_json(aggregate / "exact_equivalence.json", {"synthetic": synthetic, "units": exact, "all_pass": all(x["pass"] for x in exact), "source_binding_sha256": gate["source_binding_sha256"]})
    atomic_json(aggregate / "SOURCE_RECEIPT.json", {"archived_receipt": audit["source_contract"], "approval_chain": {k: v for k, v in gate.items() if k != "binding"}, "protocol_sha256": reviewed.sha256_file(PROTOCOL), "environment": environment(), "unit_manifest_sha256": {u: reviewed.sha256_file(OUTPUT / u / "COMPLETE.json") for u in reviewed.EXPECTED_UNITS}})
    atomic_json(aggregate / "RUN_LOG.json", {"status": "DEVELOPMENT_COMPLETE_NO_CONFIRMATION", "scientific_fit_count": 5000, "elapsed_seconds": time.perf_counter() - started, "bootstrap": bootstrap_log, "official_test_accessed": False, "p1_outputs_read": False, "hypothesis_judgment": "descriptive_development_evidence; positioning_gate_and_untouched_confirmation_still_required", "environment": environment()})
    complete_bundle(aggregate, sorted(files), gate["source_binding_sha256"], source_digest, 5000, time.perf_counter() - started)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--structure", action="store_true", help="structural and synthetic preparation only (default)")
    action.add_argument("--execute", action="store_true", help="external frozen/audited/authorized development run only")
    args = parser.parse_args()
    try:
        require_cpu_one()
        with threadpool_limits(limits=1):
            if args.execute:
                execute()
            else:
                report = structure_measurement()
                atomic_json(PREP / "STRUCTURAL_COST_FORECAST.json", report)
                print(json.dumps({k: v for k, v in report.items() if k != "synthetic_equivalence"}, sort_keys=True))
    except (GateError, OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(error), "official_test_accessed": False}), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
