"""Bounded strict-nested revision. Development assets only; resumable outer units."""
from pathlib import Path
import argparse
import dataclasses
import json
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import paper_b_cbis_p3 as cb
from paper2_mpjc.pilot import _seed, case_kfold_splits, _oof_probe_scores, _fit_probe_blocks
from paper2_evidence_handoff.data import EvidenceCohort, CONCEPT_STATES, CONCEPT_VALUE_MAP, DIAGNOSIS_GROUPS, LABELS
from run_derm7pt_cf_mpjc_d1 import _config_from_contract_mapping

OUT = Path(__file__).resolve().parent / 'panels'
GRID = [.001, .01, .1, 1., 10., 100.]
PROTOCOL = dict(version='deadline-strict-nested-v1', date='2026-10-10', encoder='OpenAI CLIP ViT-B/16', only_encoder_changed=True,
    domains={'cbis': dict(repetitions=1, outer=5, inner='remaining four original patient folds', clip=1e-6),
             'derm': dict(repetitions=5, outer=5, inner='original strict_meta four case folds; four probe folds inside each T\\H', clip=1e-12)},
    c_grid=GRID, head='balanced L2 lbfgs logistic, independent C per representation',
    selection='mean inner class-equal NLL; ties smallest C',
    standardization='coordinate mean/std ddof0 fitted on inner-training panel; std<=1e-12 replaced by 1',
    conditions='Analysis 1 only: S, C, Cflex, and M on nested predicted-concept panels',
    exclusion='inner training G probes exclude E,H,G; inner H probe excludes E,H; final T OOF and T-fit predicts E',
    inference='10000 paired cluster bootstrap draws; repetitions stay together; conditional on fitted panels',
    protected='No reserved test access, no new cohorts, no outcome-dependent search')

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))+'\n')

def derm_load():
    # Intentionally does not call the old loader, which opens official test indexes.
    with np.load(Path(__file__).resolve().parent/'derm7pt_clip_vit_b16_features.npz') as a:
        assert str(a['cohort_role'].item()) == 'DERM7PT_OFFICIAL_TRAIN_VALID_DEV609_ONLY'
        idx, x = a['metadata_index'].astype(int), a['derm_embeddings'].astype(float)
    md = pd.read_csv(ROOT/'data/Derm7pt/derm7pt_dev609_only_v1_20260908.csv').set_index('metadata_index').loc[idx]
    concepts = {n: np.array([states.index(CONCEPT_VALUE_MAP[n][str(v).strip().lower()]) for v in md[n]]) for n, states in CONCEPT_STATES.items()}
    y = np.array([int(DIAGNOSIS_GROUPS[str(v).strip().lower()] == 'MEL') for v in md.diagnosis])
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    cohort = EvidenceCohort('DERM7PT', np.array([f'derm7pt:{i:04d}' for i in idx]), idx, x, y, LABELS, concepts, CONCEPT_STATES)
    cohort.validate()
    assert len(y)==609 and np.bincount(y).tolist()==[460,149]
    return cohort

def make_panel(domain, rep, fold):
    if domain == 'cbis':
        config=json.loads((ROOT/'configs/paper_b_cbis_p3_minimal_protocol_final_v1_20260910.json').read_text())
        archive=ROOT/config['cohort']['archive']
        full=cb.load_cohort(archive, cb.STABLE_CONCEPTS, complete_only=False)
        co=cb.load_cohort(archive, cb.STABLE_CONCEPTS, complete_only=True)
        with np.load(Path(__file__).resolve().parent/'cbis_clip_vit_b16_breast_features.npz') as clip:
            assert np.array_equal(clip['unit_id'].astype(str), co.unit_id.astype(str))
            x = clip['embeddings'].astype(float)
        co=dataclasses.replace(co, x=x)
        # Fit sets repeated with H/G exchanged have identical recipes and seeds.
        original=cb.fit_concept_probes
        cache={}
        def cached(cohort, rows, **kwargs):
            key=tuple(sorted(np.asarray(rows).tolist()))
            if key not in cache: cache[key]=original(cohort,rows,**kwargs)
            return cache[key]
        cb.fit_concept_probes=cached
        try:
            p=cb.build_outer_concept_panel(co,fold,probe_c=1.,max_iter=10000,seed=20260910,minimum_state_support=10)
        finally: cb.fit_concept_probes=original
        panels=[(s.fit_rows,s.heldout_rows,s.nested_oof_fit_probability,s.heldout_probability) for s in p.selection]
        return co.y,co.unit_id,co.patient_id,co.concepts.astype(float),None,p.fit_rows,p.validation_rows,p.final_oof_probability,p.validation_probability,panels,dict(manifests=p.manifests,selection_manifests=[s.manifests for s in p.selection],probe_bundles=len(cache))
    co=derm_load()
    cfg=_config_from_contract_mapping(json.loads((ROOT/'configs/cf_mpjc_d1_derm7pt.json').read_text()))
    t,e=case_kfold_splits(len(co.y),cfg.outer_folds,_seed(cfg.seed,'outer',rep))[fold]
    splits=case_kfold_splits(len(t),cfg.calibration_folds,_seed(cfg.seed,'strict_meta',rep,fold))
    panels=[]; manifests=[]; raw=np.full((len(t),co.concept_dimension),np.nan)
    for h,(tr,va) in enumerate(splits):
        f,hrows=t[tr],t[va]
        assert not np.intersect1d(f,np.r_[hrows,e]).size
        x,_,_,m=_oof_probe_scores(co,f,cfg,seed_parts=('strict_meta_inner',rep,fold,h))
        z,_=_fit_probe_blocks(co,f,hrows,cfg,('strict_meta_predict',rep,fold,h))
        raw[va]=z
        panels.append((f,hrows,x,z))
        manifests.append(dict(fit_case_ids=co.case_id[f],heldout_case_ids=co.case_id[hrows],inner=m))
    val,_=_fit_probe_blocks(co,t,e,cfg,('outer_validation_probe_fit_all',rep,fold))
    # Ensure new selection folds remain the same as the archived main workflow.
    with np.load(ROOT/'results/paper_b_p1_matched_training_r1_20260910/matched_train_exposures.npz') as a:
        mask=(a['repetition']==rep)&(a['outer_fold']==fold)
        assert np.array_equal(a['cohort_row'][mask],t), 'development ordering changed'
        for h,(_,va) in enumerate(splits): assert np.all(a['meta_fold'][mask][va]==h)
    return co.y,co.case_id,co.case_id,co.gold_features(),list(map(len,co.concept_states.values())),t,e,raw,val,panels,dict(selection_manifests=manifests,probe_bundles=21)

def transform(x,representation,widths,epsilon):
    if representation=='soft': return x.copy()
    if widths is None:
        if representation=='hard': return (x>=.5).astype(float)
        z=np.clip(x,epsilon,1-epsilon); return np.log(z)-np.log1p(-z)
    blocks=[]; start=0
    for width in widths:
        block=x[:,start:start+width]; start+=width
        if representation=='hard': blocks.append(np.eye(width)[block.argmax(axis=1)])
        else:
            z=np.log(np.clip(block,epsilon,1.)); blocks.append(z[:,1:]-z[:,:1])
    return np.concatenate(blocks,axis=1)

def ce_nll(y,p):
    loss=-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1))
    assert set(y)=={0,1}
    return float(np.mean([loss[y==k].mean() for k in (0,1)]))

def fit(x,y,c,max_iter):
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        return LogisticRegression(C=c,class_weight='balanced',max_iter=max_iter,solver='lbfgs',random_state=20260912).fit(x,y)

def scale(x,z,on):
    mean=x.mean(0) if on else np.zeros(x.shape[1]); sd=x.std(0) if on else np.ones(x.shape[1]); sd=np.where(sd>1e-12,sd,1.)
    return (x-mean)/sd,(z-mean)/sd,mean,sd

def run_unit(domain,rep,fold):
    dest=OUT/f'{domain}_r{rep}_f{fold}'; dest.mkdir(parents=True,exist_ok=True)
    if (dest/'COMPLETE.json').exists(): return
    started=time.monotonic(); cache=dest/'panel.npz'
    if not cache.exists():
        y,ids,groups,gold,widths,t,e,raw,val,panels,audit=make_panel(domain,rep,fold)
        data=dict(y=y,ids=ids,groups=groups,gold=gold,widths=np.array([] if widths is None else widths),t=t,e=e,raw=raw,val=val)
        for h,values in enumerate(panels):
            for key,v in zip(('f','h','x','z'),values): data[f'{key}{h}']=v
        np.savez_compressed(cache,**data)
        dump(dest/'generation.json',dict(seconds=time.monotonic()-started,**audit))
    # This stage materializes only the nested concept-score panels. Analysis 1
    # is fitted once in run_analysis.py; the representation/capacity analyses
    # from the source script are deliberately outside this check.
    dump(dest/'COMPLETE.json',dict(domain=domain,repetition=rep,outer_fold=fold,
                                   seconds=time.monotonic()-started,
                                   artifact='nested_concept_score_panel'))
    print(json.dumps(dict(completed=dest.name,seconds=round(time.monotonic()-started,2))),flush=True)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--domain',choices=['cbis','derm','all'],default='all'); parser.add_argument('--pilot',action='store_true'); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    freeze=OUT/'protocol.json'
    if freeze.exists(): assert json.loads(freeze.read_text())==PROTOCOL
    else: dump(freeze,PROTOCOL)
    with threadpool_limits(limits=1):
        for domain in (['cbis','derm'] if args.domain=='all' else [args.domain]):
            for rep in range(1 if args.pilot or domain=='cbis' else 5):
                for fold in range(1 if args.pilot else 5): run_unit(domain,rep,fold)

if __name__=='__main__': main()
