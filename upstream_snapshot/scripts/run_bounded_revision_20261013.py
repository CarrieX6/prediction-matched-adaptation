"""Finite post-review checks; development fitting only, archived external predictions.

All recipes are fixed in protocol.json before any new outcome is computed.
The primary archived experiment is never overwritten.
"""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import dataclasses
import json
import sys
import time
import warnings
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import paper_b_cbis_p3 as cb
import run_deadline_nested_20260912 as old
from run_cflex_df_sensitivity_20261013 import Basis, VARIANTS, auc_cache, weighted_auc

OUT = ROOT/'paper_b_readiness/BOUNDED_REVISION_20261013'
ARCHIVE = ROOT/'results/deadline_nested_20260912'
CF = ROOT/'paper_b_readiness/CFLEX_CONTROL_20260917/full/units'
GRID = old.GRID
PROTOCOL = {
 'status':'post-review exploratory supplements; original primary results retained',
 'non_concept_controls':{'dimensions':{'cbis':10,'derm':12},'PCA':'fit on allowed F or T embeddings only; H/E excluded; shared coordinate system within each head fit',
   'random_projection_seeds':[20261025,20261026,20261027], 'projection':'fixed orthonormal Gaussian directions; all seeds reported, no best-seed selection',
   'scaling':'training-only coordinate standardization for PCA, random, and matched soft-score control',
   'selection':'six original C values, mean of four inner-H class-equal NLLs; smallest C on ties'},
 'L1':{'solver':'liblinear','seed':20261028,'scaling':'training-only standardization','C_grid':GRID,
   'selection':'four inner-H validation losses; no outer or external selection'},
 'CBIS_repeat':{'new_partition_seeds':[20261031,20261032], 'outer':'5-fold StratifiedGroupKFold, patient groups',
   'upstream':'full nested refit, fixed probe C=1, minimum state support=10; no reuse of old OOF predictions',
   'heads':'original raw-score M and four-coefficient Cflex; full-chain reference-head exclusion',
   'failure':'report a fixed-support failure; do not change support or seed'},
 'learning_curve':{'domain':'CBIS original partition','fractions':[0.5,0.75,1.0], 'seed':20261033,
   'scope':'diagnostic-head training size only; archived nested upstream panels fixed',
   'subset':'nested random patient subsets within each F/T, H/E unchanged; both heads use the same subset'},
 'PH2':{'scope':'primary 25-model mean probabilities; class-specific distance associations and exact ranking-pair changes; no label-based fitting',
   'distance':'Ledoit-Wolf on 609 source OOF mean score vectors, 12 free coordinates',
   'decomposition':'mean per-model M minus reference logit; not a decomposition of the nonlinear Cflex probability'},
 'inference':{'bootstrap':10000,'seed':20261034,'unit':'patients for CBIS, cases for Derm/PH2; same draw across arms and repetitions',
   'scope':'conditional on fitted models; exploratory intervals, not multiplicity-adjusted'},
 'stop':'one execution of these recipes; null or unfavorable outcomes retained; no outcome-dependent extension'
}

def dump(path, value):
    path.write_text(json.dumps(value,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else str(a))+'\n')

def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'protocol.json'
    if path.exists(): assert json.loads(path.read_text())==PROTOCOL
    else: dump(path,PROTOCOL)

def cbis_load():
    config=json.loads((ROOT/'configs/paper_b_cbis_p3_minimal_protocol_final_v1_20260910.json').read_text())
    full=cb.load_cohort(ROOT/config['cohort']['archive'],cb.STABLE_CONCEPTS,complete_only=False)
    co=cb.load_cohort(ROOT/config['cohort']['archive'],cb.STABLE_CONCEPTS,complete_only=True)
    x,_=cb.validate_embedding_release(ROOT,config,full)
    return dataclasses.replace(co,x=cb.subset_x(full,co,x))

def fit(x,y,c,l1=False):
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        return LogisticRegression(C=c,class_weight='balanced',solver='liblinear' if l1 else 'lbfgs',
          penalty='l1' if l1 else 'l2',max_iter=10000,tol=1e-4,random_state=20261028 if l1 else 20260912).fit(x,y)

def loss(y,p):
    p=np.clip(p,1e-12,1-1e-12)
    l=-y*np.log(p)-(1-y)*np.log1p(-p)
    return .5*(l[y==0].mean()+l[y==1].mean())

def select(blocks,y,l1=False):
    records=[]
    for c in GRID:
        ls=[loss(y[h],fit(x,y[f],c,l1).predict_proba(z)[:,1]) for f,h,x,z in blocks]
        records.append({'C':c,'loss':float(np.mean(ls)),'fold_losses':ls})
    record=min(records,key=lambda v:v['loss'])
    return record['C'],records

def panel_read(name):
    with np.load(ARCHIVE/name/'panel.npz') as a: return {k:a[k] for k in a.files}

def frame(data,name,arm,p):
    domain,rep,fold=name.split('_'); e=data['e']
    return pd.DataFrame({'domain':domain,'rep':int(rep[1:]),'fold':int(fold[1:]),'case_id':data['ids'][e],
       'group_id':data['groups'][e],'y':data['y'][e],'arm':arm,'p':p})

def projection(x,f,h,kind,dim,seed):
    if kind=='pca':
        model=PCA(n_components=dim,svd_solver='full').fit(x[f]); a,b=model.transform(x[f]),model.transform(x[h])
    else:
        rng=np.random.default_rng(seed)
        w=np.linalg.qr(rng.normal(size=(x.shape[1],dim)),mode='reduced')[0]
        a,b=x[f]@w,x[h]@w
    a,b,_,_=old.scale(a,b,True)
    return a,b

def controls():
    co=cbis_load(); derm=old.derm_load()
    for domain,x,dim,reps in [('cbis',co.x,10,1),('derm',derm.x,12,5)]:
      for rep in range(reps):
       for fold in range(5):
        name=f'{domain}_r{rep}_f{fold}'; dest=OUT/'controls'/name; dest.mkdir(parents=True,exist_ok=True)
        if (dest/'COMPLETE.json').exists(): continue
        d=panel_read(name); y,t,e=d['y'],d['t'],d['e']; rows=[]; audit={}
        assert np.array_equal(y,co.y if domain=='cbis' else derm.y)
        recipes=[('PCA','pca',None)]+[(f'Random{s-20261024}','random',s) for s in PROTOCOL['non_concept_controls']['random_projection_seeds']]
        for arm,kind,seed in recipes:
            blocks=[]
            for h in range(4):
                f,hr=d[f'f{h}'],d[f'h{h}']; a,b=projection(x,f,hr,kind,dim,seed); blocks.append((f,hr,a,b))
            c,records=select(blocks,y); a,b=projection(x,t,e,kind,dim,seed)
            model=fit(a,y[t],c); rows.append(frame(d,name,arm,model.predict_proba(b)[:,1])); audit[arm]={'C':c,'selection':records}
        blocks=[]
        for h in range(4):
            f,hr=d[f'f{h}'],d[f'h{h}']; a,b,_,_=old.scale(d[f'x{h}'],d[f'z{h}'],True); blocks.append((f,hr,a,b))
        a,b,mean,sd=old.scale(d['raw'],d['val'],True)
        for l1,arm in [(False,'StandardizedM'),(True,'L1M')]:
            c,records=select(blocks,y,l1); model=fit(a,y[t],c,l1)
            rows.append(frame(d,name,arm,model.predict_proba(b)[:,1]))
            # Center within a categorical block for the same gauge as primary stability.
            w=(model.coef_.reshape(-1)/sd)
            widths=d['widths']; start=0
            for width in widths:
                w[start:start+width]-=w[start:start+width].mean(); start+=width
            audit[arm]={'C':c,'selection':records,'centered_raw_weights':w.tolist(),'nonzero_count':int(np.count_nonzero(model.coef_))}
        q=pd.read_csv(CF/name/'predictions.csv')
        assert q.case_id.astype(str).tolist()==d['ids'][e].astype(str).tolist()
        for arm in ['M','Cflex']: rows.append(frame(d,name,arm,q[f'p_{arm}'].to_numpy()))
        pd.concat(rows).to_csv(dest/'predictions.csv',index=False); dump(dest/'heads.json',audit); dump(dest/'COMPLETE.json',{'units':1})
        print('controls complete',name,flush=True)

def subset_rows(rows,groups,fraction,seed):
    patients=np.unique(groups[rows]); rng=np.random.default_rng(seed); order=rng.permutation(patients)
    chosen=order[:max(2,int(np.ceil(len(order)*fraction)))]
    return rows[np.isin(groups[rows],chosen)]

def fullchain(d,folds):
    y,gold,t,e=d['y'],d['gold'],d['t'],d['e']; q=np.full(len(t),np.nan); pos={r:i for i,r in enumerate(t)}; blocks=[]
    for h in range(4):
        f,hr,xf,zh=[d[f'{k}{h}'] for k in ('f','h','x','z')]
        gfolds=sorted(set(folds[f])); gs=[(f[folds[f]!=g],f[folds[f]==g]) for g in gfolds]
        c,_=select([(k,g,gold[k],gold[g]) for k,g in gs],y)
        ref=fit(gold[f],y[f],c); qh=ref.decision_function(zh)
        q[[pos[r] for r in hr]]=qh
        qd=np.full(len(f),np.nan)
        for k,g in gs:
            js=[(k[folds[k]!=j],k[folds[k]==j]) for j in sorted(set(folds[k]))]
            c,_=select([(a,b,gold[a],gold[b]) for a,b in js],y)
            model=fit(gold[k],y[k],c)
            local=np.flatnonzero(np.isin(f,g)); qd[local]=model.decision_function(xf[local])
        assert np.isfinite(qd).all()
        basis=Basis.fit(qd,VARIANTS[4]); blocks.append((f,hr,basis.transform(qd),basis.transform(qh)))
    assert np.isfinite(q).all()
    c,_=select([(d[f'f{h}'],d[f'h{h}'],gold[d[f'f{h}']],gold[d[f'h{h}']]) for h in range(4)],y)
    ref=fit(gold[t],y[t],c); qe=ref.decision_function(d['val'])
    basis=Basis.fit(q,VARIANTS[4])
    return blocks,basis.transform(q),basis.transform(qe)

def archived_scalar(d,name):
    """Use original full-chain logit draws so the head-only curve fixes upstream fits."""
    train=pd.read_csv(CF/name/'training_logits.csv')
    outer=pd.read_csv(CF/name/'predictions.csv')
    blocks=[]
    for h in range(4):
        f,hr=d[f'f{h}'],d[f'h{h}']
        a=train[(train.stage=='D_local_fullchain_oof_new_reference')&(train.H==h)].set_index('row_index').loc[f]
        b=train[(train.stage=='T_fullchain_oof_cached')&(train.H==h)].set_index('row_index').loc[hr]
        assert np.array_equal(a.y.to_numpy(),d['y'][f]) and np.array_equal(b.y.to_numpy(),d['y'][hr])
        qd,qh=a.raw_reference_logit.to_numpy(),b.raw_reference_logit.to_numpy()
        basis=Basis.fit(qd,VARIANTS[4]); blocks.append((f,hr,basis.transform(qd),basis.transform(qh)))
    tq=train[train.stage=='T_fullchain_oof_cached']
    assert not tq.row_index.duplicated().any()
    tq=tq.set_index('row_index').loc[d['t']].raw_reference_logit.to_numpy()
    assert outer.row_index.tolist()==d['e'].tolist()
    basis=Basis.fit(tq,VARIANTS[4])
    return blocks,basis.transform(tq),basis.transform(outer.logit_S.to_numpy())

def curve_unit(d,name,scalar,fraction):
    y,t,e=d['y'],d['t'],d['e']; blocks,a,b=scalar
    seed=20261033+int(name[-1]); group=d['groups']; sub=subset_rows(t,group,fraction,seed)
    out=[]; audit={}
    for arm in ['M','Cflex']:
        base=([(d[f'f{h}'],d[f'h{h}'],d[f'x{h}'],d[f'z{h}']) for h in range(4)] if arm=='M' else blocks)
        small=[]
        for h,(f,hr,x,z) in enumerate(base):
            rows=subset_rows(f,group,fraction,seed+h*100); local=np.flatnonzero(np.isin(f,rows)); small.append((rows,hr,x[local],z))
        c,record=select(small,y); local=np.flatnonzero(np.isin(t,sub))
        tx,ex=(d['raw'],d['val']) if arm=='M' else (a,b)
        model=fit(tx[local],y[sub],c); q=frame(d,name,arm,model.predict_proba(ex)[:,1]); q['fraction']=fraction; out.append(q)
        audit[arm]={'C':c,'n_training_patients':len(np.unique(group[sub])),'selection':record}
    return pd.concat(out),audit

def stability():
    co=cbis_load(); y=co.y
    # Validate the head-only replay using archived nested training logits.
    for fold in range(5):
        name=f'cbis_r0_f{fold}'; dest=OUT/'learning'/name; dest.mkdir(parents=True,exist_ok=True)
        if (dest/'COMPLETE.json').exists(): continue
        d=panel_read(name); scalar=archived_scalar(d,name)
        cfblocks,a,b=scalar; c,_=select(cfblocks,y); p=fit(a,y[d['t']],c).predict_proba(b)[:,1]
        original=pd.read_csv(CF/name/'predictions.csv'); maxdiff=float(np.max(np.abs(p-original.p_Cflex.to_numpy())))
        archived_c=json.loads((CF/name/'selections.json').read_text())['Cflex']['selected_C']
        assert c==archived_c and maxdiff<0.005, ('archived scalar replay',name,c,archived_c,maxdiff)
        result=[]; records={'original_Cflex_max_abs_difference':maxdiff}
        for fraction in [.5,.75]:
            q,record=curve_unit(d,name,scalar,fraction); result.append(q); records[str(fraction)]=record
        for arm in ['M','Cflex']:
            q=frame(d,name,arm,original[f'p_{arm}'].to_numpy()); q['fraction']=1.; result.append(q)
        pd.concat(result).to_csv(dest/'predictions.csv',index=False); dump(dest/'selection.json',records); dump(dest/'COMPLETE.json',{'rep':0})
        print('learning complete',name,maxdiff,flush=True)
    for rep,seed in enumerate([20261031,20261032],start=1):
        splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed); folds=np.full(len(y),-1)
        for fold,(_,e) in enumerate(splitter.split(co.x,y,co.patient_id)): folds[e]=fold
        current=dataclasses.replace(co,outer_fold=folds); cache={}; original=cb.fit_concept_probes
        def cached(cohort,rows,**kwargs):
            key=tuple(sorted(np.asarray(rows).tolist()))
            if key not in cache: cache[key]=original(cohort,rows,**kwargs)
            return cache[key]
        cb.fit_concept_probes=cached
        try:
          for fold in range(5):
            name=f'cbis_r{rep}_f{fold}'; dest=OUT/'repeated'/name; dest.mkdir(parents=True,exist_ok=True)
            if (dest/'COMPLETE.json').exists() or (dest/'FAILED.json').exists(): continue
            started=time.time()
            try:
                panel=cb.build_outer_concept_panel(current,fold,probe_c=1.,max_iter=10000,seed=20260910,minimum_state_support=10)
            except ValueError as error:
                if 'minimum state support' not in str(error): raise
                dump(dest/'FAILED.json',{'reason':str(error),'partition_seed':seed}); print('fixed support failure',name,str(error),flush=True); continue
            d=dict(y=y,ids=co.unit_id,groups=co.patient_id,gold=co.concepts.astype(float),t=panel.fit_rows,e=panel.validation_rows,raw=panel.final_oof_probability,val=panel.validation_probability)
            for h,s in enumerate(panel.selection):
                for key,v in zip(('f','h','x','z'),(s.fit_rows,s.heldout_rows,s.nested_oof_fit_probability,s.heldout_probability)): d[f'{key}{h}']=v
            np.savez_compressed(dest/'panel.npz',**d,folds=folds)
            scalar=fullchain(d,folds); rows=[]; records={}
            for arm,blocks,a,b in [('M',[(d[f'f{h}'],d[f'h{h}'],d[f'x{h}'],d[f'z{h}']) for h in range(4)],d['raw'],d['val']),('Cflex',*scalar)]:
                c,rec=select(blocks,y); model=fit(a,y[d['t']],c); rows.append(frame(d,name,arm,model.predict_proba(b)[:,1])); records[arm]={'C':c,'selection':rec,'coef':model.coef_.tolist()}
            pd.concat(rows).to_csv(dest/'predictions.csv',index=False); dump(dest/'heads.json',records); dump(dest/'lineage.json',{'partition_seed':seed,'selection':[s.manifests for s in panel.selection],'final':panel.manifests}); dump(dest/'COMPLETE.json',{'seconds':time.time()-started})
            print('repeat complete',name,round(time.time()-started,1),flush=True)
        finally: cb.fit_concept_probes=original

def summary(frame,outname):
    tables=[]; rng=np.random.default_rng(20261034)
    for domain,g in frame.groupby('domain'):
        arms=g.arm.unique().tolist(); reps=sorted(g.rep.unique()); first=g[(g.arm==arms[0])&(g.rep==reps[0])].sort_values('case_id'); ids=first.case_id.tolist(); y=first.y.to_numpy(int)
        groups=first.group_id.to_numpy(); _,cluster=np.unique(groups,return_inverse=True); n=cluster.max()+1
        ps={}; ls={}; bs={}; caches={}
        for arm in arms:
            blocks=[g[(g.arm==arm)&(g.rep==r)].sort_values('case_id') for r in reps]
            if any(b.case_id.tolist()!=ids for b in blocks): raise ValueError('Incomplete repetition: '+domain+arm)
            p=np.stack([b.p.to_numpy(float) for b in blocks]); ps[arm]=p; z=np.clip(p,1e-12,1-1e-12); ls[arm]=(-y*np.log(z)-(1-y)*np.log1p(-z)).mean(0); bs[arm]=((p-y)**2).mean(0); caches[arm]=[auc_cache(y,row) for row in p]
        def measures(w):
            return {arm:np.array([.5*sum(np.average(ls[arm][y==c],weights=w[y==c]) for c in (0,1)),np.average(ls[arm],weights=w),np.average(bs[arm],weights=w),np.mean([weighted_auc(cache,w) for cache in caches[arm]])]) for arm in arms}
        point=measures(np.ones(len(y))); draws={arm:[] for arm in arms}
        for i in range(10000):
            w=np.bincount(rng.integers(0,n,n),minlength=n)[cluster]
            if any(w[y==c].sum()==0 for c in (0,1)): continue
            values=measures(w)
            for arm in arms: draws[arm].append(values[arm])
        for arm in arms:
            arr=np.array(draws[arm]); record={'domain':domain,'arm':arm,'n':len(y),'repetitions':len(reps),'metrics':{}}
            for j,metric in enumerate(['class_equal_NLL','natural_NLL','Brier','AUC']):
                record['metrics'][metric]={'value':point[arm][j],'ci95':np.quantile(arr[:,j],[.025,.975]).tolist()}
                if arm!='M' and 'M' in arms:
                    sign=1 if j==3 else -1
                    diff=sign*(np.array(draws['M'])[:,j]-arr[:,j]); record['metrics'][metric]['gain_M_over_arm']={'value':sign*(point['M'][j]-point[arm][j]),'ci95':np.quantile(diff,[.025,.975]).tolist()}
            tables.append(record)
        for rep in reps:
            b=g[g.rep==rep]; rows=[]
            for arm in arms:
                t=b[b.arm==arm]; yy=t.y.to_numpy(int); p=t.p.to_numpy(float); rows.append({'domain':domain,'rep':int(rep),'arm':arm,'class_equal_NLL':loss(yy,p),'AUC':roc_auc_score(yy,p)})
            pd.DataFrame(rows).to_csv(OUT/f'{outname}_{domain}_rep{rep}.csv',index=False)
    dump(OUT/f'{outname}.json',tables)

def summarize():
    predictions=list((OUT/'controls').glob('*/predictions.csv'))
    if predictions: summary(pd.concat([pd.read_csv(p) for p in predictions]),'controls_summary')
    curves=list((OUT/'learning').glob('*/predictions.csv'))
    if curves:
        curve=pd.concat([pd.read_csv(p) for p in curves])
        for fraction,g in curve.groupby('fraction'): summary(g,f'learning_{fraction}')
    repeated=list((OUT/'repeated').glob('*/predictions.csv'))
    if repeated:
        parts=[pd.read_csv(p) for p in repeated]
        for f in range(5):
            d=panel_read(f'cbis_r0_f{f}'); q=pd.read_csv(CF/f'cbis_r0_f{f}'/'predictions.csv')
            for arm in ['M','Cflex']: parts.append(frame(d,f'cbis_r0_f{f}',arm,q[f'p_{arm}'].to_numpy()))
        repeated=pd.concat(parts); repeated.to_csv(OUT/'repeated_all_predictions.csv',index=False); summary(repeated,'repeated_summary')
    signs=[]
    for domain in ['cbis','derm']:
        files=list((OUT/'controls').glob(domain+'*/heads.json'))
        for arm in ['L1M','StandardizedM']:
            weights=np.array([json.loads(p.read_text())[arm]['centered_raw_weights'] for p in files])
            if len(weights): signs.append({'domain':domain,'arm':arm,'folds':len(weights),'coordinates':weights.shape[1],'sign_crossing':int(((weights.min(0)<0)&(weights.max(0)>0)).sum()),'nonzero_per_fit':[json.loads(p.read_text())[arm]['nonzero_count'] for p in files]})
    dump(OUT/'coefficient_stability.json',signs)

def ph2():
    from analyze_ph2_failure_connection_20261012 import reference_coded_head, raw_head
    from analyze_interface_shift_null_20261012 import free_columns
    inp=ROOT/'paper_b_readiness/INTERFACE_SHIFT_NULL_20261012/analysis_input.npz'
    with np.load(inp) as a: d={k:a[k] for k in a.files}
    with np.load(ROOT/'paper_b_readiness/EXPLANATION_BASELINE_EXTERNAL_20260930/PH2/private/individual_model_predictions.npz') as a:
        probabilities=a['probabilities']; units=a['units'].astype(str); arms=a['arms'].astype(str); assert np.array_equal(a['case_id'],d['ph2_case_id'])
    free=free_columns(); source=d['derm_scores'][:,free]; target=d['ph2_scores_models'][:,:,free].mean(0)
    distance=np.sqrt(LedoitWolf().fit(source).mahalanobis(target)); pm=probabilities[:,:,list(arms).index('M')].mean(0); pc=probabilities[:,:,list(arms).index('Cflex')].mean(0); delta=pm-pc; y=d['ph2_y'].astype(int)
    rng=np.random.default_rng(20261034); results={'ensemble':'25 fixed source models; mean probabilities','association':{}}
    for label in ['all',0,1]:
        idx=np.arange(len(y)) if label=='all' else np.flatnonzero(y==label); a,b=distance[idx],delta[idx]; draws=[]
        for i in range(10000):
            ii=rng.integers(0,len(idx),len(idx)); draws.append(spearmanr(a[ii],b[ii]).statistic)
        results['association'][str(label)]={'n':len(idx),'rho':float(spearmanr(a,b).statistic),'ci95':np.quantile(draws,[.025,.975]).tolist()}
    pos,neg=np.flatnonzero(y==1),np.flatnonzero(y==0)
    r_m=pm[pos,None]>pm[neg]; r_c=pc[pos,None]>pc[neg]
    lost=(r_c&~r_m).sum(1); gained=(r_m&~r_c).sum(1)
    results['ranking']={'positive_negative_pairs':len(pos)*len(neg),'lost_correct_pairs':int(lost.sum()),'gained_correct_pairs':int(gained.sum()),'net_correct_pair_change':int((gained-lost).sum()),'AUC_M':roc_auc_score(y,pm),'AUC_Cflex':roc_auc_score(y,pc),'positive_cases_with_net_loss':int((lost>gained).sum()),'positive_cases_with_net_gain':int((gained>lost).sum())}
    # Verify exact linear per-model decomposition, then average its components.
    contributions=[]; intercepts=[]
    for i,unit in enumerate(units):
        heads=json.loads((ARCHIVE/unit/'heads.json').read_text()); bm,wm=reference_coded_head(heads['raw_soft_nested']); br,wr=reference_coded_head(heads['raw_gold_nested']); q=d['ph2_scores_models'][i][:,free]
        bm0,wm0=raw_head(heads['raw_soft_nested']); br0,wr0=raw_head(heads['raw_gold_nested']); allq=d['ph2_scores_models'][i]
        np.testing.assert_allclose(expit(bm0+allq@wm0),probabilities[i,:,list(arms).index('M')],atol=1e-10)
        parts=q*(wm-wr); np.testing.assert_allclose(bm-br+parts.sum(1),(bm0+allq@wm0)-(br0+allq@wr0),atol=1e-10); contributions.append(parts); intercepts.append(bm-br)
    contrib=np.mean(contributions,axis=0); names=d['coordinate_names'][free].astype(str)
    rows=[]
    for label in [0,1]:
        mask=y==label
        for j,name in enumerate(names): rows.append({'class':label,'coordinate':name,'mean_M_minus_reference_logit_contribution':contrib[mask,j].mean(),'mean_absolute_contribution':np.abs(contrib[mask,j]).mean()})
    pd.DataFrame(rows).to_csv(OUT/'PH2_class_contributions.csv',index=False)
    df=pd.DataFrame({'case_id':d['ph2_case_id'],'y':y,'distance':distance,'p_M':pm,'p_Cflex':pc,'delta_p':delta})
    df['lost_pairs']=0;df['gained_pairs']=0;df.loc[pos,'lost_pairs']=lost;df.loc[pos,'gained_pairs']=gained
    df.to_csv(OUT/'PH2_primary_case_connections.csv',index=False); dump(OUT/'PH2_primary_connection.json',results)
    print(json.dumps(results),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['controls','stability','summarize','ph2'],required=True);args=parser.parse_args();freeze()
    with threadpool_limits(limits=1): globals()[args.mode]()
if __name__=='__main__': main()
