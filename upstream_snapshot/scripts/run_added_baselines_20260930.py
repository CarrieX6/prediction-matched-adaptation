"""Post hoc frozen-feature baselines; archived splits, source-only selection."""
from pathlib import Path
import argparse
import dataclasses
import hashlib
import json
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from run_deadline_nested_20260912 import derm_load, cb
from paper2_evidence_handoff.data import CONCEPT_STATES

OUT = ROOT / 'paper_b_readiness/EXPLANATION_BASELINE_EXTERNAL_20260930'
ARCHIVE = ROOT / 'results/deadline_nested_20260912'
UNITS = [f'cbis_r0_f{f}' for f in range(5)] + [f'derm_r{r}_f{f}' for r in range(5) for f in range(5)]
GRIDS = {'LR': [.001, .01, .1, 1., 10., 100.], 'MLP': [1e-6, 1e-5, 1e-4, .001, .01, .1]}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')


def nll(y, p):
    p = np.clip(p, 1e-12, 1-1e-12)
    losses = -(y*np.log(p)+(1-y)*np.log1p(-p))
    return float(np.mean([losses[y == k].mean() for k in (0, 1)]))


def load_cohort(domain):
    if domain == 'derm':
        co = derm_load()
        return co.x, co.y, co.case_id, co.case_id
    cfg = json.loads((ROOT/'configs/paper_b_cbis_p3_minimal_protocol_final_v1_20260910.json').read_text())
    full = cb.load_cohort(ROOT/cfg['cohort']['archive'], cb.STABLE_CONCEPTS, complete_only=False)
    co = cb.load_cohort(ROOT/cfg['cohort']['archive'], cb.STABLE_CONCEPTS, complete_only=True)
    x, _ = cb.validate_embedding_release(ROOT, cfg, full)
    return cb.subset_x(full, co, x), co.y, co.unit_id, co.patient_id


def fit_model(x, y, kind, parameter, seed):
    mean, sd = x.mean(0), x.std(0)
    sd = np.where(sd > 1e-12, sd, 1.)
    z = (x-mean)/sd
    model = {'kind': kind, 'parameter': parameter, 'seed': seed, 'mean': mean, 'scale': sd}
    if kind == 'LR':
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            fit = LogisticRegression(C=parameter, class_weight='balanced', solver='lbfgs', max_iter=10000, random_state=seed).fit(z, y)
        model.update(coef=fit.coef_[0], intercept=float(fit.intercept_[0]), iterations=int(fit.n_iter_[0]))
    else:
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        optimizer = torch.optim.Adam(net.parameters(), lr=.001)
        tx, ty = torch.tensor(z, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        tw = torch.tensor(len(y)/(2*np.bincount(y)[y]), dtype=torch.float32)
        for _ in range(200):
            optimizer.zero_grad()
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(net(tx).flatten(), ty, reduction='none')*tw).mean()
            penalty = .5*parameter*sum((p*p).sum() for name, p in net.named_parameters() if name.endswith('weight'))
            objective = loss + penalty
            objective.backward()
            optimizer.step()
        assert torch.isfinite(objective)
        model.update(state={k: v.detach().numpy() for k, v in net.state_dict().items()}, epochs=200, final_training_objective=float(objective.detach()))
    return model


def predict(model, x):
    z = (x-np.asarray(model['mean']))/np.asarray(model['scale'])
    if model['kind'] == 'LR':
        return expit(z@np.asarray(model['coef'])+model['intercept'])
    s = {k: np.asarray(v) for k, v in model['state'].items()}
    return expit((np.maximum(0, z@s['0.weight'].T+s['0.bias'])@s['2.weight'].T+s['2.bias']).ravel())


def run_unit(unit, cohort):
    dest = OUT/'baseline_units'/unit
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/'COMPLETE.json').exists():
        return
    started = time.monotonic()
    domain, rep, fold = unit.split('_')
    x, y, ids, groups = cohort
    with np.load(ARCHIVE/unit/'panel.npz') as a:
        data = {k: a[k] for k in a.files}
    assert np.array_equal(y, data['y']) and np.array_equal(ids.astype(str), data['ids'].astype(str))
    assert np.array_equal(groups.astype(str), data['groups'].astype(str))
    t, e = data['t'], data['e']
    assert not set(groups[t]) & set(groups[e])
    predictions = pd.DataFrame(dict(domain=domain, repetition=int(rep[1:]), outer_fold=int(fold[1:]), row_index=e, case_id=ids[e], group_id=groups[e], y=y[e]))
    selections, fitted = [], {}
    for kind, grid in GRIDS.items():
        risks = []
        for parameter in grid:
            losses = []
            for h in range(4):
                f, v = data[f'f{h}'], data[f'h{h}']
                assert set(f) | set(v) == set(t)
                assert not set(groups[f]) & set(groups[v])
                m = fit_model(x[f], y[f], kind, parameter, 17)
                losses.append(nll(y[v], predict(m, x[v])))
            risks.append(float(np.mean(losses)))
            selections.append(dict(kind=kind, parameter=parameter, fold_nll=losses, mean_nll=risks[-1]))
        chosen = grid[int(np.argmin(risks))]
        seeds = [17] if kind == 'LR' else [17, 29, 43]
        models = [fit_model(x[t], y[t], kind, chosen, seed) for seed in seeds]
        fitted[kind] = models
        ps = [predict(m, x[e]) for m in models]
        predictions[f'p_{kind}'] = np.mean(ps, axis=0)
        for seed, p in zip(seeds, ps):
            predictions[f'p_{kind}_seed{seed}'] = p
    predictions.to_csv(dest/'predictions.csv', index=False)
    dump(dest/'models.json', fitted)
    dump(dest/'selection.json', selections)
    dump(dest/'COMPLETE.json', dict(unit=unit, seconds=time.monotonic()-started, feature_dimension=x.shape[1], train_count=len(t), eval_count=len(e), protocol_sha256=hashlib.sha256((OUT/'PROTOCOL.md').read_bytes()).hexdigest()))
    print(unit, 'complete', round(time.monotonic()-started, 1), 'seconds', flush=True)


def weights():
    rows, checks = [], []
    for unit in UNITS:
        domain = unit.split('_')[0]
        with np.load(ARCHIVE/unit/'panel.npz') as a:
            data = {k: a[k] for k in a.files}
        models = json.loads((ARCHIVE/unit/'heads.json').read_text())
        published = pd.read_csv(ROOT/'paper_b_readiness/p1_complete/units'/unit/'predictions.csv')
        assert np.array_equal(published.row_index, data['e'])
        names = list(cb.STABLE_CONCEPTS) if domain == 'cbis' else [f'{name}: {state}' for name, states in CONCEPT_STATES.items() for state in states]
        for key, arm, column in [('raw_soft_nested', 'M', 'p_M'), ('raw_gold_nested', 'Reference', 'p_S')]:
            m = models[key]
            coef = np.asarray(m['coef'])[0]/np.asarray(m['scale'])
            intercept = float(m['intercept'][0])-np.dot(coef, m['mean'])
            p = expit(data['val']@coef+intercept)
            error = float(np.max(np.abs(p-published[column])))
            assert error < 1e-7, (unit, arm, error)
            centered = coef.copy()
            adjusted = intercept
            if domain == 'derm':
                start = 0
                for width in data['widths']:
                    offset = coef[start:start+width].mean()
                    centered[start:start+width] -= offset
                    adjusted += offset
                    start += width
            preservation = float(np.max(np.abs(p-expit(data['val']@centered+adjusted))))
            assert preservation < 1e-10
            checks.append(dict(unit=unit, arm=arm, prediction_max_error=error, centering_max_error=preservation))
            for j, name in enumerate(names):
                rows.append(dict(unit=unit, domain=domain, arm=arm, concept=name, coefficient=float(centered[j]), original_coefficient=float(coef[j]), C=m['C'], intercept=adjusted))
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'weight_coefficients.csv', index=False)
    summary = []
    for (domain, arm, concept), f in frame.groupby(['domain', 'arm', 'concept'], sort=False):
        v = f.coefficient.to_numpy()
        summary.append(dict(domain=domain, arm=arm, concept=concept, median=np.median(v), minimum=v.min(), maximum=v.max(), sign_agreement=max(np.mean(v>0), np.mean(v<0)), fits=len(v), crosses_zero=bool(v.min()<0<v.max())))
    pd.DataFrame(summary).to_csv(OUT/'weight_stability.csv', index=False)
    dump(OUT/'weight_prediction_checks.json', checks)
    print('All 60 saved heads reproduced; block centering preserves predictions.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-only', action='store_true')
    parser.add_argument('--unit')
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        weights()
        if not args.weights_only:
            cohorts = {}
            for unit in ([args.unit] if args.unit else UNITS):
                domain = unit.split('_')[0]
                if domain not in cohorts:
                    cohorts[domain] = load_cohort(domain)
                run_unit(unit, cohorts[domain])


if __name__ == '__main__':
    main()
