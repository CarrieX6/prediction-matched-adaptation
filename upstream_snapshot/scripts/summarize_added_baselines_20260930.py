"""Paired case/patient bootstrap for frozen-feature diagnostic baselines."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'paper_b_readiness/EXPLANATION_BASELINE_EXTERNAL_20260930'
METRICS = ['class_equal_nll', 'natural_nll', 'brier', 'auc']
ARMS = ['S', 'C', 'Cflex', 'M', 'LR', 'MLP']


def metrics(y, p, w):
    """Each row of w is a bootstrap count vector; AUC handles exact ties."""
    p = np.clip(p, 1e-12, 1-1e-12)
    loss = -(y*np.log(p)+(1-y)*np.log1p(-p))
    pos, neg = w*y, w*(1-y)
    np_, nn = pos.sum(1), neg.sum(1)
    ce = .5*((pos@loss)/np_+(neg@loss)/nn)
    natural = (w@loss)/w.sum(1)
    brier = (w@((p-y)**2))/w.sum(1)
    order = np.argsort(p, kind='stable')
    _, starts = np.unique(p[order], return_index=True)
    pg = np.add.reduceat(pos[:, order], starts, axis=1)
    ng = np.add.reduceat(neg[:, order], starts, axis=1)
    auc = (pg*(np.cumsum(ng, axis=1)-.5*ng)).sum(1)/(np_*nn)
    return np.column_stack([ce, natural, brier, auc])


def main():
    frames = []
    for dest in sorted((OUT/'baseline_units').iterdir()):
        assert (dest/'COMPLETE.json').exists(), dest
        direct = pd.read_csv(dest/'predictions.csv')
        old = pd.read_csv(ROOT/'paper_b_readiness/CFLEX_CONTROL_20260917/full/units'/dest.name/'predictions.csv')
        keys = ['domain', 'repetition', 'outer_fold', 'row_index', 'case_id', 'group_id', 'y']
        frame = direct.merge(old[keys+[f'p_{a}' for a in ARMS[:4]]], on=keys, validate='one_to_one')
        assert len(frame) == len(direct) == len(old)
        frames.append(frame)
    assert len(frames) == 30
    data = pd.concat(frames, ignore_index=True)
    data.to_csv(OUT/'paired_predictions.csv', index=False)
    point_rows, contrast_rows, verification = [], [], []
    for domain, data_domain in data.groupby('domain'):
        groups = np.sort(data_domain.group_id.unique())
        mapping = {g: i for i, g in enumerate(groups)}
        panels = [f.sort_values('row_index') for _, f in data_domain.groupby('repetition')]
        nreps = 1 if domain == 'cbis' else 5
        assert len(panels) == nreps
        assert all(len(f) == (655 if domain == 'cbis' else 609) for f in panels)
        point = np.zeros((len(ARMS), 4))
        for panel in panels:
            y = panel.y.to_numpy()
            for j, arm in enumerate(ARMS):
                p = panel[f'p_{arm}'].to_numpy()
                val = metrics(y, p, np.ones((1, len(y))))[0]
                expected = [.5*(log_loss(y[y == 0], p[y == 0], labels=[0, 1])+log_loss(y[y == 1], p[y == 1], labels=[0, 1])), log_loss(y, p), brier_score_loss(y, p), roc_auc_score(y, p)]
                np.testing.assert_allclose(val, expected, atol=1e-12)
                point[j] += val/nreps
                # Independent weighted-library check including duplicated cases.
                w = np.random.default_rng(7).integers(1, 4, len(y))
                actual_auc = metrics(y, p, w[None, :])[0, 3]
                assert abs(actual_auc-roc_auc_score(y, p, sample_weight=w)) < 1e-12
        boot = np.zeros((10000, len(ARMS), 4))
        rng = np.random.default_rng(20260930)
        for offset in range(0, 10000, 250):
            counts = rng.multinomial(len(groups), np.full(len(groups), 1/len(groups)), size=250)
            for panel in panels:
                y = panel.y.to_numpy()
                weights = counts[:, [mapping[g] for g in panel.group_id]]
                assert np.all(weights@y > 0) and np.all(weights@(1-y) > 0)
                for j, arm in enumerate(ARMS):
                    boot[offset:offset+250, j] += metrics(y, panel[f'p_{arm}'].to_numpy(), weights)/nreps
        for j, arm in enumerate(ARMS):
            for k, metric in enumerate(METRICS):
                lo, hi = np.quantile(boot[:, j, k], [.025, .975])
                point_rows.append(dict(domain=domain, arm=arm, metric=metric, estimate=point[j, k], low=lo, high=hi))
        # Positive contrasts favor the candidate: loss_baseline - loss_candidate, AUC_candidate - AUC_baseline.
        for baseline, candidate in [('Cflex', 'M'), ('M', 'LR'), ('M', 'MLP'), ('LR', 'MLP')]:
            b, c = ARMS.index(baseline), ARMS.index(candidate)
            for k, metric in enumerate(METRICS):
                sign = 1 if metric == 'auc' else -1
                values = sign*(boot[:, c, k]-boot[:, b, k])
                lo, hi = np.quantile(values, [.025, .975])
                contrast_rows.append(dict(domain=domain, baseline=baseline, candidate=candidate, metric=metric, estimate=sign*(point[c, k]-point[b, k]), low=lo, high=hi))
        verification.append(dict(domain=domain, cases=len(panels[0]), groups=len(groups), repetitions=nreps, bootstrap_draws=10000, seed=20260930, sklearn_metric_checks='passed'))
    pd.DataFrame(point_rows).to_csv(OUT/'baseline_metrics.csv', index=False)
    pd.DataFrame(contrast_rows).to_csv(OUT/'baseline_paired_contrasts.csv', index=False)
    (OUT/'metric_verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    print(pd.DataFrame(point_rows).pivot(index=['domain', 'arm'], columns='metric', values='estimate').round(6).to_string())
    print(pd.DataFrame(contrast_rows).query("baseline == 'M'").round(6).to_string(index=False))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
