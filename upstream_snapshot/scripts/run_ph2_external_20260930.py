"""PH2 transfer of frozen Derm7pt models, with separate audit and outcome gates.

No PH2 observations are used to fit a model, choose a parameter or recalibrate.
The original clinical diagnosis is the endpoint; histology is incomplete.
Local images, model parameters and case-level records are not redistributable.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import re
import sys
import warnings
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts'),
    str(ROOT/'vendor/iclr_gate1a_anchored_mnpd_snapshot_20260904')]
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.fft import dctn
from scipy.special import expit, softmax
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

OUT = ROOT/'paper_b_readiness/EXPLANATION_BASELINE_EXTERNAL_20260930/PH2'
ARCHIVE = ROOT/'results/deadline_nested_20260912'
UNITS = [f'derm_r{r}_f{f}' for r in range(5) for f in range(5)]
ARMS = ['S', 'C', 'Cflex', 'M', 'LR', 'MLP']
PH2_SHA = '3588ec5f0f85736c755065f1713ae760cccdafb5a4e2e5f0e78ed4054ca37a57'
PROTOCOL_SHA = 'dbc4debb22afeb58e832240ed7e3e2e830a72507f0043639ee681742e3473b0a'
CHECKPOINT_SHA = '52cc993c5c5ff962bd0c60931874bc001e7e9b41666a385530f4a036294576be'


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else str(v))+'\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def phash(im):
    a = np.asarray(im.convert('L').resize((32, 32), Image.Resampling.LANCZOS), float)
    d = dctn(a, type=2, norm='ortho')[:8, :8].ravel()
    # Exclude DC from the median and fix its bit to zero.
    bits = d > np.median(d[1:])
    bits[0] = False
    return np.packbits(bits)


def rgb_sha(im):
    return hashlib.sha256(str(im.size).encode()+im.convert('RGB').tobytes()).hexdigest()


def audit(archive):
    assert sha(archive) == PH2_SHA, 'Unexpected source archive: audit as a new input'
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None, 'Archive CRC failure'
        text = z.read('PH2Dataset/PH2_dataset.txt').decode('latin1')
        rows = [re.split(r'\s*\|+\s*', line.strip().strip('|').strip()) for line in text.splitlines() if re.match(r'\|+\s*IMD', line)]
        assert len(rows) == 200 and all(len(row) == 10 for row in rows)
        annotations = {v[0]: v for v in rows}
        assert len(annotations) == 200
        # Read the original workbook directly, without an Excel dependency.
        x = zipfile.ZipFile(io.BytesIO(z.read('PH2Dataset/PH2_dataset.xlsx')))
        ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        shared = [''.join(e.itertext()) for e in ET.fromstring(x.read('xl/sharedStrings.xml')).findall('s:si', ns)]
        sheet = ET.fromstring(x.read('xl/worksheets/sheet1.xml'))
        workbook = {}
        for row in sheet.findall('s:sheetData/s:row', ns):
            cells = {re.sub(r'\d', '', c.attrib['r']): shared[int(c.findtext('s:v', namespaces=ns))] if c.attrib.get('t') == 's' else c.findtext('s:v', '', ns) for c in row}
            name = cells.get('A', '')
            if re.fullmatch(r'IMD\d+', name):
                selected = [i for i, col in enumerate(('C', 'D', 'E')) if cells.get(col, '').strip().upper() == 'X']
                assert len(selected) == 1
                workbook[name] = (selected[0], cells.get('B', '').strip())
        assert set(workbook) == set(annotations)
        originals = [n for n in z.namelist() if re.search(r'/IMD\d+_Dermoscopic_Image/IMD\d+\.bmp$', n)]
        assert len(originals) == 200
        for name in sorted(originals):
            case = Path(name).stem
            a = annotations[case]
            clinical, hist = int(a[2]), a[1].strip()
            assert workbook[case] == (clinical, hist), f'TXT/XLSX disagreement: {case}'
            data = z.read(name)
            dest = OUT/'private/images'/f'{case}.bmp'
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                assert dest.read_bytes() == data
            else:
                dest.write_bytes(data)
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                records.append(dict(case_id=case, clinical_diagnosis=clinical, y=int(clinical == 2), histological_diagnosis=hist,
                    archive_member=name, image=str(dest.relative_to(ROOT)), width=im.width, height=im.height, mode=im.mode,
                    file_sha256=hashlib.sha256(data).hexdigest(), rgb_sha256=rgb_sha(im)))
        for name in ('PH2_dataset.txt', 'PH2_dataset.xlsx', 'Readme.txt'):
            (OUT/'private'/name).write_bytes(z.read('PH2Dataset/'+name))
    frame = pd.DataFrame(records)
    frame.to_csv(OUT/'private/cases.csv', index=False)
    assert frame.file_sha256.nunique() == frame.rgb_sha256.nunique() == 200
    # Include every available Derm7pt image (clinical and dermoscopic; all splits).
    # No held-out diagnosis metadata are read for this overlap audit.
    source_root = ROOT/'data/Derm7pt/release_v0/images'
    source = sorted(p for p in source_root.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
    assert source
    source_records, source_hashes = [], []
    for path in source:
        with Image.open(path) as im:
            im.load()
            source_records.append(dict(path=str(path.relative_to(ROOT)), rgb_sha256=rgb_sha(im), file_sha256=sha(path)))
            source_hashes.append(phash(im))
    hashes = np.stack(source_hashes)
    pop = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    pairs, nearest = [], []
    for row in records:
        with Image.open(ROOT/row['image']) as im:
            variants = [im.rotate(a, expand=True) for a in (0, 90, 180, 270)]
            variants += [ImageOps.mirror(v) for v in variants.copy()]
            distances = np.stack([pop[np.bitwise_xor(hashes, phash(v))].sum(axis=1) for v in variants])
        best = distances.min(axis=0)
        for j in np.argsort(best, kind='stable')[:5]:
            nearest.append(dict(case_id=row['case_id'], derm_image=source_records[j]['path'], phash_distance=int(best[j]), transform_index=int(distances[:, j].argmin())))
        for j in np.flatnonzero(best <= 10):
            pairs.append(dict(case_id=row['case_id'], derm_image=source_records[j]['path'], phash_distance=int(best[j]), transform_index=int(distances[:, j].argmin())))
    pd.DataFrame(source_records).to_csv(OUT/'private/derm7pt_image_hashes.csv', index=False)
    pd.DataFrame(nearest).to_csv(OUT/'private/nearest_image_pairs.csv', index=False)
    pd.DataFrame(pairs, columns=['case_id', 'derm_image', 'phash_distance', 'transform_index']).to_csv(OUT/'private/overlap_candidates.csv', index=False)
    exact = set(frame.rgb_sha256) & {r['rgb_sha256'] for r in source_records}
    report = dict(archive=str(archive), archive_sha256=sha(archive), archive_size=Path(archive).stat().st_size, crc_pass=True,
        n_images=len(frame), clinical_counts=frame.clinical_diagnosis.value_counts().sort_index().to_dict(),
        binary_counts=frame.y.value_counts().sort_index().to_dict(), txt_xlsx_diagnosis_agreement=True,
        histology_available=int(frame.histological_diagnosis.ne('').sum()), histology_available_by_clinical=frame.assign(available=frame.histological_diagnosis.ne('')).groupby('clinical_diagnosis').available.sum().to_dict(),
        dimensions=frame.groupby(['width', 'height']).size().reset_index(name='n').to_dict('records'),
        image_modes=frame['mode'].value_counts().to_dict(), unique_rgb=frame.rgb_sha256.nunique(),
        derm7pt_images_screened=len(source), exact_rgb_overlaps=len(exact), phash_candidates=len(pairs),
        phash_screen='63 variable bits; 8 rotations/reflections; <=10 Hamming distance requires review',
        overlap_screen_pass=(not exact and not pairs), official_origin_verified=False,
        limitations=['Image checks do not prove patient independence.', 'Patient identifiers are not provided.', 'Clinical diagnosis is the endpoint; only 41 histological fields are nonblank.', 'Archive origin remains awaiting author provenance or official binary match.'])
    dump(OUT/'archive_audit.json', report)
    print(json.dumps(report, indent=2), flush=True)


def probe_predict(model, x):
    p = np.zeros((len(x), model['n_classes']))
    if model['constant'] is not None:
        p[:, model['constant']] = 1.
    else:
        z = x@np.asarray(model['coef']).T+np.asarray(model['intercept'])
        partial = np.column_stack([1-expit(z[:, 0]), expit(z[:, 0])]) if len(model['classes']) == 2 else softmax(z, axis=1)
        p[:, np.asarray(model['classes'], int)] = partial
    return p/p.sum(axis=1, keepdims=True)


def readout(unit, q, x):
    from run_added_baselines_20260930 import predict
    heads = json.loads((ARCHIVE/unit/'heads.json').read_text())
    def logit(h):
        return (((q-np.asarray(h['mean']))/np.asarray(h['scale']))@np.asarray(h['coef']).T+np.asarray(h['intercept'])).ravel()
    z = logit(heads['raw_gold_nested'])
    result = {'S': expit(z), 'M': expit(logit(heads['raw_soft_nested']))}
    cal = json.loads((ROOT/'paper_b_readiness/p1_complete/units'/unit/'calibrators.json').read_text())['calibrators']['C']
    result['C'] = expit(cal['slope']*z+cal['intercept'])
    folder = ROOT/'paper_b_readiness/CFLEX_CONTROL_20260917/full/units'/unit
    basis = json.loads((folder/'basis.json').read_text())['T_final_basis']
    models = [m for m in json.loads((folder/'models.json').read_text()) if m['role'] == 'Cflex_final_T']
    assert len(models) == 1
    m, k = models[0], np.asarray(basis['knots'])
    nonlinear = [(np.maximum(z-k[j], 0)**3-(k[4]-k[j])/(k[4]-k[3])*np.maximum(z-k[3], 0)**3+(k[3]-k[j])/(k[4]-k[3])*np.maximum(z-k[4], 0)**3)/(k[4]-k[0])**2 for j in range(3)]
    v = (np.column_stack([z, *nonlinear])-basis['train_mean'])/basis['train_population_sd']
    result['Cflex'] = expit((v@np.asarray(m['coef']).T+np.asarray(m['intercept'])).ravel())
    direct = json.loads((OUT.parent/'baseline_units'/unit/'models.json').read_text())
    for arm in ('LR', 'MLP'):
        result[arm] = np.mean([predict(m, x) for m in direct[arm]], axis=0)
    return result


def replay():
    from run_deadline_nested_20260912 import derm_load, _config_from_contract_mapping
    from paper2_mpjc.pilot import _BalancedClassifier, _seed, case_kfold_splits
    co = derm_load()
    cfg = _config_from_contract_mapping(json.loads((ROOT/'configs/cf_mpjc_d1_derm7pt.json').read_text()))
    reports = []
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        warnings.filterwarnings('ignore', category=FutureWarning, message=".*multi_class.*")
        for unit in UNITS:
            _, r, f = unit.split('_')
            r, f = int(r[1:]), int(f[1:])
            with np.load(ARCHIVE/unit/'panel.npz') as a:
                t, e, target_q = a['t'], a['e'], a['val']
                assert np.array_equal(co.case_id, a['ids'])
            t0, e0 = case_kfold_splits(len(co.y), cfg.outer_folds, _seed(cfg.seed, 'outer', r))[f]
            assert np.array_equal(t, t0) and np.array_equal(e, e0)
            fitted, probs = [], []
            for j, (name, states) in enumerate(co.concept_states.items()):
                seed = _seed(cfg.seed, 'outer_validation_probe_fit_all', r, f, j, name)
                classifier = _BalancedClassifier(len(states), cfg.probe_c, cfg.max_iter, seed).fit(co.x[t], co.concepts[name][t])
                m = dict(name=name, states=states, n_classes=len(states), constant=classifier.constant_, C=cfg.probe_c, seed=seed)
                if classifier.model_ is not None:
                    m.update(coef=classifier.model_.coef_, intercept=classifier.model_.intercept_, classes=classifier.model_.classes_)
                # Check serialized-parameter inference against the library object.
                p = probe_predict(m, co.x[e])
                assert np.max(np.abs(p-classifier.predict_proba(co.x[e]))) < 1e-12
                fitted.append(m)
                probs.append(p)
            q = np.concatenate(probs, axis=1)
            q_error = float(np.max(np.abs(q-target_q)))
            assert q_error < 1e-10, f'Concept replay mismatch {unit}: {q_error}'
            pred = readout(unit, q, co.x[e])
            published = pd.read_csv(ROOT/'paper_b_readiness/p1_complete/units'/unit/'predictions.csv').set_index('case_id').loc[co.case_id[e]]
            spline = pd.read_csv(ROOT/'paper_b_readiness/CFLEX_CONTROL_20260917/full/units'/unit/'predictions.csv').set_index('case_id').loc[co.case_id[e]]
            direct = pd.read_csv(OUT.parent/'baseline_units'/unit/'predictions.csv').set_index('case_id').loc[co.case_id[e]]
            errors = {arm: float(np.max(np.abs(pred[arm]-(spline if arm == 'Cflex' else direct if arm in ('LR', 'MLP') else published)['p_'+arm].to_numpy()))) for arm in ARMS}
            assert max(errors.values()) < 1e-10, errors
            dump(OUT/'private/recovered_models'/f'{unit}.json', dict(unit=unit, training_case_ids=co.case_id[t], models=fitted, panel_sha256=sha(ARCHIVE/unit/'panel.npz')))
            reports.append(dict(unit=unit, concept_max_abs_error=q_error, readout_errors=errors))
            print(unit, 'replayed', q_error, max(errors.values()), flush=True)
    dump(OUT/'model_replay_verification.json', dict(pass_all=True, n_outer_models=25, n_concept_models=175,
        tolerance=1e-10, units=reports, source_feature_sha256=sha(ROOT/'data/Derm7pt/derm7pt_biomedclip_dev609_only_v1_20260908.npz'),
        protocol_sha256=sha(OUT.parent/'PROTOCOL.md'), code_sha256=sha(Path(__file__)), external_data_used_in_fit=False))


def features(device_name, checkpoint):
    """Label-blind preparation, after a fixed source-image replay check."""
    import torch
    import timm
    import torchvision
    from extract_derm7pt_biomedclip_features import BiomedCLIPVisual, _encode
    assert sha(OUT.parent/'PROTOCOL.md') == PROTOCOL_SHA
    assert sha(checkpoint) == CHECKPOINT_SHA
    assert json.loads((OUT/'model_replay_verification.json').read_text())['pass_all']
    assert json.loads((OUT/'overlap_manual_review.json').read_text())['screen_pass']
    output = OUT/'private/ph2_biomedclip_features.npz'
    assert not output.exists(), 'Refusing to replace extracted features'
    source_path = ROOT/'data/Derm7pt/derm7pt_biomedclip_dev609_only_v1_20260908.npz'
    meta = pd.read_csv(ROOT/'data/Derm7pt/derm7pt_dev609_only_v1_20260908.csv').set_index('metadata_index')
    with np.load(source_path) as old:
        positions = np.argsort(old['metadata_index'], kind='stable')[:12]
        ids, target = old['metadata_index'][positions], old['derm_embeddings'][positions]
    source_images = [ROOT/'data/Derm7pt/release_v0/images'/str(meta.loc[i, 'derm']) for i in ids]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device_name)
    model = BiomedCLIPVisual(checkpoint).float().eval().to(device)
    actual = _encode(model, source_images, device, 16)
    error = float(np.max(np.abs(actual-target)))
    report = dict(pass_source_replay=error <= 1e-5, source_metadata_indices=ids,
        max_abs_error=error, prespecified_tolerance=1e-5,
        source_feature_sha256=sha(source_path), checkpoint_sha256=sha(checkpoint),
        protocol_sha256=PROTOCOL_SHA, code_sha256=sha(Path(__file__)),
        extractor_sha256=sha(ROOT/'scripts/extract_derm7pt_biomedclip_features.py'),
        torch=torch.__version__, torchvision=torchvision.__version__, timm=timm.__version__,
        device=device_name, batch_size=16, dtype='float32',
        external_labels_used=False, fitted_parameters_changed=False)
    dump(OUT/'encoder_replay_verification.json', report)
    assert report['pass_source_replay'], 'Source image encoder did not reproduce archived embeddings'
    # Do not load diagnosis columns in this stage.
    cases = pd.read_csv(OUT/'private/cases.csv', usecols=['case_id', 'image', 'file_sha256'])
    paths = [ROOT/p for p in cases.image]
    assert all(sha(p) == h for p, h in zip(paths, cases.file_sha256, strict=True))
    embeddings = _encode(model, paths, device, 16)
    assert embeddings.shape == (200, 512) and np.isfinite(embeddings).all()
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1, atol=2e-7)
    np.savez_compressed(output, case_id=cases.case_id.to_numpy(dtype=str), embeddings=embeddings,
        checkpoint_sha256=np.asarray(CHECKPOINT_SHA), archive_sha256=np.asarray(PH2_SHA))
    dump(OUT/'feature_preparation.json', dict(pass_all=True, n_images=200, dimensions=512,
        feature_sha256=sha(output), archive_sha256=PH2_SHA, encoder_receipt_sha256=sha(OUT/'encoder_replay_verification.json'),
        labels_used=False, predictions_generated=False, outcome_metrics_computed=False,
        completed_utc=datetime.now(timezone.utc).isoformat()))
    print('Prepared 200 label-blind PH2 embeddings; no external predictions or outcomes computed.', flush=True)


def external_metrics(y, p, weights):
    """Clip only NLL inputs; retain raw probabilities for Brier and AUC ties."""
    clipped = np.clip(p, 1e-12, 1-1e-12)
    loss = -(y*np.log(clipped)+(1-y)*np.log1p(-clipped))
    pos, neg = weights*y, weights*(1-y)
    np_, nn = pos.sum(1), neg.sum(1)
    ce = .5*((pos@loss)/np_+(neg@loss)/nn)
    natural = (weights@loss)/weights.sum(1)
    brier = (weights@((p-y)**2))/weights.sum(1)
    order = np.argsort(p, kind='stable')
    _, starts = np.unique(p[order], return_index=True)
    pg = np.add.reduceat(pos[:, order], starts, axis=1)
    ng = np.add.reduceat(neg[:, order], starts, axis=1)
    auc = (pg*(np.cumsum(ng, axis=1)-.5*ng)).sum(1)/(np_*nn)
    return np.column_stack([ce, natural, brier, auc])


def evaluate():
    """Run only after documented provenance, overlap, model and encoder checks."""
    from summarize_added_baselines_20260930 import METRICS
    from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
    assert sha(OUT.parent/'PROTOCOL.md') == PROTOCOL_SHA
    origin_path = OUT/'official_origin_receipt.json'
    assert origin_path.exists(), 'Missing acquisition provenance; external outcome evaluation remains gated'
    origin = json.loads(origin_path.read_text())
    assert origin['archive_sha256'] == PH2_SHA and origin['official_origin_verified'] is True
    assert origin['evidence'] and origin['verification_method']
    overlap = json.loads((OUT/'overlap_manual_review.json').read_text())
    assert overlap['archive_sha256'] == PH2_SHA and overlap['screen_pass'] is True
    assert overlap['candidate_file_sha256'] == sha(OUT/'private/overlap_candidates.csv')
    assert json.loads((OUT/'model_replay_verification.json').read_text())['pass_all']
    assert json.loads((OUT/'encoder_replay_verification.json').read_text())['pass_source_replay']
    prep = json.loads((OUT/'feature_preparation.json').read_text())
    feature_path = OUT/'private/ph2_biomedclip_features.npz'
    assert prep['pass_all'] and prep['feature_sha256'] == sha(feature_path)
    lock = json.loads((OUT/'implementation_lock.json').read_text())
    assert lock['runner_sha256'] == sha(Path(__file__))
    assert lock['metrics_sha256'] == sha(ROOT/'scripts/summarize_added_baselines_20260930.py')
    assert not (OUT/'private/ensemble_predictions.csv').exists(), 'Refusing repeated external evaluation'
    cases = pd.read_csv(OUT/'private/cases.csv')
    with np.load(feature_path) as f:
        assert np.array_equal(f['case_id'], cases.case_id.to_numpy())
        x = f['embeddings'].astype(np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True).clip(1e-12)
    y = cases.y.to_numpy(dtype=int)
    assert len(y) == 200 and np.array_equal(np.bincount(y), [160, 40])
    all_predictions = []
    for unit in UNITS:
        recovered = json.loads((OUT/'private/recovered_models'/f'{unit}.json').read_text())
        assert recovered['panel_sha256'] == sha(ARCHIVE/unit/'panel.npz')
        q = np.concatenate([probe_predict(m, x) for m in recovered['models']], axis=1)
        predictions = readout(unit, q, x)
        all_predictions.append(np.column_stack([predictions[a] for a in ARMS]))
    per_model = np.stack(all_predictions)
    assert per_model.shape == (25, 200, 6) and np.isfinite(per_model).all()
    assert np.all((per_model >= 0) & (per_model <= 1))
    probabilities = per_model.mean(axis=0)
    np.savez_compressed(OUT/'private/individual_model_predictions.npz', probabilities=per_model,
        units=np.asarray(UNITS), arms=np.asarray(ARMS), case_id=cases.case_id.to_numpy(dtype=str))
    frame = cases[['case_id', 'y']].copy()
    for j, arm in enumerate(ARMS):
        frame['p_'+arm] = probabilities[:, j]
    frame.to_csv(OUT/'private/ensemble_predictions.csv', index=False)
    points, boots = [], np.zeros((10000, len(ARMS), 4))
    for j, arm in enumerate(ARMS):
        p = probabilities[:, j]
        clipped = np.clip(p, 1e-12, 1-1e-12)
        point = external_metrics(y, p, np.ones((1, 200)))[0]
        expected = [.5*(log_loss(y[y == 0], clipped[y == 0], labels=[0, 1])+log_loss(y[y == 1], clipped[y == 1], labels=[0, 1])),
            log_loss(y, clipped), brier_score_loss(y, p), roc_auc_score(y, p)]
        np.testing.assert_allclose(point, expected, atol=1e-12)
        weights = np.random.default_rng(7).integers(1, 4, len(y))
        weighted = external_metrics(y, p, weights[None, :])[0]
        np.testing.assert_allclose(weighted[1:], [log_loss(y, clipped, sample_weight=weights),
            brier_score_loss(y, p, sample_weight=weights), roc_auc_score(y, p, sample_weight=weights)], atol=1e-12)
        points.append(point)
    points = np.asarray(points)
    rng = np.random.default_rng(20260930)
    for offset in range(0, 10000, 250):
        weights = rng.multinomial(200, np.full(200, 1/200), size=250)
        assert np.all(weights@y > 0) and np.all(weights@(1-y) > 0)
        for j in range(len(ARMS)):
            boots[offset:offset+250, j] = external_metrics(y, probabilities[:, j], weights)
    metric_rows, contrast_rows = [], []
    for j, arm in enumerate(ARMS):
        for k, metric in enumerate(METRICS):
            low, high = np.quantile(boots[:, j, k], [.025, .975])
            metric_rows.append(dict(arm=arm, metric=metric, estimate=points[j, k], low=low, high=high))
    # Positive values favor M for every metric. All five contrasts are retained.
    m = ARMS.index('M')
    for baseline in [a for a in ARMS if a != 'M']:
        b = ARMS.index(baseline)
        for k, metric in enumerate(METRICS):
            direction = 1 if metric == 'auc' else -1
            low, high = np.quantile(direction*(boots[:, m, k]-boots[:, b, k]), [.025, .975])
            contrast_rows.append(dict(baseline=baseline, candidate='M', metric=metric,
                estimate=direction*(points[m, k]-points[b, k]), low=low, high=high))
    pd.DataFrame(metric_rows).to_csv(OUT/'external_metrics.csv', index=False)
    pd.DataFrame(contrast_rows).to_csv(OUT/'external_paired_contrasts.csv', index=False)
    dump(OUT/'external_verification.json', dict(pass_all=True, n_cases=200, n_positive=40,
        n_source_models=25, bootstrap_draws=10000, seed=20260930, resampling_unit='image/case',
        patient_identifiers_available=False, sklearn_metric_checks='passed',
        ensemble='unweighted mean of 25 source model probabilities', external_fitting=False,
        loss_probability_clip=1e-12, intervals='percentile; conditional on fixed fitted models',
        official_origin_receipt_sha256=sha(origin_path), implementation_lock_sha256=sha(OUT/'implementation_lock.json'),
        completed_utc=datetime.now(timezone.utc).isoformat()))
    print(pd.DataFrame(metric_rows).pivot(index='arm', columns='metric', values='estimate').round(6).to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['audit', 'replay', 'features', 'evaluate'])
    parser.add_argument('--archive', type=Path, default=Path('/home/Datasets/Medical/Medical_Datasets/PH2Dataset.zip'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--checkpoint', type=Path, default=Path('/home/xzf/.cache/huggingface/hub/models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/blobs')/CHECKPOINT_SHA)
    args = parser.parse_args()
    if args.stage == 'audit':
        audit(args.archive)
    elif args.stage == 'replay':
        replay()
    elif args.stage == 'features':
        features(args.device, args.checkpoint)
    else:
        with threadpool_limits(limits=1):
            evaluate()


if __name__ == '__main__':
    main()
