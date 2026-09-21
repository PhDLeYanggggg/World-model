"""Fixed source-only native cost-head training and out-of-scene diagnostics."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required before Torch import')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scripts import run_m3w_native_nested as nested
from scripts.run_m3w_native_forecast import array_hash, assert_current, immutable_json, json_write
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.evaluation.m3w_native_cost_readout import fixed_interventions, conditional_report
from src.world_model.m3w_native_gain_harm import (
    cost_features, preprocess, fit_ridge, predict_ridge, build_head, fit_neural, predict_neural,
)
from src.world_model.m3w_native_nested import require_source_producer

CONFIG = 'configs/m3w_native_gain_harm_v1.json'
CODE = ('scripts/run_m3w_native_gain_harm.py', 'src/world_model/m3w_native_gain_harm.py',
        'src/evaluation/m3w_native_cost_readout.py', 'tests/test_m3w_native_gain_harm.py')


def load():
    reg = json.loads((ROOT/CONFIG).read_text())
    old, _, data, rows, parent_identity, outer = nested.load()
    if (reg['sites'] != old['sites'] or reg['seeds'] != old['seeds']
            or reg['arms'] != ['ridge', 'mse', 'underharm4']
            or reg['feature_dim'] != 355 or reg['training']['steps'] != 3000
            or any(reg[k] for k in ('threshold_search', 'model_selection', 'risk_calibration',
                                    'original_val_test_readout', 'main_external_readout', 'deployment'))):
        raise ValueError('Fixed exploratory native cost-head design required')
    bindings = dict(parent_identity['source_bindings'])
    def bind(path, expected=None):
        sha = file_digest(ROOT/path)
        if expected is not None and sha != expected:
            raise ValueError('Changed frozen binding: '+path)
        bindings[path] = sha
    for path, sha in reg['bindings'].items():
        bind(path, sha)
    for path in (CONFIG, reg['registration'], *CODE):
        bind(path)
    public = ROOT/old['reports']
    analysis = json.loads((public/'analysis.json').read_text())
    checked = json.loads((public/'verification_with_replay.json').read_text())
    exported = json.loads((public/'materialized_views.json').read_text())
    if (analysis['identity'] != parent_identity or not checked['all_checks_passed']
            or checked['analysis_sha256'] != file_digest(public/'analysis.json')
            or exported['source_analysis_sha256'] != file_digest(public/'analysis.json')
            or not exported['all_checks_passed'] or exported['outer_rows_in_training'] != 0):
        raise ValueError('Verified nested training and physical exports required')
    for t in analysis['training']:
        bind(t['checkpoint'], t['checkpoint_sha256'])
    manifest_path = f"{old['output']}/cost_views.json"
    bind(manifest_path, analysis['views_manifest_sha256'])
    manifest = json.loads((ROOT/manifest_path).read_text())
    expected = {(s, seed) for s in reg['sites'] for seed in reg['seeds']}
    views = {}
    for record in exported['views']:
        site, seed = record['outer_site'], record['seed']
        key = f'{site}_seed{seed}'
        if (site, seed) not in expected or key in views:
            raise ValueError('Unexpected or duplicate training view')
        bind(record['receipt_path'], record['receipt_sha256'])
        meta = json.loads((ROOT/record['receipt_path']).read_text())
        raw = next(v for v in manifest['views'] if (v['outer_site'], v['seed']) == (site, seed))
        if (meta['source_manifest_sha256'] != file_digest(ROOT/manifest_path)
                or meta['training_producers'] != [g['producer'] for g in raw['groups']]
                or meta['outer_prediction'] != outer[f'{site}_native_coordinate_seed{seed}']):
            raise ValueError('Materialized producer lineage differs from verified source')
        for g in raw['groups']:
            require_source_producer(g['producer'], outer_site=site, row_site=g['inner_site'],
                                    roster=reg['sites'], seed=seed)
        for kind in ('inputs', 'targets'):
            bind(meta[kind+'_path'], meta[kind+'_sha256'])
        views[key] = meta
    if len(views) != len(expected):
        raise ValueError('All twelve physical training views required')
    identity = dict(source_bindings=bindings, parent_identity=parent_identity,
        torch=torch.__version__, numpy=np.__version__, architecture=platform.machine(),
        runtime=dict(torch_threads=4, interop_threads=1, num_workers=0),
        scope=reg['scope'])
    return reg, data, rows, identity, views


def training_view(meta, data):
    with np.load(ROOT/meta['inputs_path'], allow_pickle=False) as z:
        if set(z.files) != {'ids', 'prediction'}:
            raise ValueError('Supervision cannot enter the input archive')
        ids, candidate = z['ids'].copy(), z['prediction'].copy()
    expected = np.flatnonzero(data['sites'] != meta['outer_site'])
    np.testing.assert_array_equal(ids, expected)
    if array_hash(ids) != meta['query_ids_sha256']:
        raise ValueError('Changed global query alignment')
    x, same = cost_features(data['geometry'][ids], candidate, data['scale'][ids])
    with np.load(ROOT/meta['targets_path'], allow_pickle=False) as z:
        np.testing.assert_array_equal(z['ids'], ids)
        y = np.column_stack((z['benefit'], z['harm']))
        cv = z['baseline_ade'].copy()
    pr = preprocess(x, y, cv, data['sites'][ids], meta['outer_site'])
    return ids, x, same, y, pr


def common_identity(identity, meta, ids, x, same, y, pr):
    return dict(identity=identity, outer_site=meta['outer_site'], seed=meta['seed'],
        training_sites=pr['training_sites'], training_query_ids_sha256=array_hash(ids),
        features_sha256=array_hash(x, same), targets_sha256=array_hash(y),
        preprocessing_sha256=array_hash(pr['mean'], pr['std'], pr['weights'], pr['constant'],
                                       np.array([pr['cost_scale'], pr['positive_easy_cut'], pr['hard_cut']])),
        training_producers=meta['training_producers'], inputs_sha256=meta['inputs_sha256'],
        targets_archive_sha256=meta['targets_sha256'])


def receipt(path, ti, arm, settings):
    r = json.loads(path.read_text())
    if r['identity'] != ti or r['arm'] != arm or not r['complete']:
        raise ValueError('Changed or incomplete head receipt')
    if file_digest(ROOT/r['checkpoint']) != r['checkpoint_sha256']:
        raise ValueError('Changed head checkpoint')
    if arm != 'ridge' and r['fit']['step'] != settings['steps']:
        raise ValueError('Incomplete fixed neural budget')
    return r


def train(reg, data, identity, views, beat, args):
    root = ROOT/reg['output']
    if args.view and args.view not in views:
        raise ValueError('Unregistered outer/seed view')
    for key, meta in views.items():
        if args.view and key != args.view:
            continue
        ids, x, same, y, pr = training_view(meta, data)
        ti = common_identity(identity, meta, ids, x, same, y, pr)
        for arm in reg['arms']:
            if args.arm and arm != args.arm:
                continue
            directory = root/'trials'/key/arm
            rp = directory/'complete.json'
            if rp.exists():
                receipt(rp, ti, arm, reg['training'])
                beat(state='cached_verified_complete', view=key, arm=arm); continue
            directory.mkdir(parents=True, exist_ok=True)
            beat(state='training_registered_head', view=key, arm=arm, supported_rows=int(pr['known'].sum()))
            if arm == 'ridge':
                started = time.monotonic(); head = fit_ridge(x, y, pr, alpha=reg['ridge_alpha'])
                cp = directory/'checkpoint.pt'
                if cp.exists():
                    raise ValueError('Unreceipted ridge checkpoint: preserve and investigate')
                temp = cp.with_suffix('.tmp')
                torch.save(dict(identity=ti, head=head, preprocess=pr), temp); os.replace(temp, cp)
                fit = dict(seconds=time.monotonic()-started, supported_rows=int(pr['known'].sum()), step=0)
            else:
                _, fit = fit_neural(x, y, data['sites'][ids], same, pr, seed=meta['seed'], arm=arm,
                    settings=reg['training'], identity=ti, directory=directory, resume=args.resume,
                    stop_at=args.stop_at, heartbeat=lambda **v:beat(view=key, arm=arm, **v))
                if not fit['complete']:
                    beat(state='pilot_complete', view=key, arm=arm, fit=fit); return
                cp = directory/'checkpoint.pt'
            assert_current(identity)
            immutable_json(rp, dict(identity=ti, arm=arm, complete=True, fit=fit,
                checkpoint=str(cp.relative_to(ROOT)), checkpoint_sha256=file_digest(cp),
                rows=len(x), supported_rows=int(pr['known'].sum()), feature_dim=x.shape[1],
                positive_easy_cut=pr['positive_easy_cut'], hard_cut=pr['hard_cut'],
                cost_scale=pr['cost_scale'], independent_confirmation=False, risk_calibrated=False))
    beat(state='registered_training_call_complete', risk_calibrated=False)


def evaluate(reg, data, identity, views, beat, verify):
    root, public = ROOT/reg['output'], ROOT/reg['reports']
    # Require all fixed endpoints before opening the new held-source readout.
    records = {}
    for key, meta in views.items():
        for arm in reg['arms']:
            p = root/'trials'/key/arm/'complete.json'
            r = json.loads(p.read_text())
            if r['identity']['identity'] != identity:
                raise ValueError('Mixed head experiment identity')
            records[key, arm] = receipt(p, r['identity'], arm, reg['training'])
    n = len(data['sites']); complete = data['valid'].all(1)
    baseline = data['geometry'][:, 332:356].reshape(-1, 12, 2)
    cv_ade, cv_fde = native_errors(baseline, data['target'], data['valid'], data['scale'])
    kinds = ['constant', 'ridge_raw', 'ridge_clipped', 'mse', 'underharm4']
    stats, arrays, bindings, replay_rows, paired_samplers = [], {}, [], 0, 0
    masks = dict(hard=np.zeros(n, bool), positive_easy=np.zeros(n, bool))
    for key, meta in views.items():
        ids_train, x_train, same_train, y_train, pr = training_view(meta, data)
        ti = common_identity(identity, meta, ids_train, x_train, same_train, y_train, pr)
        for arm in reg['arms']:
            receipt(root/'trials'/key/arm/'complete.json', ti, arm, reg['training'])
        site, seed = meta['outer_site'], meta['seed']
        p_record = meta['outer_prediction']['prediction']
        if file_digest(ROOT/p_record['path']) != p_record['sha256']:
            raise ValueError('Changed frozen outer forecast')
        with np.load(ROOT/p_record['path'], allow_pickle=False) as z:
            ids, candidate = z['ids'].copy(), z['prediction'].copy()
        np.testing.assert_array_equal(ids, np.flatnonzero(data['sites'] == site))
        x, same = cost_features(data['geometry'][ids], candidate, data['scale'][ids])
        ade, fde = native_errors(candidate, data['target'][ids], data['valid'][ids], data['scale'][ids])
        gain = cv_ade[ids]-ade; benefit, harm = np.maximum(gain, 0), np.maximum(-gain, 0)
        outputs = {'constant':np.tile(pr['constant'], (len(ids), 1))}
        outputs['constant'][same] = 0
        ridge = torch.load(ROOT/records[key, 'ridge']['checkpoint'], map_location='cpu', weights_only=False)
        if ridge['identity'] != ti:
            raise ValueError('Wrong ridge producer identity')
        outputs['ridge_raw'] = predict_ridge(ridge['head'], x, same, pr)
        outputs['ridge_clipped'] = np.maximum(outputs['ridge_raw'], 0)
        matched_draws = None
        for arm in ('mse', 'underharm4'):
            state = torch.load(ROOT/records[key, arm]['checkpoint'], map_location='cpu', weights_only=False)
            if (state['identity'] != ti or state['step'] != reg['training']['steps']
                    or int(state['draws'][~pr['known']].sum()) != 0
                    or state['seed'] != seed or state['settings'] != reg['training']):
                raise ValueError('Invalid completed neural head checkpoint')
            for k in ('mean', 'std', 'weights', 'known', 'constant'):
                np.testing.assert_array_equal(state['preprocess'][k], pr[k])
            if int(state['draws'].sum()) != reg['training']['steps']*reg['training']['batch_size']:
                raise ValueError('Wrong head training budget')
            if matched_draws is None:
                matched_draws = state['draws']
            else:
                np.testing.assert_array_equal(matched_draws, state['draws'])
                paired_samplers += 1
            model = build_head(reg['training']['width'], pr, seed); model.load_state_dict(state['model'])
            outputs[arm] = predict_neural(model, x, same, pr)
        path, mp = root/'scores'/(key+'.npz'), root/'scores'/(key+'.json')
        current = dict(ids=ids, **outputs)
        if verify and (not path.exists() or not mp.exists()):
            raise ValueError('Verify cannot create missing scores')
        if mp.exists() and file_digest(path) != json.loads(mp.read_text())['sha256']:
            raise ValueError('Changed cached cost-score bytes')
        nested.write_arrays(path, current)
        meta_score = dict(view=key, ids_sha256=array_hash(ids), sha256=file_digest(path),
            path=str(path.relative_to(ROOT)), head_checkpoints={a:records[key, a]['checkpoint_sha256'] for a in reg['arms']},
            identity=ti, feature_sha256=array_hash(x, same), labels_in_score_file=False)
        immutable_json(mp, meta_score); bindings.append(dict(view=key, path=meta_score['path'], sha256=meta_score['sha256']))
        replay_rows += len(ids)*len(kinds)
        masks['hard'][ids] = cv_ade[ids] >= pr['hard_cut']
        masks['positive_easy'][ids] = (cv_ade[ids] > 0) & (cv_ade[ids] <= pr['positive_easy_cut'])
        for kind, score in outputs.items():
            stats.append(dict(view=key, outer_site=site, seed=seed, kind=kind,
                conditional=conditional_report(score, benefit, harm, same)))
            for rule, use in fixed_interventions(score, same).items():
                k = kind, rule, seed
                if k not in arrays:
                    arrays[k] = dict(ade=np.full(n, np.nan), fde=np.full(n, np.nan),
                                     use=np.zeros(n, bool), seen=np.zeros(n, bool))
                d = arrays[k]
                if d['seen'][ids].any():
                    raise ValueError('Duplicate outer prediction rows')
                d['ade'][ids] = np.where(use, ade, cv_ade[ids]); d['fde'][ids] = np.where(use, fde, cv_fde[ids])
                d['use'][ids] = use; d['seen'][ids] = True
        beat(state='held_source_cost_scores_verified', view=key, rows=len(ids))
    def score(a, b, mask=None):
        if mask is None:
            mask = np.ones(n, bool)
        return paired_scene_metrics(a[mask], b[mask], data['sites'][mask], expected_scenes=reg['sites'],
            dataset='sdd', coordinate_unit='annotation_pixel', bootstrap_resamples=3000, seed=38113)
    summary = {}
    zero = complete & (cv_ade == 0)
    for kind in kinds:
        summary[kind] = {}
        for rule in ('positive_gain', 'harm_fraction_0p1'):
            per_seed = {}
            for seed in reg['seeds']:
                d = arrays[kind, rule, seed]
                if not d['seen'].all():
                    raise ValueError('Missing held-source predictions')
                zero_harm = float(d['ade'][zero].max()) if zero.any() else None
                per_seed[str(seed)] = dict(ADE=score(d['ade'], cv_ade), FDE=score(d['fde'], cv_fde),
                    complete_ADE=score(d['ade'], cv_ade, complete),
                    positive_easy_q25_diagnostic=score(d['ade'], cv_ade, masks['positive_easy']),
                    zero_CV_complete=score(d['ade'], cv_ade, zero),
                    zero_CV_max_absolute_harm=zero_harm,
                    zero_CV_empirical_no_added_error=zero_harm == 0 if zero_harm is not None else None,
                    hard_q75_diagnostic=score(d['ade'], cv_ade, masks['hard']),
                    intervention_rate=float(d['use'].mean()), unknown_selected_rows=int((d['use'] & ~np.isfinite(cv_ade)).sum()))
            mean_ade = np.mean([arrays[kind, rule, s]['ade'] for s in reg['seeds']], axis=0)
            summary[kind][rule] = dict(seeds=per_seed, mean_seed_ADE=score(mean_ade, cv_ade),
                mean_seed_hard=score(mean_ade, cv_ade, masks['hard']),
                mean_seed_positive_easy_diagnostic=score(mean_ade, cv_ade, masks['positive_easy']),
                mean_seed_zero_CV=score(mean_ade, cv_ade, zero))
    result = dict(identity=identity, result_source='fresh_run_native_cost_head_training_and_held_source_diagnostics',
        neural_fits=24, ridge_fits=12, constant_controls=12,
        optimizer_updates=sum(r['fit']['step'] for r in records.values()),
        summed_fit_seconds=sum(r['fit']['seconds'] for r in records.values()),
        indexed_source_rows=n, feature_dim=355, paired_sampler_checks=paired_samplers,
        conditional=stats, summary=summary,
        training=[dict(view=k, arm=a, fit=r['fit'], checkpoint=r['checkpoint'], checkpoint_sha256=r['checkpoint_sha256']) for (k,a),r in records.items()],
        score_archives=bindings, independent_confirmation=False, risk_calibrated=False,
        threshold_search=False, model_selection=False, positive_easy_definition='training_positive_CV_ADE_q25_diagnostic_not_formal_gate',
        deployment=False, stage5c_executed=False, smc_enabled=False)
    assert_current(identity); immutable_json(public/'analysis.json', result)
    if verify:
        immutable_json(public/'verification.json', dict(analysis_sha256=file_digest(public/'analysis.json'),
            cost_score_rows_replayed=replay_rows, head_endpoints_checked=len(records),
            preprocessing_views_recomputed=len(views), all_checks_passed=True, new_training=False))
    beat(state='evaluation_verified' if verify else 'evaluation_complete', heads=len(records))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-only', action='store_true'); p.add_argument('--resume', action='store_true')
    p.add_argument('--evaluate', action='store_true'); p.add_argument('--verify', action='store_true')
    p.add_argument('--view'); p.add_argument('--arm', choices=('ridge', 'mse', 'underharm4'))
    p.add_argument('--stop-at', type=int)
    args = p.parse_args()
    if args.stop_at is not None and (not args.view or args.arm not in ('mse', 'underharm4') or args.evaluate or args.verify):
        raise ValueError('Training pilot must identify a neural view/arm')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg, data, rows, identity, views = load()
    root = ROOT/reg['output']; root.mkdir(parents=True, exist_ok=True)
    lock = (root/'execution.lock').open('a'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    def beat(**values):
        event = dict(pid=os.getpid(), timestamp_unix=time.time(), **values)
        json_write(root/'heartbeat.json', event)
        with (root/'events.jsonl').open('a') as f:
            f.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume last atomic checkpoint')))
    if args.audit_only:
        beat(state='preflight_pass', bindings=len(identity['source_bindings']), views=len(views), rows=len(data['sites'])); return
    immutable_json(root/'identity.json', identity)
    if args.evaluate or args.verify:
        evaluate(reg, data, identity, views, beat, args.verify)
    else:
        train(reg, data, identity, views, beat, args)


if __name__ == '__main__':
    main()
