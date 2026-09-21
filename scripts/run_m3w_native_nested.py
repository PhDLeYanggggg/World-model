"""Build real pair-excluded producers and clean source cost-learning views."""
import argparse
import fcntl
from itertools import combinations
import json
import os
from pathlib import Path
import platform
import signal
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 .venv-pytorch required')
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(key, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scripts import run_m3w_native_forecast as parent
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_native_forecast import fit_trial, predict
from src.world_model.m3w_native_nested import nested_fold_design, require_source_producer, cost_supervision
from src.world_model.m3w_supervised_intervention import build_forecaster

CONFIG = 'configs/m3w_native_nested_v1.json'
CODE = ('scripts/run_m3w_native_nested.py', 'src/world_model/m3w_native_nested.py',
        'tests/test_m3w_native_nested.py')


def producer_record(key, training_sites, excluded, seed, sites):
    return dict(id=key, training_sites=sorted(training_sites), excluded_sites=sorted(excluded),
        preprocessing_fit_sites=sorted(training_sites), seed=seed, parents=[],
        initialization='random_seed', checkpoint_selection_sites=[], calibration_sites=[],
        research_design_exposed_sites=list(sites), objective='native_coordinate')


def load():
    reg = json.loads((ROOT/CONFIG).read_text())
    previous, data, rows, old_identity = parent.load()
    if (reg['parent_config'] != parent.CONFIG or reg['sites'] != previous['sites'] or reg['seeds'] != previous['seeds']
            or reg['objective'] != 'native_coordinate' or reg['steps_per_fit'] != previous['training']['steps']
            or reg['models'] != 18 or reg['optimizer_updates'] != 72000
            or any(reg[k] for k in ('cost_head_training', 'threshold_selection', 'risk_calibration',
                                    'original_val_test_readout', 'main_outer_external_readout', 'deployment'))):
        raise ValueError('Changed fixed source-only prerequisite design')
    bindings = dict(old_identity['source_bindings'])
    def bind(path, sha=None):
        actual = file_digest(ROOT/path)
        if sha is not None and actual != sha:
            raise ValueError('Changed binding: '+path)
        bindings[path] = actual
    for path, sha in reg['bindings'].items():
        bind(path, sha)
    for path in (CONFIG, reg['decision'], *CODE):
        bind(path)
    analysis = json.loads((ROOT/reg['parent_analysis']).read_text())
    if analysis['identity'] != old_identity or analysis['models'] != 24:
        raise ValueError('Changed parent experiment identity')
    preds = {r['trial']:r for r in analysis['predictions']}
    outer = {}
    for site in reg['sites']:
        for seed in reg['seeds']:
            fold, ti, key = parent.specification(previous, data, old_identity, site, 'native_coordinate', seed)
            receipt_path = f"{previous['output']}/trials/{key}/complete.json"
            receipt = parent.completed(ROOT/receipt_path, ti, previous['training'])
            bind(receipt_path); bind(receipt['checkpoint'], receipt['checkpoint_sha256'])
            prediction = preds[key]
            bind(prediction['path'], prediction['sha256'])
            bind(str(Path(prediction['path']).with_suffix('.json')))
            entry = producer_record(key, ti['training_sites'], [site], seed, reg['sites'])
            require_source_producer(entry, outer_site=site, row_site=site, roster=reg['sites'], seed=seed)
            outer[key] = dict(producer=entry, checkpoint=receipt['checkpoint'],
                checkpoint_sha256=receipt['checkpoint_sha256'], prediction=prediction,
                row_ids_sha256=parent.array_hash(fold['held_ids']))
    identity = dict(source_bindings=bindings, registration_sha256=bindings[CONFIG],
        parent_identity=old_identity, population_sha256=old_identity['population_sha256'],
        architecture=platform.machine(), torch=torch.__version__, numpy=np.__version__,
        independent_confirmation=False, runtime=previous['runtime'])
    return reg, previous, data, rows, identity, outer


def specification(reg, data, identity, pair, seed):
    fold = nested_fold_design(data, pair)
    key = '__'.join(sorted(pair))+f'_seed{seed}'
    producer = producer_record(key, sorted(set(data['sites'][fold['train_ids']])), pair, seed, reg['sites'])
    for outer, inner in (pair, tuple(reversed(pair))):
        require_source_producer(producer, outer_site=outer, row_site=inner, roster=reg['sites'], seed=seed)
    ti = dict(identity=identity, producer=producer, seed=seed, excluded_sites=sorted(pair),
        train_ids_sha256=parent.array_hash(fold['train_ids']), held_ids_sha256=parent.array_hash(fold['held_ids']),
        factors_sha256=parent.array_hash(fold['factors']), normalizers=fold['normalizers'])
    return fold, ti, key


def write_arrays(path, arrays):
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != set(arrays):
                raise ValueError('Existing archive schema changed')
            for k, a in arrays.items():
                if z[k].dtype != np.asarray(a).dtype:
                    raise ValueError('Existing archive dtype changed')
                np.testing.assert_array_equal(z[k], a)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp.npz')
        np.savez(tmp, **arrays); os.replace(tmp, path)


def independent_label_check(prediction, baseline, target, valid, scale, labels):
    errors = []
    for p in (prediction, baseline):
        safe = np.where(valid[..., None], target, 0).astype(float)
        dx = p[:, :, 0].astype(float)-safe[:, :, 0]
        dy = p[:, :, 1].astype(float)-safe[:, :, 1]
        d = np.sqrt(dx*dx+dy*dy)*scale[:, None]
        count = valid.sum(1)
        ade = np.divide(np.where(valid, d, 0).sum(1), count,
            out=np.full(len(p), np.nan), where=count > 0)
        fde = np.where(valid[:, -1], d[:, -1], np.nan)
        errors.append((ade, fde))
    for name, actual in [('neural_ade', errors[0][0]), ('neural_fde', errors[0][1]),
                         ('baseline_ade', errors[1][0]), ('baseline_fde', errors[1][1])]:
        np.testing.assert_allclose(labels[name], actual, rtol=1e-12, atol=1e-9, equal_nan=True)
    gain = errors[1][0]-errors[0][0]
    for name, actual in [('gain', gain), ('benefit', np.clip(gain, 0, None)), ('harm', np.clip(-gain, 0, None))]:
        np.testing.assert_allclose(labels[name], actual, rtol=1e-12, atol=1e-9, equal_nan=True)
    np.testing.assert_array_equal(labels['supported_steps'], valid.sum(1))
    np.testing.assert_array_equal(labels['complete_future'], valid.all(1))


def build_cache(reg, previous, data, rows, identity, outer, beat, *, verify=False, replay=False):
    root, public = ROOT/reg['output'], ROOT/reg['reports']
    entries = []
    for pair in combinations(reg['sites'], 2):
        for seed in reg['seeds']:
            fold, ti, key = specification(reg, data, identity, pair, seed)
            receipt = parent.completed(root/'trials'/key/'complete.json', ti, previous['training'])
            entries.append((pair, seed, fold, ti, key, receipt))
    caches, training, replay_rows = {}, [], 0
    for pair, seed, fold, ti, key, receipt in entries:
        ids = fold['held_ids']; model = None
        path = root/'predictions'/(key+'.npz')
        labels_path = root/'supervision'/(key+'.npz')
        meta_path = root/'cache_receipts'/(key+'.json')
        cp = torch.load(ROOT/receipt['checkpoint'], map_location='cpu', weights_only=False)
        if cp['identity'] != ti or cp['step'] != 4000 or cp['draws'][ids].any():
            raise ValueError('Invalid final nested checkpoint')
        np.testing.assert_array_equal(cp['train_ids'], fold['train_ids'])
        np.testing.assert_array_equal(cp['factors'], fold['factors'])
        if int(cp['draws'].sum()) != 4000*previous['training']['batch_size']:
            raise ValueError('Wrong fixed training budget')
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta['identity'] != ti or meta['checkpoint_sha256'] != receipt['checkpoint_sha256']:
                raise ValueError('Wrong cached producer identity')
            for p, sha in [(path, meta['prediction_sha256']), (labels_path, meta['supervision_sha256'])]:
                if file_digest(p) != sha:
                    raise ValueError('Changed cached prediction or label bytes')
            with np.load(path, allow_pickle=False) as z:
                if set(z.files) != {'ids', 'prediction'}:
                    raise ValueError('Labels must not enter the prediction archive')
                np.testing.assert_array_equal(z['ids'], ids); prediction = z['prediction'].copy()
        else:
            if verify:
                raise ValueError('Verify cannot construct missing caches')
            beat(state='predicting_cost_training_rows', trial=key, rows=len(ids))
            torch.manual_seed(seed); model = build_forecaster(previous['architecture'])
            model.load_state_dict(cp['model']); prediction = predict(model, data, ids)
            write_arrays(path, dict(ids=ids, prediction=prediction))
        baseline = data['geometry'][ids, 332:356].reshape(-1, 12, 2)
        labels = cost_supervision(prediction, baseline, data['target'][ids], data['valid'][ids], data['scale'][ids])
        write_arrays(labels_path, dict(ids=ids, **labels))
        independent_label_check(prediction, baseline, data['target'][ids], data['valid'][ids], data['scale'][ids], labels)
        np.testing.assert_array_equal(prediction[rows['static_history'][ids]], baseline[rows['static_history'][ids]])
        meta = dict(identity=ti, checkpoint_sha256=receipt['checkpoint_sha256'],
            prediction_path=str(path.relative_to(ROOT)), prediction_sha256=file_digest(path),
            supervision_path=str(labels_path.relative_to(ROOT)), supervision_sha256=file_digest(labels_path),
            row_ids_sha256=parent.array_hash(ids), rows=len(ids), supported_ADE_rows=int(np.isfinite(labels['gain']).sum()),
            complete_rows=int(labels['complete_future'].sum()), unknown_ADE_rows=int(np.isnan(labels['gain']).sum()))
        parent.immutable_json(meta_path, meta)
        if replay:
            if model is None:
                torch.manual_seed(seed); model = build_forecaster(previous['architecture']); model.load_state_dict(cp['model'])
            for start in sorted({0, ((len(ids)//2)//128)*128, ((len(ids)-1)//128)*128}):
                use = ids[start:start+128]
                np.testing.assert_array_equal(predict(model, data, use), prediction[start:start+128])
                replay_rows += len(use)
        caches[key] = meta
        training.append(dict(trial=key, excluded_sites=list(pair), checkpoint=receipt['checkpoint'],
            checkpoint_sha256=receipt['checkpoint_sha256'], fit=receipt['fit']))
        beat(state='cost_training_cache_verified', trial=key, rows=len(ids))
    views, exposure_checks = [], []
    for site in reg['sites']:
        for seed in reg['seeds']:
            groups, seen = [], np.zeros(len(data['sites']), bool)
            for inner in reg['sites']:
                if site == inner:
                    continue
                _, ti, key = specification(reg, data, identity, [site, inner], seed)
                check = require_source_producer(ti['producer'], outer_site=site, row_site=inner, roster=reg['sites'], seed=seed)
                exposure_checks.append(check)
                ids = np.flatnonzero(data['sites'] == inner)
                if seen[ids].any() or site in set(data['sites'][ids]):
                    raise ValueError('Head-training group overlap')
                seen[ids] = True
                groups.append(dict(inner_site=inner, producer=ti['producer'], cache=caches[key],
                    query_ids_sha256=parent.array_hash(ids), rows=len(ids),
                    supported_ADE_rows=int(data['valid'][ids].any(1).sum())))
            np.testing.assert_array_equal(np.flatnonzero(seen), np.flatnonzero(data['sites'] != site))
            outer_key = f'{site}_native_coordinate_seed{seed}'
            outer_record = outer[outer_key]
            require_source_producer(outer_record['producer'], outer_site=site, row_site=site, roster=reg['sites'], seed=seed)
            views.append(dict(outer_site=site, seed=seed, groups=groups, outer_producer=outer_record,
                training_rows=int(seen.sum()), supported_training_rows=int(data['valid'][seen].any(1).sum()),
                training_query_ids_sha256=parent.array_hash(np.flatnonzero(seen)),
                independent_confirmation=False, risk_head_fitted=False, risk_calibrated=False))
    parent.assert_current(identity)
    manifest = dict(identity=identity, views=views,
        input_contract='pack_geometry_past_only_and_cached_prediction_no_supervision_members',
        label_contract='separate_supervision_archives_not_inference_features',
        geometry_reference='same_bound_parent_population_and_global_query_order',
        risk_tolerance_selected=False, thresholds_selected=False, independent_confirmation=False)
    parent.immutable_json(root/'cost_views.json', manifest)
    report = dict(result_source='fresh_run_nested_training_and_cost_cache_cached_verified_parent_assets',
        identity=identity, completed_fits=len(entries), reused_outer_models=len(outer),
        optimizer_updates=sum(r['fit']['step'] for r in training),
        summed_fit_seconds=sum(r['fit']['seconds'] for r in training),
        training=training, ordered_exclusion_checks=exposure_checks,
        cost_views=len(views), producer_cache_rows=sum(v['rows'] for v in caches.values()),
        supported_cost_rows=sum(v['supported_ADE_rows'] for v in caches.values()),
        unknown_cost_rows=sum(v['unknown_ADE_rows'] for v in caches.values()),
        indexed_source_rows=len(data['sites']), scene_count=len(reg['sites']),
        views_manifest_sha256=file_digest(root/'cost_views.json'),
        source_fitting_exclusion_pass=True, research_design_exposed_sites=reg['sites'],
        independent_confirmation=False, risk_head_fitted=False, risk_calibrated=False,
        new_forecast_gain_claim=False, deployment=False, stage5c_executed=False, smc_enabled=False)
    parent.immutable_json(public/'analysis.json', report)
    if replay:
        parent.immutable_json(public/'verification_with_replay.json', dict(result_source='fresh_run_verification',
            analysis_sha256=file_digest(public/'analysis.json'), completed_fits=len(entries),
            checkpoint_replay_rows=replay_rows, cost_rows_independently_reduced=sum(v['rows'] for v in caches.values()),
            ordered_exclusion_checks=len(exposure_checks), head_views=len(views), all_checks_passed=True,
            independent_confirmation=False, new_training=False))
    beat(state='cache_verified' if verify else 'cache_complete', models=len(entries), views=len(views), replay_rows=replay_rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial'); p.add_argument('--stop-at', type=int); p.add_argument('--resume', action='store_true')
    p.add_argument('--audit-only', action='store_true'); p.add_argument('--cache', action='store_true')
    p.add_argument('--verify', action='store_true'); p.add_argument('--replay', action='store_true')
    args = p.parse_args()
    if args.stop_at is not None and (not args.trial or args.cache or args.verify):
        raise ValueError('Pilot must name a registered training trial')
    if args.replay and not args.verify:
        raise ValueError('Replay is a completed-cache verification action')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg, previous, data, rows, identity, outer = load()
    root, public = ROOT/reg['output'], ROOT/reg['reports']
    root.mkdir(parents=True, exist_ok=True); public.mkdir(parents=True, exist_ok=True)
    lock = (root/'execution.lock').open('a'); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    def beat(**values):
        event = dict(pid=os.getpid(), timestamp_unix=time.time(), **values)
        parent.json_write(root/'heartbeat.json', event)
        with (root/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume last atomic checkpoint')))
    parent.immutable_json(root/'identity.json', identity)
    beat(state='inputs_verified', indexed_rows=len(data['sites']), reused_outer_models=len(outer), expected_new_models=18)
    if args.audit_only:
        return
    if args.cache or args.verify:
        return build_cache(reg, previous, data, rows, identity, outer, beat, verify=args.verify, replay=args.replay)
    trials = [(pair, seed) for pair in combinations(reg['sites'], 2) for seed in reg['seeds']]
    names = {'__'.join(sorted(pair))+f'_seed{seed}' for pair,seed in trials}
    if args.trial and args.trial not in names:
        raise ValueError('Unregistered trial')
    new_updates = 0
    for pair, seed in trials:
        fold, ti, key = specification(reg, data, identity, pair, seed)
        if args.trial and args.trial != key:
            continue
        directory = root/'trials'/key
        if (directory/'complete.json').exists():
            parent.completed(directory/'complete.json', ti, previous['training'])
            beat(state='cached_verified_complete', trial=key); continue
        torch.manual_seed(seed); model = build_forecaster(previous['architecture'])
        beat(state='training_registered_trial', trial=key, training_rows=len(fold['train_ids']), excluded_sites=list(pair))
        fit = fit_trial(model, data, fold, seed=seed, settings=previous['training'], identity=ti,
            directory=directory, resume=args.resume, stop_at=args.stop_at, heartbeat=lambda **v:beat(trial=key, **v))
        new_updates += fit['new_updates']
        if fit['complete']:
            parent.assert_current(identity)
            parent.immutable_json(directory/'complete.json', dict(identity=ti, fit=fit,
                checkpoint=str((directory/'checkpoint.pt').relative_to(ROOT)),
                checkpoint_sha256=file_digest(directory/'checkpoint.pt')))
    beat(state='registered_training_call_complete', new_updates=new_updates, risk_head_fitted=False)


if __name__ == '__main__':
    main()
