"""Produce EqMotion cross-fitted costs without exposing outer or row sites."""
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
    raise RuntimeError('Native arm64 .venv-pytorch required before Torch import')
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from scripts import run_m3w_native_eqmotion as eq
from scripts import run_m3w_native_nested as nested
from scripts import run_m3w_native_forecast as parent
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_native_forecast import fit_trial, predict
from src.world_model.m3w_native_nested import require_source_producer, cost_supervision
from src.world_model.m3w_supervised_intervention import build_forecaster

CONFIG = 'configs/m3w_eqmotion_nested_v1.json'
CODE = ('scripts/run_m3w_eqmotion_nested.py', 'tests/test_m3w_eqmotion_nested.py')


def specification(reg, data, identity, pair, seed):
    fold, ti, key = nested.specification(reg, data, identity, pair, seed)
    ti['producer']['family'] = 'eqmotion_fixed_head'
    return fold, ti, key


def check_checkpoint(cp, reference, fold, ti, training):
    if cp['identity'] != ti or cp['settings'] != training or cp['seed'] != ti['seed']:
        raise ValueError('Wrong nested EqMotion checkpoint identity or training settings')
    eq.check_matching(cp, reference, fold, training['steps'], training['batch_size'])


def read_prediction(path, ids):
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {'ids', 'prediction'}:
            raise ValueError('Prediction archive must not contain target labels')
        np.testing.assert_array_equal(z['ids'], ids)
        prediction = z['prediction'].copy()
    if prediction.shape != (len(ids), 12, 2) or not np.isfinite(prediction).all():
        raise ValueError('Finite fixed-grid predictions required')
    return prediction


def load():
    reg = json.loads((ROOT/CONFIG).read_text())
    cfg, data, rows, outer_views, eq_identity = eq.load()
    previous = json.loads((ROOT/nested.CONFIG).read_text())
    if (reg['sites'] != cfg['sites'] or reg['seeds'] != cfg['seeds']
            or reg['sites'] != previous['sites'] or reg['seeds'] != previous['seeds']
            or reg['eqmotion_config'] != eq.CONFIG or reg['transformer_nested_config'] != nested.CONFIG
            or reg['models'] != 18 or reg['optimizer_updates'] != 72000
            or reg['training_draws'] != 4608000 or cfg['training']['steps'] != 4000
            or cfg['training']['batch_size'] != 64
            or reg['checkpoint_replay'] != 'three_fixed_128_row_blocks_per_producer_not_full_replay'
            or any(reg[k] for k in ('cost_head_training', 'threshold_selection', 'model_selection',
                'risk_calibration', 'closed_role_readout', 'deployment', 'stage5c_executed', 'smc_enabled'))):
        raise ValueError('Frozen source-only EqMotion prerequisite required')
    old = json.loads((ROOT/reg['transformer_nested_analysis']).read_text())
    prior = json.loads((ROOT/reg['eqmotion_analysis']).read_text())
    if (old['completed_fits'] != 18 or old['optimizer_updates'] != 72000
            or prior['identity'] != eq_identity or len(prior['training']) != 12):
        raise ValueError('Completed matching parent experiments required')
    parent.assert_current(old['identity'])
    bindings = dict(eq_identity['source_bindings'])
    for p, sha in old['identity']['source_bindings'].items():
        if p in bindings and bindings[p] != sha:
            raise ValueError('Parent source identities disagree')
        bindings[p] = sha
    def bind(p, sha=None):
        actual = file_digest(ROOT/p)
        if sha is not None and actual != sha:
            raise ValueError('Changed frozen dependency: '+p)
        bindings[p] = actual
    for p in (CONFIG, reg['registration'], *CODE, reg['transformer_nested_analysis'],
              reg['eqmotion_analysis'],
              'outputs/publication_readiness_2026_09/native_eqmotion_v1/replay.json',
              'outputs/publication_readiness_2026_09/native_nested_v1/verification_with_replay.json'):
        bind(p)
    references = {}
    for pair in combinations(reg['sites'], 2):
        for seed in reg['seeds']:
            fold, ti, key = nested.specification(previous, data, old['identity'], pair, seed)
            p = f"{previous['output']}/trials/{key}/complete.json"
            receipt = parent.completed(ROOT/p, ti, cfg['training'])
            bind(p); bind(receipt['checkpoint'], receipt['checkpoint_sha256'])
            references[key] = receipt
    archives = {r['view']:r for r in prior['archives']}
    outer = {}
    for key, view in outer_views.items():
        ti = eq.trial_identity(eq_identity, key, view)
        path = f"{cfg['output']}/trials/{key}/complete.json"
        receipt = parent.completed(ROOT/path, ti, cfg['training'])
        bind(path); bind(receipt['checkpoint'], receipt['checkpoint_sha256'])
        archive = archives[key]
        bind(archive['path'], archive['sha256'])
        read_prediction(ROOT/archive['path'], view['fold']['held_ids'])
        site, seed = view['site'], view['seed']
        producer = nested.producer_record(key, set(reg['sites'])-{site}, [site], seed, reg['sites'])
        producer['family'] = 'eqmotion_fixed_head'
        require_source_producer(producer, outer_site=site, row_site=site, roster=reg['sites'], seed=seed)
        outer[key] = dict(producer=producer, checkpoint=receipt['checkpoint'],
            checkpoint_sha256=receipt['checkpoint_sha256'], prediction=archive,
            row_ids_sha256=parent.array_hash(view['fold']['held_ids']))
    identity = dict(source_bindings=bindings, registration_sha256=bindings[CONFIG],
        population_sha256=eq_identity['population_sha256'], architecture=platform.machine(),
        model_architecture=cfg['architecture'], training=cfg['training'], runtime=cfg['runtime'],
        torch=torch.__version__, numpy=np.__version__, independent_confirmation=False)
    parent.assert_current(identity)
    return reg, cfg, data, rows, identity, references, outer


def build_cache(reg, cfg, data, identity, references, outer, beat, *, verify=False, replay=False):
    root, public = ROOT/reg['output'], ROOT/reg['reports']
    entries = []
    # Require the complete registered matrix, not a successful seed subset.
    for pair in combinations(reg['sites'], 2):
        for seed in reg['seeds']:
            fold, ti, key = specification(reg, data, identity, pair, seed)
            receipt = parent.completed(root/'trials'/key/'complete.json', ti, cfg['training'])
            entries.append((pair, seed, fold, ti, key, receipt))
    caches, training, replay_rows = {}, [], 0
    for pair, seed, fold, ti, key, receipt in entries:
        ids = fold['held_ids']; model = None
        path = root/'predictions'/(key+'.npz')
        labels_path = root/'supervision'/(key+'.npz')
        meta_path = root/'cache_receipts'/(key+'.json')
        cp = torch.load(ROOT/receipt['checkpoint'], map_location='cpu', weights_only=False)
        ref = torch.load(ROOT/references[key]['checkpoint'], map_location='cpu', weights_only=False)
        check_checkpoint(cp, ref, fold, ti, cfg['training'])
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta['identity'] != ti or meta['checkpoint_sha256'] != receipt['checkpoint_sha256']:
                raise ValueError('Wrong cached producer identity')
            if (file_digest(path) != meta['prediction_sha256']
                    or file_digest(labels_path) != meta['supervision_sha256']):
                raise ValueError('Changed cached prediction or supervision')
            prediction = read_prediction(path, ids)
        else:
            if verify:
                raise ValueError('Verification cannot construct missing caches')
            torch.manual_seed(seed); model = build_forecaster(cfg['architecture'])
            model.load_state_dict(cp['model'])
            parts = []
            for start in range(0, len(ids), 4096):
                beat(state='predicting', trial=key, completed_rows=start, rows=len(ids))
                parts.append(predict(model, data, ids[start:start+4096]))
            prediction = np.concatenate(parts)
            nested.write_arrays(path, dict(ids=ids, prediction=prediction))
        baseline = data['geometry'][ids, 332:356].reshape(-1, 12, 2)
        labels = cost_supervision(prediction, baseline, data['target'][ids], data['valid'][ids], data['scale'][ids])
        nested.write_arrays(labels_path, dict(ids=ids, **labels))
        nested.independent_label_check(prediction, baseline, data['target'][ids], data['valid'][ids], data['scale'][ids], labels)
        # EqMotion does not have the Transformer's exact static-history CV wrapper.
        meta = dict(identity=ti, checkpoint_sha256=receipt['checkpoint_sha256'],
            prediction_path=str(path.relative_to(ROOT)), prediction_sha256=file_digest(path),
            supervision_path=str(labels_path.relative_to(ROOT)), supervision_sha256=file_digest(labels_path),
            row_ids_sha256=parent.array_hash(ids), rows=len(ids),
            supported_ADE_rows=int(np.isfinite(labels['gain']).sum()),
            complete_rows=int(labels['complete_future'].sum()), unknown_ADE_rows=int(np.isnan(labels['gain']).sum()))
        parent.immutable_json(meta_path, meta)
        if replay:
            if model is None:
                torch.manual_seed(seed); model = build_forecaster(cfg['architecture']); model.load_state_dict(cp['model'])
            for start in sorted({0, ((len(ids)//2)//128)*128, ((len(ids)-1)//128)*128}):
                use = ids[start:start+128]
                np.testing.assert_array_equal(predict(model, data, use), prediction[start:start+128])
                replay_rows += len(use)
        caches[key] = meta
        training.append(dict(trial=key, excluded_sites=list(pair), checkpoint=receipt['checkpoint'],
            checkpoint_sha256=receipt['checkpoint_sha256'], fit=receipt['fit']))
        beat(state='producer_cache_verified', trial=key, rows=len(ids))
    views, exclusions = [], []
    for site in reg['sites']:
        for seed in reg['seeds']:
            groups, seen = [], np.zeros(len(data['sites']), bool)
            for inner in reg['sites']:
                if inner == site:
                    continue
                _, ti, key = specification(reg, data, identity, [site, inner], seed)
                exclusions.append(require_source_producer(ti['producer'], outer_site=site,
                    row_site=inner, roster=reg['sites'], seed=seed))
                ids = np.flatnonzero(data['sites'] == inner)
                if seen[ids].any() or site in set(data['sites'][ids]):
                    raise ValueError('Head training groups overlap held source')
                seen[ids] = True
                groups.append(dict(inner_site=inner, producer=ti['producer'], cache=caches[key],
                    query_ids_sha256=parent.array_hash(ids), rows=len(ids),
                    supported_ADE_rows=int(data['valid'][ids].any(1).sum())))
            np.testing.assert_array_equal(np.flatnonzero(seen), np.flatnonzero(data['sites'] != site))
            views.append(dict(outer_site=site, seed=seed, groups=groups,
                outer_producer=outer[f'{site}_seed{seed}'], training_rows=int(seen.sum()),
                supported_training_rows=int(data['valid'][seen].any(1).sum()),
                training_query_ids_sha256=parent.array_hash(np.flatnonzero(seen)),
                independent_confirmation=False, risk_head_fitted=False, risk_calibrated=False))
    parent.assert_current(identity)
    manifest = dict(identity=identity, views=views,
        input_contract='pack_geometry_past_only_and_cached_prediction_no_supervision_members',
        label_contract='separate_supervision_archives_not_inference_features',
        geometry_reference='same_bound_parent_population_and_global_query_order',
        risk_tolerance_selected=False, thresholds_selected=False, independent_confirmation=False)
    parent.immutable_json(root/'cost_views.json', manifest)
    report = dict(result_source='fresh_run_eqmotion_nested_training_cached_verified_outer_predictors',
        identity=identity, completed_fits=len(entries), reused_outer_models=len(outer),
        optimizer_updates=sum(r['fit']['step'] for r in training),
        training_draws=sum(r['fit']['total_draws'] for r in training),
        summed_fit_seconds=sum(r['fit']['seconds'] for r in training),
        training=training, ordered_exclusion_checks=exclusions, cost_views=len(views),
        producer_cache_rows=sum(v['rows'] for v in caches.values()),
        supported_cost_rows=sum(v['supported_ADE_rows'] for v in caches.values()),
        unknown_cost_rows=sum(v['unknown_ADE_rows'] for v in caches.values()),
        indexed_source_rows=len(data['sites']), scene_count=len(reg['sites']),
        views_manifest_sha256=file_digest(root/'cost_views.json'),
        matched_transformer_samplers=len(entries), source_fitting_exclusion_pass=True,
        research_design_exposed_sites=reg['sites'], independent_confirmation=False,
        risk_head_fitted=False, risk_calibrated=False, new_forecast_gain_claim=False,
        deployment=False, stage5c_executed=False, smc_enabled=False)
    parent.immutable_json(public/'analysis.json', report)
    if replay:
        parent.immutable_json(public/'verification_with_replay.json',
            dict(analysis_sha256=file_digest(public/'analysis.json'), completed_fits=len(entries),
                checkpoint_replay_rows=replay_rows, full_checkpoint_replay=False,
                cost_rows_independently_reduced=report['producer_cache_rows'],
                ordered_exclusion_checks=len(exclusions), head_views=len(views), all_checks_passed=True,
                independent_confirmation=False, new_training=False))
    beat(state='cache_verified' if verify else 'cache_complete', models=len(entries), views=len(views))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('audit-only', 'resume', 'cache', 'verify', 'replay'):
        parser.add_argument('--'+name, action='store_true')
    parser.add_argument('--trial'); parser.add_argument('--stop-at', type=int)
    args = parser.parse_args()
    if (sum((args.audit_only, args.cache, args.verify)) > 1 or (args.replay and not args.verify)
            or (args.stop_at is not None and (not args.trial or args.cache or args.verify or args.audit_only))
            or (args.trial and (args.audit_only or args.cache or args.verify))):
        raise ValueError('One phase; pilot names a registered fit, replay verifies existing caches')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg, cfg, data, rows, identity, references, outer = load()
    root = ROOT/reg['output']; root.mkdir(parents=True, exist_ok=True)
    def beat(**values):
        event = dict(pid=os.getpid(), timestamp_unix=time.time(), **values)
        parent.json_write(root/'heartbeat.json', event)
        with (root/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('Resume last atomic checkpoint')))
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        parent.immutable_json(root/'identity.json', identity)
        if args.audit_only:
            beat(state='preflight_pass', bindings=len(identity['source_bindings']), expected_fits=18,
                reused_outer_models=len(outer), matched_transformer_references=len(references))
            return
        if args.cache or args.verify:
            return build_cache(reg, cfg, data, identity, references, outer, beat,
                verify=args.verify, replay=args.replay)
        names = set(references)
        if args.trial and args.trial not in names:
            raise ValueError('Unregistered trial')
        for pair in combinations(reg['sites'], 2):
            for seed in reg['seeds']:
                fold, ti, key = specification(reg, data, identity, pair, seed)
                if args.trial and args.trial != key:
                    continue
                folder = root/'trials'/key
                if (folder/'complete.json').exists():
                    parent.completed(folder/'complete.json', ti, cfg['training'])
                    beat(state='cached_verified_complete', trial=key)
                    continue
                torch.manual_seed(seed); model = build_forecaster(cfg['architecture'])
                beat(state='training_started', trial=key, training_rows=len(fold['train_ids']), excluded_sites=list(pair))
                fit = fit_trial(model, data, fold, seed=seed, settings=cfg['training'], identity=ti,
                    directory=folder, resume=args.resume, stop_at=args.stop_at,
                    heartbeat=lambda **v:beat(trial=key, **v))
                if not fit['complete']:
                    beat(state='pilot_complete_not_full', trial=key, fit=fit)
                    return
                cp = torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=False)
                ref = torch.load(ROOT/references[key]['checkpoint'], map_location='cpu', weights_only=False)
                check_checkpoint(cp, ref, fold, ti, cfg['training'])
                parent.assert_current(identity)
                parent.immutable_json(folder/'complete.json', dict(identity=ti, fit=fit,
                    checkpoint=str((folder/'checkpoint.pt').relative_to(ROOT)),
                    checkpoint_sha256=file_digest(folder/'checkpoint.pt')))
                beat(state='fit_complete_matched_sampler', trial=key, updates=fit['step'])
        beat(state='registered_trial_complete' if args.trial else 'training_matrix_complete',
            models=1 if args.trial else 18, cost_head_fitted=False)


if __name__ == '__main__':
    main()
