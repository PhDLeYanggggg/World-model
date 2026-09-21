"""Registered full-source matched native/past-normalized forecast comparison."""
import argparse
import fcntl
import hashlib
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
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_native_metrics import native_errors, paired_scene_metrics
from src.evaluation.m3w_source_population import scene_mean, safe_gain
from src.world_model.m3w_supervised_intervention import build_forecaster
from src.world_model.m3w_native_forecast import pack_geometry, fold_design, fit_trial, predict

CONFIG = 'configs/m3w_native_forecast_v1.json'
CODE = ('src/world_model/m3w_native_forecast.py', 'scripts/run_m3w_native_forecast.py',
        'tests/test_m3w_native_forecast.py')


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def immutable_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Existing immutable evidence differs: '+str(path))
    else:
        json_write(path, value)


def array_hash(*values):
    digest = hashlib.sha256()
    for value in values:
        a = np.ascontiguousarray(value)
        digest.update(str(a.dtype).encode()); digest.update(str(a.shape).encode()); digest.update(a.tobytes())
    return digest.hexdigest()


def load():
    reg = json.loads((ROOT/CONFIG).read_text())
    if (reg['sites'] != ['coupa', 'deathCircle', 'gates', 'hyang'] or reg['seeds'] != [17, 29, 43]
            or reg['objectives'] != ['past_normalized', 'native_coordinate']
            or reg['model_selection'] or reg['threshold_selection']
            or reg['original_val_test_readout'] or reg['main_outer_external_readout']):
        raise ValueError('Fixed registered matrix and source-only roles required')
    hashes = {}
    def verify(path, expected=None):
        sha = file_digest(ROOT/path)
        if expected is not None and expected != sha:
            raise ValueError('Changed registered dependency: '+path)
        hashes[path] = sha
    for path in (CONFIG, reg['decision'], *CODE):
        verify(path)
    for path, sha in reg['bindings'].items():
        verify(path, sha)
    audit = json.loads((ROOT/reg['population_report']).read_text())
    for path, sha in audit['evidence_hashes'].items():
        verify(path, sha)
    verify(audit['row_archive']['path'], audit['row_archive']['sha256'])
    with np.load(ROOT/audit['row_archive']['path'], allow_pickle=False) as z:
        rows = {k:z[k] for k in z.files}
    source_cfg = json.loads((ROOT/'configs/m3w_source_population_v1.json').read_text())
    manifest_path = ROOT/source_cfg['input_manifest']
    manifest = json.loads(manifest_path.read_text())
    pieces = {k:[] for k in ('geometry', 'target', 'valid', 'scale')}
    recordings = []
    for receipt in manifest['records']:
        name = receipt['recording']
        if name.split('/')[0] not in reg['sites']:
            continue
        if receipt['original_split'] != 'train' or receipt['data_role'] != 'supervised_auxiliary_training':
            raise ValueError('Unapproved recording role')
        local = np.flatnonzero(rows['recordings'] == name)
        keys = np.load(manifest_path.parent/name/'query_keys.npy', allow_pickle=False)
        np.testing.assert_array_equal(rows['frames'][local], keys[:, 0])
        np.testing.assert_array_equal(rows['tracks'][local], [f'{name}:{int(k)}' for k in keys[:, 1]])
        for key in pieces:
            pieces[key].append(np.load(manifest_path.parent/name/(key+'.npy'), allow_pickle=False))
        recordings.extend([name]*receipt['rows'])
    data = {k:np.concatenate(v) for k,v in pieces.items()}
    np.testing.assert_array_equal(recordings, rows['recordings'])
    np.testing.assert_array_equal(data['valid'].sum(1), rows['valid_count'])
    np.testing.assert_array_equal(data['scale'], rows['scale'])
    data.update(sites=rows['sites'], recordings=rows['recordings'], tracks=rows['tracks'], frames=rows['frames'])
    if len(data['sites']) != 175756 or set(data['sites']) != set(reg['sites']):
        raise ValueError('Full source population mismatch')
    for start in range(0, len(data['sites']), 4096):
        pack_geometry(data['geometry'][start:start+4096])
    identity = dict(registration_sha256=file_digest(ROOT/CONFIG), source_bindings=hashes,
        torch=torch.__version__, numpy=np.__version__, architecture=platform.machine(),
        runtime=reg['runtime'], population_sha256=array_hash(data['recordings'], data['tracks'], data['frames']))
    return reg, data, rows, identity


def specification(reg, data, identity, site, objective, seed):
    fold = fold_design(data, site, objective)
    ti = dict(identity=identity, held_site=site, objective=objective, seed=seed,
        training_sites=sorted(set(data['sites'][fold['train_ids']])),
        held_ids_sha256=array_hash(fold['held_ids']), train_ids_sha256=array_hash(fold['train_ids']),
        factors_sha256=array_hash(fold['factors']), normalizers=fold['normalizers'])
    return fold, ti, f'{site}_{objective}_seed{seed}'


def assert_current(identity):
    for path, sha in identity['source_bindings'].items():
        if file_digest(ROOT/path) != sha:
            raise ValueError('Dependency changed during experiment: '+path)


def completed(path, identity, settings):
    if not path.exists():
        raise ValueError('All registered training endpoints must finish before evaluation')
    r = json.loads(path.read_text())
    if r['identity'] != identity or not r['fit']['complete'] or r['fit']['step'] != settings['steps']:
        raise ValueError('Changed or incomplete trial receipt')
    if file_digest(ROOT/r['checkpoint']) != r['checkpoint_sha256']:
        raise ValueError('Changed final checkpoint')
    return r


def evaluate(reg, data, rows, identity, root, reports, heartbeat, verify=False):
    entries = []
    for objective in reg['objectives']:
        for seed in reg['seeds']:
            for site in reg['sites']:
                fold, ti, key = specification(reg, data, identity, site, objective, seed)
                receipt = completed(root/'trials'/key/'complete.json', ti, reg['training'])
                entries.append((objective, seed, site, fold, ti, key, receipt))
    baseline = data['geometry'][:, 332:356].reshape(-1, 12, 2)
    cv_ade, cv_fde = native_errors(baseline, data['target'], data['valid'], data['scale'])
    np.testing.assert_allclose(cv_ade, rows['ade'][:, 1]*data['scale'], rtol=1e-12, atol=1e-10, equal_nan=True)
    costs = {}; pred_receipts = []; train_receipts = []
    for objective, seed, site, fold, ti, key, receipt in entries:
        path = root/'predictions'/(key+'.npz'); meta_path = path.with_suffix('.json')
        ids = fold['held_ids']
        if path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta['identity'] != ti or meta['checkpoint_sha256'] != receipt['checkpoint_sha256'] or file_digest(path) != meta['sha256']:
                raise ValueError('Cached prediction identity changed')
            with np.load(path, allow_pickle=False) as z:
                np.testing.assert_array_equal(z['ids'], ids); p = z['prediction'].copy()
        else:
            if verify:
                raise ValueError('Verify mode cannot create missing predictions')
            heartbeat(state='predicting_registered_held_source', trial=key, rows=len(ids))
            torch.manual_seed(seed); model = build_forecaster(reg['architecture'])
            cp = torch.load(ROOT/receipt['checkpoint'], map_location='cpu', weights_only=False)
            if cp['identity'] != ti or cp['step'] != reg['training']['steps']:
                raise ValueError('Checkpoint state differs from receipt')
            model.load_state_dict(cp['model'])
            p = predict(model, data, ids)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.tmp.npz')
            np.savez(temporary, ids=ids, prediction=p); os.replace(temporary, path)
            meta = dict(identity=ti, checkpoint_sha256=receipt['checkpoint_sha256'],
                        sha256=file_digest(path), path=str(path.relative_to(ROOT)))
            json_write(meta_path, meta)
        ade, fde = native_errors(p, data['target'][ids], data['valid'][ids], data['scale'][ids])
        key_pair = objective, seed
        if key_pair not in costs:
            costs[key_pair] = dict(ade=np.full(len(cv_ade), np.nan), fde=np.full(len(cv_ade), np.nan),
                                   seen=np.zeros(len(cv_ade), bool), hard=np.zeros(len(cv_ade), bool))
        dest = costs[key_pair]
        if dest['seen'][ids].any():
            raise ValueError('Repeated held row across source folds')
        dest['ade'][ids], dest['fde'][ids], dest['seen'][ids] = ade, fde, True
        dest['hard'][ids] = cv_ade[ids] >= fold['hard_cut']
        pred_receipts.append(dict(trial=key, path=meta['path'], sha256=meta['sha256']))
        train_receipts.append(dict(trial=key, fit=receipt['fit'], checkpoint=receipt['checkpoint'],
                                   checkpoint_sha256=receipt['checkpoint_sha256']))
    all_rows = np.ones(len(cv_ade), bool)
    def score(m, r, mask=all_rows):
        return paired_scene_metrics(m[mask], r[mask], data['sites'][mask], expected_scenes=reg['sites'],
            dataset='sdd', coordinate_unit='annotation_pixel', bootstrap_resamples=reg['bootstrap_resamples'],
            seed=reg['bootstrap_seed'])
    summary = {}
    complete = data['valid'].all(1)
    static = rows['static_history']
    for objective in reg['objectives']:
        seeds = {}
        for seed in reg['seeds']:
            c = costs[objective, seed]
            if not c['seen'].all():
                raise ValueError('Missing full-population held-source predictions')
            support = np.isfinite(cv_ade)
            seeds[str(seed)] = dict(ADE=score(c['ade'], cv_ade), FDE=score(c['fde'], cv_fde),
                complete_ADE=score(c['ade'], cv_ade, complete),
                static_history_ADE=score(c['ade'], cv_ade, static),
                moving_history_ADE=score(c['ade'], cv_ade, ~static),
                hard_training_q75_ADE=score(c['ade'], cv_ade, c['hard']),
                zero_CV_complete_easy_ADE=score(c['ade'], cv_ade, complete & (cv_ade == 0)),
                old_normalized_gain_percent=safe_gain(scene_mean(c['ade'][support]/data['scale'][support], data['sites'][support]),
                                                       scene_mean(rows['ade'][support, 1], data['sites'][support])))
        ade = np.mean([costs[objective, seed]['ade'] for seed in reg['seeds']], axis=0)
        fde = np.mean([costs[objective, seed]['fde'] for seed in reg['seeds']], axis=0)
        summary[objective] = dict(seeds=seeds, mean_seed_ADE=score(ade, cv_ade), mean_seed_FDE=score(fde, cv_fde))
    ref = np.mean([costs['past_normalized', seed]['ade'] for seed in reg['seeds']], axis=0)
    new = np.mean([costs['native_coordinate', seed]['ade'] for seed in reg['seeds']], axis=0)
    result = dict(result_source='fresh_training_and_held_source_inference_cached_verified_inputs',
        identity=identity, rows=len(cv_ade), sites=reg['sites'], recordings=len(set(data['recordings'])),
        models=len(entries), optimizer_updates=sum(r['fit']['step'] for r in train_receipts),
        summed_fit_seconds=sum(r['fit']['seconds'] for r in train_receipts),
        summary=summary, native_vs_matched_old_loss=score(new, ref),
        predictions=pred_receipts, training=train_receipts,
        primary='native_ADE_mean_within_scene_relative_gain',
        strongest_complement_reference='constant_velocity_causal_fd_fixed_by_native_metric_v1',
        seeds_averaged_as='errors_not_forecast_ensemble',
        independent_confirmation=False, threshold_search=False, new_deployment=False,
        full_easy_preservation_gate='not_assessed_new_native_risk_calibration_not_registered',
        original_val_test_rows=0, main_outer_external_rows=0, stage5c_executed=False, smc_enabled=False)
    assert_current(identity)
    immutable_json(reports/'analysis.json', result)
    lines = ['# Matched Full-Source Native-Loss Forecasting', '',
        'Fresh real Torch fits and held-source inference; source inputs hash-verified.',
        'Four previously explored SDD auxiliary training scenes, not independent confirmation.',
        'Full past-eligible population; unknown labels stay unknown. Pixel/raw-frame only.', '',
        '| Objective | Seed | Native ADE gain vs CV (%) | Worst scene (%) | Complete sensitivity (%) |',
        '| --- | ---: | ---: | ---: | ---: |']
    for objective, group in summary.items():
        for seed, metrics in group['seeds'].items():
            lines.append(f"| {objective} | {seed} | {metrics['ADE']['equal_scene_gain_percent']:.6f} | {metrics['ADE']['worst_scene_gain_percent']:.6f} | {metrics['complete_ADE']['equal_scene_gain_percent']:.6f} |")
        m = group['mean_seed_ADE']
        lines += ['', f"{objective}: mean-seed gain {m['equal_scene_gain_percent']:.6f}%; conditional scene interval {m['scene_bootstrap_ci95']}.", '']
    lines += [f"Native versus matched old-loss model: {result['native_vs_matched_old_loss']['equal_scene_gain_percent']:.6f}%.", '',
        'All seeds, per-scene errors, tails, old metric and named slices are retained in analysis.json.',
        'No best seed/checkpoint/threshold selected. Exactly static history is motion-bounded to CV;',
        'missed static starts remain evaluated. Zero-CV easy percentage is undefined, not a full safety pass.',
        'This is a loss contrast using an existing history/neighbor Transformer, not a new multimodal architecture.',
        'No independent calibration or deployment; Stage5C/SMC disabled.', '']
    text = '\n'.join(lines)
    report_path = reports/'results.md'
    if report_path.exists() and report_path.read_text() != text:
        raise ValueError('Result narrative replay mismatch')
    if not report_path.exists():
        report_path.write_text(text)
    heartbeat(state='evaluation_verified' if verify else 'evaluation_complete', models=len(entries), rows=len(cv_ade))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial'); p.add_argument('--stop-at', type=int)
    p.add_argument('--resume', action='store_true'); p.add_argument('--audit-only', action='store_true')
    p.add_argument('--evaluate', action='store_true'); p.add_argument('--verify', action='store_true')
    args = p.parse_args()
    if args.stop_at is not None and (not args.trial or args.evaluate or args.verify):
        raise ValueError('Pilot must name one registered training trial')
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    reg, data, rows, identity = load()
    root, reports = ROOT/reg['output'], ROOT/reg['reports']
    root.mkdir(parents=True, exist_ok=True); reports.mkdir(parents=True, exist_ok=True)
    lock = (root/'execution.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    def beat(**values):
        event = dict(pid=os.getpid(), timestamp_unix=time.time(), **values)
        json_write(root/'heartbeat.json', event)
        with (root/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt('SIGTERM; resume checkpoint')))
    immutable_json(root/'identity.json', identity)
    beat(state='inputs_verified', indexed_rows=len(data['sites']), original_val_test_rows=0,
         main_outer_external_rows=0, expected_models=24)
    if args.audit_only:
        return
    if args.evaluate or args.verify:
        evaluate(reg, data, rows, identity, root, reports, beat, verify=args.verify)
        return
    trials = [(site, objective, seed) for site in reg['sites'] for seed in reg['seeds'] for objective in reg['objectives']]
    names = {f'{s}_{o}_seed{k}' for s,o,k in trials}
    if args.trial is not None and args.trial not in names:
        raise ValueError('Unregistered trial')
    total_new = 0
    for site, objective, seed in trials:
        fold, ti, key = specification(reg, data, identity, site, objective, seed)
        if args.trial and args.trial != key:
            continue
        out = root/'trials'/key
        if (out/'complete.json').exists():
            completed(out/'complete.json', ti, reg['training'])
            beat(state='cached_verified_complete', trial=key, new_updates=0)
            continue
        torch.manual_seed(seed); model = build_forecaster(reg['architecture'])
        beat(state='training_registered_trial', trial=key, training_rows=len(fold['train_ids']))
        fit = fit_trial(model, data, fold, seed=seed, settings=reg['training'], identity=ti,
            directory=out, resume=args.resume, stop_at=args.stop_at,
            heartbeat=lambda **v:beat(trial=key, **v))
        total_new += fit['new_updates']
        if fit['complete']:
            assert_current(identity)
            immutable_json(out/'complete.json', dict(identity=ti, fit=fit,
                checkpoint=str((out/'checkpoint.pt').relative_to(ROOT)),
                checkpoint_sha256=file_digest(out/'checkpoint.pt')))
    beat(state='registered_training_call_complete', new_updates=total_new, held_scores_inspected=False)


if __name__ == '__main__':
    main()
