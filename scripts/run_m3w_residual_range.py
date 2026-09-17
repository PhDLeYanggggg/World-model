"""Fit-only 2x2 residual readout/loss comparison with frozen linear controls."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use the native arm64 training environment')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '4'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.build_m3w_observed_motion import load_registration
from scripts.run_m3w_objective_alignment import paired_interval
from scripts.run_m3w_track_event_sampling import check_saved
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_offline_visual_forecast import OfflineVisualForecast, forecast_metrics
from src.world_model.m3w_objective_alignment import supported_standardization
from src.world_model.m3w_observed_motion import feature_variant
from src.world_model.m3w_residual_range import ARMS, range_prediction, fit_range
from src.world_model.m3w_track_event_sampling import EVENT_NAMES, training_event_labels


def summary(trials, reg):
    lookup = {(t['variant'], t['arm'], t['seed'], t['fold']): t for t in trials}
    def matrix(v, arm, key):
        return np.array([[lookup[v, arm, s, f]['vs_CV'][key] for f in range(3)] for s in reg['seeds']])
    out = {}
    for v in reg['variants']:
        for arm in ARMS:
            rows = [t for t in trials if t['variant'] == v and t['arm'] == arm]
            out[v+'_'+arm] = dict(
                vs_CV=paired_interval(matrix(v, arm, 'primary_ADE'), matrix(v, arm, 'reference_ADE'), reg['bootstrap_resamples']),
                vs_linear_log=paired_interval(matrix(v, arm, 'primary_ADE'), matrix(v, 'linear_log', 'primary_ADE'), reg['bootstrap_resamples']),
                easy_degradation_percent=[t['vs_CV']['easy_degradation_percent'] for t in rows],
                safe_positive_fits=sum(t['vs_CV']['improvement_percent'] > 0 and
                    t['vs_CV']['easy_degradation_percent'] is not None and t['vs_CV']['easy_degradation_percent'] <= 2 for t in rows),
                training_gain_percent=[t['training_equal_scene_gain_percent'] for t in rows],
                fresh_fit_seconds=sum(t['fit']['fit_seconds'] for t in rows if arm != 'linear_log'),
                held_saturated_coordinates=sum(t['held_saturated_coordinates'] for t in rows))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--trial')
    parser.add_argument('--stop-at', type=int)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reg = json.loads(args.registration.read_text())
    if reg['role'] != 'fit_only_exploratory' or reg['arms'] != list(ARMS) or reg['cap'] != 12.:
        raise ValueError('Registered fit-only arms required')
    for path, digest in reg['bindings'].items():
        if file_digest(ROOT/path) != digest:
            raise ValueError('Changed bound file: '+path)
    old, parent, contract, source, receipt = load_registration(ROOT/reg['control_registration'])
    if reg['training'] != old['training'] or reg['seeds'] != old['seeds'] or receipt['rows'] != 11966:
        raise ValueError('Changed control budget/seeds/population')
    controls = ROOT/old['output']
    inputs = controls/'inputs'
    manifest = json.loads((inputs/'manifest.json').read_text())
    old_identity = dict(registration_sha256=file_digest(ROOT/reg['control_registration']),
        parent_protocol_sha256=contract.digest, source_manifest_sha256=file_digest(source/'data_manifest.json'))
    if manifest['identity'] != old_identity:
        raise ValueError('Feature lineage changed')
    for name, digest in manifest['arrays'].items():
        if file_digest(inputs/name) != digest:
            raise ValueError('Feature array changed')
    old_identity['motion_manifest_sha256'] = file_digest(inputs/'manifest.json')
    identity = dict(registration_sha256=file_digest(args.registration), source_identity=old_identity)
    output, reports = ROOT/reg['output'], ROOT/reg['reports']
    if not output.is_relative_to(ROOT/'data/stage_cvpr2027_experiments'):
        raise ValueError('Private output required')
    if (output/'identity.json').exists() and json.loads((output/'identity.json').read_text()) != identity:
        raise ValueError('Changed experiment identity')
    json_write(output/'identity.json', identity)
    keys = {f'{v}_{a}_seed{s}_fold{f}' for v in reg['variants'] for a in ARMS for s in reg['seeds'] for f in range(3)}
    if args.trial and args.trial not in keys:
        raise ValueError('Unknown trial')
    if args.stop_at is not None and (not args.trial or 'linear_log' in args.trial or not 0 < args.stop_at <= 4000):
        raise ValueError('Pilot needs a fresh registered trial and valid step')
    def heartbeat(value):
        value = dict(pid=os.getpid(), time_unix=time.time(), **value)
        json_write(output/'heartbeat.json', value)
        print(json.dumps(value), flush=True)
    arrays = {n:np.load(source/(n+'.npy'), mmap_mode='r') for n in
              ('geometry', 'targets', 'baselines', 'scale', 'folds', 'image_rows', 'coverage')}
    metadata = json.loads((source/'rows.json').read_text())
    tracks = np.array([f"{m['recording']}:{m['agent']}" for m in metadata])
    records = np.array([m['recording'] for m in metadata])
    if not np.array_equal(arrays['folds'], [m['fold'] for m in metadata]):
        raise ValueError('Row/fold alignment changed')
    observed = torch.cat([torch.from_numpy(arrays['coverage'][arrays['image_rows'][i:i+128], None].astype(np.float32)/9.)
                          .mean((2, 3, 4)) for i in range(0, receipt['rows'], 128)])
    motion, quality = [np.load(inputs/(n+'.npy'), mmap_mode='r') for n in ('motion', 'quality')]
    cv = receipt['baseline_names'].index('constant_velocity_causal_fd')
    base, target = torch.from_numpy(arrays['baselines'][:, cv].copy()), torch.from_numpy(arrays['targets'].copy())
    trials, replays, updates = [], [], 0
    for fold in range(3):
        train, held = np.flatnonzero(arrays['folds'] != fold), np.flatnonzero(arrays['folds'] == fold)
        allowed = np.zeros(len(target), bool); allowed[train] = True
        train_scenes = np.unique(arrays['folds'][train])
        train_error = np.linalg.norm(arrays['baselines'][train].astype(float)-arrays['targets'][train, None], axis=-1).mean(-1)
        strongest = int(np.argmin(np.mean([train_error[arrays['folds'][train] == f].mean(0) for f in train_scenes], axis=0)))
        for variant in reg['variants']:
            features = feature_variant(arrays['geometry'], motion, quality, variant)
            normalized, normalizer = supported_standardization(features[train], features)
            x = torch.from_numpy(normalized)
            def batch(ids):
                if not allowed[ids.numpy()].all():
                    raise ValueError('Held target requested during training')
                return x[ids], observed[ids], base[ids], target[ids]
            def infer(model, ids, arm):
                model.eval()
                predictions, raws = [], []
                with torch.no_grad():
                    for i in range(0, len(ids), 128):
                        use = ids[i:i+128]
                        p, raw = range_prediction(model, x[use], observed[use], base[use], arm, reg['cap'], True)
                        predictions.append(p.numpy()); raws.append(raw.numpy())
                return np.concatenate(predictions), np.concatenate(raws)
            for arm in ARMS:
                for seed in reg['seeds']:
                    key = f'{variant}_{arm}_seed{seed}_fold{fold}'
                    if args.trial and args.trial != key:
                        continue
                    trial_identity = dict(identity, variant=variant, arm=arm, seed=seed, fold=fold)
                    cp, pp, rp = [output/d/(key+ext) for d, ext in [('checkpoints','.pt'),('predictions','.npz'),('trials','.json')]]
                    is_control = arm == 'linear_log'
                    if is_control:
                        old_key = f'{variant}_row_log_seed{seed}_fold{fold}'
                        cp, pp = controls/'checkpoints'/(old_key+'.pt'), controls/'predictions'/(old_key+'.npz')
                    saved = json.loads(rp.read_text()) if rp.exists() else None
                    if saved:
                        check_saved(saved, trial_identity, cp, pp)
                        if not args.replay:
                            trials.append(saved); continue
                    torch.manual_seed(seed)
                    model = OfflineVisualForecast(features.shape[1])
                    if is_control:
                        expected = dict(old_identity, variant=variant, objective='row_log', seed=seed, fold=fold)
                        control = json.loads((controls/'trials'/(old_key+'.json')).read_text())
                        check_saved(control, expected, cp, pp)
                        state = torch.load(cp, map_location='cpu', weights_only=False)
                        if state['identity'] != expected or state['config'] != reg['training'] or state['step'] != 4000:
                            raise ValueError('Control checkpoint changed')
                        model.load_state_dict(state['model'])
                        fit = dict(control['fit'], new_updates_this_invocation=0, reused_not_fresh=True)
                    elif args.replay:
                        if saved is None:
                            raise ValueError('Replay needs completed trial')
                        state = torch.load(cp, map_location='cpu', weights_only=False)
                        if (state['identity'] != trial_identity or state['config'] != reg['training'] or state['step'] != 4000
                                or state['arm'] != arm or state['cap'] != reg['cap']
                                or not np.array_equal(state['train_ids'].numpy(), train)):
                            raise ValueError('Replay checkpoint changed')
                        model.load_state_dict(state['model'])
                    else:
                        fit = fit_range(model, batch, torch.from_numpy(train), arm=arm, cap=reg['cap'],
                            config=reg['training'], seed=seed, identity=trial_identity, checkpoint=cp,
                            heartbeat=lambda v:heartbeat(dict(trial=key, **v)), stop_at=args.stop_at)
                        updates += fit['new_updates_this_invocation']
                        if not fit['complete']:
                            heartbeat(dict(state='pilot_saved_no_held_evaluation', trial=key, step=fit['step'])); return
                    prediction, raw = infer(model, held, arm)
                    if is_control or args.replay:
                        with np.load(pp) as original:
                            if not np.array_equal(original['prediction'], prediction) or not np.array_equal(original['held_indices'], held):
                                raise ValueError('Exact forecast replay failed')
                    if args.replay:
                        replays.append(dict(trial=key, prediction_exact=True)); continue
                    if not is_control:
                        pp.parent.mkdir(parents=True, exist_ok=True)
                        np.savez(pp, prediction=prediction, held_indices=held)
                    train_pred, train_raw = infer(model, train, arm)
                    err = np.linalg.norm(train_pred.astype(float)-arrays['targets'][train], axis=-1).mean(1)
                    train_ade = np.mean([err[arrays['folds'][train] == f].mean() for f in train_scenes])
                    train_cv = np.mean([train_error[arrays['folds'][train] == f, cv].mean() for f in train_scenes])
                    slices = {}
                    held_events = training_event_labels(arrays['geometry'][held], arrays['targets'][held])
                    groups = [(name, held_events == j) for j, name in enumerate(EVENT_NAMES)]
                    groups += [(name, records[held] == name) for name in np.unique(records[held])]
                    for name, mask in groups:
                        ids = held[mask]
                        slices[name] = dict(forecast_metrics(prediction[mask], arrays['targets'][ids], arrays['baselines'][ids, cv],
                            arrays['scale'][ids], parent['easy_threshold']), tracks=len(np.unique(tracks[ids]))) if len(ids) else dict(rows=0)
                    train_static = np.all(arrays['geometry'][train, :16] == 0, axis=1)
                    result = dict(identity=trial_identity, trial=key, variant=variant, arm=arm, seed=seed, fold=fold,
                        result_source='cached_verified_control_replay' if is_control else 'fresh_run_native_torch_cached_verified_inputs',
                        fit=fit, train_rows=len(train), held_rows=len(held), training_equal_scene_gain_percent=float(100*(1-train_ade/train_cv)),
                        training_static_ADE=float(err[train_static].mean()) if train_static.any() else None,
                        training_static_CV_ADE=float(train_error[train_static, cv].mean()) if train_static.any() else None,
                        training_static_prediction_absolute_quantiles=np.quantile(np.abs(train_pred[train_static]), [.5,.9,.99,1]).tolist() if train_static.any() else [],
                        held_saturated_coordinates=int((np.abs(raw)>=reg['cap']).sum()) if arm.startswith('sinh') else 0,
                        vs_CV=forecast_metrics(prediction, arrays['targets'][held], arrays['baselines'][held, cv], arrays['scale'][held], parent['easy_threshold']),
                        vs_train_selected_strongest=forecast_metrics(prediction, arrays['targets'][held], arrays['baselines'][held, strongest], arrays['scale'][held], parent['easy_threshold']),
                        train_selected_strongest=receipt['baseline_names'][strongest], slices=slices,
                        constant_training_columns=int(normalizer['constant'].sum()),
                        checkpoint_path=str(cp.relative_to(ROOT)), prediction_path=str(pp.relative_to(ROOT)),
                        checkpoint_sha256=file_digest(cp), prediction_sha256=file_digest(pp))
                    json_write(rp, result); trials.append(result)
                    heartbeat(dict(state='held_fit_evaluated', trial=key, gain=result['vs_CV']['improvement_percent']))
    if args.replay:
        json_write(reports/'replay.json', dict(identity=identity, trials=replays, status='exact', new_optimizer_updates=0))
        heartbeat(dict(state='replay_complete', trials=len(replays))); return
    if args.trial:
        return
    if len(trials) != 72:
        raise ValueError('Incomplete comparison')
    report_path = reports/'report.json'
    if report_path.exists() and updates == 0:
        previous = json.loads(report_path.read_text())
        if previous['identity'] != identity or previous['trials'] != trials:
            raise ValueError('Completed report changed')
        heartbeat(dict(state='completed_resume_verified', trials=72, new_optimizer_updates=0)); return
    summaries = summary(trials, reg)
    json_write(report_path, dict(identity=identity, complete=True, fresh_fits=54, cached_verified_controls=18,
        optimizer_updates_in_fresh_fits=216000, rows=11966, trials=trials, summary=summaries,
        primary='past_normalized_ADE', aggregation='equal_physical_scene_and_seed',
        observation_mode='offline_annotated_not_strict_sensor_as_of', deployment=False,
        development_calibration_confirmation_opened=False, stage5c_executed=False, smc_enabled=False))
    lines = ['# Residual Range and Objective Comparison', '',
        '54 fresh Torch fits; 18 exactly replayed controls. Fixed-end checkpoints, no selection on held outcomes.',
        'All 11,966 previously exposed fit windows retained. No independent confirmation or deployment.', '',
        '| Feature / arm | Gain vs CV (%) | Gain vs linear/log (%) | Safe positive held fits |',
        '| --- | ---: | ---: | ---: |']
    for key, value in summaries.items():
        lines.append(f'| {key} | {value["vs_CV"]["gain_percent"]:.5f} | {value["vs_linear_log"]["gain_percent"]:.5f} | {value["safe_positive_fits"]}/9 |')
    lines += ['', 'Sinh cap is numerical only, not a physical bound. Asinh is a training target transform, not a changed evaluation metric.',
              'All scene intervals are exploratory resampling over three exposed scenes.', '']
    (reports/'report.md').write_text('\n'.join(lines))
    heartbeat(dict(state='complete', comparisons=72, fresh_fits=54))


if __name__ == '__main__':
    main()
