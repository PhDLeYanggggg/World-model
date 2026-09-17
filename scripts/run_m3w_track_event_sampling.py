"""Matched train-only sampling experiment; sealed scientific roles stay closed."""
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
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write
from src.world_model.m3w_offline_visual_forecast import OfflineVisualForecast, forecast_metrics
from src.world_model.m3w_objective_alignment import supported_standardization, geometry_prediction
from src.world_model.m3w_observed_motion import feature_variant
from src.world_model.m3w_track_event_sampling import (
    MODES, EVENT_NAMES, train_distribution, training_event_labels, distribution_summary, fit_sampled,
)


def summarize(trials, reg):
    lookup = {(t['variant'], t['mode'], t['seed'], t['fold']): t for t in trials}
    def matrix(v, m, field):
        return np.array([[lookup[v, m, s, f]['vs_CV'][field] for f in range(3)] for s in reg['seeds']])
    summary = {}
    for variant in reg['variants']:
        for mode in reg['modes']:
            chosen = [t for t in trials if t['variant'] == variant and t['mode'] == mode]
            pred = matrix(variant, mode, 'primary_ADE')
            easy = [t['vs_CV']['easy_degradation_percent'] for t in chosen]
            summary[variant+'_'+mode] = dict(
                vs_CV=paired_interval(pred, matrix(variant, mode, 'reference_ADE'), reg['bootstrap_resamples']),
                vs_row_control=paired_interval(pred, matrix(variant, 'row_uniform', 'primary_ADE'), reg['bootstrap_resamples']),
                vs_quality_control=paired_interval(pred, matrix('quality_control', mode, 'primary_ADE'), reg['bootstrap_resamples']),
                positive_held_fits=sum(t['vs_CV']['improvement_percent'] > 0 for t in chosen),
                easy_passing_fits=sum(v is not None and v <= 2 for v in easy),
                easy_degradation_percent=[v for v in easy],
                training_primary_gain_percent=[t['training_equal_scene_gain_percent'] for t in chosen],
                fresh_fit_seconds=sum(t['fit']['fit_seconds'] for t in chosen if mode != 'row_uniform'))
    return summary


def check_saved(record, identity, checkpoint, prediction):
    if (record['identity'] != identity or record['checkpoint_sha256'] != file_digest(checkpoint)
            or record['prediction_sha256'] != file_digest(prediction)):
        raise ValueError('Completed trial identity or artifact changed')


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
    if reg['role'] != 'fit_only_exploratory' or reg['modes'] != list(MODES):
        raise ValueError('Fixed fit-only sampling registration required')
    for path, digest in reg['bindings'].items():
        if file_digest(ROOT/path) != digest:
            raise ValueError('Changed bound input: '+path)
    old_reg, parent, contract, source, receipt = load_registration(ROOT/reg['control_registration'])
    if (reg['training'] != old_reg['training'] or reg['seeds'] != old_reg['seeds']
            or reg['variants'] != ['quality_control', 'directed'] or receipt['rows'] != 11966):
        raise ValueError('Changed matched training budget, features, seeds or cohort')
    controls = ROOT/old_reg['output']
    inputs = controls/'inputs'
    manifest = json.loads((inputs/'manifest.json').read_text())
    old_identity = dict(registration_sha256=file_digest(ROOT/reg['control_registration']),
        parent_protocol_sha256=contract.digest, source_manifest_sha256=file_digest(source/'data_manifest.json'))
    if manifest['identity'] != old_identity:
        raise ValueError('Old feature cache identity changed')
    for name, digest in manifest['arrays'].items():
        if file_digest(inputs/name) != digest:
            raise ValueError('Feature cache changed')
    old_identity['motion_manifest_sha256'] = file_digest(inputs/'manifest.json')
    identity = dict(registration_sha256=file_digest(args.registration), source_identity=old_identity)
    output, reports = ROOT/reg['output'], ROOT/reg['reports']
    if not output.is_relative_to(ROOT/'data/stage_cvpr2027_experiments'):
        raise ValueError('Private artifact path required')
    if (output/'identity.json').exists() and json.loads((output/'identity.json').read_text()) != identity:
        raise ValueError('Experiment identity changed')
    json_write(output/'identity.json', identity)
    keys = {f'{v}_{m}_seed{s}_fold{f}' for v in reg['variants'] for m in MODES for s in reg['seeds'] for f in range(3)}
    if args.trial and args.trial not in keys:
        raise ValueError('Unknown trial')
    if args.stop_at is not None and (not args.trial or 'row_uniform' in args.trial
                                    or not 0 < args.stop_at <= reg['training']['updates']):
        raise ValueError('Pilot requires a new registered fit and valid step')
    def heartbeat(value):
        value = dict(pid=os.getpid(), time_unix=time.time(), **value)
        json_write(output/'heartbeat.json', value)
        print(json.dumps(value), flush=True)
    a = {n:np.load(source/(n+'.npy'), mmap_mode='r') for n in
         ('geometry', 'targets', 'baselines', 'scale', 'folds', 'image_rows', 'coverage')}
    metadata = json.loads((source/'rows.json').read_text())
    records = np.array([m['recording'] for m in metadata])
    tracks = np.array([f"{m['recording']}:{m['agent']}" for m in metadata])
    if not np.array_equal(a['folds'], [m['fold'] for m in metadata]):
        raise ValueError('Row/fold lineage mismatch')
    observed = torch.cat([torch.from_numpy(a['coverage'][a['image_rows'][i:i+128], None].astype(np.float32)/9.)
                         .mean((2, 3, 4)) for i in range(0, receipt['rows'], 128)])
    motion, quality = [np.load(inputs/(n+'.npy'), mmap_mode='r') for n in ('motion', 'quality')]
    cv = receipt['baseline_names'].index('constant_velocity_causal_fd')
    base, target = torch.from_numpy(a['baselines'][:, cv].copy()), torch.from_numpy(a['targets'].copy())
    trials, replay, executed_updates = [], [], 0
    for fold in range(3):
        train, held = np.flatnonzero(a['folds'] != fold), np.flatnonzero(a['folds'] == fold)
        allowed = np.zeros(len(target), bool); allowed[train] = True
        train_scenes = np.unique(a['folds'][train])
        train_errors = np.linalg.norm(a['baselines'][train].astype(float)-a['targets'][train, None], axis=-1).mean(-1)
        strongest = int(np.argmin(np.mean([train_errors[a['folds'][train] == f].mean(0) for f in train_scenes], axis=0)))
        for variant in reg['variants']:
            features = feature_variant(a['geometry'], motion, quality, variant)
            normalized, normalizer = supported_standardization(features[train], features)
            x = torch.from_numpy(normalized)
            def batch(ids):
                if not allowed[ids.numpy()].all():
                    raise ValueError('Held rows in training batch')
                return x[ids], observed[ids], base[ids], target[ids]
            def infer(model, ids):
                model.eval()
                with torch.no_grad():
                    return np.concatenate([geometry_prediction(model, x[use], observed[use], base[use]).numpy()
                                           for use in (ids[i:i+128] for i in range(0, len(ids), 128))])
            for mode in MODES:
                p, events = train_distribution(train, a['folds'], tracks, a['geometry'], a['targets'], mode)
                for seed in reg['seeds']:
                    key = f'{variant}_{mode}_seed{seed}_fold{fold}'
                    if args.trial and args.trial != key:
                        continue
                    trial_identity = dict(identity, variant=variant, mode=mode, seed=seed, fold=fold)
                    cp, pp, rp = [output/d/(key+ext) for d, ext in
                                  [('checkpoints', '.pt'), ('predictions', '.npz'), ('trials', '.json')]]
                    is_control = mode == 'row_uniform'
                    if is_control:
                        old_key = f'{variant}_row_log_seed{seed}_fold{fold}'
                        cp, pp = controls/'checkpoints'/(old_key+'.pt'), controls/'predictions'/(old_key+'.npz')
                    saved = json.loads(rp.read_text()) if rp.exists() else None
                    if saved:
                        check_saved(saved, trial_identity, cp, pp)
                        if not args.replay:
                            trials.append(saved)
                            continue
                    torch.manual_seed(seed)
                    model = OfflineVisualForecast(features.shape[1])
                    draw_counts = None
                    if is_control:
                        expected = dict(old_identity, variant=variant, objective='row_log', seed=seed, fold=fold)
                        old = json.loads((controls/'trials'/(old_key+'.json')).read_text())
                        check_saved(old, expected, cp, pp)
                        state = torch.load(cp, map_location='cpu', weights_only=False)
                        if state['identity'] != expected or state['config'] != reg['training'] or state['step'] != 4000:
                            raise ValueError('Control checkpoint changed')
                        model.load_state_dict(state['model'])
                        fit = dict(old['fit'], new_updates_this_invocation=0, reused_training_not_fresh=True)
                    elif args.replay:
                        if saved is None:
                            raise ValueError('Missing completed trial')
                        state = torch.load(cp, map_location='cpu', weights_only=False)
                        if (state['identity'] != trial_identity or state['config'] != reg['training']
                                or state['step'] != 4000 or state['mode'] != mode
                                or not np.array_equal(state['probabilities'].numpy(), p)
                                or not np.array_equal(state['train_ids'].numpy(), train)):
                            raise ValueError('Replay mismatch')
                        model.load_state_dict(state['model'])
                    else:
                        fit = fit_sampled(model, batch, torch.from_numpy(train), p, mode=mode,
                            config=reg['training'], seed=seed, identity=trial_identity, checkpoint=cp,
                            heartbeat=lambda v:heartbeat(dict(trial=key, **v)), stop_at=args.stop_at)
                        executed_updates += fit['new_updates_this_invocation']
                        draw_counts = fit.pop('draw_counts')
                        if not fit['complete']:
                            heartbeat(dict(state='pilot_saved_no_held_evaluation', trial=key, step=fit['step']))
                            return
                    prediction = infer(model, held)
                    if is_control or args.replay:
                        with np.load(pp) as original:
                            if (not np.array_equal(original['prediction'], prediction)
                                    or not np.array_equal(original['held_indices'], held)):
                                raise ValueError('Prediction/row alignment replay differs')
                    if args.replay:
                        replay.append(dict(trial=key, prediction_exact=True, source='cached_verified'))
                        continue
                    if not is_control:
                        pp.parent.mkdir(parents=True, exist_ok=True)
                        np.savez(pp, prediction=prediction, held_indices=held)
                    train_pred = infer(model, train)
                    train_error = np.linalg.norm(train_pred.astype(float)-a['targets'][train], axis=-1).mean(1)
                    train_ade = np.mean([train_error[a['folds'][train] == f].mean() for f in train_scenes])
                    train_cv = np.mean([train_errors[a['folds'][train] == f, cv].mean() for f in train_scenes])
                    # Held labels are opened only after the fixed-end model is frozen, for descriptive slices.
                    held_events = training_event_labels(a['geometry'][held], a['targets'][held])
                    slices = {}
                    groups = [(name, held_events == i) for i, name in enumerate(EVENT_NAMES)]
                    groups += [(r, records[held] == r) for r in np.unique(records[held])]
                    for name, mask in groups:
                        ids = held[mask]
                        slices[name] = dict(forecast_metrics(prediction[mask], a['targets'][ids],
                            a['baselines'][ids, cv], a['scale'][ids], parent['easy_threshold']),
                            tracks=len(np.unique(tracks[ids]))) if len(ids) else dict(rows=0, status='no_support')
                    result = dict(identity=trial_identity, trial=key, variant=variant, mode=mode, seed=seed, fold=fold,
                        result_source='cached_verified_exact_control_replay' if is_control else 'fresh_run_native_torch_fit_cached_verified_inputs',
                        fit=fit, train_rows=len(train), held_rows=len(held),
                        training_distribution=distribution_summary(p, a['folds'][train], tracks[train], events, draw_counts),
                        vs_CV=forecast_metrics(prediction, a['targets'][held], a['baselines'][held, cv], a['scale'][held], parent['easy_threshold']),
                        vs_train_selected_strongest=forecast_metrics(prediction, a['targets'][held], a['baselines'][held, strongest], a['scale'][held], parent['easy_threshold']),
                        training_equal_scene_gain_percent=float(100*(1-train_ade/train_cv)), slices=slices,
                        train_selected_strongest=receipt['baseline_names'][strongest],
                        constant_training_columns=int(normalizer['constant'].sum()),
                        checkpoint_path=str(cp.relative_to(ROOT)), prediction_path=str(pp.relative_to(ROOT)),
                        checkpoint_sha256=file_digest(cp), prediction_sha256=file_digest(pp))
                    json_write(rp, result); trials.append(result)
                    heartbeat(dict(state='held_fit_evaluated', trial=key, source=result['result_source'], gain=result['vs_CV']['improvement_percent']))
    if args.replay:
        json_write(reports/'replay.json', dict(identity=identity, trials=replay, status='exact', new_optimizer_updates=0))
        heartbeat(dict(state='replay_complete', fits=len(replay)))
        return
    if args.trial:
        return
    if len(trials) != 72:
        raise ValueError('Incomplete matched comparison')
    report_path = reports/'report.json'
    if report_path.exists() and executed_updates == 0:
        old = json.loads(report_path.read_text())
        if old['identity'] != identity or old['trials'] != trials:
            raise ValueError('Completed report changed')
        heartbeat(dict(state='completed_resume_verified', trials=72, new_optimizer_updates=0))
        return
    summary = summarize(trials, reg)
    json_write(report_path, dict(identity=identity, complete=True, fresh_fits=54, cached_verified_controls=18,
        fresh_optimizer_updates=54*reg['training']['updates'], trials=trials, summary=summary,
        rows=receipt['rows'], primary='past_normalized_ADE', aggregation='equal_physical_scene_and_seed',
        result_source='54_fresh_fits_18_cached_verified_controls',
        observation_mode='offline_annotated_not_strict_sensor_as_of', deployment=False,
        development_calibration_confirmation_opened=False, stage5c_executed=False, smc_enabled=False))
    lines = ['# Matched Track/Event Sampling', '',
        '54 fresh Torch fits and 18 exact-replayed cached controls. Fixed-end checkpoints; all 11,966 fit windows retained.',
        'Three historically used physical scenes, not independent confirmation. Targets only drive supervised training sampling and evaluation.', '',
        '| Feature/sampling | Gain vs CV (%) | Gain vs row control (%) | Positive folds | Easy passes |',
        '| --- | ---: | ---: | ---: | ---: |']
    for key, s in summary.items():
        lines.append(f'| {key} | {s["vs_CV"]["gain_percent"]:.4f} | {s["vs_row_control"]["gain_percent"]:.4f} | {s["positive_held_fits"]}/9 | {s["easy_passing_fits"]}/9 |')
    lines += ['', '2,000 scene bootstrap draws are exploratory; overlapping windows are not independent observations.',
              'No new deployment, metric/physical time, Stage5C, SMC or foundation claim.', '']
    (reports/'report.md').write_text('\n'.join(lines))
    heartbeat(dict(state='complete', comparisons=72, fresh_fits=54, fresh_optimizer_updates=216000))


if __name__ == '__main__':
    main()
