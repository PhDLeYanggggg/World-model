"""Fixed matched spatial-motion forecasts, no sealed-role evaluation or tuning."""
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
from src.world_model.m3w_objective_alignment import supported_standardization, geometry_prediction, fit_objective
from src.world_model.m3w_spatial_motion import VARIANTS, feature_variant


def summarize(trials, reg):
    lookup = {(t['variant'], t['objective'], t['seed'], t['fold']): t for t in trials}
    def matrix(variant, objective, field):
        return np.array([[lookup[variant, objective, seed, fold]['vs_CV'][field]
                          for fold in range(3)] for seed in reg['seeds']])
    summary = {}
    for objective in reg['objectives']:
        reference = matrix(VARIANTS[0], objective, 'reference_ADE')
        control = matrix(VARIANTS[0], objective, 'primary_ADE')
        mag = matrix('lowpass_grid', objective, 'primary_ADE')
        for variant in VARIANTS:
            chosen = [t for t in trials if t['variant'] == variant and t['objective'] == objective]
            pred = matrix(variant, objective, 'primary_ADE')
            summary[variant + '_' + objective] = dict(
                vs_CV=paired_interval(pred, reference, reg['bootstrap_resamples']),
                vs_quality_control=paired_interval(pred, control, reg['bootstrap_resamples']),
                vs_lowpass_grid=paired_interval(pred, mag, reg['bootstrap_resamples']),
                positive_held_folds=sum(t['vs_CV']['improvement_percent'] > 0 for t in chosen),
                easy_gate_passing_folds=sum(t['vs_CV']['easy_degradation_percent'] <= 2 for t in chosen),
                easy_degradation_range=[min(t['vs_CV']['easy_degradation_percent'] for t in chosen),
                                        max(t['vs_CV']['easy_degradation_percent'] for t in chosen)],
                training_primary_gain_range=[min(t['training_equal_scene_gain_percent'] for t in chosen),
                                              max(t['training_equal_scene_gain_percent'] for t in chosen)],
                binary_oracle_gain_percent_diagnostic=float(100 * (1 - np.mean([
                    t['binary_oracle_ADE_diagnostic'] for t in chosen]) / reference.mean())),
                fit_seconds=sum(t['fit']['fit_seconds'] for t in chosen))
    return summary


def heartbeat_payload(value):
    return {'pid': os.getpid(), 'time_unix': time.time(), **value}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--trial')
    parser.add_argument('--stop-at', type=int)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reg, parent, contract, source, receipt = load_registration(args.registration)
    if reg['variants'] != list(VARIANTS) or reg['objectives'] != ['row_log']:
        raise ValueError('Fixed feature and objective contrasts required')
    if reg['seeds'] != parent['seeds'] or receipt['rows'] != 11966:
        raise ValueError('Changed cohort or seeds')
    output, reports = ROOT / reg['output'], ROOT / reg['reports']
    if not output.is_relative_to(ROOT / 'data/stage_cvpr2027_experiments'):
        raise ValueError('Private artifact directory required')
    inputs = output / 'inputs'
    manifest = json.loads((inputs / 'manifest.json').read_text())
    identity = dict(registration_sha256=file_digest(args.registration), parent_protocol_sha256=contract.digest,
                    source_manifest_sha256=file_digest(source / 'data_manifest.json'))
    if manifest['identity'] != identity:
        raise ValueError('Input extraction identity mismatch')
    for name, digest in manifest['arrays'].items():
        if file_digest(inputs / name) != digest:
            raise ValueError('Changed motion inputs')
    identity['motion_manifest_sha256'] = file_digest(inputs / 'manifest.json')
    if (output / 'identity.json').exists() and json.loads((output / 'identity.json').read_text()) != identity:
        raise ValueError('Training identity changed')
    json_write(output / 'identity.json', identity)
    valid_trials = {f'{v}_{o}_seed{s}_fold{f}' for v in VARIANTS for o in reg['objectives']
                    for s in reg['seeds'] for f in range(3)}
    if args.trial and args.trial not in valid_trials:
        raise ValueError('Unknown trial')
    if args.stop_at is not None and (not args.trial or not 0 < args.stop_at <= reg['training']['updates']):
        raise ValueError('Pilot must specify one trial and a valid step')
    def heartbeat(value):
        value = heartbeat_payload(value)
        json_write(output / 'heartbeat.json', value)
        print(json.dumps(value), flush=True)
    a = {n: np.load(source / (n + '.npy'), mmap_mode='r') for n in
         ('geometry', 'targets', 'baselines', 'scale', 'folds', 'image_rows', 'coverage')}
    motion = np.load(inputs / 'motion.npy', mmap_mode='r')
    quality = np.load(inputs / 'quality.npy', mmap_mode='r')
    observed = torch.cat([torch.from_numpy(a['coverage'][a['image_rows'][i:i+128], None].astype(np.float32) / 9.)
                         .mean((2, 3, 4)) for i in range(0, receipt['rows'], 128)])
    cv = receipt['baseline_names'].index('constant_velocity_causal_fd')
    base, target = torch.from_numpy(a['baselines'][:, cv].copy()), torch.from_numpy(a['targets'].copy())
    metadata = json.loads((source / 'rows.json').read_text())
    records = np.array([m['recording'] for m in metadata])
    stationary = np.all(a['geometry'][:, :16] == 0, axis=1)
    trials, replay = [], []
    for fold in range(3):
        train, held = np.flatnonzero(a['folds'] != fold), np.flatnonzero(a['folds'] == fold)
        allowed = np.zeros(len(target), bool)
        allowed[train] = True
        train_scenes = sorted(set(a['folds'][train]))
        train_errors = np.linalg.norm(a['baselines'][train].astype(float) - a['targets'][train, None], axis=-1).mean(-1)
        strongest = int(np.argmin(np.mean([train_errors[a['folds'][train] == f].mean(0) for f in train_scenes], axis=0)))
        for variant in VARIANTS:
            features = feature_variant(a['geometry'], motion, quality, variant)
            normalized, normalizer = supported_standardization(features[train], features)
            x = torch.from_numpy(normalized)
            def batch(ids):
                if not allowed[ids.numpy()].all():
                    raise ValueError('Held rows cannot enter training')
                return x[ids], observed[ids], base[ids], target[ids]
            def infer(model, ids):
                model.eval()
                values = []
                with torch.no_grad():
                    for start in range(0, len(ids), 128):
                        use = ids[start:start + 128]
                        values.append(geometry_prediction(model, x[use], observed[use], base[use]).numpy())
                return np.concatenate(values)
            for objective in reg['objectives']:
                for seed in reg['seeds']:
                    key = f'{variant}_{objective}_seed{seed}_fold{fold}'
                    if args.trial and args.trial != key:
                        continue
                    cp, pp, rp = output/'checkpoints'/(key+'.pt'), output/'predictions'/(key+'.npz'), output/'trials'/(key+'.json')
                    trial_identity = dict(identity, variant=variant, objective=objective, seed=seed, fold=fold)
                    saved = json.loads(rp.read_text()) if rp.exists() else None
                    if saved:
                        if saved['identity'] != trial_identity or saved['checkpoint_sha256'] != file_digest(cp) or saved['prediction_sha256'] != file_digest(pp):
                            raise ValueError('Completed trial changed')
                        if not args.replay:
                            trials.append(saved)
                            continue
                    torch.manual_seed(seed)
                    model = OfflineVisualForecast(features.shape[1])
                    if args.replay:
                        if saved is None:
                            raise ValueError('Missing completed trial')
                        state = torch.load(cp, map_location='cpu', weights_only=False)
                        if state['identity'] != trial_identity or state['step'] != reg['training']['updates'] or state['config'] != reg['training']:
                            raise ValueError('Replay checkpoint identity mismatch')
                        model.load_state_dict(state['model'])
                    else:
                        fit = fit_objective(model, batch, torch.from_numpy(train), a['folds'][train],
                            arm=objective, config=reg['training'], seed=seed, identity=trial_identity,
                            checkpoint=cp, heartbeat=lambda v: heartbeat(dict(trial=key, **v)), stop_at=args.stop_at)
                        if not fit['complete']:
                            heartbeat(dict(state='pilot_saved_no_held_evaluation', trial=key, step=fit['step']))
                            return
                    prediction = infer(model, held)
                    if args.replay:
                        with np.load(pp) as old:
                            exact = np.array_equal(old['prediction'], prediction) and np.array_equal(old['held_indices'], held)
                        if not exact:
                            raise ValueError('Checkpoint prediction replay differs')
                        replay.append(dict(trial=key, prediction_exact=True))
                        continue
                    pp.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(pp, prediction=prediction, held_indices=held)
                    error = np.linalg.norm(prediction.astype(float) - a['targets'][held], axis=-1).mean(1)
                    reference = np.linalg.norm(a['baselines'][held, cv].astype(float) - a['targets'][held], axis=-1).mean(1)
                    training_prediction = infer(model, train)
                    training_error = np.linalg.norm(training_prediction.astype(float) - a['targets'][train], axis=-1).mean(1)
                    train_ade = np.mean([training_error[a['folds'][train] == f].mean() for f in train_scenes])
                    train_cv = np.mean([train_errors[a['folds'][train] == f, cv].mean() for f in train_scenes])
                    slices = {}
                    for label, mask in [('stationary', stationary[held]), ('nonstationary', ~stationary[held]),
                                       *[(r, records[held] == r) for r in sorted(set(records[held]))]]:
                        ids = held[mask]
                        slices[label] = forecast_metrics(prediction[mask], a['targets'][ids], a['baselines'][ids, cv],
                            a['scale'][ids], parent['easy_threshold']) if len(ids) else dict(rows=0, status='no_support')
                    result = dict(identity=trial_identity, trial=key, variant=variant, objective=objective, seed=seed,
                        fold=fold, fit=fit, train_rows=len(train), held_rows=len(held),
                        result_source='fresh_run_native_torch_fit_cached_verified_inputs',
                        vs_CV=forecast_metrics(prediction, a['targets'][held], a['baselines'][held, cv], a['scale'][held], parent['easy_threshold']),
                        vs_train_selected_strongest=forecast_metrics(prediction, a['targets'][held], a['baselines'][held, strongest], a['scale'][held], parent['easy_threshold']),
                        training_equal_scene_gain_percent=float(100 * (1 - train_ade / train_cv)),
                        binary_oracle_ADE_diagnostic=float(np.minimum(error, reference).mean()), slices=slices,
                        stationary_positive_harm_share=float(np.maximum(error-reference, 0)[stationary[held]].sum() / max(np.maximum(error-reference, 0).sum(), 1e-12)),
                        train_selected_strongest=receipt['baseline_names'][strongest],
                        constant_training_columns=int(normalizer['constant'].sum()), checkpoint_sha256=file_digest(cp), prediction_sha256=file_digest(pp))
                    json_write(rp, result)
                    trials.append(result)
                    heartbeat(dict(state='held_fit_evaluated', trial=key, gain=result['vs_CV']['improvement_percent']))
    if args.replay:
        json_write(reports/'replay.json', dict(identity=identity, result_source='cached_verified_checkpoint_inference', trials=replay))
        heartbeat(dict(state='replay_complete', fits=len(replay)))
        return
    if args.trial:
        return
    if len(trials) != len(valid_trials):
        raise ValueError('Incomplete comparison')
    summary = summarize(trials, reg)
    report = dict(identity=identity, complete=True, result_source='fresh_run_real_training', rows=receipt['rows'],
        input_source='fresh_run_motion_cached_verified_images', trials=trials, summary=summary,
        primary='past_normalized_ADE', aggregation='equal_physical_scene_and_seed',
        observation_mode='offline_annotated_not_strict_sensor_as_of', deployment=False, submission_ready=False,
        development_calibration_confirmation_opened=False, stage5c_executed=False, smc_enabled=False)
    json_write(reports/'report.json', report)
    lines = ['# Spatial Image Motion Forecasting', '', 'Fit-only exploratory comparison; no independent confirmation.', '',
        '| Variant/objective | Gain vs CV (%) | Gain vs quality control (%) | Positive held fits | Easy passes | Binary oracle (%) |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key, s in summary.items():
        lines.append(f'| {key} | {s["vs_CV"]["gain_percent"]:.4f} | {s["vs_quality_control"]["gain_percent"]:.4f} | {s["positive_held_folds"]}/9 | {s["easy_gate_passing_folds"]}/9 | {s["binary_oracle_gain_percent_diagnostic"]:.4f} |')
    lines += ['', 'Three historically used fit scenes;2,000 paired scene bootstrap draws are exploratory only.',
              'No physical calibration, time-unit, strict online-sensor or deployment claim.', '']
    (reports/'report.md').write_text('\n'.join(lines))
    heartbeat(dict(state='complete', fits=len(trials), summary=summary))


if __name__ == '__main__':
    main()
