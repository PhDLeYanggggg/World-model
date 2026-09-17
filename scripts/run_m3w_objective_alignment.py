"""Fixed objective/sampling comparison on fit scenes; no new test selection."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use native arm64 .venv-pytorch')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '4'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write, verify_registration
from src.world_model.m3w_offline_visual_forecast import OfflineVisualForecast, forecast_metrics
from src.world_model.m3w_objective_alignment import (
    ARMS, fit_objective, geometry_prediction, sampling_weights, supported_standardization,
)


def load_experiment(path):
    reg = json.loads(path.read_text())
    if reg['role'] != 'fit_only_exploratory' or reg['arms'] != list(ARMS) or not reg['bindings']:
        raise ValueError('Fixed fit-only registration required')
    for name, digest in reg['bindings'].items():
        if file_digest(ROOT / name) != digest:
            raise ValueError('Changed registered dependency: ' + name)
    parent, contract = verify_registration(ROOT, ROOT / reg['input_registration'])
    data = ROOT / parent['output'] / 'inputs'
    receipt = json.loads((data / 'data_manifest.json').read_text())
    if receipt['rows'] != 11966 or reg['seeds'] != parent['seeds']:
        raise ValueError('Changed cohort or seeds')
    for name, digest in receipt['arrays'].items():
        if file_digest(data / name) != digest:
            raise ValueError('Changed source input: ' + name)
    arrays = {k: np.load(data / (k + '.npy'), mmap_mode='r') for k in
              ('geometry', 'targets', 'baselines', 'scale', 'folds', 'image_rows', 'coverage')}
    observed = []
    for start in range(0, receipt['rows'], 128):
        rows = arrays['image_rows'][start:start + 128]
        coverage = torch.from_numpy(arrays['coverage'][rows, None].astype(np.float32) / 9.)
        observed.append(coverage.mean((2, 3, 4)))
    arrays['observed'] = torch.cat(observed)
    identity = {'registration_sha256': file_digest(path), 'parent_protocol_sha256': contract.digest,
                'source_data_manifest_sha256': file_digest(data / 'data_manifest.json')}
    return reg, parent, receipt, arrays, identity


def paired_interval(model, reference, repeats):
    model, reference = np.asarray(model), np.asarray(reference)
    if model.shape != (3, 3) or reference.shape != model.shape:
        raise ValueError('Exactly three seeds by three fit scenes required')
    rng = np.random.default_rng(917)
    draws = rng.integers(0, 3, size=(repeats, 3))
    m, r = model.mean(0), reference.mean(0)
    gain = 100 * (1 - m[draws].mean(1) / r[draws].mean(1))
    return {'gain_percent': float(100 * (1 - model.mean() / reference.mean())),
            'exploratory_scene_ci95_percent': np.quantile(gain, [.025, .975]).tolist(),
            'per_seed_gain_percent': (100 * (1 - model.mean(1) / reference.mean(1))).tolist(),
            'per_scene_gain_percent': (100 * (1 - m / r)).tolist(),
            'scene_clusters': 3, 'bootstrap_resamples': repeats,
            'not_independent_confirmation': True}


def summarize(trials, reg):
    lookup = {(t['arm'], t['seed'], t['fold']): t for t in trials}
    matrices = {arm: np.array([[lookup[arm, seed, fold]['vs_CV']['primary_ADE']
        for fold in range(3)] for seed in reg['seeds']]) for arm in ARMS}
    baseline = np.array([[lookup['row_log', seed, fold]['vs_CV']['reference_ADE']
        for fold in range(3)] for seed in reg['seeds']])
    summary = {}
    for arm in ARMS:
        ts = [t for t in trials if t['arm'] == arm]
        summary[arm] = {'vs_CV': paired_interval(matrices[arm], baseline, reg['bootstrap_resamples']),
            'vs_row_log': paired_interval(matrices[arm], matrices['row_log'], reg['bootstrap_resamples']),
            'binary_oracle_gain_percent_diagnostic': float(100 * (
                1 - np.mean([t['binary_oracle_ADE_diagnostic'] for t in ts]) / baseline.mean())),
            'positive_held_folds': sum(t['vs_CV']['improvement_percent'] > 0 for t in ts),
            'easy_gate_passing_folds': sum(t['vs_CV']['easy_degradation_percent'] <= 2 for t in ts),
            'easy_degradation_percent_range': [min(t['vs_CV']['easy_degradation_percent'] for t in ts),
                                               max(t['vs_CV']['easy_degradation_percent'] for t in ts)],
            'easy_absolute_harm_range': [min(t['vs_CV']['easy_absolute_harm'] for t in ts),
                                         max(t['vs_CV']['easy_absolute_harm'] for t in ts)],
            'fit_seconds': sum(t['fit']['fit_seconds'] for t in ts)}
    pairs = [('row_ade', 'row_log'), ('scene_log', 'row_log'), ('scene_ade', 'row_ade'),
             ('scene_ade', 'scene_log'), ('scene_ade_harm', 'scene_ade')]
    contrasts = {a + '_vs_' + b: paired_interval(matrices[a], matrices[b], reg['bootstrap_resamples'])
                 for a, b in pairs}
    return summary, contrasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--trial')
    parser.add_argument('--stop-at', type=int)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    if args.stop_at is not None and not args.trial:
        raise ValueError('Partial pilot requires an explicit trial')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reg, parent, receipt, a, identity = load_experiment(args.registration)
    valid_trials = {f'{arm}_seed{seed}_fold{fold}' for arm in ARMS for seed in reg['seeds'] for fold in range(3)}
    if args.trial and args.trial not in valid_trials:
        raise ValueError('Unknown registered trial')
    if args.stop_at is not None and not 0 < args.stop_at <= reg['training']['updates']:
        raise ValueError('Pilot stop must be a positive registered update')
    output, reports = ROOT / reg['output'], ROOT / reg['reports']
    if not output.is_relative_to(ROOT / 'data/stage_cvpr2027_experiments'):
        raise ValueError('Keep weights and predictions in ignored data cache')
    if (output / 'identity.json').exists() and json.loads((output / 'identity.json').read_text()) != identity:
        raise ValueError('Existing run has a different identity')
    json_write(output / 'identity.json', identity)
    def heartbeat(value):
        payload = {'pid': os.getpid(), 'time_unix': time.time(), **value}
        json_write(output / 'heartbeat.json', payload)
        print(json.dumps(payload), flush=True)
    cv = receipt['baseline_names'].index('constant_velocity_causal_fd')
    base = torch.from_numpy(a['baselines'][:, cv].copy())
    target = torch.from_numpy(a['targets'].copy())
    stationary = np.all(a['geometry'][:, :16] == 0, axis=1)
    metadata = json.loads((ROOT / parent['output'] / 'inputs/rows.json').read_text())
    records = np.asarray([r['recording'] for r in metadata])
    trials, replay, audit = [], [], {}
    for seed in reg['seeds']:
        for fold in range(3):
            train, held = np.flatnonzero(a['folds'] != fold), np.flatnonzero(a['folds'] == fold)
            z, normalizer = supported_standardization(a['geometry'][train], a['geometry'])
            x = torch.from_numpy(z)
            allowed_train = np.zeros(len(z), bool)
            allowed_train[train] = True
            def train_batch(ids):
                if not allowed_train[ids.numpy()].all():
                    raise ValueError('Held fit rows cannot enter a training batch')
                return x[ids], a['observed'][ids], base[ids], target[ids]
            def infer(model, ids):
                pred = []
                model.eval()
                with torch.no_grad():
                    for start in range(0, len(ids), 128):
                        chunk = ids[start:start + 128]
                        pred.append(geometry_prediction(model, x[chunk], a['observed'][chunk], base[chunk]).numpy())
                return np.concatenate(pred)
            errors = np.linalg.norm(a['baselines'][train].astype(np.float64) - a['targets'][train, None], axis=-1).mean(-1)
            train_scenes = sorted(set(a['folds'][train]))
            strongest = int(np.argmin(np.mean([errors[a['folds'][train] == f].mean(0) for f in train_scenes], axis=0)))
            if seed == reg['seeds'][0]:
                error = errors[:, cv]
                slices = {}
                for mode in ('row', 'scene'):
                    w = sampling_weights(a['folds'][train], mode).numpy()
                    static = stationary[train]
                    slices[mode] = {'stationary_rows': int(static.sum()),
                        'stationary_ADE_error_share': float((w[static] * error[static]).sum() / (w * error).sum()),
                        'stationary_logloss_share': float((w[static] * np.log1p(error[static])).sum() / (w * np.log1p(error)).sum()),
                        'stationary_log_derivative_coefficient_share': float((w[static] / (1 + error[static])).sum() / (w / (1 + error)).sum()),
                        'stationary_ADE_derivative_coefficient_share': float(w[static].sum())}
                audit[fold] = {'constant_feature_columns': np.flatnonzero(normalizer['constant']).tolist(),
                    'train_rows': len(train), 'held_rows': len(held), 'loss_weight_diagnostic': slices,
                    'coefficient_share_is_not_full_parameter_gradient': True}
            for arm in ARMS:
                key = f'{arm}_seed{seed}_fold{fold}'
                if args.trial and args.trial != key:
                    continue
                checkpoint, prediction_path, result_path = (
                    output / 'checkpoints' / (key + '.pt'), output / 'predictions' / (key + '.npz'),
                    output / 'trials' / (key + '.json'))
                trial_identity = {**identity, 'seed': seed, 'fold': fold, 'arm': arm}
                saved = json.loads(result_path.read_text()) if result_path.exists() else None
                if saved:
                    if saved['identity'] != trial_identity or file_digest(checkpoint) != saved['checkpoint_sha256'] or file_digest(prediction_path) != saved['prediction_sha256']:
                        raise ValueError('Completed trial changed')
                    if not args.replay:
                        trials.append(saved)
                        continue
                torch.manual_seed(seed)
                model = OfflineVisualForecast(receipt['geometry_dim'])
                if args.replay:
                    if saved is None:
                        raise ValueError('No completed trial to replay')
                    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
                    if state['identity'] != trial_identity or state['config'] != reg['training'] or state['step'] != reg['training']['updates']:
                        raise ValueError('Checkpoint replay identity mismatch')
                    model.load_state_dict(state['model'])
                else:
                    fit = fit_objective(model, train_batch, torch.from_numpy(train), a['folds'][train],
                        arm=arm, config=reg['training'], seed=seed, identity=trial_identity, checkpoint=checkpoint,
                        heartbeat=lambda v: heartbeat({'trial': key, **v}), stop_at=args.stop_at)
                    if not fit['complete']:
                        heartbeat({'trial': key, 'state': 'pilot_saved_no_held_eval', 'fit': fit})
                        return
                prediction = infer(model, held)
                if args.replay:
                    with np.load(prediction_path) as old:
                        exact = np.array_equal(old['held_indices'], held) and np.array_equal(old['prediction'], prediction)
                    if not exact:
                        raise ValueError('Prediction replay differs')
                    replay.append({'trial': key, 'prediction_exact': True})
                    continue
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(prediction_path, prediction=prediction, held_indices=held)
                predicted_error = np.linalg.norm(prediction.astype(np.float64) - a['targets'][held], axis=-1).mean(1)
                reference = np.linalg.norm(a['baselines'][held, cv].astype(np.float64) - a['targets'][held], axis=-1).mean(1)
                train_prediction = infer(model, train)
                train_error = np.linalg.norm(train_prediction.astype(np.float64) - a['targets'][train], axis=-1).mean(1)
                train_ref = errors[:, cv]
                train_ade = np.mean([train_error[a['folds'][train] == f].mean() for f in train_scenes])
                train_cv = np.mean([train_ref[a['folds'][train] == f].mean() for f in train_scenes])
                breakdown = {}
                for label, mask in [('stationary', stationary[held]), ('nonstationary', ~stationary[held]),
                                    *[(rid, records[held] == rid) for rid in sorted(set(records[held]))]]:
                    ids = held[mask]
                    breakdown[label] = forecast_metrics(prediction[mask], a['targets'][ids], a['baselines'][ids, cv],
                        a['scale'][ids], parent['easy_threshold']) if len(ids) else {'rows': 0, 'status': 'no_support'}
                result = {'identity': trial_identity, 'trial': key, 'arm': arm, 'seed': seed, 'fold': fold,
                    'result_source': 'fresh_run_native_torch_fit_cached_verified_inputs', 'fit': fit,
                    'train_rows': len(train), 'held_rows': len(held),
                    'vs_CV': forecast_metrics(prediction, a['targets'][held], a['baselines'][held, cv], a['scale'][held], parent['easy_threshold']),
                    'train_selected_strongest': receipt['baseline_names'][strongest],
                    'vs_train_selected_strongest': forecast_metrics(prediction, a['targets'][held], a['baselines'][held, strongest], a['scale'][held], parent['easy_threshold']),
                    'training_equal_scene_primary_gain_percent': float(100 * (1 - train_ade / train_cv)),
                    'training_row_logloss_gain_percent': float(100 * (1 - np.log1p(train_error).mean() / np.log1p(train_ref).mean())),
                    'binary_oracle_ADE_diagnostic': float(np.minimum(predicted_error, reference).mean()),
                    'slices': breakdown, 'checkpoint_sha256': file_digest(checkpoint), 'prediction_sha256': file_digest(prediction_path)}
                json_write(result_path, result)
                trials.append(result)
                heartbeat({'state': 'held_fit_evaluated', 'trial': key, 'vs_CV': result['vs_CV']})
    if args.replay:
        json_write(reports / 'replay.json', {'result_source': 'cached_verified_checkpoint_inference', 'identity': identity, 'trials': replay})
        heartbeat({'state': 'replay_complete', 'trials': len(replay)})
        return
    if args.trial:
        return
    if len(trials) != len(reg['seeds']) * 3 * len(ARMS):
        raise ValueError('Incomplete registered matrix')
    summary, contrasts = summarize(trials, reg)
    report = {'identity': identity, 'complete': True, 'result_source': 'fresh_run_real_training',
        'input_source': 'cached_verified_full_fit_inputs', 'rows': receipt['rows'], 'trials': trials,
        'summary': summary, 'paired_factor_contrasts': contrasts, 'training_weight_audit': audit,
        'primary_metric': 'past_normalized_ADE', 'aggregation': 'equal_physical_scene_and_seed',
        'observation_mode': 'offline_annotated_not_strict_sensor_as_of', 'parent_protocol_changed': False,
        'development_calibration_confirmation_opened': False, 'deployment': False, 'submission_ready': False,
        'stage5c_executed': False, 'smc_enabled': False}
    json_write(reports / 'report.json', report)
    lines = ['# Training Objective and Scene Sampling Comparison', '',
        'Fit-only, three historically used physical scenes. No independent confirmation or deployment.', '',
        '| Arm | Gain vs CV (%) | Exploratory scene interval (%) | Positive held fits | Easy gate passes | Binary oracle (%) |',
        '| --- | ---: | --- | ---: | ---: | ---: |']
    for arm, r in summary.items():
        lines.append(f'| {arm} | {r["vs_CV"]["gain_percent"]:.4f} | {r["vs_CV"]["exploratory_scene_ci95_percent"]} | {r["positive_held_folds"]}/9 | {r["easy_gate_passing_folds"]}/9 | {r["binary_oracle_gain_percent_diagnostic"]:.4f} |')
    lines += ['', 'Each fit has4,000updates. Oracle uses future labels for diagnosis only.',
        'Primary, cohort and role boundaries unchanged; no metric/seconds or real-time causality claim.', '']
    (reports / 'report.md').write_text('\n'.join(lines))
    heartbeat({'state': 'complete', 'models': len(trials), 'summary': summary})


if __name__ == '__main__':
    main()
