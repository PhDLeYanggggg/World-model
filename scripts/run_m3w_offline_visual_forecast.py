"""Full fit-cohort matched visual experiment. No development/test selection."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use .venv-pytorch/bin/python (arm64)')
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '4'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import verify_registration, build_inputs, json_write
from src.world_model.m3w_offline_visual_forecast import ARMS, OfflineVisualForecast, fit_model, forecast_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--trial', type=str)
    p.add_argument('--stop-at', type=int)
    p.add_argument('--replay', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    reg, parent = verify_registration(ROOT, args.registration)
    output, reports = ROOT / reg['output'], ROOT / reg['reports']
    if not output.is_relative_to(ROOT / 'data/stage_cvpr2027_experiments'):
        raise ValueError('Private data/checkpoints must remain in ignored cache')
    identity = {'registration_sha256': file_digest(args.registration), 'parent_protocol_sha256': parent.digest}
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / 'run_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Existing experiment identity differs')
    json_write(identity_path, identity)
    started = time.monotonic()
    def heartbeat(value):
        json_write(output / 'heartbeat.json', {'pid': os.getpid(), 'elapsed_seconds': time.monotonic() - started,
                   'timestamp_unix': time.time(), **value})
        print(json.dumps(value), flush=True)
    receipt = build_inputs(ROOT, reg, parent, output / 'inputs', identity, heartbeat)
    json_write(reports / 'data_receipt.json', receipt)
    if args.prepare_only:
        heartbeat({'state': 'inputs_complete', 'rows': receipt['rows'], 'bytes': receipt['bytes']})
        return
    a = {name: np.load(output / 'inputs' / (name + '.npy'), mmap_mode='r')
         for name in ('geometry', 'targets', 'baselines', 'scale', 'image_rows', 'rgb', 'coverage', 'folds')}
    targets = np.asarray(a['targets'])
    cv = receipt['baseline_names'].index('constant_velocity_causal_fd')
    trials, replay = [], []
    for seed in reg['seeds']:
        for fold in sorted(reg['scene_folds'].values()):
            train = np.flatnonzero(a['folds'] != fold)
            held = np.flatnonzero(a['folds'] == fold)
            train_folds = sorted(set(a['folds'][train]))
            if len(train_folds) != 2 or len(set(a['folds'][held])) != 1:
                raise ValueError('Physical-scene fold isolation failed')
            mean = a['geometry'][train].mean(0)
            std = np.maximum(a['geometry'][train].std(0), 1e-6)
            standardized = (a['geometry'] - mean) / std
            x = torch.from_numpy(np.clip(standardized, -10, 10).astype(np.float32))
            baseline_errors = np.linalg.norm(a['baselines'][train] - targets[train, None], axis=-1).mean(-1)
            by_scene = [baseline_errors[a['folds'][train] == f].mean(0) for f in train_folds]
            strongest = int(np.argmin(np.mean(by_scene, 0)))
            def batch(ids):
                ids = ids.numpy() if isinstance(ids, torch.Tensor) else np.asarray(ids)
                source_rows = a['image_rows'][ids]
                rgb = torch.from_numpy(a['rgb'][source_rows].astype(np.float32) / 255.)
                coverage = torch.from_numpy(a['coverage'][source_rows, None].astype(np.float32) / 9.)
                return (x[ids], rgb, coverage, torch.from_numpy(a['baselines'][ids, cv].copy()),
                        torch.from_numpy(targets[ids].copy()))
            for arm in ARMS:
                key = f'{arm}_seed{seed}_fold{fold}'
                if args.trial and key != args.trial:
                    continue
                trial_path = output / 'trials' / (key + '.json')
                checkpoint = output / 'checkpoints' / (key + '.pt')
                pred_path = output / 'predictions' / (key + '.npz')
                trial_identity = {**identity, 'data_manifest_sha256': file_digest(output / 'inputs/data_manifest.json'),
                                  'seed': seed, 'fold': fold, 'arm': arm}
                if trial_path.exists() and not args.replay:
                    saved = json.loads(trial_path.read_text())
                    if saved['identity'] != trial_identity or saved['checkpoint_sha256'] != file_digest(checkpoint) or saved['prediction_sha256'] != file_digest(pred_path):
                        raise ValueError('Completed trial changed')
                    trials.append(saved)
                    continue
                torch.manual_seed(seed)
                model = OfflineVisualForecast(receipt['geometry_dim'])
                if args.replay:
                    saved = json.loads(trial_path.read_text())
                    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
                    if state['identity'] != trial_identity or state['step'] != reg['training']['updates']:
                        raise ValueError('Replay checkpoint mismatch')
                    model.load_state_dict(state['model'])
                    fit = {'complete': True}
                else:
                    fit = fit_model(model, batch, torch.from_numpy(train), arm=arm,
                        config=reg['training'], seed=seed, identity=trial_identity, checkpoint=checkpoint,
                        heartbeat=lambda v: heartbeat({'trial': key, **v}), stop_at=args.stop_at)
                if not fit['complete']:
                    heartbeat({'trial': key, 'state': 'pilot_checkpoint_saved_no_held_evaluation', **fit})
                    return
                model.eval()
                predictions = []
                with torch.no_grad():
                    for start in range(0, len(held), 128):
                        values = batch(held[start:start + 128])
                        predictions.append(model(*values[:4], arm).numpy())
                prediction = np.concatenate(predictions)
                if args.replay:
                    with np.load(pred_path) as prior:
                        equal = np.array_equal(prediction, prior['prediction']) and np.array_equal(held, prior['held_indices'])
                    replay.append({'trial': key, 'prediction_exact': equal})
                    if not equal:
                        raise ValueError('Checkpoint replay differs')
                    continue
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(pred_path, prediction=prediction, held_indices=held)
                metrics = forecast_metrics(prediction, targets[held], a['baselines'][held, cv],
                    a['scale'][held], reg['easy_threshold'])
                strong_metrics = forecast_metrics(prediction, targets[held], a['baselines'][held, strongest],
                    a['scale'][held], reg['easy_threshold'])
                result = {'identity': trial_identity, 'trial': key, 'seed': seed, 'fold': fold, 'arm': arm,
                    'result_source': 'fresh_run_torch_cpu_four_threads', 'train_rows': len(train), 'held_rows': len(held),
                    'parameters': sum(p.numel() for p in model.parameters()), 'fit': fit,
                    'vs_CV': metrics, 'train_selected_strongest': receipt['baseline_names'][strongest],
                    'vs_train_selected_strongest': strong_metrics,
                    'held_rows_with_any_clipped_geometry': float(np.any(abs(standardized[held]) > 10, axis=1).mean()),
                    'checkpoint_sha256': file_digest(checkpoint), 'prediction_sha256': file_digest(pred_path)}
                json_write(trial_path, result)
                trials.append(result)
                heartbeat({'trial': key, 'state': 'held_fit_evaluated', 'vs_CV': metrics})
    if args.replay:
        json_write(reports / 'replay.json', {'result_source': 'cached_verified_exact_checkpoint_inference',
                                           'identity': identity, 'trials': replay})
        heartbeat({'state': 'replay_complete', 'trials': len(replay)})
        return
    if args.trial:
        return
    expected = len(reg['seeds']) * len(reg['scene_folds']) * len(ARMS)
    if len(trials) != expected:
        raise ValueError('Incomplete registered trial matrix')
    summary = {}
    for arm in ARMS:
        arm_trials = [t for t in trials if t['arm'] == arm]
        model_mean = np.mean([t['vs_CV']['primary_ADE'] for t in arm_trials])
        cv_mean = np.mean([t['vs_CV']['reference_ADE'] for t in arm_trials])
        strongest_mean = np.mean([t['vs_train_selected_strongest']['reference_ADE'] for t in arm_trials])
        summary[arm] = {'equal_scene_seed_gain_vs_CV_percent': float(100 * (1 - model_mean / cv_mean)),
            'equal_scene_seed_gain_vs_train_selected_strongest_percent': float(100 * (1 - model_mean / strongest_mean)),
            'positive_CV_folds': sum(t['vs_CV']['improvement_percent'] > 0 for t in arm_trials),
            'fit_seconds': sum(t['fit']['fit_seconds'] for t in arm_trials)}
    report = {'identity': identity, 'result_source': 'fresh_run_full_fit_cohort_visual_comparison',
        'complete': True, 'rows': receipt['rows'], 'trials': trials, 'summary': summary,
        'primary_metric': 'past_normalized_ADE', 'aggregation': 'equal_physical_scene_and_seed',
        'independent_physical_scenes': len(reg['scene_folds']), 'seeds': reg['seeds'],
        'development_calibration_confirmation_opened': False, 'observation_mode': reg['observation_mode'],
        'strict_online_or_metric_claim': False, 'deployment': False, 'submission_ready': False,
        'stage5c_executed': False, 'smc_enabled': False}
    json_write(reports / 'report.json', report)
    lines = ['# Offline Annotated Visual Forecasting: Full Fit Cohort', '',
        'Exploratory leave-one-physical-fit-scene-out evaluation. Not independent test evidence.', '',
        '| Arm | Equal-scene/seed gain vs CV (%) | Vs training-selected strongest (%) | Positive CV folds |',
        '| --- | ---: | ---: | ---: |']
    for arm, r in summary.items():
        lines.append(f'| {arm} | {r["equal_scene_seed_gain_vs_CV_percent"]:.4f} | {r["equal_scene_seed_gain_vs_train_selected_strongest_percent"]:.4f} | {r["positive_CV_folds"]}/9 |')
    lines += ['', '| Arm | Seed | Held fold | Primary gain vs CV (%) | Easy degradation (%) |',
              '| --- | ---: | ---: | ---: | ---: |']
    for t in trials:
        lines.append(f'| {t["arm"]} | {t["seed"]} | {t["fold"]} | {t["vs_CV"]["improvement_percent"]:.4f} | {t["vs_CV"]["easy_degradation_percent"]} |')
    lines += ['', 'Offline interpolated annotations are not strict online observations. Zara03 missing imagery is retained.',
        'Three physical fit scenes are not a confirmation set. No development/test tuning, metric/seconds or deployment claim.', '']
    (reports / 'report.md').write_text('\n'.join(lines))
    heartbeat({'state': 'complete', 'trials': len(trials), 'summary': summary})


if __name__ == '__main__':
    main()
