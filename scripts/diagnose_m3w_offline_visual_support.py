"""Prespecified frozen-model diagnostic, not a new selected deployment policy."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Use arm64 .venv-pytorch')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '4'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_constant_feature_support import clamp_constant_support
from src.world_model.m3w_offline_visual_data import verify_registration, json_write
from src.world_model.m3w_offline_visual_forecast import OfflineVisualForecast, forecast_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    args = p.parse_args()
    control = json.loads(args.registration.read_text())
    for name, digest in control['bindings'].items():
        if file_digest(ROOT / name) != digest:
            raise ValueError('Changed support-diagnostic dependency: ' + name)
    reg, _ = verify_registration(ROOT, ROOT / control['experiment_registration'])
    output, reports = ROOT / reg['output'], ROOT / reg['reports']
    report = json.loads((reports / 'report.json').read_text())
    if not report['complete'] or len(report['trials']) != 36:
        raise ValueError('The complete frozen experiment must exist before diagnosis')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    a = {k: np.load(output / 'inputs' / (k + '.npy'), mmap_mode='r')
         for k in ('geometry', 'targets', 'baselines', 'scale', 'image_rows', 'rgb', 'coverage', 'folds')}
    receipt = json.loads((output / 'inputs/data_manifest.json').read_text())
    for name, digest in receipt['arrays'].items():
        if file_digest(output / 'inputs' / name) != digest:
            raise ValueError('Input changed')
    cv = receipt['baseline_names'].index('constant_velocity_causal_fd')
    results, fold_data = [], {}
    for fold in range(3):
        train, held = a['geometry'][a['folds'] != fold], np.flatnonzero(a['folds'] == fold)
        repaired, constant = clamp_constant_support(train, a['geometry'][held])
        training_replay, _ = clamp_constant_support(train, train)
        if not np.array_equal(train, training_replay):
            raise ValueError('Diagnostic must not alter any training input')
        mean, std = train.mean(0), np.maximum(train.std(0), 1e-6)
        original = np.clip((a['geometry'][held] - mean) / std, -10, 10)
        repaired = np.clip((repaired - mean) / std, -10, 10)
        fold_data[fold] = {'constant_columns': np.flatnonzero(constant).tolist(),
            'held_changed_rows': int((original != repaired).any(1).sum()),
            'all_training_rows_unchanged': True, 'held_rows': len(held)}
        for trial in report['trials']:
            if trial['fold'] != fold:
                continue
            path = output / 'checkpoints' / (trial['trial'] + '.pt')
            if file_digest(path) != trial['checkpoint_sha256']:
                raise ValueError('Checkpoint changed')
            state = torch.load(path, map_location='cpu', weights_only=False)
            model = OfflineVisualForecast(receipt['geometry_dim'])
            model.load_state_dict(state['model'])
            model.eval()
            predictions = [[], []]
            with torch.no_grad():
                for start in range(0, len(held), 128):
                    ids = held[start:start + 128]
                    source_rows = a['image_rows'][ids]
                    rgb = torch.from_numpy(a['rgb'][source_rows].astype(np.float32) / 255.)
                    cov = torch.from_numpy(a['coverage'][source_rows, None].astype(np.float32) / 9.)
                    baseline = torch.from_numpy(a['baselines'][ids, cv].copy())
                    for destination, source in zip(predictions, (original, repaired)):
                        geometry = torch.from_numpy(source[start:start + 128].copy())
                        destination.append(model(geometry, rgb, cov, baseline, trial['arm']).numpy())
            before, after = [np.concatenate(v) for v in predictions]
            with np.load(output / 'predictions' / (trial['trial'] + '.npz')) as saved:
                if not np.array_equal(before, saved['prediction']):
                    raise ValueError('Frozen prediction replay mismatch')
            metrics = forecast_metrics(after, a['targets'][held], a['baselines'][held, cv],
                                       a['scale'][held], reg['easy_threshold'])
            result = {'trial': trial['trial'], 'arm': trial['arm'], 'fold': fold, 'seed': trial['seed'],
                'before': trial['vs_CV'], 'after': metrics, 'baseline_prediction_replay_exact': True,
                'max_prediction_change': float(np.max(abs(after - before)))}
            results.append(result)
            print(json.dumps({'trial': trial['trial'], 'after_gain': metrics['improvement_percent']}), flush=True)
    result = {'result_source': 'fresh_run_frozen_model_diagnostic_not_retraining',
        'registration_sha256': file_digest(args.registration), 'fold_support': fold_data, 'trials': results,
        'held_fit_adaptive_diagnostic': True, 'deployment_selected': False,
        'original_experiment_unchanged': True, 'new_test_or_development_access': False}
    json_write(reports / 'constant_support_diagnostic.json', result)


if __name__ == '__main__':
    main()
