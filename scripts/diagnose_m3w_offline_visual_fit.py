"""Measure final training fit to distinguish fitting failure from transfer damage."""
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
from src.world_model.m3w_offline_visual_data import verify_registration, json_write
from src.world_model.m3w_offline_visual_forecast import OfflineVisualForecast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    args = parser.parse_args()
    reg, _ = verify_registration(ROOT, args.registration)
    output, reports = ROOT / reg['output'], ROOT / reg['reports']
    report = json.loads((reports / 'report.json').read_text())
    if not report['complete'] or len(report['trials']) != 36:
        raise ValueError('Require all frozen fits')
    receipt = json.loads((output / 'inputs/data_manifest.json').read_text())
    for name, digest in receipt['arrays'].items():
        if file_digest(output / 'inputs' / name) != digest:
            raise ValueError('Data hash changed')
    a = {k: np.load(output / 'inputs' / (k + '.npy'), mmap_mode='r') for k in
         ('geometry', 'targets', 'baselines', 'scale', 'folds', 'image_rows', 'rgb', 'coverage')}
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cv = receipt['baseline_names'].index('constant_velocity_causal_fd')
    meta = json.loads((output / 'inputs/rows.json').read_text())
    stationary = np.all(a['geometry'][:, :16] == 0, axis=1)
    cv_error = np.linalg.norm(a['baselines'][:, cv].astype(np.float64) - a['targets'], axis=-1).mean(1)
    concentration = {}
    for fold in range(3):
        mask = a['folds'] == fold
        source_ids = {(r['recording'], r['agent']) for i, r in enumerate(meta) if mask[i] and stationary[i]}
        concentration[str(fold)] = {'exact_stationary_rows': int((mask & stationary).sum()),
            'exact_stationary_source_ids': len(source_ids),
            'CV_primary_error_share_from_stationary': float(cv_error[mask & stationary].sum() / cv_error[mask].sum())}
    concentration['equal_scene_primary_error_share_from_stationary'] = float(
        sum(cv_error[(a['folds'] == f) & stationary].sum() / (a['folds'] == f).sum() for f in range(3)) /
        sum(cv_error[a['folds'] == f].mean() for f in range(3)))
    results = []
    for trial in report['trials']:
        ids = np.flatnonzero(a['folds'] != trial['fold'])
        mean, std = a['geometry'][ids].mean(0), np.maximum(a['geometry'][ids].std(0), 1e-6)
        geometry = np.clip((a['geometry'][ids] - mean) / std, -10, 10)
        path = output / 'checkpoints' / (trial['trial'] + '.pt')
        if file_digest(path) != trial['checkpoint_sha256']:
            raise ValueError('Checkpoint hash changed')
        state = torch.load(path, map_location='cpu', weights_only=False)
        model = OfflineVisualForecast(receipt['geometry_dim'])
        model.load_state_dict(state['model'])
        model.eval()
        errors = []
        with torch.no_grad():
            for start in range(0, len(ids), 128):
                batch = ids[start:start + 128]
                source_rows = a['image_rows'][batch]
                pred = model(torch.from_numpy(geometry[start:start + 128].copy()),
                    torch.from_numpy(a['rgb'][source_rows].astype(np.float32) / 255.),
                    torch.from_numpy(a['coverage'][source_rows, None].astype(np.float32) / 9.),
                    torch.from_numpy(a['baselines'][batch, cv].copy()), trial['arm']).numpy()
                errors.append(np.linalg.norm(pred.astype(np.float64) - a['targets'][batch], axis=-1).mean(1))
        error = np.concatenate(errors)
        ref = np.linalg.norm(a['baselines'][ids, cv].astype(np.float64) - a['targets'][ids], axis=-1).mean(1)
        scene_error = np.mean([error[a['folds'][ids] == f].mean() for f in set(a['folds'][ids])])
        scene_ref = np.mean([ref[a['folds'][ids] == f].mean() for f in set(a['folds'][ids])])
        r = {'trial': trial['trial'], 'arm': trial['arm'], 'seed': trial['seed'], 'held_fold': trial['fold'],
            'training_rows': len(ids), 'training_log1p_ADE': float(np.log1p(error).mean()),
            'training_CV_log1p_ADE': float(np.log1p(ref).mean()),
            'training_logloss_reduction_percent': float(100 * (1 - np.log1p(error).mean() / np.log1p(ref).mean())),
            'training_equal_scene_primary_gain_percent': float(100 * (1 - scene_error / scene_ref)),
            'held_fit_primary_gain_percent': trial['vs_CV']['improvement_percent']}
        with np.load(output / 'predictions' / (trial['trial'] + '.npz')) as saved:
            held, prediction = saved['held_indices'], saved['prediction']
        held_error = np.linalg.norm(prediction.astype(np.float64) - a['targets'][held], axis=-1).mean(1)
        held_ref = np.linalg.norm(a['baselines'][held, cv].astype(np.float64) - a['targets'][held], axis=-1).mean(1)
        r['binary_CV_candidate_oracle_ADE_diagnostic'] = float(np.minimum(held_error, held_ref).mean())
        r['binary_CV_candidate_oracle_gain_percent'] = float(100 * (1 - np.minimum(held_error, held_ref).mean() / held_ref.mean()))
        r['fraction_candidate_better_than_CV'] = float((held_error < held_ref).mean())
        r['oracle_not_an_inference_model'] = True
        results.append(r)
        print(json.dumps(r), flush=True)
    json_write(reports / 'training_fit_diagnostic.json', {'result_source': 'fresh_run_frozen_checkpoint_training_inference',
        'code_sha256': file_digest(Path(__file__)), 'registration_sha256': file_digest(args.registration),
        'adaptive_fit_only_diagnosis': True, 'results': results, 'label_error_concentration': concentration,
        'new_training_or_selection': False})


if __name__ == '__main__':
    main()
