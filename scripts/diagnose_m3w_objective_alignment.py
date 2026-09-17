"""Post-fit error decomposition; labels never become model inputs or thresholds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_offline_visual_data import json_write, verify_registration


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', type=Path, required=True)
    args = p.parse_args()
    reg = json.loads(args.registration.read_text())
    for name, digest in reg['bindings'].items():
        if file_digest(ROOT / name) != digest:
            raise ValueError('Registered source changed')
    parent, _ = verify_registration(ROOT, ROOT / reg['input_registration'])
    data, output, reports = ROOT / parent['output'] / 'inputs', ROOT / reg['output'], ROOT / reg['reports']
    report = json.loads((reports / 'report.json').read_text())
    if not report['complete'] or len(report['trials']) != 45:
        raise ValueError('All registered fits must complete before this diagnostic')
    manifest = json.loads((data / 'data_manifest.json').read_text())
    for name, digest in manifest['arrays'].items():
        if file_digest(data / name) != digest:
            raise ValueError('Input changed')
    a = {k: np.load(data / (k + '.npy'), mmap_mode='r') for k in ('geometry', 'targets', 'baselines', 'folds', 'scale')}
    recording = np.asarray([r['recording'] for r in json.loads((data / 'rows.json').read_text())])
    stationary = np.all(a['geometry'][:, :16] == 0, axis=1)
    future_static = np.all(a['targets'] == 0, axis=(1, 2))
    cv = manifest['baseline_names'].index('constant_velocity_causal_fd')
    rows, domains = [], {}
    for fold in range(3):
        ids = np.flatnonzero((a['folds'] == fold) & stationary)
        domains[fold] = {'stationary_history_rows': len(ids), 'stationary_past_and_future_rows': int(future_static[ids].sum()),
            'future_max_radius_normalized_quantiles': np.quantile(np.linalg.norm(a['targets'][ids], axis=-1).max(1),
                [0, .25, .5, .75, .95, 1]).tolist() if len(ids) else None,
            'evaluation_scale_quantiles_dataset_local': np.quantile(a['scale'][ids], [0, .5, 1]).tolist() if len(ids) else None}
    for trial in report['trials']:
        path = output / 'predictions' / (trial['trial'] + '.npz')
        if file_digest(path) != trial['prediction_sha256']:
            raise ValueError('Frozen predictions changed')
        with np.load(path) as saved:
            ids, pred = saved['held_indices'], saved['prediction'].astype(np.float64)
        target = a['targets'][ids].astype(np.float64)
        reference = np.linalg.norm(a['baselines'][ids, cv].astype(np.float64) - target, axis=-1).mean(1)
        error = np.linalg.norm(pred - target, axis=-1).mean(1)
        extra, static = np.maximum(error - reference, 0), stationary[ids]
        pieces = {}
        for name, mask in [('stationary_history', static), ('moving_history', ~static),
                           ('stationary_past_and_future', static & future_static[ids]),
                           ('stationary_past_future_movement', static & ~future_static[ids])]:
            if not mask.any():
                pieces[name] = {'rows': 0, 'status': 'no_support'}
                continue
            displacement = np.linalg.norm(pred[mask], axis=-1).mean(1)
            truth = np.linalg.norm(target[mask], axis=-1).mean(1)
            pieces[name] = {'rows': int(mask.sum()), 'mean_ADE': float(error[mask].mean()),
                'mean_CV_ADE': float(reference[mask].mean()), 'mean_harm': float((error[mask] - reference[mask]).mean()),
                'share_of_total_positive_error_increase': float(extra[mask].sum() / extra.sum()) if extra.sum() > 0 else None,
                'mean_predicted_distance_from_current_normalized': float(displacement.mean()),
                'mean_observed_future_distance_from_current_normalized': float(truth.mean()),
                'mean_predicted_distance_dataset_local': float((displacement * a['scale'][ids[mask]]).mean()),
                'mean_observed_future_distance_dataset_local': float((truth * a['scale'][ids[mask]]).mean())}
        native = {}
        for rid in sorted(set(recording[ids])):
            mask = recording[ids] == rid
            native_error = float((error[mask] * a['scale'][ids[mask]]).mean())
            native_cv = float((reference[mask] * a['scale'][ids[mask]]).mean())
            native[rid] = {'rows': int(mask.sum()), 'model_ADE': native_error, 'CV_ADE': native_cv,
                'gain_percent': 100 * (1 - native_error / native_cv) if native_cv > 0 else None,
                'coordinate_claim': 'dataset_local_unverified_diagnostic_only'}
        rows.append({'trial': trial['trial'], 'arm': trial['arm'], 'seed': trial['seed'], 'fold': trial['fold'],
                     'slices': pieces, 'native_by_recording_diagnostic': native})
    result = {'result_source': 'fresh_run_decomposition_of_hash_verified_frozen_predictions',
        'registration_sha256': file_digest(args.registration), 'code_sha256': file_digest(Path(__file__)),
        'adaptive_fit_only_diagnostic': True, 'future_motion_categories_are_labels_only': True,
        'new_training_or_model_selection': False, 'domains': domains, 'trials': rows,
        'development_calibration_confirmation_opened': False,
        'motion_labels_are_annotation_categories_not_physical_gold': True}
    json_write(reports / 'failure_decomposition.json', result)
    print(json.dumps({'domains': domains, 'trials': len(rows)}))


if __name__ == '__main__':
    main()
