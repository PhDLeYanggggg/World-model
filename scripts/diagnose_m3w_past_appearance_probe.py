"""Post-fit train/held gap diagnosis; never selects a checkpoint or threshold."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if platform.system() == 'Darwin' and platform.machine() != 'arm64':
        raise SystemExit('Use native arm64 interpreter')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '4'
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    from src.evaluation.m3w_experiment_contract import file_digest
    from src.evaluation.m3w_stationary_scene_context import NAMES
    from src.evaluation.m3w_stationary_start_probe import score_probabilities
    from src.world_model.m3w_past_appearance_probe import PastAppearanceProbe, pixel_delta_to_native
    from scripts.run_m3w_stationary_start_probe import atomic_json

    report = json.loads(args.metrics.read_text())
    completion = json.loads((args.study / 'completion.json').read_text())
    registration_path = ROOT / 'configs/m3w_past_appearance_probe.json'
    registration = json.loads(registration_path.read_text())
    if (not report['complete_registered_budget']
            or completion['report_sha256'] != file_digest(args.metrics)
            or file_digest(registration_path) != report['identity']['registration_sha256']
            or file_digest(args.cache) != report['identity']['cache_sha256']):
        raise ValueError('Changed or incomplete original experiment')
    for path, digest in registration['bindings'].items():
        if file_digest(ROOT / path) != digest:
            raise ValueError('Changed registered source: ' + path)
    with np.load(args.cache, allow_pickle=False) as a:
        x = a['geometry'].copy()
        images = torch.tensor(a['rgb'].astype(np.float32) / 255 - .5)
        mask = torch.tensor(a['mask'].astype(np.float32))
        xy, h = torch.tensor(a['image_xy']), torch.tensor(a['homography'])
        rows = json.loads(str(a['rows_json']))
    source = ROOT / registration['source_cache']
    if file_digest(source) != registration['source_cache_sha256']:
        raise ValueError('Changed label source')
    with np.load(source, allow_pickle=False) as a:
        target, scale, label = [a[k].copy() for k in ('native', 'parent_scale', 'start')]
    names = NAMES + ['image_x_to_normal', 'image_x_to_tangent',
                     'image_y_to_normal', 'image_y_to_tangent']
    assert len(names) == x.shape[1]
    folds, trials = {}, []
    for trial in report['trials']:
        name = f'fold{trial["fold"]}_seed{trial["seed"]}_{trial["arm"]}'
        checkpoint = args.study / (name + '.pt')
        if file_digest(checkpoint) != trial['checkpoint_sha256']:
            raise ValueError('Changed checkpoint')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['identity'] != trial['trial_identity']:
            raise ValueError('Changed checkpoint identity')
        ids = np.asarray(state['identity']['train_rows'])
        held = np.asarray([i for i, row in enumerate(rows) if row['fit_fold'] == trial['fold']])
        mean = np.asarray(state['identity']['normalization_mean'])
        std = np.asarray(state['identity']['normalization_std'])
        standardized = (x - mean) / std
        geometry = torch.tensor(np.clip(standardized, -10, 10).astype(np.float32))
        fold = str(trial['fold'])
        if fold not in folds:
            feature_shift = [
                {'name': feature, 'train_std_floored': bool(std[j] <= 1e-6),
                 'held_clip_fraction': float(np.mean(np.abs(standardized[held, j]) > 10)),
                 'held_absolute_z_median': float(np.median(np.abs(standardized[held, j])))}
                for j, feature in enumerate(names)
            ]
            folds[fold] = {
                'train_rows': len(ids), 'held_rows': len(held),
                'train_agents': len({rows[i]['agent_id'] for i in ids}),
                'held_agents': len({rows[i]['agent_id'] for i in held}),
                'train_change_rate': float(label[ids].mean()),
                'held_change_rate': float(label[held].mean()),
                'held_any_clipped_feature_fraction': float(np.mean((np.abs(standardized[held]) > 10).any(1))),
                'feature_shift': feature_shift,
            }
        model = PastAppearanceProbe(x.shape[1])
        model.load_state_dict(state['model'])
        model.eval()
        predictions, probabilities = [], []
        with torch.no_grad():
            for start in range(0, len(ids), 32):
                batch = ids[start:start + 32]
                delta, logits = model(geometry[batch], images[batch], mask[batch], trial['arm'])
                predictions.append(pixel_delta_to_native(delta, xy[batch], h[batch]).numpy())
                probabilities.append(torch.sigmoid(logits).numpy())
        prediction, probability = np.concatenate(predictions), np.concatenate(probabilities)
        cv_error = np.linalg.norm(target[ids], axis=-1).mean(1) / scale[ids]
        error = np.linalg.norm(prediction - target[ids], axis=-1).mean(1) / scale[ids]
        prior = float((label[ids].sum() + 1) / (len(ids) + 2))
        trials.append({
            'fold': trial['fold'], 'seed': trial['seed'], 'arm': trial['arm'],
            'train_gain_vs_cv_pct': float(100 * (1 - error.mean() / cv_error.mean())),
            'train_classification': score_probabilities(label[ids], probability, prior=prior),
            'held_gain_vs_cv_pct': trial['trajectory_unrestricted']['gain_vs_cv_pct'],
            'held_classification': trial['classification'],
        })
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('New workspace output file required')
    atomic_json(output, {
        'result_source': 'fresh_post_fit_diagnosis_of_hash_verified_models',
        'original_report_sha256': file_digest(args.metrics),
        'diagnostic_code_sha256': file_digest(Path(__file__)),
        'folds': folds, 'trials': trials, 'training_performed': False,
        'held_results_used_for_selection': False,
        'causal_explanation_proven': False, 'new_deployment': False,
    })
    grouped = {}
    for fold in (0, 1):
        grouped[str(fold)] = {}
        for arm in registration['arms']:
            selected = [t for t in trials if t['fold'] == fold and t['arm'] == arm]
            grouped[str(fold)][arm] = {
                'train_gain_mean': float(np.mean([t['train_gain_vs_cv_pct'] for t in selected])),
                'held_gain_mean': float(np.mean([t['held_gain_vs_cv_pct'] for t in selected])),
                'train_brier_lift_mean': float(np.mean([t['train_classification']['brier_lift_over_prior'] for t in selected])),
            }
    print(json.dumps({'seed_means': grouped, 'folds': folds}), flush=True)


if __name__ == '__main__':
    main()
