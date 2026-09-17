"""Descriptive same-mask decomposition of registered frozen-model controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_stationary_scene_context import NAMES
from src.evaluation.m3w_appearance_support_control import fit_box
from scripts.run_m3w_stationary_start_probe import atomic_json


def masked_errors(prediction, target, scale, switch):
    return np.linalg.norm(prediction * switch[:, None, None] - target, axis=-1).mean(1) / scale


def decompose(original, treated, target, scale, original_switch, treated_switch):
    base = masked_errors(original, target, scale, original_switch)
    same_mask = masked_errors(treated, target, scale, original_switch)
    actual = masked_errors(treated, target, scale, treated_switch)
    floor = np.linalg.norm(target, axis=-1).mean(1) / scale
    prediction_effect = float((same_mask - base).mean())
    gate_effect = float((actual - same_mask).mean())
    total = float((actual - base).mean())
    if not np.isclose(prediction_effect + gate_effect, total, atol=1e-10, rtol=1e-12):
        raise ValueError('Error decomposition does not close')
    return {'fixed_original_mask_gain_pct': float(100 * (1 - same_mask.mean() / floor.mean())),
            'actual_gain_pct': float(100 * (1 - actual.mean() / floor.mean())),
            'prediction_component_error_change': prediction_effect,
            'gate_component_error_change': gate_effect,
            'total_error_change': total,
            'original_switch_count': int(original_switch.sum()),
            'treated_switch_count': int(treated_switch.sum()),
            'same_mask_count': int(original_switch.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reg = json.loads(args.registration.read_text())
    for path, digest in reg['bindings'].items():
        if file_digest(ROOT / path) != digest:
            raise ValueError('Changed control source')
    complete = json.loads((args.study / 'completion.json').read_text())
    metrics_path = args.report_dir / 'metrics.json'
    if (complete['report_sha256'] != file_digest(metrics_path)
            or complete['identity']['registration_sha256'] != file_digest(args.registration)):
        raise ValueError('Changed control report/registration')
    report = json.loads(metrics_path.read_text())
    original_reg = json.loads((ROOT / reg['original_registration']).read_text())
    source = ROOT / original_reg['source_cache']
    if file_digest(source) != original_reg['source_cache_sha256']:
        raise ValueError('Changed source labels')
    with np.load(source, allow_pickle=False) as a:
        target, scale = a['native'].copy(), a['parent_scale'].copy()
    with np.load(ROOT / reg['cache'], allow_pickle=False) as a:
        x, mask = a['geometry'].copy(), a['mask'].copy()
        rows = json.loads(str(a['rows_json']))
    names = NAMES + ['image_x_to_normal', 'image_x_to_tangent',
                     'image_y_to_normal', 'image_y_to_tangent']
    folds, trials = {}, []
    for trial in report['trials']:
        name = f'fold{trial["fold"]}_seed{trial["seed"]}_{trial["arm"]}'
        receipt, predictions = args.study / (name + '.json'), args.study / (name + '.npz')
        record = json.loads(receipt.read_text())
        if (file_digest(receipt) != complete['receipts'][name]
                or file_digest(predictions) != record['prediction_sha256']):
            raise ValueError('Changed row predictions/receipt')
        with np.load(predictions, allow_pickle=False) as a:
            held = a['held'].copy()
            old = a['original_prediction'].copy()
            original_switch = (a['original_probability'] >= .9) & mask[held].all(1)
            for treatment in ('jacobian_box', 'all_feature_box'):
                current_switch = (a[treatment + '_probability'] >= .9) & mask[held].all(1)
                value = decompose(old, a[treatment + '_prediction'], target[held], scale[held],
                                  original_switch, current_switch)
                expected = trial['treatments'][treatment]['guarded']['gain_vs_cv_pct']
                if not np.isclose(value['actual_gain_pct'], expected, atol=1e-10, rtol=0):
                    raise ValueError('Original score not reproduced')
                trials.append({'fold': trial['fold'], 'seed': trial['seed'], 'arm': trial['arm'],
                               'treatment': treatment, **value})
        fold = str(trial['fold'])
        if fold not in folds:
            train = np.array([i for i, r in enumerate(rows) if r['fit_fold'] != trial['fold']])
            lower, upper = fit_box(x[train])
            outside = (x[held] < lower) | (x[held] > upper)
            folds[fold] = {'inside_box_rows': int((~outside.any(1)).sum()),
                           'held_rows': len(held),
                           'feature_outside_fraction': {key: float(outside[:, j].mean()) for j, key in enumerate(names)}}
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or output.exists():
        raise ValueError('Use a new workspace output file')
    grouped = {}
    for fold in (0, 1):
        grouped[str(fold)] = {}
        for arm in ('geometry', 'current_rgb', 'past_rgb'):
            grouped[str(fold)][arm] = {}
            for treatment in ('jacobian_box', 'all_feature_box'):
                records = [t for t in trials if t['fold'] == fold and t['arm'] == arm and t['treatment'] == treatment]
                grouped[str(fold)][arm][treatment] = {
                    k: float(np.mean([t[k] for t in records])) for k in
                    ('fixed_original_mask_gain_pct', 'actual_gain_pct', 'prediction_component_error_change',
                     'gate_component_error_change', 'total_error_change')}
    atomic_json(output, {'result_source': 'fresh_run_descriptive_decomposition_of_verified_cached_predictions',
                'code_sha256': file_digest(Path(__file__)), 'original_control_report_sha256': file_digest(metrics_path),
                'folds': folds, 'trials': trials, 'seed_means': grouped,
                'interpretation': 'negative error components mean less harm; path-dependent algebra, not causal identification',
                'model_or_threshold_selection': False, 'new_training': False, 'new_deployment': False})
    print(json.dumps({'seed_means': grouped, 'folds': folds}), flush=True)


if __name__ == '__main__':
    main()
