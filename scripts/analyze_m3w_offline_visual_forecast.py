"""Prespecified paired scene/seed analysis; overlapping windows are not IID."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from src.world_model.m3w_offline_visual_data import verify_registration, json_write
from src.world_model.m3w_offline_visual_forecast import forecast_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', required=True, type=Path)
    args = p.parse_args()
    reg, _ = verify_registration(ROOT, args.registration)
    output, reports = ROOT / reg['output'], ROOT / reg['reports']
    report = json.loads((reports / 'report.json').read_text())
    arrays = {k: np.load(output / 'inputs' / (k + '.npy'), mmap_mode='r')
              for k in ('targets', 'baselines', 'scale', 'folds', 'image_rows', 'coverage')}
    meta = json.loads((output / 'inputs/rows.json').read_text())
    receipt = json.loads((output / 'inputs/data_manifest.json').read_text())
    cv = receipt['baseline_names'].index('constant_velocity_causal_fd')
    lookup, breakdown = {}, []
    for trial in report['trials']:
        with np.load(output / 'predictions' / (trial['trial'] + '.npz')) as saved:
            pred, ids = saved['prediction'], saved['held_indices']
        lookup[(trial['arm'], trial['seed'], trial['fold'])] = trial['vs_CV']['primary_ADE']
        for rid in sorted({meta[i]['recording'] for i in ids}):
            mask = np.array([meta[i]['recording'] == rid for i in ids])
            actual = ids[mask]
            metrics = forecast_metrics(pred[mask], arrays['targets'][actual], arrays['baselines'][actual, cv],
                                       arrays['scale'][actual], reg['easy_threshold'])
            breakdown.append({'trial': trial['trial'], 'recording': rid, **metrics})
    paired = {}
    scene_names = sorted(reg['scene_folds'], key=reg['scene_folds'].get)
    for arm, reference in [('mask_only', 'geometry'), ('current_rgb', 'geometry'),
                            ('past_rgb', 'geometry'), ('current_rgb', 'mask_only'),
                            ('past_rgb', 'mask_only'), ('past_rgb', 'current_rgb')]:
        model = np.array([[lookup[(arm, seed, f)] for f in range(3)] for seed in reg['seeds']])
        base = np.array([[lookup[(reference, seed, f)] for f in range(3)] for seed in reg['seeds']])
        rng = np.random.default_rng(917)
        draws = rng.integers(0, 3, size=(reg['bootstrap_resamples'], 3))
        site_model, site_base = model.mean(0), base.mean(0)
        gains = 100 * (1 - site_model[draws].mean(1) / site_base[draws].mean(1))
        paired[arm + '_vs_' + reference] = {
            'equal_scene_seed_gain_percent': float(100 * (1 - model.mean() / base.mean())),
            'scene_bootstrap_ci95_percent': np.quantile(gains, [.025, .975]).tolist(),
            'per_seed_gain_percent': (100 * (1 - model.mean(1) / base.mean(1))).tolist(),
            'per_scene_seedmean_gain_percent': dict(zip(scene_names, (100 * (1 - site_model / site_base)).tolist())),
            'independent_scene_units': 3, 'bootstrap_resamples': reg['bootstrap_resamples'],
            'overlapping_windows_bootstrapped_as_IID': False,
            'confirmatory': False}
    result = {'result_source': 'fresh_run_analysis_of_frozen_fit_predictions',
        'paired': paired, 'per_recording': breakdown,
        'caution': 'Only three previously exposed fit scenes; bootstrap does not establish generalization or safety.',
        'no_threshold_or_checkpoint_selected': True}
    json_write(reports / 'paired_analysis.json', result)
    lines = ['# Paired Visual Contribution and Uncertainty', '',
        'Equal physical-scene means, then equal training-seed means. 2,000 paired scene resamples.',
        'Only three historical fit scenes: exploratory intervals, not a deployment certificate.', '',
        '| Contrast | Gain (%) | Scene-bootstrap 95% interval (%) | Seed gains (%) |',
        '| --- | ---: | --- | --- |']
    for name, r in paired.items():
        lines.append(f'| {name} | {r["equal_scene_seed_gain_percent"]:.4f} | {r["scene_bootstrap_ci95_percent"]} | {r["per_seed_gain_percent"]} |')
    lines += ['', 'The denominator is the named neural reference, not CV. See the main report for absolute baseline comparisons.',
              'A less damaging visual model is not automatically better than a strong causal baseline.', '']
    (reports / 'paired_analysis.md').write_text('\n'.join(lines))
    print(json.dumps(result['paired'], indent=2))


if __name__ == '__main__':
    main()
