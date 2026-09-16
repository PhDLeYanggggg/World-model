"""Inspect development metric sensitivity; never select a new deployment policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.evaluation.m3w_experiment_contract import file_digest


def concentration(rows):
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ValueError('Nonempty scale, floor ADE, candidate ADE rows required')
    if not np.isfinite(values).all() or (values[:, 0] <= 0).any() or (values[:, 1:] < 0).any():
        raise ValueError('Finite positive scales and nonnegative errors required')
    scale, baseline, candidate = values.T
    count = max(1, len(values) // 100)
    tail = np.argsort(baseline)[-count:]
    small = scale <= .01

    def stats(mask):
        if not mask.any():
            return {'rows': 0}
        b, c = baseline[mask], candidate[mask]
        native_b, native_c = b * scale[mask], c * scale[mask]
        return {'rows': int(mask.sum()), 'mean_scale': float(scale[mask].mean()),
                'normalized_floor_ade': float(b.mean()), 'normalized_candidate_ade': float(c.mean()),
                'native_floor_ade': float(native_b.mean()), 'native_candidate_ade': float(native_c.mean()),
                'floor_normalized_error_share': float(b.sum() / baseline.sum()) if baseline.sum() else None,
                'floor_native_error_share': float(native_b.sum() / (baseline * scale).sum())
                                           if (baseline * scale).sum() else None}

    tail_mask = np.zeros(len(values), dtype=bool)
    tail_mask[tail] = True
    return {'all': stats(np.ones(len(values), dtype=bool)),
            'scale_at_most_0p01_dataset_local': stats(small),
            'scale_above_0p01_dataset_local': stats(~small),
            'top_approximately_one_percent_by_normalized_floor_error': stats(tail_mask)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text())
    result = {'result_source': 'fresh_run_analysis_of_hash_verified_development_caches',
              'input_source': 'cached_verified', 'metrics_sha256': file_digest(args.metrics),
              'scope': 'development_diagnostic_no_new_policy_or_independent_confirmation',
              'scale_cutoff_is_dataset_local_not_comparable_across_units': True, 'seeds': {}}
    for seed in metrics['seeds']:
        report = args.study_dir / f'seed{seed}_development/development_report.json'
        if file_digest(report) != metrics['source_manifests'][str(report.resolve().relative_to(ROOT))]:
            raise ValueError('Development report hash changed')
        grouped = {}
        for receipt in report.parent.glob('*.receipt.json'):
            saved = json.loads(receipt.read_text())
            if saved['key'][0] != f'seed{seed}_neural_cost_conservative':
                continue
            cache = receipt.with_name(receipt.name.replace('.receipt.json', '.rows.json'))
            if file_digest(cache) != saved['cache_sha256']:
                raise ValueError('Cached development rows changed')
            for row in json.loads(cache.read_text()):
                if row['baseline_ade'] is not None:
                    grouped.setdefault(row['recording_id'], []).append(
                        [row['scale'], row['baseline_ade'], row['arms']['uncontrolled']['ade']])
        if not grouped:
            raise ValueError('No verified development rows')
        result['seeds'][str(seed)] = {recording: concentration(rows) for recording, rows in grouped.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'error_concentration.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# Development Error Concentration', '',
             'Fresh descriptive analysis of hash-verified caches; no metric, threshold or sample selection changes.', '',
             '| Seed | Recording | Rows | Top ~1% rows | Normalized floor error share % | Native floor error share % |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for seed, recordings in result['seeds'].items():
        for name, parts in recordings.items():
            tail = parts['top_approximately_one_percent_by_normalized_floor_error']
            lines.append(f"| {seed} | {name} | {parts['all']['rows']} | {tail['rows']} | "
                         f"{100*tail['floor_normalized_error_share']:.3f} | {100*tail['floor_native_error_share']:.3f} |")
    lines += ['', 'The top-tail slice reads labels for diagnosis only, never inference.',
              'The 0.01 scale slice is recording-local and cannot be pooled as a metric-unit threshold.',
              'Three seeds share the same floor labels; repeated floor values are not independent evidence.',
              'This diagnoses weighting sensitivity, not an error in the original annotations or a proven causal training failure.', '']
    (args.output_dir / 'error_concentration.md').write_text('\n'.join(lines))
    print(json.dumps({'output': str(args.output_dir), 'seeds': list(result['seeds'])}))


if __name__ == '__main__':
    main()
