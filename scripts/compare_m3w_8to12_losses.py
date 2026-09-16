"""Compare the completed fixed-budget loss ablation without selecting a test winner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.evaluation.m3w_experiment_contract import file_digest


def paired_rows(mse, robust):
    if mse['task'] != robust['task'] or mse['seeds'] != robust['seeds']:
        raise ValueError('Task and registered seeds must match')
    if mse['baselines'] != robust['baselines']:
        raise ValueError('Baseline rows, metrics, units or recording roles changed')
    result = []
    for seed in mse['seeds']:
        candidates = []
        for report in (mse, robust):
            rows = [r for r in report['comparisons'] if r['seed'] == seed and r['arm'] == 'uncontrolled']
            if not rows or any(r['ade'] != rows[0]['ade'] or r['fde'] != rows[0]['fde'] for r in rows):
                raise ValueError('Policy controls must use identical candidate forecasts')
            candidates.append(rows[0])
        a, b = candidates
        for key in ('complete_queries', 'fde_endpoint_label_queries', 'past_supported_queries', 'scene_count'):
            if a[key] != b[key]:
                raise ValueError('Compared forecast support changed')
        result.append({'seed': seed, 'mse': a, 'robust': b,
                       'candidate_ade_reduction_vs_mse_pct': float(100 * (a['ade'] - b['ade']) / a['ade'])
                                                            if a['ade'] > 0 else None,
                       'mse_oracle_diagnostic': mse['baseline_candidate_oracle_diagnostic'][str(seed)],
                       'robust_oracle_diagnostic': robust['baseline_candidate_oracle_diagnostic'][str(seed)],
                       'mse_selection': mse['selection_by_seed'][str(seed)]['selected'],
                       'robust_selection': robust['selection_by_seed'][str(seed)]['selected']})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mse', type=Path, required=True)
    parser.add_argument('--robust', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(p.read_text()) for p in (args.mse, args.robust)]
    rows = paired_rows(*reports)
    result = {'result_source': 'fresh_run_paired_analysis_of_completed_development_experiments',
              'input_sha256': {str(p): file_digest(p) for p in (args.mse, args.robust)},
              'scope': 'adaptive_development_only_not_independent_confirmation', 'paired_seeds': rows,
              'training_loss_values_comparable': False, 'independent_confirmation': False,
              'new_deployment': False, 'stage5c_executed': False, 'smc_enabled': False}
    result['seed_descriptive_statistics'] = {
        version: {'uncontrolled_gain_mean_pct': float(np.mean([r[version]['all_improvement_pct'] for r in rows])),
                  'uncontrolled_gain_sd_pct': float(np.std([r[version]['all_improvement_pct'] for r in rows], ddof=1))
                                               if len(rows) > 1 else None}
        for version in ('mse', 'robust')}
    native = []
    for version, report in zip(('mse', 'robust'), reports):
        for seed in report['seeds']:
            values = report['baseline_candidate_oracle_diagnostic'][str(seed)]['per_recording_native_units']
            for recording, metrics in values.items():
                causal = report['baselines'][recording]['dataset_local_unverified_baselines']
                best = min(causal, key=lambda name: causal[name]['ade'])
                error = causal[best]['ade']
                native.append({'version': version, 'seed': seed, 'recording': recording,
                    **metrics, 'development_best_causal_descriptive': best, 'development_best_causal_ade': error,
                    'gain_vs_development_best_causal_pct': 100 * (error - metrics['candidate_ade']) / error,
                    'baseline_selection_scope': 'descriptive_development_only_not_changed_frozen_floor'})
    result['native_baseline_context'] = native
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'robust_loss_comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# MSE vs Smooth-L1: Paired Development Ablation', '',
             'Fresh analysis of completed real experiments; same rows, model, updates, seeds and evaluation.',
             'Adaptive development, not a new untouched test. Training loss values use different units and are not compared.', '',
             '| Seed | MSE gain vs CV % | Smooth-L1 gain vs CV % | Candidate ADE reduction vs MSE % | MSE oracle % | Robust oracle % | Robust selection |',
             '| --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in rows:
        lines.append(f"| {r['seed']} | {r['mse']['all_improvement_pct']:.3f} | {r['robust']['all_improvement_pct']:.3f} | "
                     f"{r['candidate_ade_reduction_vs_mse_pct']:.3f} | {r['mse_oracle_diagnostic']['oracle_gain_pct']:.3f} | "
                     f"{r['robust_oracle_diagnostic']['oracle_gain_pct']:.3f} | {r['robust_selection']['arm']} |")
    lines.extend(['', 'Primary ADE is past-normalized, not meters. Positive improvement versus the failed MSE model',
                  'does not mean improvement versus CV, preservation of easy cases, or a useful joint-selection mechanism.',
                  'The JSON preserves absolute/relative easy errors, per-recording native-coordinate results and all seed decisions.',
                  'Oracle rows read future labels for evaluation only. They are not inference scores or model results.',
                  'One University development site cannot support a physical-scene CI; seed SD is training variability only.', ''])
    lines.extend(['## Native-Coordinate Strong Baseline Context', '',
                  '| Loss | Seed | Recording | Candidate ADE | CV gain % | Lowest development causal ADE | Gain vs that baseline % |',
                  '| --- | --- | --- | ---: | ---: | ---: | ---: |'])
    for r in native:
        lines.append(f"| {r['version']} | {r['seed']} | {r['recording']} | {r['candidate_ade']:.6f} | "
                     f"{r['candidate_gain_pct']:.3f} | {r['development_best_causal_ade']:.6f} | "
                     f"{r['gain_vs_development_best_causal_pct']:.3f} |")
    lines += ['', 'Units remain recording-local and unverified. The lowest development causal score is',
              'descriptive context, not a test-selected floor or a retrospective protocol change.',
              'Do not interpret a CV-only gain as an advantage over the stronger causal comparator.', '']
    (args.output_dir / 'robust_loss_comparison.md').write_text('\n'.join(lines))
    print(json.dumps(result['seed_descriptive_statistics']))


if __name__ == '__main__':
    main()
