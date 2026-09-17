"""Compare all registered residual seeds without changing the primary metric."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_m3w_8to12_losses import paired_rows
from src.evaluation.m3w_experiment_contract import file_digest


def validate_complete(report):
    if report['seeds'] != [17, 29, 43] or report.get('independent_confirmation') is not False:
        raise ValueError('All three registered development seeds required')
    expected = {(s, n) for s in report['seeds'] for n in ('full', 'hold0', 'hold1', 'hold2', 'neural_cost')}
    if len(report['fits']) != len(expected) or {(r['seed'], r['model']) for r in report['fits']} != expected:
        raise ValueError('Complete full/fold/cost fits required')
    if any(not r['training_complete'] or r['steps'] != (1000 if r['model'] == 'neural_cost' else 10000)
           for r in report['fits']):
        raise ValueError('No partial or shortened fit budgets')


def compare(skip, bounded, reference=None):
    for report in (skip, bounded):
        validate_complete(report)
    if skip['protocol_sha256'] != bounded['protocol_sha256']:
        raise ValueError('Residual arms must share the frozen protocol')
    paired = []
    for row in paired_rows(skip, bounded):
        paired.append({'seed': row['seed'], 'baseline_skip': row['mse'], 'motion_bounded': row['robust'],
            'bounded_ade_reduction_vs_skip_pct': row['candidate_ade_reduction_vs_mse_pct'],
            'skip_oracle_diagnostic': row['mse_oracle_diagnostic'],
            'bounded_oracle_diagnostic': row['robust_oracle_diagnostic'],
            'skip_selection': row['mse_selection'], 'bounded_selection': row['robust_selection']})
    reports = {'baseline_skip': skip, 'motion_bounded': bounded}
    if reference is not None:
        validate_complete(reference)
        # Support and causal baseline errors must match even across protocol versions.
        paired_rows(reference, skip)
        reports['v6_absolute_reference'] = reference
    native, descriptive = [], {}
    for name, report in reports.items():
        errors = [next(r for r in report['comparisons'] if r['seed'] == seed and r['arm'] == 'uncontrolled')
                  for seed in report['seeds']]
        gains = [r['all_improvement_pct'] for r in errors]
        descriptive[name] = {'gain_mean_pct': statistics.mean(gains), 'gain_seed_sd_percentage_points': statistics.stdev(gains),
            'uncontrolled_by_seed': errors, 'selection_by_seed': report['selection_by_seed'],
            'result_source': 'cached_verified_completed_reference' if name == 'v6_absolute_reference' else 'fresh_run_completed_training',
            'seed_sd_is_not_scene_uncertainty': True}
        for seed in report['seeds']:
            for recording, metrics in report['baseline_candidate_oracle_diagnostic'][str(seed)]['per_recording_native_units'].items():
                causal = report['baselines'][recording]['dataset_local_unverified_baselines']
                best = min(causal, key=lambda k: causal[k]['ade'])
                error = causal[best]['ade']
                native.append({'model': name, 'seed': seed, 'recording': recording, **metrics,
                    'best_development_causal_descriptive': best, 'best_development_causal_ade': error,
                    'gain_vs_best_development_causal_pct': 100*(error-metrics['candidate_ade'])/error,
                    'frozen_floor_unchanged': True})
    return {'paired_seeds': paired, 'seed_descriptive_statistics': descriptive, 'native_baseline_context': native}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip', type=Path, required=True)
    parser.add_argument('--bounded', type=Path, required=True)
    parser.add_argument('--reference-v6', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    paths = [args.skip, args.bounded] + ([args.reference_v6] if args.reference_v6 else [])
    result = compare(*[json.loads(p.read_text()) for p in paths])
    result.update(result_source='fresh_run_paired_analysis_of_completed_development_experiments',
        input_sha256={str(p): file_digest(p) for p in paths}, analysis_code_sha256=file_digest(Path(__file__)),
        scope='adaptive_development_not_independent_confirmation', new_deployment=False,
        independent_confirmation=False, stage5c_executed=False, smc_enabled=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/'residual_parameterization_comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = ['# Paired Baseline-Relative Parameterizations', '',
        'All three seeds and complete budgets. Same data, target, metric, policies and CV floor.',
        'Skip versus bounded changes only the residual amplitude mapping. v6 also differs in skip/initialization; its comparison is not a bound-only ablation.', '',
        '| Seed | Skip gain vs CV % | Bounded gain vs CV % | Bounded vs skip ADE reduction % | Skip oracle % | Bounded oracle % | Skip choice | Bounded choice |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |']
    for r in result['paired_seeds']:
        lines.append(f"| {r['seed']} | {r['baseline_skip']['all_improvement_pct']:.6f} | {r['motion_bounded']['all_improvement_pct']:.6f} | "
            f"{r['bounded_ade_reduction_vs_skip_pct']:.6f} | {r['skip_oracle_diagnostic']['oracle_gain_pct']:.6f} | "
            f"{r['bounded_oracle_diagnostic']['oracle_gain_pct']:.6f} | {r['skip_selection']['arm']} | {r['bounded_selection']['arm']} |")
    lines += ['', '## Absolute and Relative Easy Error', '',
        '| Model | Seed | CV easy ADE | Candidate easy ADE | Easy degradation % | Hard gain % |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for name, stats in result['seed_descriptive_statistics'].items():
        for r in stats['uncontrolled_by_seed']:
            lines.append(f"| {name} | {r['seed']} | {r['easy_baseline_ade']:.8f} | {r['easy_selected_ade']:.8f} | {r['easy_degradation_pct']:.6f} | {r['hard_improvement_pct']:.6f} |")
    lines += ['', '## Native-Coordinate Causal Context', '',
        '| Model | Seed | Recording | Candidate ADE | CV gain % | Gain over best development causal % |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    for r in result['native_baseline_context']:
        lines.append(f"| {r['model']} | {r['seed']} | {r['recording']} | {r['candidate_ade']:.6f} | {r['candidate_gain_pct']:.6f} | {r['gain_vs_best_development_causal_pct']:.6f} |")
    lines += ['', 'Native values remain recording-local and do not replace the primary metric. The development-best causal alternative is descriptive, not a new selected floor.',
        'Oracle uses labels only for diagnosis. Training-seed variation is not a scene CI; one physical development site cannot establish a cross-scene result.',
        'Full family tables retain every learned head/policy and fallback decision. No deployment, metric/seconds, Stage5C or SMC claim.', '']
    (args.output_dir/'residual_parameterization_comparison.md').write_text('\n'.join(lines))
    print(json.dumps({k: {s: v[s] for s in ('gain_mean_pct', 'gain_seed_sd_percentage_points')}
                      for k, v in result['seed_descriptive_statistics'].items()}))


if __name__ == '__main__':
    main()
