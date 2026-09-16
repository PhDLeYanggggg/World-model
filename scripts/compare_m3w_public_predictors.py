"""Compare fixed-budget public/local predictors; retain every seed and failed arm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scripts.compare_m3w_8to12_losses import paired_rows
from src.evaluation.m3w_experiment_contract import file_digest


def compare(local, public):
    if local['protocol_sha256'] != public['protocol_sha256']:
        raise ValueError('Paired study requires the same frozen protocol')
    for report in (local, public):
        if report.get('independent_confirmation') is not False:
            raise ValueError('This comparison is development-only')
        expected = {(s, n) for s in report['seeds'] for n in ('full', 'hold0', 'hold1', 'hold2', 'neural_cost')}
        actual = {(r['seed'], r['model']) for r in report['fits']}
        if expected != actual or len(report['fits']) != len(expected):
            raise ValueError('All declared full/fold/cost fits required')
        if any(not r['training_complete'] or r['steps'] != (1000 if r['model'] == 'neural_cost' else 10000)
               for r in report['fits']):
            raise ValueError('All fits must complete the unchanged registered budget')
    rows = []
    for paired in paired_rows(local, public):
        rows.append({'seed': paired['seed'], 'transformer': paired['mse'], 'eqmotion_K1': paired['robust'],
            'eqmotion_ade_reduction_vs_transformer_pct': paired['candidate_ade_reduction_vs_mse_pct'],
            'transformer_oracle_diagnostic': paired['mse_oracle_diagnostic'],
            'eqmotion_oracle_diagnostic': paired['robust_oracle_diagnostic'],
            'transformer_selection': paired['mse_selection'], 'eqmotion_selection': paired['robust_selection']})
    native = []
    for name, report in (('transformer', local), ('eqmotion_K1', public)):
        for seed in report['seeds']:
            for recording, metrics in report['baseline_candidate_oracle_diagnostic'][str(seed)]['per_recording_native_units'].items():
                causal = report['baselines'][recording]['dataset_local_unverified_baselines']
                best = min(causal, key=lambda k: causal[k]['ade'])
                error = causal[best]['ade']
                native.append({'model': name, 'seed': seed, 'recording': recording, **metrics,
                    'best_development_causal_descriptive': best, 'best_development_causal_ade': error,
                    'gain_vs_best_development_causal_pct': 100 * (error - metrics['candidate_ade']) / error,
                    'frozen_floor_unchanged': True})
    return {'paired_seeds': rows, 'native_baseline_context': native,
            'seed_descriptive_statistics': {
                name: {'gain_mean_pct': float(np.mean([r[name]['all_improvement_pct'] for r in rows])),
                       'gain_seed_sd_pct': float(np.std([r[name]['all_improvement_pct'] for r in rows], ddof=1))
                                          if len(rows) > 1 else None}
                for name in ('transformer', 'eqmotion_K1')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transformer', type=Path, required=True)
    parser.add_argument('--eqmotion', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    reports = [json.loads(p.read_text()) for p in (args.transformer, args.eqmotion)]
    result = compare(*reports)
    result.update({'result_source': 'fresh_run_paired_analysis_of_completed_real_development_training',
        'input_sha256': {str(p): file_digest(p) for p in (args.transformer, args.eqmotion)},
        'analysis_code_sha256': file_digest(Path(__file__)),
        'protocol_sha256': reports[0]['protocol_sha256'],
        'scope': 'historically_exposed_development_only_not_published_EqMotion_best_of_20',
        'independent_confirmation': False, 'new_deployment': False,
        'stage5c_executed': False, 'smc_enabled': False})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'paired_predictor_comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# Matched Public-Core vs Local Predictor', '',
        'Fresh real training and evaluation on historically exposed development data.',
        'Same obs8/pred12 task, rows, complete-aligned past neighbors, seeds, updates and loss.',
        'Equal update/sample budgets do not mean equal model capacity or compute. Per-fit device and elapsed time are retained in the input metrics; explicit CPU recovery is not hidden as MPS-only execution.',
        'EqMotion is a fixed K=1 author-core adaptation, not published best-of-20 performance.', '',
        '| Seed | Transformer gain vs CV % | EqMotion-K1 gain vs CV % | EqMotion gain vs Transformer % | Local oracle % | Public oracle % | Local selection | Public selection |',
        '| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |']
    for r in result['paired_seeds']:
        lines.append(f"| {r['seed']} | {r['transformer']['all_improvement_pct']:.3f} | {r['eqmotion_K1']['all_improvement_pct']:.3f} | "
            f"{r['eqmotion_ade_reduction_vs_transformer_pct']:.3f} | {r['transformer_oracle_diagnostic']['oracle_gain_pct']:.3f} | "
            f"{r['eqmotion_oracle_diagnostic']['oracle_gain_pct']:.3f} | {r['transformer_selection']['arm']} | {r['eqmotion_selection']['arm']} |")
    lines += ['', 'Primary errors are past-normalized ADE, not meters. Oracle reads labels only for diagnosis.',
        'One physical development scene prevents a meaningful scene confidence interval. Seed SD is not scene uncertainty.',
        'Full reports retain easy absolute/relative error, hard error, switch rate, harm, all policies and label denominators.', '',
        '## Native-Coordinate Strong Baseline Context', '',
        '| Model | Seed | Recording | Candidate ADE | CV gain % | Best development causal ADE | Gain over that baseline % |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for r in result['native_baseline_context']:
        lines.append(f"| {r['model']} | {r['seed']} | {r['recording']} | {r['candidate_ade']:.6f} | {r['candidate_gain_pct']:.3f} | "
                     f"{r['best_development_causal_ade']:.6f} | {r['gain_vs_best_development_causal_pct']:.3f} |")
    lines += ['', 'The development-best causal baseline is descriptive context, not a changed floor or a test-selected deployment rule.',
        'Raw-frame t+50, actual-count-matched control, real deferral control, independent calibration/confirmation and visual-scene contribution remain open.', '']
    (args.output_dir / 'paired_predictor_comparison.md').write_text('\n'.join(lines))
    print(json.dumps(result['seed_descriptive_statistics']))


if __name__ == '__main__':
    main()
