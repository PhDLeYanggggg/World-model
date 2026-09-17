"""Summarize every fixed deferral setting; no best-setting selection."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_development_evaluation import content_digest


def collect(report, registration):
    rows, controls, pairs = [], [], []
    if set(report['results']) != set(registration['families']) or report['selected_variant'] is not None:
        raise ValueError('Complete unselected family required')
    seeds = json.loads((ROOT / registration['parent_protocol']).read_text())['seeds']
    for family, seed_results in report['results'].items():
        if set(seed_results) != set(map(str, seeds)):
            raise ValueError('All registered seeds must finish')
        for seed, result in seed_results.items():
            if set(result['fits']) != set(registration['variants']):
                raise ValueError('Incomplete fixed comparator grid')
            if not result['OOF_features_exactly_matched'] or not result['fresh_forecast_scoring_exactly_matches_parent']:
                raise ValueError('Matched-control replay failed')
            summaries = list(result['summaries'].values())
            if len(summaries) != 4:
                raise ValueError('All four parent cost-head/policy combinations required')
            for variant, fit in result['fits'].items():
                if not fit['training_complete'] or fit['steps'] != registration['variants'][variant]['fit_settings']['steps']:
                    raise ValueError('Incomplete registered head budget')
                arm = 'deferral_' + variant
                value = summaries[0]['arms'][arm]
                if any(s['arms'][arm] != value for s in summaries):
                    raise ValueError('Deferral must not change with comparison policy')
                rows.append({'family': family, 'seed': int(seed), 'variant': variant,
                    'ade': value['all']['selected_error'], 'baseline_ade': value['all']['baseline_error'],
                    'gain_pct': value['all']['improvement_pct'], 'hard_gain_pct': value['hard']['improvement_pct'],
                    'easy_degradation_pct': 100 * value['easy']['degradation_fraction'],
                    'easy_ade': value['easy']['selected_error'], 'easy_baseline_ade': value['easy']['baseline_error'],
                    'positive_harm': value['all']['mean_positive_harm'],
                    'switch_pct_all_past_supported': 100 * value['switch_rate_all_past_supported'],
                    'fde': value['secondary_fde']['selected_error'], 'fde_gain_pct': value['secondary_fde']['improvement_pct'],
                    'complete_ade_rows': value['all']['count'], 'physical_scenes': summaries[0]['physical_scenes'],
                    'scene_ci_status': value['all']['bootstrap']['status'],
                    'easy_pass': value['easy']['degradation_fraction'] <= .02,
                    'positive_primary_gain': value['all']['improvement_pct'] > 0})
            for name, summary in result['summaries'].items():
                for arm in ('floor', 'uncontrolled', 'independent', 'scene_uniform', 'joint'):
                    value = summary['arms'][arm]
                    controls.append({'family': family, 'seed': int(seed), 'candidate': name, 'arm': arm,
                        'gain_pct': value['all']['improvement_pct'], 'ade': value['all']['selected_error'],
                        'hard_gain_pct': value['hard']['improvement_pct'],
                        'easy_degradation_pct': 100 * value['easy']['degradation_fraction'],
                        'switch_pct_all_past_supported': 100 * value['switch_rate_all_past_supported']})
                for variant, values in summary['paired_comparisons'].items():
                    for control, value in values.items():
                        pairs.append({'family': family, 'seed': int(seed), 'candidate': name,
                                      'variant': variant, 'control': control, **value})
    return rows, controls, pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    args = parser.parse_args()
    study = args.study.resolve()
    report = json.loads((study / 'comparison.json').read_text())
    identity = json.loads((study / 'run_identity.json').read_text())
    completion = json.loads((study / 'completion.json').read_text())
    if completion != {'identity_sha256': content_digest(identity), 'report_sha256': file_digest(study / 'comparison.json')}:
        raise ValueError('Incomplete or changed comparison')
    reg_path = ROOT / report['registration']['path']
    if file_digest(reg_path) != report['registration']['sha256']:
        raise ValueError('Registration changed')
    registration = json.loads(reg_path.read_text())
    rows, controls, pairs = collect(report, registration)
    aggregate, fits = [], []
    for family in registration['families']:
        for variant in registration['variants']:
            part = [r for r in rows if r['family'] == family and r['variant'] == variant]
            aggregate.append({'family': family, 'variant': variant, 'seeds': len(part),
                'mean_gain_pct': float(np.mean([r['gain_pct'] for r in part])),
                'seed_sd_gain_pp': float(np.std([r['gain_pct'] for r in part], ddof=1)),
                'easy_pass_seeds': sum(r['easy_pass'] for r in part),
                'positive_gain_seeds': sum(r['positive_primary_gain'] for r in part),
                'positive_gain_and_easy_pass_seeds': sum(r['easy_pass'] and r['positive_primary_gain'] for r in part)})
        for seed, result in report['results'][family].items():
            target = study / f'{family}_seed{seed}'
            costs = []
            for cache in sorted(target.glob('fold*.npz')):
                receipt = json.loads(cache.with_suffix('.json').read_text())
                if receipt['cache_sha256'] != file_digest(cache):
                    raise ValueError('OOF target cache changed')
                with np.load(cache, allow_pickle=False) as data:
                    costs.append(data['targets'])
            raw = np.concatenate(costs)
            for variant, fit in result['fits'].items():
                if file_digest(ROOT / fit['checkpoint']) != fit['checkpoint_sha256']:
                    raise ValueError('Fitted checkpoint changed')
                saved = json.loads((target / variant / 'fit_report.json').read_text())
                if any(saved[k] != v for k, v in fit.items()):
                    raise ValueError('Fit report changed')
                cap = registration['variants'][variant]['cost_bound']
                bounded = np.minimum(raw / cap, 1)
                fits.append({'family': family, 'seed': int(seed), 'variant': variant,
                    'checkpoint_sha256': fit['checkpoint_sha256'], 'OOF_feature_identity': fit['identity']['oof_feature_identity'],
                    'training_rows': fit['training_rows'], 'feature_dimension': fit['feature_dimension'], 'steps': fit['steps'],
                    'fit_seconds_including_checkpoint': fit['elapsed_seconds'],
                    'mean_loss_first50': float(np.mean(saved['losses'][:50])),
                    'mean_loss_last50': float(np.mean(saved['losses'][-50:])),
                    'clip_fraction_by_action': fit['cost_clip_fraction'],
                    'both_costs_clipped_fraction': float(np.mean((raw > cap).all(1))),
                    'raw_cost_tie_fraction': float(np.mean(raw[:, 0] == raw[:, 1])),
                    'bounded_cost_tie_fraction': float(np.mean(bounded[:, 0] == bounded[:, 1]))})
    out = args.report_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    metrics = {'result_source': 'fresh_run_summary_of_complete_training_and_verified_parent_replay',
               'registration': report['registration'], 'source_report_sha256': file_digest(study / 'comparison.json'),
               'summary_code_sha256': file_digest(Path(__file__)), 'aggregate': aggregate, 'rows': rows,
               'controls': controls, 'paired_comparisons': pairs, 'fits': fits,
               'per_recording': {f'{f}/{seed}': next(iter(r['summaries'].values()))['arms']
                    for f, seeds in report['results'].items() for seed, r in seeds.items()},
               'independent_confirmation': False, 'eligible_for_selection': False, 'deployment_approved': False}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
    with (out / 'metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# v7 Cost-Sensitive Deferral: Complete Registered Control', '',
        'All 24 new heads completed 1,000 updates. Existing forecasting checkpoints are hash-verified, not retrained.',
        'Each head uses the same 11,966 OOF queries and 306 causal features as the original gain/harm heads.',
        'Fresh floor/candidate scoring exactly matches all parent query exports before ordinary controls are reused.', '',
        'Primary: eight observed / twelve predicted native annotation steps, past-normalized ADE, equal physical scene.',
        'Dataset-local geometry is unverified. Neither seconds nor metres are claimed.',
        'Only one exposed physical development site is available. Three training seeds are not three independent sites;',
        'scene CI is not estimable here. No threshold tuning, best-deferrer selection or deployment occurs.', '',
        'The unconstrained two-action loss follows the single-expert adaptation of',
        '[Mao et al., ICML 2024](https://proceedings.mlr.press/v235/mao24d.html).',
        'This is not a reproduction of their experimental results or a calibrated risk guarantee.', '',
        '## Every Registered Setting', '',
        '| Predictor | Deferrer | Mean ADE gain vs CV | Seed SD (pp) | Positive seeds | Easy <=2% seeds | Both |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in aggregate:
        lines.append(f"| {r['family']} | {r['variant']} | {r['mean_gain_pct']:+.6f}% | {r['seed_sd_gain_pp']:.6f} | {r['positive_gain_seeds']}/3 | {r['easy_pass_seeds']}/3 | {r['positive_gain_and_easy_pass_seeds']}/3 |")
    lines += ['', '## All Seed Results', '',
              '| Predictor | Seed | Deferrer | Gain | Hard gain | Easy degradation | Switch rate |',
              '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['family']} | {r['seed']} | {r['variant']} | {r['gain_pct']:+.6f}% | {r['hard_gain_pct']:+.6f}% | {r['easy_degradation_pct']:.4f}% | {r['switch_pct_all_past_supported']:.3f}% |")
    lines += ['', '## Scope of the Comparison', '',
        'All four original cost-head/policy settings are retained in metrics.json, with 480 paired deferrer/control',
        'comparisons. They share forecasters, OOF inputs and query populations, not realized intervention counts',
        'or imposed risk budgets. The M3W guards are absent from deferral by design. Lower guarded error alone',
        'therefore cannot establish a superior learning objective; coverage/budget is a confound.',
        'The previously completed actual-count-matched joint/unary experiment still shows no joint advantage.', '',
        'Cost clipping changes the fit objective, not evaluation. Per-head clip/tie fractions and first/last-50',
        'minibatch-loss means are in metrics.json; loss reduction does not prove convergence or forecasting lift.',
        'All per-recording normalized metrics, absolute easy error, positive harm, FDE and switch denominators remain available.',
        'Raw50 is the completed parent supplementary task and is not re-evaluated for these new deferrers.', '',
        'Stage5C and SMC remain disabled. No new deployment or submission-ready claim follows from this comparator.', '']
    (out / 'results.md').write_text('\n'.join(lines))
    print(json.dumps({'heads': len(rows), 'pairs': len(pairs), 'aggregate': aggregate}, indent=2))


if __name__ == '__main__':
    main()
