"""Verify paired contrasts and export aggregate-only EqMotion control tables."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'outputs/publication_readiness_2026_09/protected_eqmotion_controls_v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_contrast(left, right, report):
    """Rebuild paired site differences and bootstrap, not independent-arm CIs."""
    scenes = left['expected_scenes']
    assert scenes == right['expected_scenes']
    assert report['unit'] == 'physical_scene'
    differences = np.array([left['by_scene'][s]['gain_percent'] -
                            right['by_scene'][s]['gain_percent'] for s in scenes])
    assert np.isfinite(differences).all() and report['resamples'] >= 2000
    np.testing.assert_allclose(differences, report['scene_differences_pp'], atol=1e-10)
    np.testing.assert_allclose(differences.mean(), report['mean_gain_difference_pp'], atol=1e-10)
    draws = np.random.default_rng(report['seed']).integers(
        len(scenes), size=(report['resamples'], len(scenes)))
    interval = np.percentile(differences[draws].mean(1), [2.5, 97.5])
    np.testing.assert_allclose(interval, report['ci95_pp'], atol=1e-10)


def check_groups(left, right, report):
    check_contrast(left['ADE'], right['ADE'], report['ADE'])
    for group in ('hard', 'positive_easy'):
        check_contrast(left['subsets'][group], right['subsets'][group], report[group])
    return 3


def summarize(name, report, indexed_rows):
    seeds = report['seeds']
    return dict(
        policy=name, ADE_gain_percent=report['ADE']['equal_scene_gain_percent'],
        ADE_ci95=report['ADE']['scene_bootstrap_ci95'],
        FDE_gain_percent=report['FDE']['equal_scene_gain_percent'],
        hard_gain_percent=report['subsets']['hard']['equal_scene_gain_percent'],
        positive_easy_gain_percent=report['subsets']['positive_easy']['equal_scene_gain_percent'],
        worst_site_seed_easy_degradation_percent=report['worst_site_seed_easy_degradation_percent'],
        mean_seed_switch_rate_percent=100*np.mean([r['selected'] for r in seeds.values()])/indexed_rows,
        selected_unknown_by_seed={s:r['selected_unknown'] for s,r in seeds.items()},
        zero_CV_harmed_by_seed={s:r['zero_CV_harmed'] for s,r in seeds.items()})


def main():
    path = REPORT / 'analysis.json'
    analysis = json.loads(path.read_text())
    analysis_hash = digest(path)
    for file in ('replay.json', 'independent_verification.json'):
        record = json.loads((REPORT / file).read_text())
        assert record['all_checks_passed'] and record['analysis_sha256'] == analysis_hash
    assert not analysis['deployment'] and not analysis['independent_confirmation']
    assert not analysis['stage5c_executed'] and not analysis['smc_enabled']
    eq = analysis['summaries']
    count = check_groups(eq['eqmotion__neural__strict_stop'], eq['eqmotion__forest__strict_stop'],
                         analysis['neural_minus_forest'])
    contrasts = []
    for key, comparison in analysis['comparisons'].items():
        action, head = key.split('__')
        matched = analysis['matched_count'][key]
        for mode,left,right,contrast in (
                ('strict', eq['eqmotion__'+head+'__strict_stop'], comparison['comparator'],
                 comparison['eqmotion_minus_comparator']),
                ('matched_count', matched['eqmotion'], matched['comparator'],
                 matched['eqmotion_minus_comparator'])):
            count += check_groups(left,right,contrast)
            contrasts.append(dict(comparator=action, head=head, mode=mode,
                EqMotion_ADE_gain_percent=left['ADE']['equal_scene_gain_percent'],
                comparator_ADE_gain_percent=right['ADE']['equal_scene_gain_percent'],
                difference_pp=contrast['ADE']['mean_gain_difference_pp'],
                CI_low_pp=contrast['ADE']['ci95_pp'][0], CI_high_pp=contrast['ADE']['ci95_pp'][1],
                EqMotion_hard_gain_percent=left['subsets']['hard']['equal_scene_gain_percent'],
                comparator_hard_gain_percent=right['subsets']['hard']['equal_scene_gain_percent'],
                EqMotion_worst_easy_degradation_percent=left['worst_site_seed_easy_degradation_percent'],
                comparator_worst_easy_degradation_percent=right['worst_site_seed_easy_degradation_percent']))
    summaries = [summarize(k,v,analysis['rows']) for k,v in eq.items()]
    summaries += [summarize(k+'__strict_stop',r['comparator'],analysis['rows'])
                  for k,r in analysis['comparisons'].items()]
    fits = analysis['fits']
    loss_lines = ['# New Forest Fitting Losses', '',
        'Twelve full-EqMotion forests were newly fitted; no forecasting or neural cost model was retrained.',
        'Each fixed 128-tree fit uses its paired neural head\'s actual 768,000 sample draws.',
        'The loss is draw-weighted mean squared error of benefit/harm fractions on fitting rows',
        'with positive forecast disagreement. It is not validation loss, ADE, or a convergence claim.', '',
        '| View | Trees | First recorded MSE (16 trees) | Final MSE (128 trees) | Effective rows | Fit seconds |',
        '|---|---:|---:|---:|---:|---:|']
    for r in fits:
        f = r['fit']
        assert f['complete'] and f['trees'] == 128 and f['sampled_rows'] == 768000
        assert [t['trees'] for t in f['trace']] == list(range(16,129,16))
        loss_lines.append(f"| {r['view']} | 128 | {f['trace'][0]['fitting_fraction_MSE']:.8f} | "
            f"{f['trace'][-1]['fitting_fraction_MSE']:.8f} | {f['effective_fit_rows']} | {f['seconds']:.3f} |")
    fit_seconds = sum(r['fit']['seconds'] for r in fits)
    loss_lines += ['', f'Summed fitting-loop time: {fit_seconds:.3f} seconds. This excludes loading,',
        'between-fit overhead and wall-clock interruptions. Full 16-tree traces remain in analysis.json.',
        'All budgets completed without sample reduction. Lower fitting loss alone does not establish safer selection.']
    (REPORT / 'training_losses.md').write_text('\n'.join(loss_lines)+'\n')
    with (REPORT / 'paired_controls.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(contrasts[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(contrasts)
    compact = dict(analysis_sha256=analysis_hash, result_source=analysis['result_source'],
        rows=analysis['rows'], complete_rows=analysis['complete_rows'], unknown_rows=analysis['unknown_rows'],
        estimates='equal_site_available_point_ADE_gain_over_causal_CV',
        sites=4, seeds=[17,29,43], new_forests=12, cached_verified_neural_heads=12,
        new_forecasters=0, fitting_seconds=fit_seconds, fitting_seconds_are_not_wall_time=True,
        policies=summaries, paired_contrasts=contrasts, neural_minus_forest=analysis['neural_minus_forest'],
        no_leakage_scope=analysis['no_leakage_scope'], independent_confirmation=False,
        deployment=False, external_readout=False, stage5c_executed=False, smc_enabled=False)
    (REPORT / 'compact_results.json').write_text(json.dumps(compact,indent=2)+'\n')
    verification = dict(all_checks_passed=True, analysis_sha256=analysis_hash,
        verifier_sha256=digest(Path(__file__)), paired_contrasts_checked=count,
        resamples_each=3000, resampling_unit='physical_scene',
        same_agent_independent_arithmetic=True, independent_research_confirmation=False)
    (REPORT / 'contrast_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification,indent=2))


if __name__ == '__main__':
    main()
