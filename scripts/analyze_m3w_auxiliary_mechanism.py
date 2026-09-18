"""Analyze all frozen source controls without choosing models or thresholds."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.analyze_m3w_sdd_auxiliary import event_error_summary
from scripts.run_m3w_objective_alignment import paired_interval
from src.evaluation.m3w_experiment_contract import file_digest
from src.world_model.m3w_auxiliary_mechanism import ALL_ARMS, load_mechanism_registration
from src.world_model.m3w_offline_visual_data import json_write


def analyze_report(report, seeds, modalities):
    trials = report['trials']
    look = {(t['arm'], t['modality'], t['seed'], t['fold']): t for t in trials}
    required = {(a, m, s, f) for a in ALL_ARMS for m in modalities for s in seeds for f in range(3)}
    if not report['complete'] or len(trials) != 108 or set(look) != required:
        raise ValueError('Complete unique 108-fit matrix required')

    def matrix(arm, modality, field='primary_ADE'):
        return np.array([[look[arm, modality, s, f]['vs_CV'][field]
                          for f in range(3)] for s in seeds])

    pairs = [('sdd_aux', 'main4k'), ('sdd_permuted', 'main4k'),
             ('sdd_aux', 'sdd_permuted'), ('no_aux', 'main4k'), ('sdd_aux', 'no_aux')]
    contrasts, decomposition, groups = {}, {}, {}
    for modality in modalities:
        reference = matrix('main4k', modality, 'reference_ADE')
        for arm in ALL_ARMS:
            np.testing.assert_array_equal(matrix(arm, modality, 'reference_ADE'), reference)
        for first, second in pairs:
            a, b = matrix(first, modality), matrix(second, modality)
            contrasts[f'{modality}:{first}_vs_{second}'] = dict(
                **paired_interval(a, b, 2000), absolute_ADE_difference=float((a-b).mean()),
                first_better_fits=int((a < b).sum()), total_fits=9)
        real, main4k, main6k = [matrix(a, modality) for a in ('sdd_aux', 'main4k', 'no_aux')]
        source = real-main4k
        exposure = main4k-main6k
        total = real-main6k
        np.testing.assert_allclose(source+exposure, total, rtol=1e-12, atol=1e-12)
        decomposition[modality] = dict(
            actual_source_minus_main4k_ADE=float(source.mean()),
            main4k_minus_main6k_ADE=float(exposure.mean()),
            actual_source_minus_main6k_ADE=float(total.mean()),
            interpretation='Arithmetic decomposition, not mediation or independent causal identification; negative ADE difference is better.',
            relative_percentages_are_not_additive=True)

    for arm in ALL_ARMS:
        for modality in modalities:
            ts = [look[arm, modality, s, f] for s in seeds for f in range(3)]
            events = {}
            for name in ('static_stays', 'static_moves', 'moving_stops', 'moving_turns',
                         'other_motion', 'hard_train_q75'):
                ss = [t['slices'][name] for t in ts if t['slices'].get(name, {}).get('rows', 0)]
                if ss:
                    events[name] = dict(**event_error_summary(ss),
                        rows_per_seed=sum(s['rows'] for s in ss)//len(seeds),
                        physical_sites=sum(bool(look[arm, modality, seeds[0], f]['slices'].get(name, {}).get('rows', 0))
                                           for f in range(3)))
            def limits(field, section='vs_CV'):
                values = [t[section][field] if section else t[field] for t in ts]
                values = [v for v in values if v is not None]
                return [min(values), max(values)] if values else None
            groups[f'{arm}_{modality}'] = dict(
                events=events,
                positive_held_fits=sum(t['vs_CV']['improvement_percent'] > 0 for t in ts),
                easy_gate_fits=sum(t['vs_CV']['easy_degradation_percent'] is not None and
                                  t['vs_CV']['easy_degradation_percent'] <= 2 for t in ts),
                safe_positive_fits=sum(t['vs_CV']['improvement_percent'] > 0 and
                    t['vs_CV']['easy_degradation_percent'] is not None and
                    t['vs_CV']['easy_degradation_percent'] <= 2 for t in ts),
                train_gain_range_percent=limits('train_equal_scene_gain_percent', None),
                held_gain_range_percent=limits('improvement_percent'),
                easy_relative_degradation_range_percent=limits('easy_degradation_percent'),
                easy_absolute_harm_range=limits('easy_absolute_harm'),
                tail_ADE_p95_range=limits('tail_ADE_p95'),
                pixel_vs_mask=paired_interval(matrix(arm, modality), matrix(arm, 'mask_only'), 2000))
    return dict(contrasts=contrasts, decomposition=decomposition, groups=groups,
                no_new_model_selected=True, independent_confirmation=False,
                weighting='Equal physical site after averaging seeds; only three previously exposed sites.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    args = parser.parse_args()
    reg, _ = load_mechanism_registration(ROOT, args.registration)
    reports = ROOT/reg['reports']
    path = reports/'report.json'
    report = json.loads(path.read_text())
    for trial in report['trials']:
        for kind in ('checkpoint', 'prediction'):
            if file_digest(ROOT/trial[kind+'_path']) != trial[kind+'_sha256']:
                raise ValueError('Completed artifact changed')
    result = dict(result_source='fresh_run_analysis_of_54_new_and_54_cached_verified_fits',
                  report_sha256=file_digest(path),
                  **analyze_report(report, reg['seeds'], reg['modalities']))
    json_write(reports/'analysis.json', result)
    lines = ['# Source Mechanism Contrasts', '',
             'All 108 fixed results are retained: 54 fresh fits and 54 hash-verified previous fits.',
             'No model or threshold is selected here. Three reused sites do not provide independent confirmation.', '',
             '| Input / contrast | Gain (%) | Descriptive site CI (%) | Better seed-site fits |',
             '| --- | ---: | --- | ---: |']
    for name, value in result['contrasts'].items():
        ci = value['exploratory_scene_ci95_percent']
        lines.append(f"| {name} | {value['gain_percent']:.5f} | [{ci[0]:.5f}, {ci[1]:.5f}] | {value['first_better_fits']}/9 |")
    lines += ['', 'Positive gain is relative to the named neural control, not necessarily to causal CV.',
              'main4k matches main-training exposure, not total compute.',
              'Source permutation matches draws/compute and preserves recording/support strata; it does not remove all dependence.',
              'The absolute-ADE decomposition in analysis.json is arithmetic, not a causal mediation result.',
              'All event strata use evaluation labels only. They are not inference inputs.',
              'No metric, seconds-level, true-3D, foundation, Stage5C or SMC claim.', '']
    (reports/'contrasts.md').write_text('\n'.join(lines))
    print(json.dumps(dict(contrasts=result['contrasts'], decomposition=result['decomposition']), indent=2))


if __name__ == '__main__':
    main()
