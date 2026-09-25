"""Publish every amended coverage contrast and scoped completion evidence."""
import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_m3w_european_hurdle_support_coverage as run
from scripts.report_m3w_european_protected_motion import compact_metric


def verify():
    a = json.loads((run.PUBLIC / 'analysis.json').read_text())
    r = json.loads((run.PUBLIC / 'replay.json').read_text())
    v = json.loads((run.PUBLIC / 'separate_verification.json').read_text())
    sha = run.digest(run.PUBLIC / 'analysis.json')
    if not r['all_passed'] or not v['all_passed'] or r['analysis_sha256'] != sha or v['analysis_sha256'] != sha:
        raise ValueError('Both complete replay and separate verification required')
    if (v['scalar_sorting_choice_arrays'] != 216 or v['coordinate_metric_reductions'] != 864
            or v['additive_decompositions'] != 108 or len(a['views']) != 216 or a['parent_control_views_reproduced'] != 72):
        raise ValueError('All amended matrix cells required')
    if v['verifier_sha256'] != run.digest(ROOT / 'scripts/verify_m3w_european_hurdle_coverage.py'):
        raise ValueError('Verifier changed since execution')
    run.assert_identity(a['identity'])
    run.read_decisions(a['identity'])
    return a


def fmt(value):
    return 'undefined' if value is None else f'{value:.6f}'


def counts(rows):
    defined = [x for x in rows if x is not None]
    return dict(comparisons=len(rows), defined=len(defined),
        positive_points=sum(x['mean_gain_difference_pp'] > 0 for x in defined),
        positive_CI=sum(x['ci95_pp'][0] > 0 for x in defined),
        negative_CI=sum(x['ci95_pp'][1] < 0 for x in defined),
        range_pp=[min(x['mean_gain_difference_pp'] for x in defined), max(x['mean_gain_difference_pp'] for x in defined)] if defined else None)


def reports(a):
    summary = dict(result_source=a['result_source'], analysis_sha256=run.digest(run.PUBLIC / 'analysis.json'),
                   views={}, decomposition=a['decomposition'], groups={}, new_training=False, deployment_changed=False)
    lines = ['# Frozen Risk Ranking at Matched Coverage', '',
        'All 216 views after the pre-readout support amendment. No new training or deployable rule.',
        'Four fitting/eight producer-excluded localities per group, shared twelve opened development localities.',
        'Original decisions retain all support. Common-pool counterfactuals can violate the risk rule.', '',
        '| View | ADE gain vs CV (%) | FDE gain (%) | Hard gain (%) | Worst easy degradation (%) | Zero-CV harm | Switch (%) | Predicted risk violations |',
        '|---|---:|---:|---:|---:|---|---:|---:|']
    for key, row in a['views'].items():
        summary['views'][key] = {k: row[k] for k in ('switch_rate', 'selected_rows', 'selected_unknown_ADE',
            'selected_unknown_FDE', 'predicted_risk_violations', 'zero_CV', 'decision_sha256', 'observed_preservation')}
        summary['views'][key].update(ADE_vs_CV={s: compact_metric(m) for s, m in row['ADE_vs_CV'].items()},
                                     FDE_vs_CV=compact_metric(row['FDE_vs_CV']))
        easy = row['ADE_vs_CV']['easy']['worst_scene_gain_percent']
        vals = [row['ADE_vs_CV']['all']['equal_scene_gain_percent'], row['FDE_vs_CV']['equal_scene_gain_percent'],
                row['ADE_vs_CV']['hard']['equal_scene_gain_percent'], None if easy is None else -easy]
        lines.append(f"| {key} | {' | '.join(fmt(x) for x in vals)} | {row['zero_CV']['harmed_rows']}/{row['zero_CV']['rows']} | {100*row['switch_rate']:.6f} | {row['predicted_risk_violations']} |")
    components = ('full_total', 'support_difference', 'total', 'ranking_at_product_count',
                  'coverage_with_hurdle_ranking', 'coverage_with_product_ranking', 'ranking_at_hurdle_count')
    lines += ['', '## Additive Components', '',
        'Percentage points of CV-normalized improvement, not percentages against changing policy denominators.',
        'Positive ranking components favor hurdle ordering. Both paths retained; no unique causal attribution.', '',
        '| Group | Subset | Component | Mean (pp) | Conditional 95% CI (pp) |', '|---|---|---|---:|---|']
    with (run.PUBLIC / 'contrasts.csv').open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['group', 'subset', 'component', 'mean_pp', 'ci_low_pp', 'ci_high_pp'])
        for key, subsets in a['decomposition'].items():
            for subset, value in subsets.items():
                for name in components:
                    row = value[name]
                    mean, ci = (None, [None, None]) if row is None else (row['mean_gain_difference_pp'], row['ci95_pp'])
                    writer.writerow([key, subset, name, mean, *ci])
                    lines.append(f'| {key} | {subset} | {name} | {fmt(mean)} | [{fmt(ci[0])}, {fmt(ci[1])}] |')
    for candidate in ('neural', 'damping097'):
        summary['groups'][candidate] = {}
        for event in ('all', 'easy'):
            rows = [r for k, r in a['decomposition'].items() if k.startswith(candidate + '_') and k.endswith('_' + event)]
            assert len(rows) == 9
            summary['groups'][candidate][event] = {subset: {name: counts([r[subset][name] for r in rows])
                for name in components} for subset in ('all', 'easy', 'hard')}
    for family in ('ade', 'fde'):
        with (run.PUBLIC / f'locality_{family}_metrics.csv').open('w', newline='') as f:
            writer = csv.writer(f, lineterminator='\n')
            fields = ('rows', 'model_error', 'reference_error', 'gain_percent', 'model_p95', 'model_p99')
            writer.writerow(['view', 'subset', 'locality', *fields])
            for key, row in a['views'].items():
                metrics = row['ADE_vs_CV'] if family == 'ade' else {'FDE': row['FDE_vs_CV']}
                for subset, metric in metrics.items():
                    for site, m in metric['by_scene'].items():
                        writer.writerow([key, subset, site, *[m.get(x) for x in fields]])
    with (run.PUBLIC / 'selection_accounting.csv').open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        fields = ('rows', 'selected', 'selected_known_ADE', 'predicted_risk_violations',
                  'selected_positive_harm_mean', 'selected_net_gain_mean')
        writer.writerow(['view', 'locality', *fields])
        for key, row in a['views'].items():
            for site, m in row['by_locality'].items():
                writer.writerow([key, site, *[m[x] for x in fields]])
    safety = {}
    for candidate in ('neural', 'damping097'):
        safety[candidate] = {}
        for event in ('all', 'easy'):
            safety[candidate][event] = {}
            for arm in run.ARMS:
                rows = [v for k, v in a['views'].items()
                        if k.startswith(candidate + '_') and k.endswith('_' + event + '_' + arm)]
                assert len(rows) == 9
                easy = [-r['ADE_vs_CV']['easy']['worst_scene_gain_percent'] for r in rows]
                safety[candidate][event][arm] = dict(
                    views=len(rows), observed_preservation_views=sum(r['observed_preservation'] for r in rows),
                    easy_degradation_above_two_percent_views=sum(x > 2 for x in easy),
                    worst_positive_easy_degradation_percent=max(easy),
                    zero_reference_harm_views=sum(r['zero_CV']['harmed_rows'] > 0 for r in rows),
                    predicted_risk_violation_views=sum(r['predicted_risk_violations'] > 0 for r in rows),
                    switch_rate_percent_range=[100 * min(r['switch_rate'] for r in rows),
                                               100 * max(r['switch_rate'] for r in rows)])
    (run.PUBLIC / 'safety_summary.json').write_text(json.dumps(safety, indent=2) + '\n')
    lines += ['', 'Detector-track image pixels, obs8/pred12 rawstride12. Conditional, dependent, unadjusted locality CIs.',
        'Not seconds, metric, human gold, physical safety, independent confirmation, true 3D or foundation.',
        'No deployment promotion, Stage5C or SMC. Initial support mismatch is preserved in the parent amendment.', '']
    (run.PUBLIC / 'results.md').write_text('\n'.join(lines))
    (run.PUBLIC / 'summary_metrics.json').write_text(json.dumps(summary, separators=(',', ':'), allow_nan=False) + '\n')
    (run.PUBLIC / 'group_metrics.json').write_text(json.dumps(summary['groups'], indent=2) + '\n')


def completion(a):
    events = [json.loads(x) for x in (run.PRIVATE / 'events.jsonl').read_text().splitlines()]
    last = {e['pid']: e for e in events}
    for pid, e in last.items():
        process = subprocess.run(['ps', '-p', str(pid), '-o', 'args='], capture_output=True, text=True)
        if e['state'] != 'phase_complete' or 'hurdle_coverage.py' in process.stdout or 'hurdle_support_coverage.py' in process.stdout:
            raise ValueError('Amended process not terminal')
    parent = run.parent.PUBLIC / 'completion_checks.json'
    tests = json.loads(parent.read_text())['scoped_test_files'] + [
        'tests/test_m3w_hurdle_coverage.py', 'tests/test_m3w_hurdle_support_coverage.py']
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', *tests], cwd=ROOT, capture_output=True, text=True)
    (run.PUBLIC / 'scoped_tests.txt').write_text(result.stdout + result.stderr)
    if result.returncode or not re.search(r'\b227 passed\b', result.stdout):
        raise ValueError('Scoped tests failed; inspect saved output')
    receipt = dict(result_source='fresh_run_matched_count_replay_completion',
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), analysis_sha256=run.digest(run.PUBLIC / 'analysis.json'),
        summary_sha256=run.digest(run.PUBLIC / 'summary_metrics.json'), reporter_sha256=run.digest(Path(__file__)),
        safety_summary_sha256=run.digest(run.PUBLIC / 'safety_summary.json'),
        test_report_sha256=run.digest(run.PUBLIC / 'scoped_tests.txt'), new_training=False,
        decision_groups=36, views=216, parent_controls=72, matched_ranking_contrasts=72,
        separate_sorting_choices=216, separate_metric_reductions=864, additive_decompositions=108,
        tests_passed=227, scoped_test_files=tests, full_legacy_suite_run=False,
        amended_terminal_pids=sorted(last), amended_phases_finished=True,
        original_attempt='pid52141_exit1_common_support_guard_before_new_readout_preserved',
        reserved_roles_opened=False, deployment_changed=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False)
    (run.PUBLIC / 'completion_checks.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(views=216, tests=227, amended_processes_finished=True)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--complete', action='store_true')
    args = parser.parse_args()
    a = verify(); reports(a)
    if args.complete:
        completion(a)


if __name__ == '__main__':
    main()
