"""Separate arithmetic and reference-boundary checks for donor conditioning."""
import argparse
import json
import math
from pathlib import Path
import statistics

from scripts import run_m3w_selection_conditioned_analogue as run
from scripts import verify_m3w_selected_analogue as old
from src.evaluation import m3w_selected_analogue as legacy_summary

require, compare = old.require, old.compare


def verify(rows, summary, expected):
    keys = [(r['context'], r['source'], r['seed']) for r in rows]
    require(len(keys) == len(set(keys)) == 72 and set(keys) == set(expected), 'Complete72 frozen views required')
    legacy_checks = []
    for prefix, refkey in (('pooled', 'full_references'), ('selected', 'references')):
        names = dict(zip(('origin', 'estimation', prefix+'_mean', prefix+'_nearest32', prefix+'_shuffled32'), old.ARMS))
        mapped = [dict(r, scores={names[k]: v for k, v in r['scores'].items() if k in names},
                       references=r[refkey]) for r in rows]
        # Reuse the separately implemented count/risk receipt validator, not the new screen function.
        legacy_checks.append(old.verify(mapped, legacy_summary.summarize(mapped, expected), expected)['checks'])
    banks = 0; decompositions = 0
    for r in rows:
        require(set(r['scores']) == set(run.api.ARMS), 'Eight frozen arms required')
        require(len(r['references']) == len(r['full_references']), 'Reference roster changed')
        for sel, full in zip(r['references'], r['full_references']):
            source, rec = json.loads(sel['recording'])
            require(source == r['source'] == sel['query_source'] and rec == sel['local_recording'], 'Namespace mismatch')
            for key in ('recording', 'query_rows', 'query_sha256', 'causal_donors'):
                require(sel[key] == full[key], 'Query/normalization cohort changed')
            require(sel['unselected_donors'] == 0 and sel['known_donors'] <= full['known_donors'], 'Unselected labels entered bank')
            if sel['supported']:
                require(all(sel[key] == full[key] for key in ('mean_sha256', 'std_sha256')), 'Normalizer refit on selection')
            require(set(sel['known_donors_by_source']) == set(full['known_donors_by_source']), 'Cross-context donor')
            for bank in (sel, full):
                require(sum(bank['known_donors_by_source'].values()) == bank['known_donors'], 'Donor counts differ')
                require(all(isinstance(n, int) and 0 <= n <= bank['known_donors'] for n in bank['donor_events']), 'Bad donor events')
                require(all(math.isfinite(x) and 0 <= x <= bank['known_donors']+1e-8 for x in bank['donor_fraction_sums']), 'Bad donor fractions')
            banks += 1
        for prefix in ('pooled', 'selected'):
            score = r['scores'][prefix+'_nearest32']
            for t in ('all', 'easy'):
                d = r['decomposition'][prefix][t]
                require(d['known'] == score['supported_known'] and d['events'] == score[t]['events'], 'Decomposition population changed')
                require(0 <= d['predicted_event_mass'] <= d['known']+1e-8, 'Invalid probability mass')
                require(d['positive_queries_with_positive_donors']+d['positive_queries_without_positive_donors'] == d['events'], 'Zero-event support hidden')
                require(all(math.isfinite(d[k]) and d[k] >= 0 for k in
                    ('brier_sum', 'event_harm_SSE', 'non_event_harm_SSE', 'harm_SSE', 'positive_severity_SSE')), 'Invalid nonnegative loss')
                compare(d['harm_SSE'], d['event_harm_SSE']+d['non_event_harm_SSE'])
                compare(d['predicted_harm_on_events']+d['predicted_harm_on_non_events'], score[t]['predicted_harm'])
                compare(d['observed_harm'], score[t]['observed_harm'])
                decompositions += 1
    means = {a: {t: statistics.fmean(r['scores'][a][t]['full_cohort_MSE'] for r in rows)
                 if all(r['scores'][a][t]['full_cohort_MSE'] is not None for r in rows) else None
                 for t in ('all', 'easy')} for a in run.api.ARMS}
    wins = 0
    for context, seed in sorted({(r['context'], r['seed']) for r in rows}):
        group = [r for r in rows if (r['context'], r['seed']) == (context, seed)]
        require(len(group) == 4, 'Four sources per pair required')
        m = {a: [r['scores'][a]['easy']['full_cohort_MSE'] for r in group]
             for a in ('origin', 'pooled_nearest32', 'selected_nearest32')}
        if all(None not in values for values in m.values()):
            wins += statistics.fmean(m['selected_nearest32']) < min(statistics.fmean(m['origin']), statistics.fmean(m['pooled_nearest32']))-1e-12
    n, o, f = (means[a] for a in ('selected_nearest32', 'origin', 'pooled_nearest32'))
    checks = dict(all_views_supported=all(r['scores']['selected_nearest32']['unsupported'] == 0 for r in rows),
        easy_MSE_beats_origin=n['easy'] is not None and n['easy'] < o['easy']-1e-12,
        easy_MSE_beats_full_bank=n['easy'] is not None and n['easy'] < f['easy']-1e-12,
        all_MSE_not_worse_than_full_bank=n['all'] is not None and n['all'] <= f['all']+1e-12,
        at_least12_of18_pair_wins=wins >= 12)
    rebuilt = dict(views=72, equal_view_MSE=means, pair_wins=int(wins), magnitude_screen=checks,
        magnitude_screen_pass=all(checks.values()), occurrences={k: sum(r['scores']['selected_nearest32'][k] for r in rows)
        for k in ('rows', 'known', 'unknown', 'supported', 'unsupported')}, fitting_diagnostic_only=True,
        whole_pipeline_OOF=False, held_inference=False, independent_roles_read=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False)
    compare(rebuilt, summary)
    return dict(legacy_arm_target_checks=sum(c['arm_targets'] for c in legacy_checks),
        selected_reference_checks=banks, decomposition_checks=decompositions,
        magnitude_conditions_checked=len(checks), magnitude_screen_pass=all(checks.values()))


def execute(check):
    reg, _, _ = run.admission(); done = json.loads((run.PUBLIC/'complete.json').read_text())
    require(done['registration_sha256'] == run.sha(run.REG) and done['summary_sha256'] == run.sha(run.PUBLIC/'summary.json'), 'Freeze changed')
    olddone = json.loads((run.parent.PUBLIC/'complete.json').read_text())
    oldrefs = {Path(r['path']).name: r for r in olddone['groups']}; rows = []
    for ref in done['groups']:
        p = run.ROOT/ref['path']
        require(not p.is_symlink() and p.resolve().is_relative_to((run.PUBLIC/'groups').resolve()) and run.sha(p) == ref['sha256'], 'Group binding changed')
        r = json.loads(p.read_text()); require(r['registration_sha256'] == run.sha(run.REG), 'Registration changed')
        prev = run.parent.PUBLIC/'groups'/p.name
        require(run.sha(prev) == oldrefs[p.name]['sha256'] == r['parent_group_sha256'], 'Parent changed')
        oldrow = json.loads(prev.read_text())
        require(r['parent_cohort_sha256'] == oldrow['parent_cohort_sha256'], 'Frozen query changed')
        for arm in run.api.ARMS[:5]:
            compare(r['scores'][arm], oldrow['scores'][arm])
            require(r['prediction_sha256'][arm] == oldrow['prediction_sha256'][arm], 'Parent predictions changed')
        rows.append(r)
    result = verify(rows, json.loads((run.PUBLIC/'summary.json').read_text()), run.parent.previous.previous.parent.expected(reg))
    result.update(complete_sha256=run.sha(run.PUBLIC/'complete.json'), verifier_sha256=run.sha(Path(__file__)),
        summary_sha256=run.sha(run.PUBLIC/'summary.json'), prior_control_comparisons=360,
        result_source='cached_verified_separate_receipt_arithmetic', new_inference=False, independent_replication=False)
    p = run.PUBLIC/'verification.json'
    if check: require(result == json.loads(p.read_text()), 'Verification replay differs')
    else: run.once(p, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--check', action='store_true'); execute(p.parse_args().check)
