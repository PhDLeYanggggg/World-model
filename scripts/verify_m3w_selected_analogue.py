"""Independent receipt arithmetic for the fixed-selection analogue probe."""
import argparse
import json
import math
from pathlib import Path
import statistics

from scripts.verify_m3w_risk_decoupling import compare, require

ARMS = ('origin', 'estimation', 'source_mean', 'nearest32', 'shuffled_nearest32')
ANALOGUES = ARMS[2:]


def verify(rows, summary, expected):
    keys = [(r['context'], r['source'], r['seed']) for r in rows]
    require(len(keys) == len(set(keys)) == 72 and set(keys) == set(expected), 'Complete72 views required')
    checks = dict(views=0, arm_targets=0, recording_banks=0)
    for row in rows:
        require(row['scope'] == 'fit' and row['prior_predictions_exact'] and
                not any(row[k] for k in ('whole_pipeline_OOF', 'held_inference',
                    'independent_roles_read', 'deployment_changed')), 'Evidence scope changed')
        require(set(row['scores']) == set(ARMS), 'All five controls required')
        ref = row['scores']['nearest32']
        for arm in ARMS:
            s = row['scores'][arm]
            counts = ('rows', 'known', 'unknown', 'supported', 'unsupported',
                      'supported_known', 'supported_unknown')
            require(all(isinstance(s[k], int) and s[k] >= 0 for k in counts), 'Invalid count')
            require(s['rows'] == s['known']+s['unknown'] == s['supported']+s['unsupported'], 'Lost rows')
            require(s['supported'] == s['supported_known']+s['supported_unknown'] and
                    s['supported_known'] <= s['known'] and s['supported_unknown'] <= s['unknown'], 'Lost support')
            require(all(s[k] == ref[k] for k in ('rows', 'known', 'unknown')), 'Fixed cohort changed')
            if arm in ANALOGUES:
                require(all(s[k] == ref[k] for k in counts), 'Analogue support mismatch')
            else:
                require(s['supported'] == s['rows'], 'Frozen neural control support changed')
            require(0 <= s['benefit_harm_envelope_exceedances'] <= s['supported'], 'Invalid envelope count')
            for target in ('all', 'easy'):
                v = s[target]; n = s['supported_known']
                require((v['full_cohort_MSE'] is None) == (n == 0 or s['unsupported'] > 0), 'Unsupported primary MSE')
                require((v['supported_only_MSE'] is None) == (n == 0), 'Known support not reported')
                for k in ('full_cohort_MSE', 'supported_only_MSE'):
                    require(v[k] is None or (math.isfinite(v[k]) and v[k] >= 0), 'Invalid squared error')
                if v['full_cohort_MSE'] is not None:
                    compare(v['full_cohort_MSE'], v['supported_only_MSE'])
                require(0 <= v['events'] <= n and 0 <= v['false_safe_rows'] <= n, 'Invalid event count')
                compare(v['event_prevalence'], v['events']/n if n else None)
                require((v['AUROC'] is None) == (v['events'] == 0 or v['events'] == n), 'Undefined AUROC hidden')
                require((v['AP'] is None) == (v['events'] == 0), 'Undefined AP hidden')
                for k in ('AUROC', 'AP'):
                    require(v[k] is None or 0 <= v[k] <= 1, 'Invalid ranking metric')
                require(all(math.isfinite(v[k]) and v[k] >= 0 for k in
                            ('observed_harm', 'predicted_harm')), 'Invalid harm mass')
                if arm in ANALOGUES or not ref['unsupported']:
                    for k in ('events', 'event_prevalence', 'observed_harm'):
                        compare(v[k], ref[target][k])
                checks['arm_targets'] += 1
        banks = row['references']
        require(len({r['recording'] for r in banks}) == len(banks), 'Duplicate query recording')
        require(sum(r['query_rows'] for r in banks) == ref['rows'], 'Reference queries incomplete')
        require(sum(r['query_rows'] for r in banks if r['supported']) == ref['supported'], 'Unsupported queries hidden')
        for bank in banks:
            require(bank['same_recording_donors'] == 0 and
                    bank['causal_donors'] >= bank['known_donors'] >= 0 and
                    bank['supported'] == (bank['known_donors'] >= 32), 'Donor boundary changed')
            require(not bank['supported'] or (bank['donor_recordings'] > 0 and
                    math.isfinite(bank['mean_neighbor_distance']) and bank['mean_neighbor_distance'] >= 0), 'Invalid reference support')
            checks['recording_banks'] += 1
        checks['views'] += 1
    means = {a: {t: {m: statistics.fmean(r['scores'][a][t][m] for r in rows)
                    if all(r['scores'][a][t][m] is not None for r in rows) else None
                    for m in ('full_cohort_MSE', 'AUROC', 'AP')}
                    for t in ('all', 'easy')} for a in ARMS}
    pairs = sorted({(r['context'], r['seed']) for r in rows})
    require(len(pairs) == 18, 'All18 head-context pairs required')
    wins = 0
    for context, seed in pairs:
        group = [r for r in rows if (r['context'], r['seed']) == (context, seed)]
        require(len(group) == 4, 'Four sources per pair required')
        values = {a: [r['scores'][a]['easy']['full_cohort_MSE'] for r in group] for a in ANALOGUES}
        if all(None not in v for v in values.values()):
            wins += statistics.fmean(values['nearest32']) < min(
                statistics.fmean(values['source_mean']), statistics.fmean(values['shuffled_nearest32']))-1e-12
    conditions = {f'easy_MSE_vs_{other}': means['nearest32']['easy']['full_cohort_MSE'] is not None and
                  means[other]['easy']['full_cohort_MSE'] is not None and
                  means['nearest32']['easy']['full_cohort_MSE'] < means[other]['easy']['full_cohort_MSE']-1e-12
                  for other in ('source_mean', 'shuffled_nearest32')}
    conditions['at_least12_of18_pair_wins'] = wins >= 12
    rebuilt = dict(views=72, equal_view_metrics=means, pair_wins=wins, information_screen=conditions,
        conditional_signal_screen_pass=all(conditions.values()),
        occurrences={k: sum(r['scores']['nearest32'][k] for r in rows) for k in
            ('rows', 'known', 'unknown', 'supported', 'unsupported', 'supported_known', 'supported_unknown')},
        incomplete_MSE_views=sum(r['scores']['nearest32']['easy']['full_cohort_MSE'] is None for r in rows),
        same_recording_donors=0, fitting_diagnostic_only=True, whole_pipeline_OOF=False,
        held_inference=False, independent_roles_read=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False)
    compare(summary, rebuilt)
    return dict(checks=checks, summary_recomputed=True, screen_conditions_checked=3,
                information_screen_pass=all(conditions.values()))


def execute(check):
    from scripts import run_m3w_selected_analogue as run
    reg, _, _ = run.admission()
    done = json.loads((run.PUBLIC/'complete.json').read_text())
    require(done['summary_sha256'] == run.sha(run.PUBLIC/'summary.json') and
            done['registration_sha256'] == run.sha(run.REG), 'Completion binding changed')
    rows = []
    for receipt in done['groups']:
        p = run.ROOT/receipt['path']
        require(not p.is_symlink() and p.resolve().is_relative_to((run.PUBLIC/'groups').resolve()) and
                run.sha(p) == receipt['sha256'], 'Group binding changed')
        row = json.loads(p.read_text())
        require(row['registration_sha256'] == run.sha(run.REG), 'Registration changed')
        old = json.loads((run.previous.PUBLIC/'readout'/p.name).read_text())
        for arm in ('origin', 'estimation'):
            current = row['scores'][arm]; prior = old['result']['cohorts'][arm]['origin']
            for k in ('rows', 'known', 'unknown'): compare(current[k], prior[k])
            for t in ('all', 'easy'):
                compare(current[t]['full_cohort_MSE'], prior[t]['MSE'])
                for k in ('observed_harm', 'predicted_harm', 'false_safe_rows'):
                    compare(current[t][k], prior[t][k])
        rows.append(row)
    result = verify(rows, json.loads((run.PUBLIC/'summary.json').read_text()), run.previous.parent.expected(reg))
    result.update(complete_sha256=run.sha(run.PUBLIC/'complete.json'),
        summary_sha256=run.sha(run.PUBLIC/'summary.json'), verifier_sha256=run.sha(Path(__file__)),
        prior_control_cohort_checks=144, result_source='cached_verified_separate_receipt_arithmetic',
        new_inference=False, independent_external_replication=False)
    p = run.PUBLIC/'verification.json'
    if check: require(result == json.loads(p.read_text()), 'Verification replay differs')
    else: run.once(p, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true')
    execute(parser.parse_args().check)
