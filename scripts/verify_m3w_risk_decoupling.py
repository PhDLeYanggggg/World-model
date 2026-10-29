"""Receipt-only arithmetic checks, separate from frozen fitting inference."""
import argparse
import json
import math
from pathlib import Path
import statistics

ARMS = ('origin', 'coupled', 'estimation')
COHORTS = ('eligible', 'origin')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def compare(a, b):
    if isinstance(b, dict):
        require(isinstance(a, dict) and set(a) == set(b), 'Summary schema changed')
        for key in b:
            compare(a[key], b[key])
    elif b is None or isinstance(b, (bool, str, list)):
        require(a == b, 'Summary field differs')
    else:
        require(a is not None and math.isfinite(a) and
                math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-12), 'Arithmetic differs')


def verify(rows, summary, expected):
    from scripts.report_m3w_shared_train_risk import check_policy
    keys = [(r['context'], r['source'], r['seed']) for r in rows]
    require(len(keys) == len(set(keys)) == 72 and set(keys) == set(expected),
            'Complete72 unique registered views required')
    checks = dict(cohort_target_checks=0, policy_checks=0, matched_count_checks=0)
    for row in rows:
        require(row['scope'] == 'fit' and not row['held_inference'] and
                not row['independent_roles_read'], 'Data role changed')
        r = row['result']
        require(r['protected_outputs_exact'] and not r['inference_label_filter'],
                'Causal protection changed')
        require(set(r['policies']) == set(r['cohorts']) == set(ARMS), 'Missing arm')
        require(r['matched_counts'][0] == r['matched_counts'][1], 'Unmatched actions')
        require(0 <= r['matched_counts'][0] <= min(r['policies'][a]['selected_count']
                for a in ('coupled', 'estimation')), 'Matched count exceeds full policy')
        original = r['cohorts']['origin']['origin']
        require(original['rows'] == r['policies']['origin']['selected_count'] and
                original['unknown'] == r['policies']['origin']['selected_unknown'], 'Original cohort changed')
        checks['matched_count_checks'] += 1
        for arm in ARMS:
            check_policy(r['policies'][arm]); checks['policy_checks'] += 1
            for cohort in COHORTS:
                c = r['cohorts'][arm][cohort]
                reference = r['cohorts']['origin'][cohort]
                require(c['rows'] == c['known']+c['unknown'], 'Unknown rows removed')
                require(all(isinstance(c[k], int) and c[k] >= 0 and c[k] == reference[k]
                            for k in ('rows', 'known', 'unknown')), 'Fixed cohort differs')
                require(c['rows'] <= r['cohorts'][arm]['eligible']['rows'], 'Cohort outside eligibility')
                for target in ('all', 'easy'):
                    v, ref = c[target], reference[target]
                    require((v['MSE'] is None) == (c['known'] == 0), 'Undefined MSE mishandled')
                    if v['MSE'] is not None:
                        require(math.isfinite(v['MSE']) and v['MSE'] >= 0, 'Invalid MSE')
                    require(0 <= v['false_safe_rows'] <= c['known'], 'False-safe count invalid')
                    for key in ('observed_harm', 'predicted_harm', 'observed_excess', 'predicted_excess'):
                        require(math.isfinite(v[key]), 'Nonfinite score mass')
                    compare(v['observed_harm'], ref['observed_harm'])
                    compare(v['observed_excess'], ref['observed_excess'])
                    checks['cohort_target_checks'] += 1
    means, counts, safety = {}, {}, {}
    for arm in ARMS:
        means[arm], counts[arm] = {}, {}
        for cohort in COHORTS:
            values = [r['result']['cohorts'][arm][cohort] for r in rows]
            counts[arm][cohort] = {k: sum(v[k] for v in values) for k in ('rows', 'known', 'unknown')}
            means[arm][cohort] = {}
            for target in ('all', 'easy'):
                numbers = [v[target]['MSE'] for v in values]
                means[arm][cohort][target] = statistics.fmean(numbers) if None not in numbers else None
        policies = [r['result']['policies'][arm] for r in rows]
        safety[arm] = dict(selected=sum(p['selected_count'] for p in policies),
            unknown_selected=sum(p['selected_unknown'] for p in policies),
            known_easy_violations=sum(p['known_easy_positive_risk'] is not None and
                                      p['known_easy_positive_risk'] > .02+1e-12 for p in policies),
            undefined_easy=sum(p['easy_selected_risk_upper'] is None for p in policies),
            full_support=sum(p['finite_completion_supported'] for p in policies))
    conditions = {}
    for reference in ('origin', 'coupled'):
        for cohort in COHORTS:
            for target in ('all', 'easy'):
                a, b = means['estimation'][cohort][target], means[reference][cohort][target]
                strict = (cohort, target) == ('origin', 'easy')
                conditions[f'{cohort}_{target}_vs_{reference}'] = (a is not None and b is not None and
                    (a < b-1e-12 if strict else a <= b+1e-12))
    pairs = sorted({(r['context'], r['seed']) for r in rows})
    require(len(pairs) == 18, 'Complete18 matched head pairs required')
    wins = 0
    for c, s in pairs:
        group = [r for r in rows if (r['context'], r['seed']) == (c, s)]
        require(len(group) == 4, 'Four source views per head pair required')
        values = {a: [r['result']['cohorts'][a]['origin']['easy']['MSE'] for r in group] for a in ARMS}
        if all(None not in v for v in values.values()):
            wins += statistics.fmean(values['estimation']) < min(
                statistics.fmean(values['origin']), statistics.fmean(values['coupled']))-1e-12
    conditions['at_least12_of18_easy_MSE_pair_wins'] = wins >= 12
    utility = {}
    for k in ('full_utility_pp', 'matched_utility_pp'):
        values = [r['result'][k] for r in rows]
        utility[k] = statistics.fmean(values) if None not in values else None
    rebuilt = dict(mean_view_MSE=means, cohort_counts=counts, easy_pair_wins=wins,
        fit_routing_conditions=conditions, fit_eligible_for_later_held_design=all(conditions.values()),
        failed_conditions=[k for k, v in conditions.items() if not v],
        fitting_policy_diagnostics=safety, utility_diagnostics=utility,
        fitted_diagnostics_not_validation=True, independent_confirmation=False, held_inference=False,
        deployment_changed=False, stage5c_executed=False, smc_enabled=False)
    compare(summary, rebuilt)
    return dict(checks=checks, summary_recomputed=True, routing_conditions_checked=len(conditions),
                easy_pair_wins=wins, fit_eligible_for_later_held_design=all(conditions.values()))


def execute(check):
    from scripts import run_m3w_risk_decoupling as run
    reg, _, _ = run.admission(); run.models(reg)
    public = run.PUBLIC
    finish = json.loads((public/'complete.json').read_text())
    require(finish['summary_sha256'] == run.sha(public/'summary.json') and
            finish['registration_sha256'] == run.sha(run.REG) and
            finish['training_freeze_sha256'] == run.sha(public/'training_freeze.json'), 'Freeze changed')
    rows = []
    for ref in finish['readouts']:
        path = run.ROOT/ref['path']
        require(not path.is_symlink() and path.resolve().is_relative_to((public/'readout').resolve())
                and run.sha(path) == ref['sha256'], 'Receipt changed')
        row = json.loads(path.read_text())
        require(row['registration_sha256'] == run.sha(run.REG) and
                row['training_freeze_sha256'] == run.sha(public/'training_freeze.json'), 'Row lineage changed')
        rows.append(row)
    result = verify(rows, json.loads((public/'summary.json').read_text()), run.parent.expected(reg))
    result.update(complete_sha256=run.sha(public/'complete.json'), summary_sha256=run.sha(public/'summary.json'),
        verifier_sha256=run.sha(Path(__file__)), result_source='cached_verified_separate_receipt_arithmetic',
        new_inference=False, independent_external_replication=False, held_inference=False,
        deployment_changed=False)
    path = public/'verification.json'
    if check:
        require(result == json.loads(path.read_text()), 'Verification differs')
    else:
        run.once(path, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true')
    execute(parser.parse_args().check)
