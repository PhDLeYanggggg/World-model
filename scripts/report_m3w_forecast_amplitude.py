"""Describe every fixed output control; do not select a coefficient from labels."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts import diagnose_m3w_forecast_amplitude as run


def number(value):
    return 'not_estimable' if value is None else f'{value:+.6f}'


def interval(value):
    if value['CI95'] is None:
        return 'not_estimable'
    return number(value['point']) + ' [' + ', '.join(number(x) for x in value['CI95']) + ']'


def risk_causes(records):
    def known_violation(p):
        return any(p[k] is not None and p[k] > .02+1e-12
                   for k in ('known_positive_risk', 'known_easy_positive_risk'))
    return dict(required_record_views=len(records),
        supported=sum(p['budget_status']['supported'] for p in records),
        empty_selected=sum(p['selected_count'] == 0 for p in records),
        undefined_easy=sum(p['easy_selected_risk_upper'] is None for p in records),
        known_selected_risk_violation=sum(known_violation(p) for p in records),
        unsupported_with_selected_unknown_no_known_violation=sum(
            not p['budget_status']['supported'] and p['selected_unknown'] > 0 and not known_violation(p) for p in records),
        failing_conditions={key: sum(not p['budget_status']['conditions'][key] for p in records)
            for key in ('all_selected_risk_upper', 'easy_selected_risk_upper', 'easy_degradation_upper')},
        overlapping_counts_not_unique_causal_attribution=True)


def build():
    summary = json.loads((run.PUBLIC/'summary.json').read_text())
    verification = json.loads((run.PUBLIC/'verification.json').read_text())
    complete = json.loads((run.PUBLIC/'complete.json').read_text())
    if (not verification['exact_replay'] or verification['groups'] != 144 or verification['amplitude_views'] != 720
            or verification['summary_sha256'] != run.sha(run.PUBLIC/'summary.json')):
        raise ValueError('Full separate exact replay required')
    checks, groups = [], []
    if len(complete['groups']) != 144:
        raise ValueError('All completed group references required')
    for ref in complete['groups']:
        path = ROOT/ref['path']
        if run.sha(path) != ref['sha256']:
            raise ValueError('Frozen group changed')
        group = json.loads(path.read_text()); groups.append(group)
        checks.extend(group['numerical_convexity'].values())
    if len(checks) != 720:
        raise ValueError('All numerical convexity checks required')
    feasible = []
    for family, amplitudes in summary['families'].items():
        for alpha, stats in amplitudes.items():
            for name in (*run.api.ARMS, 'factorized'):
                s, c = stats['safety'][name], stats['floor_contrasts'][name]
                if (s['supported_record_views'] == s['required_record_views']
                        and c['CI95'] is not None and c['CI95'][0] > 0):
                    feasible.append(dict(family=family, alpha=float(alpha), bank=name))
    cfg, values = run.load()
    neural_increment, causes = {}, {}
    for family in cfg['families']:
        neural_increment[family], causes[family] = {}, {}
        for alpha in cfg['amplitudes']:
            rows = [dict(source=g['source'], seed=g['seed'], result=g['amplitudes'][str(alpha)])
                    for g in groups if g['family'] == family]
            neural_increment[family][str(alpha)] = {mode+'_'+bound: run.readout.interval(rows,
                lambda r, m=mode, b=bound: r['comparisons']['physical_neural_minus_physical_only'][m][b], values[1])
                for mode in ('full', 'matched') for bound in ('known_pp', 'lower_pp', 'upper_pp')}
            names = [n for bank in (*run.api.ARMS, 'factorized') for n in (bank, bank+'_same_set')]
            causes[family][str(alpha)] = {name: risk_causes(
                [p[name] for r in rows for p in r['result']['recordings'].values()]) for name in names}
    gates = dict(all144groups720amplitude_views_complete=True, endpoint_exact=True,
        numerical_convexity_checks=720, positive_fitting_full_risk_controls=feasible,
        at_least_one_fitting_full_risk_and_completion_gain=bool(feasible),
        any_coefficient_selected_for_deployment=False, learned_costs_are_calibrated=False,
        held_scored=False, new_optimizer_updates=0, new_trees=0,
        independent_confirmation=False, deployment_promoted=False, submission_ready=False,
        stage5c_executed=False, smc_enabled=False)
    return dict(result_source='fresh_report_exact_verified_fitting_output_controls',
        summary_sha256=run.sha(run.PUBLIC/'summary.json'), summary=summary, gates=gates,
        descriptive_neural_bank_increment=neural_increment,
        descriptive_risk_failure_counts=causes,
        execution={key: complete[key] for key in ('pid', 'seconds', 'peak_RSS_bytes')},
        replay=json.loads((run.PRIVATE/'verification_runtime.json').read_text()),
        numerical_convexity=dict(checks=720,
            max_excess=max(c['max_numerical_excess'] for c in checks),
            known_row_views=sum(c['known_rows'] for c in checks),
            unknown_row_views=sum(c['unknown_rows'] for c in checks), independent_samples_claim=False))


def render(value):
    data = value['summary']; execution, replay = value['execution'], value['replay']
    names = (*run.api.ARMS, 'factorized')
    lines = ['# Fixed Forecast Amplitude: Fitting-Only Results', '', '## Provenance', '',
        'Fresh: all 144 output-control groups / 720 amplitude views and a separate exact',
        'inference replay. Cached_verified: source packets, frozen trajectories and cost',
        'estimators. Not_run: new learning, held/independent evaluation or deployment.', '',
        f"PID {execution['pid']}, exit 0, takes {execution['seconds']:.6f}s; peak RSS {execution['peak_RSS_bytes']:,} B.",
        f"Replay PID {replay['pid']}, exit 0, takes {replay['seconds']:.6f}s.",
        'All original endpoint choices and full metrics match the preceding experiment.',
        'Every fixed-mask convexity check passes. This numerical property does not',
        'certify the original learned conditional costs or real-world safety.', '',
        '## Complete Reselected Policies', '',
        '| Family | Alpha | Bank | Known ADE gain vs floor (%) | Required recording risk support | Undefined easy | Known all/easy violations | Selected (unknown) |',
        '|---|---:|---|---:|---:|---:|---:|---:|']
    for family, amplitudes in data['families'].items():
        for alpha in data['amplitudes']:
            stats = amplitudes[str(alpha)]
            for name in names:
                s = stats['safety'][name]
                lines.append(f"| {family} | {alpha:g} | {name} | {number(stats['accuracy'][name]['point'])} | "
                    f"{s['supported_record_views']}/{s['required_record_views']} | {s['undefined_easy_record_views']} | "
                    f"{s['known_all_violations']}/{s['known_easy_violations']} | {s['selected']} ({s['selected_unknown']}) |")
    lines += ['', 'All coefficients and banks are reported, not an outcome-selected winning model.',
        'Every risk row retains all required recordings. Undefined risk is not safe zero.', '',
        '## Reselection Versus The Same Frozen Endpoint Policy', '',
        'Positive pp means the fractional policy improves its own alpha=1 endpoint.',
        'Count matching concerns utility only, never the safety contract.', '',
        '| Family | Alpha | Bank | Full known change (pp), fitting 95% CI | Full completion-lower (pp), fitting 95% CI | Matched known change (pp), fitting 95% CI |',
        '|---|---:|---|---|---|---|']
    for family, amplitudes in data['families'].items():
        for alpha in data['amplitudes']:
            for name in names:
                c = amplitudes[str(alpha)]['contrasts'][name]
                lines.append(f'| {family} | {alpha:g} | {name} | ' + ' | '.join(
                    interval(c[k]) for k in ('full_known_pp', 'full_lower_pp', 'matched_known_pp')) + ' |')
    lines += ['', '## Output Change Without Reselection', '',
        'These controls retain the endpoint policy selection set, except analytic no-ops.',
        'They distinguish a smaller output from the newly admitted actions in the table above.', '',
        '| Family | Alpha | Bank | Known ADE gain vs floor (%) | Recording support | Undefined easy | Selected (unknown) |',
        '|---|---:|---|---:|---:|---:|---:|']
    for family, amplitudes in data['families'].items():
        for alpha in data['amplitudes']:
            for bank in names:
                name = bank + '_same_set'; stats = amplitudes[str(alpha)]; s = stats['safety'][name]
                lines.append(f"| {family} | {alpha:g} | {bank} | {number(stats['accuracy'][name]['point'])} | "
                    f"{s['supported_record_views']}/{s['required_record_views']} | {s['undefined_easy_record_views']} | "
                    f"{s['selected']} ({s['selected_unknown']}) |")
    lines += ['', '## Neural Bank Versus The Physical Bank At The Same Amplitude', '',
        'Descriptive contrasts from the same verified groups, not an alpha-selection rule.', '',
        '| Family | Alpha | Mixed minus physical known gain (pp), fitting 95% CI | Completion-lower (pp), fitting 95% CI |',
        '|---|---:|---|---|']
    for family in data['families']:
        for alpha in data['amplitudes']:
            c = value['descriptive_neural_bank_increment'][family][str(alpha)]
            lines.append(f'| {family} | {alpha:g} | ' + interval(c['full_known_pp']) + ' | ' + interval(c['full_lower_pp']) + ' |')
    lines += ['', '## Risk Failure Accounting', '',
        '| Family | Alpha | Policy | Known selected-risk violation views | Unsupported with selected unknown but no known violation | Undefined easy |',
        '|---|---:|---|---:|---:|---:|']
    for family in data['families']:
        for alpha in data['amplitudes']:
            for name in ('physical_neural', 'physical_neural_same_set'):
                c = value['descriptive_risk_failure_counts'][family][str(alpha)][name]
                lines.append(f"| {family} | {alpha:g} | {name} | {c['known_selected_risk_violation']} | "
                    f"{c['unsupported_with_selected_unknown_no_known_violation']} | {c['undefined_easy']} |")
    lines += ['', 'These counts overlap and do not uniquely identify why a prediction is wrong.',
        'An unknown-outcome bound failing is not evidence that the unknown realized error',
        'was large. Missing support is not imputed away or treated as observed harm.', '',
        '## Evidence Boundaries', '',
        'Cost transport uses convex-loss surrogate benefit/harm bounds, not freshly fitted',
        'expected fractional errors. Actual ADE/FDE and canonical completion envelopes',
        'are recomputed from the new paths; endpoint errors are not multiplied by alpha.',
        'Labels and future masks enter scoring only. The zero-amplitude engineering',
        'control is an analytic no-op and does not pass empty selected-risk support.', '',
        'Intervals use 3,000 locality resamples after repeated context/head-seed averaging.',
        'They are fitting-resubstitution, not generalization, whole-pipeline OOF or',
        'independent confirmation. Endpoint percentages that lack a required denominator',
        'remain not_estimable in summary.json. Completion utility divides by fixed known',
        'floor-error mass, not an identified complete-population error denominator.', '',
        'The 444 recording views per family repeat 74 unique fitting recordings. Neither',
        'query/amplitude repetitions nor three head seeds create new forecasting pipelines.',
        'The factorized policy still uses twice the cost-estimator capacity. Even a positive',
        'fitting amplitude control cannot establish learned neural dynamics or deployment.', '',
        'EuropeanSquares SOURCE detector-track midpoint pixels, obs8/pred12/stride12 raw',
        'frames; not SDD human gold, metric/seconds, true 3D or foundation evidence.',
        'Independent roles and DroneCrowd stay closed. No Stage5C or SMC.', '',
        '## Gates', '', *['- '+key+': '+json.dumps(result) for key, result in value['gates'].items()], '']
    return '\n'.join(lines)


if __name__ == '__main__':
    value = build(); run.once(run.PUBLIC/'evidence.json', value); run.once(run.PUBLIC/'gates.json', value['gates'])
    path = run.PUBLIC/'readout.md'; content = render(value)
    if path.exists() and path.read_text() != content:
        raise ValueError('Frozen output report changed')
    path.write_text(content)
    print(json.dumps(dict(execution=value['execution'], gates=value['gates'])))
