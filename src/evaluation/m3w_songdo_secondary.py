"""Report secondary endpoints from completed Songdo score receipts only."""

from collections import Counter
from math import isfinite

from src.evaluation import m3w_songdo_matched_readout as readout


ABSOLUTE = ('available_ade_native', 'available_fde_native', 'available_ade_p95_native',
            'complete_roster_ade_native', 'complete_roster_fde_native')
PAIR = ('known_pp', 'lower_pp', 'relative_available_native', 'relative_lower_native')
RISK = ('completion_all_risk_upper', 'completion_easy_risk_upper', 'known_easy_net_degradation')


def aggregate(rows, components, metrics):
    views = {f: {int(s): {'matched_contrasts': {'arm': r}} for s, r in v.items()} for f, v in rows.items()}
    flat = [r for v in rows.values() for r in v.values()]
    result = {}
    for k in metrics:
        values = [r[k] for r in flat if r.get(k) is not None]
        if not all(isfinite(x) for x in values):
            raise ValueError(f'Nonfinite secondary metric: {k}')
        r = readout.component_interval(views, components, 'arm', k)
        estimates = [(x, g) for g, x in r['by_component'].items() if x is not None]
        worst = max(estimates, default=None) if k in ABSOLUTE else None
        result[k] = dict(**r, defined_seed_views=len(values), required_seed_views=len(flat),
                         worst_estimable_source=worst[1] if worst else None,
                         worst_estimable_source_value=worst[0] if worst else None)
    return result


def section(rows, components, metrics):
    result = aggregate(rows, components, metrics)
    return dict(metrics=result, worst_flight_seed_available_ade_native=max(
        (r['available_ade_native'] for v in rows.values() for r in v.values()
         if r.get('available_ade_native') is not None), default=None))


def risk_status(row):
    risk = row['risk']
    if risk['selected'] != 2:
        raise ValueError('An edited arm must retain the exact two-edit budget')
    ratios = [risk[k] for k in RISK]
    if any(x is not None and not isfinite(x) for x in ratios):
        raise ValueError('Nonfinite original risk ratio')
    supported = all(x is not None and x <= .02 for x in ratios)
    if supported != risk['supported']:
        raise ValueError('Original risk flags disagree with their components')
    gains = [row[k]['lower_pp'] for k in ('vs_cv', 'vs_ols4')]
    dual = supported and all(x is not None and x > 0. for x in gains)
    if dual != row['dual_pass']:
        raise ValueError('Original dual-reference condition changed')
    return dict(edited=1, original_risk_pass=int(supported), dual_pass=int(dual),
                undefined_risk=int(any(x is None for x in ratios)),
                measured_risk_violation=int(any(x is not None and x > .02 for x in ratios)),
                no_edit_not_certified=0)


def summarize(records, components, *, matched_only):
    if set(records) != set(components) or not records:
        raise ValueError('Expected the registered confirmation flights')
    seeds = {str(s) for s in readout.SEEDS}
    if any(set(v) != seeds for v in records.values()):
        raise ValueError('A flight lacks all three frozen seeds')
    for v in records.values():
        for r in v.values():
            if ('released' in r) == matched_only:
                raise ValueError('Risk or release data does not match registered mode')
    matched = {f: {s: r['matched'] for s, r in v.items()} for f, v in records.items()}

    def build(get):
        return {f: {s: get(r) for s, r in v.items()} for f, v in records.items()}

    report = dict(flights=len(records), components=len(set(components.values())),
                  flight_seed_views=len(records)*len(seeds), matched_only=matched_only,
                  matched_seed_views=sum(r['matched_eligible'] for v in matched.values() for r in v.values()),
                  matched={}, matched_joint_minus={}, full_roster_motion_reference={})
    for a in ('cv', 'ols4', *readout.ARMS):
        rows = build(lambda r: r['matched']['arms'][a] if r['matched']['matched_eligible'] else {})
        stats = report['matched'][a] = section(rows, components, (*ABSOLUTE, 'query_intervention_rate'))
        if a in ('cv', 'ols4'):
            full = build(lambda r: r['matched']['arms'][a])
            report['full_roster_motion_reference'][a] = section(full, components, ABSOLUTE)
            continue
        for ref in ('vs_cv', 'vs_ols4'):
            stats[ref] = aggregate(build(lambda r: r['matched']['arms'][a][ref]
                                 if r['matched']['matched_eligible'] else {}), components, PAIR)
        if not matched_only:
            risk = Counter()
            for v in rows.values():
                for r in v.values():
                    if r:
                        risk.update(risk_status(r))
            stats['original_risk'] = dict(risk)
    for a in readout.ARMS[:-1]:
        report['matched_joint_minus'][a] = aggregate(build(lambda r: r['matched']['matched_contrasts'].get(a, {})),
                                                       components, PAIR)
    if matched_only:
        report['risk_and_release'] = 'not_run_no_calibration'
        return report
    report['released'] = {}
    for a in readout.ARMS:
        rows = build(lambda r: r['released'][a]['metrics'])
        stats = report['released'][a] = section(rows, components, (*ABSOLUTE, 'intervention_rate'))
        counts, by_source = Counter(), {g: Counter() for g in sorted(set(components.values()))}
        for f, v in records.items():
            for s, r in v.items():
                p = rows[f][s]
                selected = r['released'][a]['edited']
                if selected:
                    if not r['matched']['matched_eligible'] or p != r['matched']['arms'][a]:
                        raise ValueError('Released forecast is not the frozen proposal')
                    q = risk_status(p)
                else:
                    cv = r['matched']['arms']['cv']
                    if (any(p[k] != cv[k] for k in ABSOLUTE) or p['vs_cv'] != cv['vs_cv']
                            or p['risk'] is not None or p['intervention_rate'] != 0.):
                        raise ValueError('A fallback must match the original CV receipt and not be certified')
                    q = dict(edited=0, original_risk_pass=0, dual_pass=0, undefined_risk=0,
                             measured_risk_violation=0, no_edit_not_certified=1)
                counts.update(q)
                by_source[components[f]].update(q)
        stats['risk'] = dict(counts)
        stats['risk_by_source'] = {g: dict(v) for g, v in by_source.items()}
        for ref in ('vs_cv', 'vs_ols4'):
            stats[ref] = aggregate(build(lambda r: r['released'][a]['metrics'][ref]), components, PAIR)
        stats['edited_only'] = aggregate(build(lambda r: {
            'signed_easy_degradation': r['released'][a]['metrics']['risk']['known_easy_net_degradation'],
            'easy_known_gain_percent': r['released'][a]['metrics']['easy_known_gain_percent'],
            'hard_known_gain_percent': r['released'][a]['metrics']['hard_known_gain_percent'],
        } if r['released'][a]['edited'] else {}), components,
            ('signed_easy_degradation', 'easy_known_gain_percent', 'hard_known_gain_percent'))
    return report
