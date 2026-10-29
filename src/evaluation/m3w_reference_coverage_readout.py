"""Exact comparator replay and fixed-population diagnostic bookkeeping."""
import numpy as np


def replay_control_metrics(rows, previous):
    frozen = {(r['trial'], r['site'], r['endpoint'], r['subset']):r['metric']
              for r in previous['rows']}
    if len(frozen) != len(previous['rows']) or len(rows) != len(frozen):
        raise ValueError('Every frozen endpoint/locality/subset must be retained exactly once')
    seen, checks = set(), 0
    for row in rows:
        key = row['trial'], row['site'], row['endpoint'], row['subset']
        if key not in frozen or key in seen:
            raise ValueError('Changed or duplicated comparator population')
        seen.add(key)
        for contrast, old_prefix in (('metric', 'new'), ('original_ADE_contrast', 'old')):
            for suffix in ('gain_CV', 'gain_selected'):
                actual, expected = row[contrast]['old_'+suffix], frozen[key][old_prefix+'_'+suffix]
                if actual is None or expected is None:
                    if actual != expected:
                        raise ValueError('Comparator support changed')
                else:
                    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                checks += 1
    return checks


def budget_counts(records):
    groups = {}
    for row in records:
        key = '/'.join((row['arm'], row['subset'], row['reference']))
        group = groups.setdefault(key, dict(required_views=0, known_pass=0, known_fail=0,
                    not_estimable=0, net_degradation_gt2=0, views_with_unknown_labels=0))
        group['required_views'] += 1
        ratio, flag = row['known_positive_harm_percent'], row['known_budget_pass']
        if ratio is None:
            if flag is not None or row['support_status'] != 'not_estimable':
                raise ValueError('Empty risk must stay not_estimable')
            group['not_estimable'] += 1
        else:
            if not np.isfinite(ratio) or ratio < 0 or flag != (ratio <= 2.):
                raise ValueError('Unchanged 2% positive-harm rule required')
            group['known_pass' if flag else 'known_fail'] += 1
        net = row['known_net_degradation_percent']
        group['net_degradation_gt2'] += net is not None and net > 2.
        group['views_with_unknown_labels'] += row['unknown_rows'] > 0
        if row['complete_population_budget_claim']:
            raise ValueError('Candidate forecast diagnostics are not full-policy certification')
    for group in groups.values():
        if group['required_views'] != group['known_pass']+group['known_fail']+group['not_estimable']:
            raise ValueError('Required view omitted')
    return groups
