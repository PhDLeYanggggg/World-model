"""Recover both missing-position endpoints from the original paired scalars."""

from math import fsum, isfinite


CONTROLS = ('fixed_independent', 'fixed_uniform', 'fixed_no_pair')


def mean(values):
    return fsum(values)/len(values)


def group_bounds(evidence, arm):
    rows = []
    for group in evidence['groups']:
        views = [v['paired'][arm] for clip in group for v in clip]
        if not views:
            raise ValueError('Empty original source group')
        for known, low in views:
            if ((known is None) != (low is None) or (known is not None and (
                    not (isfinite(known) and isfinite(low)) or low > known+1e-12))):
                raise ValueError('Invalid archived contrast or missing-label debit')
        if any(k is None for k, _ in views):
            continue
        known, low = (mean([v[i] for v in views]) for i in (0, 1))
        rows.append((low, known, 2*known-low))
    return rows


def replay(evidence, reference, exact_interval):
    groups = evidence['groups']
    if len(groups) != 13 or sum(map(len, groups)) != 15 or any(
            [v['seed'] for v in c] != [17, 29, 43] for g in groups for c in g):
        raise ValueError('Unexpected WEPDTOF evaluation roster')
    if reference['views'] != 45 or reference['arms']['fixed_joint']['dual_passes'] != 0:
        raise ValueError('The original risk and utility decision did not reproduce')
    contrasts = {}
    for arm in CONTROLS:
        rows = group_bounds(evidence, arm)
        if len(rows) != 10:
            raise ValueError('The original matched-source support changed')
        lower, observed, upper = map(list, zip(*rows, strict=True))
        baseline = reference['matched'][arm]
        if (abs(mean(lower)-baseline['estimate']) > 1e-12 or baseline['estimable_sources'] != len(rows)):
            raise ValueError('The original lower estimate did not reproduce')
        contrasts[arm] = dict(identification_envelope_pp=[mean(lower), mean(upper)],
                              observed_pp=mean(observed),
                              empirical_bootstrap_outer_pp=[baseline['exact_ci95'][0], exact_interval(upper)[1]])
    return dict(views=45, clips=15, sources=13, matched_estimable_sources=10,
                original_fixed_joint_dual_passes=0, contrasts=contrasts)
