"""Reconstruct the frozen external decision table from published scalars only."""

from collections import Counter
from hashlib import sha256
from itertools import combinations_with_replacement
import json
from math import factorial, fsum, isfinite
from pathlib import Path

ARMS = ('cv', 'joint', 'ols4', 'spline', 'fixed_joint', 'fixed_independent', 'fixed_uniform', 'fixed_no_pair')
FIXED = ARMS[4:]
CONTRASTS = ('independent', 'uniform', 'no_pair', *FIXED[1:])
LIMIT = .02
SEEDS = [17, 29, 43]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def supported(ratios):
    require(len(ratios) == 3 and all(v is None or isfinite(v) for v in ratios), 'Invalid risk values')
    require(all(v is None or v >= 0. for v in ratios[:2]), 'Positive-harm bound cannot be negative')
    return all(v is not None and v <= LIMIT for v in ratios)


def mean(x):
    return fsum(x)/len(x)


def source_means(groups, value):
    cells = [[value(view) for clip in group for view in clip] for group in groups]
    return [mean(cell) for cell in cells if cell and all(x is not None for x in cell)]


def quantile(values, p):
    a = sorted(values)
    index = (len(a)-1)*p
    lower = int(index)
    return a[lower]+(index-lower)*(a[min(lower+1, len(a)-1)]-a[lower])


def exact_interval(values):
    """Enumerate unordered draws with exact ordered-draw multiplicity."""
    n = len(values)
    require(0 < n <= 10, 'Exact enumeration restricted to at most ten sources')
    if len(set(values)) == 1:
        return [values[0], values[0]]
    facts = [factorial(i) for i in range(n+1)]
    mass = Counter()
    for pick in combinations_with_replacement(range(n), n):
        denominator = 1
        for count in Counter(pick).values():
            denominator *= facts[count]
        mass[mean([values[i] for i in pick])] += facts[n]//denominator
    total = n**n
    require(sum(mass.values()) == total, 'Exact bootstrap multiplicities are incomplete')
    cumulative, low, high = 0, None, None
    for value, count in sorted(mass.items()):
        cumulative += count
        if low is None and 40*cumulative >= total:
            low = value
        if 40*cumulative >= 39*total:
            high = value
            break
    return [low, high]


def interval(values, total, resamples, *, exact=False):
    n = len(values)
    out = dict(estimate=None, ci95=None, estimable_sources=n, total_sources=total)
    if n:
        pick = resamples[str(n)]
        require(len(pick) == 3000*n and all(type(i) is int and 0 <= i < n for i in pick),
                'Invalid bootstrap indices')
        draws = [mean([values[i] for i in pick[j:j+n]]) for j in range(0, len(pick), n)]
        out.update(estimate=mean(values), ci95=[quantile(draws, .025), quantile(draws, .975)])
    if exact:
        out['exact_ci95'] = exact_interval(values) if n else None
    return out


def joint_state(row, mass):
    all_upper, easy_upper, easy_net = row['risk']
    require(type(row['edits']) is int and type(row['unknown_edits']) is int and
            0 <= row['unknown_edits'] <= row['edits'], 'Invalid edit counts')
    risk = supported(row['risk'])
    require(not risk or row['edits'] > 0, 'A zero-edit view cannot pass the risk rule')
    require(mass >= 0. and isfinite(mass), 'Invalid common CV error mass')
    cv, ols = row['lower_vs_cv'], row['lower_vs_ols4']
    dual = risk and mass > 0 and cv is not None and ols is not None and min(cv, ols) > 0
    reason = ('no_intervention' if not row['edits'] else 'undefined_easy_reference' if easy_upper is None else
              'risk_pass' if risk else 'violates_original_risk')
    return risk, dual, reason


def replay(evidence, resamples):
    groups = evidence['groups']
    views = [v for g in groups for clip in g for v in clip]
    require(len(groups) == 13 and sum(map(len, groups)) == 15 and len(views) == 45,
            'Wrong source or recording roster')
    for group in groups:
        for clip in group:
            require([v['seed'] for v in clip] == SEEDS, 'Wrong seed roster')
    for v in views:
        require(set(v['arms']) == set(ARMS) and set(v['paired']) == set(CONTRASTS), 'Wrong arm roster')
        require(len({v['arms'][k]['edits'] for k in FIXED}) == 1, 'Fixed-count controls differ in edit count')
    arms = {}
    for arm in ARMS:
        cells = [v['arms'][arm] for v in views]
        states = [joint_state(v, a['cv_mass']) for v, a in zip(cells, views, strict=True)]
        fields = ('ade_px', 'fde_px', 'lower_vs_cv', 'lower_vs_ols4', 'p95_ade_px', 'edit_rate')
        arms[arm] = {key: interval(source_means(groups, lambda v: v['arms'][arm][key]), len(groups), resamples)
                     for key in fields}
        arms[arm].update(edits=sum(a['edits'] for a in cells), risk_passes=sum(s[0] for s in states),
                         dual_passes=sum(s[1] for s in states), reasons=dict(Counter(s[2] for s in states)),
                         undefined_easy=sum(a['risk'][1] is None for a in cells),
                         unknown_edits=sum(a['unknown_edits'] for a in cells),
                         positive_ols4=sum(a['lower_vs_ols4'] is not None and a['lower_vs_ols4'] > 0 for a in cells))
    matched = {}
    for arm in CONTRASTS:
        for v in views:
            known, lower = v['paired'][arm]
            require((known is None) == (lower is None), 'Paired missingness must agree')
            require(lower is None or (isfinite(lower) and isfinite(known) and lower <= known + 1e-12),
                    'A missing-step debit cannot improve the conservative contrast')
        vals = source_means(groups, lambda v: v['paired'][arm][1])
        matched[arm] = interval(vals, len(groups), resamples, exact=True)
    return dict(sources=len(groups), views=len(views), arms=arms, matched=matched)


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/'manifest.json').read_text())
    for name in ('verify.py', 'evidence.json', 'resamples.json', 'README.md'):
        require(sha256((root/name).read_bytes()).hexdigest() == manifest['members'][name], f'Changed file: {name}')
    result = replay(json.loads((root/'evidence.json').read_text()), json.loads((root/'resamples.json').read_text()))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
