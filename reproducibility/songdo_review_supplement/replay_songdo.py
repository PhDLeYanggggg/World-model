"""Recompute Songdo calibration and confirmation from anonymous scalars."""

from hashlib import sha256
import json
from math import fsum, isfinite
from pathlib import Path


ARMS = ('independent', 'additive', 'same_family', 'joint')
SEEDS = [17, 29, 43]
PAIR = ('known_pp', 'lower_pp', 'relative_available_native', 'relative_lower_native')
ABSOLUTE = ('available_ade_native', 'available_fde_native', 'available_ade_p95_native')
RISK_LIMIT = .02


def mean(v):
    return fsum(v)/len(v)


def ratio_state(r):
    if len(r) != 3 or any(x is not None and not isfinite(x) for x in r) or any(
            x is not None and x < 0 for x in r[:2]):
        raise ValueError('Invalid original risk ratios')
    return all(x is not None and x <= RISK_LIMIT for x in r)


def released(score, threshold):
    return threshold is not None and score >= threshold


def risk_counts(cells, threshold_by_seed, arm):
    out = dict(edited=0, original_risk_pass=0, dual_pass=0, undefined_risk=0,
               measured_risk_violation=0, no_edit_not_certified=0)
    for cell in cells:
        a = cell['arms'][arm]
        if threshold_by_seed is not None and not released(a['score'], threshold_by_seed[str(cell['seed'])][arm]):
            out['no_edit_not_certified'] += 1
            continue
        r = a['risk']
        ok = ratio_state(r)
        gains = a['gains']
        if len(gains) != 2 or any(x is not None and not isfinite(x) for x in gains):
            raise ValueError('Invalid paired conservative gains')
        out['edited'] += 1
        out['original_risk_pass'] += ok
        out['undefined_risk'] += any(v is None for v in r)
        out['measured_risk_violation'] += any(v is not None and v > RISK_LIMIT for v in r)
        out['dual_pass'] += ok and all(x is not None and x > 0. for x in gains)
    return out


def calibration(groups):
    views = [v for g in groups for f in g for v in f if v['arms'] is not None]
    thresholds, counts = {}, {}
    for seed in SEEDS:
        t, c = {}, {}
        for arm in ARMS:
            cells = [v['arms'][arm] for v in views if v['seed'] == seed]
            boundary = max((v['score'] for v in cells if not ratio_state(v['risk'])), default=-float('inf'))
            passing = [v['score'] for v in cells if v['score'] > boundary]
            t[arm] = min(passing, default=None)
            c[arm] = dict(eligible=len(cells), released=len(passing))
        thresholds[str(seed)], counts[str(seed)] = t, c
    return thresholds, counts


def interval(groups, getter, draws):
    points, support = [], 0
    for g in groups:
        flights = []
        for f in g:
            values = [getter(v) for v in f]
            if any(x is not None and not isfinite(x) for x in values):
                raise ValueError('Nonfinite score')
            support += sum(x is not None for x in values)
            if all(x is not None for x in values):
                flights.append(mean(values))
        points.append(mean(flights) if flights else None)
    if any(p is None for p in points):
        return dict(point=None, ci95=None, estimable_components=sum(p is not None for p in points),
                    defined_seed_views=support)
    vals = sorted(mean([points[i] for i in sample]) for sample in draws)

    def quantile(q):
        x = (len(vals)-1)*q
        a = int(x)
        return vals[a]+(x-a)*(vals[min(a+1, len(vals)-1)]-vals[a])

    return dict(point=mean(points), ci95=[quantile(.025), quantile(.975)],
                estimable_components=len(points), defined_seed_views=support)


def replay(evidence):
    if evidence['original_risk_limit'] != RISK_LIMIT:
        raise ValueError('The original risk limit changed')
    fit, held = (evidence[k] for k in ('calibration', 'confirmation'))
    if [(len(g), sum(map(len, g))) for g in (fit, held)] != [(3, 40), (14, 160)]:
        raise ValueError('Expected all calibration and confirmation flights in linked components')
    for g in (*fit, *held):
        for f in g:
            if [r['seed'] for r in f] != SEEDS or any(v['arms'] is not None and (
                    set(v['arms']) != set(ARMS) or any(not isfinite(a['score']) for a in v['arms'].values())) for v in f):
                raise ValueError('Changed arm, score or seed roster')
    cells = [v for g in held for f in g for v in f]
    if any(c['arms'] is None for c in cells):
        raise ValueError('Original confirmation has matched proposals in all views')
    draws = evidence['source_resamples']
    if len(draws) != 3000 or any(len(s) != 14 or any(type(i) is not int or i not in range(14) for i in s) for s in draws):
        raise ValueError('Expected the original component resampling design')
    threshold, counts = calibration(fit)
    if threshold != evidence['calibration_thresholds']:
        raise ValueError('Replayed thresholds differ from the pre-confirmation freeze')
    contrasts = {a: {k: interval(held, lambda v: v['contrasts'][a][k], draws) for k in PAIR}
                 for a in ARMS[:-1]}
    matched = {a: {k: interval(held, lambda v: v['accuracy'][a][k], draws) for k in ABSOLUTE}
               for a in ('cv', 'ols4', *ARMS)}
    actual = {a: {k: interval(held, lambda v: v['accuracy'][a if released(
        v['arms'][a]['score'], threshold[str(v['seed'])][a]) else 'cv'][k], draws) for k in ABSOLUTE} for a in ARMS}
    return dict(flights=160, source_components=14, views=len(cells), thresholds=threshold,
                calibration_counts=counts, contrasts=contrasts, matched=matched, actual=actual,
                matched_risk={a: risk_counts(cells, None, a) for a in ARMS},
                actual_risk={a: risk_counts(cells, threshold, a) for a in ARMS})


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/'manifest.json').read_text())
    for name in ('evidence.json', 'verify.py', 'README.md'):
        if sha256((root/name).read_bytes()).hexdigest() != manifest[name]:
            raise ValueError('Changed Songdo replay member')
    print(json.dumps(replay(json.loads((root/'evidence.json').read_text())), sort_keys=True, allow_nan=False))
