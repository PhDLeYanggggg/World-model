"""Verify the arbitrary-path OLS4 exclusion from aggregate query lengths."""

from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
import json
from math import fsum, isclose, isfinite
from pathlib import Path
from statistics import median


HORIZON = 12
SEEDS = (17, 29, 43)
POSITIVE_PP = 1e-6


def slopes(queries, kind):
    bins = defaultdict(list)
    for q in queries:
        n = q['actors']
        lengths = q[kind]
        if (type(n) is not int or n <= 0 or len(lengths) != HORIZON
                or any(not isfinite(x) or x < 0 for x in lengths)):
            raise ValueError('Invalid aggregate query slopes')
        for k, x in zip(range(HORIZON, 0, -1), lengths, strict=True):
            if x:
                bins[Fraction(k, n)].append(x)
    return [(float(s), fsum(z)) for s, z in sorted(bins.items(), reverse=True)]


def envelope(items, budget):
    """A feasible fractional allocation and a matching weak-duality upper bound."""
    if not isfinite(budget) or budget < 0:
        raise ValueError('Invalid recorded peak budget')
    remaining, multiplier, terms = budget, 0., []
    for slope, length in items:
        spend = min(remaining, length)
        terms.append(slope*spend)
        remaining -= spend
        if spend < length:
            multiplier = slope
            break
    primal = fsum(terms)
    dual = fsum([multiplier*budget, *(max(s-multiplier, 0.)*z for s, z in items)])
    if not isclose(primal, dual, rel_tol=1e-11, abs_tol=1e-10):
        raise ValueError('Relaxed allocation and dual certificate disagree')
    return dual


def minimum_budget(items, target):
    if target <= 0.:
        return 0.
    spent = []
    remaining = target
    for slope, length in items:
        if remaining <= slope*length:
            return fsum([*spent, remaining/slope])
        spent.append(length)
        remaining -= slope*length
    return None


def evaluate(row):
    queries = row['queries']
    budget = fsum(q['budget'] for q in queries)
    ref, known, missing = (row[k] for k in ('cv_error_mass', 'known_ols4_advantage', 'missing_disagreement'))
    if not all(isfinite(x) for x in (ref, known, missing)) or ref <= 0. or missing < 0. or not queries:
        raise ValueError('Invalid reference support')
    gap = known+missing
    all_ = {k: slopes(queries, k) for k in ('available', 'charged')}
    result = {}
    for key, kind, offset in (('cv', 'available', 0.), ('ols4', 'charged', -gap)):
        local = fsum(envelope(slopes([q], kind), q['budget']) for q in queries)
        pooled = envelope(all_[kind], budget)
        if pooled+1e-9 < local:
            raise ValueError('Pooling cannot shrink the feasible set')
        result[key] = {'query': 100*(local+offset)/ref, 'record': 100*(pooled+offset)/ref}
    needed = {k: minimum_budget(all_[k], z) for k, z in (('available', known), ('charged', gap))}
    result.update(peak_budget=budget, required_peak=needed,
                  required_multiple={k: x/budget if budget > 0 and x is not None else None for k, x in needed.items()})
    return result


def replay(data):
    rows, seen, fold = [], defaultdict(set), {}
    for row in data['views']:
        rec, src, seed = (row[k] for k in ('recording', 'source', 'seed'))
        if seed in seen[rec] or seed not in SEEDS or fold.setdefault(rec, src) != src:
            raise ValueError('Duplicate or cross-source recording')
        seen[rec].add(seed)
        rows.append(evaluate(row))
    if len(rows) != 72 or set(seen) != set(range(24)) or set(fold.values()) != set(range(9)) or any(
            v != set(SEEDS) for v in seen.values()):
        raise ValueError('The certificate requires all three seeds of the frozen 24 recordings')
    modes, multiples = {}, {}
    for scope in ('query', 'record'):
        points = [r['ols4'][scope] for r in rows]
        excluded = [x for x in points if x < -POSITIVE_PP]
        modes[scope] = dict(excluded=len(excluded), median_ols4_upper_pp=median(points),
                            closest_excluded_to_zero_pp=max(excluded, default=None))
    for kind in ('available', 'charged'):
        r = [x['required_multiple'][kind] for x in rows if x['required_multiple'][kind] is not None]
        multiples[kind] = dict(views=len(r), median=median(r), ties_within_budget=sum(x <= 1. for x in r))
    return dict(recordings=len(seen), sources=len(set(fold.values())), views=len(rows), upper_bounds=modes,
                break_even_multiples=multiples, zero_budget_views=Counter(x['peak_budget'] == 0. for x in rows)[True],
                per_view=rows)


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/'manifest.json').read_text())
    if set(manifest) != {'README.md', 'evidence.json', 'verify.py'} or any(
            sha256((root/name).read_bytes()).hexdigest() != h for name, h in manifest.items()):
        raise ValueError('Certificate manifest does not match')
    print(json.dumps(replay(json.loads((root/'evidence.json').read_text())), indent=2))


if __name__ == '__main__':
    main()
