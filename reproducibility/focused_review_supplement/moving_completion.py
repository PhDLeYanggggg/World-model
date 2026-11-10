"""Replay all three Moving ceilings using grouped error slopes."""

from collections import Counter, defaultdict
from fractions import Fraction
from math import fsum, isclose, isfinite
from statistics import median


KINDS = ('unfavorable_completion', 'observed_steps_only', 'favorable_completion')


def favorable(queries, budget, peak):
    if not queries or not isfinite(budget) or budget < 0.:
        raise ValueError('Invalid completion budget')
    for q in queries:
        if (set(q) != {'actors', 'budget', 'favorable_lengths', 'favorable_ray'}
                or type(q['actors']) is not int or q['actors'] < 1
                or not isfinite(q['budget']) or q['budget'] < 0.
                or type(q['favorable_ray']) is not int or not 0 <= q['favorable_ray'] <= peak.HORIZON
                or len(q['favorable_lengths']) != peak.HORIZON
                or any(not isfinite(w) or w < 0. for w in q['favorable_lengths'])):
            raise ValueError('Invalid aggregate completion slopes')
    # Marginal gain below the strongest unbounded slope cannot improve the allocation.
    ray = max(Fraction(q['favorable_ray'], q['actors']) for q in queries)
    items = defaultdict(list)
    for q in queries:
        for i, width in enumerate(q['favorable_lengths'], 1):
            slope = Fraction(i, q['actors']) - ray
            if slope > 0 and width:
                items[slope].append(width)
    increment = peak.envelope([(float(s), fsum(w)) for s, w in sorted(items.items(), reverse=True)], budget)
    return float(ray)*budget + increment


def bounds(row, old, peak):
    queries, completion = row['queries'], row['completion_queries']
    budget = fsum(q['budget'] for q in queries)
    if sorted((q['actors'], q['budget']) for q in queries) != sorted(
            (q['actors'], q['budget']) for q in completion):
        raise ValueError('The completion analysis must use the same budgets')
    ref, known, d = (row[k] for k in ('cv_error_mass', 'known_ols4_advantage', 'missing_disagreement'))
    observed = dict(query=fsum(peak.envelope(peak.slopes([q], 'available'), q['budget']) for q in queries),
                    record=peak.envelope(peak.slopes(queries, 'available'), budget))
    upper = dict(query=fsum(favorable([q], q['budget'], peak) for q in completion),
                 record=favorable(completion, budget, peak))
    result = {}
    for scope in ('query', 'record'):
        low, mid, high = old['ols4'][scope], 100*(observed[scope]-known)/ref, 100*(upper[scope]-known+d)/ref
        if not low <= mid + 1e-8 or not mid <= high + 1e-8:
            raise ValueError('Inconsistent nested completion limits')
        if not d and not any(q['favorable_ray'] for q in completion) and not isclose(low, high, abs_tol=1e-8):
            raise ValueError('A complete-label view must have a unique ceiling')
        result[scope] = dict(zip(KINDS, (low, mid, high), strict=True))
    if any(result['record'][k] < result['query'][k] - 1e-8 for k in KINDS):
        raise ValueError('A pooled budget cannot hurt the hindsight ceiling')
    return result


def summarize(views, results, limit):
    n_records = len({v['recording'] for v in views})
    n_sources = len({v['source'] for v in views})
    report = {}
    for scope in ('query', 'record'):
        rows = [r[scope] for r in results]
        summary = {}
        for k in KINDS:
            excluded = [r[k] < -limit for r in rows]
            remaining = [v for v, no in zip(views, excluded, strict=True) if not no]
            summary[k] = dict(excluded_views=sum(excluded),
                              all_seeds_excluded_recordings=n_records-len({v['recording'] for v in remaining}),
                              all_recordings_excluded_sources=n_sources-len({v['source'] for v in remaining}),
                              median_upper_pp=median(r[k] for r in rows))
        split = Counter('negative_even_if_favorable' if r[KINDS[2]] < -limit else
                        'negative_under_conservative_score_only' if r[KINDS[0]] < -limit else
                        'not_excluded_by_conservative_score' for r in rows)
        report[scope] = dict(**summary, classification=dict(split))
    return report


def report(rows, old, peak):
    result = [bounds(v, p, peak) for v, p in zip(rows, old['per_view'], strict=True)]
    by_source = []
    for i in sorted({v['source'] for v in rows}):
        group = [(v, r['record']) for v, r in zip(rows, result, strict=True) if v['source'] == i]
        by_source.append(dict(recordings=len({v['recording'] for v, _ in group}), views=len(group),
                              excluded_views={k: sum(r[k] < -peak.POSITIVE_PP for _, r in group) for k in KINDS}))
    by_source.sort(key=lambda row: (row['recordings'], tuple(row['excluded_views'].values())))
    return dict(views=old['views'], sources=old['sources'], recordings=old['recordings'], seeds=list(peak.SEEDS),
                completion_envelope=summarize(rows, result, peak.POSITIVE_PP), per_source_record_pooled=by_source)


def attach(base, augmentation):
    keys = ('recording', 'source', 'seed')
    rows = []
    for row, extension in zip(base['views'], augmentation['views'], strict=True):
        if (set(extension) != {*keys, 'completion_queries'}
                or any(row[k] != extension[k] for k in keys)):
            raise ValueError('Completion views do not align with the original certificate')
        rows.append(dict(**row, completion_queries=extension['completion_queries']))
    return rows
