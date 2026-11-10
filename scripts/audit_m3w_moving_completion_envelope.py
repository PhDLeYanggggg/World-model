"""Separate displacement and missing-label limits on frozen Moving development."""

from collections import Counter, defaultdict
from fractions import Fraction
import json
from math import fsum, isclose, isfinite
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import package_m3w_moving_peak_certificate as prior

np, verify = prior.np, prior.verify
OUT = prior.OUT.parent/'moving_completion_envelope_v1'
KINDS = ('unfavorable_completion', 'observed_steps_only', 'favorable_completion')


def segments(error, valid):
    """Slopes of sum(min(radius, known_error)) plus missing_count * radius."""
    if error.shape != valid.shape or error.ndim != 2 or error.shape[1] != verify.HORIZON or not len(error):
        raise ValueError('Expected a nonempty query on the frozen prediction grid')
    if not np.isfinite(error).all() or np.any(error < 0) or np.any(error[~valid] != 0.):
        raise ValueError('Invalid observed error')
    values = np.sort(error, axis=1)
    increments = np.diff(values, axis=1, prepend=0.)
    missing = (~valid).sum(1)
    slope = np.arange(verify.HORIZON, 0, -1)[None, :] + missing[:, None]
    widths = np.bincount(slope[increments > 0.], weights=increments[increments > 0.], minlength=13)
    if len(widths) != 13:
        raise ValueError('An observed segment overlaps the missing-step ray')
    return widths[1:].tolist(), int(missing.max())


def aggregate(cv, ols, target, valid, query, radii):
    base = prior.aggregate(cv, ols, target, valid, query, radii)
    ids, group, sizes = np.unique(query, return_inverse=True, return_counts=True)
    target = np.where(valid[..., None], target, 0.)
    error = np.where(valid, np.linalg.norm(cv-target, axis=-1), 0.)
    queries = []
    for i, n in enumerate(sizes):
        at = group == i
        lengths, ray = segments(error[at], valid[at])
        queries.append(dict(actors=int(n), budget=float(radii[i]), favorable_lengths=lengths,
                            favorable_ray=ray))
    if len(ids) != len(base['queries']):
        raise ValueError('Query grouping differs from the archived bound')
    base['completion_queries'] = queries
    return base


def favorable(queries, budget):
    # Baseline dual price is the strongest unbounded missing-step slope.
    if not queries or not isfinite(budget) or budget < 0.:
        raise ValueError('Invalid completion budget')
    ray = max(Fraction(q['favorable_ray'], q['actors']) for q in queries)
    items = defaultdict(list)
    for q in queries:
        for i, width in enumerate(q['favorable_lengths'], 1):
            slope = Fraction(i, q['actors']) - ray
            if slope > 0 and width:
                items[slope].append(width)
    increment = verify.envelope([(float(s), fsum(w)) for s, w in sorted(items.items(), reverse=True)], budget)
    return float(ray)*budget + increment


def bounds(row, old):
    queries, completion = row['queries'], row['completion_queries']
    budget = fsum(q['budget'] for q in queries)
    if sorted((q['actors'], q['budget']) for q in queries) != sorted(
            (q['actors'], q['budget']) for q in completion):
        raise ValueError('The completion analysis must use the same budgets')
    ref, known, d = (row[k] for k in ('cv_error_mass', 'known_ols4_advantage', 'missing_disagreement'))
    observed = dict(query=fsum(verify.envelope(verify.slopes([q], 'available'), q['budget']) for q in queries),
                    record=verify.envelope(verify.slopes(queries, 'available'), budget))
    upper = dict(query=fsum(favorable([q], q['budget']) for q in completion),
                 record=favorable(completion, budget))
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


def summarize(views, results):
    limit = verify.POSITIVE_PP
    report = {}
    for scope in ('query', 'record'):
        rows = [r[scope] for r in results]
        summary = {}
        for k in KINDS:
            excluded = [r[k] < -limit for r in rows]
            remaining = [v for v, no in zip(views, excluded, strict=True) if not no]
            summary[k] = dict(excluded_views=sum(excluded),
                              all_seeds_excluded_recordings=24-len({v['recording'] for v in remaining}),
                              all_recordings_excluded_sources=9-len({v['source'] for v in remaining}),
                              median_upper_pp=median(r[k] for r in rows))
        split = Counter('negative_even_if_favorable' if r[KINDS[2]] < -limit else
                        'negative_under_conservative_score_only' if r[KINDS[0]] < -limit else
                        'not_excluded_by_conservative_score' for r in rows)
        report[scope] = dict(**summary, classification=dict(split))
    return report


def run():
    payload, expected, hashes = prior.export(aggregate_view=aggregate)
    old = verify.replay(payload)
    prior.check(old, expected)
    rows = payload['views']
    result = [bounds(v, p) for v, p in zip(rows, old['per_view'], strict=True)]
    by_source = []
    for i in sorted({v['source'] for v in rows}):
        group = [(v, r['record']) for v, r in zip(rows, result, strict=True) if v['source'] == i]
        by_source.append(dict(recordings=len({v['recording'] for v, _ in group}), views=len(group),
                              excluded_views={k: sum(r[k] < -verify.POSITIVE_PP for _, r in group) for k in KINDS}))
    by_source.sort(key=lambda row: (row['recordings'], tuple(row['excluded_views'].values())))
    return dict(views=old['views'], sources=old['sources'], recordings=old['recordings'], seeds=list(verify.SEEDS),
                completion_envelope=summarize(rows, result), per_source_record_pooled=by_source,
                archive_input_sha256=hashes, result_source='fresh_post_hoc_bound_on_frozen_exposed_development',
                original_2pct_unchanged=True, independent_confirmation=False, external_outcomes_read=False,
                learned_policy=False, risk_evaluated=False, submission_ready=False)


if __name__ == '__main__':
    prior.audit.torch.set_num_threads(4)
    prior.audit.torch.set_num_interop_threads(1)
    result = prior.old.stable(run())
    OUT.mkdir(parents=True, exist_ok=True)
    prior.old.once(OUT/'receipt.json', result)
    print(json.dumps(result, indent=2))
