"""Deterministic, masked relative-trajectory evaluation on a common query roster."""

import numpy as np


def evaluate(predictions, target, valid, counts, comparisons, *, block=512):
    """Compare unordered within-query pairs, giving each supported query equal weight."""
    if (target.ndim != 3 or target.shape[1:] != (12, 2) or valid.shape != target.shape[:2]
            or valid.dtype != bool or not np.isfinite(target[valid]).all()
            or counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < 1) or counts.sum() != len(target) or block < 1
            or any(p.shape != target.shape or not np.isfinite(p).all() for p in predictions.values())
            or any(a not in predictions or b not in predictions or a == b for a, b in comparisons.values())):
        raise ValueError('Expected aligned twelve-step trajectories, masks and ordered query counts')
    offsets = np.r_[0, np.cumsum(counts)]
    eligible = np.flatnonzero(counts >= 2)
    if not len(eligible):
        raise ValueError('No pairs in this cohort')
    n = len(eligible)
    pair_steps = counts[eligible]*(counts[eligible]-1)*6
    known = np.zeros(n, dtype=int)
    endpoints = np.zeros(n, dtype=int)
    observed = {a: np.zeros((n, 2)) for a in predictions}
    missing = {k: np.zeros(n) for k in comparisons}
    for start in range(0, n, block):
        q = eligible[start:start+block]
        span = slice(start, start+len(q))
        idx = [np.triu_indices(counts[j], k=1) + offsets[j] for j in q]
        i, j = np.concatenate(idx, axis=1)
        group = np.repeat(np.arange(len(q)), [v.shape[1] for v in idx])

        def sum_query(v):
            return np.bincount(group, weights=v, minlength=len(q))

        ok = valid[i] & valid[j]
        known[span] = sum_query(ok.sum(1))
        endpoints[span] = sum_query(ok[:, -1])
        truth = np.where(ok[..., None], target[i].astype(float)-target[j], 0.)
        rel = {a: p[i].astype(float)-p[j] for a, p in predictions.items()}
        for a, z in rel.items():
            error = np.where(ok, np.linalg.norm(z-truth, axis=-1), 0.)
            observed[a][span, 0] = sum_query(error.sum(1))
            observed[a][span, 1] = sum_query(error[:, -1])
        for k, (a, b) in comparisons.items():
            distance = np.linalg.norm(rel[a]-rel[b], axis=-1)
            missing[k][span] = sum_query(np.where(ok, 0., distance).sum(1))

    def observed_query_mean(error, denom):
        have = denom > 0
        return error[have]/denom[have]

    def mean(x):
        return float(x.mean()) if len(x) else None

    def absolute(v):
        ade = observed_query_mean(v[:, 0], known)
        return dict(relative_ade=mean(ade), relative_fde=mean(observed_query_mean(v[:, 1], endpoints)),
                    query_relative_ade_p95=float(np.quantile(ade, .95)) if len(ade) else None)

    contrasts = {}
    for k, (a, b) in comparisons.items():
        gain = observed[b][:, 0]-observed[a][:, 0]
        debit = missing[k]
        contrasts[k] = dict(observed=mean(observed_query_mean(gain, known)),
                            observed_contribution=mean(gain/pair_steps),
                            lower=mean((gain-debit)/pair_steps), upper=mean((gain+debit)/pair_steps))
    return dict(support=dict(rows=len(target), queries=len(counts), pair_queries=n,
                             observed_queries=int((known > 0).sum()), endpoint_queries=int((endpoints > 0).sum()),
                             complete_pair_queries=int((known == pair_steps).sum()),
                             pair_steps=int(pair_steps.sum()), observed_pair_steps=int(known.sum())),
                absolute={a: absolute(v) for a, v in observed.items()}, contrasts=contrasts)
