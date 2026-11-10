"""Remove equally many labeled steps from fully observed vehicle-pair queries."""

import numpy as np

HIDDEN_STEPS = (3, 6)
RANDOM_REPLICATES = 32


def pair_step_terms(predictions, target, valid, counts, comparisons, *, block=512):
    if (target.ndim != 3 or target.shape[1:] != (12, 2)
            or valid.shape != target.shape[:2] or valid.dtype != bool
            or counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < 1) or counts.sum() != len(target) or block < 1
            or not np.isfinite(target[valid]).all()
            or any(p.shape != target.shape or not np.isfinite(p).all() for p in predictions.values())):
        raise ValueError('Expected the common twelve-step query roster')
    offsets = np.r_[0, np.cumsum(counts)]
    incomplete = np.add.reduceat((~valid).any(1), offsets[:-1])
    q = np.flatnonzero((counts >= 2) & (incomplete == 0))
    if not len(q):
        raise ValueError('No fully labeled pair query')
    gains = {k: np.zeros((len(q), 12)) for k in comparisons}
    distances = {k: np.zeros_like(v) for k, v in gains.items()}
    for start in range(0, len(q), block):
        batch = q[start:start+block]
        i, j = np.concatenate([np.triu_indices(counts[x], k=1)+offsets[x] for x in batch], axis=1)
        pairs = counts[batch]*(counts[batch]-1)//2
        edges = np.r_[0, np.cumsum(pairs)][:-1]
        z = target[i].astype(float)-target[j]
        paths = {k: v[i].astype(float)-v[j] for k, v in predictions.items()}
        err = {k: np.linalg.norm(v-z, axis=-1) for k, v in paths.items()}
        span = slice(start, start+len(batch))
        for k, (a, b) in comparisons.items():
            gains[k][span] = np.add.reduceat(err[b]-err[a], edges, axis=0)/pairs[:, None]
            debit = np.linalg.norm(paths[a]-paths[b], axis=-1)
            distances[k][span] = np.add.reduceat(debit, edges, axis=0)/pairs[:, None]
    return gains, distances, dict(queries=len(q), pair_steps=int((counts[q]*(counts[q]-1)*6).sum()))


def hide_plan(queries, n, seed):
    if not 0 < n < 12 or queries < 1:
        raise ValueError('Expected at least one observed and one hidden step')
    steps = np.arange(12)
    head = np.broadcast_to(steps < n, (queries, 12))
    tail = np.broadcast_to(steps >= 12-n, (queries, 12))
    rng = np.random.default_rng([seed, n])
    uniform = rng.random((RANDOM_REPLICATES, queries, 12)).argsort(axis=-1).argsort(axis=-1) < n
    return head, tail, uniform


def masked(gain, distance, hidden):
    n = hidden.sum(-1)
    if (gain.shape != distance.shape or hidden.shape[-2:] != gain.shape or hidden.dtype != bool
            or not np.all(n == n.flat[0]) or not 0 < n.flat[0] < gain.shape[-1]):
        raise ValueError('Masking rates must match across all queries and replicates')
    truth = float(gain.mean())
    observed = np.where(hidden, 0., gain).sum(-1)
    contribution = observed.mean(-1)/gain.shape[-1]
    debit = np.where(hidden, distance, 0.).mean(axis=(-2, -1))
    lower, upper = contribution-debit, contribution+debit
    if np.any(lower > truth+1e-10) or np.any(upper < truth-1e-10):
        raise ValueError('Complete-grid result fell outside its masking bound')
    conditional = observed.mean(-1)/(gain.shape[-1]-n.flat[0])
    return dict(observed=conditional, bias=conditional-truth, lower=lower, upper=upper)


def evaluate(gain, distance, seed):
    out = {k: dict(full=float(x.mean()), masked={}) for k, x in gain.items()}
    for n in HIDDEN_STEPS:
        head, tail, uniform = hide_plan(len(next(iter(gain.values()))), n, seed)
        for k, x in gain.items():
            out[k]['masked'][str(n)] = {
                name: {m: float(v) for m, v in masked(x, distance[k], mask).items()}
                for name, mask in (('first', head), ('last', tail))
            }
            out[k]['masked'][str(n)]['uniform'] = {m: v.tolist()
                for m, v in masked(x, distance[k], uniform).items()}
    return out
