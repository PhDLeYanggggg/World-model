"""Causal recency adaptation using forecast errors wholly inside the history."""
import numpy as np

WINDOWS = (2, 4, 6)
ORIGINS = (3, 4, 5, 6)
PRIOR = (.25, .50, .25)
TEMPERATURE = .5
ARMS = ('fixed_prefix', 'query_prefix', 'agent_prefix')


def _velocity(h, k):
    x = np.arange(k, dtype=float)-(k-1)/2.
    return np.einsum('nkd,k->nd', h[:, -k:]-h[:, -1:], x)/float(x@x)


def _history(history):
    h = np.asarray(history, float)
    if h.ndim != 3 or h.shape[1:] != (8, 2) or not np.isfinite(h).all() or not len(h):
        raise ValueError('A nonempty batch of exactly eight finite observed positions is required')
    return h


def prefix_weights(history):
    h = _history(history)
    errors, mass = np.zeros((len(h), len(WINDOWS))), 0.
    for t in ORIGINS:
        for lead in (1, 2):
            if t+lead >= h.shape[1]:
                continue
            weight = (1. if lead == 1 else .5)*.8**(6-t)
            # Compare velocity errors at a common scale, independent of lead.
            actual = (h[:, t+lead]-h[:, t])/lead
            for j, k in enumerate(WINDOWS):
                velocity = _velocity(h[:, :t+1], min(k, t+1))
                errors[:, j] += weight*np.linalg.norm(actual-velocity, axis=1)
            mass += weight
    scale = np.maximum(.05, np.sqrt(np.square(np.diff(h, axis=1)).sum(2).mean(1)))
    logits = np.log(PRIOR)-errors/(mass*TEMPERATURE*scale[:, None])
    weight = np.exp(logits-logits.max(1, keepdims=True))
    return weight/weight.sum(1, keepdims=True)


def candidates(history, query):
    h = _history(history)
    q = np.asarray(query)
    if q.shape != (len(h),):
        raise ValueError('Every past row must have one aligned query')
    a = prefix_weights(h)
    _, inv, size = np.unique(q, return_inverse=True, return_counts=True)
    uniform = np.stack([np.bincount(inv, weights=v)/size for v in a.T], axis=1)[inv]
    v = np.stack([_velocity(h, k) for k in WINDOWS], axis=1)
    t = np.arange(1, 13)[None, :, None]
    weights = (np.broadcast_to(PRIOR, a.shape), uniform, a)
    return {k: t*np.einsum('nm,nmd->nd', w, v)[:, None] for k, w in zip(ARMS, weights, strict=True)}
