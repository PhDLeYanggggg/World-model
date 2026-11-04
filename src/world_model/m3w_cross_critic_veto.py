"""A fixed second-critic risk veto and causal same-count selection control."""
import numpy as np
from src.world_model.m3w_multiaction_cost import select_actions

ARMS = ('adaptive_self', 'frozen_self', 'cross_veto', 'self_count_matched')


def choices(proposer, critic, allowed, groups, ids):
    a, b = np.asarray(proposer), np.asarray(critic)
    mask, ids = np.asarray(allowed), np.asarray(ids)
    n = len(a)
    if (a.shape != (n, 7, 5) or b.shape != a.shape or mask.shape != (n, 7)
            or mask.dtype != bool or ids.shape != (n,) or ids.dtype.kind not in 'iu'
            or len(np.unique(ids)) != n or not np.isfinite(a).all() or not np.isfinite(b).all()
            or (a < 0).any() or (b < 0).any()):
        raise ValueError('Aligned finite causal moments, eligibility and unique row IDs required')
    if not groups or any(np.asarray(g).dtype.kind not in 'iu' or not len(g) for g in groups):
        raise ValueError('Nonempty integer query groups required')
    covered = np.concatenate(groups)
    if not np.array_equal(np.sort(covered), np.arange(n)):
        raise ValueError('Every row must belong to exactly one causal query')
    own, utility = select_actions(a, mask, arm='physical_neural')
    other, _ = select_actions(b, mask, arm='physical_neural')
    veto = np.full(n, -1, np.int64)
    selected = np.flatnonzero(own >= 0)
    cost = b[selected, own[selected]]
    passed = ((cost[:, 2] > 0) & (cost[:, 3] > 0)
        & (cost[:, 1] <= .02*cost[:, 2]) & (cost[:, 4] <= .02*cost[:, 3]))
    kept = selected[passed]
    veto[kept] = own[kept]
    matched = np.full(n, -1, np.int64)
    for group in groups:
        k = int((veto[group] >= 0).sum())
        candidates = group[own[group] >= 0]
        order = np.lexsort((ids[candidates], -utility[candidates]))
        take = candidates[order[:k]]
        matched[take] = own[take]
        if k != int((matched[group] >= 0).sum()):
            raise ValueError('Match the intervention count in every query, including empty ones')
    return dict(adaptive_self=own, frozen_self=other, cross_veto=veto, self_count_matched=matched)


def rollout(paths, floor, choice):
    p, f, c = map(np.asarray, (paths, floor, choice))
    n = len(f)
    if (p.shape != (n, 7, 12, 2) or f.shape != (n, 12, 2) or c.shape != (n,)
            or c.dtype.kind not in 'iu' or (c < -1).any() or (c >= 7).any()
            or not np.isfinite(p).all() or not np.isfinite(f).all()):
        raise ValueError('Fixed finite causal paths and aligned legal action indices required')
    result = f.copy()
    selected = np.flatnonzero(c >= 0)
    result[selected] = p[selected, c[selected]]
    return result
