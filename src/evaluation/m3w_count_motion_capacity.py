"""Offline count allocation under causal credits and a cumulative movement cap."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, hstack, vstack

from src.evaluation.m3w_streaming_menu_capacity import POSITIVE_PP, respects_tokens, token_rows


def movement_prefix(cv, path, rows):
    """Sum of selected agents' peak Euclidean displacements from CV."""
    peak = np.linalg.norm(path - cv, axis=-1).max(axis=1)
    cost = np.zeros(rows.shape)
    present = rows >= 0
    cost[present] = peak[rows[present]]
    if not np.isfinite(cost).all() or (cost[present] <= 0).any():
        raise ValueError('A nominated edit must have finite positive movement')
    return np.column_stack((np.zeros(len(rows)), cost.cumsum(axis=1)))


def within_cap(cost, cap):
    return cost <= cap + 1e-9 * cap


def allocate(legal, movement, total, cap, *, score=None, terms=None, seconds=2.):
    """Exactly one objective: label-blind score or label-aware original maximin."""
    if (score is None) == (terms is None):
        raise ValueError('Provide exactly one objective')
    nq = len(legal)
    if (legal.shape != (nq, 3) or movement.shape != legal.shape or not legal[:, 0].all()
            or (legal[:, 2] & ~legal[:, 1]).any() or not np.isfinite(movement[legal]).all()
            or (movement[:, 0] != 0.).any() or (movement[:, 1:][legal[:, 1:]] <= 0.).any()
            or total < 0 or total > int(legal[:, 1].sum())//5 or not np.isfinite(cap)
            or cap < 0. or bool(total) != (cap > 0.)):
        raise ValueError('Invalid menu or recorded movement budget')
    if not total:
        return np.zeros(nq, dtype=int) if score is not None else None, 'zero_realized'

    q, n = np.nonzero(legal[:, 1:] & within_cap(movement[:, 1:], cap))
    n += 1
    k = len(n)
    if not k:
        return None, 'movement_infeasible'
    cols = np.arange(k)
    choice = coo_array((np.ones(k), (q, cols)), shape=(nq, k+1)).tocsc()
    by_query = coo_array((n.astype(float), (q, cols)), shape=choice.shape).tocsc()
    costs = movement[q, n] / cap
    mats = [choice, csc_array(np.r_[n, 0.][None]), csc_array(np.r_[costs, 0.][None])]
    lo = np.r_[np.full(nq, -np.inf), total, -np.inf]
    hi = np.r_[np.ones(nq), total, 1. + 1e-9]
    if score is not None:
        if score.shape != legal.shape or not np.isfinite(score[legal]).all():
            raise ValueError('Invalid frozen prefix scores')
        obj = np.r_[score[q, 0].astype(float) - score[q, n].astype(float), 0.]
        low, high = 0., 0.
    else:
        gain, risk, easy, offset, easy_cap, mass = terms
        support = easy[q, n]
        if not support.any() or mass <= 0.:
            return None, 'no_support'
        g, r = gain[q, n].T, risk[q, n].T
        scale = np.abs(r).max(1).clip(1.)
        mats += [csc_array(np.column_stack((r / scale[:, None], np.zeros(3)))),
                 csc_array(np.r_[support, 0.][None]),
                 csc_array(np.column_stack((-g / mass, np.ones(2))))]
        lo = np.r_[lo, np.full(3, -np.inf), 1., -np.inf, -np.inf]
        hi = np.r_[hi, 0., 0., easy_cap / scale[2], np.inf, offset / mass]
        obj, low, high = np.r_[np.zeros(k), -1.], -np.inf, np.inf
    mat = vstack(mats, format='csc')
    tail, available = token_rows(by_query, legal[:, 1])
    mat = vstack((hstack((mat, csc_array((mat.shape[0], nq)))), tail), format='csc')
    # HiGHS presolve excluded a feasible exact-boundary option beside near-ties.
    sol = milp(c=np.r_[obj, np.zeros(nq)], integrality=np.r_[np.ones(k), 0., np.zeros(nq)],
               bounds=Bounds(np.r_[np.zeros(k), low, np.zeros(nq)],
                             np.r_[np.ones(k), high, np.full(nq, 10.)]),
               constraints=LinearConstraint(mat, np.r_[lo, np.full(2*nq, -np.inf)], np.r_[hi, available]),
               options=dict(time_limit=seconds, mip_rel_gap=0., presolve=False))
    if sol.x is not None:
        take = sol.x[:k] > .5
        count = np.zeros(nq, dtype=int)
        count[q[take]] = n[take]
        if (len(set(q[take])) != take.sum() or count.sum() != total
                or not respects_tokens(count, legal[:, 1])
                or not within_cap(movement[np.arange(nq), count].sum(), cap)):
            return None, 'unresolved_numerical_check'
        if score is not None:
            return (count, 'score_optimal') if sol.status == 0 else (None, 'unresolved')
        if float(100. * np.min(offset + g[:, take].sum(1)) / mass) > POSITIVE_PP:
            return count, 'candidate'
    if score is None and sol.status == 2:
        return None, 'risk_infeasible'
    dual = getattr(sol, 'mip_dual_bound', None)
    if score is None and dual is not None and np.isfinite(dual) and -100. * dual <= POSITIVE_PP:
        return None, 'no_strict_positive_bound'
    return None, 'unresolved'
