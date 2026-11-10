"""Certify budget-matched maximin gain as a function of edit exchanges."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, hstack, vstack

from src.evaluation.m3w_count_motion_capacity import respects_tokens, token_rows, within_cap


def exchange_count(plan, anchor):
    return int(np.maximum(plan - anchor, 0).sum())


def solve(terms, movement, cap, anchor, radius, *, seconds=2.):
    """Return a proposed incumbent, a conservative upper bound, and status."""
    gain, risk, support, legal, offset, easy_cap, mass = terms
    n, total = len(anchor), int(anchor.sum())
    if (radius < 1 or anchor.shape != (len(legal),) or (anchor < 0).any() or (anchor > 2).any()
            or not legal[np.arange(n), anchor].all() or not within_cap(movement[np.arange(n), anchor].sum(), cap)
            or not respects_tokens(anchor, legal[:, 1])):
        raise ValueError('Invalid exchange anchor or radius')
    if total == 0 or mass <= 0.:
        return None, None, 'infeasible'
    q, k = np.nonzero(legal[:, 1:] & within_cap(movement[:, 1:], cap))
    k += 1
    on = support[q, k]
    if not on.any():
        return None, None, 'infeasible'

    p = len(k)
    columns = np.arange(p)
    one = coo_array((np.ones(p), (q, columns)), shape=(n, p+1)).tocsc()
    spent = coo_array((k.astype(float), (q, columns)), shape=one.shape).tocsc()
    g, r = gain[q, k].T, risk[q, k].T
    scale = np.abs(r).max(1).clip(1.)
    mats = (one, csc_array(np.r_[k, 0.][None]), csc_array(np.r_[movement[q, k]/cap, 0.][None]),
            csc_array(np.r_[np.maximum(k - anchor[q], 0), 0.][None]),
            csc_array(np.column_stack((r/scale[:, None], np.zeros(3)))),
            csc_array(np.r_[on, 0.][None]),
            csc_array(np.column_stack((-g/mass, np.ones(2)))))
    lo = np.r_[np.full(n, -np.inf), total, -np.inf, -np.inf, np.full(3, -np.inf), 1., -np.inf, -np.inf]
    # The tiny outward allowance only relaxes the upper-bound problem.
    hi = np.r_[np.ones(n), total, 1. + 1e-9, radius, np.array([0., 0., easy_cap])/scale + 1e-10,
               np.inf, offset/mass]
    mat = vstack(mats, format='csc')
    tail, limits = token_rows(spent, legal[:, 1])
    mat = vstack((hstack((mat, csc_array((mat.shape[0], n)))), tail), format='csc')
    ans = milp(c=np.r_[np.zeros(p), -1., np.zeros(n)], integrality=np.r_[np.ones(p), 0., np.zeros(n)],
               bounds=Bounds(np.r_[np.zeros(p), -np.inf, np.zeros(n)], np.r_[np.ones(p), np.inf, np.full(n, 10.)]),
               constraints=LinearConstraint(mat, np.r_[lo, np.full(2*n, -np.inf)], np.r_[hi, limits]),
               options=dict(time_limit=seconds, mip_rel_gap=0., presolve=False))
    dual = getattr(ans, 'mip_dual_bound', None)
    upper = float(-100.*dual + 1e-7) if dual is not None and np.isfinite(dual) else None
    plan = None
    if ans.x is not None:
        use = ans.x[:p] > .5
        c = np.zeros(n, dtype=int)
        c[q[use]] = k[use]
        if (len(set(q[use])) == use.sum() and c.sum() == total and exchange_count(c, anchor) <= radius
                and within_cap(movement[np.arange(n), c].sum(), cap) and respects_tokens(c, legal[:, 1])):
            plan = c
    status = 'infeasible' if ans.status == 2 else 'bounded' if upper is not None else 'unresolved'
    return plan, upper, status
