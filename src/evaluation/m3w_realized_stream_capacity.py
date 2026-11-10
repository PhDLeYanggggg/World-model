"""Exact realized-count hindsight optimization with a frozen nested action menu."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, hstack, vstack

from src.evaluation.m3w_streaming_menu_capacity import POSITIVE_PP, respects_tokens, token_rows


def solve(gain, risk, easy, legal, offset, easy_cap, mass, total, *, seconds=2.):
    """Return query prefix counts and a bound on maximin conservative gain."""
    qn = len(legal)
    if (legal.shape != (qn, 3) or not legal[:, 0].all() or np.any(legal[:, 2] & ~legal[:, 1])
            or gain.shape != (qn, 3, 2) or risk.shape != (qn, 3, 3) or easy.shape != legal.shape
            or not np.isfinite(gain[legal]).all() or not np.isfinite(risk[legal]).all()
            or total < 0 or total > int(legal[:, 1].sum())//5):
        raise ValueError('Invalid frozen menu or realized quota')
    if total == 0:
        return None, dict(status='zero_realized', upper_pp=None)
    q, n = np.nonzero(legal[:, 1:])
    n += 1
    support = easy[q, n]
    if not support.any() or mass <= 0.:
        return None, dict(status='no_support', upper_pp=None)
    g, r = gain[q, n].T, risk[q, n].T
    variables = len(n)
    col = np.arange(variables)
    one = coo_array((np.ones(variables), (q, col)), shape=(qn, variables+1)).tocsc()
    spent = coo_array((n.astype(float), (q, col)), shape=one.shape).tocsc()
    scale = np.abs(r).max(1).clip(1.)
    mat = vstack((one, csc_array(np.r_[n, 0.][None]),
                  csc_array(np.column_stack((r/scale[:, None], np.zeros(3)))),
                  csc_array(np.r_[support, 0.][None]),
                  csc_array(np.column_stack((-g/mass, np.ones(2))))), format='csc')
    lower = np.r_[np.full(qn, -np.inf), total, np.full(3, -np.inf), 1., -np.inf, -np.inf]
    upper = np.r_[np.ones(qn), total, 0., 0., easy_cap/scale[2], np.inf, offset/mass]
    tail, available = token_rows(spent, legal[:, 1])
    mat = vstack((hstack((mat, csc_array((mat.shape[0], qn)))), tail), format='csc')
    bounds = Bounds(np.r_[np.zeros(variables), -np.inf, np.zeros(qn)],
                    np.r_[np.ones(variables), np.inf, np.full(qn, 10.)])
    sol = milp(c=np.r_[np.zeros(variables), -1., np.zeros(qn)],
               integrality=np.r_[np.ones(variables), 0., np.zeros(qn)], bounds=bounds,
               constraints=LinearConstraint(mat, np.r_[lower, np.full(2*qn, -np.inf)], np.r_[upper, available]),
               options=dict(time_limit=seconds, mip_rel_gap=0.))
    dual = getattr(sol, 'mip_dual_bound', None)
    upper_pp = float(-100.*dual) if dual is not None and np.isfinite(dual) else None
    if sol.x is not None:
        take = sol.x[:variables] > .5
        count = np.zeros(qn, dtype=int)
        count[q[take]] = n[take]
        if len(set(q[take])) != take.sum() or count.sum() != total or not respects_tokens(count, legal[:, 1]):
            raise ValueError('Solver returned an infeasible budget allocation')
        achieved = float(100.*np.min(offset+g[:, take].sum(1))/mass)
        if achieved > POSITIVE_PP:
            return count, dict(status='candidate', upper_pp=upper_pp, witness_pp=achieved)
    status = ('risk_infeasible' if sol.status == 2 else 'no_strict_positive_bound'
              if upper_pp is not None and upper_pp <= POSITIVE_PP else 'unresolved')
    return None, dict(status=status, upper_pp=upper_pp)
