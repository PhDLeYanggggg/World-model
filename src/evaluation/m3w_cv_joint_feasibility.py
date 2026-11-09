"""Maximin full-grid utility with an optional edit budget and original CV risk."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_array, csc_array, vstack

from src.evaluation.m3w_cv_anchor_budget import candidate_terms, risk_excess
from src.evaluation.m3w_selection_forecast_readout import errors


POSITIVE_PP = 1e-6


def coefficients(cv, ols, bank, y, valid, query, easy_cut):
    _, group, counts = np.unique(query, return_inverse=True, return_counts=True)
    w = 1. / counts[group]
    mass, base, gain, _ = candidate_terms(cv, ols, bank, y, valid)
    ref = float(w @ mass)
    known = valid.any(1)
    cv_ade, _ = errors(cv, y, valid)
    easy = known & (cv_ade > 0.) & (cv_ade <= easy_cut)
    signed = np.column_stack([np.where(easy, errors(p, y, valid)[0] - cv_ade, 0.) for p in bank.swapaxes(0, 1)])
    risk = np.concatenate((risk_excess(cv, bank, y, valid, easy_cut), signed[..., None]), axis=-1)
    # Dominance is local to a recipient, so neither query counts nor support change.
    legal = (bank != cv[:, None]).any((2, 3))
    keep = legal.copy()
    quality = np.concatenate((gain, -risk), axis=-1)
    for i in range(bank.shape[1]):
        best = quality[:, i:i+1]
        no_worse = (quality >= best).all(-1)
        wins = (quality > best).any(-1) | (np.arange(bank.shape[1]) < i)
        keep[:, i] &= ~(legal & no_worse & wins).any(1)
    row, arm = np.nonzero(keep)
    return dict(row=row, arm=arm, group=group, counts=counts, ref=ref, easy=easy[row],
                risk=risk[row, arm], easy_cap=.02 * float(cv_ade[easy].sum()),
                offset=np.array([0., w @ base]) * (100. / ref) if ref > 0. else np.zeros(2),
                gain=gain[row, arm] * w[row, None] * (100. / ref) if ref > 0. else np.zeros((len(row), 2)),
                actions_before_prune=int(legal.sum()))


def constraints(data, cap):
    row = data['row']
    n = len(row)
    col = np.arange(n)
    num_q = len(data['counts'])
    by_row = coo_array((np.ones(n), (row, col)), shape=(len(data['group']), n+1))
    by_query = coo_array((np.ones(n), (data['group'][row], col)), shape=(num_q, n+1))
    scale = np.abs(data['risk']).max(0).clip(1.)
    risk = csc_array(np.column_stack((data['risk'].T / scale[:, None], np.zeros(3))))
    support = csc_array(np.r_[-data['easy'].astype(float), 0.][None])
    gain = csc_array(np.column_stack((-data['gain'].T, np.ones(2))))
    matrix = vstack((by_row, by_query, risk, support, gain), format='csc')
    rhs = np.r_[np.ones(len(data['group'])), np.minimum(cap, data['counts']),
                0., 0., data['easy_cap'] / scale[2], -1., data['offset']]
    return matrix, rhs


def solve(data, cap, seconds=1.5):
    n = len(data['row'])
    if data['ref'] <= 0. or not data['easy'].any():
        return None, dict(status='undefined_support', upper_pp=None, lp_status=None, mip_status=None)
    mat, rhs = constraints(data, cap)
    c = np.r_[np.zeros(n), -1.]
    lp = linprog(c, A_ub=mat, b_ub=rhs, bounds=[(0., 1.)]*n+[(None, None)], method='highs',
                 options={'time_limit': seconds})
    upper = float(-lp.fun) if lp.status == 0 else None
    log = dict(status='unresolved', upper_pp=upper, lp_status=int(lp.status), mip_status=None)
    if lp.status == 2:
        return None, dict(log, status='risk_infeasible')
    if upper is not None and upper < -POSITIVE_PP:
        return None, dict(log, status='nonpositive_bound')
    # An integral LP solution is already optimal; check the rounding explicitly.
    if lp.status == 0:
        b = lp.x[:-1] > .5
        if np.max(np.abs(lp.x[:-1] - b)) <= 1e-8 and np.all(mat @ np.r_[b, -lp.fun] <= rhs + 1e-7):
            return b, log
    m = milp(c, integrality=np.r_[np.ones(n), 0.],
             bounds=Bounds(np.r_[np.zeros(n), -np.inf], np.r_[np.ones(n), np.inf]),
             constraints=LinearConstraint(mat, -np.inf, rhs), options={'time_limit': seconds, 'mip_rel_gap': 0.})
    d = getattr(m, 'mip_dual_bound', None)
    if d is not None and np.isfinite(d):
        upper = min(upper, -float(d)) if upper is not None else -float(d)
    log.update(upper_pp=upper, mip_status=int(m.status))
    if m.status == 2:
        return None, dict(log, status='risk_infeasible')
    if upper is not None and upper < -POSITIVE_PP:
        return None, dict(log, status='nonpositive_bound')
    return (m.x[:-1] > .5 if m.x is not None else None), log
