"""Hindsight capacity under matched count and CV-relative peak budgets."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_array, csc_array, vstack

from src.evaluation.m3w_cv_anchor_budget import candidate_terms, risk_excess
from src.evaluation.m3w_cv_joint_feasibility import POSITIVE_PP
from src.evaluation.m3w_selection_forecast_readout import errors


def radius_envelope(cap, weight, budget):
    """Fill descending weighted slopes of sum_t min(cap_it, radius_i)."""
    length = np.diff(np.sort(cap, axis=1), axis=1, prepend=0.).ravel()
    slope = (weight[:, None] * np.arange(cap.shape[1], 0, -1)).ravel()
    order = np.argsort(-slope, kind='stable')
    z = length[order]
    spend = np.clip(budget - (np.cumsum(z) - z), 0., z)
    return float(slope[order] @ spend)


def arbitrary_bound(cv, ols, target, valid, query, radii):
    """Upper bound each reference separately, without count or risk constraints."""
    _, group, size = np.unique(query, return_inverse=True, return_counts=True)
    w = 1. / size[group]
    y = np.where(valid[..., None], target, 0.)
    ce = np.linalg.norm(cv - y, axis=-1)
    ref = float(w @ np.where(valid, ce, 0.).sum(1))
    if ref <= 0.:
        return {'query': None, 'record': None}
    caps = np.where(valid[None], ce, np.stack((np.zeros_like(ce), np.linalg.norm(ols - cv, axis=-1))))
    _, base, _, _ = candidate_terms(cv, ols, cv[:, None], target, valid)
    offset = np.array([0., w @ base])
    order = np.argsort(group, kind='stable')
    blocks = np.split(order, size.cumsum()[:-1])
    fixed = sum((np.array([radius_envelope(caps[k, b], w[b], r) for k in range(2)])
                 for b, r in zip(blocks, radii, strict=True) if r > 0.), start=np.zeros(2))
    pooled = np.array([radius_envelope(c, w, radii.sum()) for c in caps])
    if (pooled + 1e-6 < fixed).any():
        raise ValueError('Pooling cannot shrink the relaxed feasible set')
    return {k: (100. * (offset + g) / ref).tolist() for k, g in (('query', fixed), ('record', pooled))}


def design(cv, ols, bank, target, valid, query, easy_cut, edits, radii, mode):
    _, group, size = np.unique(query, return_inverse=True, return_counts=True)
    if mode == 'record':
        group_budget = np.zeros(len(cv), dtype=int)
        edits, radii = np.array([edits.sum()]), np.array([radii.sum()])
    elif mode == 'query':
        group_budget = group
    else:
        raise ValueError('Unknown budget')
    w = 1. / size[group]
    mass, base, gain, _ = candidate_terms(cv, ols, bank, target, valid)
    ref = float(w @ mass)
    cv_ade, _ = errors(cv, target, valid)
    easy = valid.any(1) & (cv_ade > 0.) & (cv_ade <= easy_cut)
    signed = np.column_stack([np.where(easy, errors(p, target, valid)[0] - cv_ade, 0.)
                              for p in bank.swapaxes(0, 1)])
    risk = np.concatenate((risk_excess(cv, bank, target, valid, easy_cut), signed[..., None]), axis=-1)
    radius = np.linalg.norm(bank - cv[:, None], axis=-1).max(-1)
    legal = (radius > 0.) & (radius <= radii[group_budget, None] + 1e-9) & (edits[group_budget, None] > 0)
    # A better path on the same row must also be no more expensive.
    quality = np.concatenate((gain, -risk, -radius[..., None]), axis=-1)
    keep = legal.copy()
    for a in range(bank.shape[1]):
        k = quality[:, a:a+1]
        win = (quality > k).any(-1) | (np.arange(bank.shape[1]) < a)
        keep[:, a] &= ~(legal & (quality >= k).all(-1) & win).any(1)
    row, arm = np.nonzero(keep)
    normalizer = 100. / ref if ref > 0. else 0.
    return dict(row=row, arm=arm, peak=radius[row, arm], group=group_budget[row],
                edits=edits, radii=radii, ref=ref, easy=easy[row], risk=risk[row, arm],
                easy_cap=.02 * float(cv_ade[easy].sum()), offset=np.array([0., w @ base]) * normalizer,
                gain=gain[row, arm] * w[row, None] * normalizer, before_prune=int(legal.sum()))


def constraints(d):
    row, group = d['row'], d['group']
    _, row = np.unique(row, return_inverse=True)
    n = len(row)
    col = np.arange(n)
    shape = (len(d['edits']), n+1)
    by_row = coo_array((np.ones(n), (row, col)), shape=(int(row.max())+1, n+1))
    by_group = coo_array((np.ones(n), (group, col)), shape=shape)
    peak_scale = np.maximum(d['radii'], 1.)
    by_peak = coo_array((d['peak'] / peak_scale[group], (group, col)), shape=shape)
    scale = np.abs(d['risk']).max(0).clip(1.)
    risk = csc_array(np.column_stack((d['risk'].T / scale[:, None], np.zeros(3))))
    support = csc_array(np.r_[-d['easy'].astype(float), 0.][None])
    gain = csc_array(np.column_stack((-d['gain'].T, np.ones(2))))
    return vstack((by_row, by_group, by_peak, risk, support, gain), format='csc'), np.r_[
        np.ones(by_row.shape[0]), d['edits'], d['radii'] / peak_scale,
        0., 0., d['easy_cap'] / scale[2], -1., d['offset']]


def solve(d, seconds):
    log = dict(status='unresolved', upper_pp=None, lp_status=None, mip_status=None)
    if d['ref'] <= 0. or not d['easy'].any():
        return None, dict(log, status='undefined_support')
    a, b = constraints(d)
    n = len(d['row'])
    c = np.r_[np.zeros(n), -1.]
    lp = linprog(c, A_ub=a, b_ub=b, bounds=[(0., 1.)]*n+[(None, None)], method='highs',
                 options={'time_limit': seconds})
    log['lp_status'] = int(lp.status)
    if lp.status == 2:
        return None, dict(log, status='risk_infeasible')
    if lp.status == 0:
        upper = float(-lp.fun)
        log['upper_pp'] = upper
        if upper < -POSITIVE_PP:
            return None, dict(log, status='nonpositive_bound')
        picked = lp.x[:-1] > .5
        if (np.max(np.abs(lp.x[:-1]-picked)) <= 1e-8
                and np.all(a @ np.r_[picked, upper] <= b + 1e-8)):
            return picked, log
    result = milp(c, integrality=np.r_[np.ones(n), 0.],
                  bounds=Bounds(np.r_[np.zeros(n), -np.inf], np.r_[np.ones(n), np.inf]),
                  constraints=LinearConstraint(a, -np.inf, b),
                  options={'time_limit': seconds, 'mip_rel_gap': 0.})
    log['mip_status'] = int(result.status)
    if result.status == 2:
        return None, dict(log, status='risk_infeasible')
    dual = getattr(result, 'mip_dual_bound', None)
    if dual is not None and np.isfinite(dual):
        log['upper_pp'] = -float(dual) if log['upper_pp'] is None else min(log['upper_pp'], -float(dual))
        if log['upper_pp'] < -POSITIVE_PP:
            return None, dict(log, status='nonpositive_bound')
    return (result.x[:-1] > .5 if result.x is not None else None), log
