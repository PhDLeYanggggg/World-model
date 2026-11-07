"""Offline multi-action ceiling under frozen counts or per-query caps."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, vstack

from src.evaluation.m3w_recording_switch_ceiling import BUDGET, moments, policy


def solve(floor, paths, allowed, raw, target, valid, queries, recording, easy_cut, hard_cut, *, fixed, seconds=5.):
    _, group, size = np.unique(queries, return_inverse=True, return_counts=True)
    n, m = allowed.shape
    raw_count = np.bincount(group[raw >= 0], minlength=len(size))
    cap = raw_count if fixed else np.minimum(2, size-1)
    if not cap.any():
        return dict(status='infeasible', reason='no_selected_count_support')
    per_action = [moments(floor, paths[:, j], target, valid, easy_cut) for j in range(m)]
    risk = np.stack([np.column_stack([p[k] for p in per_action]) for k in ('all_excess','easy_excess','easy_signed')])
    radius = np.column_stack([p['radius'] for p in per_action])
    util = np.column_stack([p['utility'] for p in per_action])
    ref = per_action[0]['ref_easy']; full_easy = BUDGET * ref.sum()
    legal = allowed & (radius > 0.) & (cap[group] > 0)[:, None]
    if not (legal.any(1) & (ref > 0.)).any():
        return dict(status='infeasible', reason='no_eligible_easy_reference')
    if fixed:
        lower = np.where(legal[None], risk, np.inf).min(2)
        minimum = np.zeros(3)
        for j in np.flatnonzero(cap):
            v = lower[:, group == j]
            if (np.isfinite(v[0]).sum() < cap[j]):
                return dict(status='infeasible', reason='not_enough_eligible_rows')
            minimum += np.sort(v, axis=1)[:, :cap[j]].sum(1)
        for k, label in enumerate(('all_excess_lower_bound','easy_excess_lower_bound','signed_easy_lower_bound')):
            if minimum[k] > (full_easy if k == 2 else 0.) + 1e-9:
                return dict(status='infeasible', reason=label)
    i, a = np.where(legal); v = np.arange(len(i))
    row = coo_array((np.ones(len(v)), (i, v)), shape=(n, len(v))).tocsc()
    scene = coo_array((np.ones(len(v)), (group[i], v)), shape=(len(size), len(v))).tocsc()
    terms = risk[:, i, a]; unit = np.abs(terms).max(1).clip(1.)
    easy = (ref[i] > 0.).astype(float)
    mat = vstack((row, scene, csc_array(terms/unit[:, None]), csc_array(easy[None])), format='csc')
    lo = np.r_[np.zeros(n), cap if fixed else np.zeros(len(cap)), -np.inf, -np.inf, -np.inf, 1.]
    hi = np.r_[np.ones(n), cap, 0., 0., full_easy/unit[2], np.inf]
    utility = util[i, a] / size[group[i]]
    sol = milp(-utility/max(1., np.abs(utility).max()), integrality=np.ones(len(v)), bounds=Bounds(0., 1.),
               constraints=LinearConstraint(mat, lo, hi), options=dict(time_limit=seconds, mip_rel_gap=0.))
    if sol.status == 2:
        return dict(status='infeasible', reason='integer_constraints')
    if sol.x is None:
        return dict(status='unresolved', reason=f'solver_status_{sol.status}')
    use = sol.x > .5; rows, acts = i[use], a[use]; prediction = floor.copy()
    prediction[rows] = paths[rows, acts]
    r = policy(prediction, floor, floor, target, valid, np.full(n, recording), easy_cut=easy_cut, hard_cut=hard_cut)
    r = r['recordings'][recording]
    counts = np.bincount(group[rows], minlength=len(cap))
    if (len(np.unique(rows)) != len(rows) or (counts > cap).any() or fixed and not np.array_equal(counts, cap)
            or not r['supported'] or r['selected'] != len(rows)):
        return dict(status='unresolved', reason='original_evaluator_postcheck')
    return dict(status='optimal' if sol.status == 0 else 'feasible_unproven', risk=r,
                switches=len(rows), completion_gain_mass=float(utility[use].sum()))
