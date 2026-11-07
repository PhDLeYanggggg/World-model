"""Causal deletion decisions on an unchanged stream of direct actions."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_array, vstack

from src.evaluation.m3w_prefix_capacity import incidence
from src.evaluation.m3w_recording_switch_ceiling import BUDGET, moments

THRESHOLDS = (0., .005, .01, .02, .05, .1, .2)


def fit_stats(x, realized, scale, weight):
    y = np.column_stack((realized['remove_lower'], realized['all_excess'],
                         realized['easy_excess'], realized['easy_signed'],
                         realized['ref_easy'])) / scale[:, None]
    w = weight / weight.sum()
    return x.T @ (x*w[:, None]), x.T @ (y*w[:, None])


def coefficients(stats):
    g, h = (np.mean([v[k] for v in stats], axis=0) for k in (0, 1))
    ridge = np.eye(len(g))*.002
    ridge[0, 0] = 0.
    return np.linalg.solve(g+ridge, h)


def realized(floor, path, target, valid, cut):
    z = moments(floor, path, target, valid, cut)
    z['remove_lower'] = np.where(z['known'], -z['diff'], -z['radius'])
    return z


def deletion_rank(x, coef):
    gain, a, e, signed, ref = (x @ coef).T
    return gain + a + e + np.maximum(signed, 0.) - np.maximum(ref, 0.)


def choose(group, score, threshold):
    selected = np.zeros(len(score), dtype=bool)
    if threshold is None:
        return selected
    for j in group:
        if len(j):
            i = j[score[j].argmax()]
            selected[i] = score[i] > threshold
    return selected


def observe(z, removed, full_easy):
    keep = ~removed
    a, e, signed, ref = (z[k][keep].sum() for k in ('all_excess', 'easy_excess', 'easy_signed', 'ref_easy'))
    return bool(ref > 0 and a <= 0 and e <= 0 and signed <= BUDGET*full_easy), bool(
        full_easy > 0 and signed > BUDGET*full_easy)


def calibrate(records):
    """Return to direct unless donor evidence supports a positive policy change."""
    candidates = []
    for t in THRESHOLDS:
        passes = lost = bad = removals = 0
        mass = 0.
        for g, z, score, full_easy, w in records:
            q = choose(g, score, t)
            baseline, old_bad = observe(z, np.zeros(len(score), bool), full_easy)
            pass_new, new_bad = observe(z, q, full_easy)
            passes += int(pass_new) - int(baseline)
            lost += int(baseline and not pass_new)
            bad += int(new_bad and not old_bad)
            removals += int(q.sum())
            mass += float(w[q] @ z['remove_lower'][q])
        if passes > 0 and lost == 0 and bad == 0 and mass >= 0:
            candidates.append((passes, mass, -removals, t))
    return max(candidates)[-1] if candidates else None


def oracle(z, q, w, full_easy, protect_gain, seconds=3.):
    """Hindsight maximum retained count; original scorer must check witnesses."""
    n = len(q)
    if not n:
        return None, 'no_original_actions'
    if not (z['ref_easy'] > 0).any():
        return None, 'no_original_support'
    _, inv, size = np.unique(q, return_inverse=True, return_counts=True)
    risk = [z[k] for k in ('all_excess', 'easy_excess', 'easy_signed')]
    r = [0., 0., BUDGET*full_easy]
    if protect_gain:
        # The unknown-target penalty for a deletion has the opposite sign
        # from its gain versus the floor; it never becomes a free credit.
        delta = w*z['remove_lower']
        risk.append(delta)
        r.append(float(delta.sum()))
    unit = np.maximum(np.abs(risk).max(1), 1.)
    mat = vstack((incidence(inv, len(size)), csc_array(np.asarray(risk)/unit[:, None]),
                  csc_array((z['ref_easy'][None] > 0).astype(float))), format='csc')
    lo = np.r_[size-1, [-np.inf]*len(r), 1.]
    hi = np.r_[size, np.asarray(r)/unit, np.inf]
    result = milp(c=-np.ones(n), integrality=np.ones(n), bounds=Bounds(0., 1.),
                  constraints=LinearConstraint(mat, lo, hi),
                  options=dict(time_limit=seconds, mip_rel_gap=0.))
    if result.x is None:
        return None, 'infeasible' if result.status == 2 else 'unresolved'
    drop = result.x < .5
    if (np.bincount(inv[drop], minlength=len(size)) > 1).any():
        return None, 'unresolved'
    return drop, 'optimal' if result.status == 0 else 'feasible'
