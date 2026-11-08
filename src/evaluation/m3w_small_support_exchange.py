"""Outcome-aware support exchange capacity at the incumbent's query counts."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_array, vstack

from src.evaluation.m3w_prefix_capacity import incidence
from src.evaluation.m3w_recording_forecast_comparison import comparison_masses
from src.evaluation.m3w_recording_switch_ceiling import BUDGET, moments
from src.evaluation.m3w_selection_forecast_readout import errors

TERMS = ("all_excess", "easy_excess", "easy_signed")
POS = 1e-6


def menu(cv, bank, ols, initial, groups, y, valid, cut):
    n = len(cv)
    active = initial >= 0
    original = cv.copy()
    original[active] = bank[active, initial[active]]
    old = moments(cv, original, y, valid, cut)
    cv_error, easy, known = old["base"], old["easy"], old["known"]
    candidate_error = np.column_stack([errors(p, y, valid)[0] for p in bank.swapaxes(0, 1)])
    radius = np.linalg.norm(bank-cv[:, None], axis=-1).max(2)
    ols_old = np.linalg.norm(original-ols, axis=-1).max(1)
    ols_floor = np.linalg.norm(cv-ols, axis=-1).max(1)
    weights = np.empty(n)
    choices, matrix = [], []
    for g, q in enumerate(groups):
        weights[q] = 1./len(q)
        incoming = q[~active[q] & easy[q]]
        if not len(incoming):
            continue
        for a in q[active[q]]:
            # The replacement may not move farther from CV than the removed edit.
            allow = (radius[incoming] > 0.) & (radius[incoming] <= old["radius"][a])
            b = np.where(allow, candidate_error[incoming], np.inf).argmin(1)
            j = np.flatnonzero(allow.any(1))
            for r, k in zip(incoming[j], b[j], strict=True):
                gain = cv_error[r]-candidate_error[r, k]
                excess = max(-gain, 0.)-BUDGET*cv_error[r]
                base = weights[a]*(gain-old["diff"][a])
                unknown = not known[a]
                choices.append((g, a, r, k))
                matrix.append((excess-old["all_excess"][a], excess-old["easy_excess"][a],
                    -gain-old["easy_signed"][a], 1-int(easy[a]),
                    base-weights[a]*old["radius"][a]*unknown,
                    base+weights[a]*old["radius"][a]*unknown,
                    base+weights[a]*(ols_old[a]-ols_floor[a])*unknown))
    # Columns: three risk deltas, support count delta, and exact lower deltas
    # against incumbent, CV, and OLS4 respectively.
    choice = np.asarray(choices, dtype=int).reshape(-1, 4)
    terms = np.asarray(matrix, dtype=float).reshape(-1, 7)
    ols_cmp = comparison_masses(ols, original, cv, y, valid, weights)
    cv_cmp = comparison_masses(cv, original, cv, y, valid, weights)
    return dict(choice=choice, terms=terms, original=original, full_easy_mass=float(cv_error[easy].sum()),
                risk=np.array([old[k][active].sum() for k in TERMS]),
                support=int((easy & active).sum()), full_easy=int(easy.sum()), reference=cv_cmp["reference"],
                old_lower=np.array([cv_cmp["known_delta"]-cv_cmp["unknown_radius"],
                                    ols_cmp["known_delta"]-ols_cmp["unknown_radius"]]), weights=weights)


def solve(problem, *, single, joint, seconds=3.):
    j, t = problem["choice"], problem["terms"]
    target = max(1., problem["reference"])*POS
    upper_risk = np.array([0., 0., BUDGET*problem["full_easy_mass"]])-problem["risk"]
    if not problem["full_easy"]:
        return None, "structural_no_easy"
    if not len(j):
        return (np.empty(0, dtype=int), "unchanged_only") if (
            not joint and problem["support"] and np.all(upper_risk >= 0.)) else (None, "no_legal_exchange")
    if not problem["support"] and not (t[:, 3] > 0).any():
        return None, "no_eligible_easy_exchange"
    if single:
        z = np.vstack((np.zeros(t.shape[1]), t))
        feasible = (z[:, :3] <= upper_risk).all(1) & (z[:, 3]+problem["support"] >= 1)
        if joint:
            feasible &= (z[:, 4] >= target) & (z[:, 5:]+problem["old_lower"] >= target).all(1)
        choices = np.flatnonzero(feasible)
        if not len(choices):
            return None, "infeasible"
        best = int(choices[z[choices, 4].argmax()])
        return np.array([best-1], dtype=int) if best else np.empty(0, dtype=int), "optimal"
    block = np.r_[t[:, :4].T, t[:, 4:].T] if joint else t[:, :4].T
    lower = np.r_[[-np.inf]*3, 1-problem["support"]]
    upper = np.r_[upper_risk, np.inf]
    if joint:
        lower = np.r_[lower, target, target-problem["old_lower"]]
        upper = np.r_[upper, [np.inf]*3]
    scale = np.abs(block).max(1).clip(1.)
    g = np.unique(j[:, 0], return_inverse=True)[1]
    count = incidence(g, int(g.max())+1)
    mat = vstack((count, csc_array(block/scale[:, None])), format="csc")
    lo = np.r_[np.full(count.shape[0], -np.inf), lower/scale]
    hi = np.r_[np.ones(count.shape[0]), upper/scale]
    objective = -t[:, 4]/max(1., np.abs(t[:, 4]).max())
    result = milp(objective, integrality=np.ones(len(j)), bounds=Bounds(0., 1.),
                  constraints=LinearConstraint(mat, lo, hi),
                  options=dict(time_limit=seconds, mip_rel_gap=0.))
    if result.x is None:
        return None, "infeasible" if result.status == 2 else "unresolved"
    take = np.flatnonzero(result.x > .5)
    if np.unique(j[take, 0]).size != len(take):
        raise ValueError("Exchange count violated")
    return take, "optimal" if result.status == 0 else "feasible_unresolved"


def paths(problem, selected, initial, cv, bank):
    choose = initial.copy()
    for _, drop, add, action in problem["choice"][selected]:
        choose[drop], choose[add] = -1, action
    use = choose >= 0
    out = cv.copy()
    out[use] = bank[use, choose[use]]
    return out, choose
