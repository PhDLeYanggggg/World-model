"""Hindsight capacity at a shared sum-of-peak-displacements budget."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, vstack

from src.evaluation.m3w_opendd_location import paired
from src.evaluation.m3w_recording_switch_ceiling import BUDGET, moments


def radius_frontier(cv, ols, y, valid, eligible):
    """Exact risk-free OLS4 upper frontier with arbitrary per-step directions.

    Given row radius r, an observed step can recover min(r, CV error);
    an unobserved step can recover min(r, CV-to-OLS distance) from its
    triangle debit. Each is attainable by projecting onto the CV ball.
    Concavity makes global descending-slope allocation optimal.
    """
    cap = np.where(valid, np.linalg.norm(cv - y, axis=-1),
                   np.linalg.norm(cv - ols, axis=-1))
    knots = np.sort(cap[eligible], axis=1)
    lengths = np.diff(knots, axis=1, prepend=0.)
    slopes = np.arange(cap.shape[1], 0, -1)
    # All rows share integer slopes; sum the corresponding segment lengths.
    width = lengths.sum(0)
    return width, slopes, float(paired(ols, cv, y, valid)[1].sum())


def upper_gain(frontier, budget):
    width, slope, base = frontier
    consumed = np.r_[0., np.cumsum(width)[:-1]]
    return float(base + np.clip(budget - consumed, 0., width) @ slope)


def break_even_radius(frontier):
    """Infimum budget for nonnegative lower gain; None if unattainable."""
    width, slope, base = frontier
    if base >= 0.:
        return 0.
    gain = np.cumsum(width * slope)
    if gain[-1] < -base:
        return None
    j = int(np.searchsorted(gain, -base))
    return float(width[:j].sum() + (-base - (gain[j - 1] if j else 0.)) / slope[j])


def coefficients(cv, ols, bank, y, valid, easy_cut):
    """Do not discard complete-row actions: better paths can cost more."""
    actions = bank.swapaxes(0, 1)
    mass = [moments(cv, p, y, valid, easy_cut) for p in actions]
    radius = np.column_stack([m["radius"] for m in mass])
    row, arm = np.nonzero(radius > 0.)
    base = paired(ols, cv, y, valid)[1].sum(1)
    risk = np.stack([np.column_stack([m[k] for m in mass])[row, arm]
                     for k in ("all_excess", "easy_excess", "easy_signed")])
    gain = np.stack([np.column_stack([paired(ref, p, y, valid)[1].sum(1)
                                      for p in actions])[row, arm] for ref in (cv, ols)])
    gain[1] -= base[row]
    return dict(row=row, arm=arm, cost=radius[row, arm], rows=len(y), risk=risk,
                easy=mass[0]["easy"][row], easy_cap=BUDGET * mass[0]["ref_easy"].sum(),
                gain=gain, offset=np.array([0., base.sum()]),
                cv_mass=np.linalg.norm(cv - y, axis=-1)[valid].sum())


def solve(m, budget, seconds=15.):
    row, n = m["row"], len(m["row"])
    if budget <= 0. or m["cv_mass"] <= 0. or not m["easy"].any():
        return None, dict(status="insufficient_budget_or_support", upper_pp=None)
    one = coo_array((np.ones(n), (row, np.arange(n))), shape=(m["rows"], n + 1))
    scale = np.abs(m["risk"]).max(1).clip(1.)
    risk = csc_array(np.column_stack((m["risk"] / scale[:, None], np.zeros(3))))
    support_cost = csc_array(np.column_stack((np.stack((m["easy"], m["cost"] / budget)), np.zeros(2))))
    utility = csc_array(np.column_stack((-m["gain"] / m["cv_mass"], np.ones(2))))
    matrix = vstack((one, risk, support_cost, utility), format="csc")
    lo = np.r_[np.full(m["rows"] + 3, -np.inf), 1., -np.inf, -np.inf, -np.inf]
    hi = np.r_[np.ones(m["rows"]), 0., 0., m["easy_cap"] / scale[2], np.inf, 1.,
               m["offset"] / m["cv_mass"]]
    sol = milp(c=np.r_[np.zeros(n), -1.], integrality=np.r_[np.ones(n), 0.],
        bounds=Bounds(np.r_[np.zeros(n), -np.inf], np.r_[np.ones(n), np.inf]),
        constraints=LinearConstraint(matrix, lo, hi), options=dict(time_limit=seconds, mip_rel_gap=0.))
    bound = getattr(sol, "mip_dual_bound", None)
    return (sol.x[:-1] > .5 if sol.x is not None else None), dict(
        status={0: "optimal", 2: "infeasible"}.get(sol.status, "unresolved"),
        solver_status=int(sol.status),
        upper_pp=float(-100 * bound) if bound is not None and np.isfinite(bound) else None)
