"""Dual-reference capacity with optional exact query replacement counts."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, vstack

from src.evaluation.m3w_dual_reference_capacity import action_masses, solve
from src.evaluation.m3w_recording_forecast_comparison import comparison_masses
from src.evaluation.m3w_selection_forecast_readout import policy


STATES = ("verified_both", "negative_upper", "risk_infeasible", "unresolved")


def solve_free(m, n_rows, seconds):
    rows = m["row"]
    n = len(rows)
    if not n or not m["easy"].any():
        return None, dict(infeasible=True, solver_status=None, upper_gain_mass=None)
    one = coo_array((np.ones(n), (rows, np.arange(n))), shape=(n_rows, n+1))
    scale = np.maximum(np.max(abs(m["risk"]), axis=1), 1.)
    risk = csc_array(np.column_stack((m["risk"]/scale[:, None], np.zeros(3))))
    support = csc_array(np.r_[m["easy"].astype(float), 0.][None])
    objective = csc_array(np.column_stack((-m["utility"]/m["scale"], np.ones(2))))
    mat = vstack((one, risk, support, objective), format="csc")
    lo = np.r_[np.full(n_rows+3, -np.inf), 1., -np.inf, -np.inf]
    hi = np.r_[np.ones(n_rows), 0., 0., m["easy_cap"]/scale[2], np.inf, m["offset"]/m["scale"]]
    sol = milp(np.r_[np.zeros(n), -1.], integrality=np.r_[np.ones(n), 0.],
        bounds=Bounds(np.r_[np.zeros(n), -np.inf], np.r_[np.ones(n), np.inf]),
        constraints=LinearConstraint(mat, lo, hi), options=dict(time_limit=seconds, mip_rel_gap=0.))
    upper = getattr(sol, "mip_dual_bound", None)
    return (None if sol.x is None else sol.x[:-1] > .5), dict(
        solver_status=int(sol.status), infeasible=sol.status == 2,
        upper_gain_mass=float(-upper*m["scale"]) if upper is not None and np.isfinite(upper) else None)


def evidence(cv, ols, bank, y, valid, inverse, counts, weight, rec, cut, rows, arms):
    if len(np.unique(rows)) != len(rows):
        raise ValueError("At most one replacement per row")
    p = cv.copy()
    p[rows] = bank[rows, arms]
    on = np.any(p != cv, axis=(1, 2))
    if int(on.sum()) != len(rows):
        raise ValueError("A replacement may not be an identity path")
    if counts is not None and not np.array_equal(np.bincount(inverse[on], minlength=len(counts)), counts):
        raise ValueError("Witness did not reproduce exact query counts")
    risk = policy(p, cv, cv, y, valid, np.full(len(cv), rec),
                  easy_cut=cut["easy"], hard_cut=cut["hard"])["recordings"][rec]
    masses = [comparison_masses(b, p, cv, y, valid, weight) for b in (cv, ols)]
    gain = [v["known_delta"]-v["unknown_radius"] for v in masses]
    return dict(gain_mass=gain, reference=masses[0]["reference"], risk=risk, switched=int(on.sum()))


def assess(cv, ols, bank, y, valid, inverse, counts, weight, rec, cut, *, budgets=(2., 8.)):
    m = action_masses(cv, ols, bank, y, valid, weight, cut["easy"])
    tol = 1e-7*m["scale"]
    for seconds in budgets:
        take, s = (solve(m, inverse, counts, seconds) if counts is not None
                   else solve_free(m, len(cv), seconds))
        result = dict(s, status="unresolved", solve_seconds_budget=seconds, tolerance_mass=tol)
        if take is not None:
            rows, arms = m["row"][take], m["arm"][take]
            val = evidence(cv, ols, bank, y, valid, inverse, counts, weight, rec, cut, rows, arms)
            np.testing.assert_allclose(val["gain_mass"], m["offset"]+m["utility"][:, take].sum(1),
                                       atol=tol, rtol=1e-10)
            if val["risk"]["supported"] and min(val["gain_mass"]) > tol:
                return dict(result, status="verified_both", evidence=val, witness=[rows.tolist(), arms.tolist()])
        if s["infeasible"]:
            return dict(result, status="risk_infeasible")
        if s["upper_gain_mass"] is not None and s["upper_gain_mass"] < -tol:
            return dict(result, status="negative_upper")
    return result
