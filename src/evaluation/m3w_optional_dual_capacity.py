"""Optional-recipient full-grid maximin capacity with the original CV risk."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, vstack

from src.evaluation.m3w_opendd_location import paired
from src.evaluation.m3w_recording_switch_ceiling import BUDGET, moments


def coefficients(cv, ols, bank, y, valid, easy_cut):
    values = [moments(cv, p, y, valid, easy_cut) for p in bank.swapaxes(0, 1)]
    radius = np.column_stack([m["radius"] for m in values])
    legal = radius > 0.
    known = valid.any(1)
    complete = valid.all(1)
    # On complete rows the smallest-error nonidentity path dominates all other nonidentity choices.
    best = np.where(legal, np.column_stack([m["diff"] for m in values]), -np.inf).argmax(1)
    row, arm = np.nonzero(legal & (~complete[:, None] | (np.arange(bank.shape[1]) == best[:, None])))
    base_ols = paired(ols, cv, y, valid)[1].sum(1)
    cv_mass = np.linalg.norm(cv-y, axis=-1)[valid].sum()
    m = values[0]
    risk = np.stack([np.column_stack([d[k] for d in values])[row, arm]
                     for k in ("all_excess", "easy_excess", "easy_signed")])
    gain = np.stack([np.column_stack([paired(ref, p, y, valid)[1].sum(1)
                                     for p in bank.swapaxes(0, 1)])[row, arm] for ref in (cv, ols)])
    gain[1] -= base_ols[row]
    return dict(row=row, arm=arm, risk=risk, easy=m["easy"][row], easy_count=int(m["easy"].sum()),
        easy_cap=BUDGET*float(m["ref_easy"].sum()), cv_mass=float(cv_mass), gain=gain,
        offset=np.array([0., base_ols.sum()]), rows=len(cv), known=known)


def solve(m, seconds=2.):
    row = m["row"]
    n = len(row)
    if not m["easy"].any() or m["cv_mass"] <= 0.:
        return None, dict(status="absent_easy" if m["easy_count"] == 0 else "no_legal_easy",
                          solver_status=None, upper_pp=None)
    support = csc_array(np.r_[m["easy"], 0.][None])
    by_agent = coo_array((np.ones(n), (row, np.arange(n))), shape=(m["rows"], n+1))
    s = np.abs(m["risk"]).max(1).clip(1.)
    risk = csc_array(np.column_stack((m["risk"]/s[:, None], np.zeros(3))))
    utility = csc_array(np.column_stack((-m["gain"]/m["cv_mass"], np.ones(2))))
    matrix = vstack((by_agent, risk, support, utility), format="csc")
    lower = np.r_[np.full(m["rows"]+3, -np.inf), 1., -np.inf, -np.inf]
    upper = np.r_[np.ones(m["rows"]), 0., 0., m["easy_cap"]/s[2], np.inf, m["offset"]/m["cv_mass"]]
    sol = milp(c=np.r_[np.zeros(n), -1.], integrality=np.r_[np.ones(n), 0.],
        bounds=Bounds(np.r_[np.zeros(n), -np.inf], np.r_[np.ones(n), np.inf]),
        constraints=LinearConstraint(matrix, lower, upper), options=dict(time_limit=seconds, mip_rel_gap=0.))
    dual = getattr(sol, "mip_dual_bound", None)
    bound = float(-100*dual) if dual is not None and np.isfinite(dual) else None
    status = "risk_infeasible" if sol.status == 2 else "solved" if sol.status == 0 else "unresolved"
    return (sol.x[:-1] > .5 if sol.x is not None else None), dict(status=status, solver_status=int(sol.status), upper_pp=bound)
