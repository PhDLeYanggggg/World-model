"""Label-aware maximin certificates for fixed menus and chronological budgets."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, hstack, vstack

from src.evaluation.m3w_opendd_location import paired
from src.evaluation.m3w_optional_dual_capacity import coefficients
from src.evaluation.m3w_selection_forecast_readout import policy

POSITIVE_PP = 1e-6


def respects_tokens(count, eligible, credit=1, store=10, cost=5):
    balance = 0
    for n, e in zip(count, eligible, strict=True):
        balance = min(store, balance + credit*int(e))-cost*int(n)
        if balance < 0:
            return False
    return True


def token_rows(by_query, eligible, credit=1, store=10, cost=5):
    """A feasible retained balance exists iff causal token replay can pay every edit."""
    q = len(eligible)
    diag = np.arange(q)
    identity = coo_array((np.ones(q), (diag, diag)), shape=(q, q)).tocsc()
    prev = coo_array((np.ones(q-1), (diag[1:], diag[:-1])), shape=(q, q)).tocsc()
    mat = vstack((hstack((cost*by_query, identity)), hstack((cost*by_query, identity-prev))), format="csc")
    return mat, np.r_[np.full(q, store), credit*np.asarray(eligible, dtype=float)]


def assess(cv, ols, bank, y, valid, query, rec, cuts, eligible, cap, nested=None,
           streaming=False, seconds=2.):
    """Maximize conservative gain versus CV and OLS4, subject to the original risk."""
    m = coefficients(cv, ols, bank, y, valid, cuts[0])
    if m["easy_count"] == 0:
        return {"status": "absent_easy"}
    budget = int(np.sum(eligible))//5
    if budget == 0:
        return {"status": "zero_quota"}
    row, arm = m["row"], m["arm"]
    on = cap[query[row]] > 0
    row, arm = row[on], arm[on]
    gain, risk, easy = m["gain"][:, on], m["risk"][:, on], m["easy"][on]
    if not easy.any():
        return {"status": "no_legal_easy"}
    n, nq = len(row), len(cap)
    cols = np.arange(n)
    by_query = coo_array((np.ones(n), (query[row], cols)), shape=(nq, n+1)).tocsc()
    by_agent = coo_array((np.ones(n), (row, cols)), shape=(len(cv), n+1))
    scale = np.abs(risk).max(1).clip(1.)
    mats = [by_query, by_agent, csc_array(np.r_[np.ones(n), 0.][None]),
            csc_array(np.column_stack((risk/scale[:, None], np.zeros(3)))),
            csc_array(np.r_[easy, 0.][None]),
            csc_array(np.column_stack((-gain/m["cv_mass"], np.ones(2))))]
    lo = np.r_[np.full(nq+len(cv)+4, -np.inf), 1., -np.inf, -np.inf]
    hi = np.r_[cap, np.ones(len(cv)), budget, 0., 0., m["easy_cap"]/scale[2], np.inf, m["offset"]/m["cv_mass"]]
    if nested is not None:
        # Only one arm exists per nominated agent; edit two implies edit one.
        first, second = nested.T
        has_second = second >= 0
        slot = np.full(len(cv), 0.)
        slot[first[has_second]], slot[second[has_second]] = -1., 1.
        mats.append(coo_array((slot[row], (query[row], cols)), shape=(nq, n+1)))
        lo, hi = np.r_[lo, np.full(nq, -np.inf)], np.r_[hi, np.zeros(nq)]
    mat = vstack(mats, format="csc")
    obj = np.r_[np.zeros(n), -1.]
    integrality = np.r_[np.ones(n), 0.]
    low = np.r_[np.zeros(n), -np.inf]
    high = np.r_[np.ones(n), np.inf]
    if streaming:
        tail, upper = token_rows(by_query, eligible)
        mat = vstack((hstack((mat, csc_array((mat.shape[0], nq)))), tail), format="csc")
        lo, hi = np.r_[lo, np.full(2*nq, -np.inf)], np.r_[hi, upper]
        obj, integrality = np.r_[obj, np.zeros(nq)], np.r_[integrality, np.zeros(nq)]
        low, high = np.r_[low, np.zeros(nq)], np.r_[high, np.full(nq, 10.)]
    sol = milp(c=obj, integrality=integrality, bounds=Bounds(low, high),
        constraints=LinearConstraint(mat, lo, hi), options=dict(time_limit=seconds, mip_rel_gap=0.))
    dual = getattr(sol, "mip_dual_bound", None)
    out = {"status": "unresolved", "upper_pp": float(-100.*dual) if dual is not None and np.isfinite(dual) else None}
    if sol.x is not None:
        take = sol.x[:n] > .5
        pred = cv.copy()
        pred[row[take]] = bank[row[take], arm[take]]
        edits = (pred != cv).any((1, 2))
        count = np.bincount(query[edits], minlength=nq)
        if (edits.sum() != take.sum() or (count > cap).any() or edits.sum() > budget
                or (streaming and not respects_tokens(count, eligible))):
            raise ValueError("Incumbent violates exact count constraints")
        if nested is not None and (edits[second[has_second]] & ~edits[first[has_second]]).any():
            raise ValueError("Incumbent violates the nominated prefix")
        checked = policy(pred, cv, cv, y, valid, np.full(len(cv), rec), easy_cut=cuts[0], hard_cut=cuts[1])["recordings"][str(rec)]
        gains = 100.*np.array([paired(ref, pred, y, valid)[1].sum() for ref in (cv, ols)])/m["cv_mass"]
        np.testing.assert_allclose(gains, 100.*(m["offset"]+gain[:, take].sum(1))/m["cv_mass"], atol=1e-10, rtol=1e-8)
        out.update(witness_pp=float(gains.min()), edits=int(edits.sum()))
        if checked["supported"] and gains.min() > POSITIVE_PP:
            out["status"] = "verified_positive_both"
            return out
    if sol.status == 2:
        out["status"] = "risk_infeasible"
    elif out["upper_pp"] is not None and out["upper_pp"] <= POSITIVE_PP:
        out["status"] = "no_strict_positive_bound"
    return out
