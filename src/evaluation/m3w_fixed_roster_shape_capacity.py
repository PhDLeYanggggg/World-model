"""Shape-only risk capacity with no ability to change which rows are edited."""

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, vstack

from src.evaluation.m3w_opendd_location import paired
from src.evaluation.m3w_optional_dual_capacity import coefficients
from src.evaluation.m3w_selection_forecast_readout import policy
from src.evaluation.m3w_streaming_menu_capacity import POSITIVE_PP


def assess(cv, ols, bank, target, valid, rec, cuts, active, seconds=2.):
    if not active.any():
        return "zero_edits"
    masked = np.where(active[:, None, None, None], bank, cv[:, None])
    m = coefficients(cv, ols, masked, target, valid, cuts[0])
    if not m["easy"].any():
        return "absent_easy" if not m["easy_count"] else "no_selected_easy"
    row, arm = m["row"], m["arm"]
    n = len(row)
    by_row = coo_array((np.ones(n), (row, np.arange(n))), shape=(len(cv), n+1))
    r = m["risk"]
    scale = np.abs(r).max(1).clip(1.)
    mat = vstack((by_row, csc_array(np.column_stack((r/scale[:, None], np.zeros(3)))),
        csc_array(np.r_[m["easy"], 0.][None]), csc_array(np.column_stack((-m["gain"]/m["cv_mass"], np.ones(2))))), format="csc")
    lo = np.r_[active.astype(float), np.full(3, -np.inf), 1., -np.inf, -np.inf]
    hi = np.r_[active.astype(float), 0., 0., m["easy_cap"]/scale[2], np.inf, m["offset"]/m["cv_mass"]]
    res = milp(c=np.r_[np.zeros(n), -1.], integrality=np.r_[np.ones(n), 0.],
        bounds=Bounds(np.r_[np.zeros(n), -np.inf], np.r_[np.ones(n), np.inf]),
        constraints=LinearConstraint(mat, lo, hi), options=dict(time_limit=seconds, mip_rel_gap=0.))
    if res.x is not None:
        use = res.x[:n] > .5
        pred = cv.copy()
        pred[row[use]] = bank[row[use], arm[use]]
        if use.sum() != active.sum() or not np.array_equal((pred != cv).any((1, 2)), active):
            raise ValueError("A capacity witness changed the edit roster")
        rr = policy(pred, cv, cv, target, valid, np.full(len(cv), rec), easy_cut=cuts[0], hard_cut=cuts[1])["recordings"][str(rec)]
        gain = 100.*np.array([paired(ref, pred, target, valid)[1].sum() for ref in (cv, ols)])/m["cv_mass"]
        np.testing.assert_allclose(gain, 100.*(m["offset"]+m["gain"][:, use].sum(1))/m["cv_mass"], atol=1e-10, rtol=1e-8)
        if rr["supported"] and gain.min() > POSITIVE_PP:
            return "verified_positive_both"
    if res.status == 2:
        return "risk_infeasible"
    dual = getattr(res, "mip_dual_bound", None)
    if dual is not None and -100.*dual <= POSITIVE_PP:
        return "no_strict_positive_bound"
    return "unresolved"
