"""Create recording-level training labels; never used by past-only inference."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, vstack

from src.evaluation.m3w_recording_switch_ceiling import BUDGET, moments
from src.evaluation.m3w_recording_forecast_comparison import comparison_masses
from src.evaluation.m3w_selection_forecast_readout import errors, policy


def teacher(cv, ols, paths, y, valid, offsets, cap, recording, cut, seconds=6.):
    """Optimize the shared risk budget, restricting teachers to known rows."""
    size = np.diff(offsets)
    query = np.repeat(np.arange(len(size)), size)
    n = len(cv)
    target = np.zeros(n, dtype=bool)
    known = valid.any(1)
    w = 1./size[query]
    ref, _ = errors(cv, y, valid)
    easy = known & (ref > 0.) & (ref <= cut['easy'])
    can = np.flatnonzero(known & (cap[query] > 0))
    e = np.column_stack([np.where(np.any((p[can] != cv[can]), axis=(1, 2)),
                     errors(p[can], y[can], valid[can])[0], np.inf) for p in paths.transpose(1, 0, 2, 3)])
    arm = e.argmin(1)
    row = can[np.isfinite(e.min(1))]
    arm = arm[np.isfinite(e.min(1))]
    if not easy[row].any():
        return target, dict(status='no_eligible_easy_reference', positive=False)
    m = moments(cv[row], paths[row, arm], y[row], valid[row], cut['easy'])
    risk = np.column_stack([m[k] for k in ('all_excess', 'easy_excess', 'easy_signed')])
    unit = np.abs(risk).max(0).clip(1.)
    q = query[row]
    membership = coo_array((np.ones(len(row)), (q, np.arange(len(row)))), shape=(len(size), len(row)))
    matrix = vstack((membership, csc_array((risk/unit).T), csc_array(easy[row][None])), format='csc')
    ub = np.r_[cap, 0., 0., BUDGET*ref[easy].sum()/unit[2], np.inf]
    lb = np.r_[np.full(len(ub)-1, -np.inf), 1.]
    value = w[row]*m['diff']
    opt = milp(-value/max(1., np.abs(value).max()), integrality=np.ones(len(row)), bounds=Bounds(0., 1.),
               constraints=LinearConstraint(matrix, lb, ub), options=dict(time_limit=seconds, mip_rel_gap=0.))
    state = dict(status={0:'optimal', 1:'timed', 2:'infeasible'}.get(opt.status, 'unresolved'), positive=False)
    if opt.x is None:
        return target, state
    use = opt.x > .5
    if np.any(np.bincount(q[use], minlength=len(cap)) > cap):
        return target, dict(status='failed_count_recheck', positive=False)
    pred = cv.copy()
    pred[row[use]] = paths[row[use], arm[use]]
    check = policy(pred, cv, cv, y, valid, np.full(n, recording),
                   easy_cut=cut['easy'], hard_cut=cut['hard'])['recordings'][recording]
    if not check['supported']:
        return target, dict(status='failed_original_risk_recheck', positive=False)
    cmp = comparison_masses(ols, pred, cv, y, valid, w)
    gain = cmp['known_delta']-cmp['unknown_radius']
    cv_cmp = comparison_masses(ols, cv, cv, y, valid, w)
    np.testing.assert_allclose(gain, cv_cmp['known_delta']-cv_cmp['unknown_radius']+value[use].sum(),
                               rtol=1e-10, atol=1e-7)
    if gain > 0.:
        target[row[use]] = True
    return target, dict(state, positive=gain > 0., verified_feasible=True, gain_mass=gain,
                        selected=int(target.sum()))
