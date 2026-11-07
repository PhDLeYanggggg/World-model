"""Hindsight maximin utility under exact switches and original recording risk."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, vstack

from src.evaluation.m3w_recording_switch_ceiling import BUDGET, moments
from src.evaluation.m3w_selection_forecast_readout import errors


def action_masses(cv, ols, bank, target, valid, weight, easy_cut):
    """Encode each nonidentity action as an increment from the CV fallback."""
    known = valid.any(1)
    cv_error, _ = errors(cv, target, valid)
    ols_error, _ = errors(ols, target, valid)
    ols_cv_radius = np.linalg.norm(ols-cv, axis=-1).max(1)
    offset = float(weight @ np.where(known, ols_error-cv_error, -ols_cv_radius))
    easy = known & (cv_error > 0.) & (cv_error <= easy_cut)
    fields = ('all_excess', 'easy_excess', 'easy_signed', 'utility')
    values = [moments(cv, p, target, valid, easy_cut) for p in bank.swapaxes(0, 1)]
    radius = np.stack([m['radius'] for m in values], axis=1)
    legal = radius > 0.
    utility = np.stack([m['utility'] for m in values], axis=1)
    # On observed rows, lower ADE dominates for both references and all risk checks.
    best = np.where(legal, utility, -np.inf).argmax(1)
    rows, arms = np.nonzero(legal & (~known[:, None] | (np.arange(bank.shape[1]) == best[:, None])))
    data = {key: np.stack([m[key] for m in values], axis=1)[rows, arms] for key in fields}
    dist = np.linalg.norm(ols[rows]-bank[rows, arms], axis=-1).max(1)
    ols_increment = np.where(known[rows], data['utility'], ols_cv_radius[rows]-dist)
    return dict(row=rows, arm=arms, risk=np.stack([data[k] for k in fields[:3]]),
                easy=easy[rows], easy_cap=BUDGET*float(cv_error[easy].sum()),
                utility=np.stack((data['utility'], ols_increment))*weight[rows],
                offset=np.array([0., offset]), scale=max(1., float(weight[known] @ cv_error[known])))


def solve(m, query_inverse, counts, seconds):
    """Maximize the smaller CV/OLS4 conservative gain, not only the CV gain."""
    rows = m['row']
    n, agents = len(rows), len(query_inverse)
    if not n or not m['easy'].any():
        return None, dict(solver_status=None, infeasible=True, upper_gain_mass=None)
    by_query = query_inverse[rows]
    columns = np.arange(n)
    eq = coo_array((np.ones(n), (by_query, columns)), shape=(len(counts), n+1))
    one_path = coo_array((np.ones(n), (rows, columns)), shape=(agents, n+1))
    scale = np.maximum(np.max(abs(m['risk']), axis=1), 1.)
    risk = csc_array(np.column_stack((m['risk']/scale[:, None], np.zeros(3))))
    support = csc_array(np.r_[m['easy'].astype(float), 0.][None])
    utility = csc_array(np.column_stack((-m['utility']/m['scale'], np.ones(2))))
    mat = vstack((eq, one_path, risk, support, utility), format='csc')
    lower = np.r_[counts, np.full(agents+3, -np.inf), 1., -np.inf, -np.inf]
    upper = np.r_[counts, np.ones(agents), 0., 0., m['easy_cap']/scale[2], np.inf, m['offset']/m['scale']]
    sol = milp(c=np.r_[np.zeros(n), -1.], integrality=np.r_[np.ones(n), 0.],
               bounds=Bounds(np.r_[np.zeros(n), -np.inf], np.r_[np.ones(n), np.inf]),
               constraints=LinearConstraint(mat, lower, upper),
               options=dict(time_limit=seconds, mip_rel_gap=0.))
    bound = getattr(sol, 'mip_dual_bound', None)
    status = dict(solver_status=int(sol.status), infeasible=sol.status == 2,
                  upper_gain_mass=float(-bound*m['scale']) if bound is not None and np.isfinite(bound) else None)
    return (sol.x[:-1] > .5 if sol.x is not None else None), status
