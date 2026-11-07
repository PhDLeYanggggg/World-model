"""Recording-level capacity with separable recipient choices and fixed query caps."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, vstack

from src.evaluation.m3w_joint_slate_capacity import BUDGET, comparison_masses, errors, moments, policy


def solve(cv, ols, bank, y, valid, offsets, limit, allowed, recording, easy_cut, hard_cut, seconds=4.):
    counts = np.diff(offsets)
    if (bank.ndim != 4 or bank.shape[0] != len(cv) or bank.shape[2:] != cv.shape[1:]
            or allowed.shape != bank.shape[:2] or offsets[0] != 0 or offsets[-1] != len(cv)
            or len(limit) != len(counts) or np.any(counts < 1)
            or np.any((limit < 0) | (limit > np.minimum(2, counts-1)))):
        raise ValueError('Misaligned recipient budget')
    query = np.repeat(np.arange(len(counts)), counts)
    weights = 1./counts[query]
    reference, _ = errors(cv, y, valid)
    easy = valid.any(1) & (reference > 0.) & (reference <= easy_cut)
    baseline = comparison_masses(ols, cv, cv, y, valid, weights)
    offset = baseline['known_delta']-baseline['unknown_radius']
    row, arm = np.nonzero(allowed & (limit[query, None] > 0))
    m = moments(cv[row], bank[row, arm], y[row], valid[row], easy_cut)
    best = np.full(len(cv), -np.inf)
    legal = m['radius'] > 0.
    # For one known recipient, the minimum-error legal path weakly
    # dominates all others in every risk constraint and in utility.
    np.maximum.at(best, row[legal], m['diff'][legal])
    keep = np.flatnonzero(legal & (~m['known'] | (m['diff'] == best[row])))
    if keep.size:
        first = np.r_[True, np.diff(row[keep]) != 0]
        keep = keep[first | ~m['known'][keep]]
    row, arm = row[keep], arm[keep]
    n = len(row)
    base = dict(feasible=False, positive=False, reference_mass=baseline['reference'],
                decisions=n, easy_rows=int(easy.sum()))
    support = (m['ref_easy'][keep] > 0.).astype(float)
    if not n or not support.any():
        return dict(base, status='no_selected_easy_candidate', proved_infeasible=True, optimum_proved=False)
    risk = np.column_stack([m[k][keep] for k in ('all_excess', 'easy_excess', 'easy_signed')])
    unknown = ~m['known'][keep]
    before = np.linalg.norm(cv[row]-ols[row], axis=-1).max(1)
    after = np.linalg.norm(bank[row, arm]-ols[row], axis=-1).max(1)
    benefit = weights[row]*np.where(unknown, before-after, m['diff'][keep])
    # Known rows now have one alternative. Only rows with multiple
    # remaining alternatives need a separate mutual-exclusion constraint.
    multiple = np.bincount(row, minlength=len(cv))[row] > 1
    duplicates, repeat = np.unique(row[multiple], return_inverse=True)
    q = query[row]
    columns = np.arange(n)
    membership = coo_array((np.ones(n), (q, columns)), shape=(len(counts), n))
    exclusive = coo_array((np.ones(len(repeat)), (repeat, columns[multiple])), shape=(len(duplicates), n))
    unit = np.abs(risk).max(0).clip(1.)
    matrix = vstack((membership, exclusive, csc_array((risk/unit).T), csc_array(support[None])), format='csc')
    upper = np.r_[limit, np.ones(len(duplicates)), 0., 0., BUDGET*reference[easy].sum()/unit[2], np.inf]
    lower = np.r_[np.full(len(upper)-1, -np.inf), 1.]
    scale = max(1., np.abs(benefit).max())
    sol = milp(-benefit/scale, integrality=np.ones(n), bounds=Bounds(0., 1.),
               constraints=LinearConstraint(matrix, lower, upper),
               options=dict(time_limit=seconds, mip_rel_gap=0.))
    dual = getattr(sol, 'mip_dual_bound', None)
    out = dict(base, status={0:'optimal', 1:'timed', 2:'infeasible'}.get(sol.status, 'unresolved'),
               optimum_proved=bool(sol.status == 0), proved_infeasible=bool(sol.status == 2),
               upper_mass=float(offset-dual*scale) if dual is not None and np.isfinite(dual) else None)
    if sol.x is None:
        return out
    take = sol.x > .5
    picked = row[take]
    if len(np.unique(picked)) != len(picked) or np.any(np.bincount(q[take], minlength=len(counts)) > limit):
        return dict(out, status='unresolved_integrality', optimum_proved=False)
    pred = cv.copy()
    pred[picked] = bank[picked, arm[take]]
    check = policy(pred, cv, cv, y, valid, np.full(len(cv), recording),
                   easy_cut=easy_cut, hard_cut=hard_cut)['recordings'][recording]
    if not check['supported'] or check['selected'] != len(picked):
        return dict(out, status='unresolved_original_risk_recheck', optimum_proved=False)
    scored = comparison_masses(ols, pred, cv, y, valid, weights)
    gain = scored['known_delta']-scored['unknown_radius']
    np.testing.assert_allclose(gain, offset+benefit[take].sum(), rtol=1e-10, atol=1e-7)
    return dict(out, feasible=True, positive=gain > 0., witness_mass=gain,
                switches=len(picked), unknown_switches=check['unknown_selected'])
