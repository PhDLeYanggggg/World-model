"""Hindsight capacity of fixed forecast choices under recording risk and prefix budgets."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, hstack, vstack

from src.evaluation.m3w_recording_switch_ceiling import BUDGET, moments
from src.evaluation.m3w_recording_forecast_comparison import comparison_masses
from src.evaluation.m3w_selection_forecast_readout import errors, policy
from src.world_model.m3w_causal_record_ledger import EPS


MODES = ('free_capped', 'fixed', 'fixed_terminal', 'fixed_prefix')


def prepare(floor, bank, direct, direct_choice, target, valid, groups, easy_cut, estimated, legal):
    n, k = legal.shape
    order = np.concatenate(groups)
    if len(order) != n or not np.array_equal(np.sort(order), np.arange(n)):
        raise ValueError('Chronological groups must partition the recording')
    sizes = np.asarray([len(g) for g in groups])
    group = np.empty(n, dtype=int)
    group[order] = np.repeat(np.arange(len(groups)), sizes)
    fixed = np.bincount(group[direct_choice >= 0], minlength=len(groups))
    capped = np.minimum(2, sizes-1)
    if (fixed > capped).any() or estimated.shape != (2, n, k) or not np.isfinite(estimated).all():
        raise ValueError('Invalid frozen counts or estimated costs')
    arr = [moments(floor, bank[:, j], target, valid, easy_cut) for j in range(k)]
    names = ('radius', 'all_excess', 'easy_excess', 'easy_signed')
    radius, all_cost, easy_cost, signed = [np.column_stack([m[name] for m in arr]) for name in names]
    known = valid.any(1)
    direct_ade = errors(direct, target, valid)[0]
    by_action = np.column_stack([errors(bank[:, j], target, valid)[0] for j in range(k)])
    weight = 1. / sizes[group]
    # Selecting nothing on a row has a nonzero paired effect when direct switches there.
    offset = np.where(known, direct_ade-arr[0]['base'],
                      -np.linalg.norm(floor-direct, axis=-1).max(1))
    choices = np.where(known[:, None], direct_ade[:, None]-by_action,
                       -np.linalg.norm(bank-direct[:, None], axis=-1).max(2))
    gain = (choices-offset[:, None]) * weight[:, None]
    return dict(group=group, fixed=fixed, capped=capped, ref=arr[0]['ref_easy'],
                reference=float(weight @ arr[0]['base']), offset=float(weight @ offset), gain=gain,
                estimated=estimated, legal=legal & (radius>0), risk=np.stack((all_cost,easy_cost,signed)),
                weight=weight)


def incidence(index, n_rows):
    return coo_array((np.ones(len(index)), (index, np.arange(len(index)))),
                     shape=(n_rows, len(index))).tocsc()


def prefix_matrix(estimated, at, nq):
    """Use a sparse balance recursion, not an action-by-prefix dense matrix."""
    na = len(at)
    event = incidence(at, nq)
    q = np.arange(nq)
    balance = coo_array((np.r_[np.ones(nq), -np.ones(nq-1)],
                         (np.r_[q, q[1:]], np.r_[q, q[:-1]])), shape=(nq,nq)).tocsc()
    blank = csc_array((nq,nq))
    unit = np.max(np.abs(estimated), axis=1).clip(1.)
    blocks = [event.multiply((a/s)[None]).tocsc() for a, s in zip(estimated, unit)]
    assert blocks[0].shape == (nq, na)
    return vstack((hstack((blocks[0], balance, blank)), hstack((blocks[1], blank, balance))), format='csc')


def undominated(legal, gain, costs):
    """Replace dominated actions on the same row without changing support or counts."""
    better = gain[:, :, None] >= gain[:, None, :]
    strictly = gain[:, :, None] > gain[:, None, :]
    for c in costs:
        better &= c[:, :, None] <= c[:, None, :]
        strictly |= c[:, :, None] < c[:, None, :]
    j = np.arange(gain.shape[1])
    tie = j[:,None] < j[None,:]
    return legal & ~(legal[:, :, None] & better & (strictly | tie)).any(axis=1)


def solve(problem, floor, bank, direct, target, valid, recording, easy_cut, hard_cut, mode, seconds):
    if mode not in MODES:
        raise ValueError('Unknown capacity comparison')
    group, fixed, ref = (problem[k] for k in ('group','fixed','ref'))
    cap = problem['capped'] if mode == 'free_capped' else fixed
    free = mode == 'free_capped'
    if not cap.any():
        return dict(status='infeasible', reason='zero_switch_count')
    legal = problem['legal'] & (cap[group,None]>0)
    if not ((ref>0) & legal.any(1)).any():
        return dict(status='infeasible', reason='no_eligible_easy_reference')
    n = len(group)
    costs = problem['risk']
    if mode in ('fixed_terminal', 'fixed_prefix'):
        costs = np.concatenate((costs,problem['estimated']),axis=0)
    legal = undominated(legal,problem['gain'],costs)
    row, action = np.where(legal)
    g = group[row]
    if not free and (np.bincount(group[legal.any(1)], minlength=len(cap)) < cap).any():
        return dict(status='infeasible', reason='not_enough_eligible_rows')
    realized = problem['risk'][:,row,action]
    unit = np.abs(realized).max(1).clip(1.)
    pred = problem['estimated'][:,row,action]
    easy_upper = BUDGET*float(ref.sum())
    if not free:
        # Separate minima can refute joint feasibility; they cannot certify it.
        best = np.where(legal[None], problem['risk'], np.inf).min(2)
        minimum = sum((np.sort(best[:,group==q], axis=1)[:,:cap[q]].sum(1)
                       for q in np.flatnonzero(cap)), start=np.zeros(3))
        for z, upper, name in zip(minimum, (0.,0.,easy_upper), ('all_harm','easy_harm','easy_net')):
            if z > upper+1e-9:
                return dict(status='infeasible', reason=name+'_necessary_bound')
    matrix = vstack((incidence(row,n), incidence(g,len(cap)), csc_array(realized/unit[:,None]),
                     csc_array((ref[row]>0.)[None].astype(float))), format='csc')
    lower = np.r_[np.zeros(n), np.zeros(len(cap)) if free else cap, [-np.inf]*3, 1.]
    upper = np.r_[np.ones(n), cap, 0., 0., easy_upper/unit[2], np.inf]
    slack = 0
    if mode == 'fixed_terminal':
        matrix = vstack((matrix,csc_array(pred/np.abs(pred).max(1).clip(1.)[:,None])), format='csc')
        lower = np.r_[lower, -np.inf, -np.inf]
        upper = np.r_[upper, 0., 0.]
    if mode == 'fixed_prefix':
        slack = 2*len(cap)
        matrix = vstack((hstack((matrix,csc_array((matrix.shape[0],slack)))),
                         prefix_matrix(pred,g,len(cap))), format='csc')
        lower = np.r_[lower, np.zeros(slack)]
        upper = np.r_[upper, np.zeros(slack)]
    value = problem['gain'][row,action]
    objective_unit = max(np.abs(value).max(),1.)
    result = milp(c=np.r_[-value/objective_unit, np.zeros(slack)],
                  integrality=np.r_[np.ones(len(row)), np.zeros(slack)],
                  bounds=Bounds(0.,np.r_[np.ones(len(row)),np.full(slack,np.inf)]),
                  constraints=LinearConstraint(matrix,lower,upper),
                  options=dict(time_limit=seconds,mip_rel_gap=0.,presolve=mode!='fixed_prefix'))
    if result.status == 2:
        return dict(status='infeasible',reason='joint_constraints')
    dual = getattr(result,'mip_dual_bound',None)
    bound = None if dual is None or not np.isfinite(dual) else float(problem['offset']-dual*objective_unit)
    if result.x is None:
        return dict(status='unresolved', reason=f'solver_{result.status}', paired_upper_mass=bound)
    keep = result.x[:len(row)]>.5
    j, a = row[keep],action[keep]
    counts = np.bincount(g[keep],minlength=len(cap))
    if len(np.unique(j)) != len(j) or (counts > cap).any() or not free and not np.array_equal(counts,cap):
        raise ValueError('Returned selection does not respect the registered counts')
    prediction = floor.copy()
    prediction[j] = bank[j,a]
    risk = policy(prediction,floor,floor,target,valid,np.full(n,recording),
                  easy_cut=easy_cut,hard_cut=hard_cut)['recordings'][recording]
    actual = comparison_masses(direct,prediction,floor,target,valid,problem['weight'])
    cv = comparison_masses(floor,prediction,floor,target,valid,problem['weight'])
    expected = float(problem['offset']+value[keep].sum())
    if not np.isclose(actual['known_delta']-actual['unknown_radius'],expected,rtol=1e-8,atol=1e-8):
        raise ValueError('Paired objective differs from the unchanged scorer')
    prefix = np.cumsum(np.stack([np.bincount(g[keep],weights=z[keep],minlength=len(cap))
                                 for z in pred]),axis=1)
    exceeds = mode == 'fixed_prefix' and (prefix>EPS).any()
    exceeds |= mode == 'fixed_terminal' and (prefix[:,-1]>EPS).any()
    if not risk['supported'] or exceeds or risk['selected'] != len(j):
        return dict(status='unresolved',reason='original_postcheck',paired_upper_mass=bound)
    return dict(status='optimal' if result.status==0 else 'feasible_unproven', switches=len(j),
                paired_lower_mass=expected, paired_upper_mass=bound,
                vs_cv_lower_mass=cv['known_delta']-cv['unknown_radius'], risk=risk)
