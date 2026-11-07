"""Future-label-only ceiling under the original recording risk contract."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, vstack

from src.evaluation.m3w_selection_forecast_readout import errors, policy

BUDGET = .02


def moments(floor, forecast, target, valid, easy_cut):
    cv, _ = errors(floor,target,valid); ne, _ = errors(forecast,target,valid)
    known = valid.any(1); easy = known & (cv>0) & (cv<=easy_cut)
    radius = np.linalg.norm(forecast-floor,axis=-1).max(1)
    base = np.where(known,cv,0.); diff = np.where(known,cv-ne,0.)
    harm = np.maximum(-diff,0.); ref_easy = np.where(easy,base,0.)
    unknown = np.where(known,0.,radius)
    return dict(known=known,easy=easy,radius=radius,base=base,ref_easy=ref_easy,diff=diff,
        utility=diff-unknown, all_excess=harm+unknown-BUDGET*base,
        easy_excess=np.where(easy,harm,0.)+unknown-BUDGET*ref_easy,
        easy_signed=np.where(easy,-diff,0.))


def free_count(m, weight):
    """All nonharmful rows are useful, plus at most one necessary easy anchor."""
    legal = m['radius']>0
    take = legal & m['known'] & (m['diff']>=0)
    if (take & m['easy']).any():
        return take,'analytic_optimum'
    anchors = np.flatnonzero(legal & m['easy'] & (m['easy_excess']<=0))
    if not len(anchors):
        return None,'no_positive_reference_easy_action_within_2_percent'
    best = anchors[np.argmax(weight[anchors]*m['diff'][anchors])]
    take[best] = True
    return take,'analytic_optimum'


def matched_count(m, query, weight, seconds=15.):
    """Test the old per-query count, not a newly claimed risk-equivalent control."""
    _, group, sizes = np.unique(query,return_inverse=True,return_counts=True)
    limit = np.minimum(2,sizes-1)
    n = len(query)
    equal = coo_array((np.ones(n),(group,np.arange(n))),shape=(len(sizes),n)).tocsc()
    risk = np.stack((m['all_excess'],m['easy_excess'],m['easy_signed']))
    scale = np.max(abs(risk),axis=1); scale = np.where(scale>0,scale,1.)
    ref = m['ref_easy']>0
    mat = vstack((equal,csc_array(risk/scale[:,None]),csc_array(ref[None].astype(float))),format='csc')
    upper = np.r_[limit,0.,0.,BUDGET*m['ref_easy'].sum()/scale[2],np.inf]
    lower = np.r_[limit,-np.inf,-np.inf,-np.inf,1.]
    value = weight*m['utility']; value_scale = max(abs(value).max(),1.)
    sol = milp(c=-value/value_scale,integrality=np.ones(n),bounds=Bounds(0.,(m['radius']>0).astype(float)),
        constraints=LinearConstraint(mat,lower,upper),options=dict(time_limit=seconds,mip_rel_gap=0.))
    bound = getattr(sol,'mip_dual_bound',None)
    upper_mass = float(-bound*value_scale) if bound is not None and np.isfinite(bound) else None
    result = dict(solver_status=int(sol.status),optimal=bool(sol.status==0),
        infeasible=bool(sol.status==2),objective_upper_mass=upper_mass)
    if sol.x is None: return None,result
    take = sol.x>.5
    if not np.array_equal(np.bincount(group[take],minlength=len(limit)),limit) or not (take&ref).any():
        return None,dict(result,postcheck_count_or_support_failed=True,optimal=False)
    return take,result


def inspect(floor,forecast,target,valid,query,recording,easy_cut,hard_cut,seconds=15.):
    m = moments(floor,forecast,target,valid,easy_cut)
    _, inv, count = np.unique(query,return_inverse=True,return_counts=True); w = 1./count[inv]
    ref = float(w@m['base'])
    free, reason = free_count(m,w)
    match, solver = matched_count(m,query,w,seconds)
    def summarize(take):
        if take is None: return None
        actual = policy(np.where(take[:,None,None],forecast,floor),floor,floor,target,valid,
            np.full(len(take),recording),easy_cut=easy_cut,hard_cut=hard_cut)['recordings'][recording]
        return dict(supported=actual['supported'], switched=int(take.sum()),
            completion_gain_mass=float(w[take]@m['utility'][take]), risk=actual)
    a,b = summarize(free),summarize(match)
    if a is not None and not a['supported']:
        raise ValueError('Analytic free-count witness fails the exact original risk checks')
    if b is not None and not b['supported']:
        solver.update(optimal=False,postcheck_original_risk_failed=True)
    if a is not None and b is not None and b['supported'] and b['completion_gain_mass']>a['completion_gain_mass']+1e-9:
        raise AssertionError('A matched feasible policy cannot exceed the free-count ceiling')
    return dict(rows=len(query),queries=len(count),known_rows=int(m['known'].sum()),
        known_easy_rows=int(m['easy'].sum()),reference_mass=ref,
        matched_required_switches=int(np.minimum(2,count-1).sum()),free_count=a,free_reason=reason,
        matched_count=b,solver=solver,label_aware_only=True,choices_exported=False)
