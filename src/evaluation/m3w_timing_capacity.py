"""Hindsight retiming under a matched recording total, plus a past-only counterfactual."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_array, hstack, vstack

from src.evaluation import m3w_prefix_capacity as base
from src.world_model import m3w_causal_record_ledger as ledger

MODES = ('total', 'total_terminal', 'total_prefix')


def retime(groups, value, a, e, legal, mint, *, carry):
    """Earn switch slots only from the current/past direct decisions."""
    choice = np.full(len(value), -1, dtype=np.int16)
    credit, balance = 0, np.zeros(2)
    for g in groups:
        credit = (credit if carry else 0) + int((mint[g]>=0).sum())
        if credit == 0:
            continue
        u, ac, ec, ok = (x[g] for x in (value, a, e, legal))
        if credit == 1 and len(g)>1:
            p = ledger.query_choice(u, ac, ec, ok, balance, count=1)
        else:
            p = ledger.query_choice(u, ac, ec, ok, balance)
        if p is None:
            continue
        total = ledger.totals(p, u, ac, ec)
        if total[0]<=0.:
            continue
        balance -= total[1:]
        credit -= int((p>=0).sum())
        choice[g] = p
    return choice


def solve(p, floor, bank, direct, target, valid, recording, easy_cut, hard_cut, mode, seconds):
    if mode not in MODES:
        raise ValueError('Unknown recording-total comparison')
    group, cap, ref = p['group'], p['capped'], p['ref']
    total = int(p['fixed'].sum())
    if total == 0:
        return dict(status='infeasible',reason='zero_recording_total')
    legal = p['legal'] & (cap[group,None]>0)
    if not ((ref>0) & legal.any(1)).any():
        return dict(status='infeasible',reason='no_eligible_easy_reference')
    costs = p['risk'] if mode=='total' else np.concatenate((p['risk'],p['estimated']))
    legal = base.undominated(legal,p['gain'],costs)
    j,k = np.where(legal)
    q = group[j]
    n, nq, nz = len(group), len(cap), len(j)
    raw, pred = p['risk'][:,j,k], p['estimated'][:,j,k]
    unit = np.abs(raw).max(1).clip(1.)
    # One equality fixes the recording total without fixing any query's count.
    mat = vstack((base.incidence(j,n),base.incidence(q,nq),
                  csc_array(np.ones((1,nz))),csc_array(raw/unit[:,None]),
                  csc_array((ref[j]>0.)[None].astype(float))),format='csc')
    lo = np.r_[np.zeros(n+nq),total,[-np.inf]*3,1.]
    hi = np.r_[np.ones(n),cap,total,0.,0.,base.BUDGET*ref.sum()/unit[2],np.inf]
    slack = 0
    if mode=='total_terminal':
        mat = vstack((mat,csc_array(pred/np.abs(pred).max(1).clip(1.)[:,None])),format='csc')
        lo, hi = np.r_[lo,-np.inf,-np.inf], np.r_[hi,0.,0.]
    if mode=='total_prefix':
        slack = 2*nq
        mat = vstack((hstack((mat,csc_array((mat.shape[0],slack)))),
                      base.prefix_matrix(pred,q,nq)),format='csc')
        lo, hi = np.r_[lo,np.zeros(slack)], np.r_[hi,np.zeros(slack)]
    value = p['gain'][j,k]
    scale = max(1.,np.abs(value).max())
    sol = milp(c=np.r_[-value/scale,np.zeros(slack)],
               integrality=np.r_[np.ones(nz),np.zeros(slack)],
               bounds=Bounds(0.,np.r_[np.ones(nz),np.full(slack,np.inf)]),
               constraints=LinearConstraint(mat,lo,hi),
               options=dict(time_limit=seconds,mip_rel_gap=0.,presolve=False))
    if sol.status==2:
        return dict(status='infeasible',reason='joint_constraints')
    dual = getattr(sol,'mip_dual_bound',None)
    bound = None if dual is None or not np.isfinite(dual) else float(p['offset']-dual*scale)
    if sol.x is None:
        return dict(status='unresolved',reason=f'solver_{sol.status}',paired_upper_mass=bound)
    use = sol.x[:nz]>.5
    rows, acts = j[use],k[use]
    counts = np.bincount(q[use],minlength=nq)
    if len(rows)!=total or len(np.unique(rows))!=total or (counts>cap).any():
        raise ValueError('Recording total, query cap or row exclusion violated')
    proposal = floor.copy()
    proposal[rows] = bank[rows,acts]
    risk = base.policy(proposal,floor,floor,target,valid,np.full(n,recording),
                       easy_cut=easy_cut,hard_cut=hard_cut)['recordings'][recording]
    prefix = np.cumsum(np.stack([np.bincount(q[use],weights=z[use],minlength=nq)
                                 for z in pred]),axis=1)
    invalid = mode=='total_prefix' and (prefix>ledger.EPS).any()
    invalid |= mode=='total_terminal' and (prefix[:,-1]>ledger.EPS).any()
    if invalid or not risk['supported'] or risk['selected']!=total:
        return dict(status='unresolved',reason='original_postcheck',paired_upper_mass=bound)
    actual = base.comparison_masses(direct,proposal,floor,target,valid,p['weight'])
    lower = actual['known_delta']-actual['unknown_radius']
    if not np.isclose(lower,p['offset']+value[use].sum(),rtol=1e-8,atol=1e-8):
        raise ValueError('Expected paired objective must agree with original scoring')
    return dict(status='optimal' if sol.status==0 else 'feasible_unproven', switches=total,
                paired_lower_mass=lower,paired_upper_mass=bound,risk=risk,
                retimed_switches=int(np.maximum(counts-p['fixed'],0).sum()))
