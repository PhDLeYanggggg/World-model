"""Bound the minimum number of full-bank edits under the original recording rule."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_array, vstack

from src.evaluation.m3w_cv_budget_menu_factorial import incidence
from src.evaluation.m3w_cv_joint_feasibility import POSITIVE_PP


def solve(d, ceiling, seconds):
    n = len(d['row'])
    if not n or not d['easy'].any() or d['ref']<=0.:
        return None,dict(status='undefined_support',lower=None)
    scale = np.abs(d['risk']).max(0).clip(1.)
    mat = vstack((incidence(d['row'],len(d['group'])),
                  csc_array(d['risk'].T/scale[:,None]),
                  csc_array(d['easy'][None].astype(float)),
                  csc_array(d['gain'].T),
                  csc_array(np.ones((1,n)))),format='csc')
    lo = np.r_[np.zeros(len(d['group'])),-np.inf,-np.inf,-np.inf,1.,
               POSITIVE_PP-d['offset'],0.]
    hi = np.r_[np.ones(len(d['group'])),0.,0.,d['easy_cap']/scale[2],np.inf,
               np.inf,np.inf,ceiling]
    result = milp(np.ones(n),integrality=np.ones(n),bounds=Bounds(np.zeros(n),np.ones(n)),
        constraints=LinearConstraint(mat,lo,hi),options={'time_limit':seconds,'mip_rel_gap':.005})
    dual = getattr(result,'mip_dual_bound',None)
    lower = max(1,int(np.ceil(dual-1e-5))) if dual is not None and np.isfinite(dual) else 1
    return (np.flatnonzero(result.x>.5) if result.x is not None else None),dict(
        status='infeasible' if result.status==2 else 'bounded',lower=lower,solver_status=int(result.status))


def verified_plan(d, known, verify, seconds):
    """Retain a verified incumbent; a solver status alone is not a witness."""
    upper = None
    if known is not None:
        proof = verify(known)
        if not proof['verified']:
            raise ValueError('Invalid inherited recording witness')
        upper = dict(actions=np.asarray(known).tolist(),edits=proof['edits'],gain_pp=proof['gain_pp'])
    z,log = solve(d,upper['edits'] if upper else len(d['group']),seconds)
    if z is not None:
        a = np.column_stack((d['row'][z],d['arm'][z]))
        proof = verify(a)
        if proof['verified'] and (upper is None or proof['edits']<=upper['edits']):
            upper = dict(actions=a.tolist(),edits=proof['edits'],gain_pp=proof['gain_pp'])
    if upper and (log['status']=='infeasible' or log['lower']>upper['edits']):
        raise ValueError('Count bound contradicts a verified incumbent')
    return dict(**log,incumbent=upper)
