"""Factorial hindsight capacity under common CV-anchored recording constraints."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_array, csc_array, vstack

from src.evaluation.m3w_cv_anchor_budget import candidate_terms, risk_excess, row_ranks
from src.evaluation.m3w_cv_joint_feasibility import POSITIVE_PP, coefficients
from src.evaluation.m3w_selection_forecast_readout import errors


def incidence(group,n):
    return coo_array((np.ones(len(group)),(group,np.arange(len(group)))),shape=(n,len(group)))


def optimize(discrete,lo,hi,gain,risk,easy,cap,offset,seconds):
    """Return an incumbent or a conservative negative/undefined status."""
    n = len(gain)
    if not n or not np.any(easy):
        return None,'undefined_support',None
    scale = np.abs(risk).max(0).clip(1.)
    risk = csc_array(np.column_stack((risk.T/scale[:,None],np.zeros(3))))
    support = csc_array(np.r_[easy,0.][None])
    utility = csc_array(np.column_stack((-gain.T,np.ones(2))))
    a = csc_array((discrete.data,(discrete.row,discrete.col)),shape=(discrete.shape[0],n+1))
    mat = vstack((a,risk,support,utility),format='csc')
    lhs = np.r_[lo,-np.inf,-np.inf,-np.inf,1.,-np.inf,-np.inf]
    rhs = np.r_[hi,0.,0.,cap/scale[2],np.inf,offset]
    # Finding any independently verifiable positive incumbent is sufficient.
    result = milp(np.r_[np.zeros(n),-1.],integrality=np.r_[np.ones(n),0.],
        bounds=Bounds(np.r_[np.zeros(n),-np.inf],np.r_[np.ones(n),np.inf]),
        constraints=LinearConstraint(mat,lhs,rhs),
        options={'time_limit':seconds,'mip_rel_gap':.02,'objective_target':-1e-4})
    d = getattr(result,'mip_dual_bound',None)
    upper = -float(d) if d is not None and np.isfinite(d) else None
    if result.status==2:
        return None,'risk_infeasible',upper
    if upper is not None and upper < -POSITIVE_PP:
        return None,'nonpositive_bound',upper
    return (np.flatnonzero(result.x[:-1]>.5) if result.x is not None else None),'unresolved',upper


def full(c,o,b,y,v,q,easy,anchor,fixed,seconds):
    d = coefficients(c,o,b,y,v,q,easy)
    if d['ref']<=0.:
        return None,'undefined_support',None
    quotas = np.bincount(d['group'][anchor>=0],minlength=len(d['counts']))
    a = np.flatnonzero(quotas[d['group'][d['row']]]>0) if fixed else np.arange(len(d['row']))
    row = d['row'][a]
    if fixed and (np.bincount(d['group'][np.unique(row)],minlength=len(quotas))<quotas).any():
        return None,'risk_infeasible',None
    r = incidence(row,len(c))
    if fixed:
        g = incidence(d['group'][row],len(quotas))
        r = vstack((r,g),format='coo')
    take,status,upper = optimize(r,np.r_[np.zeros(len(c)),quotas] if fixed else np.zeros(len(c)),
        np.r_[np.ones(len(c)),quotas] if fixed else np.ones(len(c)),d['gain'][a],d['risk'][a],d['easy'][a],
        d['easy_cap'],d['offset'],seconds)
    return (np.column_stack((d['row'][a[take]],d['arm'][a[take]])) if take is not None else None),status,upper


def prefix_orders(c,o,b):
    radius = np.linalg.norm(b-c[:,None],axis=-1)
    legal = radius.max(2)>1e-8
    agreement = np.linalg.norm(c-o,axis=-1)[:,None]-np.linalg.norm(b-o[:,None],axis=-1)
    scores = [agreement.mean(2)-t*radius.mean(2) for t in (0.,.25,.75,1.25)]
    scores.append(agreement[...,8:].mean(2)-.25*radius[...,:4].mean(2))
    for score in scores:
        s = np.where(legal,score,-np.inf)
        arm = s.argmax(1)
        value = s[np.arange(len(s)),arm]
        rows = np.argsort(-value,kind='stable')
        rows = rows[np.isfinite(value[rows])]
        yield rows,arm[rows]


def narrow(c,o,b,y,v,q,easy_cut,anchor,fixed,seconds):
    _,inv,counts = np.unique(q,return_inverse=True,return_counts=True)
    w = 1./counts[inv]
    mass,delta,gain,_ = candidate_terms(c,o,b,y,v)
    ref = float(w@mass)
    if ref<=0.:
        return None,'undefined_support',None
    cv_error,_ = errors(c,y,v)
    easy = v.any(1)&(cv_error>0.)&(cv_error<=easy_cut)
    signed = np.column_stack([np.where(easy,errors(p,y,v)[0]-cv_error,0.) for p in b.swapaxes(0,1)])
    terms = np.concatenate((gain*(100.*w/ref)[:,None,None],risk_excess(c,b,y,v,easy_cut),
        signed[...,None],np.broadcast_to(easy[:,None,None],(*signed.shape,1))),axis=-1)
    order,off = row_ranks(q)
    blocks,choices = [],[]
    for a,z in zip(off[:-1],off[1:],strict=True):
        g = order[a:z]
        active = anchor[g]>=0
        k = int(active.sum())
        if fixed and k==0:
            continue
        edits = [(g[active],anchor[g][active],k)] if k else []
        values = [terms[edits[0][0],edits[0][1]].sum(0)] if k else []
        if not fixed:
            edits.append((g[:0],anchor[g[:0]],0))
            values.append(np.zeros(6))
        for row,arm in prefix_orders(c[g],o[g],b[g]):
            row = g[row]
            n = len(row)
            if fixed and n<k:
                raise ValueError('Cannot reproduce the fixed-count narrow menu')
            totals = terms[row,arm].cumsum(0)
            if fixed:
                values.append(totals[k-1])
                edits.append((row,arm,k))
            else:
                values.extend(totals)
                edits.extend((row,arm,j) for j in range(1,n+1))
        blocks.append(np.stack(values))
        choices.extend(edits)
    if not blocks:
        return None,'undefined_support',None
    u = np.concatenate(blocks)
    groups = incidence(np.repeat(np.arange(len(blocks)),[len(x) for x in blocks]),len(blocks))
    ones = np.ones(len(blocks))
    take,status,upper = optimize(groups,ones,ones,u[:,:2],u[:,2:5],u[:,5],.02*cv_error[easy].sum(),
                                 np.array([0.,100.*(w@delta)/ref]),seconds)
    if take is None:
        return None,status,upper
    edits = [np.column_stack((r[:k],arm[:k])) for r,arm,k in (choices[i] for i in take)]
    return np.concatenate(edits),status,upper
