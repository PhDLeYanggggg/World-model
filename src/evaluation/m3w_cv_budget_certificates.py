"""Cheap cardinality certificates and witness reduction without integer search."""
import numpy as np
from scipy.optimize import minimize

from src.evaluation.m3w_cv_anchor_budget import candidate_terms, risk_excess
from src.evaluation.m3w_cv_joint_feasibility import POSITIVE_PP
from src.evaluation.m3w_selection_forecast_readout import errors


def lower_bound(design):
    """Nonnegative multipliers give lower bounds even without convergence."""
    row = design['row']
    a = np.c_[design['gain'],-design['risk'],design['easy']]
    b = np.r_[POSITIVE_PP-design['offset'],0.,0.,-design['easy_cap'],1.]
    scale = np.maximum(np.abs(a).max(0),np.abs(b)).clip(1.)
    a,b = a/scale,b/scale
    starts = np.r_[0,np.flatnonzero(np.diff(row))+1]
    groups = np.cumsum(np.r_[True,np.diff(row)!=0])-1
    ix = np.arange(len(row))
    best = 1.

    def fun(mult):
        nonlocal best
        z = mult.clip(0.)
        cost = 1.-(a*z).sum(1)
        minimum = np.minimum.reduceat(cost,starts)
        picked = np.minimum.reduceat(np.where(cost==minimum[groups],ix,len(ix)),starts)[minimum<0.]
        dual = float(b@z+minimum.clip(max=0.).sum())
        best = max(best,dual)
        return -dual,a[picked].sum(0)-b

    minimize(fun,np.zeros(len(b)),jac=True,bounds=[(0.,1e5)]*len(b),method='L-BFGS-B',
             options={'maxiter':120,'ftol':1e-10,'maxls':25})
    return max(1,int(np.ceil(best-1e-4)))


def witness_terms(cv,ols,bank,y,v,q,cut,edits):
    i,j = np.asarray(edits).T
    path = bank[i,j]
    _,inv,counts = np.unique(q,return_inverse=True,return_counts=True)
    w = 1./counts[inv]
    mass,offset,*_ = candidate_terms(cv,ols,cv[:,None],y,v)
    ref = w@mass
    _,_,gain,_ = candidate_terms(cv[i],ols[i],path[:,None],y[i],v[i])
    e,_ = errors(cv,y,v)
    p,_ = errors(path,y[i],v[i])
    easy = v.any(1)&(e>0.)&(e<=cut)
    risk = risk_excess(cv[i],path[:,None],y[i],v[i],cut)[:,0]
    signed = np.where(easy[i],p-e[i],0.)
    a = np.c_[gain[:,0]*(100.*w[i]/ref)[:,None],-risk,-signed,easy[i]]
    b = np.r_[POSITIVE_PP,POSITIVE_PP-100.*(w@offset)/ref,0.,0.,-.02*e[easy].sum(),1.]
    return a,b


def prune(terms,required):
    """Delete only actions whose removal preserves every recording constraint."""
    n = len(terms)
    order = np.argsort(terms[:,1],kind='stable')
    rng = np.random.default_rng(2109)
    proposals = [order,np.argsort(terms[:,:2].min(1),kind='stable'),np.arange(n)]
    proposals.extend(rng.permutation(n) for _ in range(4))
    best = np.ones(n,dtype=bool)
    # Keep a strictly positive gain rather than accepting equality at the evaluator threshold.
    floor = required+np.array([1e-8,1e-8,0.,0.,0.,0.])
    for order in proposals:
        total = terms.sum(0)
        keep = np.ones(n,dtype=bool)
        for i in order:
            v = total-terms[i]
            if (v>=floor).all():
                keep[i] = False
                total = v
        if keep.sum()<best.sum():
            best = keep
    return best
