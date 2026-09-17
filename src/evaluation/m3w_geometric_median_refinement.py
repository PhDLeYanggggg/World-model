"""Numerical refinement for registered conditional medians, without held labels."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def refine_geometric_median(points, weights, initial, *, tolerance=1e-8):
    y,w,x = np.asarray(points,float),np.asarray(weights,float),np.asarray(initial,float)
    if (y.ndim!=2 or y.shape[1]!=2 or w.shape!=(len(y),) or x.shape!=(2,)
            or not np.isfinite(y).all() or not np.isfinite(w).all() or not np.isfinite(x).all()
            or np.any(w<0) or w.sum()<=0 or not np.isfinite(tolerance) or tolerance<=0):
        raise ValueError('Finite support, initial point and valid weights required')
    active=w>0
    support,inverse=np.unique(y[active],axis=0,return_inverse=True)
    w=np.bincount(inverse,weights=w[active],minlength=len(support)); w/=w.sum()
    scale=float(np.linalg.norm(support,axis=1).max())
    if scale==0:
        return np.zeros(2),{'converged':True,'method':'zero_support','gap_bound':0.,'objective':0.}
    y,x=support/scale,x/scale
    def quantities(z):
        d=z-y; r=np.linalg.norm(d,axis=1); nz=r>0
        g=(d[nz]*(w[nz]/r[nz])[:,None]).sum(0)
        gap=max(float(np.linalg.norm(g))-float(w[~nz].sum()),0.)*float(r.max())
        return float(w@r),g,gap
    # Nonsmooth optima may be support atoms; test their exact subgradients first.
    for atom in y:
        risk,_,gap=quantities(atom)
        if gap==0:
            return atom*scale,{'converged':True,'method':'certified_support_atom',
                'gap_bound':0.,'objective':risk*scale}
    objective=lambda z:quantities(z)[0]
    derivative=lambda z:quantities(z)[1]
    answer=minimize(objective,x,jac=derivative,method='BFGS',options={'gtol':1e-11,'maxiter':5000})
    risk,_,gap=quantities(answer.x)
    if risk>objective(x)+1e-12:
        raise ValueError('Refinement increased conditional training risk')
    return answer.x*scale,{'converged':bool(gap<=tolerance),'method':'BFGS_after_atom_check',
        'gap_bound':gap*scale,'objective':risk*scale,'optimizer_success':bool(answer.success),
        'optimizer_status':int(answer.status),'iterations':int(answer.nit)}
