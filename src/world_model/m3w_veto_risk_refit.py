"""Nested donor residuals for causal predicted-budget vetoes."""
import numpy as np

from src.world_model import m3w_causal_record_ledger as ledger


def prefix(groups,a,e,choice):
    balance = np.zeros(2)
    before = np.empty((len(choice),2))
    for g in groups:
        before[g] = balance
        costs = ledger.totals(choice[g],a[g],a[g],e[g])[1:]
        if (costs>balance+ledger.EPS).any():
            raise ValueError('Original predicted budget cannot be overdrawn')
        balance -= costs
    return before


def descriptors(x,scale,a,e,slack,legal):
    cost = np.stack((a,e),axis=-1)
    v = np.arcsinh(np.clip(cost/scale[:,None,None],-100.,100.))
    s = np.arcsinh(np.clip(slack/scale[:,None],-100.,100.))
    s = np.broadcast_to(s[:,None],v.shape)
    veto = legal & (cost>slack[:,None]+ledger.EPS).any(-1)
    return np.concatenate((x,v,s),axis=-1),veto


def moments(x,y,use,weights):
    k,d = x.shape[1:]
    g = np.zeros((k,d,d))
    b = np.zeros((k,d,2))
    for j in range(k):
        ix = use[:,j]
        w,z = weights[ix],x[ix,j]
        if len(w):
            w = w/w.sum()
            g[j] = z.T@(w[:,None]*z)
            b[j] = z.T@(w[:,None]*y[ix,j])
    return g,b


def coefficients(moments):
    g,b = (np.mean(v,axis=0) for v in zip(*moments))
    return np.linalg.solve(g+.002*np.eye(g.shape[-1]),b)


def reopen(x,coef,scale,radius,a,e,slack,veto,witness):
    base = np.stack((a,e),axis=-1)
    limit = radius+.02*scale[:,None]
    delta = np.einsum('nad,adc->nac',x,coef)*scale[:,None,None]
    proposed = base+np.clip(delta,-limit[...,None],limit[...,None])
    released = veto & (proposed<=slack[:,None]+ledger.EPS).all(-1)
    r = np.flatnonzero(witness>=0)
    released[r,witness[r]] = False
    cost = np.where(released[...,None],proposed,base)
    return cost[...,0],cost[...,1],released
