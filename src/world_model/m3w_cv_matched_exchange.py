"""One exchange per causal query at identical peak intervention mass."""
import numpy as np
import torch
from torch.nn import functional as F

from src.evaluation.m3w_cv_anchor_budget import candidate_terms, risk_excess
from src.evaluation.m3w_selection_forecast_readout import errors
from src.world_model.m3w_cv_fixed_recipient_support import peak
from src.world_model.m3w_cv_joint_selector import pair_context
from src.world_model.m3w_cv_minimum_budget import action_features


def shortlist(cv,ols,anchor,selected,groups):
    gap = peak(ols-cv)
    r = peak(anchor-cv)
    credit = (np.linalg.norm(ols-cv,axis=-1)-np.linalg.norm(ols-anchor,axis=-1)).sum(1)
    outgoing,incoming,query = [],[],[]
    for k,g in enumerate(groups):
        s,u = g[selected[g]],g[~selected[g]]
        s = s[np.argsort(credit[s],kind='stable')[:4]]
        u = u[np.argsort(-gap[u],kind='stable')[:8]]
        i,j = np.where((gap[u][None]>=r[s,None]) & (r[s,None]>0.))
        outgoing.extend(s[i]); incoming.extend(u[j]); query.extend(np.full(len(i),k))
    return tuple(np.asarray(x,dtype=int) for x in (outgoing,incoming,query))


def incoming_path(cv,ols,radius):
    delta = ols-cv
    r = peak(delta)
    if (radius>r).any() or (radius<=0.).any():
        raise ValueError('Cannot amplify or use a trivial exchange path')
    return cv+(radius/r)[:,None,None]*delta


def features(past,h,cv,ols,scale,old,new,i,j):
    ctx,_,_ = pair_context(h[i],h[j],cv[i],cv[j])
    return np.c_[action_features(past[i],cv[i],ols[i],old,scale[i]),
        action_features(past[j],cv[j],ols[j],new,scale[j]),np.arcsinh(ctx)].astype(np.float32)


def targets(c,o,path,y,v,cut):
    _,_,g,_ = candidate_terms(c,o,path[:,None],y,v)
    r = risk_excess(c,path[:,None],y,v,cut)[:,0]
    e,_ = errors(c,y,v)
    p,_ = errors(path,y,v)
    easy = v.any(1)&(e>0.)&(e<=cut)
    return np.c_[g[:,0]/12.,r,np.where(easy,p-e,0.)]


def objective(net,x,y):
    z = net(x)
    return F.smooth_l1_loss(z,y)+.2*F.smooth_l1_loss(z[:,:2].amin(1),y[:,:2].amin(1))


def decode(score,q,threshold,eligible=None):
    if eligible is not None:
        score = np.where(eligible,score,-np.inf)
    chosen = []
    start = np.r_[0,np.flatnonzero(np.diff(q))+1,len(q)]
    for l,r in zip(start[:-1],start[1:],strict=True):
        if l==r:
            continue
        i = l+int(score[l:r].argmax())
        if score[i]>threshold:
            chosen.append(i)
    return np.asarray(chosen,dtype=int)


def apply(cv,anchor,i,j,path,pick):
    z = anchor.copy()
    s,t = i[pick],j[pick]
    if len(np.unique(np.r_[s,t]))!=2*len(s):
        raise ValueError('Overlapping exchange')
    z[s],z[t] = cv[s],path[pick]
    np.testing.assert_allclose(peak(z[t]-cv[t]),peak(anchor[s]-cv[s]),rtol=1e-6,atol=1e-8)
    if (z!=cv).any((1,2)).sum()!=(anchor!=cv).any((1,2)).sum():
        raise ValueError('Exchange changed intervention count')
    return z
