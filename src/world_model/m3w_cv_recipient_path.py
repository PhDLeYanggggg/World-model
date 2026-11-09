"""Independent recipient scores and displacement-limited within-row path ranking."""
import numpy as np
import torch
from torch.nn import functional as F


def recipient_features(past,cv,ols,scale):
    delta = np.log1p(np.linalg.norm(ols-cv,axis=-1)/scale[:,None])
    return np.c_[past,delta].astype(np.float32)


def value(z):
    return z[...,:2].min(-1)-z[...,2:5].clip(0.).sum(-1)


def paired_loss(net,x,target):
    y = net(x)[...,:5]
    base = F.smooth_l1_loss(y,target)
    estimate = y[...,:2].amin(-1)-y[...,2:5].clamp_min(0.).sum(-1)
    realized = target[...,:2].amin(-1)-target[...,2:5].clamp_min(0.).sum(-1)
    margin = realized[:,0]-realized[:,1]
    pair = F.softplus(-margin.sign()*(estimate[:,0]-estimate[:,1]))
    weight = margin.abs().clamp(max=2.)
    return base+.2*(pair*weight).sum()/weight.sum().clamp_min(1.)


def nominations(cost,new,legal,radius):
    a = np.where(legal,value(cost),-np.inf)
    old = a.argmax(1)
    row = np.arange(len(old))
    # The frozen action always remains in the candidate set.
    support = legal & (radius<=radius[row,old,None]+1e-9)
    path = np.where(support,value(new),-np.inf).argmax(1)
    return a[row,old],old,path


def select(rank,arm,groups,scale,*,fraction=.25,margin=0.,match=None):
    score = rank*scale
    out = np.full(len(arm),-1,np.int8)
    for q in groups:
        n = int((match[q]>=0).sum()) if match is not None else max(1,int(np.floor(fraction*len(q))))
        at = q[np.argsort(-score[q],kind='stable')[:n]]
        at = at[np.isfinite(score[at])]
        if match is None:
            at = at[rank[at]>margin]
        elif len(at)!=n:
            raise ValueError('Exact query count unavailable')
        out[at] = arm[at]
    return out
