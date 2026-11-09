"""Asymmetric easy-risk regression and lead-resolved completion utility."""
import numpy as np
import torch
from torch.nn import functional as F


def tail_loss(net,x,risk,cat):
    logits = net(x)
    estimate = logits[:,:2]
    error = F.smooth_l1_loss(estimate,risk,reduction='none')
    keep = torch.stack((cat!=1,cat==0),1)
    weight = 1.+2.*(risk[:,1]>estimate[:,1]).to(error.dtype)
    error = error*torch.stack((torch.ones_like(weight),weight),1)
    return ((error*keep).sum(0)/keep.sum(0).clamp_min(1)).sum()+F.cross_entropy(logits[:,2:5],cat)


def observed_targets(cv,action,y,valid,scale):
    y = np.where(valid[...,None],y,0.)
    return (np.linalg.norm(cv-y,axis=-1)-np.linalg.norm(action-y,axis=-1))/scale[:,None]


def benefit_loss(net,x,y,valid):
    err = F.smooth_l1_loss(net(x),y,reduction='none')
    return (err*valid).sum()/valid.sum().clamp_min(1)


def score(cv,ols,path,benefit,visible,scale):
    """Estimate the old mean of two CV-anchored incremental lower utilities."""
    dist = lambda a,b:np.linalg.norm(a-b,axis=-1)
    debit = .5*(-dist(path,cv)+dist(ols,cv)-dist(path,ols))/scale[:,None]
    flat = visible.mean(1,keepdims=True)
    return tuple((p*benefit+(1.-p)*debit).mean(1) for p in (flat,visible))
