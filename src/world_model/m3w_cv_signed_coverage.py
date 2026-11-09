"""Signed risk targets and conservative swaps for CV-anchored edit selection."""
import numpy as np
import torch
from torch.nn import functional as F

from src.evaluation.m3w_cv_anchor_budget import candidate_terms, risk_excess
from src.evaluation.m3w_selection_forecast_readout import errors


def labels(cv, ols, paths, y, valid, scale, easy_cut):
    _, _, g, _ = candidate_terms(cv,ols,paths,y,valid)
    c,_ = errors(cv,y,valid)
    known = valid.any(1)
    easy = known & (c>0.) & (c<=easy_cut)
    risk = risk_excess(cv,paths,y,valid,easy_cut)/scale[:,None,None]
    gain = g/(valid.shape[1]*scale[:,None,None])
    cat = np.where(easy,0,np.where(known,2,1))
    return gain,risk,cat


def risk_loss(net,x,risk,cat,conditional):
    a = net(x)
    delta = F.smooth_l1_loss(a[:,:2],risk,reduction='none')
    if conditional:
        valid = torch.stack((cat!=1,cat==0),dim=1)
        delta = (delta*valid).sum(0)/valid.sum(0).clamp_min(1)
    else:
        delta = delta.mean(0)
    return delta.sum()+F.cross_entropy(a[:,2:5],cat)


def decode_risk(logits,radius,conditional):
    p = logits[:,2:5].softmax(1)
    raw = logits[:,:2]
    if conditional:
        raw = raw*torch.stack((p[:,0]+p[:,2],p[:,0]),1)+p[:,1,None]*radius[:,None]
    return raw,p[:,0]


def signed_swap(cv,p,y,v,s):
    """Same query, choose one of two agents; positive means first agent."""
    clean = np.where(v[...,None],y,0.)
    advantage = np.linalg.norm(cv-clean,axis=-1)-np.linalg.norm(p-clean,axis=-1)
    known = np.where(v,advantage,0.).sum(2)
    width = np.where(v,0.,np.linalg.norm(p-cv,axis=-1)).sum((1,2))
    change = known[:,0]-known[:,1]
    return np.sign(change)*np.maximum(np.abs(change)-width,0.)/(v.shape[2]*s.mean(1))


def gain_loss(net,x,target,preference,scale):
    p = net(x)
    pooled = p.mean(-1)*scale/scale.mean(1,keepdim=True)
    delta = pooled[:,0]-pooled[:,1]
    pair = (preference.abs().clamp(max=1.)*F.softplus(-preference.sign()*delta/.15)).mean()
    return F.smooth_l1_loss(p,target)+.25*pair
