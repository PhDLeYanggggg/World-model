"""Separate future observation support from conditional OLS4 replacement gain."""
import numpy as np
import torch
from torch import nn

from src.world_model import m3w_ols_replacement as prior


AMPLITUDES = (.25, .5, 1.)
ARMS = ('direct', 'factorized_ego', 'factorized_neighbor')
NEIGHBOR_WIDTH = 6
EGO_WIDTH = 8


def neighbor_windows(history, groups):
    h = np.asarray(history)
    neighbors = np.zeros((len(h), 8, 8, 2), np.float32)
    valid = np.zeros((len(h), 8), bool)
    for q in groups:
        k = min(8, len(q)-1)
        if not k:
            continue
        origin = h[q, -1]
        dist = np.square(origin[:,None]-origin[None]).sum(2)
        np.fill_diagonal(dist,np.inf)
        nearest = np.argsort(dist,axis=1,kind='stable')[:,:k]
        neighbors[q,:k] = h[q[nearest]]-origin[:,None,None]
        valid[q,:k] = True
    return neighbors,valid


def neighborhood(h, n, mask, scale):
    if n.shape != (len(h),8,8,2) or mask.shape != (len(h),8):
        raise ValueError('Expected up to eight aligned past neighbors')
    xy = n[:,:,-1]/scale[:,None,None]
    dist = np.linalg.norm(xy,axis=2)
    size = mask.sum(1)
    den = np.maximum(size,1)
    med_speed = (h[:,-1]-h[:,-4])/(3*scale[:,None])
    local_speed = (n[:,:,-1]-n[:,:,-4])/(3*scale[:,None,None])
    rel = local_speed-med_speed[:,None]
    center = (xy*mask[...,None]).sum(1)/den[:,None]
    radius = np.linalg.norm(center,axis=1)
    away = -(center*med_speed).sum(1)/np.maximum(radius,1e-6)
    nearest = np.min(np.where(mask,dist,np.inf),axis=1)
    return np.column_stack((size/8.,np.log1p(np.where(size,nearest,0.)),
        np.log1p((dist*mask).sum(1)/den),np.log1p(radius),away,
        (np.linalg.norm(rel,axis=2)*mask).sum(1)/den)).clip(-16.,16.).astype(np.float32)


def design(h,n,mask):
    x,cv,ols,bank,scale = prior.features(h)
    z = neighborhood(h,n,mask,scale)
    a = np.asarray(AMPLITUDES,dtype=float)
    delta = (bank-ols[:,None])[:,:,None]*a[None,None,:,None,None]
    bank = ols[:,None]+delta.reshape(len(h),-1,12,2)
    x = np.repeat(x,len(a),axis=1)
    amplitude = np.broadcast_to(np.tile(a,len(prior.NAMES))[None,:,None],(*x.shape[:2],1))
    features = np.concatenate((x,amplitude),axis=2).astype(np.float32)
    debit = np.linalg.norm(delta.reshape(len(h),-1,12,2),axis=3).max(2)/(12.*scale[:,None])
    return features,z,cv,ols,bank,scale,debit.astype(np.float32)


class GainSupport(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.cost = prior.network(width+NEIGHBOR_WIDTH)
        self.support = nn.Sequential(nn.Linear(EGO_WIDTH+NEIGHBOR_WIDTH,24),nn.SiLU(),nn.Linear(24,1))

    def forward(self, x, neighbor, *, ego_support):
        z = neighbor[:,None].expand(-1,x.shape[1],-1)
        cost = prior.prediction(self.cost,torch.cat((x,z),dim=-1))
        inp = torch.cat((x[:,0,:EGO_WIDTH],torch.zeros_like(neighbor) if ego_support else neighbor),dim=1)
        return cost,self.support(inp).squeeze(1)


def objective(cost, logit, target, missing, *, factorized):
    fit = torch.nn.functional.smooth_l1_loss(cost,target,beta=.05,reduction='none')
    if factorized:
        known = ~missing
        fit = (fit*known[:,None,None]).sum()/(known.sum().clamp_min(1)*fit.shape[1]*fit.shape[2])
    else:
        fit = fit.mean()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logit,missing.float())
    return fit+.25*bce,fit,bce


def risk_score(cost, missing_probability, debit, *, factorized):
    utility = cost[...,0]-cost[...,1]
    if not factorized:
        return utility
    p = missing_probability[:,None]
    # Missing target: negative utility debit and equally sized positive harm.
    return (1-p)*utility-2*p*debit


def choose(score, legal, groups, *, counts=None):
    if score.shape != legal.shape or not np.isfinite(score).all():
        raise ValueError('Expected finite aligned candidate scores')
    selected = np.full(len(score),-1,np.int16)
    score = np.where(legal,score,-np.inf)
    best = score.argmax(1)
    values = score[np.arange(len(score)),best]
    for j,q in enumerate(groups):
        use = q[(values[q] > 0.) if counts is None else legal[q].any(1)]
        limit = min(2,len(q)-1,len(use)) if counts is None else int(counts[j])
        if not 0 <= limit <= min(2,len(q)-1,len(use)):
            raise ValueError('Realized intervention count is not feasible for comparator')
        use = use[np.argsort(-values[use],kind='stable')[:limit]]
        selected[use] = best[use]
    return selected
