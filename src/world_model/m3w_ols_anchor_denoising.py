"""Isolate causal position-anchor and velocity perturbations around OLS4."""
import numpy as np
import torch
from torch import nn

from src.world_model.m3w_ols_replacement import velocity


FITS = ('learned_anchor','learned_slope')
ARMS = ('cv','ols4','ols8','raw_line4','fixed_anchor','fixed_slope',*FITS)


def design(history):
    h = np.asarray(history,dtype=float)
    if h.ndim != 3 or h.shape[1:] != (8,2) or not np.isfinite(h).all():
        raise ValueError('Complete eight-step causal histories required')
    t = np.arange(1.,13.)
    shift = h-h[:,-1:]
    v4,v8 = (velocity(shift,n) for n in (4,8))
    dv = np.diff(shift,axis=1)
    scale = np.sqrt(np.square(dv).sum(2).mean(1)).clip(.05)
    vcv = dv[:,-1]
    floor = v4[:,None]*t[None,:,None]
    bias = shift[:,-4:].mean(1)+1.5*v4
    vel_delta = v8-v4
    radius,s = (np.linalg.norm(a,axis=1) for a in (bias,vel_delta))
    # The full comparison uses the same maximum displacement in either direction.
    delta = np.divide(vel_delta,s[:,None],out=np.zeros_like(vel_delta),where=s[:,None] > 1e-8)*radius[:,None]
    anchor = bias[:,None]*np.minimum(t/3.,1.)[None,:,None]
    slope = delta[:,None]*(t/12.)[None,:,None]
    step = np.linalg.norm(dv,axis=2)/scale[:,None]
    residual = shift[:,-4:]-bias[:,None]-v4[:,None]*np.arange(-3,1)[None,:,None]
    strength = np.linalg.norm(residual,axis=2)/scale[:,None]
    align = (bias*v4).sum(1)/(radius*np.linalg.norm(v4,axis=1)).clip(1e-8)
    features = np.column_stack((np.log1p(scale),step[:,-1],step.mean(1),step.std(1),step[:,-3:].mean(1),
        np.linalg.norm(v4,axis=1)/scale,np.linalg.norm(v8,axis=1)/scale,radius/scale,s/scale,
        np.linalg.norm(vcv-v4,axis=1)/scale,strength.mean(1),strength.max(1),align))
    paths = dict(cv=vcv[:,None]*t[None,:,None],ols4=floor,ols8=v8[:,None]*t[None,:,None],
                 raw_line4=floor+bias[:,None],fixed_anchor=floor+anchor,fixed_slope=floor+slope)
    return features.clip(-16.,16.).astype(np.float32),paths,anchor,slope,scale,s>1e-8


def network(width):
    return nn.Sequential(nn.Linear(width,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU(),nn.Linear(32,1))


def loss(net,x,floor,delta,target,valid,scale):
    a = net(x).sigmoid().view(-1,1,1)
    p = floor+a*delta
    clean = torch.where(valid[...,None],target,0.)
    error = torch.linalg.vector_norm(p-clean,dim=2)-torch.linalg.vector_norm(floor-clean,dim=2)
    known = valid.any(1)
    diff = torch.where(valid,error,0.).sum(1)/valid.sum(1).clamp_min(1)
    debit = torch.linalg.vector_norm(a*delta,dim=2).amax(1)
    cost = torch.where(known,diff+diff.relu(),2.*debit)/scale
    return cost.mean()
