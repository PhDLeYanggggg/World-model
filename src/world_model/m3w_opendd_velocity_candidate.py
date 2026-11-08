"""A causal, heading-aligned, direct multi-step velocity candidate."""
import numpy as np
import torch
from torch import nn


def frames(history):
    vel = np.diff(history,axis=1)
    tail = vel[:,-1]
    smoothed = vel[:,-3:].mean(1)
    axis = np.where(np.linalg.norm(tail,axis=1,keepdims=True)>.01,tail,smoothed)
    norm = np.linalg.norm(axis,axis=1,keepdims=True)
    axis = np.where(norm>1e-8,axis/norm.clip(1e-8),np.array([[1.,0.]]))
    normal = np.stack((-axis[:,1],axis[:,0]),axis=1)
    rotation = np.stack((axis,normal),axis=2)
    scale = np.sqrt(np.square(vel).sum(2).mean(1)).clip(.05)
    return rotation,scale


def design(history, context):
    if history.shape[1:] != (8,2) or context.shape != (len(history),2,2,8,2):
        raise ValueError('Require eight-step positions and two-by-two context histories')
    rot,scale = frames(history)
    origin = history[:,-1]
    past = ((history-origin[:,None]) @ rot) / scale[:,None,None]
    valid = np.isfinite(context).all(-1)
    relative = np.where(valid[...,None], context-origin[:,None,None,None],0.)
    peer = (np.einsum('ngkti,nij->ngktj',relative,rot)/10.).clip(-8.,8.)
    features = np.concatenate((past.reshape(len(past),-1),np.log1p(scale)[:,None],
        np.concatenate((peer,valid[...,None]),axis=-1).reshape(len(past),-1)),axis=1)
    return features.astype(np.float32),rot,scale


def network(width):
    return nn.Sequential(nn.Linear(width,64),nn.SiLU(),nn.Linear(64,48),nn.SiLU(),nn.Linear(48,6))


def decode(raw):
    if raw.ndim!=2 or raw.shape[1]!=6:
        raise ValueError('Expected three velocity knots')
    knots = 5. * torch.tanh(raw.reshape(-1,3,2)/5.)
    speed = nn.functional.interpolate(knots.transpose(1,2),size=12,mode='linear',align_corners=True)
    return speed.transpose(1,2).cumsum(dim=1)


def objective(pred, future, valid, cv):
    observed = torch.where(valid[...,None],future,0.)
    err = torch.sqrt((pred-observed).square().sum(-1) + .0001)
    ref = torch.sqrt((cv-observed).square().sum(-1) + .0001)
    terms = torch.where(valid,err+.25*torch.relu(err-ref),0.)
    return (terms.sum(1)/valid.sum(1).clamp_min(1)).mean()
