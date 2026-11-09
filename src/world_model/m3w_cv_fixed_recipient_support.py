"""Equal-peak actions and train-source consensus relative to uniform shrinkage."""
import numpy as np
from torch.nn import functional as F

from src.evaluation.m3w_cv_anchor_budget import candidate_terms, risk_excess
from src.evaluation.m3w_selection_forecast_readout import errors
from src.world_model.m3w_cv_minimum_budget import action_features


def peak(delta):
    return np.linalg.norm(delta,axis=-1).max(-1)


def project(cv,anchor,bank,alpha):
    """Never amplify a donor and always retain the uniform action in slot zero."""
    delta = anchor-cv
    r = peak(delta)*alpha
    d = bank-cv[:,None]
    m = peak(d)
    valid = (m>=r[:,None])&(r[:,None]>0.)
    ratio = np.divide(r[:,None],m,out=np.zeros_like(m),where=valid)
    uniform = cv+alpha*delta
    paths = np.concatenate((uniform[:,None],cv[:,None]+ratio[...,None,None]*d),axis=1)
    return paths,np.c_[r>0.,valid]


def features(past,c,o,u,p,s):
    return np.c_[action_features(past,c,o,p,s),action_features(past[:,:0],c,o,u,s)]


def targets(cv,ols,u,p,y,v,s,cut):
    paths = np.stack((u,p),axis=1)
    _,_,g,_ = candidate_terms(cv,ols,paths,y,v)
    risk = risk_excess(cv,paths,y,v,cut)
    c,_ = errors(cv,y,v)
    l,_ = errors(u,y,v)
    r,_ = errors(p,y,v)
    easy = v.any(1)&(c>0.)&(c<=cut)
    signed = np.where(easy,r-l,0.)
    clean = np.where(v[...,None],y,0.)
    gain = np.where(v,np.linalg.norm(u-clean,axis=-1)-np.linalg.norm(p-clean,axis=-1),0.).sum(1)
    debit = np.where(v,0.,np.linalg.norm(u-p,axis=-1)).sum(1)
    return np.c_[gain/(12*s),debit/(12*s),(g[:,1]-g[:,0])/(12*s[:,None]),
        (risk[:,1]-risk[:,0])/s[:,None],signed/s,v.mean(1)].clip(-8.,8.).astype(np.float32)


def objective(net,x,y):
    z = net(x)
    return F.smooth_l1_loss(z[:,:7],y[:,:7])+.15*F.binary_cross_entropy_with_logits(z[:,7],y[:,7])


def choose(pooled,sources,legal,mode):
    if mode=='uniform':
        return np.zeros(len(legal),np.int8)
    if mode not in ('pooled','consensus'):
        raise ValueError('Unknown equal-radius mode')
    bound = lambda p: p[...,0]-p[...,1].clip(0.)
    if mode=='pooled':
        score = bound(pooled)-pooled[...,4:7].clip(0.).sum(-1)
        keep = legal & (score>0.)
    else:
        score = bound(sources).min(0)
        keep = legal & (score>0.) & (sources[...,2:4].min((0,3))>=0.)
        keep &= (sources[...,4:7].max((0,3))<=0.) & (sources[...,7].min(0)>=0.)
    z = np.where(keep,score,-np.inf)
    z[:,0] = 0.
    return z.argmax(1).astype(np.int8)
