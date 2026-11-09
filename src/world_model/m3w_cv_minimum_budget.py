"""Cost scoring and minimum-edit imitation on a shared causal action bank."""
import numpy as np
import torch
from torch.nn import functional as F

from src.evaluation.m3w_cv_anchor_budget import candidate_terms, risk_excess
from src.evaluation.m3w_selection_forecast_readout import errors
from src.world_model.m3w_cv_admission import features


def query_context(past,q):
    _,inv,n = np.unique(q,return_inverse=True,return_counts=True)
    summary = np.zeros((len(n),past.shape[1]),np.float64)
    np.add.at(summary,inv,past)
    return np.c_[past,summary[inv]/n[inv,None],np.log1p(n[inv])].astype(np.float32)


def action_features(past,cv,ols,action,scale):
    extra = np.log1p(np.linalg.norm(action-ols,axis=-1)/scale[:,None])
    return np.c_[features(past,action-cv,scale),extra].astype(np.float32)


def targets(cv,ols,action,y,valid,scale,cut):
    _,_,g,_ = candidate_terms(cv,ols,action[:,None],y,valid)
    risk = risk_excess(cv,action[:,None],y,valid,cut)[:,0]
    ce,_ = errors(cv,y,valid)
    pe,_ = errors(action,y,valid)
    easy = valid.any(1)&(ce>0.)&(ce<=cut)
    signed = np.where(easy,pe-ce,0.)
    raw = np.c_[g[:,0]/(12.*scale[:,None]),risk/scale[:,None],signed/scale]
    return raw.clip(-8.,8.).astype(np.float32)


def objective(net,x,cost,teacher,valid_teacher,use_teacher):
    output = net(x)
    loss = F.smooth_l1_loss(output[:,:5],cost)
    if use_teacher:
        pos = valid_teacher & teacher
        neg = valid_teacher & ~teacher
        bce = F.binary_cross_entropy_with_logits(output[:,5],teacher.float(),reduction='none')
        loss = loss+.15*((bce*pos).sum()/pos.sum().clamp_min(1)+(bce*neg).sum()/neg.sum().clamp_min(1))
    return loss


def choose(pred,legal,groups,scale,fraction,penalty,teacher,match=None,guard=False):
    gain = pred[...,:2].min(-1)
    risk = pred[...,2:5]
    score = gain-penalty*risk.clip(0.).sum(-1)
    if teacher:
        score = score+.20*np.tanh(pred[...,5])
    score = np.where(legal,score,-np.inf)
    arm = score.argmax(1)
    row = np.arange(len(arm))
    val = score[row,arm]*scale
    out = np.full(len(arm),-1,dtype=np.int8)
    for q in groups:
        k = int((match[q]>=0).sum()) if match is not None else max(1,int(np.floor(fraction*len(q))))
        at = q[np.argsort(-val[q],kind='stable')[:k]]
        at = at[np.isfinite(val[at])]
        if match is not None:
            if len(at)!=k:
                raise ValueError('Matched query did not keep its edit count')
        else:
            at = at[val[at]>0.]
            if guard and len(at):
                a = arm[at]
                balance = np.cumsum(risk[at,a]*scale[at,None],axis=0)
                feasible = np.flatnonzero((balance<=0.).all(1))
                at = at[:(feasible[-1]+1 if len(feasible) else 0)]
        out[at] = arm[at]
    return out
