"""Feasible recording substitutions and additive/nonadditive slate scorers."""
import numpy as np
import torch
from torch import nn

from src.evaluation.m3w_cv_anchor_budget import candidate_terms, risk_excess
from src.evaluation.m3w_selection_forecast_readout import errors
from src.world_model.m3w_cv_admission import features
from src.world_model.m3w_record_joint_actions import local_edges


RISK_EPS = 1e-8


def path_for(cv,bank,action):
    p = cv.copy()
    at = np.flatnonzero(action>=0)
    p[at] = bank[at,action[at]]
    return p


def top_action(score,legal,k):
    best = np.where(legal,score,-np.inf).argmax(1)
    val = np.where(legal[np.arange(len(best)),best],score[np.arange(len(best)),best],-np.inf)
    order = np.argsort(-val,kind='stable')[:k]
    if len(order)!=k or not np.isfinite(val[order]).all():
        raise ValueError('Slate cannot reach its fixed count')
    a = np.full(len(score),-1,dtype=np.int8)
    a[order] = best[order]
    return a


def menu(cv,ols,bank,k):
    radius = np.linalg.norm(bank-cv[:,None],axis=-1)
    legal = radius.max(2)>1e-8
    agreement = np.linalg.norm(cv-ols,axis=-1)[:,None]-np.linalg.norm(bank-ols[:,None],axis=-1)
    # The fifth slot specifically values the end of the forecast.
    rank = [agreement.mean(2)-t*radius.mean(2) for t in (0.,.25,.75,1.25)]
    rank.append(agreement[...,8:].mean(2)-.25*radius[...,:4].mean(2))
    return [top_action(s,legal,k) for s in rank]


def descriptor(past,h,cv,path,scale,action,edges):
    d = path-cv
    selected = action>=0
    x = features(past[selected],d[selected],scale[selected])
    if not len(x):
        raise ValueError('Fallback does not require a slate score')
    i,j = edges
    if len(i):
        normal = (scale[i]+scale[j])*.5
        dv = d[i]-d[j]
        rel = h[i,-1:]-h[j,-1:]+cv[i]-cv[j]
        width = np.linalg.norm(dv,axis=-1)/normal[:,None]
        contraction = (np.linalg.norm(rel,axis=-1)-np.linalg.norm(rel+dv,axis=-1)).clip(0.)/normal[:,None]
        graph = (width.mean(),width.max(),contraction.mean(),contraction.max(),
                 (selected[i]&selected[j]).mean(),(np.linalg.norm(h[i,-1]-h[j,-1],axis=-1)/normal).mean())
    else:
        graph = (0.,)*6
    return np.r_[x.mean(0),x.std(0),np.arcsinh((*graph,selected.mean(),np.log1p(len(h))))].astype(np.float32)


def contributions(cv,ols,path,y,valid,w,easy_cut):
    """CV-anchored action increment and exact original risk excesses."""
    ref,offset,g,_ = candidate_terms(cv,ols,path[:,None],y,valid)
    e,_ = errors(cv,y,valid)
    p,_ = errors(path,y,valid)
    easy = valid.any(1)&(e>0.)&(e<=easy_cut)
    signed = np.where(easy,p-e,0.)
    easy_selected = easy & (path!=cv).any((1,2))
    risk = risk_excess(cv,path[:,None],y,valid,easy_cut)[:,0].sum(0)
    return np.r_[w@g[:,0],risk,signed.sum(),easy_selected.sum()],float(w@ref),float(w@offset),.02*e[easy].sum()


def verdict(increments,ref,offset,easy_cap):
    gain = 100.*np.min(increments[:2]+[0.,offset])/ref
    ok = bool(gain>1e-6 and (increments[2:4]<=RISK_EPS).all()
              and increments[4]<=easy_cap+RISK_EPS and increments[5]>.5)
    return gain,ok


def preference(left,right,feasible):
    if feasible and left[1]!=right[1]:
        return 1 if left[1] else -1
    return int(np.sign(left[0]-right[0])) if abs(left[0]-right[0])>1e-9 else 0


class SlateScore(nn.Module):
    def __init__(self,width,joint):
        super().__init__()
        self.width = width
        self.additive = nn.Linear(width,1)
        self.joint = nn.Sequential(nn.Linear(width*2+8,32),nn.SiLU(),nn.Linear(32,1)) if joint else None
        if self.joint is not None:
            nn.init.zeros_(self.joint[-1].weight)
            nn.init.zeros_(self.joint[-1].bias)

    def forward(self,x):
        y = self.additive(x[...,:self.width]).squeeze(-1)
        return y if self.joint is None else y+self.joint(x).squeeze(-1)


def loss(net,x,y):
    score = net(x)
    z = torch.nn.functional.softplus(-y*(score[:,0]-score[:,1]))
    use = y!=0.
    return (z*use).sum()/use.sum().clamp_min(1)
