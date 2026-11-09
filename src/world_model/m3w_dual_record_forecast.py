"""Train a CV-anchored causal forecast with a recording-risk surrogate."""
import torch
from torch import nn

from src.world_model.m3w_causal_velocity_spline import Spline


class Forecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.spline = Spline()
        self.gate = nn.Sequential(nn.Flatten(),nn.Linear(42,24),nn.SiLU(),nn.Linear(24,1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, history, peers, mask, frame, scale, velocity, cv):
        raw = self.spline(history,peers,mask,frame,scale,velocity,True)
        gate = (.5+.5*self.gate(history).squeeze(1)).clamp(0.,1.)
        return cv+gate[:,None,None]*(raw-cv)


def components(pred, cv, ols, target, valid, weight, cut):
    clean = torch.where(valid[...,None],target,0.)
    pe,ce,oe = (torch.linalg.vector_norm(z-clean,dim=-1) for z in (pred,cv,ols))
    dr_cv, dr_ols = (torch.linalg.vector_norm(pred-z,dim=-1) for z in (cv,ols))
    ref = (weight*torch.where(valid,ce,0.).sum(1)).sum().clamp_min(1e-4)
    gains = torch.stack([(weight*torch.where(valid,error-pe,-debit).sum(1)).sum()/ref
                         for error,debit in ((ce,dr_cv),(oe,dr_ols))])
    n = valid.sum(1)
    ce = torch.where(valid,ce,0.).sum(1)/n.clamp_min(1)
    pe = torch.where(valid,pe,0.).sum(1)/n.clamp_min(1)
    easy = (n>0)&(ce>0.)&(ce<=cut)
    selected = dr_cv.amax(1)>0.
    unknown = torch.where(n==0,dr_cv.amax(1),0.).sum()
    harm = (pe-ce).clamp_min(0.)
    ref_all = (ce*selected).sum()
    ref_easy = (ce*easy*selected).sum()
    ref_all_easy = (ce*easy).sum()
    sums = torch.stack((harm.sum()+unknown,(harm*easy).sum()+unknown,((pe-ce)*easy).sum()))
    refs = torch.stack((ref_all,ref_easy,ref_all_easy))
    excess = (sums-.02*refs)/(refs.detach().clamp_min(.05*ce.sum()+1e-4))
    # Continuous support incentive is used only for optimization; scoring uses exact edits.
    strength = dr_cv.amax(1)/(torch.linalg.vector_norm(cv,dim=-1).amax(1).detach().clamp_min(.05))
    easy_strength = ((ce*easy)*strength.clamp(max=1.)).sum()/ref_all_easy.clamp_min(1e-4)
    support = torch.where(ref_all_easy>0.,(.02-easy_strength).clamp_min(0.),0.)
    return gains,excess,support


def objective(pred, cv, ols, target, valid, weight, cut, constrained):
    gain,excess,support = components(pred,cv,ols,target,valid,weight,cut)
    if not constrained:
        return -gain[1]
    return -gain.min() + 2.*excess.clamp_min(0.).sum() + support
