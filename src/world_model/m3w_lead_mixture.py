"""Bounded analytic path mixtures with shared or lead-dependent weights."""
import torch
from torch import nn


class LeadMixture(nn.Module):
    def __init__(self, width, experts):
        super().__init__()
        self.experts = experts
        self.head = nn.Sequential(nn.Linear(width,48),nn.SiLU(),nn.Linear(48,48),nn.SiLU(),
                                  nn.Linear(48,3*experts))
        with torch.no_grad():
            self.head[-1].weight.zero_()
            self.head[-1].bias.reshape(3,experts).zero_()[:,1] = 3.

    def forward(self, x, paths, lead_dependent):
        w = self.head(x).reshape(-1,3,self.experts).softmax(-1)
        u = torch.linspace(0.,1.,paths.shape[-2],device=x.device,dtype=x.dtype)
        if lead_dependent:
            basis = torch.stack(((1-u)**2,2*u*(1-u),u**2),dim=1)
        else:
            basis = x.new_full((len(u),3),1/3)
        weights = torch.einsum('tk,nke->nte',basis,w)
        return torch.einsum('nte,netd->ntd',weights,paths)


def objective(path, cv, ols, target, valid, scale, easy_cut):
    y = torch.where(valid[...,None],target,0.)
    norm = lambda x:torch.linalg.vector_norm(x,dim=-1)
    error = norm(path-y)
    ref = norm(cv-y)
    upper = torch.where(valid,error-norm(ols-y),norm(path-ols)).mean(1)
    count = valid.sum(1).clamp_min(1)
    known = valid.any(1)
    ref_ade = torch.where(valid,ref,0.).sum(1)/count
    signed = torch.where(valid,error-ref,0.).sum(1)/count
    easy = known & (ref_ade>0.) & (ref_ade<=easy_cut)
    missing = torch.where(known,0.,norm(path-cv).amax(1))
    all_cost = torch.where(known,signed.relu()-.02*ref_ade,0.)+missing
    easy_cost = torch.where(easy,signed.relu()-.02*ref_ade,0.)+missing
    return ((upper+2.*(all_cost.relu()+easy_cost.relu()))/scale).mean()
