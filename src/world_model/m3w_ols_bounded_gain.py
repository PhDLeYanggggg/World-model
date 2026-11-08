"""Predict bounded gain as a fraction of known forecast disagreement."""
import torch

from src.world_model.m3w_ols_replacement import network


ARMS = ('bounded_full','bounded_no_hindcast','bounded_no_neighbors')


def logits(net,x,near,arm):
    if arm not in ARMS:
        raise ValueError('Unknown bounded-gain feature arm')
    if arm == 'bounded_no_hindcast':
        x = x.clone()
        x[..., -5:-1] = 0.
    if arm == 'bounded_no_neighbors':
        near = torch.zeros_like(near)
    n = near[:,None].expand(-1,x.shape[1],-1)
    return net(torch.cat((x,n),dim=-1))


def fractional_target(gain,harm,debit):
    valid = debit > 1e-6
    den = debit.clamp_min(1e-6)
    g,h = gain/den,harm/den
    if (g[valid].abs() > 1.00001).any() or (h[valid] > 1.00001).any() or (h[valid] < 0.).any():
        raise ValueError('Forecast gain exceeds its geometric disagreement bound')
    return torch.stack(((g.clamp(-1.,1.)+1.)/2.,h.clamp(0.,1.)),dim=-1),valid


def objective(z,target,valid):
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z,target,reduction='none')
    return (loss*valid[...,None]).sum()/(2*valid.sum().clamp_min(1))


def score(z,debit):
    f,h = z.sigmoid().unbind(-1)
    return (2*f-1-h)*debit
