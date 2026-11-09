"""Interpolate causal references and gate a learned deviation by predicted support."""
import torch
from torch import nn
from torch.nn import functional as F

from src.world_model.m3w_dual_record_forecast import components


MODES = ('bridge', 'lead_mean', 'causal_support')


class Forecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal = nn.GRU(6, 40, batch_first=True)
        self.peers = nn.Sequential(nn.Linear(6, 24), nn.SiLU(), nn.Linear(24, 24))
        self.trunk = nn.Sequential(nn.Linear(65, 48), nn.SiLU())
        self.motion = nn.Linear(48, 9)
        self.support = nn.Linear(48, 12)
        nn.init.zeros_(self.motion.weight)
        nn.init.zeros_(self.motion.bias)
        t = torch.arange(1., 13.)
        u = t/12.
        self.register_buffer('basis', torch.stack(((1.-u).square(), 2.*u*(1.-u), u.square()), 1))
        self.register_buffer('integral', 12.*torch.stack(
            (u-u.square()+u.pow(3)/3., u.square()-2.*u.pow(3)/3., u.pow(3)/3.), 1))

    def forward(self, history, peers, mask, frame, scale, velocity, cv, ols, lead_mean, mode):
        _, temporal = self.temporal(history)
        ctx = (self.peers(peers)*mask[..., None]).sum(1)/mask.sum(1).clamp_min(1.)[:, None]
        size = torch.linalg.vector_norm(ols[:, -1]-cv[:, -1], dim=1)/scale.clamp_min(.05)
        state = self.trunk(torch.cat((temporal[0], ctx, size.log1p().clamp(max=8.)[:, None]), 1))
        motion = self.motion(state)
        a = (motion[:, :3] @ self.basis.T).sigmoid()
        bridge = cv + a[..., None]*(ols-cv)
        logits = self.support(state)
        if mode == 'bridge':
            return bridge, logits
        if mode not in MODES:
            raise ValueError('Unknown reference bridge arm')
        control = .75*motion[:, 3:].reshape(-1, 3, 2).tanh()
        local = torch.einsum('tk,nkd->ntd', self.integral, control)
        offset = torch.einsum('ntk,ndk->ntd', local, frame)*scale[:, None, None]
        gate = lead_mean if mode == 'lead_mean' else logits.detach().sigmoid()
        return bridge+gate[..., None]*offset, logits


def loss(pred, logits, cv, ols, target, valid, weight, cut):
    g, risk, support = components(pred, cv, ols, target, valid, weight, cut)
    aux = F.binary_cross_entropy_with_logits(logits, valid.to(logits.dtype), reduction='none').mean(1)
    return (-g[1] + 2.*(-g[0]).clamp_min(0.) + 2.*risk.clamp_min(0.).sum()
            + support + .20*(weight @ aux)/weight.sum().clamp_min(1e-8))
