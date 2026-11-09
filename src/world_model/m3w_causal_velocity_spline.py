"""Causal, motion-frame velocity extrapolation with a bounded integrated spline."""
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn

from src.world_model.m3w_prefix_expert_mixture import slope


NEIGHBORS = 4
MODES = ('ego', 'peers', 'completion')
HORIZON = 12


def causal_inputs(history, groups):
    h = np.asarray(history, dtype=np.float64)
    if h.ndim != 3 or h.shape[1:] != (8, 2) or not np.isfinite(h).all():
        raise ValueError('Require eight finite past positions')
    dv = np.diff(h, axis=1)
    v4 = slope(h, 4)
    scale = np.sqrt(np.square(dv).sum(2).mean(1)).clip(.05)
    norm = np.linalg.norm(v4, axis=1)
    longest = dv[np.arange(len(h)), np.square(dv).sum(2).argmax(1)]
    direction = np.where((norm > 1e-8)[:, None], v4, longest)
    direction /= np.linalg.norm(direction, axis=1).clip(1e-8)[:, None]
    frame = np.stack((direction, np.column_stack((-direction[:, 1], direction[:, 0]))), axis=2)
    pos = np.einsum('ntd,ndk->ntk', h[:, 1:]-h[:, -1:], frame)/scale[:, None, None]
    vel = np.einsum('ntd,ndk->ntk', dv, frame)/scale[:, None, None]
    acc = np.diff(vel, axis=1, prepend=vel[:, :1])
    tokens = np.concatenate((pos, vel, acc), axis=2).clip(-24., 24.)
    peers = np.zeros((len(h), NEIGHBORS, 6))
    present = np.zeros((len(h), NEIGHBORS), bool)
    covered = np.zeros(len(h), bool)
    for g in groups:
        if covered[g].any() or len(np.unique(g)) != len(g):
            raise ValueError('Overlapping query membership')
        covered[g] = True
        if len(g) < 2:
            continue
        k = min(NEIGHBORS, len(g)-1)
        _, indices = cKDTree(h[g, -1]).query(h[g, -1], k=k+1)
        # Self may not be first when two agents share the same observed position.
        indices = np.where(indices == np.arange(len(g))[:, None], len(g), indices)
        order = np.argsort(indices == len(g), axis=1, kind='stable')[:, :k]
        near = g[np.take_along_axis(indices, order, axis=1)]
        d = h[near, -1]-h[g, None, -1]
        dv_peer = v4[near]-v4[g, None]
        ttc = np.clip(-np.sum(d*dv_peer, axis=-1)/np.square(dv_peer).sum(2).clip(1e-10), 0., HORIZON)
        closest = np.linalg.norm(d+ttc[..., None]*dv_peer, axis=2)
        local = [np.einsum('nkd,ndj->nkj', z, frame[g])/scale[g, None, None] for z in (d, dv_peer)]
        peers[g, :k] = np.concatenate((*local, ttc[..., None]/HORIZON,
                                       closest[..., None]/scale[g, None, None]), axis=2)
        present[g, :k] = True
    if not covered.all():
        raise ValueError('Query roster is incomplete')
    return tuple(a.astype(np.float32) for a in
                 (tokens, peers.clip(-24., 24.), present, frame, scale, v4))


class Spline(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal = nn.GRU(6, 40, batch_first=True)
        self.peers = nn.Sequential(nn.Linear(6, 24), nn.SiLU(), nn.Linear(24, 24))
        self.head = nn.Sequential(nn.Linear(64, 48), nn.SiLU(), nn.Linear(48, 6))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        t = torch.arange(1., HORIZON+1)
        u = t/HORIZON
        self.register_buffer('lead', t)
        self.register_buffer('integral', HORIZON*torch.stack(
            (u-u.square()+u.pow(3)/3, u.square()-2*u.pow(3)/3, u.pow(3)/3), dim=1))

    def forward(self, tokens, peers, mask, frame, scale, velocity, with_peers):
        _, temporal = self.temporal(tokens)
        z = (self.peers(peers)*mask[..., None]).sum(1)/mask.sum(1).clamp_min(1.)[:, None]
        context = z if with_peers else torch.zeros_like(z)
        controls = .75*self.head(torch.cat((temporal[0], context), dim=1)).reshape(-1, 3, 2).tanh()
        local = torch.einsum('tk,nkd->ntd', self.integral, controls)
        delta = torch.einsum('ntk,ndk->ntd', local, frame)*scale[:, None, None]
        return velocity[:, None]*self.lead[None, :, None] + delta


def loss(pred, velocity, target, valid, scale, completion):
    t = torch.arange(1., HORIZON+1, dtype=pred.dtype, device=pred.device)
    ols = velocity[:, None]*t[None, :, None]
    y = torch.where(valid[..., None], target, 0.)
    delta = torch.linalg.vector_norm(pred-y, dim=-1)-torch.linalg.vector_norm(ols-y, dim=-1)
    debit = torch.linalg.vector_norm(pred-ols, dim=-1) if completion else torch.zeros_like(delta)
    return (torch.where(valid, delta, debit).mean(1)/scale).mean()
