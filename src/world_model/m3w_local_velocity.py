"""Learn bounded velocity innovations from matched past-only neighborhoods."""

import numpy as np
import torch
from torch import nn

from src.world_model.m3w_ols_replacement import velocity


ARMS = ("near", "outer", "ego")


def inputs(geometry):
    g = np.asarray(geometry)
    if g.ndim != 2 or g.shape[1] != 476 or not np.isfinite(g).all():
        raise ValueError("Expected finite native past-only geometry")
    h = g[:, :16].reshape(-1, 8, 2).astype(float)
    n = g[:, 38:166].reshape(-1, 8, 8, 2).astype(float)
    valid = g[:, 230:294].reshape(-1, 8, 8).all(axis=-1)
    if (valid[:, 1:] & ~valid[:, :-1]).any():
        raise ValueError("Expected a prefix of observed neighbors")
    dv = np.diff(h, axis=1)
    scale = np.sqrt(np.square(dv).sum(2).mean(1)).clip(.05)
    v = velocity(h, 4)
    mag = np.linalg.norm(v, axis=1)
    heading = np.divide(v, mag[:, None], out=np.zeros_like(v), where=mag[:, None] > 1e-8)
    heading[mag <= 1e-8] = (1., 0.)
    basis = np.stack((heading, np.stack((-heading[:, 1], heading[:, 0]), axis=1)), axis=1)
    local = lambda vec: np.einsum("n...i,nji->n...j", vec, basis) / scale.reshape((-1,) + (1,) * (vec.ndim-1))
    ego = np.column_stack((local(dv).reshape(-1, 14), np.log1p(scale), mag/scale))
    peer_v = np.diff(n, axis=2)
    offsets = n[:, :, -1] - h[:, None, -1]
    peer_features = np.concatenate((local(offsets),
        local(peer_v - dv[:, None]).reshape(-1, 8, 14)), axis=-1)
    count = valid.sum(1)
    # Split the same causal sorted bank into disjoint, equally sized halves.
    k = count // 2
    j = np.arange(8)
    masks = dict(near=j < k[:, None],
                 outer=(j >= (count-k)[:, None]) & valid,
                 ego=np.zeros_like(valid))
    t = np.arange(1., 13.)
    # Three-step integrated transient plus a sustained velocity shift.
    # Each ramp is zero at the observation endpoint and bounded by t.
    ramp = np.minimum(t, 3.)
    ols = v[:, None] * t[None, :, None]
    cv = dv[:, -1:] * t[None, :, None]
    a = (peer_v[:, :, -1] - peer_v[:, :, -4]) / 3
    local_a = (a * masks["near"][..., None]).sum(1) / k.clip(1)[:, None]
    limit = .15 * scale / np.linalg.norm(local_a, axis=1).clip(1e-8)
    fixed = ols + (local_a * np.minimum(1., limit)[:, None])[:, None] * (ramp * .5)[None, :, None]
    return (ego.clip(-16., 16.).astype(np.float32), peer_features.clip(-32., 32.).astype(np.float32),
            masks, basis.astype(np.float32), scale.astype(np.float32),
            dict(cv=cv, ols4=ols, fixed_neighbor=fixed))


class VelocityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.peer = nn.Sequential(nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU())
        self.readout = nn.Sequential(nn.Linear(49, 48), nn.SiLU(), nn.Linear(48, 4))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, ego, peers, mask, basis, scale):
        count = mask.sum(1)
        enc = self.peer(peers)
        pool = (enc * mask[..., None]).sum(1) / count.clamp_min(1)[:, None]
        z = self.readout(torch.cat((ego, pool, count[:, None]/4.), dim=1)).reshape(-1, 2, 2)
        # Joint per-step speed perturbation <= 0.5 RMS past step size.
        z = .25 * z / torch.linalg.vector_norm(z, dim=2, keepdim=True).clamp_min(1.)
        vectors = (z @ basis) * scale[:, None, None]
        t = torch.arange(1., 13., device=ego.device)
        return vectors[:, :1] * t[:, None] + vectors[:, 1:] * t.clamp_max(3.)[:, None]


def lower(forecast, floor, target, valid):
    """Use per-step disagreement for missing labels; no future in inference."""
    clean = torch.where(valid[..., None], target, 0.)
    d = torch.linalg.vector_norm(forecast-clean, dim=-1) - torch.linalg.vector_norm(floor-clean, dim=-1)
    debit = torch.linalg.vector_norm(forecast-floor, dim=-1)
    return torch.where(valid, d, debit).mean(dim=1)


def scores(forecast, reference, target, valid):
    clean = np.where(valid[..., None], target, 0.)
    diff = np.linalg.norm(reference-clean, axis=2)-np.linalg.norm(forecast-clean, axis=2)
    dist = np.linalg.norm(reference-forecast, axis=2)
    available = np.where(valid, diff, 0.).sum(1) / valid.sum(1).clip(1)
    return dict(observed=available,
                conservative=np.where(valid.any(1), available, -dist.max(1)),
                full_grid_lower=np.where(valid, diff, -dist).mean(1))
