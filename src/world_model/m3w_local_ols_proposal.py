"""Bounded, heading-aligned trajectory proposals anchored on causal OLS4."""
import numpy as np
import torch
from torch import nn


STEPS, BATCH, AMPLITUDES = 240, 128, (0., .25, .5, 1.)


def describe(history, groups):
    """Build local and neighbor context using the eight observed positions."""
    h = np.asarray(history, dtype=np.float64)
    if h.ndim != 3 or h.shape[1:] != (8, 2) or not np.isfinite(h).all():
        raise ValueError('Require eight finite past positions')
    v = np.diff(h, axis=1)
    recent = v[:, -4:].mean(1)
    norm = np.linalg.norm(recent, axis=-1)
    scale = np.maximum(np.linalg.norm(v, axis=-1).mean(1), 1.)
    axis = np.divide(recent, norm[:, None], out=np.zeros_like(recent), where=norm[:, None] > 1e-9)
    axis[norm <= 1e-9, 0] = 1.
    rot = np.stack((axis, np.stack((-axis[:, 1], axis[:, 0]), axis=1)), axis=1)
    local = np.einsum('ntd,nkd->ntk', v, rot)/scale[:, None, None]
    own = np.column_stack((local.reshape(len(h), -1), np.log1p(scale)))
    context = np.zeros((len(h), 7))
    covered = np.zeros(len(h), dtype=int)
    for rows in groups:
        rows = np.asarray(rows, dtype=int)
        np.add.at(covered, rows, 1)
        if len(rows) < 2:
            continue
        p = h[rows, -1]
        z = p[None, :, :] - p[:, None, :]
        dist = np.linalg.norm(z, axis=-1)
        np.fill_diagonal(dist, np.inf)
        j = dist.argmin(1)
        rel = z[np.arange(len(rows)), j]
        delta_v = recent[rows[j]]-recent[rows]
        shared = np.broadcast_to(recent[rows].mean(0), rel.shape)
        vec = np.stack((rel, delta_v, shared), axis=1)
        x = np.einsum('ntd,nkd->ntk', vec, rot[rows])/scale[rows, None, None]
        context[rows] = np.column_stack((x.reshape(len(rows), 6), np.full(len(rows), np.log1p(len(rows)-1))))
    if not np.all(covered == 1):
        raise ValueError('Query roster must partition the population exactly')
    return own, context, rot, scale


def make_net(width):
    net = nn.Sequential(nn.Linear(width, 32), nn.SiLU(), nn.Linear(32, 4))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


def shift(net, x, rot, scale):
    coef = net(x).reshape(-1, 2, 2)
    bounded = coef/(1.+torch.linalg.vector_norm(coef, dim=-1, keepdim=True))
    step = torch.arange(1, 13, device=x.device, dtype=x.dtype)
    t = (step/12.)[None, :, None]
    local = (1.-t)*bounded[:, :1] + t*bounded[:, 1:]
    return torch.einsum('ntk,nkd->ntd', local, rot) * (.5*scale[:, None, None]*step[None, :, None])


def loss(delta, ols, target, valid, scale):
    n = valid.sum(1)
    target = torch.where(valid[..., None], target, ols)
    err = torch.linalg.vector_norm(ols+delta-target, dim=-1)
    base = torch.linalg.vector_norm(ols-target, dim=-1)
    known = ((err-base)*valid).sum(1)/n.clamp_min(1)
    unknown = torch.linalg.vector_norm(delta, dim=-1).amax(1)
    return (torch.where(n > 0, known, unknown)/scale).mean()
