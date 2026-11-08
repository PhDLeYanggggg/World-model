"""Bounded causal motion experts and in-history forecast reliability."""

import numpy as np
import torch
from torch import nn

from src.world_model.m3w_ols_replacement import velocity


EXPERTS = ("ols4", "fast", "long", "median", "trend")
ARMS = ("replay", "history", "global_mix")
STEPS = np.arange(1., 13.)


def _experts(h, t):
    diff = np.diff(h, axis=1)
    v = velocity(h, 4)
    long = velocity(h, h.shape[1])
    typical = np.median(diff, axis=1)
    short = velocity(h, 3)
    previous = velocity(h[:, :-2], 3)
    trend = (short - previous) / 2.
    # A finite acceleration impulse: after three steps the velocity is constant.
    impulse = (np.minimum(t, 3.) * (np.minimum(t, 3.) + 1.) / 2.
               + 3. * np.maximum(t - 3., 0.))
    t = t[None, :, None]
    return np.stack((v[:, None] * t, diff[:, -1:] * t, long[:, None] * t,
                     typical[:, None] * t,
                     v[:, None] * t + trend[:, None] * impulse[None, :, None]), axis=1)


def _bound(paths, t, scale):
    delta = paths - paths[:, :1]
    limit = .35 * scale[:, None, None, None] * t[None, None, :, None]
    return delta * np.minimum(1., limit / np.linalg.norm(delta, axis=-1, keepdims=True).clip(1e-8))


def design(history):
    h = np.asarray(history, dtype=np.float64)
    if h.ndim != 3 or h.shape[1:] != (8, 2) or not np.isfinite(h).all():
        raise ValueError("Need eight finite, observed 2D positions per row")
    diff = np.diff(h, axis=1)
    scale = np.sqrt(np.square(diff).sum(2).mean(1)).clip(.05)
    v = velocity(h, 4)
    speed = np.linalg.norm(v, axis=-1)
    heading = v / speed[:, None].clip(1e-8)
    heading[speed <= 1e-8] = (1., 0.)
    left = np.stack((-heading[:, 1], heading[:, 0]), axis=1)
    normalized = diff / scale[:, None, None]
    ego = np.column_stack((np.einsum("nti,ni->nt", normalized, heading),
                           np.einsum("nti,ni->nt", normalized, left),
                           np.log1p(scale), speed / scale))
    evidence = []
    for n in (5, 6, 7):
        p = h[:, :n]
        prefix_scale = np.sqrt(np.square(np.diff(p, axis=1)).sum(2).mean(1)).clip(.05)
        t = STEPS[:min(2, 8 - n)]
        pred = _experts(p, t)
        pred = pred[:, :1] + _bound(pred, t, prefix_scale)
        observed = h[:, n:n+len(t)] - p[:, -1:]
        error = np.linalg.norm(pred - observed[:, None], axis=-1).mean(2) / prefix_scale[:, None]
        evidence.append(np.column_stack((error[:, 0], error[:, 1:] - error[:, :1])))
    raw = _experts(h, STEPS)
    deltas = _bound(raw, STEPS, scale)
    return (ego.clip(-12., 12.).astype(np.float32),
            np.concatenate(evidence, axis=1).clip(-12., 12.).astype(np.float32),
            deltas.astype(np.float32), scale.astype(np.float32),
            dict(ols4=raw[:, 0], cv=raw[:, 1], ols8=raw[:, 2],
                 median=raw[:, 3], bounded_trend=raw[:, 0] + deltas[:, 4]))


def arm_features(ego, evidence, arm):
    if arm not in ARMS:
        raise ValueError(f"Unknown context arm: {arm}")
    return np.column_stack((np.zeros_like(ego) if arm == "global_mix" else ego,
                            evidence if arm == "replay" else np.zeros_like(evidence)))


class ReplayMotion(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(31, 48), nn.SiLU(), nn.Linear(48, len(EXPERTS)))
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([3., 0., 0., 0., 0.]))

    def forward(self, x):
        return self.net(x).softmax(dim=-1)


def mix(weights, deltas):
    return torch.einsum("nk,nkhd->nhd", weights, deltas)


def objective(delta, floor, target, valid, scale):
    clean = torch.where(valid[..., None], target, 0.)
    excess = (torch.linalg.vector_norm(floor + delta - clean, dim=-1)
              - torch.linalg.vector_norm(floor - clean, dim=-1))
    debit = torch.linalg.vector_norm(delta, dim=-1)
    full_grid = torch.where(valid, excess, debit).mean(1)
    # A fixed observed component prevents missing steps from being the only gradient.
    observed = torch.where(valid, excess, 0.).sum(1) / valid.sum(1).clamp_min(1)
    return ((full_grid + .25 * observed) / (12. * scale)).mean()
