"""Past-only, whole-path mixture of analytic motion experts."""

import numpy as np
import torch
from torch import nn


EXPERTS = ("cv", "ols4", "ols8", "damped_ols4", "stable_turn", "stationary")
ARMS = ("static", "kinematic", "prefix")


def slope(h, width):
    h = h[:, -width:]
    t = np.arange(h.shape[1], dtype=float)
    t -= t.mean()
    return np.einsum("t,ntd->nd", t, h - h[:, -1:]) / (t @ t)


def bank(h, horizon=12):
    """Relative path bank on the uniform native annotation grid."""
    h = np.asarray(h, float)
    if h.ndim != 3 or h.shape[1] < 4 or h.shape[2] != 2 or not np.isfinite(h).all():
        raise ValueError("Require at least four finite past positions")
    t = np.arange(1, horizon + 1)[None, :, None]
    v, v4, v8 = h[:, -1] - h[:, -2], slope(h, 4), slope(h, 8)
    recent, prior = slope(h, 3), slope(h[:, :-1], 3)
    cross = prior[:, 0] * recent[:, 1] - prior[:, 1] * recent[:, 0]
    angle = np.arctan2(cross, (prior * recent).sum(1)).clip(-np.pi / 32., np.pi / 32.)
    valid_turn = (np.linalg.norm(prior, axis=1) > 1e-8) & (np.linalg.norm(recent, axis=1) > 1e-8)
    angle = np.where(valid_turn, angle, 0.)
    phase = t[..., 0] * angle[:, None]
    cs, sn = np.cos(phase), np.sin(phase)
    vx, vy = v4[:, :1], v4[:, 1:]
    turn = np.stack((vx * cs - vy * sn, vx * sn + vy * cs), axis=-1).cumsum(1)
    straight = v4[:, None] * t
    return np.stack((v[:, None] * t, straight, v8[:, None] * t,
        v4[:, None] * (-np.expm1(-.05 * t) / .05), turn, np.zeros_like(straight)), axis=1)


def features(history, arm):
    h = np.asarray(history, float)
    if h.shape[1:] != (8, 2) or arm not in ARMS:
        raise ValueError("Require the fixed eight-step input and a registered arm")
    dv = np.diff(h, axis=1)
    scale = np.sqrt(np.square(dv).sum(2).mean(1)).clip(1.)
    v = slope(h, 4)
    u = v / np.linalg.norm(v, axis=1).clip(1e-12)[:, None]
    along = (dv * u[:, None]).sum(2) / scale[:, None]
    side = (dv[..., 1] * u[:, None, 0] - dv[..., 0] * u[:, None, 1]) / scale[:, None]
    motion = np.column_stack((along, side, np.log(scale)))
    regrets = []
    for end in range(4, 8):
        y = h[:, end:] - h[:, end - 1:end]
        p = bank(h[:, :end], 8 - end)
        error = np.linalg.norm(p - y[:, None], axis=-1) / scale[:, None, None]
        regrets.append((error - error[:, 1:2]).reshape(len(h), -1))
    history_error = np.concatenate(regrets, axis=1)
    x = np.column_stack((motion if arm != "static" else np.zeros_like(motion),
                         history_error if arm == "prefix" else np.zeros_like(history_error)))
    return np.arcsinh(x).astype(np.float32), scale.astype(np.float32)


class Mixture(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(width, 32), nn.SiLU(), nn.Linear(32, len(EXPERTS)))
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():
            self.head[-1].bias.zero_()
            self.head[-1].bias[1] = 3.

    def forward(self, x, paths):
        w = self.head(x).softmax(1)
        return (paths * w[:, :, None, None]).sum(1), w


def loss(net, x, paths, scale, target, valid):
    p, _ = net(x, paths)
    safe = torch.where(valid[..., None], target, 0.)
    error = torch.linalg.vector_norm(p - safe, dim=-1) / scale[:, None]
    return torch.where(valid, error, 0.).sum() / valid.sum().clamp_min(1)


def full_masses(left, right, cv, target, valid, weight):
    """Each unobserved step pays its own paired distance; denominator is CV mass."""
    a, b, c = (np.linalg.norm(p - target, axis=-1) for p in (left, right, cv))
    d = np.linalg.norm(left - right, axis=-1)
    return dict(reference=float(weight @ np.where(valid, c, 0.).sum(1)),
                known_delta=float(weight @ np.where(valid, a - b, 0.).sum(1)),
                unknown_radius=float(weight @ np.where(valid, 0., d).sum(1)))
