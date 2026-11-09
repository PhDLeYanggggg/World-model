"""Predict available-label gain and positive harm for a fixed analytic bank."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.world_model import m3w_dronecrowd_motion_floor as floor


METHODS = ("gain", "individual", "scene_budget", "matched_gain")
TOLERANCE = .02


def features(history):
    h = np.asarray(history, dtype=float)
    if h.ndim != 3 or h.shape[1:] != (8, 2) or not np.isfinite(h).all():
        raise ValueError("Only finite eight-step histories are inputs")
    delta = np.diff(h, axis=1)
    scale = np.linalg.norm(delta, axis=-1).mean(1).clip(.05)
    heading = h[:, -1]-h[:, 0]
    norm = np.linalg.norm(heading, axis=-1)
    u = heading/np.where(norm > 0., norm, 1.)[:, None]
    u[norm == 0.] = (1., 0.)
    rot = np.stack((u, np.column_stack((-u[:, 1], u[:, 0]))), axis=-1)
    tail = ((h-h[:, -1:]) @ rot)/scale[:, None, None]
    probe = []
    for k in range(4, 8):
        c = floor.base(h[:, :k])
        actual = h[:, k]-h[:, k-1]
        probe.append(np.linalg.norm(c[:, :, 0]+c[:, :, 1]-actual[:, None], axis=-1)/scale[:, None])
    probe = np.stack(probe, axis=-1)
    ref = np.column_stack((tail[:, :-1].reshape(len(h), -1), np.log1p(scale),
                          np.full(len(h), np.log1p(len(h))), probe[:, 0]))
    coeff = floor.base(h)
    paths = floor.paths(coeff)
    off = paths[:, 1:]-paths[:, :1]
    aligned = (off @ rot[:, None])/scale[:, None, None, None]
    code = np.broadcast_to(np.eye(4)[None], (len(h), 4, 4))
    option = np.concatenate((aligned.reshape(len(h), 4, -1), probe[:, 1:],
                             (probe[:, :1]-probe[:, 1:]), code), axis=2)
    return ref.clip(-20., 20.).astype(np.float32), option.clip(-40., 40.).astype(np.float32), paths, scale


def targets(paths, y, valid, scale, cutoff):
    """Future is used only for training moments, never to define the feature roster."""
    known = valid.any(1)
    actual = np.where(valid[..., None], y, 0.)
    error = np.where(valid[:, None], np.linalg.norm(paths-actual[:, None], axis=-1), 0.)
    ade = error.sum(2)/valid.sum(1).clip(1)[:, None]
    cv = ade[:, 0]
    easy = known & (cv > 0.) & (cv <= cutoff)
    change = cv[:, None]-ade[:, 1:]
    radius = np.linalg.norm(paths[:, 1:]-paths[:, :1], axis=-1).max(-1)
    unknown = np.where(known[:, None], 0., radius)
    harm = np.maximum(-change, 0.)
    moments = np.stack((change-unknown, harm+unknown, harm*easy[:, None]+unknown), axis=-1)
    ref = np.column_stack((cv, cv*easy))
    return ((moments/scale[:, None, None]).astype(np.float32), (ref/scale[:, None]).astype(np.float32))


class HarmHead(nn.Module):
    def __init__(self, actor_width, option_width):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(actor_width, 48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU())
        self.refs = nn.Linear(32, 2)
        self.action = nn.Sequential(nn.Linear(32+option_width, 48), nn.SiLU(), nn.Linear(48, 3))

    def forward(self, actor, option):
        x = self.actor(actor)
        z = self.action(torch.cat((x[:, None].expand(-1, option.shape[1], -1), option), dim=-1))
        return torch.cat((z[..., :1], F.softplus(z[..., 1:])), dim=-1), F.softplus(self.refs(x))


def loss(out, ref, target, reference):
    gain = F.smooth_l1_loss(out[..., 0], target[..., 0], beta=.2)
    # The optimum for a nonnegative moment is its conditional mean, not
    # a mean on the log scale (which understates rare large harm).
    deviance = lambda p, y: (p-y*p.clamp_min(1e-8).log()).mean()
    harms = deviance(out[..., 1:], target[..., 1:])
    mass = deviance(ref, reference)
    return gain+harms+mass


def choose(moments, reference, legal):
    """One query only. All decisions use predicted moments, not observed labels."""
    if (moments.shape[:2] != legal.shape or moments.shape[-1] != 3
            or reference.shape != (len(moments), 2) or not np.isfinite(moments).all()
            or not np.isfinite(reference).all() or (reference < 0.).any() or (moments[..., 1:] < 0.).any()):
        raise ValueError("Invalid predicted action moments")
    gain, harms = moments[..., 0], moments[..., 1:]
    allowed = legal & (gain > 0.)
    i = np.arange(len(gain))

    def best(use):
        q = np.where(use, gain, -np.inf)
        j = q.argmax(1)
        return np.where(use.any(1), j+1, 0), j, q[i, j]

    a, j, g = best(allowed)
    individual, _, _ = best(allowed & (reference[:, None] > 0.).all(2)
                            & (harms <= TOLERANCE*reference[:, None]).all(2))
    order = np.flatnonzero(a)
    order = order[np.argsort(-g[order], kind="stable")]
    # A separate gain-only control inherits the same number of edits but
    # ranks them by gain alone. The scene rule ranks by predicted harm excess.
    excess = harms[i, j]-TOLERANCE*reference
    priority = g-excess.clip(0.).sum(1)
    ordered = order[np.argsort(-priority[order], kind="stable")]
    feasible = np.flatnonzero((excess[ordered].cumsum(0) <= 0.).all(1)
                              & (reference[ordered].cumsum(0) > 0.).all(1))
    count = int(feasible[-1]+1) if len(feasible) else 0
    scene, matched = np.zeros_like(a), np.zeros_like(a)
    scene[ordered[:count]] = a[ordered[:count]]
    matched[order[:count]] = a[order[:count]]
    return np.stack((a, individual, scene, matched), axis=1).astype(np.uint8)
