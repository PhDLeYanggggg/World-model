"""Factor missing-outcome probability from conditional forecast costs."""

import numpy as np
from torch import nn
from torch.nn import functional as F

MOTION = 13
GEOMETRY = 6


def observability_features(history, groups):
    v = np.diff(history, axis=1)
    speed = np.linalg.norm(v, axis=-1)
    acc = np.linalg.norm(np.diff(v, axis=1), axis=-1)
    x = np.zeros((len(history), MOTION+GEOMETRY), dtype=np.float32)
    x[:, :MOTION] = np.log1p(np.c_[speed, acc]).clip(0., 12.)
    for q in groups:
        pos = history[q, -1]
        diff = pos-np.median(pos, axis=0)
        dist = np.linalg.norm(diff, axis=1)
        spread = max(1., float(np.median(dist)))
        d = np.linalg.norm(pos[:, None]-pos[None], axis=2)
        np.fill_diagonal(d, np.inf)
        near = d.min(1) if len(q) > 1 else np.zeros(1)
        velocity = v[q, -3:].mean(1)
        outward = (diff*velocity).sum(1)/(dist.clip(1e-8)*np.linalg.norm(velocity, axis=1).clip(1e-8))
        x[q, MOTION:] = np.c_[np.full(len(q), np.log1p(len(q))), np.log1p(dist/spread),
                             np.log1p(near/spread), outward, diff/spread].clip(-12., 12.)
    return x


def costs(conditional, p_missing, radius):
    """Compose unconditional gain/excess estimates with the known path radius."""
    p = np.broadcast_to(p_missing, radius.shape)
    debit = p*radius
    z = conditional*(1.-p[..., None])
    return z + np.stack((-debit, debit, debit), axis=-1)


class Head(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conditional = nn.Sequential(nn.Linear(width, 40), nn.Tanh(), nn.Linear(40, 24),
                                         nn.Tanh(), nn.Linear(24, 3))
        self.motion = nn.Sequential(nn.Linear(MOTION, 24), nn.Tanh(), nn.Linear(24, 1))
        self.geometry = nn.Sequential(nn.Linear(MOTION+GEOMETRY, 24), nn.Tanh(), nn.Linear(24, 1))

    def forward(self, x):
        a, b = x[:, :-(MOTION+GEOMETRY)], x[:, -(MOTION+GEOMETRY):]
        return self.conditional(a), self.motion(b[:, :MOTION]), self.geometry(b)


def loss(net, x, y):
    z, m, g = net(x)
    missing = y[:, 3:4]
    conditional = (F.smooth_l1_loss(z, y[:, :3], beta=.1, reduction="none")
                   * (1.-missing)).sum()/((1.-missing).sum().clamp(min=1.)*3.)
    return conditional+.25*(F.binary_cross_entropy_with_logits(m, missing)
                             +F.binary_cross_entropy_with_logits(g, missing))
