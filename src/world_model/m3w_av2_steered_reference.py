"""Change the direction of two fixed edits while preserving their radial budgets."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ARMS = ("ade", "guarded")
MAX_ANGLE = np.pi/4.
KNOTS = 3


def inputs(d, anchor):
    cv, ols, h = (np.asarray(d[k], dtype=float) for k in ("cv", "ols4", "h"))
    rows = np.flatnonzero(np.any(anchor != cv, axis=(1, 2)))
    if len(rows) != 2:
        return None
    h = h[rows]
    speed = np.linalg.norm(np.diff(h, axis=1), axis=-1).mean(1).clip(.05)
    direction = h[:, -1]-h[:, 0]
    norm = np.linalg.norm(direction, axis=1)
    u = direction/np.where(norm > 1e-8, norm, 1.)[:, None]
    u[norm <= 1e-8] = (1., 0.)
    frame = np.stack((u, np.column_stack((-u[:, 1], u[:, 0]))), axis=-1)
    encode = lambda x: ((x@frame)/speed[:, None, None]).reshape(2, -1)
    c, o, delta = cv[rows], ols[rows], anchor[rows]-cv[rows]
    x = np.column_stack((encode(h-h[:, -1:]), encode(c), encode(delta), encode(o-c), np.log1p(speed)))
    return rows, x.clip(-32., 32.).astype(np.float32), c, o, delta, float(speed.mean())


class Steer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, KNOTS))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        z = self.layers(x)
        return MAX_ANGLE*F.interpolate(z.reshape(-1, 1, KNOTS), size=12, mode="linear", align_corners=True).reshape(*x.shape[:-1], 12).tanh()


def rotate(delta, angles, lib):
    cs, sn = lib.cos(angles), lib.sin(angles)
    x, y = delta[..., 0], delta[..., 1]
    return lib.stack((cs*x-sn*y, sn*x+cs*y), -1)


def label(d, recipient, y, valid, cut):
    cv, ols = d["cv"], d["ols4"]
    safe = np.where(valid[..., None], y, 0.)
    c, o = (np.linalg.norm(p-safe, axis=-1) for p in (cv, ols))
    mean = np.where(valid, c, 0.).sum(1)/valid.sum(1).clip(1)
    easy = (mean > 0.) & (mean <= cut)
    outside = np.ones(len(cv), dtype=bool)
    outside[recipient] = False
    gain_other = np.where(valid, o-c, -np.linalg.norm(ols-cv, axis=-1))[outside].sum()
    return safe[recipient], valid[recipient], easy[recipient], np.array([gain_other, mean[easy].sum()])


def objective(angles, c, o, delta, scale, y, v, easy, context, guarded):
    path = c+rotate(delta, angles, torch)
    norm = lambda x: torch.linalg.vector_norm(x, dim=-1)
    e, cv, ols = (norm(x-y) for x in (path, c, o))
    denom = v.sum((1, 2)).clamp_min(1)*scale
    ade = (e*v).sum((1, 2))/denom
    if not guarded:
        return ade.mean()
    gain_cv = torch.where(v, cv-e, -norm(delta)).sum((1, 2))
    gain_ols = torch.where(v, ols-e, -norm(path-o)).sum((1, 2))+context[:, 0]
    avg = ((e-cv)*v).sum(2)/v.sum(2).clamp_min(1)
    ref = (cv*v).sum(2)/v.sum(2).clamp_min(1)
    debit = torch.where(v.any(2), 0., norm(delta).amax(2))
    all_excess = (avg.relu()+debit-.02*ref).sum(1)
    easy_excess = ((avg.relu()-.02*ref)*easy+debit).sum(1)
    signed = (avg*easy).sum(1)-.02*context[:, 1]
    risk = torch.stack((all_excess, easy_excess, signed), 1).relu().sum(1)/(2.*scale)
    floors = ((-gain_cv).relu()+(-gain_ols).relu())/denom
    return (ade+floors+2.*risk).mean()
