"""Learn displacement directions, then project them to a fixed two-edit budget."""

import numpy as np
import torch
from torch import nn

from src.world_model import m3w_av2_steered_reference as steer


ARMS = ("soft", "ols4_projection", "raw_projection", "guarded_projection")


def inputs(d, anchor, rows):
    rows = np.asarray(rows)
    c, o, h = (d[k][rows] for k in ("cv", "ols4", "h"))
    offset = anchor[rows]-c
    radius = np.linalg.norm(offset, axis=-1)
    scale = np.linalg.norm(np.diff(h, axis=1), axis=-1).mean(1).clip(.05)
    u = h[:, -1]-h[:, 0]
    norm = np.linalg.norm(u, axis=1)
    u = u/np.maximum(norm, 1e-8)[:, None]
    u[norm < 1e-8] = [1., 0.]
    frame = np.stack((u, np.column_stack((-u[:, 1], u[:, 0]))), axis=-1)
    encode = lambda p: ((p@frame)/scale[:, None, None]).reshape(len(rows), -1)
    x = np.column_stack((encode(h-h[:, -1:]), encode(c), encode(o-c),
                        radius/scale[:, None], np.log1p(scale)))
    return x.clip(-32., 32.).astype(np.float32), c, o, offset, frame, scale


def head(width):
    return nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 24))


def residual(net, x, frame, scale):
    z = net(x).reshape(*x.shape[:-1], 12, 2)
    t = torch.arange(1, 13, dtype=z.dtype, device=z.device).view(12, 1)
    return ((.5*t*z.tanh()) @ frame.transpose(-1, -2))*scale[..., None, None]


def project(direction, offset):
    norm = torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
    r = torch.linalg.vector_norm(offset, dim=-1, keepdim=True)
    return torch.where(norm > 1e-6, r*direction/norm.clamp_min(1e-6), offset)


def target(d, rows, y, valid, cutoff):
    return steer.label(d, rows, y, valid, cutoff)


def objectives(r, cv, ols, offset, scale, y, valid, easy, context):
    norm = lambda z: torch.linalg.vector_norm(z, dim=-1)
    avg_scale = scale.mean(1)
    denom = valid.sum((1, 2)).clamp_min(1)*avg_scale
    raw = (norm(cv+r-y)*valid).sum((1, 2))/denom
    change = project(r, offset)
    p = cv+change
    err, base, other = (norm(a-y) for a in (p, cv, ols))
    known = (err*valid).sum((1, 2))/denom
    gain_cv = torch.where(valid, base-err, -norm(change)).sum((1, 2))
    gain_ols = torch.where(valid, other-err, -norm(p-ols)).sum((1, 2))+context[:, 0]
    floors = ((-gain_cv).relu()+(-gain_ols).relu())/denom
    count = valid.sum(2).clamp_min(1)
    avg = ((err-base)*valid).sum(2)/count
    ref = (base*valid).sum(2)/count
    debit = torch.where(valid.any(2), 0., norm(change).amax(2))
    risk_all = (avg.relu()+debit-.02*ref).sum(1)
    risk_easy = ((avg.relu()-.02*ref)*easy+debit).sum(1)
    signed = (avg*easy).sum(1)-.02*context[:, 1]
    risk = torch.stack((risk_all, risk_easy, signed), 1).relu().sum(1)/(2.*avg_scale)
    return raw.mean(), known.mean(), floors.mean(), risk.mean()
