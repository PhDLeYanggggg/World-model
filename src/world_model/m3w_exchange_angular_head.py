"""Learn smooth residual direction while preserving the correction norm."""

import numpy as np
import torch
from torch import nn


def features(history, cv, ols, fraction):
    h = history-history[:, -1:]
    v = ols[:, -1]-cv[:, -1]
    length = np.linalg.norm(v, axis=-1)
    u = np.divide(v, length[:, None], out=np.broadcast_to([1., 0.], v.shape).copy(), where=length[:, None] > 1e-10)
    scale = np.linalg.norm(np.diff(h, axis=1), axis=-1).mean(1).clip(.05)
    local = np.stack(((h*u[:, None]).sum(-1), h[..., 1]*u[:, None, 0]-h[..., 0]*u[:, None, 1]), axis=-1)
    x = np.column_stack(((local/scale[:, None, None]).reshape(len(h), -1), fraction,
                         np.log1p(scale), length/(cv.shape[1]*scale))).clip(-16., 16.)
    return x.astype(np.float32), scale


class AngularHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def angle(z, steps):
    t = torch.linspace(-1., 1., steps, device=z.device, dtype=z.dtype)
    return torch.pi/2*torch.tanh(z[:, :1]+z[:, 1:]*t)


def rotate(z, cv, anchor):
    t = angle(z, cv.shape[1])
    x, y = (anchor-cv).unbind(-1)
    c, s = t.cos(), t.sin()
    return cv+torch.stack((x*c-y*s, x*s+y*c), dim=-1)


def losses(pred, cv, ols, target, valid, scale):
    e = torch.linalg.vector_norm(pred-target, dim=-1)
    n = pred.shape[1]*scale
    observed = (e*valid).sum(1)/n

    def gain(ref):
        e0 = torch.linalg.vector_norm(ref-target, dim=-1)
        debit = torch.linalg.vector_norm(ref-pred, dim=-1)
        return ((e0-e)*valid-debit*(~valid)).sum(1)/n

    return observed.mean(), -torch.minimum(gain(cv), gain(ols)).mean()
