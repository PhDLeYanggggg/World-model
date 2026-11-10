"""Predict a linear path by mixing causal, robust velocity estimates."""

import numpy as np
import torch
from torch import nn


NAMES = ('cv', 'ols4', 'ols8', 'median4', 'median8')


def features(history):
    h = np.asarray(history, dtype=float)
    if h.ndim != 3 or h.shape[1:] != (8, 2) or not np.isfinite(h).all():
        raise ValueError('Require eight finite past positions per track')
    past = h-h[:, -1:]

    def slopes(k):
        window = past[:, -k:]
        x = np.arange(k)-(k-1)/2.
        i, j = np.triu_indices(k, 1)
        return np.einsum('nkd,k->nd', window, x)/(x @ x), np.median(
            (window[:, j]-window[:, i])/(j-i)[None, :, None], axis=1)

    o4, m4 = slopes(4)
    o8, m8 = slopes(8)
    delta = np.diff(past, axis=1)
    v = np.stack((delta[:, -1], o4, o8, m4, m8), axis=1)
    scale = np.sqrt(np.square(delta).sum(2).mean(1)).clip(1e-8)
    vnorm = v/scale[:, None, None]
    x = np.column_stack((np.linalg.norm(delta, axis=2)/scale[:, None], np.linalg.norm(vnorm, axis=2),
                         np.einsum('nkd,nd->nk', vnorm[:, [0, 2, 3, 4]], vnorm[:, 1])))
    return x.astype(np.float32), v, scale


class VelocityMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(16, 16), nn.SiLU(), nn.Linear(16, len(NAMES)))

    def forward(self, x, slopes):
        w = self.net(x).softmax(1)
        return (w[..., None]*slopes).sum(1), w


def loss(velocity, cv, future, valid, *, preserve):
    t = torch.arange(1, 13, device=velocity.device)[None, :, None]

    def errors(v):
        e = torch.linalg.vector_norm(v[:, None]*t-future, dim=-1)
        return (e*valid).sum(1)/valid.sum(1).clamp(min=1)

    error = errors(velocity)
    cv_error = errors(cv)
    usable = valid.any(1)
    mae = error[usable].mean()
    pos = usable & (cv_error > 1e-6)
    guard = torch.zeros((), dtype=mae.dtype, device=mae.device)
    if preserve and pos.any():
        easy = pos & (cv_error <= torch.quantile(cv_error[pos], .25))
        guard = torch.relu(error[easy]-1.02*cv_error[easy]).mean()
    return mae+2*guard, mae, guard
