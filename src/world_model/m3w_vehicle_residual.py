"""Bounded CV residual and matched observed/complete-grid training objectives."""

import numpy as np
import torch

from src.world_model import m3w_opendd_velocity_candidate as motion

PEAK_BOUND = 2.


def encode(history):
    rotation, scale = motion.frames(history)
    h = ((history-history[:, -1:]) @ rotation) / scale[:, None, None]
    x = np.column_stack((h.reshape(len(history), 16), np.log1p(scale)))
    cv = (h[:, -1:] - h[:, -2:-1]) * np.arange(1., 13.)[None, :, None]
    return x.astype(np.float32), cv.astype(np.float32), rotation, scale


def predict(net, x, cv):
    delta = motion.decode(net(x))
    peak = delta.norm(dim=-1).amax(dim=-1).clamp_min(PEAK_BOUND)
    return cv + delta * (PEAK_BOUND / peak)[:, None, None]


def loss(pred, cv, target, valid, *, completion):
    target = torch.where(valid[..., None], target, 0.)
    err = (pred-target).norm(dim=-1)
    known = torch.where(valid, err, 0.).sum(dim=1)
    if not completion:
        return (known / pred.shape[1]).mean()
    debit = torch.where(valid, 0., (pred-cv).norm(dim=-1)).sum(dim=1)
    return ((known + debit) / pred.shape[1]).mean()
