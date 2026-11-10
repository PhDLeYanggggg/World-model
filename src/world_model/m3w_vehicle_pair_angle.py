"""Matched pair models for the incremental value of changing angles."""

import numpy as np
import torch
from torch import nn

from src.world_model.m3w_vehicle_nominee_rank import WIDTH


def features(x, scales):
    if x.shape != (3, WIDTH) or scales.shape != (3, 2) or not np.isfinite(scales).all() or (scales < 0.).any():
        raise ValueError('Expected incumbent plus two alternatives')
    ratios = scales[1:]/scales[1:].max(0).clip(1e-12)
    pair = np.column_stack((np.broadcast_to(x[0], x[1:].shape), x[1:], ratios)).astype(np.float32)
    return pair, ratios


def network():
    return nn.Sequential(nn.Linear(2*WIDTH+2, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 3))


def loss(net, batch, direct):
    x, y = batch
    pred = net(x)
    use = y[:, 2:4] if direct else y[:, :2]
    fit = nn.functional.smooth_l1_loss(pred[:, :2].tanh(), use, beta=.15)
    return fit+.5*nn.functional.binary_cross_entropy_with_logits(pred[:, 2], y[:, 4])


def choose(logits, ratio):
    if logits.shape[-2:] != (2, 3) or ratio.shape != logits[..., :2].shape or not np.isfinite(logits).all():
        raise ValueError('Invalid pair predictions')
    z = np.tanh(logits[..., :2])*ratio
    p = 1./(1.+np.exp(-logits[..., 2].clip(-80., 80.)))
    floor = np.zeros((len(z), 1))
    score = np.maximum(z.min(-1), 0.)*p
    return (np.column_stack((floor, z[..., 0])).argmax(1), np.column_stack((floor, score)).argmax(1),
            score.max(1))
