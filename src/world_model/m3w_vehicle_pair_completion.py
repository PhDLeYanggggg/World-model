"""Separate observed pair utility from unavailable-step costs."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.world_model import m3w_vehicle_pair_learning as base

ARMS = ('direct_aux', 'factorized')


class PairValue(base.PairValue):
    def __init__(self):
        super().__init__(joint=True)
        self.head[-1] = nn.Linear(32, 5)


def targets(old, observed, scale, arm):
    known = np.asarray(observed).reshape(-1, 1)/scale
    if arm == 'direct_aux':
        return np.column_stack((old, known)).astype(np.float32)
    if arm != 'factorized':
        raise ValueError('Unknown pair completion target')
    debit = known-old[:, 2:4]
    if np.any(debit[:, 0] < -1e-5):
        raise ValueError('Missing CV disagreement cannot increase conservative gain')
    return np.column_stack((old[:, :2], known, debit)).astype(np.float32)


def loss(net, batch):
    a, b, relative, truth = batch
    prediction = net(a, b, relative)
    return (F.binary_cross_entropy_with_logits(prediction[:, :2], truth[:, :2])
            + F.smooth_l1_loss(prediction[:, 2:], truth[:, 2:], beta=.1))


def merit(raw, arm):
    if arm == 'direct_aux':
        return base.merit(raw[:, :4])
    if arm != 'factorized':
        raise ValueError('Unknown pair completion rank')
    cv = raw[:, 2]-np.maximum(raw[:, 3], 0.)
    ols = raw[:, 2]-raw[:, 4]
    return base.merit(np.column_stack((raw[:, :2], cv, ols)))
