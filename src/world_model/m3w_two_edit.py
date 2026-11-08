"""Matched two-edit controls with a shared candidate-level value model."""

import numpy as np
import torch
from scipy.special import expit
from torch import nn


def pairs(rows, query, capacity):
    i, j = np.triu_indices(len(rows), k=1)
    legal = (rows[i] != rows[j]) & ((query[i] != query[j]) | (np.asarray(capacity)[query[i]] >= 2))
    return i[legal], j[legal]


def rank(value, left, right, separation_debit):
    gain, harm = value[:, :2], value[:, 2:5]
    w = np.array([2., 4., 1.])
    p = expit(value[:, 5])
    independent = gain.min(1)-np.maximum(harm, 0.)@w+.025*p
    joint_gain = (gain[left]+gain[right]).min(1)
    joint = joint_gain-np.maximum(harm[left]+harm[right], 0.)@w+.025*(p[left]+p[right]-p[left]*p[right])
    return dict(independent=(independent[left]+independent[right]).argmax(),
                joint=joint.argmax(), joint_geometry=(joint-.10*np.asarray(separation_debit)).argmax())


def network(width):
    return nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 6))


def loss(net, x, y):
    z = net(x)
    err = nn.functional.smooth_l1_loss(z[:, :5], y[:, :5], beta=.05, reduction="none")
    return (err*z.new_tensor([1., 1., 2., 3., 1.])).mean() + .10*nn.functional.binary_cross_entropy_with_logits(z[:, 5], y[:, 5])
