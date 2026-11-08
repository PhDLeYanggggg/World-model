"""Train candidate moments and one-edit recording events in the same head."""

import numpy as np
import torch
from scipy.special import expit
from torch import nn


def labels(gain, risk, easy, ols_offset, easy_mass):
    safe = easy & (risk <= [0., 0., .02*easy_mass]).all(1)
    qualified = safe & (gain[:, 0] > 0.) & (gain[:, 1]+ols_offset > 0.)
    return np.stack((easy, safe, qualified), axis=1)


def network(width):
    return nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 9))


def loss(net, x, y):
    z = net(x)
    moment = nn.functional.smooth_l1_loss(z[:, :5], y[:, :5], beta=.05, reduction="none")
    e, s, q = y[:, 5:].T

    def masked(pred, target, weight):
        err = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction="none")
        return (err*weight).sum()/weight.sum().clamp(min=1.)

    binary = nn.functional.binary_cross_entropy_with_logits(z[:, [5, 8]], y[:, [5, 7]])
    binary += .5*masked(z[:, 6], s, e) + .5*masked(z[:, 7], q, s)
    return (moment*z.new_tensor([1., 1., 2., 3., 1.])).mean()+.10*binary


def scores(z):
    p = expit(z[:, 5:])
    return dict(direct=p[:, 3], factorized=p[:, :3].prod(1))
