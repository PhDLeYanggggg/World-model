"""Causal recipient assignment for one or two fixed displacement slots."""

import numpy as np
import torch
from torch import nn

from src.evaluation.m3w_opendd_location import paired
from src.evaluation.m3w_recording_switch_ceiling import moments
from src.world_model import m3w_shape_menu_value as shape

ARMS = ("frozen_prefix", "gain", "support", "joint_local", "joint_prefix")
AMPS = (.25, .5, .75, 1.)


def features(history, amplitude):
    x, cv, ols, bank, _, _, scale = shape.features(history, amplitude)
    backtest = shape.menu.prefix_rank(history, amplitude, amplitude > 0.)
    best = backtest.mean(0).argmax(1)
    return x, cv, ols, bank[np.arange(len(bank)), best], scale


def target(cv, ols, path, y, valid, cut, scale):
    m = moments(cv, path, y, valid, cut)
    gain = paired(cv, path, y, valid)[1].mean(1)
    ols_gain = paired(ols, path, y, valid)[1].mean(1) - paired(ols, cv, y, valid)[1].mean(1)
    v = np.stack((gain, ols_gain, m["all_excess"], m["easy_excess"], m["easy_signed"]), axis=1)
    return np.c_[(v/scale[:, None]).clip(-8., 8.), m["easy"]].astype(np.float32)


class Value(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 6))

    def forward(self, x):
        return self.net(x)


def objective(net, x, y):
    z = net(x)
    err = nn.functional.smooth_l1_loss(z[:, :5], y[:, :5], beta=.025, reduction="none")
    return (err*z.new_tensor([1., 1., 2., 4., 1.])).mean() + .10*nn.functional.binary_cross_entropy_with_logits(z[:, 5], y[:, 5])


def feasible(roster, radii, native_peak):
    """Within-query replacement cannot exceed the recipient's native OLS4 bridge cap."""
    rows = np.asarray(roster, dtype=int)
    r = np.asarray(radii, dtype=float)
    cap = np.asarray(native_peak, dtype=float)
    if (len(r) not in (1, 2) or len(rows) <= len(r) or (r <= 0.).any()
            or len(set(rows)) != len(rows) or not np.isfinite(r).all() or not np.isfinite(cap[rows]).all()):
        raise ValueError("Require positive fixed slots and at least one unedited recipient")
    slot, index = np.nonzero((cap[rows][None] > 0.) & (cap[rows][None]+1e-10 >= r[:, None]))
    picked = rows[index]
    return slot, picked, (r[slot]/cap[picked]).clip(max=1.)


def assign(rows, slots, value, probability, n_slots, coverage_weight=0.):
    """Jointly maximize value plus a marginal coverage proxy; no independence guarantee."""
    first, = np.nonzero(slots == 0)
    if n_slots == 1:
        return first[[(value[first]+coverage_weight*probability[first]).argmax()]]
    second, = np.nonzero(slots == 1)
    p0, p1 = probability[first, None], probability[None, second]
    z = value[first, None]+value[None, second]+coverage_weight*(p0+p1-p0*p1)
    z = np.where(rows[first, None] != rows[None, second], z, -np.inf)
    if not np.isfinite(z).any():
        raise ValueError("No distinct assignment for two frozen slots")
    i, j = np.unravel_index(z.argmax(), z.shape)
    return np.array([first[i], second[j]])
