"""Gain and risk readout for a fixed two-agent position projection."""

import numpy as np
import torch
from torch import nn

from src.evaluation.m3w_cv_anchor_budget import candidate_terms
from src.world_model import m3w_av2_assignment_constraints as constraints
from src.world_model import m3w_av2_assignment_cost as cost

ARMS = ("transferred_cost", "risk_replay", "gain_only", "gain_risk")


def inputs(d, bank, opts, cut):
    context, _, pair, scale = cost.inputs(d, bank, opts, cut)
    return constraints.features(context, pair), scale


def targets(d, bank, opts, y, valid, cut, scale):
    _, base, gains, _ = candidate_terms(d["cv"], d["ols4"], bank, y, valid)
    i, j = opts.T
    gain = np.tanh((gains[i, 0]+gains[j, 1]+np.array([0., base.sum()]))/(2.*y.shape[1]*scale))
    status = constraints.labels(d, bank, opts, y, valid, cut)
    return gain.astype(np.float32), status


def head(width):
    return nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, 4))


def loss(prediction, gain, status):
    f = nn.functional
    g = f.smooth_l1_loss(prediction[:, :2].tanh(), gain, beta=.2)
    support = f.binary_cross_entropy_with_logits(prediction[:, 2], status[:, 0].float())
    bce = f.binary_cross_entropy_with_logits(prediction[:, 3], status[:, 1].float(), reduction="none")
    risk = (bce*status[:, 0]).sum()/status[:, 0].sum().clamp_min(1)
    return g+support+risk, (g, support, risk)


def scores(out):
    # Same fitted trajectory gains in both arms; only the risk and support factors differ.
    gain = np.exp(np.tanh(out[..., :2]).min(-1))
    prob = 1./(1.+np.exp(-np.clip(out[..., 2:], -25., 25.)))
    return gain, gain*prob.prod(-1)
