"""Candidate-conditioned gain and risk predictions from causal geometry."""

import numpy as np
import torch
from torch import nn

from src.world_model.m3w_donor_motion_shape import motion_frame
from src.world_model.m3w_opendd_hardness_gate import features as motion_features


def features(history, cv, path):
    _, rot, scale = motion_frame(history)
    z = ((path-cv) @ rot)/scale[:, None, None]
    if z.shape != (len(history), 12, 2) or not np.isfinite(z).all():
        raise ValueError("Expected finite causal candidate paths")
    radius = np.linalg.norm(z, axis=-1).max(1)
    return np.c_[motion_features(history), z.reshape(len(z), 24), np.log1p(radius)].clip(-16., 16.).astype(np.float32), scale


def network(width):
    return nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 5))


def loss(net, x, target):
    z = net(x)
    err = nn.functional.smooth_l1_loss(z[:, :4], target[:, :4], beta=.05, reduction="none")
    support = nn.functional.binary_cross_entropy_with_logits(z[:, 4], target[:, 4])
    return (err*z.new_tensor([1., 2., 4., 1.])).mean() + .10*support


def scores(z, scale):
    value = z[:, :4]*scale[:, None]
    risk_penalty = np.maximum(value[:, 1:], 0.) @ np.array([2., 4., 1.])
    return value[:, 0], value[:, 0]-risk_penalty


def ordered_pairs(sites):
    """One producer per unordered exclusion pair, evaluated on both endpoints."""
    sites = tuple(sorted(sites))
    if len(sites) < 3 or len(set(sites)) != len(sites):
        raise ValueError("Require distinct fitting and excluded sources")
    for i, a in enumerate(sites):
        for b in sites[i+1:]:
            yield (a, b), tuple(s for s in sites if s not in (a, b))
