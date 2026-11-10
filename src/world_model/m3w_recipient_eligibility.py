"""Predict the eligibility and value of one matched-budget recipient swap."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

THRESHOLD = .5


def network(width):
    return nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 48), nn.SiLU(),
                         nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, 2))


def loss(pred, target):
    gain = F.smooth_l1_loss(pred[:, 0], target[:, 0], beta=.2)
    eligibility = F.binary_cross_entropy_with_logits(pred[:, 1], target[:, 1])
    return gain + eligibility, gain, eligibility


def rank(normalized_gain, probability, distance):
    if (not len(distance) or normalized_gain.shape != distance.shape or probability.shape != distance.shape
            or not np.isfinite(normalized_gain).all() or not np.isfinite(probability).all()
            or np.any((probability < 0.) | (probability > 1.)) or not np.isfinite(distance).all()
            or (distance <= 0.).any()):
        raise ValueError('Expected a finite, nonempty legal menu')
    gain = normalized_gain * distance
    eligible = probability >= THRESHOLD
    best_gain = int(gain.argmax())
    best_risk = (int(np.where(eligible, gain, -np.inf).argmax()) if eligible.any()
                 else int(np.lexsort((gain, probability))[-1]))
    guarded = best_risk if eligible[best_risk] and gain[best_risk] > 0. else -1
    return best_gain, best_risk, guarded
