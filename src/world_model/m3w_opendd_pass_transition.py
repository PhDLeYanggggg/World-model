"""Predict explicit rescue and destruction events for a support candidate."""

import numpy as np
from torch import nn
from torch.nn import functional as F


def targets(original_pass, candidate_pass, loses_reference):
    passed = np.asarray(candidate_pass, bool)
    return np.column_stack((passed & (not original_pass), ~passed & original_pass,
                            np.asarray(loses_reference, bool))).astype(np.float32)


def probabilities(logits):
    return np.exp(-np.logaddexp(0., -np.asarray(logits)))


def advantage(p, rescue_only=False):
    return p[:, 0] if rescue_only else p[:, 0]-p[:, 1]


def network(width):
    return nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 24),
                         nn.SiLU(), nn.Linear(24, 3))


def loss(net, x, y):
    z = net(x)
    # A rescue is defined only for a previously failing recording, and vice versa.
    # Store both unconditional event targets so the head need not see prior outcomes.
    return F.binary_cross_entropy_with_logits(z, y)
