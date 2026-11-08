"""Separate replacement reference loss from predicted intervention gain."""

import numpy as np
from torch import nn
from torch.nn import functional as F


def reference_targets(gain_has_reference, old_support_has_reference, candidate_has_reference):
    unique_old = bool(old_support_has_reference and not gain_has_reference)
    missing_new = ~np.asarray(candidate_has_reference, dtype=bool)
    return np.column_stack((unique_old & missing_new, np.full(len(missing_new), unique_old), missing_new))


def network(width):
    return nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 24),
                         nn.SiLU(), nn.Linear(24, 3))


def loss(net, x, y):
    z = net(x)
    first = F.binary_cross_entropy_with_logits(z[:, :2], y[:, :2])
    conditional = F.binary_cross_entropy_with_logits(z[:, 2], y[:, 2], reduction="none")
    return first + .25*(conditional*y[:, 1]).sum()/y[:, 1].sum().clamp(min=1.)


def scores(logits):
    p = np.exp(-np.logaddexp(0., -np.asarray(logits)))
    return np.column_stack((p[:, 0], p[:, 1]*p[:, 2]))


def matched_easy(easy, keep):
    """Outcome-free, retrospective count match among already emitted proposals."""
    out = np.zeros(len(keep), dtype=bool)
    out[np.argsort(-np.asarray(easy), kind="stable")[:np.count_nonzero(keep)]] = True
    return out
