"""Exact one-edit support labels under the unchanged CV recording risk rule."""

import numpy as np
from torch import nn
from torch.nn import functional as F


def targets(cv_error, candidate_error, known, easy_cut, weights, ols_minus_cv_lower):
    """One known easy edit with positive gain automatically satisfies 2%."""
    cv = np.asarray(cv_error)
    delta = cv[:, None]-candidate_error
    cv_supported = (np.asarray(known) & (cv > 0.) & (cv <= easy_cut))[:, None] & (delta > 0.)
    both = cv_supported & (ols_minus_cv_lower+weights[:, None]*delta > 0.)
    return np.stack((cv_supported, both), axis=-1)


def network(width):
    return nn.Sequential(nn.Linear(width, 32), nn.Tanh(), nn.Linear(32, 2))


def loss(net, x, y):
    return F.binary_cross_entropy_with_logits(net(x), y)
