"""Treat missing relative-motion targets as intervals during fitting."""

import numpy as np
from torch.nn import functional as F


def relative_bounds(paths, reference, target, valid):
    """Full-grid paired relative-motion bounds with observed labels fixed."""
    i, j = np.triu_indices(len(reference), 1)
    if not len(i):
        return np.zeros((len(paths), 2))
    visible = valid[i] & valid[j]
    ref = reference[i]-reference[j]
    pred = paths[:, i]-paths[:, j]
    y = np.where(visible[..., None], target[i]-target[j], 0.)
    known = np.where(visible, np.linalg.norm(ref-y, axis=-1)-np.linalg.norm(pred-y, axis=-1), 0.).sum((1, 2))
    debit = np.where(visible, 0., np.linalg.norm(pred-ref, axis=-1)).sum((1, 2))
    return np.column_stack((known-debit, known+debit))/visible.size


def loss(net, batch):
    x, y = batch
    pred = net(x)
    estimate = pred[:, :2].tanh()
    low, high = y[:, 1], y[:, 3]
    relative_target = estimate[:, 1].detach().clamp(min=low, max=high)
    traj = F.smooth_l1_loss(estimate[:, 0], y[:, 0], beta=.15)
    relative = F.smooth_l1_loss(estimate[:, 1], relative_target, beta=.15)
    return .5*(traj+relative)+.5*F.binary_cross_entropy_with_logits(pred[:, 2], y[:, 2])
