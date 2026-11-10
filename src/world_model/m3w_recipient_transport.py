"""Rank a single recipient swap without changing its displacement budget."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def proposals(history, offsets, design, cv, anchor, recipients):
    """Keep one edit, and transport the other in the new actor's heading frame."""
    recipients = np.asarray(recipients)
    n = len(cv)
    if (recipients.shape != (2,) or len(set(recipients)) != 2
            or np.any((recipients < 0) | (recipients >= n))):
        raise ValueError('Require two distinct in-query recipients')
    moving = np.linalg.norm(history[:, -1] - history[:, -4], axis=1)
    delta = (anchor - cv).astype(np.float64)
    if set(np.flatnonzero(np.any(delta != 0., axis=(1, 2)))) != set(recipients):
        raise ValueError('The incumbent does not have exactly the declared edits')
    headings = (history[:, -1] - history[:, -4]).astype(np.float64)
    headings /= np.linalg.norm(headings, axis=1, keepdims=True).clip(1e-12)
    candidates = np.setdiff1d(np.flatnonzero(moving >= .15), recipients)
    drop = np.repeat(recipients[moving[recipients] >= .15], len(candidates))
    add = np.tile(candidates, len(recipients[moving[recipients] >= .15]))
    perpendicular = headings[:, ::-1] * [-1., 1.]
    basis = np.stack((headings, perpendicular), axis=-1)
    local = delta[drop] @ basis[drop]
    transported = local @ basis[add].transpose(0, 2, 1)
    np.testing.assert_allclose(np.linalg.norm(transported, axis=-1), np.linalg.norm(delta[drop], axis=-1),
                               rtol=1e-10, atol=1e-10)
    scale = np.linalg.norm(local, axis=-1).max(axis=1).clip(1e-10)
    position = ((offsets[add]-offsets[drop])[:, None] @ basis[drop])[:, 0]
    retained = np.where(drop == recipients[0], recipients[1], recipients[0])
    context = np.column_stack((np.log1p(scale), np.sign(position)*np.log1p(np.abs(position)),
                               np.log1p(np.linalg.norm(offsets[add]-offsets[retained], axis=1)),
                               np.log1p(np.linalg.norm(offsets[drop]-offsets[retained], axis=1))))
    x = np.column_stack((design[drop], design[add], design[retained], context,
                         (local / scale[:, None, None]).reshape(len(drop), cv.shape[1]*2))).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite causal candidate features')
    return drop, add, transported + cv[add], x


def targets(cv, anchor, future, valid, drop, add, new):
    """Direct conservative improvement versus the incumbent, per swap distance."""
    def paired(before, after, ids):
        bound = np.linalg.norm(after-before, axis=-1)
        error = np.linalg.norm(before-future[ids], axis=-1) - np.linalg.norm(after-future[ids], axis=-1)
        return np.where(valid[ids], error, -bound).sum(axis=1), bound.sum(axis=1)

    a, sa = paired(anchor[drop], cv[drop], drop)
    b, sb = paired(cv[add], new, add)
    return np.clip((a+b)/(sa+sb).clip(1e-12), -1., 1.).astype(np.float32)


def network(width):
    return nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 48), nn.SiLU(),
                         nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, 1), nn.Flatten(0))


def objective(pred, target, robust):
    """Each row is an equally sized donor minibatch; compare excess vs no-swap."""
    losses = (F.smooth_l1_loss(pred, target, reduction='none', beta=.2)
              - F.smooth_l1_loss(torch.zeros_like(target), target, reduction='none', beta=.2)).mean(1)
    return (torch.logsumexp(20.*losses, dim=0)-np.log(len(losses)))/20. if robust else losses.mean()


def decision(scores):
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite predicted gain')
    return int(scores.argmax()) if len(scores) and scores.max() > 0. else -1
