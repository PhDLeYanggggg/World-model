"""Qualification logits for release of a fixed two-edit proposal."""

import torch
from torch.nn import functional as F


def label_targets(target):
    risk, dual = target[:, 0] > .5, target[:, 1] > .5
    full = dual & (target[:, 4] >= 0.) & (target[:, 5] > 0.)
    return torch.stack((risk, dual, full), dim=1)


def loss(net, batch, nested):
    *inputs, values = batch
    y = label_targets(values)
    p = net(*inputs)[:, :3]
    target = y.to(p.dtype)
    bce = F.binary_cross_entropy_with_logits(p, target, reduction='none')
    if not nested:
        return bce.mean()
    conditional = (y[:, 0], y[:, 1])
    terms = [bce[:, 0].mean()]
    for k, mask in enumerate(conditional, 1):
        terms.append((bce[:, k]*mask).sum()/mask.sum().clamp(min=1))
    return sum(terms)/3.


def log_score(logits, nested):
    p = F.logsigmoid(logits[:, :3])
    return p.sum(dim=1) if nested else p[:, 2]
