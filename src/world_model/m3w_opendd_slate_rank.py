"""Within-count slate ranking with distinct gain and harm predictions."""

from itertools import combinations

import numpy as np
import torch
from torch import nn


def matched_advantage(left, right, observed, debit):
    """Observed gain and missing-step debit only on changed recipients."""
    add = np.setdiff1d(left, right, assume_unique=True)
    remove = np.setdiff1d(right, left, assume_unique=True)
    return observed[add].sum()-observed[remove].sum(), debit[add].sum()+debit[remove].sum()


def examples(packet, max_size, observed, debit, risk, proxy):
    for k in range(1, max_size+1):
        ids = np.flatnonzero(packet.size == k)
        for i, j in combinations(ids, 2):
            a, b = packet.rows[i], packet.rows[j]
            center, radius = matched_advantage(a, b, observed, debit)
            robust = np.sign(center)*max(abs(center)-radius, 0.)
            harms = risk[a].sum(0)-risk[b].sum(0)
            values = [proxy[a].sum(), proxy[b].sum()]
            target = np.r_[robust, harms, values]/packet.scale
            yield packet.features[[i, j]], target.clip(-12., 12.)


def network(width, scalar=False):
    return nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(),
                         nn.Linear(32, 1 if scalar else 4))


def loss(net, x, target, scalar=False):
    z = net(x)
    if scalar:
        return nn.functional.smooth_l1_loss(z[..., 0], target[:, 4:], beta=.05)
    diff = z[:, 0]-z[:, 1]
    gain = nn.functional.smooth_l1_loss(diff[:, 0], target[:, 0], beta=.05)
    decisive = (target[:, 0] != 0.).to(z.dtype)
    rank = nn.functional.binary_cross_entropy_with_logits(
        diff[:, 0]/.1, (target[:, 0] > 0.).to(z.dtype), reduction="none")
    harms = nn.functional.smooth_l1_loss(diff[:, 1:], target[:, 1:4], beta=.05)
    return gain+.1*(rank*decisive).mean()+harms


def decision(scores, anchor, guard=False):
    """The first candidate is not required to be the anchor."""
    if not np.isfinite(scores).all():
        raise ValueError("Ranker emitted a non-finite score")
    if not guard:
        return int(np.argmax(scores[:, 0]))
    delta = scores-scores[anchor]
    good = (delta[:, 0] > 0.) & (delta[:, 1:] <= 0.).all(1)
    return int(np.argmax(np.where(good, delta[:, 0], -np.inf))) if good.any() else int(anchor)
