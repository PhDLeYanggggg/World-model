"""Label-blind exact exchange menus and a relative transition head."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.evaluation.m3w_count_motion_capacity import respects_tokens, within_cap


def exchange_menu(anchor, legal, movement, cap):
    q = np.arange(len(anchor))
    spend = movement[q, anchor].sum()
    if (not legal[q, anchor].all() or not within_cap(spend, cap) or not respects_tokens(anchor, legal[:, 1])):
        raise ValueError('Invalid frozen exchange anchor')
    destinations = q[(anchor < 2) & legal[q, np.minimum(anchor + 1, 2)]]
    result = []
    for src in np.flatnonzero(anchor):
        drop = movement[src, anchor[src]] - movement[src, anchor[src] - 1]
        dst = destinations[destinations != src]
        extra = movement[dst, anchor[dst]+1] - movement[dst, anchor[dst]]
        for j in dst[within_cap(spend - drop + extra, cap)]:
            p = exchanged(anchor, src, j)
            if within_cap(movement[q, p].sum(), cap) and respects_tokens(p, legal[:, 1]):
                result.append((src, j))
    return np.asarray(result, dtype=int).reshape(-1, 2)


def exchanged(anchor, src, dst):
    p = anchor.copy()
    p[src] -= 1
    p[dst] += 1
    return p


def features(x, legal, movement, anchor, cap, pairs):
    src, dst = pairs.T
    scale = max(float(cap), np.finfo(float).tiny)
    if len(pairs) and cap <= 0.:
        raise ValueError('Positive edit pairs require positive cap')
    removed = movement[src, anchor[src]] - movement[src, anchor[src]-1]
    added = movement[dst, anchor[dst]+1] - movement[dst, anchor[dst]]
    n = max(len(anchor)-1, 1)
    pair = np.column_stack((x[src], x[dst], anchor[src], anchor[dst], src/n, dst/n,
                            removed/scale, added/scale))
    summary = np.r_[(x * anchor[:, None]).sum(0) / max(1, anchor.sum()),
                    x[legal[:, 1]].mean(0) if legal[:, 1].any() else np.zeros(x.shape[1]),
                    np.log1p(anchor.sum()), np.log1p(legal[:, 1].sum()),
                    movement[np.arange(len(anchor)), anchor].sum()/scale]
    context = np.broadcast_to(summary, (len(pairs), len(summary)))
    return np.column_stack((pair, context)).astype(np.float32), (added-removed)


class ExchangeRanker(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 4))

    def forward(self, x):
        return self.model(x)


def objective(pred, transition, gain):
    return (F.cross_entropy(pred[:, :3], transition, weight=pred.new_tensor([2., 1., 2.]))
            + F.smooth_l1_loss(pred[:, 3], gain, beta=.05))


def rank(pred, *, margin=.10, guard=True):
    p = torch.softmax(pred[:, :3], dim=1)
    s = p[:, 2] - p[:, 0]
    if guard:
        s = s.masked_fill(pred[:, 3] <= 0., -torch.inf)
    if not len(s):
        return None
    i = int(s.argmax())
    return i if float(s[i]) > margin else None
