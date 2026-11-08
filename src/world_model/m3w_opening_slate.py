"""Diverse first-offer shortlists and matched pointwise/listwise models."""

import numpy as np
from torch import nn
from torch.nn import functional as F


def shortlist(score, row, displacement, size, mode):
    order = np.argsort(-score, kind="stable")
    if mode == "flat":
        return order[:size]
    angle = np.arctan2(displacement[:, 1], displacement[:, 0])
    sector = np.floor((angle+np.pi)*4./np.pi).astype(int) % 8
    seen = set()
    unique, rest = [], []
    for j in order:
        k = int(row[j]) if mode == "row" else int(sector[j])
        (rest if k in seen else unique).append(j)
        seen.add(k)
    return np.array((unique+rest)[:size], dtype=int)


def network(width):
    return nn.Sequential(nn.Linear(width, 32), nn.Tanh(), nn.Linear(32, 16),
                         nn.Tanh(), nn.Linear(16, 1))


def pointwise(net, x, quality):
    return F.smooth_l1_loss(net(x).squeeze(-1), quality, beta=.2)


def listwise(net, x, quality):
    score = net(x).squeeze(-1)
    target = F.softmax(quality/.25, dim=1)
    return F.smooth_l1_loss(score, quality, beta=.2)+.5*(
        -(target*F.log_softmax(score, dim=1)).sum(1).mean())
