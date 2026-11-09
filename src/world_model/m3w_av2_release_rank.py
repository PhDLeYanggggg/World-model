"""Query-balanced strict qualification and cross-scene tail ranking."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def head(width):
    return nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, 1))


def label_pools(queries):
    """Columns are all-check pass and genuinely supported."""
    if any(q.ndim != 2 or q.shape[1] != 2 or np.any(q[:, 0] & ~q[:, 1]) for q in queries):
        raise ValueError("Invalid exact joint labels")
    return tuple([np.flatnonzero(m) for m in mask] for mask in (
        [q[:, 0] for q in queries],
        [~q[:, 0] for q in queries],
        [q[:, 1] & ~q[:, 0] for q in queries],
    ))


def sample(pools, rng, n, rivals):
    positive, negative, hard = pools
    neg_ids, hard_ids = (np.flatnonzero([len(v) for v in p]) for p in (negative, hard))
    pos_ids = [i for i, rows in enumerate(positive) if len(rows) and np.any(neg_ids != i)]
    if not pos_ids:
        raise ValueError("No cross-scene qualification contrasts")
    for i in rng.choice(pos_ids, n):
        negatives = []
        for k in range(rivals):
            use_hard = k % 2 == 0 and np.any(hard_ids != i)
            roster, rows = (hard_ids, hard) if use_hard else (neg_ids, negative)
            j = int(rng.choice(roster[roster != i]))
            negatives.append((j, int(rng.choice(rows[j]))))
        yield (int(i), int(rng.choice(positive[i]))), negatives


def loss(positive, negative, rank_weight):
    point = (F.softplus(-positive).mean()+F.softplus(negative).mean())/2
    rank = F.softplus(1.+negative.max(dim=1).values-positive).mean()
    return point+rank_weight*rank, point, rank
