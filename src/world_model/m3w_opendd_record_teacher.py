"""Same-count slate labels with and without recording-risk supervision."""

from dataclasses import replace

import numpy as np
from torch import nn


def at_count(p, count):
    """Restrict a packet to the selected cardinality for the exact solver."""
    rows = (p.rows[0],)+tuple(p.rows[j] for j in np.flatnonzero((p.size == count) & (p.size > 0)))
    if count and len(rows) == 1:
        raise ValueError("The fixed count has no legal slate")
    return replace(p, release=count, rows=rows, fixed=np.arange(count+1))


def labels(p, count, chosen, lower):
    ids = np.flatnonzero(p.size == count)
    options = [p.rows[i] for i in ids]
    target = [i for i, rows in enumerate(options) if np.isin(rows, chosen).all()]
    if len(target) != 1:
        raise ValueError("Exact witness must choose one slate at this count")
    if len(ids) <= 1:
        return None
    # Both teachers see the same packet; only the nominated option differs.
    greedy = int(np.argmax([lower[r].sum() for r in options]))
    x = np.zeros((5, p.features.shape[1]), dtype=np.float32)
    x[:len(ids)] = p.features[ids]
    return x, [greedy, target[0], len(ids)]


def network(width):
    return nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))


def loss(net, x, z, teacher):
    score = net(x).squeeze(-1)
    legal = score.new_tensor(range(x.shape[1]))[None] < z[:, 2:3]
    return nn.functional.cross_entropy(score.masked_fill(~legal, -1e9), z[:, teacher])
