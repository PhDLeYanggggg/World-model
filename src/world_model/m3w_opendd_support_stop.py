"""Finite causal stopping for a released support slot."""

import numpy as np
from torch import nn

DEADLINE = 2


def stop(packets, policy):
    """Packets contain only a current recipient and current two-head scores."""
    release = None
    for t, (row, now, ahead) in enumerate(packets):
        if release is None and row is not None:
            release = t
        if release is None:
            continue
        age = t-release
        if age > DEADLINE:
            break
        if row is None:
            continue
        if not np.isfinite([now, ahead]).all():
            raise ValueError("Nonfinite support stopping scores")
        if (policy == "immediate" or (policy == "wait_one" and age >= 1)
                or (policy == "myopic" and (now >= 0. or age == DEADLINE))
                or (policy == "continue" and (ahead >= 0. or age == DEADLINE))):
            return t, int(row)
    return None


def targets(rewards):
    """Observed-label contrasts on a donor's first released two-query window."""
    first = next((i for i, r in enumerate(rewards) if r is not None), None)
    if first is None:
        return
    use = rewards[first:first+DEADLINE+1]
    for i, value in enumerate(use):
        if value is not None:
            best = max([0.]+[v for v in use[i+1:] if v is not None])
            yield first+i, value, value-best


def network(width):
    return nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, 2))


def loss(net, x, target):
    return nn.functional.smooth_l1_loss(net(x), target, beta=.05)
