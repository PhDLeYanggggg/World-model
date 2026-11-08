"""Keep the support edit intact while ranking matched gain alternatives."""

import numpy as np
from torch import nn
from torch.nn import functional as F


def same_support(anchor, candidate):
    a = [z for z in anchor if z[0] == 1]
    b = [z for z in candidate if z[0] == 1]
    return (len(a) == len(b) and
            all(i == j and np.array_equal(x, y) for (_, i, x), (_, j, y) in zip(a, b, strict=True)))


def allowed(anchor, candidate, support_seen, protected=True):
    if not same_support(anchor, candidate):
        return False
    old = next(j for r, j, _ in anchor if r == 0)
    new = next(j for r, j, _ in candidate if r == 0)
    return not protected or support_seen or old == new


def choose(z, legal, guard=False):
    if z.shape != (len(legal), 7) or not legal[0] or not np.isfinite(z).all():
        raise ValueError("Invalid gain menu")
    lower = z[:, 0]-z[0, 0]-np.maximum(z[:, 1], 0.)
    lower[0] = 0.
    if guard:
        legal = (legal & (z[:, 2:5] <= z[0, 2:5]).all(1)
                 & (z[:, 5] >= z[0, 5]) & (z[:, 6] <= z[0, 6]))
    return int(np.where(legal, lower, -np.inf).argmax())


def network(width):
    return nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 32),
                         nn.SiLU(), nn.Linear(32, 7))


def loss(net, x, y):
    z = net(x)
    point = F.smooth_l1_loss(z[:, :5], y[:, :5], beta=.05)
    joint = F.binary_cross_entropy_with_logits(z[:, 5], y[:, 5])
    ref_loss = F.binary_cross_entropy_with_logits(z[:, 6], y[:, 6], pos_weight=z.new_tensor(4.))
    return point+.1*joint+.2*ref_loss
