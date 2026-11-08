"""Choose matched-radius directions using two conservative reference contrasts."""

import numpy as np
from torch import nn
from torch.nn import functional as F


def network(width):
    return nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 24),
                         nn.SiLU(), nn.Linear(24, 5))


def loss(net, x, y):
    return F.smooth_l1_loss(net(x), y, beta=.05)


def choose(z, legal, mode):
    if z.shape != (len(legal), 5) or not legal[0] or not np.isfinite(z).all():
        raise ValueError("Invalid direction menu")
    z = z-z[0]
    score = z[:, 0] if mode == "cv" else z[:, :2].min(1)
    take = legal & ((z[:, 2:] <= 0.).all(1) if mode == "guard" else True)
    return int(np.where(take, score, -np.inf).argmax())


def joint(z, w, legal, other):
    """Select two co-observed shapes under an estimated non-increase in risk."""
    total = (z-z[0])[:, None]+(w-w[0])[None]
    take = legal[:, None] & other[None] & (total[..., 2:] <= 0.).all(2)
    score = np.where(take, total[..., :2].min(2), -np.inf)
    return tuple(map(int, np.unravel_index(score.argmax(), score.shape)))
