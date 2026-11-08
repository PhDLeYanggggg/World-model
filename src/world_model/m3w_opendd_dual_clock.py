"""Past-only two-role stopping with a separate retrospective budget match."""

from itertools import product

import numpy as np
from torch import nn
from torch.nn import functional as F

LOOKAHEAD = 2


def proposals(packet, spent=()):
    used = {r for r, _ in spent}
    occupied = {i for _, i in spent}
    menu = tuple((None, *(i for i in rows if i not in occupied)) if r not in used else (None,)
                 for r, rows in enumerate(packet))
    for p in product(*menu):
        if p[0] is not None and p[0] == p[1]:
            continue
        yield tuple((r, i) for r, i in enumerate(p) if i is not None)


def stream(packets, score, mode):
    """Score present actions against a present state, without terminal cues."""
    spent = ()
    for q, packet in enumerate(packets):
        if len(spent) == 2:
            break
        menu = tuple(x for x in proposals(packet, spent) if x)
        if not menu:
            continue
        z = np.asarray(score(q, packet, spent, menu))
        if z.shape != (len(menu), 3) or not np.isfinite(z).all():
            raise ValueError("Invalid current decision scores")
        value = z[:, 0] if mode == "myopic" else np.minimum(z[:, 1], z[:, 2])
        if value.max() > 0.:
            spent += menu[int(value.argmax())]
    return spent


def earliest(packets, roles):
    """Label-free retrospective control: reproduce only the spent role set."""
    if not roles:
        return ()
    items = {r: [(q, rank, i) for q, p in enumerate(packets) for rank, i in enumerate(p[r])]
             for r in sorted(roles)}
    if len(items) == 1:
        r, z = next(iter(items.items()))
        if z:
            return ((r, z[0][2]),)
    else:
        ranked = [(min(a, b), max(a, b), ra, rb, ia, ib) for a, ra, ia in items[0]
                  for b, rb, ib in items[1] if ia != ib]
        if ranked:
            *_, a, b = min(ranked)
            return ((0, a), (1, b))
    raise ValueError("A realized role budget has no matched control")


def describe(features, q, packet, spent, act):
    width = features[0].shape[1]
    rows = [np.zeros(width+1) for _ in range(4)]
    for offset, seq in ((0, spent), (2, act)):
        for r, i in seq:
            rows[offset+r] = np.r_[features[r][i], 1.]
    return np.concatenate((*rows, np.log1p([q, len(packet[0]), len(packet[1])]))).astype(np.float32)


def network(width):
    return nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 32),
                         nn.SiLU(), nn.Linear(32, 3))


def loss(net, x, y):
    return F.smooth_l1_loss(net(x), y, beta=.1)
