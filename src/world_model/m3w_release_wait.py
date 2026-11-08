"""Release-anchored wait decisions with an untouched-policy credit reserve."""

import numpy as np
from torch import nn

AGE = 2


class WaitValue(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, 4))

    def forward(self, x):
        return self.net(x)


def loss(net, x, target):
    return nn.functional.smooth_l1_loss(net(x), target, beta=.02)


def schedule(query, rows, slots, score, release, original, eligible, cap, defer):
    """An immediate anchor is never displaced by an earlier deferred edit."""
    order = np.argsort(query, kind="stable")
    off = np.r_[0, np.bincount(query, minlength=len(cap)).cumsum()]
    released = np.r_[0, np.bincount(release, minlength=len(cap)).cumsum()]
    paid = np.zeros(len(release), dtype=bool)
    ref = balance = 0
    selected = []
    for t in range(len(cap)):
        arrived = np.arange(released[t], released[t+1])
        ref = min(10, ref+int(eligible[t]))-5*len(arrived)
        keep = original[arrived[~defer[arrived]]]
        balance = min(10, balance+int(eligible[t]))-5*len(keep)
        if not 0 <= ref <= balance:
            raise ValueError("An untouched anchor cannot be funded")
        selected.extend(keep)
        paid[slots[keep]] = True
        used = set(rows[keep])
        limit = min((balance-ref)//5, cap[t]-len(keep))
        if not limit:
            continue
        ix = order[off[t]:off[t+1]]
        rs = slots[ix]
        on = ix[defer[rs] & ~paid[rs] & (release[rs] < t) & (t <= release[rs]+AGE) & (score[ix] > 0.)]
        for e in on[np.argsort(-score[on], kind="stable")]:
            if rows[e] in used or paid[slots[e]]:
                continue
            used.add(rows[e])
            paid[slots[e]] = True
            selected.append(e)
            balance -= 5
            limit -= 1
            if not limit:
                break
    return np.asarray(selected, dtype=int)
