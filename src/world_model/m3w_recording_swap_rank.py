"""Compare budget-feasible swaps across queries in one recording."""

import numpy as np
import torch
from torch import nn

from src.evaluation.m3w_opendd_location import paired
from src.evaluation.m3w_streaming_menu_capacity import respects_tokens
from src.world_model.m3w_direct_count_value import accumulate


def components(cv, bank, y, valid, row, arm):
    out = []
    for p in bank.swapaxes(0, 1):
        known, lower = paired(cv, p, y, valid)
        out.append(np.stack((known.mean(1), (known-lower).mean(1)), axis=-1))
    return accumulate(np.stack(out, axis=1), row, arm)


def swaps(plans, legal, recordings, rng, limit=32):
    """Hold per-recording edit count and chronological credit exactly fixed."""
    edges = set()
    for r in np.unique(recordings):
        ix = np.flatnonzero(recordings == r)
        lm = legal[ix]
        for plan in plans:
            v = plan[ix]
            source = np.flatnonzero(v > 0)
            dest = np.flatnonzero((v == 0) & lm[:, 1])
            if not len(source) or not len(dest):
                continue
            local = set()
            for _ in range(limit*5):
                a, b = int(rng.choice(source)), int(rng.choice(dest))
                c = int(v[a])
                if not lm[b, c] or (a, b, c) in local:
                    continue
                vv = v.copy()
                vv[a], vv[b] = 0, c
                if respects_tokens(vv, lm[:, 1]):
                    local.add((a, b, c))
                if len(local) == limit:
                    break
            edges.update((min(ix[a], ix[b]), max(ix[a], ix[b]), c) for a, b, c in local)
    return np.asarray(sorted(edges), dtype=np.int64).reshape(-1, 3)


def target(parts, edges, scale):
    a, b, c = edges.T
    left, right = parts[a, c], parts[b, c]
    diff = right[:, 0]-left[:, 0]
    debit = right[:, 1]+left[:, 1]
    normalizer = (scale[a]+scale[b])*.5
    return (np.sign(diff)*np.maximum(np.abs(diff)-debit, 0.)/normalizer).astype(np.float32)


class RankValue(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 48), nn.SiLU(), nn.Linear(48, 4))

    def forward(self, x):
        v = self.net(x).reshape(len(x), 2, 2)
        return torch.cat((v.new_zeros((len(x), 1, 2)), v), dim=1)


def objective(net, x, y, legal, left, right, count, w, margin, rank):
    ok = legal.clone()
    ok[:, 0] = False
    fit = nn.functional.smooth_l1_loss(net(x)[ok], y[ok], beta=.06)
    if not rank or not len(left):
        return fit
    i = torch.arange(len(count))
    both = net(torch.cat((left, right)))
    l, r = both.chunk(2)
    d = r[i, count].min(1).values*w[:, 1]-l[i, count].min(1).values*w[:, 0]
    known = margin != 0.
    fit_pair = nn.functional.smooth_l1_loss(d, margin, beta=.04, reduction="none")
    sign = nn.functional.binary_cross_entropy_with_logits(d/.04, (margin > 0.).float(), reduction="none")
    rank_loss = fit_pair+.035*sign*(margin.abs()/.04).clamp(max=4.)
    return fit+((rank_loss*known).sum()/known.sum().clamp_min(1))
