"""Matched direct and event/severity supervision for selected easy harm."""
import platform

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before importing Torch')

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ('raw', 'ridge_risk', 'direct', 'factorized')


class HarmHead(nn.Module):
    def __init__(self, dims, event_rate, severity):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dims, 32), nn.SiLU(), nn.Linear(32, 2))
        p = torch.tensor([event_rate, severity], dtype=torch.float32).clamp(.001, .999)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.logit(p))

    def forward(self, x):
        return self.net(x)


def loss(logits, target, factorized):
    event = (target > 0.).to(target.dtype)
    expected = logits.sigmoid().prod(1)
    marginal = F.binary_cross_entropy(expected.clamp(1e-7, 1.-1e-7), target)
    if not factorized:
        return marginal
    occurrence = F.binary_cross_entropy_with_logits(logits[:, 0], event)
    magnitude = F.binary_cross_entropy_with_logits(logits[:, 1], target, reduction='none')
    conditional = (magnitude * event).sum()/event.sum().clamp_min(1.)
    return marginal + .5*(occurrence + conditional)


def admissible(utility, all_excess, easy_excess, legal):
    return legal & (utility > 0.) & (all_excess <= 0.) & (easy_excess <= 0.)


def matched(groups, count_from, score, eligible):
    """Choose within a fixed, label-free pool at the exact raw query count."""
    z = np.where(eligible, score, -np.inf)
    best = z.argmax(1); value = z[np.arange(len(z)), best]
    out = np.full(len(z), -1, dtype=np.int16)
    for g in groups:
        n = int((count_from[g] >= 0).sum())
        if n:
            ix = g[np.argsort(-value[g], kind='stable')[:n]]
            if not np.isfinite(value[ix]).all():
                raise ValueError('Matched count must be possible inside the original pool')
            out[ix] = best[ix]
    return out


def sampled_batches(donor_queries, steps, batch_size, seed):
    """Draw source, query, then candidate; repeats are intentional."""
    rng = np.random.default_rng(seed)
    donor = rng.integers(len(donor_queries), size=steps*batch_size)
    out = np.empty(len(donor), dtype=np.int64)
    for k, queries in enumerate(donor_queries):
        at = np.flatnonzero(donor == k)
        chosen = rng.integers(len(queries), size=len(at))
        out[at] = [q[rng.integers(len(q))] for q in (queries[c] for c in chosen)]
    return out.reshape(steps, batch_size)
