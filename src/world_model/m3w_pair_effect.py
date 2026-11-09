"""Learn the non-additive effect of two edits on observed relative motion."""

import numpy as np
import torch
from torch import nn


def observed_effect(ref, delta, label, valid):
    """Compute E10 + E01 - E00 - E11, only where both futures are observed."""
    a, b = delta[:, 0], delta[:, 1]
    observed = valid.all(axis=1)
    residual = np.where(observed[..., None], ref[:, 0]-ref[:, 1]-label[:, 0]+label[:, 1], 0.)
    errors = [np.linalg.norm(x, axis=-1) for x in (residual+a, residual-b, residual, residual+a-b)]
    effect = errors[0]+errors[1]-errors[2]-errors[3]
    count = observed.sum(1)
    return (np.where(observed, effect, 0.).sum(1)/np.maximum(count, 1), count)


class PairEffect(nn.Module):
    def __init__(self, slate, weight=0.):
        super().__init__()
        self.slate = slate.requires_grad_(False)
        self.effect = nn.Sequential(nn.Linear(52, 32), nn.SiLU(), nn.Linear(32, 1))
        self.weight = weight

    def encode(self, x, geometry):
        a, b = self.slate.action(x).unbind(-2)
        return torch.cat((a+b, (a-b).abs(), geometry), dim=-1)

    def forward(self, x, geometry, on):
        score = (self.slate.unary(self.slate.action(x)).squeeze(-1)*on).sum(-1)
        delta = self.effect(self.encode(x, geometry)).squeeze(-1)
        return score, score + self.weight * delta * on.all(-1)
