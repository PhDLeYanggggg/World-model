"""Train-only scaling and keep-relative objectives for legal allocation slates."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ('linear_absolute', 'linear_advantage', 'mlp_absolute', 'mlp_advantage')


def normalization(table):
    total, first, second = 0, None, None
    for x, _, valid in table:
        z = x[valid].double()
        if not torch.isfinite(z).all():
            raise ValueError('Nonfinite training features')
        if not len(z):
            continue
        a, b = z.sum(0), z.square().sum(0)
        first = a if first is None else first+a
        second = b if second is None else second+b
        total += len(z)
    if not total:
        raise ValueError('No valid training options')
    mean = first/total
    scale = (second/total-mean.square()).clamp_min(0).sqrt().clamp_min(.1)
    return dict(mean=mean.float().tolist(), scale=scale.float().tolist(), options=total,
                std_floor=.1, clip=8., fit_role='training_features_only')


class Selector(nn.Module):
    def __init__(self, arm, norm):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Unknown registered arm')
        self.register_buffer('mean', torch.tensor(norm['mean'], dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(norm['scale'], dtype=torch.float32))
        width = len(norm['mean'])
        self.net = (nn.Linear(width, 1) if arm.startswith('linear') else
                    nn.Sequential(nn.Linear(width, 48), nn.GELU(), nn.Linear(48, 32),
                                  nn.GELU(), nn.Linear(32, 1)))

    def forward(self, x):
        return self.net(((x-self.mean)/self.scale).clamp(-8., 8.)).squeeze(-1)


def objective(scores, y, valid, *, relative):
    if scores.shape != y.shape or y.shape != valid.shape or not valid[:, 0].all():
        raise ValueError('Misaligned keep-inclusive slate')
    target = y.to(scores.dtype)
    if relative:
        mask = valid[:, 1:]
        error = (torch.tanh(scores[:, 1:]-scores[:, :1])-(target[:, 1:]-target[:, :1])).square()
    else:
        mask = valid
        error = F.binary_cross_entropy_with_logits(scores, target, reduction='none')
    return ((error*mask).sum(1)/mask.sum(1).clamp_min(1)).mean()


def choose(scores, valid):
    if scores.shape != valid.shape or not valid[..., 0].all() or not torch.isfinite(scores[valid]).all():
        raise ValueError('Invalid legal option scores')
    return scores.masked_fill(~valid, float('-inf')).argmax(-1)


def select_arm(validation):
    """Fixed tie order; nonpositive inner gain retains the incumbent everywhere."""
    if set(validation) != set(ARMS) or not all(np.isfinite(v) for v in validation.values()):
        raise ValueError('Incomplete inner validation')
    best = max(ARMS, key=lambda k: validation[k])
    return best if validation[best] > 0. else 'keep'
