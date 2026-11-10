"""Compare exact nominee classification with operating-budget ranking."""

import numpy as np
import torch
from torch import nn

from src.world_model import m3w_vehicle_action_value as action
from src.world_model import m3w_vehicle_opportunity as opportunity

WIDTH = opportunity.WIDTH+action.WIDTH+1


def features(angles, actor, angle, margin, n):
    if not 0 <= actor < 2 or not 1 <= angle <= angles.shape[1]:
        raise ValueError('Expect an actual nominated rotation')
    return np.r_[opportunity.summarize(angles, n), angles[actor, angle-1], margin].astype(np.float32)


def network():
    return nn.Sequential(nn.Linear(WIDTH, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))


def budget_pair_loss(logit, y, site):
    terms = []
    for g in site.unique():
        use = site == g
        a, b = logit[use & (y == 1.)], logit[use & (y == 0.)]
        if a.numel() and b.numel():
            n = max(1, (15*b.numel()+99)//100)
            high = b.topk(n, sorted=False).values
            terms.append(nn.functional.softplus(high[:, None]-a[None]).mean())
    return torch.stack(terms).mean() if terms else logit.sum()*0.


def loss(net, batch, budget=False):
    x, y, site = batch
    pred = net(x)[:, 0]
    bce = nn.functional.binary_cross_entropy_with_logits(pred, y)
    return bce+.5*budget_pair_loss(pred, y, site) if budget else bce
