"""Estimate easy reference support and preserve it when moving fixed edit slots."""

import numpy as np
import torch
from torch import nn

from src.evaluation.m3w_selection_forecast_readout import errors
from src.world_model.m3w_easy_reference_rank import motion


def prefix_features(h, q):
    return motion(h, [np.flatnonzero(q == k) for k in np.unique(q)]).astype(np.float32)


def targets(cv, y, valid, cut):
    ade, _ = errors(cv, y, valid)
    eligible = valid.any(1) & (ade > 0.) & (ade <= cut)
    return eligible, np.where(eligible, ade, 0.)


class EasySupport(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 32), nn.SiLU(), nn.Linear(32, 16), nn.SiLU(), nn.Linear(16, 2))

    def forward(self, x):
        v = self.net(x)
        return v[:, 0], nn.functional.softplus(v[:, 1])

    def loss(self, x, eligible, mass):
        logit, amount = self(x)
        classification = nn.functional.binary_cross_entropy_with_logits(logit, eligible)
        regression = nn.functional.smooth_l1_loss(amount, torch.log1p(mass), reduction='none')
        return classification + (regression*eligible).sum()/eligible.sum().clamp_min(1.)

    @torch.inference_mode()
    def estimate(self, x):
        l, a = self(torch.from_numpy(x))
        p = torch.sigmoid(l)
        return p.numpy(), (p*torch.expm1(a.clamp_max(10.))).numpy()


def guard(cv, fixed, pooled, estimate, blocks):
    if len(estimate) != len(cv) or not np.isfinite(estimate).all() or np.any(estimate < 0.):
        raise ValueError('Require finite nonnegative support per agent')
    result = pooled.copy()
    take = {k: (np.linalg.norm(v-cv, axis=-1).max(1) > 0.) for k, v in (('fixed', fixed), ('pooled', pooled))}
    rejected, differs = 0, 0
    for ix in blocks:
        if np.array_equal(fixed[ix], pooled[ix]):
            continue
        differs += 1
        first = float(estimate[ix][take['fixed'][ix]].sum())
        next_ = float(estimate[ix][take['pooled'][ix]].sum())
        if next_ + 1e-12 < first:
            result[ix] = fixed[ix]
            rejected += 1
    return result, dict(changed_blocks=differs, rejected_blocks=rejected)
