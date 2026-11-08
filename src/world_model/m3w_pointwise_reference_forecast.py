"""Observed positive-harm training against the stronger per-row reference."""
import torch

from src.world_model.m3w_dual_reference_forecast import objective


def _pointwise_terms(pred, target, valid, cv, ols, scale):
    y = torch.where(valid[..., None], target, 0.)

    def ade(x):
        return torch.where(valid, torch.linalg.vector_norm(x-y, dim=-1), 0.).sum(1)/valid.sum(1).clamp_min(1)

    e, c, o = (ade(x) for x in (pred, cv, ols))
    # Replaces the existing positive CV term by the maximum of both harms.
    excess = (torch.relu(e-torch.minimum(c, o))-torch.relu(e-c))/scale
    return excess, (o < e) & (e <= c)


def pointwise_excess(pred, target, valid, cv, ols, scale):
    return _pointwise_terms(pred, target, valid, cv, ols, scale)[0]


def loss(pred, target, valid, cv, ols, scale, easy_cut, *, treatment):
    base, refs, penalty = objective(pred, target, valid, cv, ols, scale, easy_cut, dual=False)
    extra, active = _pointwise_terms(pred, target, valid, cv, ols, scale)
    return base+(.5*extra.mean() if treatment else 0.), refs, penalty, extra.mean(), active.sum()
