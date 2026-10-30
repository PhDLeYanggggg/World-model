"""Identified cost moments plus a directional original-budget score objective."""
from contextlib import contextmanager

import torch

from src.world_model import m3w_active_bank_cost as active

OBJECTIVES = ('original', 'symmetric_score', 'upper_score')
BUDGET = .02
ORIGINAL_LOSS = active.completion.base.continuous_loss
ORIGINAL_OBJECTIVE = active.completion.OBJECTIVE


def score_penalty(prediction, target, real, rms, *, underprediction_weight):
    if (prediction.shape != target.shape or target.ndim != 3 or target.shape[-1] != 5
            or real.shape != target.shape[:2] or real.dtype != torch.bool
            or rms.shape != (5,) or not torch.isfinite(rms).all() or (rms <= 0).any()
            or not torch.isfinite(prediction).all() or not torch.isfinite(target).all()
            or (target < 0).any() or underprediction_weight not in (1., 4.)):
        raise ValueError('Finite five moments, active supervision mask and fixed score weights required')
    # The reference prediction is part of the deployed risk inequality, not a fixed denominator.
    predicted = prediction[..., [1, 4]] - BUDGET * prediction[..., [2, 3]]
    observed = target.detach()[..., [1, 4]] - BUDGET * target.detach()[..., [2, 3]]
    scale = torch.sqrt(rms[[1, 4]].square() + (BUDGET * rms[[2, 3]]).square())
    error = (predicted - observed) / scale
    weight = torch.where(error < 0, underprediction_weight, 1.)
    per_action = (weight * error.square()).mean(-1)
    return ((per_action * real).sum(1) / real.sum(1).clamp_min(1)).mean()


@contextmanager
def installed(objective_arm):
    if objective_arm not in OBJECTIVES:
        raise ValueError('One of three registered score objectives required')
    base, parent = active.completion.base, active.completion
    previous_loss, previous_name = base.continuous_loss, parent.OBJECTIVE

    def loss(prediction, target, real, rms, *, underharm_weight, ranking_weight):
        values = ORIGINAL_LOSS(prediction, target, real, rms,
            underharm_weight=underharm_weight, ranking_weight=ranking_weight)
        if objective_arm == 'original':
            return values
        extra = score_penalty(prediction, target, real, rms,
            underprediction_weight=4. if objective_arm == 'upper_score' else 1.)
        return dict(values, total=values['total'] + extra)

    if objective_arm != 'original':
        base.continuous_loss = loss
        parent.OBJECTIVE = ORIGINAL_OBJECTIVE + '+unit_weight_' + objective_arm + '_original2percent_budget'
    try:
        yield
    finally:
        base.continuous_loss, parent.OBJECTIVE = previous_loss, previous_name


def fit(data, *, objective_arm, **kwargs):
    with installed(objective_arm):
        return active.fit(data, **kwargs)
