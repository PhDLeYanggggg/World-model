"""Online selected-action false-admission loss; frozen causal inference unchanged."""
from contextlib import contextmanager

import torch
from src.world_model import m3w_budget_score_cost as parent

OBJECTIVES = ('original', 'upper_score', 'admitted_score')


def admission_penalty(prediction, target, real, rms):
    if (prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 5
            or real.shape != prediction.shape[:2] or real.dtype != torch.bool or rms.shape != (5,)
            or not torch.isfinite(prediction).all() or not torch.isfinite(target).all()
            or (prediction < 0).any() or (target < 0).any()
            or not torch.isfinite(rms).all() or (rms <= 0).any()):
        raise ValueError('Finite known-only supervised moments and frozen RMS required')
    # Recompute the causal winning action every update, with no label-based routing.
    with torch.no_grad():
        p = prediction.detach().to(torch.float64)
        gain = p[..., 0] - p[..., 1]
        admissible = (real & (gain > 0) & (p[..., 2] > 0) & (p[..., 3] > 0)
            & (p[..., 1] <= .02 * p[..., 2]) & (p[..., 4] <= .02 * p[..., 3]))
        choice = gain.masked_fill(~admissible, -torch.inf).argmax(1)
        selected = admissible.any(1)
    at = torch.arange(len(prediction), device=prediction.device)
    pred = prediction[at, choice]
    truth = target.detach()[at, choice]
    q_pred = pred[:, [1, 4]] - .02 * pred[:, [2, 3]]
    q_true = truth[:, [1, 4]] - .02 * truth[:, [2, 3]]
    scale = torch.sqrt(rms[[1, 4]].square() + (.02 * rms[[2, 3]]).square())
    # Labels supervise the error of that choice; they never choose the action.
    false = selected[:, None] & (q_true > 0)
    loss = (torch.relu((q_true - q_pred) / scale).square() * false).mean()
    return loss, dict(selected_supervised_rows=int(selected.sum()),
        false_admitted_channels=int(false.sum()))


@contextmanager
def installed(objective_arm):
    if objective_arm not in OBJECTIVES:
        raise ValueError('Registered admission objective required')
    with parent.installed('original' if objective_arm == 'original' else 'upper_score'):
        base = parent.active.completion.base
        previous, name = base.continuous_loss, parent.active.completion.OBJECTIVE
        if objective_arm == 'admitted_score':
            def loss(prediction, target, real, rms, **kwargs):
                terms = previous(prediction, target, real, rms, **kwargs)
                extra, _ = admission_penalty(prediction, target, real, rms)
                return dict(terms, total=terms['total'] + extra)
            base.continuous_loss = loss
            parent.active.completion.OBJECTIVE = name + '+unit_online_chosen_false_admission'
        try:
            yield
        finally:
            base.continuous_loss = previous
            parent.active.completion.OBJECTIVE = name


def fit(data, *, objective_arm, **kwargs):
    with installed(objective_arm):
        return parent.active.fit(data, **kwargs)
