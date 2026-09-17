"""Observed-motion residual parameterizations, not calibrated safety guarantees."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn


INPUT_KEYS = {'history', 'history_mask', 'neighbors', 'neighbor_mask',
              'baseline', 'prediction_time', 'request_mask'}
MODES = {'baseline_skip', 'motion_bounded'}


def radial_squash(delta):
    """delta / sqrt(1 + ||delta||^2), with finite large-vector intermediates."""
    if not torch.isfinite(delta).all():
        raise ValueError('Nonfinite predicted residual')
    scale = delta.abs().amax(-1, keepdim=True).clamp_min(1.).detach()
    scaled = delta / scale
    return scaled / torch.sqrt(scale.reciprocal().square() + scaled.square().sum(-1, keepdim=True))


def past_motion_budget(inputs):
    if set(inputs) != INPUT_KEYS:
        raise ValueError('Only the declared causal input schema is accepted')
    history, mask = inputs['history'], inputs['history_mask']
    request, times = inputs['request_mask'], inputs['prediction_time']
    if (history.ndim != 3 or history.shape[-1] != 3 or history.shape[1] < 2
            or mask.shape != history.shape[:2] or mask.dtype != torch.bool or not mask.all()
            or request.dtype != torch.bool or request.shape != times.shape or not request.any(1).all()):
        raise ValueError('Full observed ego history and explicit requested times required')
    if (not torch.isfinite(history).all() or (history[..., 2] > 0).any()
            or (history[:, 1:, 2] <= history[:, :-1, 2]).any()):
        raise ValueError('Finite ordered past-only history required')
    baseline = torch.where(request[..., None], inputs['baseline'], 0.)
    times = torch.where(request, times, 0.)
    if not torch.isfinite(baseline).all() or not torch.isfinite(times).all() or (times[request] <= 0).any():
        raise ValueError('Finite causal baseline and positive requested times required')
    path = (history[:, 1:, :2] - history[:, :-1, :2]).norm(dim=-1).sum(-1)
    extent = torch.maximum(path, baseline.norm(dim=-1).amax(-1))
    budget = extent[:, None] * times / times.amax(-1, keepdim=True)
    if not torch.isfinite(budget).all():
        raise ValueError('Nonfinite observed motion budget')
    return budget.detach()


class BaselineRelativeForecaster(nn.Module):
    """Same backbone, zero initial correction, optionally ego-motion bounded.

    An exactly stationary observed history stays on its stationary CV floor in
    bounded mode, including when the future starts moving. Such misses remain
    in loss/evaluation. No target-derived easy label is consulted at inference.
    """
    def __init__(self, predictor, *, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError('Unknown output parameterization')
        self.predictor, self.mode = predictor, mode
        core = predictor
        while hasattr(core, 'predictor'):
            core = core.predictor
        if not hasattr(core, 'output') or not isinstance(core.output[-1], nn.Linear):
            raise ValueError('An explicit Transformer displacement output is required')
        nn.init.zeros_(core.output[-1].weight)
        nn.init.zeros_(core.output[-1].bias)
        self.source_identity = {
            'base_predictor': getattr(predictor, 'source_identity', {}),
            'parameterization_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'output_parameterization': mode,
            'initial_forecast': 'exact_declared_causal_baseline',
            'bounded_rule': 'ego_path_or_baseline_extent_times_requested_fraction_radial_squash'
                if mode == 'motion_bounded' else None,
            'target_or_evaluation_scale_changed': False,
            'safety_guarantee': False,
        }

    def forward(self, inputs):
        if set(inputs) != INPUT_KEYS:
            raise ValueError('Only the declared causal input schema is accepted')
        delta = self.predictor(inputs)
        if self.mode == 'motion_bounded':
            delta = past_motion_budget(inputs)[..., None] * radial_squash(delta)
        request = inputs['request_mask']
        baseline = torch.where(request[..., None], inputs['baseline'], 0.)
        return torch.where(request[..., None], baseline + delta, 0.)
