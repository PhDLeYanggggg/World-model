"""Causal reference-complete output support, not a prediction-risk guarantee."""
import math
from pathlib import Path
import hashlib
import platform

if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')

import torch

from src.world_model.m3w_baseline_relative_forecaster import (
    BaselineRelativeForecaster, INPUT_KEYS, past_motion_budget, radial_squash,
)
from src.world_model.m3w_dimensionless_correction import DimensionlessAgentTrackSourceForecaster
from src.world_model.m3w_european_source_forecast import baseline_torch


def reference_motion_budget(inputs, *, enabled=True, margin=.05):
    original = past_motion_budget(inputs)
    if not enabled:
        return original
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError('A fixed finite positive reference-interior margin is required')
    request = inputs['request_mask']
    if request.shape[1] != 12:
        raise ValueError('The registered twelve requested steps are required')
    center = torch.where(request[..., None], inputs['baseline'], 0.)
    result = original
    for index in (1, 3):
        reference = baseline_torch(inputs['history'], index)
        reference = torch.where(request[..., None], reference, 0.)
        distance = (reference-center).norm(dim=-1)
        result = torch.maximum(result, (1+margin)*distance)
    if not torch.isfinite(result).all():
        raise ValueError('Finite past-only reference budget required')
    return result.detach()


class ReferenceCoverageForecaster(BaselineRelativeForecaster):
    def __init__(self, predictor, *, enabled=True, margin=.05):
        super().__init__(predictor, mode='motion_bounded')
        if not math.isfinite(margin) or margin <= 0:
            raise ValueError('Finite positive reference margin required')
        self.enabled, self.margin = bool(enabled), margin
        self.source_identity.update(
            reference_coverage_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            coverage_enabled=self.enabled, reference_margin=self.margin,
            coverage_references=['causal_CV', 'causal_damping_097'],
            safety_guarantee=False,
        )

    def forward(self, inputs):
        if not self.enabled:
            return super().forward(inputs)
        if set(inputs) != INPUT_KEYS:
            raise ValueError('Only the frozen causal input schema is accepted')
        delta = self.predictor(inputs)
        radius = reference_motion_budget(inputs, margin=self.margin)
        baseline = torch.where(inputs['request_mask'][..., None], inputs['baseline'], 0.)
        result = baseline + radius[..., None]*radial_squash(delta)
        return torch.where(inputs['request_mask'][..., None], result, 0.)


class ReferenceCompleteAgentTrackForecaster(DimensionlessAgentTrackSourceForecaster):
    """Same initialized core and parameter keys; only the causal radius changes."""
    def __init__(self, baseline_index, *, width=64, heads=4, layers=2, enabled=True, margin=.05):
        super().__init__(baseline_index, width=width, heads=heads, layers=layers)
        self.predictor = ReferenceCoverageForecaster(self.predictor.predictor,
                                                     enabled=enabled, margin=margin)
