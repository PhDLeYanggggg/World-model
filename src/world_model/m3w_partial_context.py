"""Versioned observed-token conditioning; old models and policies are unchanged."""
import platform
if platform.system() == 'Darwin' and platform.machine() != 'arm64':
    raise RuntimeError('Native arm64 required before Torch import')
import torch
from torch import nn
from src.world_model.m3w_context_conditioning import INPUT_KEYS
from src.world_model.m3w_european_source_forecast import BASELINES, baseline_torch
from src.world_model.m3w_supervised_intervention import PastContextForecaster
from src.world_model.m3w_baseline_relative_forecaster import BaselineRelativeForecaster


def condition_partial_inputs(inputs):
    if set(inputs) != INPUT_KEYS:
        raise ValueError('Only declared causal input schema allowed')
    h, n = inputs['history'], inputs['neighbors']
    hm, nm = inputs['history_mask'], inputs['neighbor_mask']
    if (h.ndim != 3 or h.shape[-1] != 3 or n.ndim != 4 or n.shape[0] != len(h)
            or n.shape[2:] != h.shape[1:] or hm.shape != h.shape[:2]
            or nm.shape != n.shape[:3] or hm.dtype != torch.bool or nm.dtype != torch.bool
            or not hm.all() or not torch.isfinite(h).all() or not torch.isfinite(n[nm]).all()):
        raise ValueError('Complete finite ego and explicitly masked finite neighbor observations required')
    if ((h[..., 2] > 0).any() or (h[:, 1:, 2] <= h[:, :-1, 2]).any()
            or (h[:, -1, 2] != 0).any()
            or not torch.isclose(n[..., 2], h[:, None, :, 2], rtol=1e-6, atol=1e-6)[nm].all()):
        raise ValueError('Synchronous past-only observed token timestamps required')
    safe = torch.where(nm[..., None], n, 0.)
    scale = torch.maximum(h[..., :2].norm(dim=-1).amax(1),
        safe[..., :2].norm(dim=-1).flatten(1).amax(1)).clamp_min(1.).detach()
    result = dict(inputs)
    result['history'] = torch.cat((h[..., :2]/scale[:, None, None], h[..., 2:]), -1)
    result['neighbors'] = torch.cat((safe[..., :2]/scale[:, None, None, None], safe[..., 2:]), -1)
    result['baseline'] = inputs['baseline']/scale[:, None, None]
    return result, scale


class PartialObservedConditioner(nn.Module):
    def __init__(self, predictor):
        super().__init__(); self.predictor = predictor

    def forward(self, inputs):
        conditioned, scale = condition_partial_inputs(inputs)
        return self.predictor(conditioned)*scale[:, None, None]


class PartialContextSourceForecaster(nn.Module):
    """Same parameter budget and bounded output; only causal context rule changes."""
    def __init__(self, baseline_index, *, width=64, heads=4, layers=2):
        super().__init__()
        if not 0 <= baseline_index < len(BASELINES):
            raise ValueError('Registered causal baseline required')
        self.baseline_index = baseline_index
        core = PastContextForecaster(width=width, heads=heads, layers=layers, neighbor_policy='observed_tokens')
        self.predictor = BaselineRelativeForecaster(PartialObservedConditioner(core), mode='motion_bounded')

    def forward(self, inputs):
        if set(inputs) != INPUT_KEYS:
            raise ValueError('Only declared causal input schema allowed')
        safe = dict(inputs)
        safe['baseline'] = baseline_torch(inputs['history'], self.baseline_index)
        return self.predictor(safe)
