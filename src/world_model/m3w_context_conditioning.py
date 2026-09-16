"""Past-only predictor conditioning, separate from the fixed evaluation scale."""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

INPUT_KEYS = {'history', 'history_mask', 'neighbors', 'neighbor_mask',
              'baseline', 'prediction_time', 'request_mask'}


def condition_inputs(inputs):
    if set(inputs) != INPUT_KEYS:
        raise ValueError('Only the declared causal input schema is accepted')
    h, n = inputs['history'], inputs['neighbors']
    hm, nm = inputs['history_mask'], inputs['neighbor_mask']
    if (h.ndim != 3 or h.shape[-1] != 3 or n.ndim != 4 or n.shape[0] != h.shape[0]
            or n.shape[2:] != h.shape[1:] or hm.shape != h.shape[:2] or nm.shape != n.shape[:3]
            or hm.dtype != torch.bool or nm.dtype != torch.bool or not hm.all()):
        raise ValueError('Aligned full ego history and explicit past masks required')
    if not torch.isfinite(h[hm]).all() or not torch.isfinite(n[nm]).all():
        raise ValueError('Finite observed context required')
    if (h[..., 2][hm] > 0).any() or (n[..., 2][nm] > 0).any():
        raise ValueError('Context must be past-only before eligibility filtering')
    eligible = nm.all(-1) & torch.isclose(n[..., 2], h[:, None, :, 2], atol=1e-6, rtol=1e-6).all(-1)
    observed_neighbors = torch.where(eligible[..., None, None], n[..., :2], 0.)
    ego_radius = h[..., :2].norm(dim=-1).amax(dim=1)
    neighbor_radius = observed_neighbors.norm(dim=-1).flatten(1).amax(dim=1)
    scale = torch.maximum(ego_radius, neighbor_radius).clamp_min(1.).detach()
    if not torch.isfinite(scale).all():
        raise ValueError('Nonfinite observed context scale')
    conditioned = dict(inputs)
    conditioned['history'] = torch.cat((h[..., :2] / scale[:, None, None], h[..., 2:]), -1)
    conditioned['neighbors'] = torch.cat((n[..., :2] / scale[:, None, None, None], n[..., 2:]), -1)
    conditioned['baseline'] = inputs['baseline'] / scale[:, None, None]
    return conditioned, scale


class ObservedContextConditioner(nn.Module):
    """Restore output units before loss/risk/evaluation; no labels enter scaling."""
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.source_identity = {
            'base_predictor': getattr(predictor, 'source_identity', {}),
            'input_conditioning': 'observed_joint_max_norm',
            'conditioner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'output_restored_before_loss_and_evaluation': True,
        }

    def forward(self, inputs):
        conditioned, scale = condition_inputs(inputs)
        return self.predictor(conditioned) * scale[:, None, None]
