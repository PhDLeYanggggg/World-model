"""Separate observed-view alignment from assumed future-view restoration."""
import torch

from src.world_model.m3w_camera_frame import CAMERA_KEYS, CameraFrameForecaster
from src.world_model.m3w_context_conditioning import INPUT_KEYS

ARMS = ('identity', 'past_only', 'future_only', 'both')


def operator_inputs(inputs, arm):
    if arm not in ARMS or set(inputs) != INPUT_KEYS | CAMERA_KEYS:
        raise ValueError('Declared operator arm and causal inputs only')
    n = len(inputs['history'])
    for key, steps in (('observed_camera_maps', 8), ('predicted_camera_maps', 12)):
        if inputs[key].shape != (n, steps, 2, 3) or not torch.isfinite(inputs[key]).all():
            raise ValueError('Complete finite past-derived operator grid required')
    safe = dict(inputs)
    for key, enabled in (('observed_camera_maps', arm in ('past_only', 'both')),
            ('predicted_camera_maps', arm in ('future_only', 'both'))):
        if not enabled:
            value = inputs[key]
            safe[key] = torch.eye(3, dtype=value.dtype, device=value.device)[:2].expand_as(value)
    return safe


class CameraFactorForecaster(CameraFrameForecaster):
    def __init__(self, architecture, baseline_index, operator_arm):
        if operator_arm not in ARMS:
            raise ValueError('Declared factorial operator arm required')
        super().__init__(architecture, baseline_index,
            'identity' if operator_arm == 'identity' else 'camera_frame')
        self.operator_arm = operator_arm

    def forward(self, inputs):
        return super().forward(operator_inputs(inputs, self.operator_arm))
