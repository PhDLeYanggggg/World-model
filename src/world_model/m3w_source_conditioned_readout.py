"""Train-scale output parameterization; same bounded forecast function class."""
import numpy as np
import torch

from src.world_model.m3w_source_pretrained_temporal import TemporalSourceDynamics
from src.world_model.m3w_source_temporal_centered import center_observed_tokens


def training_readout_gain(radius, loss_scale):
    values = np.asarray(radius, dtype=np.float64)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
            or np.any(values < 0) or not np.isfinite(loss_scale) or loss_scale <= 0):
        raise ValueError('Training radii and positive existing training-loss scale required')
    gain = float(np.median(values/loss_scale))
    if gain <= 0:
        raise ValueError('No positive median training restoration support')
    return max(1., gain)


class ConditionedTemporalDynamics(TemporalSourceDynamics):
    def __init__(self, arm, gain):
        if arm not in ('geometry', 'centered') or not np.isfinite(gain) or gain < 1:
            raise ValueError('Fixed arm and train-derived readout gain required')
        super().__init__('geometry' if arm == 'geometry' else 'sequence')
        self.conditioned_arm = arm
        self.register_buffer('readout_gain', torch.tensor(float(gain), dtype=torch.float64))
        # Division precedes the existing radial bound. No forecast is clipped further.
        self.head[-1].register_forward_hook(self._condition_readout)

    def _condition_readout(self, module, inputs, output):
        return output/self.readout_gain.to(dtype=output.dtype, device=output.device)

    def forward(self, geometry, appearance, coverage, engine_tag='mask_only'):
        if self.conditioned_arm == 'centered':
            appearance = center_observed_tokens(appearance, coverage, 'centered')
        return super().forward(geometry, appearance, coverage, engine_tag)
