"""Remove shared observed appearance without fitting a target-domain transform."""
from src.world_model.m3w_source_pretrained_temporal import TemporalSourceDynamics
import torch

ARMS = ('centered', 'centered_unit')


def center_observed_tokens(appearance, coverage, mode):
    n = len(appearance)
    if (mode not in ARMS or appearance.shape != (n, 8, 512) or coverage.shape != (n, 8)
            or not torch.isfinite(appearance).all() or not torch.isfinite(coverage).all()
            or torch.any((coverage < 0) | (coverage > 1))):
        raise ValueError('Finite past appearance and coverage required')
    valid = coverage > 0
    count = valid.sum(1).clamp_min(1)
    mean = (appearance*valid[..., None]).sum(1)/count[:, None]
    centered = torch.where(valid[..., None], appearance-mean[:, None], torch.zeros_like(appearance))
    if mode == 'centered_unit':
        rms = torch.sqrt(centered.square().sum((1, 2))/count).clamp_min(.001)
        centered = centered/rms[:, None, None]
    return centered


class CenteredTemporalDynamics(TemporalSourceDynamics):
    def __init__(self, mode):
        if mode not in ARMS: raise ValueError('Fixed centering arm required')
        super().__init__('sequence')
        self.centering = mode

    def forward(self, geometry, appearance, coverage, engine_tag='mask_only'):
        return super().forward(geometry, center_observed_tokens(appearance, coverage, self.centering),
                               coverage, engine_tag)
