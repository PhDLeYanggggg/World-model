"""Past-conditioned timing of an edit at exactly fixed peak departure."""

import torch
from torch import nn


def fixed_radius(displacement, envelope, fraction=.25):
    """Reallocate movement across forecast steps, keeping each row's peak."""
    raw = displacement * envelope.unsqueeze(-1)
    peak = displacement.norm(dim=-1).amax(dim=-1, keepdim=True)
    scaled_peak = raw.norm(dim=-1).amax(dim=-1, keepdim=True)
    safe = torch.where(scaled_peak > 0, scaled_peak, torch.ones_like(scaled_peak))
    return raw * (fraction * peak / safe).unsqueeze(-1)


class TemporalEnvelope(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(width, 32), nn.SiLU(), nn.Linear(32, 4))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, displacement):
        knots = .25 + 1.5 * self.head(x).sigmoid()
        shape = torch.nn.functional.interpolate(knots.unsqueeze(1), size=displacement.shape[1],
                                               mode="linear", align_corners=True).squeeze(1)
        return fixed_radius(displacement, shape), shape


def objective(edit, shape, truth, ols, valid, easy):
    """Observed dual-reference error and non-canceling harm; no mask in inputs."""
    def ade(path):
        return ((path-truth).norm(dim=-1) * valid).sum(1) / valid.sum(1).clamp(min=1)
    current = ade(edit)
    cv = ade(torch.zeros_like(edit))
    other = ade(ols)
    known = valid.any(1)
    loss = current + (current-other).clamp(min=0) + (current-cv).clamp(min=0) * (1 + 3 * easy)
    smooth = shape.diff(dim=1).square().mean()
    return (loss * known).sum() / known.sum().clamp(min=1) + .025 * smooth
