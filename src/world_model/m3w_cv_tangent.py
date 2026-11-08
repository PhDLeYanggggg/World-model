"""CV-relative small edits with a directional-derivative training ablation."""

import torch
from torch import nn


def directional_derivative(residual, step, valid):
    """One-sided derivative of observed ADE at CV, including zero-error cusps."""
    length = residual.norm(dim=-1)
    smooth = (residual*step).sum(-1)/length.clamp_min(1e-12)
    raw = torch.where(length > 1e-12, smooth, step.norm(dim=-1))
    return (raw*valid).sum(1)/valid.sum(1).clamp_min(1)


class CVTangent(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU())
        self.shape = nn.Linear(32, 8)
        self.accept = nn.Linear(32, 1)
        nn.init.zeros_(self.shape.weight)
        nn.init.zeros_(self.shape.bias)

    def forward(self, x, incumbent_unit, radius):
        h = self.encoder(x)
        knots = .30*self.shape(h).reshape(-1, 4, 2).tanh()
        change = nn.functional.interpolate(knots.transpose(1, 2), size=12,
                                           mode="linear", align_corners=True).transpose(1, 2)
        proposal = incumbent_unit+change
        norm = proposal.norm(dim=-1).amax(1).clamp_min(1e-12)
        return proposal*(radius/norm)[:, None, None], self.accept(h).squeeze(1)


def losses(offset, logit, cv, target, valid, easy, with_tangent):
    clean = torch.where(valid[..., None], target, cv)
    residual = cv-clean
    current = residual+offset
    n = valid.sum(1).clamp_min(1)
    known = valid.any(1)
    known_mass = known.sum().clamp_min(1)
    baseline = (residual.norm(dim=-1)*valid).sum(1)/n
    change = (current.norm(dim=-1)*valid).sum(1)/n-baseline
    missing = (~known)*offset.norm(dim=-1).amax(1)
    harm = change.clamp_min(0.)*(1.+3.*easy)
    utility = (change.sum()+harm.sum())/known_mass+missing.mean()
    # Acceptance learns realized improvement and a separate guard for easy harm.
    label = known & (change < 0.) & (baseline > 0.)
    guard = nn.functional.binary_cross_entropy_with_logits(logit, label.to(logit.dtype))
    if not with_tangent:
        return utility+.02*guard
    slope = directional_derivative(residual, offset, valid)
    return utility+.02*guard+.5*(slope+slope.clamp_min(0.)*easy).sum()/known_mass
