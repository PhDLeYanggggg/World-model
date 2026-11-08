"""Causal, origin-anchored velocity from reliability-weighted observations."""

import torch
from torch import nn


def line_fit(history, weights):
    """Fit a free-intercept line, then use only its velocity at the current anchor."""
    t = torch.arange(-7, 1, dtype=history.dtype, device=history.device)
    w = weights/weights.sum(1, keepdim=True)
    tc = t-(w*t).sum(1, keepdim=True)
    v = ((w*tc)[..., None]*history).sum(1)/(w*tc.square()).sum(1, keepdim=True)
    intercept = (w[..., None]*(history-t[None, :, None]*v[:, None])).sum(1)
    return v, intercept


def anchored_path(velocity):
    leads = torch.arange(1, 13, dtype=velocity.dtype, device=velocity.device)
    return velocity[:, None]*leads[None, :, None]


def recent_prior(h):
    return h.new_tensor([.05, .05, .05, .05, 1., 1., 1., 1.]).expand(len(h), -1)


def robust_path(history):
    """Fixed baseline: downweight inconsistent points without learned parameters."""
    h = history-history[:, -1:]
    prior = recent_prior(h)
    w = prior
    lead = h.new_tensor(range(-7, 1))[None, :, None]
    for _ in range(4):
        v, a = line_fit(h, w)
        r = (h-a[:, None]-v[:, None]*lead).norm(dim=-1)
        scale = .10+h.diff(dim=1).norm(dim=-1).median(1, keepdim=True).values
        w = prior/(1.+(r/scale).square()).sqrt()
    return anchored_path(line_fit(h, w)[0])


def reliability_features(history):
    h = history-history[:, -1:]
    prior = recent_prior(h)
    lead = h.new_tensor(range(-7, 1))[None, :, None]
    v, a = line_fit(h, prior)
    residual = h-a[:, None]-v[:, None]*lead
    dv = h.diff(dim=1)
    noise = .10+dv.norm(dim=-1).median(1, keepdim=True).values
    before = torch.cat((dv[:, :1], dv), 1)-v[:, None]
    after = torch.cat((dv, dv[:, -1:]), 1)-v[:, None]
    speed = v.norm(dim=-1, keepdim=True)
    axis = v/speed.clamp_min(1e-7)
    signed = (residual*axis[:, None]).sum(-1)/noise
    x = torch.stack((lead[..., 0].expand(len(h), -1)/7,
        residual.norm(dim=-1)/noise, before.norm(dim=-1)/noise, after.norm(dim=-1)/noise,
        (after-before).norm(dim=-1)/noise, signed), dim=-1).asinh()
    return h, prior, x


class WeightedVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(12, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(self, history):
        h, prior, x = reliability_features(history)
        context = x.mean(1, keepdim=True).expand_as(x)
        adjustment = self.score(torch.cat((x, context), dim=-1)).squeeze(-1)
        weights = prior*(2.*adjustment.tanh()).exp()
        return anchored_path(line_fit(h, weights)[0]), weights


def objective(pred, cv, ols, truth, valid, scale, easy, dual):
    """Fit trajectory ADE, with a matched optional dual-reference loss."""
    safe = torch.where(valid[..., None], truth, 0.)
    denom = valid.sum(1).clamp_min(1)*scale
    known = valid.any(1)

    def ade(p):
        return ((p-safe).norm(dim=-1)*valid).sum(1)/denom

    current = ade(pred)
    ordinary = current.sum()/known.sum().clamp_min(1)
    if not dual:
        return ordinary
    harm = (current-ade(cv)).clamp_min(0.)*(1.+3.*easy)+(current-ade(ols)).clamp_min(0.)
    missing = ((pred-cv).norm(dim=-1).amax(1)+(pred-ols).norm(dim=-1).amax(1))/scale
    return ordinary+(harm+missing*(~known)).mean()
