"""Masked training costs for a CV versus two-reference forecast contrast."""
import math

import torch


def causal_references(history):
    h = history-history[:, -1:]
    slope = (h[:, -4:] * h.new_tensor([-.3, -.1, .1, .3])[None, :, None]).sum(1)
    t = torch.arange(1, 13, dtype=h.dtype, device=h.device)[None, :, None]
    return (h[:, -1:]-h[:, -2:-1])*t, slope[:, None]*t


def objective(pred, target, valid, cv, ols, scale, easy_cut, *, dual):
    known = valid.any(1)
    clean = torch.where(valid[..., None], target, 0.)

    def ade(p):
        error = torch.linalg.vector_norm(p-clean, dim=-1)
        return torch.where(valid, error, 0.).sum(1)/valid.sum(1).clamp_min(1)

    actual = ade(pred)
    cv_ade = ade(cv)
    easy = known & (cv_ade > 0.) & (cv_ade <= easy_cut)

    def conservative(ref, baseline):
        unknown = torch.linalg.vector_norm(pred-ref, dim=-1).amax(1)
        return (torch.where(known, actual-baseline, unknown)/scale).mean()

    gains = torch.stack((conservative(cv, cv_ade), conservative(ols, ade(ols))))
    harm = torch.relu(actual-cv_ade)/scale
    penalty = .5*harm.mean()+2.*(harm*easy).mean()
    risk = .25*(torch.logsumexp(gains/.25, dim=0)-math.log(2.)) if dual else gains[0]
    return risk+penalty, gains, penalty
