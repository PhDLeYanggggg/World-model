"""Direct and coherent conditional-moment estimates of the original risks."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.evaluation.m3w_selection_forecast_readout import errors


class Head(nn.Module):
    def __init__(self, past_width, action_width):
        super().__init__()
        self.past_width = past_width
        self.history = nn.Sequential(nn.Linear(past_width, 48), nn.SiLU(), nn.Linear(48, 5))
        self.action = nn.Sequential(nn.Linear(action_width, 48), nn.SiLU(), nn.Linear(48, 3))

    def forward(self, x):
        return self.history(x[:, :self.past_width]), self.action(x)


def decode(history, action, radius, component):
    logits = history[:, :3]
    unknown, easy, other = logits.softmax(1).unbind(1)
    me, mo = F.softplus(history[:, 3:] - 1.).unbind(1)
    he, ho = F.softplus(action[:, :2] - 3.).unbind(1)
    if component:
        benefit = F.softplus(action[:, 2] - 3.)
        debit = unknown * radius
        observed_easy = easy * he
        moments = (observed_easy + other * ho + debit, observed_easy + debit,
                   easy * (he - benefit), easy * me + other * mo, easy * me)
    else:
        moments = (he + ho, he, action[:, 2], me + mo, me)
    return torch.stack(moments, 1), logits


def targets(cv, path, y, valid, scale, cut):
    c, _ = errors(cv, y, valid)
    p, _ = errors(path, y, valid)
    known = valid.any(1)
    easy = known & (c > 0.) & (c <= cut)
    shift = np.where(known, p-c, 0.)
    radius = np.linalg.norm(path - cv, axis=-1).max(1)
    debit = np.where(known, 0., radius)
    h = shift.clip(0.)
    target = np.column_stack((h + debit, np.where(easy, h, 0.) + debit,
        np.where(easy, shift, 0.), np.where(known, c, 0.), np.where(easy, c, 0.)))
    category = np.where(known, np.where(easy, 1, 2), 0)
    return (target / scale[:, None]).astype(np.float32), category, (radius / scale).astype(np.float32)


def objective(net, x, y, category, radius, component):
    pred, logits = decode(*net(x), radius, component)
    w = (1. + radius).square().reciprocal()
    err = (pred-y).square().mean(1)
    return (w*err).mean() + .15 * F.cross_entropy(logits, category)


def risk(moments):
    return np.concatenate((moments[..., :2] - .02 * moments[..., 3:5], moments[..., 2:3]), axis=-1)
