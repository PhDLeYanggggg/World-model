"""Independent actor values with explicit scene qualification margins."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.evaluation.m3w_cv_anchor_budget import candidate_terms, risk_excess

TEMPERATURE = .10


def components(cv, ols, bank, target, valid, cut, row, col, unit):
    _, fallback, gain, _ = candidate_terms(cv, ols, bank, target, valid)
    risk = risk_excess(cv, bank, target, valid, cut)
    y = np.where(valid[..., None], target, 0.)
    n = valid.sum(1).clip(1)
    c = np.where(valid, np.linalg.norm(cv-y, axis=-1), 0.).sum(1)/n
    p = np.where(valid[:, None], np.linalg.norm(bank-y[:, None], axis=-1), 0.).sum(2)/n[:, None]
    easy = (c > 0.) & (c <= cut)
    signed = np.where(easy[row], p[row, col]-c[row], 0.)
    local = np.column_stack((gain[row, col]/valid.shape[1], risk[row, col], signed))
    scene = np.column_stack((fallback/valid.shape[1], .02*np.where(easy, c, 0.), easy))
    scene[:, :2] /= unit[:, None]
    return local/unit[row, None], scene


def margins(action, scene, opts, row, units):
    """Five signed margins and edited-easy support on the fixed two-slot menu."""
    local = (action[opts]*units[row[opts], None]).sum(-2)
    intercept = (scene[:, :2]*units[:, None]).sum(0)
    cv, ols, all_risk, easy_risk, signed = local.unbind(-1)
    return torch.stack((cv, ols+intercept[0], -all_risk, -easy_risk, intercept[1]-signed), dim=-1)/units.mean()


def qualification(value, scene, opts, row):
    support = scene[row[opts], 2].bool().any(-1)
    return support & (value[:, :2] > 0.).all(1) & (value[:, 2:] >= 0.).all(1)


def slate_score(value, logits, opts, row):
    absent = F.logsigmoid(-logits[row[opts]]).sum(1)
    log_support = (-absent.expm1()).clamp_min(1e-8).log()
    return log_support+F.logsigmoid(value/TEMPERATURE).sum(1)


class Assignment(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.edit = nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, 5))
        self.actor = nn.Sequential(nn.Linear(21, 32), nn.SiLU(), nn.Linear(32, 24), nn.SiLU(), nn.Linear(24, 3))
        with torch.no_grad():
            self.actor[-1].bias[1] = -3.

    def forward(self, x, ctx):
        z = self.actor(ctx)
        # The easy allowance is small, nonnegative, and zero when there is no easy reference.
        cap = .02*F.softplus(z[:, 1])
        return self.edit(x), torch.stack((z[:, 0], cap*z[:, 2].sigmoid(), z[:, 2]), dim=1)


def supervised_loss(logp, label):
    return torch.where(label, -logp, -(-logp.expm1()).clamp_min(1e-8).log()).mean()
