"""Train a one-sided harm estimate on complete two-action choices."""

import torch
from torch import nn
from torch.nn import functional as F

QUANTILE = .8


def quantile_loss(estimate, actual, q=QUANTILE):
    residual = actual-estimate
    return torch.maximum(q*residual, (q-1.)*residual).mean()


class Harm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, 3))
        self.margin = nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, 3))
        self.cap = nn.Sequential(nn.Linear(21, 24), nn.SiLU(), nn.Linear(24, 1))
        with torch.no_grad():
            self.margin[-1].bias.fill_(-3.)
            self.cap[-1].bias.fill_(-3.)

    def forward(self, x, ctx, choices, row, unit):
        scale = unit[row[choices], None]/unit.mean()
        mean = (self.point(x)[choices]*scale).sum(1)
        cap = .02*(F.softplus(self.cap(ctx).flatten())*unit).sum()/unit.mean()
        mean = mean-torch.stack((cap*0., cap*0., cap))
        extra = (F.softplus(self.margin(x)[choices])*scale).sum(1)
        return mean, mean.detach()+extra
