"""Estimate two different prerequisites for releasing a fixed substitution."""

import torch
from torch import nn
from torch.nn import functional as F


class ReplayHurdle(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.anchor = nn.Sequential(nn.Linear(width, 32), nn.SiLU(), nn.Linear(32, 1))
        self.candidate = nn.Sequential(nn.Linear(2*width+1, 48), nn.SiLU(), nn.Linear(48, 24),
                                       nn.SiLU(), nn.Linear(24, 2))

    def forward(self, anchor, candidate, replay):
        x = torch.cat((candidate, candidate-anchor, replay[..., None]), dim=-1)
        return torch.cat((self.anchor(anchor), self.candidate(x)), dim=-1)


def loss(logits, labels):
    a, joint = labels.unbind(-1)
    reference = F.binary_cross_entropy_with_logits(logits[..., 0], a)
    direct = F.binary_cross_entropy_with_logits(logits[..., 1], joint)
    conditional = F.binary_cross_entropy_with_logits(logits[..., 2], joint, reduction="none")
    return reference+direct+(conditional*a).sum()/a.sum().clamp_min(1.)


def scores(logits, replay):
    a, direct, conditional = logits.sigmoid().unbind(-1)
    return torch.stack((replay, replay*a, direct, a*conditional, a, conditional), dim=-1)
