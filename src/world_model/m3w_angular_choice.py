"""Predict paired gain for a finite angular menu with an exact keep action."""

import torch
from torch import nn
from torch.nn import functional as F


DEGREES = tuple(a for a in range(-90, 91, 15) if a)


class ChoiceHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(),
                                 nn.Linear(32, 2*len(DEGREES)))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x).reshape(len(x), 2, len(DEGREES))


def targets(cv, anchor, y, valid, scale):
    a = torch.deg2rad(cv.new_tensor(DEGREES))[None, :, None]
    x, z = (anchor-cv).unsqueeze(1).unbind(-1)
    c, s = a.cos(), a.sin()
    p = cv[:, None]+torch.stack((x*c-z*s, x*s+z*c), dim=-1)
    error = torch.linalg.vector_norm(p-y[:, None], dim=-1)
    ref = torch.linalg.vector_norm(anchor-y, dim=-1)[:, None]
    disagreement = torch.linalg.vector_norm(p-anchor[:, None], dim=-1)
    normalizer = (cv.shape[1]*scale)[:, None]
    return ((ref-error)*valid[:, None]).sum(2)/normalizer, (
        disagreement*(~valid[:, None])).sum(2)/normalizer


def loss(pred, observed, debit, factor):
    z = debit if factor else observed-debit
    return F.smooth_l1_loss(pred[:, 0], observed, beta=.05)+F.smooth_l1_loss(pred[:, 1], z, beta=.05)


def choose(pred, factor, force=False):
    score = pred[:, 0]-pred[:, 1].clamp_min(0.) if factor else pred[:, 1]
    if not force:
        score = torch.cat((score.new_zeros(len(score), 1), score), dim=1)
    return score.argmax(1)+int(force)
