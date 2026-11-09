"""Separate query eligibility from within-query relative qualification."""

import torch
from torch import nn
from torch.nn import functional as F

MODES = ("direct", "query_weighted", "conditional")
POS_WEIGHT = 5.


class Qualification(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.candidate = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 32),
                                       nn.SiLU(), nn.Linear(32, 2))
        self.reference = nn.Sequential(nn.Linear(width, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, options, anchor):
        return self.candidate(options), self.reference(anchor).squeeze(-1)


def binary(z, y):
    return F.binary_cross_entropy_with_logits(z, y, pos_weight=z.new_tensor(POS_WEIGHT))


def ranking(z, head):
    positive, negative = z.chunk(2)
    return F.softplus(negative[:, head]-positive[:, head]).mean()


def probabilities(candidate, reference):
    correction = candidate.new_tensor(POS_WEIGHT).log()
    direct, conditional = (candidate-correction).sigmoid().unbind(-1)
    q = (reference-correction).sigmoid()
    # Query weighting alone cannot change within-query ordering.
    return dict(zip(MODES, (direct, q*direct, q*conditional), strict=True)), q
