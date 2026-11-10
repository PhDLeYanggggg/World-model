"""Rank interchangeable plans within a recording, not across recordings."""

from torch import nn
from torch.nn import functional as F


class ListRanker(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 48), nn.GELU(), nn.Linear(48, 32),
                                 nn.GELU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def loss(logits, target, valid, *, listwise):
    """Equal slate weights; padded options and homogeneous ranking terms are ignored."""
    y = target.to(logits.dtype)
    n = valid.sum(-1).clamp_min(1)
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
    point = ((bce * valid).sum(-1) / n).mean()
    if not listwise:
        return point
    pos = target & valid
    positives = pos.sum(-1)
    mixed = (positives > 0) & (positives < n)
    logp = F.log_softmax(logits.masked_fill(~valid, float('-inf')), dim=-1)
    logp = logp.masked_fill(~pos, 0.)
    rank = -(logp.sum(-1) / positives.clamp_min(1))
    return point + (rank * mixed).sum() / mixed.sum().clamp_min(1)
