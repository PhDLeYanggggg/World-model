"""Conditional feasibility targets for an equal-count one-edit exchange."""

import torch
from torch import nn
from torch.nn import functional as F

from src.world_model.m3w_one_exchange_ranker import ExchangeRanker, objective


class RiskRanker(ExchangeRanker):
    def __init__(self, width):
        super().__init__(width)
        self.model[-1] = nn.Linear(32, 7)


def loss(scores, transition, gain, status, factored):
    value = objective(scores[:, :4], transition, gain)
    if not factored:
        return value
    for i in range(3):
        valid = status >= i
        bce = F.binary_cross_entropy_with_logits(scores[:, 4+i], (status > i).to(scores.dtype), reduction='none')
        value = value + (bce*valid).sum()/valid.sum().clamp_min(1)
    return value


def success_log_probability(scores):
    return F.logsigmoid(scores[:, 4:]).sum(1)
