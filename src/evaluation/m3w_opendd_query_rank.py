"""Fixed-budget query allocation and differentiable opportunity regret."""
import numpy as np
import torch
from torch.nn import functional as F

from src.evaluation.m3w_opendd_location import choose


def training_loss(score, legal, cv, utility=None):
    if not legal.any(1).all():
        raise ValueError('An optimization query must contain a legal edit')
    loss = (F.smooth_l1_loss(score, cv, reduction='none', beta=.1)*legal).sum(1) / legal.sum(1)
    if utility is not None:
        best = utility.masked_fill(~legal, -torch.inf).amax(1, keepdim=True)
        prob = (score/.10).masked_fill(~legal, -torch.inf).softmax(dim=1)
        regret = torch.where(legal, best - utility, 0.)
        loss = loss + (prob * regret).sum(1)
    return loss.mean()


def restrict(legal, hindcast, offsets, mode):
    if mode == 'full':
        return legal.copy()
    restricted = np.zeros_like(legal)
    if mode == 'recipient':
        rows = np.flatnonzero(legal.any(1))
        option = np.where(legal, hindcast, -np.inf).argmax(1)
        restricted[rows, option[rows]] = True
    elif mode == 'candidate':
        chosen = choose(hindcast, legal, offsets)
        restricted[chosen >= 0] = legal[chosen >= 0]
    else:
        raise ValueError(mode)
    return restricted


def limited_queries(legal, score, offsets, scale):
    count = len(offsets)-1
    result = np.zeros(count, dtype=bool)
    strength = np.maximum.reduceat(np.where(legal, score, -np.inf).max(1), offsets[:-1]) * scale
    feasible = np.flatnonzero(np.isfinite(strength))
    if len(feasible):
        k = max(1, len(feasible)//5)
        order = np.argsort(-strength[feasible], kind='stable')
        result[feasible[order[:k]]] = True
    return result
