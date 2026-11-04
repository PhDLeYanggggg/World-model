"""Logistic two-stage deferral surrogate, Mao et al., ICML 2024, Eq. (3).

This is a loss primitive, not a trained control or a relative-risk certificate.
Costs are supervision, never inference features. Original source:
https://proceedings.mlr.press/v235/mao24d.html
"""
import math

import torch
from torch.nn import functional as F


def two_stage_logistic(logits, costs, legal, observed):
    """Return per-row losses and the supervised mask without imputing outcomes.

    For legal actions k with nonnegative costs c_k, Eq.(3) assigns weight
    sum_j(c_j)-c_k to each action's logistic loss. There is no oracle-class
    hard label, per-row weight normalization, or empirical risk-budget change.
    A one-action row has no routing decision and contributes zero.
    """
    if (logits.ndim != 2 or costs.shape != logits.shape or legal.shape != logits.shape
            or observed.shape != logits.shape[:1] or legal.dtype != torch.bool
            or observed.dtype != torch.bool or not torch.isfinite(logits).all()
            or not legal.any(1).all()):
        raise ValueError('Finite logits, legal actions and explicit observation mask required')
    supervised = observed & (legal.sum(1) > 1)
    mask = legal & supervised[:, None]
    if not torch.isfinite(costs[mask]).all() or (costs[mask] < 0).any():
        raise ValueError('Observed expert costs must be finite and nonnegative')
    # Unknown labels may be NaN. They are excluded, not set to observed zeros.
    c = torch.where(mask, costs.detach(), torch.zeros_like(costs))
    weights = torch.where(mask, c.sum(1, keepdim=True)-c, torch.zeros_like(c))
    logp = F.log_softmax(logits.masked_fill(~legal, -torch.inf), 1)
    loss = -(weights*torch.where(legal, logp, torch.zeros_like(logp))).sum(1)/math.log(2)
    return loss, supervised
