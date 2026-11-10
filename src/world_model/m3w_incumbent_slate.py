"""Compare legal exchanges with an explicitly represented unchanged allocation."""

import torch
from torch.nn import functional as F

from src.world_model.m3w_exchange_listwise import ListRanker


def incumbent_targets(option_pass, transition, lengths):
    """Recover donor-only incumbent labels from archived outcome transitions."""
    if sum(lengths) != len(option_pass) or transition.shape != option_pass.shape:
        raise ValueError('Misaligned donor transitions')
    if not lengths:
        return option_pass.new_empty(0, dtype=torch.bool)
    anchors, offset = [], 0
    for n in lengths:
        if n < 1:
            raise ValueError('Empty donor slate')
        value = option_pass[offset:offset+n].long()+1-transition[offset:offset+n]
        if not ((value == 0) | (value == 1)).all() or not (value == value[0]).all():
            raise ValueError('Inconsistent incumbent label within a donor slate')
        anchors.append(value[0].bool())
        offset += n
    return torch.stack(anchors)


def with_incumbent(features):
    """Add no-exchange at index zero using only shared causal recording context."""
    if features.ndim != 3 or not features.shape[1] or (features.shape[-1]-9) % 4:
        raise ValueError('Expected batched exchange features')
    width = (features.shape[-1]-9)//4
    context_start = 2*width+6
    anchor = torch.zeros_like(features[:, :1])
    anchor[..., context_start:] = features[:, :1, context_start:]
    values = torch.cat((anchor, features), dim=1)
    flag = values.new_zeros((*values.shape[:2], 1))
    flag[:, 0] = 1.
    return torch.cat((values, flag), dim=-1)


def objective(logits, option_pass, valid, incumbent_pass, *, pairwise):
    y = torch.cat((incumbent_pass[:, None], option_pass), dim=1).to(logits.dtype)
    mask = torch.cat((torch.ones_like(incumbent_pass[:, None]), valid), dim=1)
    if logits.shape != mask.shape:
        raise ValueError('Incumbent and option logits differ in shape')
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
    point = ((bce*mask).sum(1)/mask.sum(1)).mean()
    if not pairwise:
        return point
    difference = y[:, 1:]-y[:, :1]
    comparable = (difference != 0) & valid
    margin = logits[:, 1:]-logits[:, :1]
    pair = F.softplus(-difference*margin)
    return point + ((pair*comparable).sum(1)/comparable.sum(1).clamp_min(1)).mean()


def choices(scores):
    if scores.ndim != 1 or len(scores) < 2 or not torch.isfinite(scores).all():
        raise ValueError('Expected finite scores for an incumbent and at least one exchange')
    # Ties favor retaining the current allocation; no test-selected margin is used.
    return dict(forced=int(scores[1:].argmax())+1, optional=int(scores.argmax()))
