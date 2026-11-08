"""Matched marginal and conditionally supervised recording risk heads."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SupportMoments(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.context = nn.Sequential(nn.Linear(width, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU())
        self.reference = nn.Linear(32, 4)
        self.effects = nn.Linear(32, 6)

    def forward(self, x, bound):
        z = self.context(x)
        r = self.reference(z.mean(1))
        known, easy = r[:, :2].sigmoid().unbind(1)
        means = F.softplus(r[:, 2:])
        weights = torch.stack((known*easy, known*(1.-easy)), dim=1)
        means_weighted = weights*means
        ref = torch.stack((means_weighted.sum(1), means_weighted[:, 0]), dim=1)
        effects = self.effects(z).reshape(*bound.shape, 2, 3).softmax(-1)[..., :2]*bound[..., None, None]
        weighted = effects*weights[:, None, :, None]
        benefit, harm = weighted.sum(2).unbind(-1)
        missing = (1.-known[:, None])*bound
        cost = torch.stack((benefit-harm-missing, harm+missing, weighted[:, :, 0, 1]+missing), dim=-1)
        return ref, cost, r[:, :2], means, effects


def objective(prediction, b, gain, known, easy, radius, conditional):
    ref, costs, logits, means, effects = prediction
    benefit, harm = gain.clamp_min(0.), (-gain).clamp_min(0.)
    debit = (~known[:, None])*radius
    targets = torch.stack((gain-debit, harm+debit, easy[:, None]*harm+debit), dim=-1)
    masses = torch.stack((b, b*easy), dim=-1)
    marginal = F.mse_loss(ref, masses)+F.mse_loss(costs, targets)
    if not conditional:
        return marginal
    labels = torch.stack((known, easy), dim=1).to(b.dtype)
    likelihood = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    event = likelihood[:, 0].mean()+(known*likelihood[:, 1]).sum()/known.sum().clamp_min(1)
    mask = torch.stack((easy, known & ~easy), dim=1)
    count = mask.sum().clamp_min(1)
    conditional_ref = (mask*(means-b[:, None]).square()).sum()/count
    action = torch.stack((benefit, harm), dim=-1)
    conditional_cost = (mask[:, None, :, None]*(effects-action[:, :, None]).square()).sum()
    conditional_cost /= count*effects.shape[1]*2
    return marginal+.25*event+.5*(conditional_ref+conditional_cost)


def allocate(ref, moment, legal, query_inverse, count, weight, *, query_counts=None):
    """Exact action count, with risk-constrained ranking on a fixed price grid."""
    if not 0 <= count <= legal.any(1).sum():
        raise ValueError("Budget exceeds legal action support")
    n = len(legal)
    rows = np.arange(n)
    gain, all_harm, easy_harm = moment.transpose(2, 0, 1)
    utility = gain*weight[:, None]/weight.mean()
    excess = all_harm-.02*ref[:, :1]
    easy_excess = easy_harm-.02*ref[:, 1:]

    def top(score):
        masked = np.where(legal, score, -np.inf)
        best = masked.argmax(1)
        rank = masked[rows, best]
        if query_counts is None:
            on = np.argsort(-rank, kind="stable")[:count]
        else:
            by_query = np.lexsort((rows, -rank, query_inverse))
            sizes = np.bincount(query_inverse, minlength=len(query_counts))
            within_rank = rows-np.repeat(np.r_[0, np.cumsum(sizes)[:-1]], sizes)
            on = by_query[within_rank < query_counts[query_inverse[by_query]]]
        if len(on) != count or not np.isfinite(rank[on]).all():
            raise ValueError("Exact count includes an illegal action")
        choice = np.full(n, -1, dtype=np.int8)
        choice[on] = best[on]
        mass = ref[on].sum(0)
        ratios = [float(h[on, best[on]].sum()/m) if m > 0. else None
                  for h, m in zip((all_harm, easy_harm), mass, strict=True)]
        value = float(utility[on, best[on]].sum())
        over = max((np.inf if r is None else max(0., r-.02)) for r in ratios)
        return choice, ratios, (-over, value)

    independent = top(utility)
    alternatives = [independent]
    for price in (1., 4., 16., 64., 256.):
        for easy_weight in (1., 4.):
            alternatives.append(top(utility-price*(excess+easy_weight*easy_excess)))
    chosen = max(alternatives, key=lambda t: t[2])
    return (independent[0], independent[1]), (chosen[0], chosen[1])
