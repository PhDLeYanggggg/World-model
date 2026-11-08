"""Rank conditional gains without learning or changing support probabilities."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ARMS = ("conditional_point", "common_step_rank")


def network(width):
    net = nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU(),
                        nn.Linear(32, 12), nn.Tanh())
    nn.init.zeros_(net[-2].weight)
    nn.init.zeros_(net[-2].bias)
    return net


def common_gaps(pred, frac, debit, visible):
    common = visible[:, 0] & visible[:, 1]

    def gap(v):
        g = v * debit
        return ((g[:, 0]-g[:, 1])*common).sum(-1) / common.sum(-1).clamp_min(1)

    return gap(pred), gap(frac)


def loss(pred, frac, debit, visible, arm):
    if arm not in ARMS:
        raise ValueError("Unknown conditional-gain treatment")
    per_step = F.smooth_l1_loss(pred, frac, beta=.2, reduction="none")
    point = (per_step * debit * visible).mean()
    if arm == "conditional_point":
        return point
    estimate, realized = common_gaps(pred, frac, debit, visible)
    return point + .5*(realized.abs()*F.softplus(-realized.sign()*estimate/.025)).mean()


def combined(mu, p, distance):
    """Conditional expected known gain minus expected missing-step debit."""
    if not np.isfinite(mu).all() or not np.isfinite(p).all() or not np.isfinite(distance).all():
        raise ValueError("Expected finite conditional gain, support and disagreement")
    return ((p*mu-(1.-p))*distance).mean(-1)


def support_layout(probability, debit_score, eligible, queries):
    layout = []
    for q in queries:
        use = q[eligible[q]]
        ordered = use[np.argsort(-probability[use], kind="stable")]
        ref = use[np.argsort(-debit_score[use], kind="stable")[:2]]
        layout.extend((chunk, int(np.isin(chunk, ref).sum())) for chunk in np.array_split(ordered, 4)
                      if len(chunk))
    return layout


def matched_choices(scores, layout):
    take = np.zeros(len(scores), bool)
    for ids, count in layout:
        if not 0 <= count <= len(ids):
            raise ValueError("Invalid registered support budget")
        at = ids[np.argsort(-scores[ids], kind="stable")[:count]]
        take[at] = True
    return take
