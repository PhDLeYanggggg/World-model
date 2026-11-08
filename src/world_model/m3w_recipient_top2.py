"""Exact expected top-two gain training with fixed causal forecasts."""

import numpy as np
import torch
from torch import nn

from src.world_model import m3w_expert_risk_allocation as prior


ARMS = ("point", "top2", "top2_context")
AMPLITUDE = .25


def inputs(geometry, queries):
    x, delta, scale, paths, legal, replay = prior.design(geometry)
    j, score = prior.nominate(replay, legal)
    take = np.arange(len(j))
    valid = np.isfinite(score)
    change = np.where(valid[:, None, None], AMPLITUDE * delta[take, j], 0.)
    floor = paths["ols4"]
    magnitude = np.linalg.norm(change, axis=-1).mean(1)
    features = np.column_stack((x[take, j], np.where(valid, score, 0.)))
    context = np.zeros((len(j), 7), dtype=np.float32)
    scale_by_query = np.empty(len(j), dtype=np.float32)
    for q in queries:
        columns = features[q][:, [14, 15, 31, 32, 38, 39, 40]]
        median = np.median(columns, axis=0)
        offset = columns - median
        context[q] = offset / np.maximum(np.abs(offset).mean(0), .1)
        scale_by_query[q] = 12. * scale[q].mean()
    features = np.column_stack((features, np.zeros_like(context))).clip(-16, 16).astype(np.float32)
    contextual = features.copy()
    contextual[:, -context.shape[1]:] = context.clip(-16., 16.)
    return features, contextual, scale_by_query, paths, change, valid, magnitude, np.where(valid, score, 0.)


def top2_probability(logits, valid):
    """Inclusion probability of two weighted draws without replacement."""
    # Bounded logits and double precision avoid cancellation near a certain first draw.
    z = logits.double().clamp(-6., 6.)
    w = torch.where(valid, z.exp(), 0.)
    total = w.sum(-1, keepdim=True)
    first = w / total.clamp_min(1e-12)
    remaining = total - w
    ratio = w / remaining.clamp_min(1e-12)
    second = first * (ratio.sum(-1, keepdim=True) - ratio)
    enough = valid.sum(-1, keepdim=True) >= 2
    return (first + torch.where(enough, second, 0.)).to(logits.dtype)


def train_loss(net, x, target, valid, arm):
    output = net(x).squeeze(-1)
    if arm == "point":
        l = torch.nn.functional.smooth_l1_loss(output, target, reduction="none", beta=.02)
        return ((l * valid).sum(-1) / valid.sum(-1).clamp_min(1)).mean()
    if arm not in ARMS:
        raise ValueError(f"Unknown training arm: {arm}")
    probability = top2_probability(output / .05, valid)
    return -(probability * target).sum(-1).mean() / 2.


def network(width):
    net = nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU(), nn.Linear(32, 1))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


def choose(score, eligible, queries):
    picked = np.zeros(len(score), dtype=bool)
    for q in queries:
        legal = q[eligible[q]]
        picked[legal[np.argsort(-score[legal], kind="stable")[:2]]] = True
    return picked
