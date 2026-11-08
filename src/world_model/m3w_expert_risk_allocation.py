"""Learn per-expert utility and compare fixed-budget causal scene allocators."""

from itertools import combinations

import numpy as np
import torch
from torch import nn

from src.world_model import m3w_replay_motion as motion


FITS = ("gain_only", "risk_heads")
WEIGHTS = np.vstack((np.eye(5), [(np.eye(5)[a] + np.eye(5)[b]) / 2
                               for a, b in combinations(range(5), 2)]))[1:].astype(np.float32)
SCENE_PENALTY = .25
SHORTLIST = 12


def design(geometry):
    ego, replay, deltas, scale, paths = motion.design(geometry[:, :16].reshape(-1, 8, 2))
    delta = np.einsum("ak,nkhd->nahd", WEIGHTS, deltas)
    distance = np.linalg.norm(delta, axis=-1)
    neighbor = geometry[:, 38:166].reshape(-1, 8, 8, 2)
    mask = geometry[:, 230:294].reshape(-1, 8, 8).all(axis=-1)
    last = np.linalg.norm(neighbor[:, :, -1], axis=-1)
    near = np.where(mask, last, np.inf).min(1)
    near = np.where(np.isfinite(near), near, 0.) / scale
    row = np.column_stack((ego, replay, mask.sum(1) / 8., np.log1p(near)))
    replay_rank = -np.einsum("ak,nk->na", WEIGHTS, replay.reshape(-1, 3, 5).mean(1))
    features = np.concatenate((np.broadcast_to(row[:, None], (*delta.shape[:2], row.shape[-1])),
        np.broadcast_to(WEIGHTS[None], (*delta.shape[:2], WEIGHTS.shape[-1])),
        np.stack((distance.mean(-1), distance[..., -1]), -1) / (12. * scale[:, None, None])), -1)
    return features.astype(np.float32), delta, scale, paths, distance.max(-1) > 1e-6, replay_rank


def targets(floor, delta, target, valid, scale, easy):
    clean = np.where(valid[..., None], target, 0.)
    error = np.linalg.norm(floor - clean, axis=-1)
    excess = np.linalg.norm((floor[:, None] + delta) - clean[:, None], axis=-1) - error[:, None]
    debit = np.linalg.norm(delta, axis=-1)
    gain = np.where(valid[:, None], -excess, -debit).mean(-1)
    harm = np.where(valid[:, None], np.maximum(excess, 0.), debit).mean(-1)
    return (np.stack((gain, harm, harm * easy[:, None]), -1) / (12. * scale[:, None, None])).astype(np.float32)


def network(width):
    return nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU(), nn.Linear(32, 3))


def predict(net, x):
    raw = net(x)
    return torch.cat((raw[..., :1], raw[..., 1:].square()), -1)


def loss(net, x, y, legal, arm):
    prediction = predict(net, x)
    losses = torch.nn.functional.smooth_l1_loss(prediction, y, beta=.02, reduction="none")
    cost = losses[..., 0] if arm == "gain_only" else losses.sum(-1)
    per_row = (cost * legal).sum(1) / legal.sum(1).clamp_min(1)
    return per_row.mean()


def nominate(ranks, legal):
    masked = np.where(legal, ranks, -np.inf)
    j = masked.argmax(1)
    return j, masked[np.arange(len(j)), j]


def costs(base, switched):
    rel = base[:, None] - base[None]
    reference = np.linalg.norm(rel, axis=-1)
    a = np.maximum(reference - np.linalg.norm(switched[:, None] - base[None], axis=-1), 0.).mean(-1)
    both = np.maximum(reference - np.linalg.norm(switched[:, None] - switched[None], axis=-1), 0.).mean(-1)
    np.fill_diagonal(a, 0.)
    return a, both


def choose(ranks, legal, queries, floor=None, delta=None, origins=None):
    j, score = nominate(ranks, legal)
    names = ("independent", "pair_additive", "joint") if floor is not None else ("independent",)
    picked = {a: np.full(len(j), -1, dtype=np.int16) for a in names}
    for q in queries:
        active = q[np.isfinite(score[q])]
        ids = active[np.argsort(-score[active], kind="stable")[:SHORTLIST]]
        k = min(2, len(ids))
        picked["independent"][ids[:k]] = j[ids[:k]]
        if floor is None:
            continue
        if k <= 1:
            for name in names[1:]:
                picked[name][ids] = j[ids]
            continue
        base = floor[ids] + origins[ids, None]
        single, both = costs(base, base + delta[ids, j[ids]])
        divisor = len(ids) - 1
        unary = score[ids] - SCENE_PENALTY * single.sum(1) / divisor
        p, q = np.triu_indices(len(ids), 1)
        additive = unary[p] + unary[q]
        interaction = (both[p, q] - single[p, q] - single[q, p]) / divisor
        for name, objective in (("pair_additive", additive),
                                ("joint", additive - SCENE_PENALTY * interaction)):
            take = objective.argmax()
            which = ids[[p[take], q[take]]]
            picked[name][which] = j[which]
    return picked


def selected_paths(floor, delta, selection):
    forecast = floor.copy()
    at = np.flatnonzero(selection >= 0)
    forecast[at] += delta[at, selection[at]]
    return forecast
