"""Causal matched-count choice with an explicit learned pair interaction."""

import numpy as np
import torch
from torch import nn

from src.world_model.m3w_prefix_expert_mixture import features


def head(width, outputs):
    return nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 48), nn.SiLU(), nn.Linear(48, outputs))


def scale(h):
    return np.linalg.norm(np.diff(h, axis=1), axis=-1).mean(1).clip(1.)


def row_features(h, delta):
    past, s = features(h, 'prefix')
    edits = np.arcsinh(delta / s[:, None, None, None]).astype(np.float32)
    ids = np.broadcast_to(np.eye(delta.shape[1], dtype=np.float32), (len(h), delta.shape[1], delta.shape[1]))
    return np.concatenate((np.broadcast_to(past[:, None], (*edits.shape[:2], past.shape[-1])),
                           edits.reshape(*edits.shape[:2], -1), ids), axis=-1)


def pair_context(h1, h2, cv1, cv2):
    offset = h1[:, -1] - h2[:, -1]
    sep = np.linalg.norm(offset, axis=-1)
    axis = offset / sep[:, None].clip(1e-8)
    axis[sep < 1e-8] = 0.
    basis = np.stack((axis, np.stack((-axis[:, 1], axis[:, 0]), 1)), axis=-1)
    s = (scale(h1) + scale(h2)) / 2.
    move = (h1[:, -1] - h1[:, 0]) - (h2[:, -1] - h2[:, 0])
    p = np.concatenate((np.column_stack((sep / s, np.log(s))),
        np.einsum('nd,ndk->nk', move, basis) / s[:, None],
        (np.einsum('ntd,ndk->ntk', (cv1 - cv2)[:, 3::4], basis) / s[:, None, None]).reshape(len(s), 6)), axis=1)
    return p, basis, s


def pair_features(context, a, b):
    p, basis, s = context
    a, b = a[:, 3::4] / s[:, None, None], b[:, 3::4] / s[:, None, None]
    d = np.einsum('ntd,ndk->ntk', a - b, basis).reshape(len(a), 6)
    interaction = np.concatenate((np.linalg.norm(a, axis=-1) + np.linalg.norm(b, axis=-1),
                                  (a * b).sum(-1)), axis=1)
    return np.arcsinh(np.concatenate((p, d, interaction), axis=1)).astype(np.float32)


def pair_target(cv, truth, valid, a, b, norm):
    base = np.linalg.norm(cv - truth, axis=-1)
    changed = a - b
    gain = base - np.linalg.norm(cv + changed - truth, axis=-1)
    return (np.where(valid, gain, -np.linalg.norm(changed, axis=-1)).mean(1) / norm).clip(-8., 8.)


def rank_output(pred):
    return .5 * (pred[..., 0] + pred[..., 1]) - np.maximum(pred[..., 2], 0.) - np.maximum(pred[..., 3], 0.)


def matched_radius(cv, paths, query):
    """Keep choices but shrink each arm to the smallest total peak in its query."""
    _, inv = np.unique(query, return_inverse=True)
    delta = np.stack([p-cv for p in paths.values()])
    peak = np.linalg.norm(delta, axis=-1).max(-1)
    mass = np.stack([np.bincount(inv, weights=r) for r in peak])
    target = mass.min(0)
    weight = np.divide(target, mass, out=np.zeros_like(mass), where=mass > 0.)
    final = cv[None] + weight[:, inv, None, None] * delta
    realized = np.linalg.norm(final-cv[None], axis=-1).max(-1)
    for p in realized:
        np.testing.assert_allclose(np.bincount(inv, weights=p), target, atol=1e-8, rtol=1e-8)
    np.testing.assert_array_equal(peak > 0., realized > 0.)
    return dict(zip(paths, final, strict=True))


def choose(scores, legal, edges, solo_i, solo_j, both, cap, pair_weight):
    """Same available recipients and exact counts for all comparison arms."""
    n, k = legal.shape
    if not np.isfinite(scores).all() or scores.shape != legal.shape:
        raise ValueError('Finite causal action scores required')
    best = np.where(legal, scores, -np.inf).argmax(1)
    row = np.arange(n)
    base = np.where(legal.any(1), scores[row, best], -np.inf)
    count = min(cap, int(legal.any(1).sum()))
    direct = np.argsort(-base, kind='stable')[:count]
    uniform = None
    score = -np.inf
    for j in range(k):
        if legal[:, j].sum() >= count:
            ix = np.argsort(-np.where(legal[:, j], scores[:, j], -np.inf), kind='stable')[:count]
            value = scores[ix, j].sum()
            if value > score:
                uniform, score = (ix, j), value
    i, j = edges
    deg = np.bincount(np.r_[i, j], minlength=n).clip(1.)
    w = pair_weight * 2. / (deg[i] + deg[j])
    linear = base.copy()
    np.add.at(linear, i, w * solo_i)
    np.add.at(linear, j, w * solo_j)
    interaction = w * (both - solo_i - solo_j)
    add = np.argsort(-linear, kind='stable')[:count]
    joint = []
    for _ in range(count):
        v = int(linear.argmax())
        joint.append(v)
        mask = i == v
        np.add.at(linear, j[mask], interaction[mask])
        mask = j == v
        np.add.at(linear, i[mask], interaction[mask])
        linear[v] = -np.inf
    result = {}
    for key, ix in (('independent', direct), ('additive_pair', add), ('joint', np.asarray(joint, dtype=int))):
        result[key] = np.full(n, -1, dtype=np.int8)
        result[key][ix] = best[ix]
    result['uniform'] = result['independent'].copy()
    if uniform is not None:
        result['uniform'].fill(-1)
        result['uniform'][uniform[0]] = uniform[1]
    return result
