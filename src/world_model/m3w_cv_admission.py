"""Factorized expected-risk features and an optional matched-count decoder."""

import numpy as np
import torch
from torch.nn import functional as F


def features(past, delta, scale):
    d = delta / scale[:, None, None]
    radius = np.linalg.norm(d, axis=-1).max(-1)
    return np.concatenate((past, np.arcsinh(d).reshape(len(d), -1),
                           np.log1p(radius)[:, None]), axis=1).astype(np.float32)


def decode(logits):
    """Gain, positive mass, and two mutually exclusive support outcomes."""
    gain = logits[..., :2]
    mass = F.softplus(logits[..., 2:4])
    share = logits[..., 4:6].sigmoid()
    fraction = torch.cat((logits[..., 6:8], torch.zeros_like(logits[..., 6:7])), -1).softmax(-1)[..., :2]
    return gain, mass, mass * share, fraction


def objective(net, x, target):
    gain, mass, easy, prob = decode(net(x))
    err = sum(F.smooth_l1_loss(p, y) for p, y in
              ((gain, target[:, :2]), (mass, target[:, 2:4]), (easy, target[:, 4:6])))
    p = torch.cat((prob, 1. - prob.sum(-1, keepdim=True)), -1).clamp_min(1e-8)
    return err - (p.log() * target[:, 6:9]).sum(-1).mean()


def predicted(gain, mass, easy, probability, relative_radius):
    missing = probability[..., 1] * relative_radius
    risk = np.stack((mass[..., 1]+missing-.02*mass[..., 0],
                     easy[..., 1]+missing-.02*easy[..., 0]), axis=-1)
    preference = gain.mean(-1)
    return preference, risk, probability[..., 0]


def admissible(gain, risk, p_easy, legal, tolerance, min_easy):
    # The tolerance is on a predicted cost. It never changes the measured 2% rule.
    return legal & (gain > 0.) & (risk.max(-1) <= tolerance) & (p_easy >= min_easy)


def select(score, eligible, cap, edges=None, pair_values=None, pair_weight=.25):
    n = len(score)
    count = min(cap, int(eligible.sum()))
    linear = np.where(eligible, score, -np.inf)
    if edges is None:
        return np.argsort(-linear, kind='stable')[:count]
    i, j = edges
    solo_i, solo_j, both = pair_values.T
    deg = np.bincount(np.r_[i, j], minlength=n).clip(1)
    w = pair_weight * 2. / (deg[i] + deg[j])
    np.add.at(linear, i, w * solo_i)
    np.add.at(linear, j, w * solo_j)
    cross = w * (both-solo_i-solo_j)
    picked = []
    for _ in range(count):
        k = int(linear.argmax())
        picked.append(k)
        left, right = i == k, j == k
        np.add.at(linear, j[left], cross[left])
        np.add.at(linear, i[right], cross[right])
        linear[k] = -np.inf
    return np.asarray(picked, dtype=int)


def choose_setting(rows, expected_records):
    """Only complete, positive calibration configurations can unlock edits."""
    feasible = [r for r in rows if r['edits'] and r['passes'] == expected_records
                and min(r['gain_pp']) > 0.]
    if not feasible:
        return None
    best = max(feasible, key=lambda r: (min(r['gain_pp']), -r['edits']))
    return best['tolerance'], best['min_easy']
