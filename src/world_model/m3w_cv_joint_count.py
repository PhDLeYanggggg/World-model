"""Causal query-slate generation and shared-count comparisons."""
import numpy as np
import torch
from torch.nn import functional as F


FRACTIONS = (.25, .50, .75, 1.)
PAIR_FEATURES = 6


def menu(gain, risk, legal, agreement, anchor):
    n = len(legal)
    size = int(legal.any(1).sum())
    anchor_k = int((anchor >= 0).sum())
    if anchor_k > size:
        raise ValueError('Portfolio exceeds legal count')
    counts = sorted({0, anchor_k, *(int(np.ceil(f * size)) for f in FRACTIONS)})
    g = gain.min(-1)
    signed = risk[..., 0] + 2. * risk[..., 1] + risk[..., 2]
    scores = (g - 2. * risk.clip(0.).sum(-1), g - signed, g - 4. * signed,
              gain[..., 1] - 2. * signed, agreement)
    plans = [np.full(n, -1, dtype=np.int16)]
    independent = {0: 0}
    for k in counts[1:]:
        independent[k] = len(plans)
        seen = set()
        for score in scores:
            masked = np.where(legal, score, -np.inf)
            a = masked.argmax(1)
            top = np.argsort(-masked[np.arange(n), a], kind='stable')[:k]
            plan = np.full(n, -1, dtype=np.int16)
            plan[top] = a[top]
            token = plan.tobytes()
            if token not in seen:
                plans.append(plan)
                seen.add(token)
        if k == anchor_k and anchor.astype(np.int16).tobytes() not in seen:
            plans.append(anchor.astype(np.int16))
    return np.stack(plans), independent


def terms(gain, risk, signed, pair, scale):
    return np.concatenate((gain, risk, signed[:, None], pair[:, None]), axis=1) / scale


def score(pred, price, pair_weight):
    return pred[..., :2].min(-1) - price * pred[..., 2:5].clip(0.).sum(-1) + pair_weight * pred[..., 5]


def select(pred, counts, price, pair_weight, margin, match=None):
    values = score(pred - pred[:1], price, pair_weight)
    values[1:] -= margin
    if match is not None:
        values = np.where(counts == match, values, -np.inf)
    i = int(values.argmax())
    if not np.isfinite(values[i]):
        raise ValueError('No exact-count query option')
    return i


def loss(net, x, y, geometry):
    if not geometry:
        x = torch.cat((x[..., :-PAIR_FEATURES], torch.zeros_like(x[..., -PAIR_FEATURES:])), dim=-1)
    p = net(x)
    diff = p[:, 1] - p[:, 0]
    delta = y[:, 1] - y[:, 0]
    return F.smooth_l1_loss(p, y) + .5 * F.smooth_l1_loss(diff, delta)
