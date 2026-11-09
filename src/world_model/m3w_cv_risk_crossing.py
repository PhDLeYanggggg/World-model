"""Exact matched substitutions around a verified recording witness."""
import numpy as np
import torch
from torch.nn import functional as F


GAIN_MARGIN = 1e-6


def to_current(actions, old_fractions, amplitudes, rays):
    out = np.full(len(actions), -1, np.int16)
    at = actions >= 0
    fraction, ray = np.divmod(actions[at], rays)
    a = np.array([amplitudes.index(f) for f in old_fractions])
    out[at] = ray * len(amplitudes) + a[fraction]
    return out


def crossings(terms, required, anchor, left, left_arm, right, right_arm):
    """True risk-only crossings, with exact utility preference on the same pairs."""
    at = np.flatnonzero(anchor >= 0)
    total = terms[at, anchor[at]].sum(0)
    after = total - terms[left, left_arm] + terms[right, right_arm]
    support = after[:, 5] >= required[5]
    risk = after[:, 2:5] < required[2:5] - 1e-8
    keep = (after[:, :2] > required[:2] + 1e-9).all(1) & support & risk.any(1)
    a = (total[:2] - required[:2]).min()
    b = (after[:, :2] - required[:2]).min(1)
    utility = np.where(np.abs(a - b) > 1e-9, np.sign(a - b), 0).astype(np.int8)
    return keep, utility, risk


def objective(net, pair, utility, risk):
    z = net(pair).squeeze(-1)
    diff = z[:, 0] - z[:, 1]
    label = torch.ones_like(diff) if risk else utility.to(diff.dtype)
    loss = F.softplus(-label * diff)
    return (loss * (label != 0.)).sum() / (label != 0.).sum().clamp_min(1)


def adjusted_gain(gain, logits, scale, strength):
    return gain + strength * scale[:, None, None] * np.tanh(logits[..., None])
