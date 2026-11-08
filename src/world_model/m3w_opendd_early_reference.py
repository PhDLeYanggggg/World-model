"""Release a support edit from local reference evidence without moving gain."""

import numpy as np
from torch import nn
from torch.nn import functional as F


def choose(packets, threshold, mode):
    """Stream the frozen gain and baseline support rules plus one early admission."""
    gain_row = support = matched = None
    for rows, gain, easy, radius, logits, mass in packets:
        if gain_row is not None and support is not None:
            break
        was_open = gain_row is None
        if was_open and len(rows) and gain.max() > 0.:
            gain_row = int(rows[gain.argmax()])
        if support is not None:
            continue
        available = rows != gain_row
        old = available & (easy >= threshold)
        if old.any():
            support = matched = int(rows[np.where(old, easy, -np.inf).argmax()])
            continue
        legal = available & (radius > 0.) & (logits > 0.)
        if not was_open or mode == "original" or not legal.any():
            continue
        # Probability-only and expected safe-reference mass share an admission rule.
        p = np.exp(-np.logaddexp(0., -logits))
        rank = p if mode == "incidence" else p*np.expm1(mass.clip(0., 6.))
        i = int(np.where(legal, rank, -np.inf).argmax())
        support = int(rows[i])
        matched = int(rows[np.where(available & (radius >= radius[i]), easy, -np.inf).argmax()])
    return gain_row, support, matched


def network(width):
    return nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 24),
                         nn.SiLU(), nn.Linear(24, 2))


def loss(net, x, y, with_mass):
    z = net(x)
    gate = F.binary_cross_entropy_with_logits(z[:, 0], y[:, 0])
    if not with_mass:
        return gate
    keep = y[:, 0]
    return gate + .25*(F.smooth_l1_loss(z[:, 1], y[:, 1], reduction="none")*keep).sum()/keep.sum().clamp(min=1.)
