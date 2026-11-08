"""Generate bounded, same-radius support candidates and score their advantage."""

import numpy as np
from torch import nn
from torch.nn import functional as F


def match(cv, native, alternatives):
    """Keep the native support radius; never enlarge an alternative."""
    radius = np.linalg.norm(native-cv, axis=-1).max(1)
    bank, allowed = [native], [radius > 0.]
    for p in alternatives:
        delta = p-cv
        size = np.linalg.norm(delta, axis=-1).max(1)
        ok = (radius > 0.) & (size >= radius)
        out = cv.copy()
        out[ok] += delta[ok]*(radius[ok]/size[ok])[:, None, None]
        for previous, legal in zip(bank, allowed, strict=True):
            ok &= ~((np.abs(previous-out).max(axis=(1, 2)) <= 1e-12) & legal)
        bank.append(out)
        allowed.append(ok)
    return np.stack(bank), np.stack(allowed), radius


def first_release(rows, families, queries, scores, legal, cutoff):
    """Take the highest scoring admissible action at the first qualifying query."""
    use = np.flatnonzero(legal & (scores > cutoff))
    if not len(use):
        return None
    at = use[queries[use] == queries[use].min()]
    i = at[scores[at].argmax()]
    return int(rows[i]), int(families[i])


def network(width):
    return nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 24),
                         nn.SiLU(), nn.Linear(24, 2))


def loss(net, x, y):
    z = net(x)
    return F.smooth_l1_loss(z[:, 0], y[:, 0], beta=.1) + .2*F.binary_cross_entropy_with_logits(z[:, 1], y[:, 1])
