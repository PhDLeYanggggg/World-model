"""Learn an offset shape while keeping the incumbent CV departure fixed."""
import numpy as np
import torch

from src.world_model.m3w_opendd_velocity_candidate import decode

ARMS = ('observed', 'completion', 'easy_guard', 'peer_guard')


def shape(raw, incumbent_unit):
    # Per-step perturbation is bounded by sqrt(2)/2, so the max norm cannot vanish.
    proposal = incumbent_unit + (.1/12)*decode(raw)
    radius = torch.linalg.vector_norm(proposal, dim=-1).amax(1)
    return proposal/radius[:, None, None]


def neighbors(history, offsets):
    """Use only the last observed separation to order peer histories."""
    context = np.full((len(history), 2, 2, 8, 2), np.nan)
    for lo, hi in zip(offsets[:-1], offsets[1:], strict=True):
        h = history[lo:hi]
        d = np.linalg.norm(h[:, None, -1]-h[None, :, -1], axis=-1)
        np.fill_diagonal(d, np.inf)
        n = min(2, hi-lo-1)
        if n > 0:
            ids = np.argsort(d, axis=1, kind='stable')[:, :n]
            context[lo:hi, 0, :n] = h[ids]
    return context


def at_radius(offset, budget):
    r = np.linalg.norm(offset, axis=-1).max(1)
    if np.any((r <= 0) & (budget > 0)):
        raise ValueError('No direction for a positive intervention budget')
    scale = np.divide(budget, r, out=np.zeros_like(budget), where=r > 0)
    return offset*scale[:, None, None]
