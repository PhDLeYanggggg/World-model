"""Bounded harm moments and chronological matched-budget decisions."""

import numpy as np
import torch
from torch import nn

from src.world_model.m3w_reference_bridge_policy import WIDTH

CONTEXT = 20
BUDGET = .02


def target(cv, paths, y, valid, cut):
    """Observed harm, observed easy harm, signed easy change and reference masses."""
    n = valid.sum(1)
    known = n > 0
    safe = np.where(valid[..., None], y, 0.)
    c = np.where(valid, np.linalg.norm(cv-safe, axis=-1), 0.).sum(1) / n.clip(1)
    p = np.where(valid[:, None], np.linalg.norm(paths-safe[:, None], axis=-1), 0.).sum(-1) / n.clip(1)[:, None]
    signed = p-c[:, None]
    harm = np.maximum(signed, 0.)
    easy = known & (c > 0.) & (c <= cut)
    radius = np.linalg.norm(paths-cv[:, None], axis=-1).max(-1)
    def tile(v):
        return np.broadcast_to(v[:, None], radius.shape)
    return np.stack((harm, harm*easy[:, None], signed*easy[:, None], tile(c), tile(c*easy),
                     radius*~known[:, None]), axis=-1)


class MomentHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.context = nn.Sequential(nn.Linear(CONTEXT, 32), nn.SiLU(), nn.Linear(32, 3))
        self.action = nn.Sequential(nn.Linear(WIDTH, 48), nn.SiLU(), nn.Linear(48, 24),
                                    nn.SiLU(), nn.Linear(24, 3))

    def forward(self, x, radius):
        a = self.context(x[..., :CONTEXT])
        unknown = torch.sigmoid(a[..., 0])
        base = (1.-unknown)*nn.functional.softplus(a[..., 1])
        easy = base*torch.sigmoid(a[..., 2])
        rate, share, improvement = torch.sigmoid(self.action(x)).unbind(-1)
        observed = (1.-unknown)*radius
        harm = observed*rate
        easy_harm = harm*share
        easy_signed = easy_harm - (observed-harm)*improvement
        return torch.stack((harm, easy_harm, easy_signed, base, easy, radius*unknown), dim=-1)


def excess(z, easy_reference):
    """Positive violations of the three original risks under estimated moments."""
    h, he, signed, base, easy, unknown = np.moveaxis(z, -1, 0)
    n = np.stack((h+unknown, he+unknown, signed), axis=-1)
    d = np.stack((base, easy, np.broadcast_to(easy_reference, base.shape)), axis=-1)
    return n-BUDGET*d, d


def choose(value, slate_moments, easy, ledger, past_easy, mode):
    if mode == "soft":
        slack, _ = excess(slate_moments, easy)
        return int(np.argmax(value-np.maximum(slack, 0.).sum(-1)))
    if mode not in ("local", "prefix"):
        raise ValueError("Unknown risk mode")
    moments = slate_moments+ledger if mode == "prefix" else slate_moments
    full_easy = easy+past_easy if mode == "prefix" else easy
    slack, denom = excess(moments, full_easy)
    loss = np.maximum(slack, 0.) / denom.clip(1e-8)
    # Compare worst estimated violation, then total violation, then the shared utility.
    return int(np.lexsort((-value, loss.sum(-1), loss.max(-1)))[0])
