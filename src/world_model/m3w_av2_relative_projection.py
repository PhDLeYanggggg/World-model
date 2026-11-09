"""Bounded direction refinement with a matched relative-trajectory objective."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.world_model import m3w_av2_history_replay as history
from src.world_model import m3w_av2_projected_reference as projected
from src.world_model.m3w_av2_steered_reference import rotate


MAX_ANGLE = np.pi/4.
PAIR_WEIGHT = .75


def inputs(d, raw, recipients):
    ix = np.asarray(recipients)
    x, c, _, delta, frame, scale = projected.inputs(d, raw, ix)
    h = d["h"][ix]
    motion = h[::-1]-h
    local = lambda z: ((z@frame)/scale[:, None, None]).reshape(2, -1)
    return np.column_stack((x, x[::-1], local(motion), local(delta))).clip(-32., 32.).astype(np.float32), c, delta, scale


class Refine(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, 3))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        z = self.head(x)
        a = F.interpolate(z.reshape(-1, 1, 3), size=12, mode="linear", align_corners=True)
        return MAX_ANGLE*a.reshape(*z.shape[:-1], 12).tanh()


def labels(d, raw, ix, y, valid):
    i, j = np.triu_indices(len(raw), 1)
    effect = (i[:, None] == ix).astype(float)-(j[:, None] == ix).astype(float)
    use = np.any(effect, axis=1)
    safe = np.where(valid[..., None], y, 0.)
    return (safe[ix], valid[ix], raw[i[use]]-raw[j[use]], safe[i[use]]-safe[j[use]],
            valid[i[use]] & valid[j[use]], effect[use], len(i)*raw.shape[1])


def losses(angles, cv, delta, scale, y, valid, pair_raw, pair_y, pair_valid, effect, pair_size):
    p = cv+rotate(delta, angles, torch)
    raw = cv+delta
    norm = lambda a: torch.linalg.vector_norm(a, dim=-1)
    individual = torch.where(valid, norm(p-y), norm(p-raw)).sum((1, 2))/(valid.shape[1]*valid.shape[2])
    rel_delta = torch.einsum("bei,bitd->betd", effect, p-raw)
    rel = torch.where(pair_valid, norm(pair_raw+rel_delta-pair_y), norm(rel_delta))
    pair = rel.sum((1, 2))/pair_size
    unit = scale.mean(1)
    return (individual/unit).mean(), (pair/unit).mean()


def confidence(d, raw, path):
    """Score prospective change by replaying its fixed shapes on known prefixes."""
    cv, h = d["cv"], d["h"]
    final = h[:, -1]-h[:, -2]
    i, j = np.triu_indices(len(h), 1)
    vals = []
    for k in history.PREFIXES:
        n = len(h[0])-k
        c, _, _, scale = history.motion.rollout(h[:, :k], n)
        past = h[:, k:]-h[:, k-1:k]
        v = h[:, k-1]-h[:, k-2]
        a, b = ((c-past)+history.align((p-cv)[:, :n], final, v) for p in (raw, path))
        g = np.linalg.norm(a, axis=-1).mean()-np.linalg.norm(b, axis=-1).mean()
        q = np.linalg.norm(a[i]-a[j], axis=-1).mean()-np.linalg.norm(b[i]-b[j], axis=-1).mean()
        vals.append(min(g, q)/scale.mean())
    return float(np.exp(np.clip(np.median(vals), -25., 25.)))
