"""Past-only neighbor availability and factored reference eligibility."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.world_model.m3w_av2_paired_qualification import POS_WEIGHT, binary

CONTEXT_WIDTH = 16
MOTION_WIDTH = 6
MODES = ("motion_direct", "motion_factored", "context_direct", "context_factored")


def past_context(rows, ids, history, steps):
    """The supplied observed states can include tracks outside the forecast roster."""
    lookup = {t: i for i, t in enumerate(steps)}
    tracks = {}
    for row in rows:
        k, t = row["track_id"], row["timestep"]
        if not row["observed"] or t not in lookup:
            raise ValueError("Out-of-window state in observability inputs")
        track = tracks.setdefault(k, {})
        p = np.array([row["position_x"], row["position_y"]], dtype=float)
        if t in track or not np.isfinite(p).all():
            raise ValueError("Duplicate or invalid observed state")
        track[t] = p
    key = list(tracks)
    pos = np.zeros((len(key), len(steps), 2))
    valid = np.zeros(pos.shape[:2], dtype=bool)
    for i, k in enumerate(key):
        for t, p in tracks[k].items():
            pos[i, lookup[t]], valid[i, lookup[t]] = p, True
    index = np.array([key.index(k) for k in ids], dtype=int)
    if not valid[index].all() or not np.array_equal(pos[index], history):
        raise ValueError("Observed context does not reproduce the frozen roster")
    partial = ~valid.all(1)
    delta = np.diff(history, axis=1)
    speed = np.linalg.norm(delta, axis=-1)
    unit = max(float(np.median(speed)), .05)
    direction = np.linalg.norm(delta.sum(1), axis=-1)/speed.sum(1).clip(1e-8)
    motion = np.column_stack((np.log1p(speed.mean(1)/unit), np.log1p(speed[:, -2:].mean(1)/unit),
        np.log1p(speed.std(1)/unit), direction, np.full(len(ids), np.log1p(unit)), np.full(len(ids), np.log1p(len(ids)))))
    context = []
    for i, k in enumerate(index):
        neighbor = valid.copy()
        neighbor[k] = False
        distance = np.linalg.norm(pos-history[i], axis=-1)/unit
        near = neighbor & (distance <= 20.)
        short = near & partial[:, None]
        count, small = near.sum(0), short.sum(0)
        nearest = np.where(neighbor, distance, 80.).min(0).clip(0., 80.)
        enter = (short[:, 1:] & ~short[:, :-1]).sum()
        leave = (short[:, :-1] & ~short[:, 1:]).sum()
        context.append(np.log1p([count.mean(), count[0], count[-1], count.std(),
            small.mean(), small[0], small[-1], small.std(), enter, leave,
            nearest.mean(), nearest[-1], valid[:, 0].sum(), valid[:, -1].sum(),
            partial.sum(), (partial & valid[:, -1]).sum()]))
    return motion.astype(np.float32), np.asarray(context, dtype=np.float32), int(partial.sum())


def labels(cv, target, valid, edited, cutoff):
    error = np.where(valid, np.linalg.norm(cv-target, axis=-1), 0.).sum(1)/valid.sum(1).clip(1)
    known = valid.any(1)
    easy = known & (error > 0.) & (error <= cutoff)
    return np.column_stack((valid.mean(1), known, easy))[edited].astype(np.float32), bool(easy[edited].any())


class Gate(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(MOTION_WIDTH+CONTEXT_WIDTH, 32), nn.SiLU(), nn.Linear(32, 16), nn.SiLU())
        self.aux = nn.Linear(16, 2)
        self.query = nn.Sequential(nn.Linear(width+32, 48), nn.SiLU(), nn.Linear(48, 3))

    def forward(self, x, actors):
        h = self.actor(actors)
        return self.query(torch.cat((x, h.mean(1), h.amax(1)), dim=-1)), self.aux(h)


def gate_loss(z, aux, events, actor_targets):
    eligible, support = events.unbind(-1)
    coverage, known, easy = actor_targets.unbind(-1)
    cond = F.binary_cross_entropy_with_logits(z[:, 2], eligible,
        pos_weight=z.new_tensor(POS_WEIGHT), reduction="none")
    aux_easy = F.binary_cross_entropy_with_logits(aux[..., 1], easy, reduction="none")
    return (binary(z[:, 0], eligible) + F.binary_cross_entropy_with_logits(z[:, 1], support)
        + (cond*support).sum()/support.sum().clamp_min(1)
        + .5*F.mse_loss(aux[..., 0].sigmoid(), coverage) + .5*(aux_easy*known).sum()/known.sum().clamp_min(1))


def scores(z):
    correction = z.new_tensor(POS_WEIGHT).log()
    return (z[..., 0]-correction).sigmoid(), z[..., 1].sigmoid()*(z[..., 2]-correction).sigmoid()
