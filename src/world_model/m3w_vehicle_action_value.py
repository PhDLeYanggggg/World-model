"""Learn the value of changing one direction relative to its exact incumbent."""

import numpy as np
import torch
from torch import nn

from src.world_model.m3w_vehicle_angular_pair import candidates


OWN = 30
WIDTH = 50


def design(cv, anchor, selected, context, rotation, scale, offsets):
    """Return angle inputs and paths; no target or validity input."""
    selected = np.asarray(selected)
    bank = candidates(cv[selected], anchor[selected])[:, 1:]
    other = selected[::-1]
    rot = rotation[selected]
    unit = scale[selected, None, None]
    h = (anchor[selected]-cv[selected]) @ rot / unit
    v = (bank-cv[selected, None]) @ rot[:, None] / unit[:, None]
    p = (anchor[other]-cv[other]) @ rot / unit
    xy = ((offsets[other]-offsets[selected])[:, None] @ rot)[:, 0] / (8.*unit[:, 0])
    ix = (3, 7, 11)
    a, b = bank.shape[:2]
    features = np.concatenate((
        np.broadcast_to(context[selected, None, :17], (a, b, 17)),
        np.broadcast_to(h[:, None, ix].reshape(a, 1, 6), (a, b, 6)),
        v[:, :, ix].reshape(a, b, 6),
        np.log1p(np.linalg.norm(bank-anchor[selected, None], axis=-1).sum(-1))[..., None],
        np.broadcast_to(context[selected, None, 17:], (a, b, 12)),
        np.broadcast_to(p[:, None, ix].reshape(a, 1, 6), (a, b, 6)),
        np.broadcast_to(xy[:, None], (a, b, 2)),
    ), axis=-1)
    return features.clip(-16., 16.).astype(np.float32), bank


def gain_harm(before, after, target, valid):
    shift = np.linalg.norm(after-before, axis=-1)
    # Invalid targets must not affect observed contributions.
    y = np.where(valid[..., None], target, 0.)
    delta = np.linalg.norm(before-y, axis=-1)-np.linalg.norm(after-y, axis=-1)
    plus = np.where(valid, np.maximum(delta, 0.), 0.)
    harm = np.where(valid, np.maximum(-delta, 0.), shift)
    return plus, harm, shift


def targets(anchor, selected, bank, y, valid):
    """Positive gain, harm including unknown steps, and incident-edge lower gain."""
    out = []
    for i, a in enumerate(selected):
        p = bank[i]
        pos, harm, shift = gain_harm(anchor[a], p, y[a], valid[a])
        den = shift.sum(1).clip(1e-9)
        others = np.flatnonzero(np.arange(len(anchor)) != a)
        before = anchor[a]-anchor[others]
        after = p[:, None]-anchor[others]
        both = valid[a] & valid[others]
        add, sub, _ = gain_harm(before, after, y[a]-y[others], both)
        relative = (add-sub).sum((1, 2))/(den*len(others))
        out.append(np.column_stack((pos.sum(1)/den, harm.sum(1)/den, relative)))
    return np.asarray(out, dtype=np.float32).clip(-1., 1.)


class AngleValue(nn.Module):
    def __init__(self, with_context):
        super().__init__()
        self.with_context = with_context
        self.body = nn.Sequential(nn.Linear(WIDTH, 64), nn.SiLU(), nn.Linear(64, 48), nn.SiLU(), nn.Linear(48, 3))

    def forward(self, x):
        if not self.with_context:
            x = torch.cat((x[..., :OWN], torch.zeros_like(x[..., OWN:])), dim=-1)
        a = self.body(x)
        return torch.cat((a[..., :2].sigmoid(), a[..., 2:].tanh()), dim=-1)


def loss(net, batch):
    x, y = batch
    return nn.functional.smooth_l1_loss(net(x), y, beta=.15)


def merit(value, joint=False):
    gain = value[..., 0]-value[..., 1]
    relative = value[..., 2]
    rank = gain-.25*value[..., 1]+(.5*relative if joint else 0.)
    eligible = (gain > 0.) & (rank > 0.)
    return rank, eligible & ((relative > 0.) if joint else True)


def choose(value, joint=False, same_actor=None):
    score, eligible = merit(value, joint)
    if same_actor is not None:
        angle = int(score[same_actor].argmax())
        return (same_actor, angle+1), float(score[same_actor, angle])
    if not eligible.any():
        return (-1, -1), 0.
    row, angle = np.unravel_index(np.where(eligible, score, -np.inf).argmax(), score.shape)
    return (int(row), int(angle)+1), float(score[row, angle])
