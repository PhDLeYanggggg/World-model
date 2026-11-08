"""Shared expected-utility head for actions with identical intervention radii."""

import numpy as np
import torch
from torch import nn


def features(history, cv, bank):
    """Causal, heading-relative features; rows with zero budget are not scored."""
    h = np.asarray(history, dtype=np.float64)
    if h.shape[1:] != (8, 2) or cv.shape != (len(h), 12, 2) or bank.shape[0] != len(h) or bank.shape[2:] != (12, 2):
        raise ValueError("Expected eight past and twelve predicted steps")
    v = np.diff(h, axis=1)
    norm = np.linalg.norm(v, axis=-1)
    scale = np.sqrt(np.square(norm).mean(1))
    if not (scale > 0.).all():
        raise ValueError("Edited recipients must have nonstationary histories")
    axis = v[np.arange(len(h)), 6-np.argmax((norm > 1e-8*scale[:, None])[:, ::-1], axis=1)]
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    rotation = np.stack((axis, np.stack((-axis[:, 1], axis[:, 0]), axis=1)), axis=2)
    delta = bank - cv[:, None]
    r = np.linalg.norm(delta, axis=-1).max(2)
    if not (r > 0.).all() or not np.allclose(r, r[:, :1], atol=1e-13, rtol=1e-7):
        raise ValueError("Path choice requires a common positive radius")
    past = ((h-h[:, -1:]) @ rotation)/scale[:, None, None]
    base = (cv @ rotation)/scale[:, None, None]
    common = np.concatenate((past[:, :-1].reshape(len(h), 14), base[:, [-1]].reshape(len(h), 2),
                             np.log1p(r[:, :1]/scale[:, None])), axis=1)
    unit = (delta @ rotation[:, None])/r[:, :, None, None]
    x = np.concatenate((np.broadcast_to(common[:, None], (*r.shape, common.shape[-1])),
                        unit.reshape(*r.shape, 24)), axis=2)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite causal path descriptor")
    return x.astype(np.float32), r


def targets(cv, bank, radius, future, valid, easy_cut):
    """Signed full-grid lower utility and observed easy harm, both per radius."""
    y = np.where(valid[..., None], future, 0.)
    a = np.linalg.norm(cv-y, axis=-1)
    b = np.linalg.norm(bank-y[:, None], axis=-1)
    gain = a[:, None]-b
    debit = np.linalg.norm(bank-cv[:, None], axis=-1)
    lower = np.where(valid[:, None], gain, -debit).mean(2)/radius
    mass = valid.sum(1).clip(1)
    ref_ade = np.where(valid, a, 0.).sum(1)/mass
    easy = valid.any(1) & (ref_ade > 0.) & (ref_ade <= easy_cut)
    harm = (np.maximum(0., np.where(valid[:, None], -gain, 0.).sum(2)/mass[:, None])
            * easy[:, None]/radius)
    if not np.isfinite(lower).all() or np.max(np.abs(lower)) > 1.+1e-8 or np.max(harm) > 1.+1e-8:
        raise ValueError("Triangle-inequality bound violated")
    return np.stack((lower, harm), axis=-1).astype(np.float32)


class ActionValue(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Sequential(nn.Linear(41, 48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU(), nn.Linear(32, 2))

    def forward(self, x):
        v = self.value(x)
        return torch.stack((v[..., 0].tanh(), v[..., 1].sigmoid()), dim=-1)


def loss(pred, target):
    delta = lambda a: a[:, :, None]-a[:, None, :]
    g, h = pred.unbind(-1)
    y, harm = target.unbind(-1)
    return ((g-y).square().mean() + .5*(delta(g)-delta(y)).square().mean()
            + (h-harm).square().mean() + .5*(delta(g-h)-delta(y-harm)).square().mean())


def select(estimate, fallback_index):
    gain, harm = np.moveaxis(estimate, -1, 0)
    safe_value = gain-harm
    utility = gain.argmax(1)
    guarded = safe_value.argmax(1)
    # Keep the causal anchor when the gains disagree or tie.
    at = np.arange(len(guarded))
    agree = np.where((utility == guarded) & (safe_value[at, guarded] > safe_value[:, fallback_index]),
                     guarded, fallback_index)
    return {"utility": utility, "easy_penalty": guarded, "agreement": agree}
