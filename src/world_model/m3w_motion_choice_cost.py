"""Fixed-displacement path choice with matched order and cost targets."""

import numpy as np
import torch
from torch import nn

from src.world_model import m3w_prefix_expert_mixture as motion


ARMS = ('rank', 'pooled_cost', 'lead_cost')
EXPERTS = (0, 2, 3, 4, 5)
RADIUS_FRACTION = .5


def candidates(history):
    base = motion.bank(history)
    ols, cv = base[:, 1], base[:, 0]
    edit = base[:, EXPERTS] - ols[:, None]
    peak = np.linalg.norm(edit, axis=-1).max(-1)
    radius = RADIUS_FRACTION * np.sqrt(np.square(np.diff(history, axis=1)).sum(-1).mean(1))
    legal = (radius[:, None] > 1e-8) & (peak >= radius[:, None])
    edit *= np.divide(radius[:, None], peak, out=np.zeros_like(peak), where=legal)[..., None, None]
    return ols, cv, ols[:, None] + edit, legal, radius


def inputs(history, ols, bank, radius):
    x, _ = motion.features(history, 'prefix')
    axis = ols[:, 0]
    norm = np.linalg.norm(axis, axis=1)
    axis = np.divide(axis, norm[:, None], out=np.zeros_like(axis), where=norm[:, None] > 1e-12)
    axis[norm <= 1e-12] = (1, 0)
    edits = (bank - ols[:, None]) / radius.clip(1e-8)[:, None, None, None]
    along = np.einsum('nktd,nd->nkt', edits, axis)
    across = np.einsum('nktd,nd->nkt', edits, np.column_stack((-axis[:, 1], axis[:, 0])))
    return np.concatenate((x, along.reshape(len(x), -1), across.reshape(len(x), -1)), axis=1).astype(np.float32)


def targets(ols, bank, radius, future, valid):
    """Signed OLS error saved, per observed lead; never passed to inference."""
    safe = np.where(valid[..., None], future, 0.)
    delta = np.linalg.norm(ols - safe, axis=-1)[:, None] - np.linalg.norm(bank - safe[:, None], axis=-1)
    return np.where(valid[:, None], delta / radius.clip(1e-8)[:, None, None], 0.).astype(np.float32)


class Head(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, len(EXPERTS) * 6))

    def forward(self, x):
        return self.net(x).reshape(len(x), len(EXPERTS), 3, 2)


def costs(output):
    return output.sigmoid()


def scores(output, arm):
    if arm == 'rank':
        return 8 * (output[..., 0] - output[..., 1]).mean(-1)
    gh = costs(output)
    return (gh[..., 0] - gh[..., 1]).mean(-1)


def loss(net, arm, x, legal, delta, valid):
    pred = net(x)
    if arm == 'rank':
        gain = delta.sum(-1).masked_fill(~legal, -torch.inf)
        winners = gain >= (gain.amax(-1, keepdim=True) - 1e-6)
        logp = scores(pred, arm).masked_fill(~legal, -1e6).log_softmax(1)
        return -((logp * winners).sum(1) / winners.sum(1)).mean()
    gh = torch.stack((delta.clamp_min(0.), (-delta).clamp_min(0.)), dim=-1)
    gh = gh.reshape(len(x), len(EXPERTS), 3, 4, 2)
    support = valid.reshape(len(x), 3, 4).sum(-1)
    if arm == 'lead_cost':
        target = gh.sum(-2) / support.clamp_min(1)[:, None, :, None]
    elif arm == 'pooled_cost':
        target = gh.sum((2, 3))[:, :, None] / support.sum(1)[:, None, None, None]
    else:
        raise ValueError(arm)
    weights = (legal[:, :, None] * support[:, None]).unsqueeze(-1)
    return ((costs(pred) - target).square() * weights).sum() / (2 * weights.sum())


def choose(score, legal):
    if not np.isfinite(score).all() or score.shape != legal.shape:
        raise ValueError('Require finite, aligned action scores')
    return np.where(legal.sum(1) >= 2, np.where(legal, score, -np.inf).argmax(1), -1)


def apply(ols, bank, legal, radius, action):
    changed = legal.sum(1) >= 2
    if (not np.array_equal(action >= 0, changed) or np.any(action[~changed] != -1)
            or np.any(action[changed] >= legal.shape[1]) or not legal[changed, action[changed]].all()):
        raise ValueError('Action must match fixed recipients and legal edit menu')
    path = ols.copy()
    path[changed] = bank[changed, action[changed]]
    np.testing.assert_allclose(np.linalg.norm(path[changed] - ols[changed], axis=-1).max(1),
                               radius[changed], rtol=1e-8, atol=1e-10)
    return path
