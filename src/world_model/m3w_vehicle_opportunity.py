"""Recording-level prediction of an opportunity in a finite angle menu."""

import numpy as np
import torch
from torch import nn

from src.world_model import m3w_vehicle_action_value as action


WIDTH = 3*action.WIDTH+2


def summarize(actions, n_agents):
    if actions.ndim != 3 or actions.shape[0] != 2 or actions.shape[-1] != action.WIDTH or n_agents < 2:
        raise ValueError('Require both original recipients and their legal angles')
    x = actions.reshape(-1, action.WIDTH)
    return np.r_[x.mean(0), x.min(0), x.max(0), np.log1p(n_agents), np.log1p(len(x))].astype(np.float32)


def head():
    return nn.Sequential(nn.Linear(WIDTH, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1))


def loss(net, batch):
    x, y = batch
    return nn.functional.binary_cross_entropy_with_logits(net(x)[:, 0], y)


def nominate(value):
    score, _ = action.merit(value)
    i, j = np.unravel_index(score.argmax(), score.shape)
    return (int(i), int(j)+1), float(score[i, j])


def top_mask(score, n):
    if not 0 <= n <= len(score) or not np.isfinite(score).all():
        raise ValueError('Invalid equal-budget diagnostic')
    out = np.zeros(len(score), dtype=bool)
    out[np.argsort(-score, kind='stable')[:n]] = True
    return out
