"""Causal shape features and matched-action gain and harm targets."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def features(history, cv, ols, bank, radius):
    h = np.asarray(history, float)
    v = np.diff(h, axis=1)
    speed = np.linalg.norm(v, axis=2)
    scale = np.maximum(np.sqrt(np.square(speed).mean(1)), .05)
    d = bank-cv[:, None]
    unit = d/np.where(radius > 0, radius, 1.)[:, None, None, None]
    heading = v[:, -3:].mean(1)
    heading /= np.linalg.norm(heading, axis=1).clip(1e-8)[:, None]
    across = np.c_[-heading[:, 1], heading[:, 0]]
    u = unit[:, :, [1, 3, 7, 11]]
    magnitude = np.linalg.norm(u, axis=-1)
    along = (u*heading[:, None, None]).sum(-1)
    lateral = (u*across[:, None, None]).sum(-1)
    speed_pattern = np.c_[speed/scale[:, None], np.linalg.norm(np.diff(v, axis=1), axis=2)/scale[:, None]]
    x = np.c_[np.log1p(scale), speed_pattern, np.log1p(radius/scale)]
    motion = np.broadcast_to(x[:, None], (*d.shape[:2], x.shape[1]))
    geom = np.linalg.norm(bank[:, :, [1, 3, 7, 11]]-ols[:, None, [1, 3, 7, 11]], axis=-1)/scale[:, None, None]
    arm = np.broadcast_to(np.eye(bank.shape[1])[None], (*bank.shape[:2], bank.shape[1]))
    replay = []
    for prefix in (4, 5, 6):
        target = h[:, prefix:]-h[:, prefix-1:prefix]
        k = target.shape[1]
        local_cv = v[:, prefix-2, None]*np.arange(1, k+1)[None, :, None]
        pseudo = local_cv[:, None]+d[:, :, :k]
        error = np.linalg.norm(local_cv-target, axis=-1)[:, None]-np.linalg.norm(pseudo-target[:, None], axis=-1)
        replay.append(error.sum(2)/(k*np.where(radius > 0., radius, 1.))[:, None])
    probes = np.stack(replay, axis=-1).clip(-1., 1.)
    x = np.concatenate((motion, magnitude, along, lateral, np.log1p(geom), arm, probes), axis=-1)
    return x.clip(-12., 12.).astype(np.float32), probes.mean(-1)


def targets(cv, bank, future, valid, radius, easy_cut):
    clean = np.where(valid[..., None], future, 0.)
    e = np.linalg.norm(bank-clean[:, None], axis=-1)
    ref = np.linalg.norm(cv-clean, axis=-1)
    gain = np.where(valid[:, None], ref[:, None]-e, 0.)
    debit = np.where(valid[:, None], 0., np.linalg.norm(bank-cv[:, None], axis=-1))
    n = valid.sum(1).clip(1)
    harm = np.maximum(-gain.sum(2)/n[:, None], 0.)
    ref_mean = np.where(valid, ref, 0.).sum(1)/n
    easy = (ref_mean > 0.) & (ref_mean <= easy_cut)
    scale = np.where(radius > 0., radius, 1.)[:, None]
    return np.stack(((gain-debit).mean(2)/scale, harm/scale, easy[:, None]*harm/scale),
                    axis=-1).clip(-1., 1.).astype(np.float32)


def network(width):
    return nn.Sequential(nn.Linear(width, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 3))


def loss(net, x, target):
    return F.smooth_l1_loss(net(x), target, beta=.1)


def choose(bank, score):
    return bank[np.arange(len(bank)), np.argmax(score, axis=1)]


def fit_roster(held, recordings):
    site, day = held.split("-")
    return tuple(r for r in recordings if r.split("-")[0] != site and r.split("-")[1] != day)
