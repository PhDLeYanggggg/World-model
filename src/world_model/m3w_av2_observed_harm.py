"""Conditional positive-harm targets for source-excluded projected forecasts."""

import numpy as np
import torch
from torch import nn


def targets(cv, paths, opts, y, valid, cutoff):
    safe = np.where(valid[..., None], y, 0.)
    count = valid.sum(1).clip(1)
    base = np.where(valid, np.linalg.norm(cv-safe, axis=-1), 0.).sum(1)/count
    edit = np.where(valid[:, None], np.linalg.norm(paths-safe[:, None], axis=-1), 0.).sum(2)/count[:, None]
    harm = np.maximum(edit-base[:, None], 0.)
    easy = (base > 0.) & (base <= cutoff)
    i, j = opts.T
    all_mass = base[i]+base[j]
    easy_mass = easy[i]*base[i]+easy[j]*base[j]
    support = (all_mass > 0.) & (easy_mass > 0.)
    all_harm = harm[i, 0]+harm[j, 1]
    easy_harm = easy[i]*harm[i, 0]+easy[j]*harm[j, 1]
    numerator = np.column_stack((all_harm, easy_harm))
    mass = np.column_stack((all_mass, easy_mass))
    ratios = np.divide(numerator, mass, out=np.zeros_like(numerator), where=mass > 0.)
    fail = (ratios > .02).any(1)
    return support, fail, np.log1p(ratios/.02).clip(0., 8.).astype(np.float32)


def head(width):
    return nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, 3))


def loss(z, fail, severity):
    f = nn.functional
    classifier = f.binary_cross_entropy_with_logits(z[:, 0], fail.float())
    magnitude = f.smooth_l1_loss(f.softplus(z[:, 1:]), severity, beta=.25)
    return classifier+.5*magnitude, classifier, magnitude
