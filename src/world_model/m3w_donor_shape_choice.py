"""Learn a paired shape choice at fixed intervention slots."""

import numpy as np
import torch
from torch import nn

from src.world_model.m3w_donor_motion_shape import motion_frame


MARGINS = (0., .01, .025, .05, .10, .20)
PENALTIES = (1., 3.)


def features(history, cv, anchor, donor, radius_fraction, distance, prefix_pick):
    v, rot, scale = motion_frame(history)
    local = lambda p: ((p-cv) @ rot).reshape(len(cv), -1)/scale[:, None]
    return np.c_[v, local(anchor), local(donor), np.log1p(scale),
                 radius_fraction, distance, prefix_pick].clip(-32., 32.).astype(np.float32)


def network(width):
    return nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU(), nn.Linear(32, 4))


def loss(net, x, target):
    return nn.functional.smooth_l1_loss(net(x), target, beta=.05)


def use_donor(pred, rule):
    if rule is None:
        return np.zeros(len(pred), bool)
    margin, penalty = rule
    return pred[:, 0] - penalty*np.maximum(pred[:, 1:], 0.).mean(1) > margin


def calibrate(cells, guarded):
    """Select on out-of-fold donors only; the zero-action arm is always present."""
    rules = [None] + [(m, w) for w in (PENALTIES if guarded else (0.,)) for m in MARGINS]
    table = []
    for rule in rules:
        by_site = []
        for folds in cells.values():
            summaries = []
            for pred, delta, mass in folds:
                take = use_donor(pred, rule)
                summaries.append(np.r_[100.*delta[take].sum(0)/mass, take.mean()])
            by_site.append(np.mean(summaries, axis=0))
        avg = np.mean(by_site, axis=0)
        admissible = not guarded or np.all(avg[1:4] <= 1e-12)
        table.append(dict(rule=rule, gain=float(avg[0]), risk_delta=avg[1:4].tolist(),
                          choice_rate=float(avg[4]), admissible=bool(admissible)))
    best = max((v for v in table if v["admissible"]), key=lambda v: (v["gain"], -v["choice_rate"]))
    return best["rule"], table
