"""Two held-location shape selectors: outcome value or a latent motion proxy."""

import numpy as np
import torch
from torch import nn

from src.evaluation.m3w_opendd_location import paired
from src.evaluation.m3w_recording_switch_ceiling import moments
from src.world_model import m3w_radius_matched_turn_menu as menu
from src.world_model.m3w_opendd_velocity_candidate import frames


def features(h, amp):
    cv, ols, bank = menu.make_menu(h, 12, amp)
    rot, step = frames(h)
    scale = 12.*step
    hist = ((h-h[:, -1:])@rot)/step[:, None, None]
    relative = (bank-cv[:, None])@rot[:, None]/scale[:, None, None, None]
    probe = menu.prefix_rank(h, amp, amp > 0.).transpose(1, 0, 2)
    geometry = relative[:, :, [2, 5, 8, 11]].reshape(len(h), -1)
    x = np.c_[hist.reshape(len(h), -1), np.log1p(step), amp, geometry, probe.reshape(len(h), -1)]
    return x.clip(-16., 16.).astype(np.float32), cv, ols, bank, relative.astype(np.float32), rot, scale


def targets(cv, bank, y, valid, cut, rot, scale):
    value = []
    for path in bank.swapaxes(0, 1):
        m = moments(cv, path, y, valid, cut)
        value.append(np.column_stack((paired(bank[:, 0], path, y, valid)[1].mean(1), m["all_excess"],
                                      m["easy_excess"], m["easy_signed"])))
    risk = (np.stack(value, axis=1)/scale[:, None, None]).clip(-8., 8.).astype(np.float32)
    motion = (np.where(valid[..., None], y-cv, 0.)@rot)/scale[:, None, None]
    return risk, motion.clip(-8., 8.).astype(np.float32)


class Head(nn.Module):
    def __init__(self, width, kind):
        super().__init__()
        self.kind = kind
        self.net = nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 32), nn.SiLU(), nn.Linear(32, 24))

    def forward(self, x):
        shape = (len(menu.NAMES), 4) if self.kind == "value" else (12, 2)
        return self.net(x).reshape(len(x), *shape)


def objective(net, x, value, motion, valid):
    z = net(x)
    if net.kind == "value":
        t = nn.functional.smooth_l1_loss(z, value, beta=.06, reduction="none")
        return (t*z.new_tensor([1., 2., 4., 1.])).mean()
    t = nn.functional.smooth_l1_loss(z, motion, beta=.06, reduction="none").sum(-1)
    return ((t*valid).sum(1)/valid.sum(1).clamp_min(1)).mean()


def choose(value, future_proxy, geometry):
    g = value[..., 0].copy()
    g[:, 0] = 0.
    penalty = value[..., 1:].clip(min=0.)@np.array([2., 4., 1.])
    err = np.linalg.norm(geometry-future_proxy[:, None], axis=-1).mean(-1)
    return dict(learned_gain=g.argmax(1), learned_risk=(g-penalty).argmax(1), learned_dynamics=err.argmin(1))
