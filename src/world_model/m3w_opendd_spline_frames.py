"""Matched OLS, CV, and comoving-CV input frames for the causal spline."""
import numpy as np
import torch

from src.world_model.m3w_causal_velocity_spline import Spline, causal_inputs


MODES = ('ols_spline', 'cv_spline', 'comoving_spline')


def designs(history):
    h = np.asarray(history, dtype=float)
    # The matched EqMotion control is ego-only. No peer is exposed to any spline.
    groups = np.arange(len(h))[:, None]
    raw = causal_inputs(h, groups)
    v = (h[:, -1]-h[:, -2]).astype(np.float32)
    t = np.arange(-7., 1.)
    comoving = causal_inputs(h-(h[:, -1]-h[:, -2])[:, None]*t[None, :, None], groups)
    return dict(ols_spline=raw, cv_spline=(*raw[:-1], v), comoving_spline=(*comoving[:-1], v)), raw[-2]


def loss(net, inputs, scale, target, valid):
    p = net(*inputs, False)
    error = torch.where(valid[..., None], p-target, 0.)/scale[:, None, None]
    robust = (error.square().sum(-1)+.0025).sqrt()-.05
    return (robust.sum(1)/valid.sum(1).clamp_min(1)).mean()
