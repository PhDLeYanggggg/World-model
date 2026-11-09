"""Continuous assignment targets and an optional within-scenario listwise loss."""

import numpy as np
from torch import nn

from src.evaluation.m3w_cv_anchor_budget import candidate_terms, risk_excess


TEMPERATURE = .25
RANK_WEIGHT = .5


def targets(d, paths, opts, y, valid, cutoff, scale):
    cv, ols = d["cv"], d["ols4"]
    mass, base, gains, _ = candidate_terms(cv, ols, paths, y, valid)
    risk = risk_excess(cv, paths, y, valid, cutoff)
    clean = np.where(valid[..., None], y, 0.)
    count = valid.sum(1).clip(1)
    ref = np.where(valid, np.linalg.norm(cv-clean, axis=-1), 0.).sum(1)/count
    err = np.where(valid[:, None], np.linalg.norm(paths-clean[:, None], axis=-1), 0.).sum(2)/count[:, None]
    easy = (ref > 0.) & (ref <= cutoff)
    signed = np.where(easy[:, None], err-ref[:, None], 0.)
    i, j = opts.T
    gain = (gains[i, 0]+gains[j, 1]+np.array([0., base.sum()])).min(1)
    risks = np.column_stack((risk[i, 0]+risk[j, 1], signed[i, 0]+signed[j, 1]-.02*ref[easy].sum()))
    violation = risks.max(1)
    supported = (easy[i] | easy[j]) & (mass.sum() > 0.)
    eligible = supported & (gain > 0.) & (violation <= 0.)
    u = np.tanh(gain/(2.*y.shape[1]*scale))
    r = np.tanh(violation/scale)
    # Qualifying options always outrank violations; continuous margins rank inside each group.
    score = np.where(eligible, 1.+.5*u+.25*np.maximum(-r, 0.),
                     -.5*(~supported)-.5*np.maximum(r, 0.)+.25*np.minimum(u, 0.))
    return score.astype(np.float32), eligible


def head(width):
    return nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, 1))


def losses(pred, target, valid):
    f = nn.functional
    count = valid.sum(1).clamp_min(1)
    point = (f.smooth_l1_loss(pred, target, beta=.25, reduction="none").masked_fill(~valid, 0.).sum(1)/count).mean()
    truth = f.softmax((target/TEMPERATURE).masked_fill(~valid, -1e4), dim=1)
    logp = f.log_softmax((pred/TEMPERATURE).masked_fill(~valid, -1e4), dim=1)
    return point, -(truth*logp).sum(1).mean()


def choose(score):
    k = int(score.argmax())
    return k, float(np.exp(np.clip(score[k], -25., 25.)))
