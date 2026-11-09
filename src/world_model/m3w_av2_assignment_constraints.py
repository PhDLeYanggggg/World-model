"""Separate the support, risk and utility conditions in two-edit assignment."""

import numpy as np
import torch
from torch import nn

from src.evaluation.m3w_cv_anchor_budget import candidate_terms, risk_excess
from src.world_model import m3w_av2_assignment_cost as cost


ARMS = ("soft", "analytic", "learned")
PROB_CUT = .5


def labels(d, p, opts, y, valid, cutoff):
    cv, ols = d["cv"], d["ols4"]
    mass, fallback, gain, _ = candidate_terms(cv, ols, p, y, valid)
    risk = risk_excess(cv, p, y, valid, cutoff)
    safe = np.where(valid[..., None], y, 0.)
    count = valid.sum(1).clip(1)
    base = np.where(valid, np.linalg.norm(cv-safe, axis=-1), 0.).sum(1)/count
    e = np.where(valid[:, None], np.linalg.norm(p-safe[:, None], axis=-1), 0.).sum(2)/count[:, None]
    easy = (base > 0.) & (base <= cutoff)
    signed = np.where(easy[:, None], e-base[:, None], 0.)
    i, j = opts.T
    support = (easy[i] | easy[j]) & (mass.sum() > 0.)
    safe_risk = ((risk[i, 0]+risk[j, 1]) <= 0.).all(1) & (signed[i, 0]+signed[j, 1] <= .02*base[easy].sum())
    utility = ((gain[i, 0]+gain[j, 1]+np.array([0., fallback.sum()])) > 0.).all(1)
    return np.column_stack((support, safe_risk, utility))


def features(context, pair):
    scene = np.r_[context.mean(0), np.log1p(len(context))].astype(np.float32)
    return np.column_stack((pair, np.broadcast_to(scene, (len(pair), len(scene)))))


def head(width):
    return nn.Sequential(nn.Linear(width, 48), nn.SiLU(), nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, 3))


def loss(logits, label):
    """Chain factorization; risk and gain are fitted only on their antecedents."""
    a, b, _ = label.T
    use = torch.stack((torch.ones_like(a), a, a & b), dim=1)
    terms = nn.functional.binary_cross_entropy_with_logits(logits, label.to(logits.dtype), reduction="none")
    return ((terms*use).sum(0)/use.sum(0).clamp_min(1)).sum()


def select(context, action, prob, opts):
    i, j = opts.T
    scores = action[i, 0]+action[j, 1]
    easy = 1./(1.+np.exp(-np.clip(context[:, 0], -50., 50.)))
    support = 1.-(1.-easy[i])*(1.-easy[j])
    utility = np.minimum(scores[:, 0], scores[:, 1]+context[:, 1].sum())
    risk = (scores[:, 2:4] <= 0.).all(1) & (scores[:, 4] <= .02*np.maximum(context[:, 2], 0.).sum())
    soft, conf = cost.choose(context, action, np.zeros(len(opts)), opts)
    rules = ((support >= PROB_CUT) & risk & (utility > 0.), prob.prod(1) >= PROB_CUT)
    output = [(int(soft[0]), float(conf[0]), len(opts))]
    for ok, confidence in zip(rules, (support, prob.prod(1)), strict=True):
        # Forced-choice readout uses the unchanged soft candidate when no option is admissible.
        # The missing release score guarantees actual abstention, not an implicit release.
        k = int(np.where(ok, utility, -np.inf).argmax()) if ok.any() else int(soft[0])
        output.append((k, float(confidence[k]) if ok.any() else None, int(ok.sum())))
    return output
