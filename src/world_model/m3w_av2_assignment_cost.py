"""Structured action costs and a separate two-edit interaction residual."""

import numpy as np
from torch import nn

from src.evaluation.m3w_cv_anchor_budget import candidate_terms, risk_excess
from src.world_model import m3w_av2_history_replay as history


PAIR_WEIGHT = .5
SUPPORT_WEIGHT = .1
ARMS = ("additive", "joint")


def inputs(d, p, opts, cutoff):
    h, c, o = d["h"], d["cv"], d["ols4"]
    norm = lambda x: np.linalg.norm(x, axis=-1)
    scale = float(norm(np.diff(h, axis=1)).mean().clip(.05))
    n = len(h)
    delta = p-c[:, None]
    velocity = h[:, -1]-h[:, -2]
    replay_actor, replay_action = [], []
    for k in history.PREFIXES:
        c0, o0, _, _ = history.motion.rollout(h[:, :k], 8-k)
        y = h[:, k:]-h[:, k-1:k]
        a = history.align(delta[:, :, :8-k], velocity[:, None], (h[:, k-1]-h[:, k-2])[:, None])
        ce, oe, pe = (norm(e).mean(-1) for e in (c0-y, o0-y, (c0-y)[:, None]+a))
        replay_actor.append(np.column_stack((ce, oe))/scale)
        replay_action.append(np.stack((ce[:, None]-pe, oe[:, None]-pe), axis=-1)/scale)
    context = np.column_stack(((h-h[:, -1:]).reshape(n, -1)/scale, (o-c).reshape(n, -1)/scale,
                               np.full(n, np.log1p(cutoff/scale)), *replay_actor))
    action = np.concatenate((np.broadcast_to(context[:, None], (n, 2, context.shape[-1])),
                             delta.reshape(n, 2, -1)/scale, *replay_action), axis=-1)
    i, j = opts.T
    pair = np.column_stack((action[i, 0], action[j, 1], (h[i]-h[j]).reshape(len(i), -1)/scale))
    return *(x.clip(-24., 24.).astype(np.float32) for x in (context, action, pair)), scale


def targets(d, p, opts, y, valid, cutoff, scale):
    cv, ols = d["cv"], d["ols4"]
    _, base, gain, _ = candidate_terms(cv, ols, p, y, valid)
    risk = risk_excess(cv, p, y, valid, cutoff)
    clean = np.where(valid[..., None], y, 0.)
    norm = lambda x: np.linalg.norm(x, axis=-1)
    err = np.where(valid, norm(cv-clean), 0.).sum(1)/valid.sum(1).clip(1)
    e = np.where(valid[:, None], norm(p-clean[:, None]), 0.).sum(2)/valid.sum(1).clip(1)[:, None]
    easy = (err > 0.) & (err <= cutoff)
    signed = np.where(easy[:, None], e-err[:, None], 0.)
    action = np.concatenate((gain/y.shape[1], risk, signed[..., None]), axis=-1)/scale
    context = np.column_stack((easy, base/(y.shape[1]*scale), np.where(easy, err/scale, 0.)))
    i, j = opts.T
    both = valid[i] & valid[j]
    c = cv[i]-cv[j]
    error = c-(clean[i]-clean[j])
    delta = p-cv[:, None]
    a, b = delta[i, 0], -delta[j, 1]
    def gain_for(change):
        return np.where(both, norm(error)-norm(error+change), -norm(change)).mean(1)
    pair = (gain_for(a+b)-gain_for(a)-gain_for(b))/scale
    return context.astype(np.float32), action.astype(np.float32), pair.astype(np.float32)


class Heads(nn.Module):
    def __init__(self, widths):
        super().__init__()
        net = lambda w, o: nn.Sequential(nn.Linear(w, 48), nn.SiLU(), nn.Linear(48, 24), nn.SiLU(), nn.Linear(24, o))
        self.context = net(widths[0], 3)
        self.action = net(widths[1], 5)
        self.pair = net(widths[2], 1)


def choose(context, action, residual, opts):
    i, j = opts.T
    z = action[i, 0]+action[j, 1]
    total_ols = z[:, 1]+context[:, 1].sum()
    signed = z[:, 4]-.02*np.maximum(context[:, 2], 0.).sum()
    easy = 1./(1.+np.exp(-np.clip(context[:, 0], -50., 50.)))
    support = 1.-(1.-easy[i])*(1.-easy[j])
    margin = np.minimum(z[:, 0], total_ols)-np.maximum(z[:, 2:4], 0.).sum(1)-np.maximum(signed, 0.)
    additive = margin+SUPPORT_WEIGHT*np.log(support.clip(1e-9))
    choice = np.array([additive.argmax(), (additive+PAIR_WEIGHT*residual).argmax()])
    confidence = support[choice]/(1.+np.exp(-np.clip(margin[choice], -50., 50.)))
    return choice, confidence
