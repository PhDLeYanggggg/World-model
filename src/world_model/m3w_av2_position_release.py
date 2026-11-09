"""Causal confidence for a fixed projected position forecast."""

import numpy as np

from src.world_model import m3w_av2_history_replay as history


ARMS = ("risk_replay", "trajectory_replay", "transferred_cost")


def history_scores(d, path, edited, cutoff):
    """Replay the same edit shapes against already observed positions."""
    h, cv = d["h"], d["cv"]
    delta = path-cv
    final = h[:, -1]-h[:, -2]
    trajectory, risk, support = [], [], []
    for k in history.PREFIXES:
        n = h.shape[1]-k
        c, o, _, scale = history.motion.rollout(h[:, :k], n)
        old = h[:, k:]-h[:, k-1:k]
        v = h[:, k-1]-h[:, k-2]
        change = history.align(delta[:, :n], final, v)
        errors = [np.linalg.norm(z-old, axis=-1).mean(-1) for z in (c, o, c+change)]
        a, b, e = errors
        easy = (a > 0.) & (a <= cutoff*(n+1.)/13.)
        z = e-a
        i = edited
        excess = (
            np.maximum(z[i], 0.).sum()-.02*a[i].sum(),
            (np.maximum(z[i], 0.)[easy[i]]-.02*a[i][easy[i]]).sum(),
            z[easy].sum()-.02*a[easy].sum(),
        )
        unit = scale.mean()
        trajectory.append(min((a-e).sum(), (b-e).sum())/unit)
        risk.append(max(0., *excess)/unit)
        support.append(easy[i].any())
    safe_score = np.quantile(np.asarray(trajectory)-risk, .25)
    raw_score = np.quantile(trajectory, .25)
    return float(np.exp(np.clip(safe_score, -25., 25.))*(.25+.75*np.mean(support))), float(np.exp(np.clip(raw_score, -25., 25.)))


def choose_calibrated(rules):
    """Choose only on the number of strictly qualifying calibration releases."""
    winner = max(ARMS, key=lambda k: rules[k]["accepted"])
    return winner if rules[winner]["accepted"] else None
