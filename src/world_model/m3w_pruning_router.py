"""Past-prefix repairability scores and a predicted support reserve."""
import numpy as np

from src.world_model import m3w_record_pruning as base
from src.world_model.m3w_pruning_teacher import THRESHOLDS

ROUTING_CUTS = (0., .2, .4, .6)
RESERVE = .5


def prefix(groups, action):
    """One row per active query, computed without its suffix."""
    total, peak = np.zeros(action.shape[1]), np.zeros(action.shape[1])
    n = 0
    out = []
    for q in groups:
        now = action[q]
        total += now.sum(0)
        n += len(q)
        peak = np.maximum(peak, np.abs(now).max(0))
        out.append(np.r_[1., np.log1p(n), np.log1p(len(q)), now.mean(0), total/n, peak])
    return np.asarray(out).reshape(len(groups), 3+3*action.shape[1])


def moments(x, target):
    return x.T @ x/len(x), x.mean(0)*target


def choose(groups, score, support, router, action_cut, route_cut, reserve):
    out = np.zeros(len(score), dtype=bool)
    if action_cut is None:
        return out
    proposed = retained = 0.
    for q, r in zip(groups, router, strict=True):
        mass = float(support[q].sum())
        proposed += mass
        retained += mass
        if r < route_cut:
            continue
        for i in q[np.argsort(-score[q], kind='stable')]:
            if score[i] <= action_cut:
                break
            if reserve and retained-support[i] < RESERVE*proposed - 1e-12:
                continue
            out[i] = True
            retained -= support[i]
            break
    return out


def calibrate(records, reserve):
    best = None
    count = 0
    for r in ROUTING_CUTS:
        for t in THRESHOLDS:
            recovered = lost = extra = drops = 0
            gain = 0.
            for g, score, support, router, z, full, w in records:
                take = choose(g, score, support, router, t, r, reserve)
                old, bad = base.observe(z, np.zeros(len(take), bool), full)
                new, worse = base.observe(z, take, full)
                recovered += int(new and not old)
                lost += int(old and not new)
                extra += int(worse and not bad)
                drops += int(take.sum())
                gain += float(w[take] @ z['remove_lower'][take])
            if recovered and not lost and not extra and gain >= 0:
                count += 1
                candidate = recovered, gain, -drops, r, t
                best = candidate if best is None else max(best, candidate)
    return ((None, None) if best is None else best[-2:]), count
