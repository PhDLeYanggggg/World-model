"""Causal reference choices on the same two-profile assignment menu."""

import numpy as np

from src.world_model import m3w_av2_history_replay as history


ARMS = ("original", "soft", "ols_agreement", "prefix_robust", "guarded_soft")


def prefix_scores(h, cv, bank, opts, cutoff):
    i, j = opts.T
    delta = bank-cv[:, None]
    final_v = h[:, -1]-h[:, -2]
    score, support = [], []
    norm = lambda a: np.linalg.norm(a, axis=-1).mean(-1)
    for k in history.PREFIXES:
        n = h.shape[1]-k
        c, o, _, unit = history.motion.rollout(h[:, :k], n)
        y = h[:, k:]-h[:, k-1:k]
        change = history.align(delta[:, :, :n], final_v[:, None], (h[:, k-1]-h[:, k-2])[:, None])
        a, b, e = norm(c-y), norm(o-y), norm((c-y)[:, None]+change)
        easy = (a > 0.) & (a <= cutoff*(n+1.)/13.)
        cv_gain = a[i]+a[j]-e[i, 0]-e[j, 1]
        ols_gain = (b-a).sum()+cv_gain
        hi, hj = np.maximum(e[i, 0]-a[i], 0.), np.maximum(e[j, 1]-a[j], 0.)
        all_excess = hi+hj-.02*(a[i]+a[j])
        easy_excess = easy[i]*(hi-.02*a[i])+easy[j]*(hj-.02*a[j])
        score.append((np.minimum(cv_gain, ols_gain)-np.maximum.reduce([all_excess, easy_excess, np.zeros(len(i))]))/unit.mean())
        support.append(easy[i] | easy[j])
    return np.quantile(score, .25, axis=0), np.sum(support, axis=0) >= 2


def select(d, bank, opts, cutoff, original, soft):
    i, j = opts.T
    distance = np.linalg.norm(bank-d["ols4"][:, None], axis=-1).mean(-1)
    distance = distance[i, 0]+distance[j, 1]
    score, support = prefix_scores(d["h"], d["cv"], bank, opts, cutoff)
    robust = np.where(support, score, -np.inf).argmax() if support.any() else score.argmax()
    admit = support & (score > score[soft]+1e-10) & (distance <= distance[soft]+1e-10)
    guarded = np.where(admit, score, -np.inf).argmax() if admit.any() else soft
    return tuple(map(int, (original, soft, distance.argmin(), robust, guarded)))
