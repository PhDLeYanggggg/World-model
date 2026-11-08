"""Past-only shape controls for frozen OpenDD reserve recipients."""

import numpy as np

from src.world_model.m3w_ols_replacement import rollout

ANALYTIC = ("ols4", "ols3", "ols8", "damped4", "bounded_acceleration4")


def analytic(history):
    """Choose a trajectory family on observed 4/5/6-step prefixes only."""
    cv, ols, rest, _ = rollout(history, 12)
    bank = np.concatenate((ols[:, None], rest), axis=1)
    scores = np.zeros(bank.shape[:2])
    for n in (4, 5, 6):
        _, linear, other, scale = rollout(history[:, :n], 8-n)
        pred = np.concatenate((linear[:, None], other), axis=1)
        actual = history[:, n:]-history[:, n-1:n]
        scores += np.linalg.norm(pred-actual[:, None], axis=-1).mean(2)/scale[:, None]
    choose = scores.argmin(1)
    return cv, ols, bank[np.arange(len(choose)), choose], choose


def peak(delta):
    return np.linalg.norm(delta, axis=-1).max(1)


def match(cv, neural, ols, selected):
    """Match peak displacement with no amplification or new recipients."""
    delta = np.stack([p-cv for p in (neural, ols, selected)])
    norms = np.stack([peak(d) for d in delta])
    common = norms.min(0)
    legal = common > 1e-12
    same = []
    for d, radius in zip(delta, norms, strict=True):
        out = neural.copy()
        out[legal] = cv[legal]+d[legal]*(common[legal]/radius[legal])[:, None, None]
        same.append(out)
    return same, legal, common
