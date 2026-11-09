"""Train relative to a strong motion path without moving the risk reference."""

import numpy as np

from src.world_model.m3w_full_scene_harm_transfer import TOLERANCE, targets
from src.world_model.m3w_observation_augmentation import match_gain


PRICES = (0., .5, 1., 2., 4., 8., 16.)
ARMS = ("cv_gain", "ols_matched", "ols_gain", "ols_joint", "ols_joint_matched", "ols4")


def paired_targets(paths, y, valid, scale, cutoff):
    cv, refs = targets(paths, y, valid, scale, cutoff)
    alt = cv.copy()
    base = paths[:, 1:2]
    labels = np.where(valid[..., None], y, 0.)[:, None]
    gain = np.linalg.norm(base-labels, axis=-1)-np.linalg.norm(paths[:, 1:]-labels, axis=-1)
    gain = np.where(valid[:, None], gain, 0.).sum(2)/valid.sum(1).clip(1)[:, None]
    radius = np.linalg.norm(paths[:, 1:]-base, axis=-1).max(2)
    alt[..., 0] = (gain-np.where(valid.any(1)[:, None], 0., radius))/scale[:, None]
    return cv, alt, refs


def choose(gain, legal):
    return np.where((legal & (gain > 0.)).any(1), np.where(legal, gain, -np.inf).argmax(1), 0)


def budgeted(gain, harm, ref, changed_from_cv, legal):
    """Query-wide price search, including the costs of every unchanged OLS4 path."""
    mass = np.where(changed_from_cv[..., None], ref[:, None], 0.)
    harm = np.where(changed_from_cv[..., None], harm, 0.)
    excess = harm-TOLERANCE*mass
    prices = np.asarray([(a, b) for a in PRICES for b in PRICES])
    score = gain[None]-np.einsum("pk,nak->pna", prices, excess)
    allowed = legal & (gain > 0.)
    allowed[:, 0] = True
    idx = np.where(allowed, score, -np.inf).argmax(2)
    row = np.arange(len(gain))[None]
    supported = (mass[row, idx].sum(1) > 0.).all(1) & (excess[row, idx].sum(1) <= 0.).all(1)
    utility = gain[row, idx].sum(1)
    if not supported.any():
        return np.zeros(len(gain), dtype=np.uint8), False
    return idx[np.where(supported, utility, -np.inf).argmax()], True


def decisions(paths, cv_moments, ols_moments, refs):
    legal = (paths[:, 1:] != paths[:, 1:2]).any((2, 3))
    edited = (paths[:, 1:] != paths[:, :1]).any((2, 3))
    g0 = cv_moments[..., 0]-cv_moments[:, :1, 0]
    g1 = ols_moments[..., 0]-ols_moments[:, :1, 0]
    a, b = choose(g0, legal), choose(g1, legal)
    joint, supported = budgeted(g1, ols_moments[..., 1:], refs, edited, legal)
    return np.column_stack((a, match_gain(g1[:, 1:], legal[:, 1:], np.count_nonzero(a)), b,
        joint, match_gain(g1[:, 1:], legal[:, 1:], np.count_nonzero(joint)), np.zeros(len(a)))).astype(np.uint8), supported
