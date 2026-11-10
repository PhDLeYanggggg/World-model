"""Change only the direction of a frozen correction at each horizon step."""

import numpy as np

from src.world_model.m3w_ols_replacement import rollout
from src.world_model.m3w_radius_matched_turn_menu import turn_path


NAMES = ('original', 'short_velocity', 'long_velocity', 'acceleration', 'turn')


def match_direction(cv, anchor, proposal):
    a, p = anchor-cv, proposal-cv
    r, scale = np.linalg.norm(a, axis=-1), np.linalg.norm(p, axis=-1)
    factor = np.divide(r, scale, out=np.zeros_like(r), where=scale > 1e-10)
    return np.where((scale > 1e-10)[..., None], cv+p*factor[..., None], anchor)


def direction_bank(history, cv, anchor):
    derived, _, b, _ = rollout(history, cv.shape[1])
    if cv.shape != anchor.shape or not np.allclose(cv, derived, atol=1e-11, rtol=1e-11):
        raise ValueError('CV must be the same causal continuation for all directions')
    proposals = (b[:, 0], b[:, 1], b[:, 3], turn_path(history, cv.shape[1]))
    paths = np.stack((anchor, *(match_direction(cv, anchor, v) for v in proposals)), axis=1)
    if not np.isfinite(paths).all():
        raise ValueError('Nonfinite matched direction')
    return paths


def bridge_fraction(cv, ols, anchor):
    d = ols-cv
    mass = np.square(d).sum((1, 2))
    a = np.divide(((anchor-cv)*d).sum((1, 2)), mass, out=np.zeros_like(mass), where=mass > 0.)
    if np.any((a < -1e-10) | (a > 1.+1e-10)) or not np.allclose(
            anchor, cv+a[:, None, None]*d, atol=1e-10, rtol=1e-10):
        raise ValueError('The archived direction is not a CV-to-OLS4 bridge')
    return np.clip(a, 0., 1.)


def suffix_scores(history, a):
    """Use three prefix fits and their corresponding observed suffix errors."""
    if history.shape[1:] != (8, 2):
        raise ValueError('Require eight causal observations')
    losses = []
    for k in (4, 5, 6):
        h = history[:, :k]
        cv, ols, _, scale = rollout(h, 8-k)
        paths = direction_bank(h, cv, cv+a[:, None, None]*(ols-cv))
        y = history[:, k:]-history[:, k-1:k]
        losses.append(np.linalg.norm(paths-y[:, None], axis=-1).mean(2)/scale[:, None])
    return np.stack(losses)


def select(suffix):
    rank = suffix.mean(0)
    use = rank.argmin(1)
    better = (suffix[:, np.arange(len(use)), use] < suffix[..., 0]).all(0)
    return use, np.where(better, use, 0)
