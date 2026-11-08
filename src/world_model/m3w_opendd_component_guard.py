"""Separate neural speed and lateral corrections in an observed motion frame."""

import numpy as np


def peak(offset):
    return np.linalg.norm(offset, axis=-1).max(axis=-1)


def amplitude(offset, target):
    original = peak(offset)
    factor = np.divide(target, original, out=np.zeros_like(original), where=original > 1e-12)
    return offset*factor[:, None, None]


def split(history, cv, ols, neural):
    """Align whole-path components with the sign of the observed OLS4/CV disagreement."""
    if history.ndim != 3 or history.shape[1:] != (8, 2) or any(
            v.shape != (len(history), 12, 2) for v in (cv, ols, neural)):
        raise ValueError("Expected aligned 8/12 arrays")
    if any(not np.isfinite(v).all() for v in (history, cv, ols, neural)):
        raise ValueError("Expected finite observed inputs and predictions")
    last = history[:, -1]-history[:, -2]
    chord = history[:, -1]-history[:, 0]
    direction = np.where(np.linalg.norm(last, axis=1)[:, None] > 1e-8, last, chord)
    norm = np.linalg.norm(direction, axis=1)
    defined = norm > 1e-8
    axis = np.divide(direction, norm[:, None], out=np.zeros_like(direction), where=defined[:, None])

    def project(v):
        return (v*axis[:, None]).sum(-1)[..., None]*axis[:, None]

    delta = neural-cv
    longitudinal = project(delta)
    lateral = delta-longitudinal
    contrast = ols-cv
    along = project(contrast)
    across = contrast-along
    l_ok = defined & ((longitudinal[:, -1]*along[:, -1]).sum(1) > 1e-12)
    t_ok = defined & ((lateral[:, -1]*across[:, -1]).sum(1) > 1e-12)
    guarded = longitudinal*l_ok[:, None, None] + lateral*t_ok[:, None, None]
    return dict(full=delta, longitudinal=longitudinal, lateral=lateral, guard=guarded,
                equal_peak=amplitude(delta, peak(guarded))), defined, l_ok, t_ok


def instability(history, cv, ols):
    """Normalized disagreement of two past-only forecasts; no label-derived bins."""
    step = np.sqrt(np.square(np.diff(history, axis=1)).sum(2).mean(1)).clip(.05)
    return np.linalg.norm(ols[:, -1]-cv[:, -1], axis=1)/(12.*step)
