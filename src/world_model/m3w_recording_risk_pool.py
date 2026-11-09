"""Fixed-count forecast alternatives with recording-level risk calibration."""

import numpy as np

from src.world_model.m3w_full_scene_harm_transfer import TOLERANCE
from src.world_model.m3w_observation_augmentation import match_gain


PRICES = (0., 1., 4., 16.)
ARMS = ("independent", "price_1", "price_4", "price_16")
RULES = ("raw", "source_median", "source_envelope")


def query_menu(paths, moments, refs, frozen_independent):
    """Keep the independent count in every query but reassign actor and path."""
    legal = (paths[:, 2:] != paths[:, 1:2]).any((2, 3))
    changed_cv = (paths[:, 1:] != paths[:, :1]).any((2, 3))
    mass = np.where(changed_cv[..., None], refs[:, None], 0.)
    harm = np.where(changed_cv[..., None], moments[..., 1:], 0.)
    gain = moments[..., 0]-moments[:, :1, 0]
    excess = (harm-TOLERANCE*mass).sum(2)
    delta = excess[:, 1:]-excess[:, :1]
    count = np.count_nonzero(frozen_independent)
    rows, totals = [], []
    idx = np.arange(len(paths))
    for p in PRICES:
        c = match_gain(gain[:, 1:]-p*delta, legal, count)
        if p == 0. and not np.array_equal(c, frozen_independent):
            raise ValueError("Control paths changed")
        rows.append(c)
        totals.append(np.r_[harm[idx, c].sum(0), mass[idx, c].sum(0),
                            gain[idx, c].mean(), refs[:, 0].mean()])
    return np.column_stack(rows), np.stack(totals)


def prediction(totals):
    t = np.asarray(totals)
    if t.shape != (len(ARMS), 6) or not np.isfinite(t).all() or (t[:, :4] < 0.).any():
        raise ValueError("Expected finite aggregate forecast moments")
    risk = np.divide(t[:, :2], t[:, 2:4], out=np.full_like(t[:, :2], np.inf), where=t[:, 2:4] > 0.)
    gain = np.divide(t[:, 4], t[:, 5], out=np.full(len(t), -np.inf), where=t[:, 5] > 0.)
    return risk, gain


def select(risk, gain, offset):
    safe = np.isfinite(risk).all(1) & (risk+offset <= TOLERANCE).all(1) & (gain > 0.)
    return int(np.where(safe, gain, -np.inf).argmax()) if safe.any() else None


def group_offsets(by_source, held):
    """Fit only on whole non-held groups. Rows may include seeds, not extra sources."""
    groups = [np.asarray(v) for k, v in sorted(by_source.items()) if k != held]
    if not groups or not all(x.ndim == 3 and x.shape[1:] == (len(ARMS), 2) and np.isfinite(x).all() for x in groups):
        raise ValueError("Need complete finite source residuals")
    median = np.median([np.median(x, axis=0) for x in groups], axis=0)
    envelope = np.max([x.max(0) for x in groups], axis=0)
    return np.zeros_like(median), median, envelope
