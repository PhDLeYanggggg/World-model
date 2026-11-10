"""Outcome summaries for a few frozen forecasts, without absolute trajectories."""

import numpy as np


def compare(left, right, target, valid, normalizer):
    if (left.shape != right.shape or left.shape != target.shape or left.ndim != 3
            or left.shape[1:] != (12, 2) or valid.shape != left.shape[:2] or valid.dtype != bool
            or not np.isfinite(left).all() or not np.isfinite(right).all() or not np.isfinite(target[valid]).all()
            or not np.isfinite(normalizer) or normalizer <= 0.):
        raise ValueError('Invalid matched case geometry, observed targets or normalization')
    error = np.linalg.norm(left-target, axis=-1)-np.linalg.norm(right-target, axis=-1)
    gain = np.where(valid, error, 0.)*normalizer
    missing = np.where(valid, 0., np.linalg.norm(left-right, axis=-1))*normalizer
    point, radius = (np.r_[0., np.cumsum(x.sum(0))] for x in (gain, missing))
    return dict(observed=point.tolist(), lower=(point-radius).tolist(), upper=(point+radius).tolist(),
                actor_observed=gain.sum(1).tolist(), actor_debit=missing.sum(1).tolist())


def median_case(candidates):
    """Select the lower median with a stable tie break, never the largest effect."""
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: (x['rank'], x['member']))[(len(candidates)-1)//2]
