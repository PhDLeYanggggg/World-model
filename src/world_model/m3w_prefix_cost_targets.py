"""Label-only prefix costs and a separate past-forecast disagreement interface."""
import numpy as np


def _forecasts(baseline, candidate, scale):
    b, p, s = np.asarray(baseline, float), np.asarray(candidate, float), np.asarray(scale, float)
    if (b.ndim != 3 or b.shape[-1] != 2 or p.shape != b.shape or s.shape != (len(b),)
            or b.shape[1] != 12 or not np.isfinite(b).all() or not np.isfinite(p).all()
            or not np.isfinite(s).all() or np.any(s <= 0)):
        raise ValueError("Finite 12-step candidate/baseline forecasts and positive causal scale required")
    return b, p, s


def causal_prefix_disagreement(baseline, candidate, scale):
    """Future predictions are causal inputs; future ground truth is not accepted."""
    b, p, s = _forecasts(baseline, candidate, scale)
    distance = np.linalg.norm(p-b, axis=-1) * s[:, None]
    return np.cumsum(distance, axis=1) / np.arange(1, 13)


def supervised_prefix_costs(baseline, candidate, target, valid, scale):
    """Loss/evaluation labels only. Missing prefixes remain NaN, never safe zeros."""
    b, p, s = _forecasts(baseline, candidate, scale)
    y, mask = np.asarray(target, float), np.asarray(valid)
    if (y.shape != b.shape or mask.shape != b.shape[:2] or mask.dtype != bool
            or not np.isfinite(y[mask]).all()):
        raise ValueError("Aligned finite supported targets and boolean label mask required")
    safe = np.where(mask[..., None], y, 0)
    cv = np.linalg.norm(b-safe, axis=-1) * s[:, None]
    nn = np.linalg.norm(p-safe, axis=-1) * s[:, None]
    gain = np.cumsum(np.where(mask, cv-nn, 0), axis=1) / np.arange(1, 13)
    available = np.logical_and.accumulate(mask, axis=1)
    costs = np.stack((np.maximum(gain, 0), np.maximum(-gain, 0)), axis=-1)
    costs[~available] = np.nan
    return dict(costs=costs, available=available,
                disagreement=causal_prefix_disagreement(b, p, s))
