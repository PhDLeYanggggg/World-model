"""Label-side text precision and supplied-H projection diagnostics, not features."""
from __future__ import annotations

from decimal import Decimal
import numpy as np


def printed_interval(token):
    value = Decimal(token)
    if not value.is_finite():
        raise ValueError('Finite printed coordinate required')
    quantum = Decimal(1).scaleb(value.as_tuple().exponent)
    return value-quantum/2, value+quantum/2, quantum


def constant_compatible(tokens_xy):
    """Conservative closed rounding intervals. Not annotation-noise bounds."""
    if len(tokens_xy)<2 or any(len(row)!=2 for row in tokens_xy):
        raise ValueError('At least two printed xy positions required')
    overlap = []
    for axis in range(2):
        intervals = [printed_interval(row[axis]) for row in tokens_xy]
        overlap.append(max(v[0] for v in intervals) <= min(v[1] for v in intervals))
    return bool(all(overlap))


def project_supplied_h(xy, matrix):
    points, h = np.asarray(xy, dtype=float), np.asarray(matrix, dtype=float)
    if points.shape[-1:]!=(2,) or h.shape!=(3, 3) or not np.isfinite(points).all() or not np.isfinite(h).all():
        raise ValueError('Finite coordinate pairs and 3x3 supplied H required')
    inverse = np.linalg.inv(h)
    flat = points.reshape(-1, 2)
    homogeneous = np.c_[flat, np.ones(len(flat))] @ inverse.T
    if np.any(np.abs(homogeneous[:, 2]) <= 1e-12):
        raise ValueError('Projection at infinity is unsupported, not silently clipped')
    pixels = homogeneous[:, :2]/homogeneous[:, 2:3]
    if not np.isfinite(pixels).all():
        raise ValueError('Nonfinite projection')
    return pixels.reshape(points.shape)


def resolution_counts(maximum, native_changed, cv_error, agent_keys, run_keys,
                      thresholds=(.5, 1., 2., 5., 10.), tolerance=.001):
    maximum, changed, error = np.asarray(maximum), np.asarray(native_changed), np.asarray(cv_error)
    if (maximum.ndim!=1 or changed.shape!=maximum.shape or error.shape!=maximum.shape
            or len(agent_keys)!=len(maximum) or len(run_keys)!=len(maximum)
            or not np.isfinite(maximum).all() or not np.isfinite(error).all()
            or np.any(maximum<0) or np.any(error<0) or tolerance<0):
        raise ValueError('Aligned finite label diagnostics required')
    result = []
    for threshold in thresholds:
        selected = changed & (maximum > threshold+tolerance)
        boundary = changed & (np.abs(maximum-threshold)<=tolerance)
        result.append({'inferred_pixel_threshold': threshold, 'numerical_tolerance': tolerance,
            'rows_above': int(selected.sum()), 'agents_above': len({tuple(agent_keys[i]) for i in np.flatnonzero(selected)}),
            'runs_above': len({tuple(run_keys[i]) for i in np.flatnonzero(selected)}),
            'rows_within_numerical_boundary': int(boundary.sum()),
            'cv_error_share_above': float(error[selected].sum()/error.sum()) if error.sum()>0 else None,
            'native_cv_ade_above': float(error[selected].mean()) if selected.any() else None,
            'labels_redefined': False})
    return result


def slice_forecast_metrics(prediction, target, mask):
    p, y, mask = np.asarray(prediction), np.asarray(target), np.asarray(mask)
    if p.shape!=y.shape or p.shape[1:]!=(12, 2) or mask.dtype!=bool or mask.shape!=(len(y),):
        raise ValueError('Aligned trajectory labels, predictions and slice mask required')
    if not mask.any():
        return {'status': 'not_run_empty_slice', 'rows': 0}
    error = np.linalg.norm(p[mask]-y[mask], axis=-1).mean(1)
    floor = np.linalg.norm(y[mask], axis=-1).mean(1)
    return {'status': 'fresh_run_frozen_forecast_label_slice', 'rows': int(mask.sum()),
        'native_ade': float(error.mean()), 'cv_native_ade': float(floor.mean()),
        'gain_vs_cv_pct': float(100*(1-error.mean()/floor.mean())) if floor.mean()>0 else None,
        'native_harm': float((error-floor).mean()),
        'native_positive_harm': float(np.maximum(error-floor, 0).mean()),
        'ratio_status': 'defined' if floor.mean()>0 else 'undefined_zero_floor',
        'selection_allowed': False}
