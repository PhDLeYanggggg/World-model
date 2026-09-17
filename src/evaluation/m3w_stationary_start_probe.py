"""Past-only start identifiability on fit folds, not a forecasting evaluation."""
from __future__ import annotations

import numpy as np


NEIGHBOR_FIELDS = ('present', 'distance', 'path', 'last_speed_horizon', 'net_displacement',
                   'radial_speed_horizon', 'closing_speed_horizon', 'closest_time_horizon',
                   'closest_distance', 'straightness')
FEATURE_NAMES = ['visible_count', 'complete_aligned_count', 'degenerate_distance_scale'] + [
    f'neighbor{i}_{name}' for i in range(8) for name in NEIGHBOR_FIELDS]
GEOMETRY_COLUMNS = [0, 1, 2] + [3 + i * len(NEIGHBOR_FIELDS) + j for i in range(8) for j in (0, 1)]


def past_features(inputs):
    """No labels/reader accepted. Ratios avoid mixing native coordinate scales."""
    expected = {'history_xy', 'history_frame_offsets', 'history_velocity', 'history_mask',
                'neighbor_xy', 'neighbor_mask', 'neighbor_frame_offsets', 'baseline_rollouts',
                'prediction_frame_offsets', 'causal_features'}
    if set(inputs) != expected:
        raise ValueError('Only declared past/request input schema allowed')
    h, n = np.asarray(inputs['history_xy'], dtype=np.float64), np.asarray(inputs['neighbor_xy'], dtype=np.float64)
    t, nt = np.asarray(inputs['history_frame_offsets']), np.asarray(inputs['neighbor_frame_offsets'])
    mask, nm = np.asarray(inputs['history_mask']), np.asarray(inputs['neighbor_mask'])
    if (h.shape != (8, 2) or n.shape != (8, 8, 2) or mask.dtype != bool or nm.dtype != bool
            or mask.shape != (8,) or nm.shape != (8, 8) or not mask.all()
            or t.shape != (8,) or nt.shape != (8, 8) or not np.isfinite(h).all()
            or not np.isfinite(n[nm]).all() or np.any(t > 0) or np.any(nt[nm] > 0)
            or not np.all(np.diff(t) > 0) or t[-1] != 0):
        raise ValueError('Finite eight-step past context required')
    if np.any(np.diff(h, axis=0) != 0):
        raise ValueError('This probe requires exactly stationary ego histories')
    offsets = np.asarray(inputs['prediction_frame_offsets'])
    if offsets.shape != (12,) or not np.isfinite(offsets).all() or offsets[0] <= 0 or np.any(np.diff(offsets) <= 0):
        raise ValueError('Fixed twelve-step requested grid required')
    eligible = nm.all(1) & np.isclose(nt, t[None], atol=1e-6, rtol=1e-6).all(1)
    eligible_xy = n[eligible]
    distances = np.linalg.norm(eligible_xy[:, -1], axis=1)
    positive = distances[distances > 0]
    scale = float(np.median(positive)) if len(positive) else 1.
    rows = np.zeros((8, len(NEIGHBOR_FIELDS)), dtype=np.float64)
    for i in np.flatnonzero(eligible):
        xy = n[i] / scale
        delta = np.diff(xy, axis=0)
        path = np.linalg.norm(delta, axis=1).sum()
        displacement = np.linalg.norm(xy[-1] - xy[0])
        velocity_horizon = delta[-1] / (t[-1] - t[-2]) * offsets[-1]
        rel = xy[-1]
        dist, vv = np.linalg.norm(rel), velocity_horizon @ velocity_horizon
        radial = (rel @ velocity_horizon) / dist if dist else 0.
        tau = float(np.clip(-(rel @ velocity_horizon) / vv, 0, 10)) if vv else 10.
        closest = np.linalg.norm(rel + min(tau, 1.) * velocity_horizon)
        rows[i] = [1., dist, path, np.linalg.norm(velocity_horizon), displacement, radial,
                   max(0., -radial), tau, closest, displacement / path if path else 0.]
    features = np.r_[float(inputs['causal_features'][6]), eligible.sum(), len(positive) == 0, rows.ravel()]
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite causal ratios')
    return features


def stationary_label(history, future):
    history, future = np.asarray(history, dtype=np.float64), np.asarray(future, dtype=np.float64)
    if (history.shape != (8, 2) or future.shape != (12, 2)
            or not np.isfinite(history).all() or not np.isfinite(future).all()
            or np.any(np.diff(history, axis=0) != 0)):
        raise ValueError('Complete stationary-past and twelve-step labels required')
    distance = np.linalg.norm(future - history[-1], axis=1)
    changed = np.flatnonzero(distance > 0)
    return {'changed': bool(len(changed)), 'first_change_step': int(changed[0]+1) if len(changed) else None,
            'native_ade': float(distance.mean()), 'native_fde': float(distance[-1]),
            'native_max_displacement': float(distance.max()),
            'native_future_path': float(np.linalg.norm(np.diff(np.vstack([history[-1], future]), axis=0), axis=1).sum())}


def stationary_run(points, current):
    """Source audit only; future run boundary is NEVER an inference feature."""
    first = last = current
    step = points[current, 0] - points[current-1, 0]
    while first > 0 and points[first-1, 1] == points[current, 1] and points[first, 0]-points[first-1, 0] == step and np.array_equal(points[first-1, 2:], points[current, 2:]):
        first -= 1
    while last+1 < len(points) and points[last+1, 1] == points[current, 1] and points[last+1, 0]-points[last, 0] == step and np.array_equal(points[last+1, 2:], points[current, 2:]):
        last += 1
    return {'first_row': int(first), 'last_row': int(last),
            'starts_at_track_entry': bool(first == 0 or points[first-1, 1] != points[current, 1]),
            'ends_at_track_exit': bool(last == len(points)-1 or points[last+1, 1] != points[current, 1]),
            'observed_rows_since_run_start': int(current-first+1), 'whole_run_rows_label_only': int(last-first+1)}


def score_probabilities(y, probability, *, prior):
    from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
    y, p = np.asarray(y, dtype=int), np.asarray(probability, dtype=float)
    if not len(y) or p.shape != y.shape or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Valid held-fold probabilities required')
    baseline = np.full(len(y), prior)
    brier, base_brier = float(np.mean((p-y)**2)), float(np.mean((baseline-y)**2))
    return {'rows': len(y), 'positives': int(y.sum()), 'positive_rate': float(y.mean()),
            'train_only_prior': prior, 'brier': brier, 'prior_brier': base_brier,
            'brier_lift_over_prior': base_brier-brier,
            'log_loss': float(log_loss(y, np.column_stack([1-p, p]), labels=[0, 1])),
            'prior_log_loss': float(log_loss(y, np.column_stack([1-baseline, baseline]), labels=[0, 1])),
            'auroc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
            'average_precision': float(average_precision_score(y, p)) if y.sum() else None,
            'rank_metric_status': 'defined' if len(np.unique(y)) == 2 else 'single_class_no_auroc',
            'calibrated_risk': False}
