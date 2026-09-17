"""Proper-score information probe, separate from trajectory-model selection."""
from __future__ import annotations

import numpy as np

from src.evaluation.m3w_stationary_pooled_context import pool_context

ARMS = ('neighbors', 'quality', 'magnitude', 'directed')


def probe_features(neighbors, motion, quality, arm):
    """Only frozen past summaries enter; labels and row metadata are not accepted."""
    neighbors, motion, quality = map(np.asarray, (neighbors, motion, quality))
    n = len(neighbors)
    if (arm not in ARMS or neighbors.shape != (n, 83) or motion.shape != (n, 7, 10)
            or quality.shape != (n, 7, 5) or not np.isfinite(motion).all()
            or not np.isfinite(quality).all() or np.any((quality < 0) | (quality > 1))):
        raise ValueError('Registered finite past summary schema required')
    fields = [pool_context(neighbors)]
    if arm != 'neighbors':
        fields.append(quality.reshape(n, -1))
    if arm == 'magnitude':
        fields.append(motion[:, :, 6:].reshape(n, -1))
    if arm == 'directed':
        fields.append(motion.reshape(n, -1))
    return np.concatenate(fields, axis=1)


def join_rows(stationary_rows, all_rows, geometry, targets, labels):
    """Validate identity and labels separately; returned indices are not features."""
    keys = [(r['recording'], r['agent'], r['frame']) for r in all_rows]
    lookup = {key: i for i, key in enumerate(keys)}
    if len(lookup) != len(keys):
        raise ValueError('Duplicate full-cohort query')
    selected = [(r['recording_id'], r['agent_id'], r['frame_id']) for r in stationary_rows]
    if len(set(selected)) != len(selected) or any(key not in lookup for key in selected):
        raise ValueError('Ambiguous stationary/full-cohort join')
    ids = np.asarray([lookup[key] for key in selected], dtype=np.int64)
    static = np.all(np.asarray(geometry)[:, :16] == 0, axis=1)
    if not np.array_equal(np.sort(ids), np.flatnonzero(static)):
        raise ValueError('Stationary cohort changed or rows lost')
    expected = np.any(np.asarray(targets)[ids] != 0, axis=(1, 2)).astype(int)
    if not np.array_equal(expected, labels):
        raise ValueError('Cached supervisory labels disagree')
    for r, i in zip(stationary_rows, ids):
        full = all_rows[i]
        if (r['data_role'] != 'fit' or r['fit_fold'] != full['fold']
                or r['physical_scene'] != full['scene']):
            raise ValueError('Role/scene/fold mismatch')
    return ids


def group_brier(y, p, reference, keys):
    """Equal group weighting is descriptive, not a guarantee of independence."""
    y, p, reference, keys = map(np.asarray, (y, p, reference, keys))
    if not len(y) or any(a.shape != y.shape for a in (p, reference, keys)):
        raise ValueError('Aligned nonempty row vectors required')
    model = np.array([np.mean((p[keys == g] - y[keys == g])**2) for g in np.unique(keys)])
    ref = np.array([np.mean((reference[keys == g] - y[keys == g])**2) for g in np.unique(keys)])
    return dict(groups=len(model), brier=float(model.mean()), reference_brier=float(ref.mean()),
                brier_lift=float((ref-model).mean()), independence_established=False)


def paired_agent_interval(y, p, reference, keys, resamples=2000):
    """Conditional held-agent bootstrap, not a new-scene generalization interval."""
    y, p, reference, keys = map(np.asarray, (y, p, reference, keys))
    if p.ndim == 1:
        p = p[None]
    if reference.ndim == 1:
        reference = reference[None]
    if (p.ndim != 2 or reference.shape != p.shape or p.shape[1] != len(y)
            or keys.shape != y.shape or not len(y)):
        raise ValueError('Aligned seed-by-row probabilities required')
    # Average losses across seeds, not probabilities (which would make an ensemble).
    gain = ((reference-y)**2-(p-y)**2).mean(0)
    groups = np.unique(keys)
    values = np.array([gain[keys == g].mean() for g in groups])
    rng = np.random.default_rng(20260917)
    draws = values[rng.integers(len(values), size=(resamples, len(values)))].mean(1)
    return dict(agent_balanced_lift=float(values.mean()),
                descriptive_ci95=np.quantile(draws, [.025, .975]).tolist(),
                resamples=resamples, agents=len(groups),
                scope='fixed_fitted_models_conditional_held_agents_not_new_scenes',
                independent_confirmation=False)
