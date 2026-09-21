"""Matched native-coordinate errors; no cross-unit pooling or epsilon gains."""
from __future__ import annotations

import numpy as np


def native_errors(prediction, target, valid, native_scale):
    p, y = np.asarray(prediction, float), np.asarray(target, float)
    mask, scale = np.asarray(valid), np.asarray(native_scale, float)
    if (p.ndim != 3 or p.shape[-1] != 2 or p.shape[1] < 1 or y.shape != p.shape
            or mask.shape != p.shape[:2] or mask.dtype != bool
            or scale.shape != (len(p),) or not np.isfinite(p).all()
            or not np.isfinite(y[mask]).all() or not np.isfinite(scale).all()
            or np.any(scale <= 0)):
        raise ValueError('Finite paired predictions, masked labels and positive causal scales required')
    # Translation and rotation cancel in Euclidean error; undo only causal scale.
    distance = np.linalg.norm(p-np.where(mask[..., None], y, 0), axis=-1)*scale[:, None]
    count = mask.sum(1)
    ade = np.divide(np.where(mask, distance, 0).sum(1), count,
                    out=np.full(len(p), np.nan), where=count > 0)
    return ade, np.where(mask[:, -1], distance[:, -1], np.nan)


def paired_scene_metrics(model, reference, scenes, *, expected_scenes, dataset,
                         coordinate_unit, row_units=None, bootstrap_resamples=0, seed=38113):
    m, r, s = np.asarray(model, float), np.asarray(reference, float), np.asarray(scenes)
    roster = tuple(expected_scenes)
    if (m.ndim != 1 or r.shape != m.shape or s.shape != m.shape or not roster
            or len(set(roster)) != len(roster) or not set(s).issubset(roster)
            or not isinstance(dataset, str) or not dataset
            or not isinstance(coordinate_unit, str) or not coordinate_unit):
        raise ValueError('One dataset/unit, aligned costs and a fixed unique scene roster required')
    if row_units is not None:
        units = np.asarray(row_units)
        if units.shape != m.shape or np.any(units != coordinate_unit):
            raise ValueError('Unlike native units cannot be pooled')
    supported = np.isfinite(m) & np.isfinite(r)
    if (not np.array_equal(np.isnan(m), np.isnan(r)) or np.isinf(m).any()
            or np.isinf(r).any() or np.any(m[supported] < 0) or np.any(r[supported] < 0)):
        raise ValueError('Identical label support and finite nonnegative paired costs required')
    rows, gains = {}, []
    for scene in roster:
        use = supported & (s == scene)
        if not use.any():
            rows[scene] = dict(rows=0, status='no_supported_labels', model_error=None,
                               reference_error=None, absolute_harm=None, gain_percent=None)
            continue
        mm, rr = float(m[use].mean()), float(r[use].mean())
        gain = None if rr == 0 else float(100*(1-mm/rr))
        rows[scene] = dict(rows=int(use.sum()), model_error=mm, reference_error=rr,
            absolute_harm=mm-rr, gain_percent=gain,
            model_p95=float(np.quantile(m[use], .95)), model_p99=float(np.quantile(m[use], .99)),
            reference_p95=float(np.quantile(r[use], .95)),
            status='zero_reference_percentage_undefined' if gain is None else 'defined')
        if gain is not None:
            gains.append(gain)
    all_defined = len(gains) == len(roster)
    ci = None
    if bootstrap_resamples < 0 or int(bootstrap_resamples) != bootstrap_resamples:
        raise ValueError('Nonnegative integer bootstrap count required')
    if all_defined and len(roster) >= 2 and bootstrap_resamples:
        draws = np.random.default_rng(seed).choice(gains, size=(bootstrap_resamples, len(roster)))
        ci = np.quantile(draws.mean(1), [.025, .975]).tolist()
    return dict(dataset=dataset, coordinate_unit=coordinate_unit, indexed_rows=len(m),
        supported_rows=int(supported.sum()), unknown_rows=int((~supported).sum()),
        expected_scenes=list(roster), by_scene=rows,
        equal_scene_gain_percent=float(np.mean(gains)) if all_defined else None,
        worst_scene_gain_percent=float(min(gains)) if all_defined else None,
        scene_bootstrap_ci95=ci, bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=seed, bootstrap_unit='physical_scene',
        uncertainty_limit='conditional_exploratory_not_independent_confirmation')


def select_complement_baseline(costs, scenes, expected_scenes, *, reference_index=1):
    """Select using other source sites only; never optimize on the held scene."""
    e, s = np.asarray(costs, float), np.asarray(scenes)
    roster = tuple(expected_scenes)
    if (e.ndim != 2 or s.shape != (len(e),) or len(roster) < 2
            or len(set(roster)) != len(roster) or set(s) != set(roster)
            or not 0 <= reference_index < e.shape[1]
            or np.isinf(e).any() or np.any(e[np.isfinite(e)] < 0)
            or np.any(np.isnan(e) != np.isnan(e[:, :1]))):
        raise ValueError('Fixed roster and paired baseline support required')
    chosen, receipt = np.zeros(len(e), np.int64), {}
    for held in roster:
        train_sites = [v for v in roster if v != held]
        risks = []
        for site in train_sites:
            use = (s == site) & np.isfinite(e).all(1)
            if not use.any():
                raise ValueError('Missing complement training support')
            mean = e[use].mean(0)
            if mean[reference_index] <= 0:
                raise ValueError('Zero training reference; percentage selection undefined')
            risks.append(mean/mean[reference_index])
        score = np.mean(risks, axis=0)
        k = int(np.argmin(score))
        chosen[s == held] = k
        receipt[held] = dict(selection_scenes=train_sites, selected_index=k,
                             training_mean_relative_risk=score.tolist())
    return chosen, receipt
