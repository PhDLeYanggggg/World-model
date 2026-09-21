"""Descriptive in-sample diagnostics, never a gate or calibration estimator."""
from __future__ import annotations

import numpy as np


def conditional_cost_summary(targets, predictions, mask):
    y, p, mask = np.asarray(targets, dtype=np.float64), np.asarray(predictions, dtype=np.float64), np.asarray(mask)
    if (y.ndim != 2 or y.shape[1] != 2 or p.shape != y.shape or mask.shape != (len(y),)
            or mask.dtype != bool or not np.isfinite(y).all() or not np.isfinite(p).all()
            or np.any(y < 0) or np.any(p < 0)):
        raise ValueError('Aligned nonnegative benefit/harm arrays and Boolean membership required')
    if not mask.any():
        return dict(rows=0, target_benefit_mean=None, target_harm_mean=None,
                    predicted_benefit_mean=None, predicted_harm_mean=None,
                    benefit_mse=None, harm_mse=None, actual_net_gain_mean=None,
                    predicted_net_gain_mean=None, positive_harm_fraction=None)
    y, p = y[mask], p[mask]
    return dict(rows=len(y), target_benefit_mean=float(y[:, 0].mean()), target_harm_mean=float(y[:, 1].mean()),
        predicted_benefit_mean=float(p[:, 0].mean()), predicted_harm_mean=float(p[:, 1].mean()),
        benefit_mse=float(np.mean((p[:, 0]-y[:, 0])**2)), harm_mse=float(np.mean((p[:, 1]-y[:, 1])**2)),
        actual_net_gain_mean=float(np.mean(y[:, 0]-y[:, 1])),
        predicted_net_gain_mean=float(np.mean(p[:, 0]-p[:, 1])),
        positive_harm_fraction=float(np.mean(y[:, 1] > 0)))


def diagnose_fit(targets, predictions, recordings, policies, *, raw_ridge=None, top_fraction=.01):
    y, p = np.asarray(targets), np.asarray(predictions)
    names = np.asarray(recordings)
    all_rows = np.ones(len(y), dtype=bool)
    whole = conditional_cost_summary(y, p, all_rows)
    if (not len(y) or names.shape != (len(y),) or any(not isinstance(n, str) or not n for n in recordings)
            or not np.isfinite(top_fraction) or not 0 < top_fraction <= 1):
        raise ValueError('Nonempty identified fit rows and valid descriptive fraction required')
    if np.any((y[:, 0] > 0) & (y[:, 1] > 0)):
        raise ValueError('Realized benefit and harm must not both be positive for one sample')
    masks = dict(all=all_rows)
    if raw_ridge is not None:
        raw = np.asarray(raw_ridge)
        if raw.shape != y.shape or not np.isfinite(raw).all() or not np.array_equal(np.maximum(raw, 0), p):
            raise ValueError('Raw ridge predictions must exactly reconstruct saved readout')
        masks['ridge_harm_clipped_to_zero'] = raw[:, 1] < 0
        masks['ridge_benefit_clipped_to_zero'] = raw[:, 0] < 0
    for name, policy in policies.items():
        gain_floor, harm_cap = policy['min_predicted_gain'], policy['max_agent_predicted_harm']
        if not np.isfinite([gain_floor, harm_cap]).all() or min(gain_floor, harm_cap) < 0:
            raise ValueError('Fixed finite nonnegative eligibility rules required')
        masks['eligible_'+name] = (p[:, 0]-p[:, 1] >= gain_floor) & (p[:, 1] <= harm_cap)
    conditional = {k: conditional_cost_summary(y, p, m) for k, m in masks.items()}
    per_recording = {name: {k: conditional_cost_summary(y, p, m & (names == name)) for k, m in masks.items()}
                     for name in sorted(set(recordings))}
    yd = y.astype(np.float64)
    constant = np.broadcast_to(yd.mean(0), y.shape)
    k = max(1, int(np.ceil(top_fraction*len(y))))
    squared = np.sort(yd[:, 1]**2)
    return dict(full_fit=whole, conditional=conditional, per_recording=per_recording,
        mean_label_reference=conditional_cost_summary(y, constant, all_rows),
        top_harm_target_rows=k, top_harm_target_fraction=top_fraction,
        top_harm_squared_target_mass_fraction=float(squared[-k:].sum()/squared.sum()) if squared.sum() > 0 else None,
        cost_head_evaluation_role='in_sample_fit_not_held_out',
        per_agent_eligibility_is_not_scene_intervention=True, calibrated_risk=False, deployment_approved=False)
