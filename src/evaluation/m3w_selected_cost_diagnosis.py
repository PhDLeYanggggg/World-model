"""Frozen-prediction diagnostics; labels never construct an inference decision."""
import numpy as np

MOMENTS = ('benefit', 'harm', 'reference', 'easy_reference', 'easy_harm')


def subset_summary(prediction, target, mask, scales):
    p, y, use, scales = map(np.asarray, (prediction, target, mask, scales))
    if (p.ndim != 3 or p.shape[-1] != 5 or y.shape != p.shape or use.shape != p.shape[:2]
            or use.dtype != bool or scales.shape != (len(p),) or (scales <= 0).any()
            or not np.isfinite(p).all() or (p < 0).any() or not np.isfinite(scales).all()):
        raise ValueError('Finite predictions, explicit action subset and positive frozen scales required')
    known = np.isfinite(y).all((1, 2))
    if not (known | np.isnan(y).all((1, 2))).all() or (y[known] < 0).any():
        raise ValueError('Whole-row unknown labels must remain unknown')
    observed = use & known[:, None]; unknown = use & ~known[:, None]
    count = int(observed.sum()); scale = np.broadcast_to(scales[:, None], use.shape)
    predicted = p[observed]; truth = y[observed]
    total_p = predicted.sum(0, dtype=np.float64); total_y = truth.sum(0, dtype=np.float64)
    normalized_error = (predicted-truth)/scale[observed, None]
    def ratio(a, b):
        return float(a/b) if b > 0 else None
    result = dict(action_views=int(use.sum()), row_views=int(use.any(1).sum()),
        known_action_views=count, unknown_action_views=int(unknown.sum()),
        predicted_known_mass=dict(zip(MOMENTS, total_p.tolist())),
        observed_known_mass=dict(zip(MOMENTS, total_y.tolist())),
        normalized_MSE=dict(zip(MOMENTS, np.square(normalized_error).mean(0).tolist())) if count else None,
        normalized_bias=dict(zip(MOMENTS, normalized_error.mean(0).tolist())) if count else None,
        known_harm_underprediction_mass=float(np.maximum(truth[:, 1]-predicted[:, 1], 0).sum()),
        known_reference_inflation_mass=float(np.maximum(predicted[:, 2]-truth[:, 2], 0).sum()),
        normalized_predicted_harm_near_zero_known=int((predicted[:, 1]/scale[observed] <= 1e-6).sum()),
        near_zero_predicted_harm_but_true_harm_positive=int(((predicted[:, 1]/scale[observed] <= 1e-6) & (truth[:, 1] > 0)).sum()),
        actual_harm_positive_known=int((truth[:, 1] > 0).sum()),
        predicted_all_risk_on_known=ratio(total_p[1], total_p[2]),
        observed_all_risk_on_known=ratio(total_y[1], total_y[2]),
        predicted_easy_risk_on_known=ratio(total_p[4], total_p[3]),
        observed_easy_risk_on_known=ratio(total_y[4], total_y[3]),
        reference_inflation_ratio=ratio(total_p[2], total_y[2]),
        easy_reference_inflation_ratio=ratio(total_p[3], total_y[3]),
        harm_prediction_ratio=ratio(total_p[1], total_y[1]),
        easy_harm_prediction_ratio=ratio(total_p[4], total_y[4]),
        diagnostic_not_full_completion_safety=True)
    for name, h, r in (('all', 1, 2), ('easy', 4, 3)):
        predicted_excess = float(total_p[h]-.02*total_p[r])
        actual_excess = float(total_y[h]-.02*total_y[r])
        harm_part = float(total_y[h]-total_p[h])
        reference_part = float(.02*(total_p[r]-total_y[r]))
        if not np.isclose(actual_excess-predicted_excess, harm_part+reference_part, rtol=1e-10, atol=1e-8):
            raise ValueError('Risk-excess decomposition must be an exact accounting identity')
        result[name+'_excess_decomposition'] = dict(predicted_excess=predicted_excess,
            observed_excess=actual_excess, harm_underprediction=harm_part,
            reference_inflation=reference_part, identity_error=actual_excess-predicted_excess-harm_part-reference_part)
    return result


def audit_record(prediction, target, allowed, choices, scales, actions):
    """Compare a frozen legal bank, its passing gate and its one-action choice."""
    p, allowed, choices = np.asarray(prediction), np.asarray(allowed), np.asarray(choices)
    if allowed.dtype != bool or allowed.shape != p.shape[:2] or choices.shape != (len(p),):
        raise ValueError('Aligned frozen legal actions and choices required')
    legal = allowed.copy(); legal[:, sorted(set(range(7))-set(actions))] = False
    selected = np.zeros_like(legal); rows = np.flatnonzero(choices >= 0)
    if rows.size and (not np.isin(choices[rows], actions).all() or not legal[rows, choices[rows]].all()):
        raise ValueError('Frozen choices must be actual legal bank actions')
    selected[rows, choices[rows]] = True
    passing = (legal & (p[..., 0] > p[..., 1]) & (p[..., 2] > 0) & (p[..., 3] > 0)
               & (p[..., 1] <= .02*p[..., 2]) & (p[..., 4] <= .02*p[..., 3]))
    if (selected & ~passing).any():
        raise ValueError('Primary selected actions must satisfy the original predicted gate')
    return dict(all_legal=subset_summary(p, target, legal, scales),
        gate_passing=subset_summary(p, target, passing, scales),
        selected=subset_summary(p, target, selected, scales),
        choice_counts={str(k):int((choices == k).sum()) for k in range(-1, 7)},
        label_aware_selection=False, empty_risk='not_estimable_not_safe_zero')
