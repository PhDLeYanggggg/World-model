"""Read-only transport diagnosis; future costs describe, never choose, an action."""
import numpy as np

from src.evaluation.m3w_selected_cost_diagnosis import audit_record, subset_summary


def neural_funnel(prediction, target, allowed, choices):
    p, y, allowed, choices = map(np.asarray, (prediction, target, allowed, choices))
    known = np.isfinite(y).all((1, 2))
    legal = allowed[:, 6]
    pred_gain = p[:, 6, 0]-p[:, 6, 1]
    true_gain = y[:, 6, 0]-y[:, 6, 1]
    passing = (legal & (pred_gain > 0) & (p[:, 6, 2] > 0) & (p[:, 6, 3] > 0)
        & (p[:, 6, 1] <= .02*p[:, 6, 2]) & (p[:, 6, 4] <= .02*p[:, 6, 3]))
    useful = legal & known & (true_gain > 0)
    buckets = dict(predicted_nonpositive_gain=useful & (pred_gain <= 0),
        predicted_positive_guard_rejected=useful & (pred_gain > 0) & ~passing,
        guard_passed_other_action_chosen=useful & passing & (choices != 6),
        neural_chosen=useful & (choices == 6))
    if not np.array_equal(sum(v.astype(int) for v in buckets.values()), useful.astype(int)):
        raise ValueError("Useful neural rows must partition without changing frozen decisions")
    return dict(legal_neural_rows=int(legal.sum()), known_neural_rows=int((legal&known).sum()),
        unknown_neural_rows=int((legal&~known).sum()), useful_known_neural_rows=int(useful.sum()),
        useful_known_neural_benefit_mass=float(true_gain[useful].sum()),
        buckets={name:dict(rows=int(take.sum()),known_benefit_mass=float(true_gain[take].sum())) for name,take in buckets.items()},
        label_aware_inference=False, gain_definition="observed_positive_ADE_gain_not_a_safety_certificate")


def diagnose(prediction, target, allowed, choices, scales):
    result = audit_record(prediction, target, allowed, choices, scales, tuple(range(7)))
    neural = np.zeros_like(allowed); neural[:, 6] = allowed[:, 6]
    result["neural_legal"] = subset_summary(prediction, target, neural, scales)
    result["neural_funnel"] = neural_funnel(prediction, target, allowed, choices)
    return result


def selected_summary(prediction, target, allowed, choices, scales):
    selected = np.zeros_like(allowed)
    ix = np.flatnonzero(choices >= 0)
    if not allowed[ix, choices[ix]].all(): raise ValueError("Only frozen legal choices admitted")
    selected[ix, choices[ix]] = True
    return subset_summary(prediction, target, selected, scales)


def feature_shift(x, preprocess, groups):
    """SOURCE normalization stays frozen; these quantities never alter inference."""
    counts = np.zeros(2, np.int64); total = np.zeros(2, np.int64)
    outside_rows = 0
    for lo in range(0, len(x), 1024):
        z = np.abs((x[lo:lo+1024]-preprocess["mean"])/preprocess["std"])
        clipped = z > preprocess["clip"]
        outside_rows += int(clipped.any((1, 2)).sum())
        for i, section in enumerate((slice(0,380), slice(380,444))):
            counts[i] += int(clipped[..., section].sum()); total[i] += clipped[..., section].size
    sizes = np.asarray([len(g) for g in groups])
    return dict(rows=len(x),rows_with_any_clipped_field=outside_rows,
        geometry_rollout_clipped_fraction=float(counts[0]/total[0]),detector_clipped_fraction=float(counts[1]/total[1]),
        query_counts=dict(singleton=int((sizes==1).sum()),two_to_four=int(((sizes>=2)&(sizes<=4)).sum()),
            five_to_nine=int(((sizes>=5)&(sizes<=9)).sum()),ten_or_more=int((sizes>=10).sum())),
        normalization_fitted=False,descriptive_only=True)
