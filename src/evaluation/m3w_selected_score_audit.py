"""Label-aware score accounting for fixed selections, never inference features."""
import numpy as np


def audit(prediction, targets, scales, allowed, groups, initial, choice):
    p, y, scales = map(lambda v: np.asarray(v, float), (prediction, targets, scales))
    ok, initial, choice = map(np.asarray, (allowed, initial, choice)); n = len(p)
    if (n == 0 or p.shape != (n, 7, 5) or y.shape != p.shape
            or scales.shape != (n,) or ok.shape != (n, 7) or ok.dtype != bool
            or not np.isfinite(p).all() or (p < 0).any()
            or not np.isfinite(scales).all() or (scales <= 0).any()):
        raise ValueError('Aligned fixed costs,scales and causal availability required')
    known = np.isfinite(y).all((1, 2))
    if not (known | np.isnan(y).all((1, 2))).all() or (y[known] < 0).any():
        raise ValueError('Wholly unknown rows stay NaN; partial moment labels refuse')
    for c in (initial, choice):
        if c.shape != (n,) or not np.issubdtype(c.dtype, np.integer) or ((c < -1) | (c >= 7)).any():
            raise ValueError('Aligned legal fixed action indices required')
        at = np.flatnonzero(c >= 0)
        if not ok[at, c[at]].all(): raise ValueError('No unavailable selected action')
    if (not groups or any(np.asarray(g).ndim != 1 or not len(g)
            or not np.issubdtype(np.asarray(g).dtype, np.integer) for g in groups)
            or not np.array_equal(np.sort(np.concatenate(groups)), np.arange(n))):
        raise ValueError('Every causal row must belong to exactly one complete query')
    score, harm, reference = [], [], []
    observed_positive = np.zeros(2, int); false_safe = np.zeros(2, int); zero_ref = np.zeros(2, int)
    raw_truth = np.zeros((2, 2)); raw_pred = np.zeros((2, 2))
    empty = unknown = 0; max_error = 0.
    for g in groups:
        ix = np.asarray(g)[choice[g] >= 0]
        if not len(ix): empty += 1; continue
        if not known[ix].all(): unknown += 1; continue
        action = choice[ix]; truth, pred = y[ix, action], p[ix, action]; mass = scales[ix].sum()
        ht, hp = truth[:, [1, 4]].sum(0), pred[:, [1, 4]].sum(0)
        rt, rp = truth[:, [2, 3]].sum(0), pred[:, [2, 3]].sum(0)
        true_score = (truth[:, [1, 4]]-.02*truth[:, [2, 3]]).sum(0)/mass
        pred_score = (pred[:, [1, 4]]-.02*pred[:, [2, 3]]).sum(0)/mass
        delta_h, delta_r = (ht-hp)/mass, .02*(rp-rt)/mass
        error = true_score-pred_score
        if not np.allclose(error, delta_h+delta_r, rtol=1e-12, atol=1e-12):
            raise AssertionError('Signed-score decomposition must agree')
        max_error = max(max_error, float(np.max(np.abs(error-delta_h-delta_r))))
        score.append(error); harm.append(delta_h); reference.append(delta_r)
        observed_positive += true_score > 0
        false_safe += (pred_score <= 0) & (true_score > 0)
        zero_ref += rt == 0
        raw_truth += np.stack((ht, rt)); raw_pred += np.stack((hp, rp))
    mean = lambda values: np.mean(values, 0).tolist() if values else None
    return dict(rows=n, queries=len(groups), selected_rows=int((choice >= 0).sum()),
        selected_unknown_rows=int(((choice >= 0) & ~known).sum()),
        empty_selected_queries=empty, unknown_selected_queries=unknown, complete_selected_queries=len(score),
        changed_rows=int((choice != initial).sum()),
        introduced_rows=int(((initial < 0) & (choice >= 0)).sum()),
        removed_rows=int(((initial >= 0) & (choice < 0)).sum()),
        changed_action_rows=int(((initial >= 0) & (choice >= 0) & (choice != initial)).sum()),
        mean_signed_score_underprediction=mean(score), mean_harm_underprediction=mean(harm),
        mean_reference_bias_contribution=mean(reference),
        mean_score_squared_error=mean([v**2 for v in score]),
        known_observed_positive_excess_queries=observed_positive.tolist(),
        predicted_signed_safe_observed_positive_excess=false_safe.tolist(),
        zero_true_reference_complete_queries=zero_ref.tolist(),
        raw_truth_complete_query_harm_reference=raw_truth.tolist(),
        raw_prediction_complete_query_harm_reference=raw_pred.tolist(),
        max_abs_decomposition_error=max_error,
        missing_or_empty_imputed_as_zero=False, inference_feature=False, risk_certificate=False,
        different_selected_populations_not_causal_comparison=True)
