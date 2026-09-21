"""Fixed risk-score controls; future-defined groups are evaluation-only."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from src.evaluation.m3w_native_matched_coverage import top_count


def risk_decisions(gain_scores, risk_scores, same, current_stop, ids, anchor, cut=.01):
    scores = np.asarray(gain_scores)
    if (scores.ndim != 2 or scores.shape[1] != 2 or not np.isfinite(scores).all()
            or set(risk_scores) != {'all_harm', 'zero_reference_harm'} or not 0 <= cut <= 1):
        raise ValueError('Aligned finite gain and both fixed risk arms required')
    for mask in (same, current_stop, anchor):
        if np.asarray(mask).shape != (len(scores),) or np.asarray(mask).dtype != bool:
            raise ValueError('Aligned causal boolean masks required')
    for p in risk_scores.values():
        if p.shape != scores.shape or not np.isfinite(p).all() or np.any(p < 0) or np.any(p[:, 0] > 1):
            raise ValueError('Finite bounded event scores and nonnegative expected costs required')
    g = scores[:, 0]-scores[:, 1]
    eligible = (g > 0) & ~same
    pools = dict(net_only=eligible, stop_veto=eligible & ~current_stop,
        all_harm_guard=eligible & (risk_scores['all_harm'][:, 0] <= cut),
        zero_harm_guard=eligible & (risk_scores['zero_reference_harm'][:, 0] <= cut))
    requested = int(np.asarray(anchor, bool).sum())
    common = min(requested, *(int(v.sum()) for v in pools.values()))
    decisions = {k+'_online':v for k,v in pools.items()}
    decisions.update({k+'_matched':top_count(g, v, ids, common) for k,v in pools.items()})
    return decisions, dict(requested_count=requested, common_count=common,
        capacity={k:int(v.sum()) for k,v in pools.items()},
        risk_score_cut=cut, score_cut_is_calibrated_risk_bound=False)


def event_quality(probability, target, support):
    p, y = np.asarray(probability)[support], np.asarray(target)[support]
    if not len(p):
        return dict(rows=0, positive_rows=0, brier=None, auroc=None, auprc=None, ece=None)
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1) or not np.isin(y, [0, 1]).all():
        raise ValueError('Supported binary targets and bounded predictions required')
    bins = np.minimum((p*10).astype(int), 9)
    rows, ece = [], 0.
    for k in range(10):
        use = bins == k
        if use.any():
            predicted, observed = float(p[use].mean()), float(y[use].mean())
            ece += use.mean()*abs(predicted-observed)
            rows.append(dict(bin=k, rows=int(use.sum()), predicted=predicted, observed=observed))
    low = p <= .01
    return dict(rows=len(p), positive_rows=int(y.sum()), positive_rate=float(y.mean()),
        brier=float(np.mean((p-y)**2)),
        auroc=float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
        auprc=float(average_precision_score(y, p)) if y.sum() else None,
        ece=float(ece), reliability=rows,
        score_le_0p01_rows=int(low.sum()), score_le_0p01_positive_rows=int(y[low].sum()),
        score_le_0p01_observed_rate=float(y[low].mean()) if low.any() else None,
        calibrated_on_independent_data=False)
