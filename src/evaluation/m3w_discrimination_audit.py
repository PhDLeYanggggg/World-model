"""Own-fitting ranking versus decision errors; no threshold selection."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from src.evaluation.m3w_decision_semantics import joint_target


def summarize(score, target, weights):
    score, target, weights = (np.asarray(a, dtype=np.float64) for a in (score, target, weights))
    if score.ndim != 1 or target.shape != score.shape or weights.shape != score.shape:
        raise ValueError('Aligned vectors required')
    if not all(np.isfinite(x).all() for x in (score, target, weights)) or (weights < 0).any():
        raise ValueError('Finite vectors and nonnegative weights required')
    # Zero targets are not beneficial examples; unknown outcomes never enter.
    use = (target != 0) & (weights > 0)
    s, y, w = score[use], target[use], weights[use]
    bad = y > 0
    support = len(set(bad.tolist())) == 2
    cost = np.abs(y)*w
    admitted = s <= 0
    positive, negative = cost[bad].sum(), cost[~bad].sum()
    q = [0., .1, .5, .9, 1.]
    quantiles = lambda z: None if not len(z) else dict(zip(map(str, q), map(float, np.quantile(z, q))))
    return dict(rows=len(score), nonzero_supported_rows=len(s), harmful_rows=int(bad.sum()),
        beneficial_rows=int((~bad).sum()), zero_target_rows=int((target == 0).sum()),
        query_weighted_AUROC=float(roc_auc_score(bad, s, sample_weight=w)) if support else None,
        cost_weighted_AUROC=float(roc_auc_score(bad, s, sample_weight=cost)) if support else None,
        query_weighted_AUPRC=float(average_precision_score(bad, s, sample_weight=w)) if support else None,
        positive_weight_fraction=float(w[bad].sum()/w.sum()) if len(s) else None,
        false_safe_count=int((bad & admitted).sum()),
        false_reject_count=int((~bad & ~admitted).sum()),
        harmful_mass_admitted_fraction=float(cost[bad & admitted].sum()/positive) if positive else None,
        beneficial_mass_rejected_fraction=float(cost[~bad & ~admitted].sum()/negative) if negative else None,
        score_harmful_quantiles=quantiles(s[bad]), score_beneficial_quantiles=quantiles(s[~bad]),
        threshold=0., scores_are_not_probabilities=True, no_threshold_search=True)


def audit(risk, decision, target, allowed, observed, weights, sites):
    r, d, t = (np.asarray(v) for v in (risk, decision, target))
    if r.shape != d.shape or r.shape != t.shape or r.ndim != 3 or r.shape[1:] != (7, 2):
        raise ValueError('Aligned seven-action/two-channel scores and labels required')
    if (allowed.shape != r.shape[:2] or allowed.sum(1).max(initial=0) > 1
            or observed.shape != (len(r),) or weights.shape != observed.shape or sites.shape != observed.shape):
        raise ValueError('Aligned frozen single proposals and metadata required')
    ix, actions = np.nonzero(allowed & observed[:, None])
    y = joint_target(t)[ix, actions]
    rr, dd = r[ix, actions].max(-1), d[ix, actions].max(-1)
    score = np.maximum(rr, dd)
    metrics = {key: summarize(value, y, weights[ix]) for key, value in
        (('risk', rr), ('decision', dd), ('combined', score))}
    per_source = {str(site): summarize(score[sites[ix] == site], y[sites[ix] == site],
        weights[ix][sites[ix] == site]) for site in sorted(set(sites[ix]))}
    per_action = {str(action): summarize(score[actions == action], y[actions == action],
        weights[ix][actions == action]) for action in sorted(set(actions))}
    safe = y < 0; unsafe = y > 0
    blocking = dict(risk_only=int((safe & (rr > 0) & (dd <= 0)).sum()),
        decision_only=int((safe & (dd > 0) & (rr <= 0)).sum()), both=int((safe & (rr > 0) & (dd > 0)).sum()))
    return dict(total=metrics, per_source=per_source, per_action=per_action,
        beneficial_rejection_blocker=blocking, unknown_outcomes_excluded=int((allowed.any(1) & ~observed).sum()),
        harmful_admitted_both=int((unsafe & (score <= 0)).sum()), own_fitting_only=True,
        independent_generalization=False, safety_certificate=False)
