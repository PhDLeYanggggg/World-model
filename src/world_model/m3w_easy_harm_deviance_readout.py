"""Fixed six-arm source-development comparison for the paired cost-loss trial."""
import numpy as np

from src.world_model import m3w_temporal_auxiliary_readout as previous

CONTROLS = previous.CONTROLS
TARGET = 'easy_deviance'
COMPARATORS = CONTROLS + ('quadratic',)
ARMS = COMPARATORS + (TARGET,)
PRIMARY_COMPARATORS = ('quadratic', 'original')


def decisions(predictions, moving, support, recordings, frames, ids):
    ids, rec, frame = map(np.asarray, (ids, recordings, frames))
    if (set(predictions) != set(ARMS) or ids.ndim != 1 or len(np.unique(ids)) != len(ids)
            or rec.shape != ids.shape or frame.shape != ids.shape):
        raise ValueError('All six frozen arms and unique aligned rows required')
    actions = {k: previous.policy.eligible(p, moving, support) for k,p in predictions.items()}
    if any(v.shape != ids.shape for v in actions.values()):
        raise ValueError('Aligned prediction and metadata rows required')
    for other in COMPARATORS:
        left, right = previous.policy.matched(actions[other], actions[TARGET],
            previous.forest.core.signed(predictions[other])[:, 0],
            previous.forest.core.signed(predictions[TARGET])[:, 0], rec, frame, ids)
        actions[other+'_matched_'+TARGET] = left
        actions[TARGET+'_matched_'+other] = right
    return actions


def policy_score(y, action, envelope):
    result = previous.completion_bounds(y, action, envelope)
    for group in ('', 'easy_'):
        h = result['selected_known_'+group+'harm_mass']
        r = result['selected_known_'+group+'reference_mass']
        result['known_'+group+'positive_risk'] = h/r if r > 0 else None
    result['switch_rate'] = result['selected_count']/len(y) if len(y) else None
    return result


def selected_diagnostics(prediction, y, action, scale):
    chosen = action & np.isfinite(y).all(1)
    pred, truth = prediction[chosen].sum(0), y[chosen].sum(0)
    harm = y[chosen, 1]/scale
    quantiles = np.quantile(harm, [.5,.95,.99]).tolist() if len(harm) else [None]*3
    return dict(known_selected=int(chosen.sum()), predicted_moments=pred.tolist(),
        observed_moments=truth.tolist(),
        predicted_observed_easy_harm_ratio=float(pred[4]/truth[4]) if truth[4] > 0 else None,
        positive_harm_over_frozen_TRAIN_scale=dict(p50=quantiles[0], p95=quantiles[1],
            p99=quantiles[2], maximum=float(harm.max()) if len(harm) else None),
        unknown_outcomes_imputed=False)


def evaluate(pr, predictions, targets, envelope, moving, support, sites, recordings, frames, ids):
    y, rec, env = np.asarray(targets, float), np.asarray(recordings), np.asarray(envelope, float)
    if any(np.asarray(v).shape != (len(y),) for v in (sites, rec, frames, ids)):
        raise ValueError('Aligned evaluation rows required')
    actions = decisions(predictions, moving, support, rec, frames, ids)
    policies = {k: policy_score(y, a, env) for k,a in actions.items()}
    known = np.isfinite(y).all(1)
    den = float(y[known, 2].sum())
    scores = {arm: {name: previous.score(p, y, pr, sites, rec, frames, mask)
        for name,mask in dict(all=np.ones(len(y), bool), original_selected=actions['original'],
                              own_selected=actions[arm]).items()} for arm,p in predictions.items()}
    pairs, contrasts = {}, {}
    for other in COMPARATORS:
        for cohort in ('all', 'original_selected'):
            a,b = [scores[k][cohort]['conditional_MSE'] for k in (other, TARGET)]
            contrasts[other+'_'+cohort+'_MSE'] = b-a if a is not None and b is not None else None
        for mode in ('full', 'matched'):
            a = other if mode == 'full' else other+'_matched_'+TARGET
            b = TARGET if mode == 'full' else TARGET+'_matched_'+other
            pair = previous.paired_completion(y, actions[a], actions[b], env)
            pairs[other+'_'+mode] = pair
            contrast = policies[b]['selected_net_gain_lower_mass']-policies[a]['selected_net_gain_lower_mass']
            contrasts[other+'_'+mode+'_lower_proxy_delta_percent'] = 100*contrast/den if den > 0 else None
            contrasts[other+'_'+mode+'_paired_lower_percent'] = 100*pair['lower_mass']/den if den > 0 else None
    recording_results = {}
    for recording in np.unique(rec):
        at = rec == recording
        recording_results[str(recording)] = {k: policy_score(y[at], actions[k][at], env[at]) for k in ARMS}
    diagnostics = {k: selected_diagnostics(predictions[k], y, actions[k], pr['scale']) for k in ARMS}
    return dict(rows=len(y), known_rows=int(known.sum()), unknown_rows=int((~known).sum()),
        full_known_reference_mass=den, scores=scores, policies=policies, pairs=pairs,
        contrasts=contrasts, recording_results=recording_results, diagnostics=diagnostics,
        inference_label_filter=False), actions


def summarize(rows, expected, cfg):
    keys = lambda records: [(r['group'], r['source'], r['head_seed']) for r in records]
    if len(set(keys(rows))) != len(rows) or sorted(keys(rows)) != sorted(keys(expected)):
        raise ValueError('All registered identities required exactly once')
    if len(rows) != cfg['source_heads'] or len({r['source'] for r in rows}) != 12:
        raise ValueError('Fixed 72-view/12-locality grid required')
    groups = {}
    for row in rows:
        groups.setdefault(row['group'], set()).add(row['head_seed'])
    if any(s != set(cfg['head_seeds']) for s in groups.values()) or cfg['risk_budget'] != .02:
        raise ValueError('All three seeds and unchanged risk budget required')
    contrasts = {key: previous.interval(rows, key, cfg['bootstrap_resamples'], cfg['bootstrap_seed'])
                 for key in rows[0]['result']['contrasts']}
    primary = [k+'_'+mode+'_paired_lower_percent' for k in PRIMARY_COMPARATORS for mode in ('full', 'matched')]
    failures = ['unsupported_'+k for k in primary
                if contrasts[k]['CI95'] is None or contrasts[k]['CI95'][0] <= 0]
    safety = {}
    for name in rows[0]['result']['policies']:
        values = [r['result']['policies'][name] for r in rows]
        upper = [v['easy_selected_risk_upper'] for v in values]
        known = [v['known_easy_positive_risk'] for v in values]
        safety[name] = dict(selected_occurrences=sum(v['selected_count'] for v in values),
            unknown_selected_occurrences=sum(v['selected_unknown'] for v in values),
            undefined_easy_risk=sum(v is None for v in upper),
            known_easy_violations=sum(v is not None and v > .02+1e-12 for v in known),
            easy_upper_violations=sum(v is not None and v > .02+1e-12 for v in upper),
            worst_easy_upper=max((v for v in upper if v is not None), default=None),
            finite_completion_supported=sum(v['finite_completion_supported'] for v in values))
    # Full policy and the two primary same-count policies must retain support;
    # an empty matched query/cohort is reported, not credited as a safety pass.
    required = [TARGET]+[TARGET+'_matched_'+k for k in PRIMARY_COMPARATORS]
    for name in required:
        if safety[name]['finite_completion_supported'] != len(rows):
            failures.append('incomplete_or_unsafe_completion_'+name)
        if safety[name]['known_easy_violations'] or safety[name]['easy_upper_violations']:
            failures.append('selected_easy_risk_violated_'+name)
    return dict(groups=len(rows), localities=12, trained_cost_heads=cfg['neural_fits'],
        contrasts=contrasts, primary_contrasts=primary, safety=safety,
        failure_reasons=failures, advance_to_transfer_design=not failures,
        secondary_MSE_not_an_advancement_gate=True, all_strong_controls_retained=True,
        bootstrap_resamples=cfg['bootstrap_resamples'],
        head_seeds_are_not_forecaster_seeds=True, independent_confirmation=False,
        deployment_changed=False, transfer_evaluated=False, stage5c_executed=False, smc_enabled=False)
