"""Unchanged policy budgets, full and count-matched utility, fixed controls."""
import statistics
import numpy as np
from src.world_model import m3w_scaled_risk_readout as metrics
from src.evaluation.m3w_easy_constrained_policy import budget_status

POLICIES = ('floor', 'origin', 'stochastic', 'exact_dual', 'exact_gradient')
PAIRS = (('origin', 'exact_gradient'), ('stochastic', 'exact_dual'),
         ('stochastic', 'exact_gradient'), ('exact_dual', 'exact_gradient'))


def evaluate(args, actions, ranks):
    _, env, y, _, _, rec, frame, _ = args
    if set(actions) != set(POLICIES) or set(ranks) != set(POLICIES):
        raise ValueError('Every fixed policy required')
    known = np.isfinite(y).all(1); den = float(y[known, 2].sum())
    policies = {k: metrics.policy_score(y, a, env) for k, a in actions.items()}
    recordings = {str(r): {k: metrics.policy_score(y[rec == r], a[rec == r], env[rec == r])
        for k, a in actions.items()} for r in np.unique(rec)}
    pairs = {}
    for left, right in PAIRS:
        aa, bb = metrics.previous.policy.matched(actions[left], actions[right], ranks[left], ranks[right],
                                               rec, frame, np.arange(len(y)))
        if aa.sum() != bb.sum(): raise AssertionError('Count matching failed')
        item = dict(matched_count=int(aa.sum()), matched_empty=not bool(aa.any()),
            matched_risk_role='utility_only_not_safety_gate', empty_risk_status='not_estimable',
            matched_left=metrics.policy_score(y, aa, env), matched_right=metrics.policy_score(y, bb, env))
        for mode, a, b in (('full', actions[left], actions[right]), ('matched', aa, bb)):
            pair = metrics.previous.paired_completion(y, a, b, env)
            item[mode] = dict(**pair, lower_pp=100*pair['lower_mass']/den if den > 0 else None,
                             upper_pp=100*pair['upper_mass']/den if den > 0 else None)
        pairs[right+'_minus_'+left] = item
    return dict(rows=len(y), known=int(known.sum()), unknown=int((~known).sum()),
                full_known_reference_mass=den, policies=policies, recordings=recordings,
                comparisons=pairs, inference_label_filter=False)


def aggregate(rows):
    safety, comparisons = {}, {}
    for arm in POLICIES:
        ps = [r['result']['policies'][arm] for r in rows]
        rr = [p[arm] for r in rows for p in r['result']['recordings'].values()]
        safety[arm] = dict(selected=sum(p['selected_count'] for p in ps),
            unknown_selected=sum(p['selected_unknown'] for p in ps),
            budget_views=sum(budget_status(p)['supported'] for p in ps), required_views=len(ps),
            budget_record_views=sum(budget_status(p)['supported'] for p in rr), required_record_views=len(rr),
            known_all_violations=sum(p['known_positive_risk'] is not None and p['known_positive_risk'] > .02+1e-12 for p in ps),
            known_easy_violations=sum(p['known_easy_positive_risk'] is not None and p['known_easy_positive_risk'] > .02+1e-12 for p in ps),
            easy_degradation_violations=sum(p['easy_degradation_upper'] is not None and p['easy_degradation_upper'] > .02+1e-12 for p in ps),
            undefined_easy_views=sum(p['easy_selected_risk_upper'] is None for p in ps),
            empty_views=sum(p['selected_count'] == 0 for p in ps))
    for left, right in PAIRS:
        name = right+'_minus_'+left; ps = [r['result']['comparisons'][name] for r in rows]
        item = dict(empty_matched_views=sum(p['matched_empty'] for p in ps))
        for mode in ('full', 'matched'):
            for bound in ('lower_pp', 'upper_pp'):
                values = [p[mode][bound] for p in ps]
                item[mode+'_'+bound] = statistics.fmean(values) if values and None not in values else None
        comparisons[name] = item
    return dict(views=len(rows), rows=sum(r['result']['rows'] for r in rows),
                unknown=sum(r['result']['unknown'] for r in rows), safety=safety, comparisons=comparisons)


def summarize(rows):
    scopes = {}
    for scope in ('fit', 'inner_held'):
        part = [r for r in rows if r['scope'] == scope]
        if len(part) != 72 or len({(r['context'], r['source'], r['seed']) for r in part}) != 72:
            raise ValueError('Complete fixed 72-view cohort required')
        scopes[scope] = aggregate(part)
    p = scopes['inner_held']['safety']['exact_gradient']
    return dict(scopes=scopes, primary='exact_gradient', risk_MSE_gate=False,
        primary_full_budget_supported=p['budget_views'] == 72 and p['budget_record_views'] == p['required_record_views'],
        independent_confirmation=False, whole_pipeline_OOF=False, deployment_changed=False,
        stage5c_executed=False, smc_enabled=False)
