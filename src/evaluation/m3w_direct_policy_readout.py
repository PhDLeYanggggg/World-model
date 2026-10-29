"""Full-policy budgets and count-matched utility for frozen direct gates."""
import statistics
import numpy as np
from src.world_model import m3w_scaled_risk_readout as metrics
from src.world_model.m3w_direct_policy import ARMS
from src.evaluation.m3w_easy_constrained_policy import budget_status

POLICIES = ('floor', 'origin')+ARMS
PAIRS = (('origin', 'soft_source'), ('origin', 'soft_record'), ('origin', 'hard_record'),
         ('soft_source', 'soft_record'), ('soft_record', 'hard_record'))


def evaluate(args, actions, ranks):
    _, env, y, _, _, recs, frames, _ = args
    if set(actions) != set(POLICIES) or set(ranks) != set(POLICIES):
        raise ValueError('Every frozen policy and rank required')
    known = np.isfinite(y).all(1); den = float(y[known, 2].sum())
    policies = {a: metrics.policy_score(y, mask, env) for a, mask in actions.items()}
    records = {str(r): {a: metrics.policy_score(y[recs == r], mask[recs == r], env[recs == r])
        for a, mask in actions.items()} for r in np.unique(recs)}
    comparisons = {}
    for left, right in PAIRS:
        aa, bb = metrics.previous.policy.matched(actions[left], actions[right], ranks[left], ranks[right],
                                               recs, frames, np.arange(len(y)))
        item = dict(matched_count=int(aa.sum()), matched_empty=not bool(aa.any()),
            matched_risk_role='not_a_safety_gate', matched_risk_empty_status='not_estimable',
            matched_left=metrics.policy_score(y, aa, env), matched_right=metrics.policy_score(y, bb, env))
        if aa.sum() != bb.sum(): raise AssertionError('Count matching failed')
        for mode, a, b in (('full', actions[left], actions[right]), ('matched', aa, bb)):
            pair = metrics.previous.paired_completion(y, a, b, env)
            item[mode] = dict(**pair, lower_pp=100*pair['lower_mass']/den if den > 0 else None,
                             upper_pp=100*pair['upper_mass']/den if den > 0 else None)
        comparisons[right+'_minus_'+left] = item
    return dict(rows=len(y), known=int(known.sum()), unknown=int((~known).sum()),
                full_known_reference_mass=den, policies=policies, recordings=records,
                comparisons=comparisons, inference_label_filter=False)


def summarize(rows):
    scopes = {}
    for scope in ('fit', 'inner_held'):
        selected = [r for r in rows if r['scope'] == scope]
        if len(selected) != 72 or len({(r['context'], r['source'], r['seed']) for r in selected}) != 72:
            raise ValueError('Complete 72-view roster required for each scope')
        safety = {}
        for a in POLICIES:
            ps = [r['result']['policies'][a] for r in selected]
            records = [v[a] for r in selected for v in r['result']['recordings'].values()]
            safety[a] = dict(selected=sum(p['selected_count'] for p in ps),
                unknown_selected=sum(p['selected_unknown'] for p in ps),
                budget_views=sum(budget_status(p)['supported'] for p in ps), required_views=len(ps),
                budget_record_views=sum(budget_status(p)['supported'] for p in records), required_record_views=len(records),
                known_all_violations=sum(p['known_positive_risk'] is not None and p['known_positive_risk'] > .02+1e-12 for p in ps),
                known_easy_violations=sum(p['known_easy_positive_risk'] is not None and p['known_easy_positive_risk'] > .02+1e-12 for p in ps),
                undefined_easy_views=sum(p['easy_selected_risk_upper'] is None for p in ps),
                empty_views=sum(p['selected_count'] == 0 for p in ps))
        contrasts = {}
        for left, right in PAIRS:
            name = right+'_minus_'+left
            contrast = dict(empty_matched_views=sum(r['result']['comparisons'][name]['matched_empty'] for r in selected))
            for mode in ('full', 'matched'):
                for bound in ('lower_pp', 'upper_pp'):
                    vals = [r['result']['comparisons'][name][mode][bound] for r in selected]
                    contrast[mode+'_'+bound] = statistics.fmean(vals) if None not in vals else None
            contrasts[name] = contrast
        scopes[scope] = dict(views=len(selected), rows=sum(r['result']['rows'] for r in selected),
            unknown=sum(r['result']['unknown'] for r in selected), safety=safety, contrasts=contrasts)
    primary = scopes['inner_held']['safety']['hard_record']
    return dict(scopes=scopes, primary='hard_record', risk_MSE_gate=False,
        primary_full_roster_budget_supported=primary['budget_views'] == 72 and
            primary['budget_record_views'] == primary['required_record_views'],
        causal_trajectory_producer_changed=False, independent_confirmation=False,
        whole_pipeline_OOF=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)
