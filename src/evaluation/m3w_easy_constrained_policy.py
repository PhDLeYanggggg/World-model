"""Future-policy risk and utility separation on a fixed TRAIN roster."""
import statistics
import numpy as np
from src.evaluation.m3w_selected_tail_audit import actions
from src.world_model import m3w_scaled_risk_readout as metrics
from src.evaluation.m3w_easy_constrained import ARMS, PRIMARY

COMPARATORS=('origin','parent_blend')


def budget_status(policy):
    checks={k:policy[k] is not None and np.isfinite(policy[k]) and policy[k]<=.02+1e-12
        and (k=='easy_degradation_upper' or policy[k]>=0)
        for k in ('all_selected_risk_upper','easy_selected_risk_upper','easy_degradation_upper')}
    return dict(conditions=checks,supported=all(checks.values()),
        status='within_budget' if all(checks.values()) else 'over_budget_or_not_estimable')


def evaluate(args,predictions,eligible):
    _,env,y,_,_,recs,frames,_=args
    if set(predictions)!=set(ARMS):raise ValueError('All six fixed policy arms required')
    chosen={a:actions(p,eligible) for a,p in predictions.items()}
    known=np.isfinite(y).all(1);den=float(y[known,2].sum())
    policies={a:metrics.policy_score(y,mask,env) for a,mask in chosen.items()}
    matched={};contrasts={};pairs={}
    for other in COMPARATORS:
        left,right=metrics.previous.policy.matched(chosen[other],chosen[PRIMARY],
            predictions[other][:,0]-predictions[other][:,1],predictions[PRIMARY][:,0]-predictions[PRIMARY][:,1],
            recs,frames,np.arange(len(y)))
        matched[other]=dict(left=metrics.policy_score(y,left,env),right=metrics.policy_score(y,right,env),
            count=int(left.sum()),risk_role='reported_not_a_safety_gate',
            empty=not bool(left.any()),empty_risk_status='not_estimable' if not left.any() else 'see_bounds')
        if left.sum()!=right.sum():raise AssertionError('Matched count changed')
        for mode,l,r in (('full',chosen[other],chosen[PRIMARY]),('matched',left,right)):
            pair=metrics.previous.paired_completion(y,l,r,env);pairs[other+'_'+mode]=pair
            contrasts[other+'_'+mode]=100*pair['lower_mass']/den if den>0 else None
    recordings={str(rec):{a:metrics.policy_score(y[recs==rec],mask[recs==rec],env[recs==rec])
        for a,mask in chosen.items()} for rec in np.unique(recs)}
    return dict(rows=len(y),known=int(known.sum()),unknown=int((~known).sum()),policies=policies,
        recordings=recordings,matched=matched,pairs=pairs,contrasts=contrasts,full_known_reference_mass=den,
        full_budget_status={a:budget_status(p) for a,p in policies.items()},
        inference_label_filter=False,matched_risk_required=False,population_safety_proven=False),chosen


def summarize(rows):
    results=[r['policy'] for r in rows];safety={}
    for a in ARMS:
        policies=[r['policies'][a] for r in results]
        records=[rec[a] for r in results for rec in r['recordings'].values()]
        safety[a]=dict(selected=sum(p['selected_count'] for p in policies),
            selected_unknown=sum(p['selected_unknown'] for p in policies),
            within_budget_views=sum(budget_status(p)['supported'] for p in policies),
            required_views=len(policies),within_budget_record_views=sum(budget_status(p)['supported'] for p in records),
            required_record_views=len(records),
            undefined_easy_views=sum(p['easy_selected_risk_upper'] is None for p in policies),
            known_easy_violations=sum(p['known_easy_positive_risk'] is not None and p['known_easy_positive_risk']>.02+1e-12 for p in policies),
            completion_utility_and_budget_supported_views=sum(p['finite_completion_supported'] for p in policies))
    contrasts={k:statistics.fmean(r['contrasts'][k] for r in results)
        if all(r['contrasts'][k] is not None for r in results) else None for k in results[0]['contrasts']}
    return dict(safety=safety,paired_completion_lower_utility_pp=contrasts,
        empty_matched_views={a:sum(r['matched'][a]['empty'] for r in results) for a in COMPARATORS},
        primary_full_roster_budget_supported=safety[PRIMARY]['within_budget_views']==len(rows) and
            safety[PRIMARY]['within_budget_record_views']==safety[PRIMARY]['required_record_views'],
        policy_utility_positive_vs_both=all(v is not None and v>0 for v in contrasts.values()),
        risk_MSE_not_a_policy_gate=True,matched_risk_is_not_safety_gate=True,
        formal_generalization_or_deployment_gate_pass=False,reason='exposed_TRAIN_only_independent_evaluation_not_run')
