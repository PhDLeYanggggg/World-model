"""TRAIN-only fitting comparison; never a deployment or held-data result."""
import statistics

from src.evaluation import m3w_train_feasibility as feasibility

np, metrics, budget_status = feasibility.np, feasibility.metrics, feasibility.budget_status
ARMS = ('origin', 'exact_gradient', 'cost_sensitive')


def evaluate(args, masks, ranks):
    if set(masks) != set(ARMS) or set(ranks) != set(ARMS): raise ValueError('Fixed three policies required')
    _, env, y, _, _, rec, frame, _ = args
    policies = {a:metrics.policy_score(y,masks[a],env) for a in ARMS}
    recordings = {str(r):{a:metrics.policy_score(y[rec==r],masks[a][rec==r],env[rec==r]) for a in ARMS}
                  for r in np.unique(rec)}
    den = float(y[np.isfinite(y).all(1),2].sum())
    pairs = {}
    for other in ARMS[:-1]:
        aa,bb = metrics.previous.policy.matched(masks[other],masks['cost_sensitive'],
            ranks[other],ranks['cost_sensitive'],rec,frame,np.arange(len(y)))
        if aa.sum()!=bb.sum(): raise AssertionError('Matched action counts differ')
        item = dict(matched_count=int(aa.sum()),matched_empty=not bool(aa.any()),
            matched_left=metrics.policy_score(y,aa,env),matched_right=metrics.policy_score(y,bb,env),
            matched_role='utility_only',empty_risk_status='not_estimable')
        for mode,a,b in (('full',masks[other],masks['cost_sensitive']),('matched',aa,bb)):
            p = metrics.previous.paired_completion(y,a,b,env)
            item[mode] = dict(**p,lower_pp=100*p['lower_mass']/den if den>0 else None,
                upper_pp=100*p['upper_mass']/den if den>0 else None)
        pairs['cost_sensitive_minus_'+other] = item
    return dict(rows=len(y),policies=policies,recordings=recordings,pairs=pairs,
                full_known_reference_mass=den,scope='fitting_only',deployment_changed=False)


def summarize(rows):
    if len(rows)!=72 or len({(r['context'],r['source'],r['seed']) for r in rows})!=72:
        raise ValueError('Complete fitting cohort required')
    safety = {}
    for arm in ARMS:
        ps = [r['result']['policies'][arm] for r in rows]
        rr = [v[arm] for r in rows for v in r['result']['recordings'].values()]
        safety[arm] = dict(budget_views=sum(budget_status(p)['supported'] for p in ps),required_views=len(ps),
            budget_record_views=sum(budget_status(p)['supported'] for p in rr),required_record_views=len(rr),
            selected=sum(p['selected_count'] for p in ps),unknown=sum(p['selected_unknown'] for p in ps),
            undefined_easy_views=sum(p['easy_selected_risk_upper'] is None for p in ps),
            known_all_violations=sum(p['known_positive_risk'] is not None and p['known_positive_risk']>.02+1e-12 for p in ps),
            known_easy_violations=sum(p['known_easy_positive_risk'] is not None and p['known_easy_positive_risk']>.02+1e-12 for p in ps),
            easy_degradation_violations=sum(p['easy_degradation_upper'] is not None and p['easy_degradation_upper']>.02+1e-12 for p in ps))
    pairs = {}
    for other in ARMS[:-1]:
        name = 'cost_sensitive_minus_'+other
        ps = [r['result']['pairs'][name] for r in rows]
        pairs[name] = dict(empty_matched_views=sum(p['matched_empty'] for p in ps))
        for mode in ('full','matched'):
            for bound in ('lower_pp','upper_pp'):
                v = [p[mode][bound] for p in ps]
                pairs[name][mode+'_'+bound] = statistics.fmean(v) if None not in v else None
    return dict(safety=safety,pairs=pairs,scope='fitting_only',inner_held_scored=False,
        independent_confirmation=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)
