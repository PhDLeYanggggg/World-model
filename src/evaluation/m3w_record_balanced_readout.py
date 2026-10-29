"""Fixed TRAIN comparison of utility weighting and constraint weighting."""
import statistics

from src.evaluation import m3w_train_feasibility as feasibility

np, metrics, budget_status = feasibility.np, feasibility.metrics, feasibility.budget_status
ARMS = ('origin','cost_sensitive','record_balanced')


def evaluate(args, masks, ranks):
    if set(masks) != set(ARMS) or set(ranks) != set(ARMS):
        raise ValueError('Fixed three-policy roster required')
    _,env,y,_,_,records,frames,_ = args
    policies = {a:metrics.policy_score(y,masks[a],env) for a in ARMS}
    recordings = {str(r):{a:metrics.policy_score(y[records==r],masks[a][records==r],env[records==r])
                          for a in ARMS} for r in np.unique(records)}
    den = float(y[np.isfinite(y).all(1),2].sum())
    pairs = {}
    for other in ARMS[:-1]:
        left,right = metrics.previous.policy.matched(masks[other],masks['record_balanced'],
            ranks[other],ranks['record_balanced'],records,frames,np.arange(len(y)))
        if left.sum() != right.sum(): raise AssertionError('Matched counts differ')
        pair = dict(matched_count=int(left.sum()),matched_empty=not bool(left.any()),
            matched_left=metrics.policy_score(y,left,env),matched_right=metrics.policy_score(y,right,env),
            matched_role='utility_only',empty_risk_status='not_estimable')
        for mode,a,b in (('full',masks[other],masks['record_balanced']),('matched',left,right)):
            value = metrics.previous.paired_completion(y,a,b,env)
            pair[mode] = dict(**value,lower_pp=100*value['lower_mass']/den if den>0 else None,
                             upper_pp=100*value['upper_mass']/den if den>0 else None)
        pairs['record_balanced_minus_'+other] = pair
    return dict(rows=len(y),policies=policies,recordings=recordings,pairs=pairs,
                full_known_reference_mass=den,scope='fitting_only',deployment_changed=False)


def summarize(rows):
    if len(rows)!=72 or len({(r['context'],r['source'],r['seed']) for r in rows})!=72:
        raise ValueError('All 72 frozen fitting views required')
    safety={}
    for arm in ARMS:
        ps=[r['result']['policies'][arm] for r in rows]
        rs=[v[arm] for r in rows for v in r['result']['recordings'].values()]
        safety[arm]=dict(budget_views=sum(budget_status(p)['supported'] for p in ps),required_views=len(ps),
            budget_record_views=sum(budget_status(p)['supported'] for p in rs),required_record_views=len(rs),
            selected=sum(p['selected_count'] for p in ps),unknown=sum(p['selected_unknown'] for p in ps),
            undefined_easy_record_views=sum(p['easy_selected_risk_upper'] is None for p in rs),
            known_all_record_violations=sum(p['known_positive_risk'] is not None and p['known_positive_risk']>.02+1e-12 for p in rs),
            known_easy_record_violations=sum(p['known_easy_positive_risk'] is not None and p['known_easy_positive_risk']>.02+1e-12 for p in rs),
            easy_degradation_record_violations=sum(p['easy_degradation_upper'] is not None and p['easy_degradation_upper']>.02+1e-12 for p in rs))
    pairs={}
    for other in ARMS[:-1]:
        name='record_balanced_minus_'+other;ps=[r['result']['pairs'][name] for r in rows]
        pairs[name]=dict(empty_matched_views=sum(p['matched_empty'] for p in ps))
        for mode in ('full','matched'):
            for bound in ('lower_pp','upper_pp'):
                values=[p[mode][bound] for p in ps]
                pairs[name][mode+'_'+bound]=statistics.fmean(values) if None not in values else None
    return dict(safety=safety,pairs=pairs,scope='fitting_only',inner_held_scored=False,
        independent_confirmation=False,deployment_changed=False,stage5c_executed=False,smc_enabled=False)
