"""Four frozen decision overlays, same-count controls and unchanged support rules."""
import numpy as np

from src.world_model.m3w_policy_conditional_calibration import ARMS
from src.world_model import m3w_scaled_risk_readout as metrics
from scripts.report_m3w_shared_train_risk import check_policy, equal_number

COMPARATORS = ('raw', 'global', 'support_only')


def evaluate(y, env, recs, frames, prediction, policies):
    if set(policies) != set(ARMS):
        raise ValueError('All four registered policies required')
    if any((v & ~policies['raw']).any() for v in policies.values()):
        raise ValueError('Overlay expanded frozen policy')
    known = np.isfinite(y).all(1)
    den = float(y[known, 2].sum())
    score = prediction[:, 0]-prediction[:, 1]
    contrasts, pairs, matched = {}, {}, {}
    for other in COMPARATORS:
        a, b = metrics.previous.policy.matched(policies[other], policies['conditional'],
            score, score, recs, frames, np.arange(len(env)))
        matched[other] = {other: metrics.policy_score(y, a, env),
                         'conditional': metrics.policy_score(y, b, env)}
        for mode, left, right in (('full', policies[other], policies['conditional']), ('matched', a, b)):
            pair = metrics.previous.paired_completion(y, left, right, env)
            pairs[other+'_'+mode] = pair
            contrasts[other+'_'+mode] = 100*pair['lower_mass']/den if den > 0 else None
    return dict(rows=len(y), known_rows=int(known.sum()), unknown_rows=int((~known).sum()),
        full_known_reference_mass=den, contrasts=contrasts, pairs=pairs, matched=matched,
        policies={a: metrics.policy_score(y, v, env) for a, v in policies.items()},
        recordings={str(r): {a: metrics.policy_score(y[recs == r], v[recs == r], env[recs == r])
                            for a, v in policies.items()} for r in np.unique(recs)},
        inference_label_filter=False)


def validate(r):
    if set(r['policies']) != set(ARMS) or r['inference_label_filter']:
        raise ValueError('Incomplete or label-filtered readout')
    for a in ARMS:
        check_policy(r['policies'][a])
        for rr in r['recordings'].values():check_policy(rr[a])
        equal_number(r['policies'][a]['selected_count'],sum(v[a]['selected_count'] for v in r['recordings'].values()))
    for other in COMPARATORS:
        for p in r['matched'][other].values():check_policy(p)
        equal_number(r['matched'][other][other]['selected_count'],r['matched'][other]['conditional']['selected_count'])
        for mode in ('full','matched'):
            pair=r['pairs'][other+'_'+mode]; den=r['full_known_reference_mass']
            equal_number(pair['lower_mass'],pair['known_difference_mass']-pair['unknown_exchanged_envelope_mass'])
            equal_number(r['contrasts'][other+'_'+mode],100*pair['lower_mass']/den if den > 0 else None)


def summarize(rows, expected):
    keys=[(r['context'],r['source'],r['seed']) for r in rows]
    if len(rows)!=72 or len(set(keys))!=72 or set(keys)!=set(expected):
        raise ValueError('Complete72 unique cross-source readouts required')
    for r in rows:validate(r['result'])
    safety={}
    for a in ARMS:
        pp=[r['result']['policies'][a] for r in rows]
        safety[a]=dict(selected=sum(p['selected_count'] for p in pp),
            unknown_selected=sum(p['selected_unknown'] for p in pp),
            empty=sum(p['selected_count']==0 for p in pp),
            undefined_easy=sum(p['easy_selected_risk_upper'] is None for p in pp),
            known_easy_violations=sum(p['known_easy_positive_risk'] is not None and p['known_easy_positive_risk'] > .02+1e-12 for p in pp),
            upper_easy_violations=sum(p['easy_selected_risk_upper'] is not None and p['easy_selected_risk_upper'] > .02+1e-12 for p in pp),
            full_support=sum(p['finite_completion_supported'] for p in pp))
    contrasts={key:metrics.previous.interval(rows,key,3000,20261029) for key in rows[0]['result']['contrasts']}
    failures=[k+'_not_positive' for k,v in contrasts.items() if v['CI95'] is None or v['CI95'][0]<=0]
    if safety['conditional']['full_support']!=72:failures.append('full_policy_support_failed')
    return dict(views=72,source_localities=12,calibrators=72,neural_updates=0,arms=list(ARMS),
        safety=safety,contrasts=contrasts,failures=failures,advance=not failures,
        bootstrap_draws=3000,nominal_exposed_TRAIN_only=True,independent_confirmation=False,
        outer_development_read=False,independent_roles_read=False,deployment_changed=False,
        stage5c_executed=False,smc_enabled=False)
