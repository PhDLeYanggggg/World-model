"""Complete frozen comparator evaluation of one rational-mapped harm branch."""
import numpy as np

from src.world_model import m3w_scaled_risk_readout as metrics
from src.evaluation.m3w_full_range_harm_readout import ARMS as CONTROLS, actions
from scripts.report_m3w_shared_train_risk import check_policy, equal_number

TARGET = 'rational'
ARMS = CONTROLS+(TARGET,)


def evaluate(args, predictions, eligible):
    x, env, y, _, sites, recs, frames, pr = args
    if set(predictions) != set(ARMS):
        raise ValueError('All eight frozen policies required')
    np.testing.assert_array_equal(predictions[TARGET][:, [0, 2, 3]], predictions['anchor'][:, [0, 2, 3]])
    selected = {a: actions(p, eligible) for a, p in predictions.items()}
    policies = {a: metrics.policy_score(y, m, env) for a, m in selected.items()}
    scores = {a: metrics.previous.score(p, y, pr, sites, recs, frames, np.ones(len(x), bool))['conditional_MSE']
              for a, p in predictions.items()}
    known = np.isfinite(y).all(1)
    den = float(y[known, 2].sum())
    contrasts, pairs, matched = {}, {}, {}
    for other in CONTROLS:
        a, b = metrics.previous.policy.matched(selected[other], selected[TARGET],
            predictions[other][:, 0]-predictions[other][:, 1],
            predictions[TARGET][:, 0]-predictions[TARGET][:, 1], recs, frames, np.arange(len(x)))
        matched[other] = {other: metrics.policy_score(y, a, env), TARGET: metrics.policy_score(y, b, env)}
        contrasts[other+'_MSE'] = scores[TARGET]-scores[other] if scores[other] is not None and scores[TARGET] is not None else None
        for mode, left, right in (('full', selected[other], selected[TARGET]), ('matched', a, b)):
            pair = metrics.previous.paired_completion(y, left, right, env)
            pairs[other+'_'+mode] = pair
            contrasts[other+'_'+mode] = 100*pair['lower_mass']/den if den > 0 else None
    return dict(rows=len(x), unknown_rows=int((~known).sum()), policies=policies, scores=scores,
        contrasts=contrasts, pairs=pairs, matched=matched, full_known_reference_mass=den,
        protected_outputs_exact=True, inference_label_filter=False,
        recordings={str(r): {a: metrics.policy_score(y[recs == r], m[recs == r], env[recs == r])
                            for a, m in selected.items()} for r in np.unique(recs)})


def validate(r):
    if set(r['policies']) != set(ARMS) or r['inference_label_filter'] or not r['protected_outputs_exact']:
        raise ValueError('Complete protected causal comparison required')
    for a in ARMS:
        check_policy(r['policies'][a])
        for rec in r['recordings'].values():
            check_policy(rec[a])
        equal_number(r['policies'][a]['selected_count'], sum(v[a]['selected_count'] for v in r['recordings'].values()))
    for other in CONTROLS:
        aa = r['matched'][other]
        for p in aa.values():
            check_policy(p)
        if aa[other]['selected_count'] != aa[TARGET]['selected_count']:
            raise ValueError('Matched counts differ')
        for mode in ('full', 'matched'):
            pair = r['pairs'][other+'_'+mode]
            equal_number(pair['lower_mass'], pair['known_difference_mass']-pair['unknown_exchanged_envelope_mass'])
            den = r['full_known_reference_mass']
            equal_number(r['contrasts'][other+'_'+mode], 100*pair['lower_mass']/den if den > 0 else None)


def summarize(rows, expected, cfg):
    actual = [(r['scope'], r['context'], r['source'], r['seed']) for r in rows]
    if len(actual) != 144 or len(set(actual)) != 144 or set(actual) != set(expected):
        raise ValueError('All144 registered views required')
    out = dict(fresh_fits=18, new_optimizer_steps=36000, views=144, deployment_changed=False,
        independent_roles_read=False, outer_development_read=False, new_forecaster=False)
    for scope in ('fit', 'held'):
        group = [r for r in rows if r['scope'] == scope]
        for r in group:
            validate(r['result'])
        safety = {}
        for a in ARMS:
            pp = [r['result']['policies'][a] for r in group]
            safety[a] = dict(views=len(pp), selected=sum(v['selected_count'] for v in pp),
                known_easy_violations=sum(v['known_easy_positive_risk'] is not None and v['known_easy_positive_risk'] > .02+1e-12 for v in pp),
                upper_easy_violations=sum(v['easy_selected_risk_upper'] is not None and v['easy_selected_risk_upper'] > .02+1e-12 for v in pp),
                empty=sum(v['selected_count'] == 0 for v in pp),
                undefined_easy=sum(v['easy_selected_risk_upper'] is None for v in pp),
                full_support=sum(v['finite_completion_supported'] for v in pp))
        out[scope] = dict(safety=safety,
            mean_view_MSE={a: float(np.mean([r['result']['scores'][a] for r in group if r['result']['scores'][a] is not None])) for a in ARMS},
            contrasts={k: metrics.previous.interval(group, k, cfg['bootstrap_draws'], cfg['bootstrap_seed']) for k in group[0]['result']['contrasts']})
    failures = [a+'_'+mode+'_not_positive' for a in CONTROLS for mode in ('full', 'matched')
                if out['held']['contrasts'][a+'_'+mode]['CI95'] is None or out['held']['contrasts'][a+'_'+mode]['CI95'][0] <= 0]
    if out['held']['safety'][TARGET]['full_support'] != 72:
        failures.append('full_policy_support_failed')
    out.update(failures=failures, advance=not failures, bootstrap_draws=cfg['bootstrap_draws'],
               inference_status='nominal_exposed_TRAIN_internal_not_confirmation')
    return out
