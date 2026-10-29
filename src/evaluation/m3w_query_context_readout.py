"""Complete matched-information comparison without opening independent roles."""
import numpy as np

from src.evaluation import m3w_rational_harm_readout as parent
from scripts.report_m3w_shared_train_risk import check_policy, equal_number

metrics = parent.metrics
CONTROLS = parent.ARMS + ('self',)
ARMS = CONTROLS + ('peer',)


def evaluate(args, predictions, eligible):
    x, env, y, _, sites, recs, frames, pr = args
    if set(predictions) != set(ARMS): raise ValueError('All ten comparator arms required')
    for arm in ('self', 'peer'):
        np.testing.assert_array_equal(predictions[arm][:, [0, 2, 3]], predictions['anchor'][:, [0, 2, 3]])
    selected = {a: parent.actions(p, eligible) for a, p in predictions.items()}
    known = np.isfinite(y).all(1); den = float(y[known, 2].sum())
    scores = {a: metrics.previous.score(p, y, pr, sites, recs, frames,
        np.ones(len(x), bool))['conditional_MSE'] for a, p in predictions.items()}
    contrasts, pairs, matched = {}, {}, {}
    for other in CONTROLS:
        left, right = metrics.previous.policy.matched(selected[other], selected['peer'],
            predictions[other][:, 0]-predictions[other][:, 1],
            predictions['peer'][:, 0]-predictions['peer'][:, 1], recs, frames, np.arange(len(y)))
        matched[other] = {other: metrics.policy_score(y, left, env), 'peer': metrics.policy_score(y, right, env)}
        contrasts[other+'_MSE'] = scores['peer']-scores[other] if scores['peer'] is not None and scores[other] is not None else None
        for mode, a, b in (('full', selected[other], selected['peer']), ('matched', left, right)):
            pair = metrics.previous.paired_completion(y, a, b, env)
            pairs[other+'_'+mode] = pair
            contrasts[other+'_'+mode] = 100*pair['lower_mass']/den if den > 0 else None
    return dict(rows=len(y), unknown_rows=int((~known).sum()), scores=scores,
        policies={a: metrics.policy_score(y, v, env) for a, v in selected.items()},
        contrasts=contrasts, pairs=pairs, matched=matched, full_known_reference_mass=den,
        protected_outputs_exact=True, inference_label_filter=False,
        recordings={str(r): {a: metrics.policy_score(y[recs == r], v[recs == r], env[recs == r])
                            for a, v in selected.items()} for r in np.unique(recs)},
        selected_diagnostics={a: metrics.selected_diagnostics(p, y, selected[a], pr['scale'])
                              for a, p in predictions.items()})


def validate(r):
    if set(r['policies']) != set(ARMS) or r['inference_label_filter'] or not r['protected_outputs_exact']:
        raise ValueError('Complete causal readout required')
    for a in ARMS:
        check_policy(r['policies'][a])
        for rec in r['recordings'].values(): check_policy(rec[a])
        equal_number(r['policies'][a]['selected_count'], sum(v[a]['selected_count'] for v in r['recordings'].values()))
    for other in CONTROLS:
        for p in r['matched'][other].values(): check_policy(p)
        equal_number(r['matched'][other][other]['selected_count'], r['matched'][other]['peer']['selected_count'])
        for mode in ('full', 'matched'):
            pair = r['pairs'][other+'_'+mode]; den = r['full_known_reference_mass']
            equal_number(pair['lower_mass'], pair['known_difference_mass']-pair['unknown_exchanged_envelope_mass'])
            equal_number(r['contrasts'][other+'_'+mode], 100*pair['lower_mass']/den if den > 0 else None)


def summarize(rows, expected, cfg):
    keys = [(r['context'], r['source'], r['seed']) for r in rows]
    if len(rows) != 72 or len(set(keys)) != 72 or set(keys) != set(expected):
        raise ValueError('All72 frozen views required')
    for r in rows: validate(r['result'])
    safety = {}
    for a in ARMS:
        pp = [r['result']['policies'][a] for r in rows]
        safety[a] = dict(selected=sum(v['selected_count'] for v in pp),
            unknown_selected=sum(v['selected_unknown'] for v in pp),
            known_easy_violations=sum(v['known_easy_positive_risk'] is not None and v['known_easy_positive_risk'] > .02+1e-12 for v in pp),
            upper_easy_violations=sum(v['easy_selected_risk_upper'] is not None and v['easy_selected_risk_upper'] > .02+1e-12 for v in pp),
            empty=sum(v['selected_count'] == 0 for v in pp),
            undefined_easy=sum(v['easy_selected_risk_upper'] is None for v in pp),
            full_support=sum(v['finite_completion_supported'] for v in pp))
    contrasts = {k: metrics.previous.interval(rows, k, cfg['bootstrap_draws'], cfg['bootstrap_seed'])
                 for k in rows[0]['result']['contrasts']}
    failed = [a+'_'+m+'_not_positive' for a in CONTROLS for m in ('full', 'matched')
              if contrasts[a+'_'+m]['CI95'] is None or contrasts[a+'_'+m]['CI95'][0] <= 0]
    if safety['peer']['full_support'] != 72: failed.append('full_policy_support_failed')
    return dict(fresh_fits=36, new_optimizer_steps=72000, views=72, source_localities=12,
        safety=safety, contrasts=contrasts, failures=failed, advance=not failed,
        bootstrap_draws=cfg['bootstrap_draws'], nominal_exposed_TRAIN_only=True,
        head_seeds_are_not_forecaster_seeds=True, independent_confirmation=False,
        outer_development_read=False, independent_roles_read=False, deployment_changed=False,
        new_forecaster=False, stage5c_executed=False, smc_enabled=False)
