"""Fixed fitting cohorts and explicit go/no-go; never deployment certification."""
import numpy as np
from src.evaluation.m3w_selected_tail_audit import actions
from src.evaluation import m3w_query_context_readout as old

ARMS = ('origin', 'coupled', 'estimation')
COHORTS = ('eligible', 'origin')


def score(prediction, truth, mask, scale):
    known = mask & np.isfinite(truth).all(1)
    result = dict(rows=int(mask.sum()), known=int(known.sum()), unknown=int((mask & ~known).sum()))
    for name, col, ref in [('all', 1, 2), ('easy', 4, 3)]:
        observed = (truth[known, col]-.02*truth[known, ref])/scale
        predicted = (prediction[known, col]-.02*prediction[known, ref])/scale
        result[name] = dict(MSE=float(np.mean((observed-predicted)**2)) if known.any() else None,
            observed_harm=float(truth[known, col].sum()/scale), predicted_harm=float(prediction[known, col].sum()/scale),
            observed_excess=float(observed.sum()), predicted_excess=float(predicted.sum()),
            false_safe_rows=int(((predicted <= 0) & (observed > 1e-12)).sum()))
    return result


def evaluate(args, predictions):
    if set(predictions) != set(ARMS): raise ValueError('All registered fixed controls required')
    x, env, y, _, _, recs, frames, pr = args
    from src.world_model.m3w_risk_decoupling import api
    eligible = api.parent.previous.prior.eligible(args)
    cohorts = dict(eligible=eligible, origin=actions(predictions['origin'], eligible))
    selected = {a: actions(p, eligible) for a, p in predictions.items()}
    for p in predictions.values():
        np.testing.assert_array_equal(p[:, [0, 2, 3]], predictions['origin'][:, [0, 2, 3]])
    denominator = float(y[np.isfinite(y).all(1), 2].sum())
    left, right = old.metrics.previous.policy.matched(selected['coupled'], selected['estimation'],
        predictions['coupled'][:, 0]-predictions['coupled'][:, 1],
        predictions['estimation'][:, 0]-predictions['estimation'][:, 1], recs, frames, np.arange(len(y)))
    full = old.metrics.previous.paired_completion(y, selected['coupled'], selected['estimation'], env)
    same = old.metrics.previous.paired_completion(y, left, right, env)
    return dict(cohorts={a: {c: score(p, y, m, pr['scale']) for c, m in cohorts.items()}
                        for a, p in predictions.items()},
        policies={a: old.metrics.policy_score(y, s, env) for a, s in selected.items()},
        common_cohort_hash=api.fingerprint(cohorts['origin']),
        full_utility_pp=100*full['lower_mass']/denominator if denominator > 0 else None,
        matched_utility_pp=100*same['lower_mass']/denominator if denominator > 0 else None,
        matched_counts=[int(left.sum()), int(right.sum())],
        protected_outputs_exact=True, inference_label_filter=False)


def summarize(rows, expected):
    keys = {(r['context'], r['source'], r['seed']) for r in rows}
    if len(rows) != 72 or len(keys) != 72 or keys != set(expected):
        raise ValueError('Complete72 fitting views required')
    means = {}; counts = {}
    for arm in ARMS:
        means[arm] = {}; counts[arm] = {}
        for cohort in COHORTS:
            entries = [r['result']['cohorts'][arm][cohort] for r in rows]
            counts[arm][cohort] = {k: sum(v[k] for v in entries) for k in ('rows', 'known', 'unknown')}
            means[arm][cohort] = {target: float(np.mean([v[target]['MSE'] for v in entries]))
                if all(v[target]['MSE'] is not None for v in entries) else None for target in ('all', 'easy')}
    guard = {}
    for reference in ('origin', 'coupled'):
        for cohort in COHORTS:
            for target in ('all', 'easy'):
                a, b = means['estimation'][cohort][target], means[reference][cohort][target]
                strict = cohort == 'origin' and target == 'easy'
                guard[f'{cohort}_{target}_vs_{reference}'] = bool(a is not None and b is not None
                    and (a < b-1e-12 if strict else a <= b+1e-12))
    pairs = {(r['context'], r['seed']) for r in rows}
    wins = 0
    for context, seed in pairs:
        group = [r for r in rows if (r['context'], r['seed']) == (context, seed)]
        values = {a: [r['result']['cohorts'][a]['origin']['easy']['MSE'] for r in group] for a in ARMS}
        if all(v is not None for vv in values.values() for v in vv):
            wins += np.mean(values['estimation']) < min(np.mean(values['origin']), np.mean(values['coupled']))-1e-12
    guard['at_least12_of18_easy_MSE_pair_wins'] = bool(wins >= 12)
    safety = {}
    for arm in ARMS:
        pp = [r['result']['policies'][arm] for r in rows]
        safety[arm] = dict(selected=sum(p['selected_count'] for p in pp),
            unknown_selected=sum(p['selected_unknown'] for p in pp),
            known_easy_violations=sum(p['known_easy_positive_risk'] is not None and p['known_easy_positive_risk'] > .02+1e-12 for p in pp),
            undefined_easy=sum(p['easy_selected_risk_upper'] is None for p in pp),
            full_support=sum(p['finite_completion_supported'] for p in pp))
    return dict(mean_view_MSE=means, cohort_counts=counts, easy_pair_wins=int(wins),
        fit_routing_conditions=guard, fit_eligible_for_later_held_design=all(guard.values()),
        failed_conditions=[k for k, v in guard.items() if not v], fitting_policy_diagnostics=safety,
        utility_diagnostics={k: float(np.mean([r['result'][k] for r in rows]))
            if all(r['result'][k] is not None for r in rows) else None for k in ('full_utility_pp', 'matched_utility_pp')},
        fitted_diagnostics_not_validation=True, independent_confirmation=False,
        held_inference=False, deployment_changed=False, stage5c_executed=False, smc_enabled=False)
