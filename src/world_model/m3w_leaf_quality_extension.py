"""TRAIN-identity leaf-local quality clipping; not a calibrated risk bound."""
import numpy as np
from src.world_model import m3w_cost_support_diagnostic as diagnostic

forest = diagnostic.forest
base = diagnostic.model.positive.base
CONTROLS = ('original', 'additive', 'poisson', 'cost')


def training_bounds(state, fitted, x, envelope, targets, quality, sites, recordings, frames):
    tables = diagnostic.training_support(state, fitted, x, envelope, targets, quality, sites, recordings, frames)
    bounds = {key: np.concatenate([t[key] for t in tables]) for key in ('nodes', 'lower', 'upper')}
    np.testing.assert_array_equal(bounds['nodes'], fitted['nodes'])
    return bounds


def clip_quality(q, lower, upper):
    q, lo, hi = [np.asarray(v, float) for v in (q, lower, upper)]
    if q.ndim != 2 or q.shape[1] != 7 or lo.shape != q.shape or hi.shape != q.shape:
        raise ValueError('Aligned seven-dimensional row/leaf quality bounds required')
    if not all(np.isfinite(v).all() for v in (q, lo, hi)) or np.any(lo > hi):
        raise ValueError('Finite ordered TRAIN quality bounds required')
    return np.clip(q, lo, hi)


def predict_raw(state, fitted, bounds, x, envelope, quality):
    pr = state['preprocess']
    z, support = forest.causal_inputs(x, envelope, pr)
    q = base.standardize(quality, fitted['quality_mean'], fitted['quality_std'])
    np.testing.assert_array_equal(bounds['nodes'], fitted['nodes'])
    if len(q) != len(x):
        raise ValueError('Aligned past-only rows required')
    state['model'].set_params(n_jobs=1)
    original = state['model'].predict(z)/forest.FACTORS*pr['rms']*pr['scale']
    cost, extended = np.zeros((len(x), 8)), np.zeros((len(x), 8))
    changed = np.zeros(len(x), np.int64)
    for j, tree in enumerate(state['model'].estimators_):
        start, end = fitted['offsets'][j:j+2]
        leaves = tree.apply(z)
        at = np.searchsorted(fitted['nodes'][start:end], leaves)+start
        np.testing.assert_array_equal(fitted['nodes'][at], leaves)
        clipped = clip_quality(q, bounds['lower'][at], bounds['upper'][at])
        changed += np.any(clipped != q, 1)
        for features, total in ((q, cost), (clipped, extended)):
            values = tree.tree_.value[leaves, :, 0].copy()
            eta = np.einsum('ni,nji->nj', features-fitted['means'][at], fitted['coef'][at])-fitted['log_normalizer'][at]
            values[:, diagnostic.model.HARM] *= np.exp(np.clip(eta, -60, 60))
            total += values
    trees = len(state['model'].estimators_)
    cost = (cost/trees)/forest.FACTORS*pr['rms']*pr['scale']
    extended = (extended/trees)/forest.FACTORS*pr['rms']*pr['scale']
    for values in (cost, extended):
        np.testing.assert_array_equal(values[:, (0, 2, 3)], original[:, (0, 2, 3)])
    return dict(original=original[:, :5], cost=cost[:, :5], extended=extended[:, :5]), support, changed/trees


def evaluate(state, predictions, targets, envelope, moving, support, sites, recordings, frames, ids):
    if set(predictions) != set(CONTROLS+('extended',)):
        raise ValueError('All five frozen-comparison arms required')
    actions = {k: base.eligible(v, moving, support) for k, v in predictions.items()}
    _, query = np.unique(np.rec.fromarrays([recordings, frames]), return_inverse=True)
    for other in CONTROLS:
        a, b = base.matched(actions[other], actions['extended'], forest.core.signed(predictions[other])[:, 0],
                            forest.core.signed(predictions['extended'])[:, 0], recordings, frames, ids)
        np.testing.assert_array_equal(np.bincount(query, weights=a), np.bincount(query, weights=b))
        actions[other+'_matched_extended'], actions['extended_matched_'+other] = a, b
    policies = {k: base.completion_bounds(targets, a, envelope) for k, a in actions.items()}
    scores = {k: forest.signed_score(v, targets, state['preprocess'], sites, recordings, frames) for k, v in predictions.items()}
    den = float(targets[np.isfinite(targets).all(1), 2].sum())
    contrasts = {}
    for other in CONTROLS:
        contrasts['extended_minus_'+other+'_signed_MSE'] = scores['extended']-scores[other]
        for mode in ('full', 'matched'):
            a = policies[other if mode == 'full' else other+'_matched_extended']
            b = policies['extended' if mode == 'full' else 'extended_matched_'+other]
            contrasts['extended_minus_'+other+'_'+mode+'_utility_percent'] = (
                100*(b['selected_net_gain_lower_mass']-a['selected_net_gain_lower_mass'])/den if den > 0 else None)
    return dict(policies=policies, scores=scores, contrasts=contrasts, full_known_reference_mass=den), actions


def summarize(rows, cfg):
    out = dict(groups=len(rows), localities=len({r['source'] for r in rows}), new_training=False,
        threshold_search=False, independent_confirmation=False, transfer_evaluated=False, deployment_changed=False)
    for arm in rows[0]['result']['policies']:
        values = [r['result']['policies'][arm] for r in rows]
        risk = [v['easy_selected_risk_upper'] for v in values if v['easy_selected_risk_upper'] is not None]
        out[arm] = dict(selected=sum(v['selected_count'] for v in values),
            unknown_selected=sum(v['selected_unknown'] for v in values),
            complete_support=sum(v['finite_completion_supported'] for v in values), defined_easy_risk=len(risk),
            violations=sum(v > .02+1e-12 for v in risk), worst_easy_upper=max(risk) if risk else None,
            known_label_violations=sum(v['selected_known_easy_harm_mass']/v['selected_known_easy_reference_mass'] > .02+1e-12
                for v in values if v['selected_known_easy_reference_mass'] > 0))
    for name in rows[0]['result']['contrasts']:
        out[name] = base.interval([(r['source'], r['result']['contrasts'][name]) for r in rows],
                                  cfg['bootstrap_resamples'], cfg['bootstrap_seed'])
    reasons = []
    for other in CONTROLS:
        value = out['extended_minus_'+other+'_signed_MSE']['CI95']
        if value is None or value[1] >= 0:
            reasons.append('MSE_not_supported_vs_'+other)
        for mode in ('full', 'matched'):
            value = out['extended_minus_'+other+'_'+mode+'_utility_percent']['CI95']
            if value is None or value[0] <= 0:
                reasons.append(mode+'_utility_not_supported_vs_'+other)
    e, o = out['extended'], out['original']
    if e['complete_support'] < o['complete_support']:
        reasons.append('complete_support_reduced')
    if e['known_label_violations'] > o['known_label_violations']:
        reasons.append('more_known_risk_violations')
    if e['violations'] > o['violations']:
        reasons.append('more_upper_risk_violations')
    if e['worst_easy_upper'] is None or o['worst_easy_upper'] is None or e['worst_easy_upper'] > o['worst_easy_upper']:
        reasons.append('worst_upper_risk_not_preserved')
    out['advance_to_transfer'] = not reasons
    out['failure_reasons'] = reasons
    out['all_heads_easy_supported_within_budget'] = (
        e['complete_support'] == len(rows) and e['defined_easy_risk'] == len(rows) and e['violations'] == 0)
    return out
