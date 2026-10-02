"""Train-only terminal-value refit with fixed tree routing and reference costs."""
import numpy as np

from src.world_model import m3w_source_forest as forest
from src.world_model.m3w_component_calibration import eligible, matched
from src.world_model.m3w_unknown_outcome_bounds import completion_bounds
from src.world_model.m3w_forest_projection import interval


def fractions(y, env):
    y, env = np.asarray(y, float), np.asarray(env, float)
    if (y.shape != (len(env), 5) or not np.isfinite(y).all() or (y < 0).any()
            or not np.isfinite(env).all() or (env < 0).any()):
        raise ValueError('Known nonnegative training moments and causal envelope required')
    cost = y[:, [0, 1, 4]]
    if ((env == 0) & (cost.sum(1) > 0)).any():
        raise ValueError('Zero disagreement requires zero realized benefit and harm')
    f = np.divide(cost, env[:, None], out=np.zeros_like(cost), where=env[:, None] > 0)
    if (f[:, :2].sum(1) > 1+1e-5).any() or (f[:, 2] > f[:, 1]+1e-5).any():
        raise ValueError('Training targets exceed causal envelope beyond numerical tolerance')
    adjust = f[:, :2].sum(1) > 1
    f[adjust] /= f[adjust, :2].sum(1)[:, None]
    f[:, 2] = np.minimum(f[:, 2], f[:, 1])
    return f, int(adjust.sum())


def aggregate_leaves(leaves, y, env, weights, nodes):
    leaves, weights = np.asarray(leaves), np.asarray(weights, float)
    if (len(leaves) != len(y) or weights.shape != leaves.shape or not np.isfinite(weights).all()
            or (weights <= 0).any() or (leaves < 0).any() or (leaves >= nodes).any()):
        raise ValueError('Positive training weights and valid leaf indices required')
    f, clipped = fractions(y, env)
    mass = np.bincount(leaves, weights=weights, minlength=nodes)

    def avg(v):
        total = np.bincount(leaves, weights=weights*v, minlength=nodes)
        return np.divide(total, mass, out=np.zeros_like(total), where=mass > 0)

    original = np.column_stack([avg(y[:, j]) for j in range(5)])
    refit = np.column_stack([avg(f[:, j]) for j in range(3)])
    env_mean, env2 = avg(env), avg(env*env)
    env_var = np.maximum(env2-env_mean*env_mean, 0.)
    return dict(original=original, refit=refit, mass=mass, env_mean=env_mean,
                env_std=np.sqrt(env_var), target_roundoff_rows=clipped)


def fit(state, x, env, y, sites, recordings, frames):
    """Fit terminal values only. No validation labels accepted by this function."""
    pr = state['preprocess']
    fresh = forest.core.preprocess(x, env, y, sites, recordings, frames, training_site=pr['training_site'])
    forest.core.exact(pr, fresh)
    z, _ = forest.causal_inputs(x, env, pr)
    known = np.isfinite(y).all(1)
    w, _ = forest.core.weights(sites, recordings, frames, known)
    truth = forest.transformed_targets(y[known], pr)
    inputs = dict(features=forest.fingerprint(z[known]), targets=forest.fingerprint(truth),
        weights=forest.fingerprint(w[known]), known_mask=forest.fingerprint(known))
    assert inputs == state['input_hashes'], 'Training membership or targets differ from frozen forest'
    z, yy, ee, ww = z[known], y[known], env[known], w[known]
    values, offsets, stats = [], [0], []
    validated_leaves = 0
    for tree in state['model'].estimators_:
        leaves = tree.apply(z)
        a = aggregate_leaves(leaves, yy, ee, ww, tree.tree_.node_count)
        used = a['mass'] > 0
        original = tree.tree_.value[:, :, 0]/forest.FACTORS*pr['rms']*pr['scale']
        np.testing.assert_allclose(a['original'][used], original[used, :5], rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(a['mass'][used], tree.tree_.weighted_n_node_samples[used], rtol=2e-6, atol=1e-10)
        values.append(a['refit'])
        stats.append(np.column_stack((a['mass'], a['env_mean'], a['env_std'])))
        offsets.append(offsets[-1]+tree.tree_.node_count)
        validated_leaves += int(used.sum())
    return dict(values=np.concatenate(values), offsets=np.array(offsets, dtype=np.int64),
        stats=np.concatenate(stats), known_training_rows=int(known.sum()),
        unknown_training_rows=int((~known).sum()),
        training_target_roundoff_rows=fractions(yy, ee)[1],
        input_hashes=inputs, original_leaves_reconstructed=validated_leaves)


def predict(state, fitted, x, env):
    """Causal inference: reference moments retain the exact frozen implementation."""
    original, support = forest.predict(state, x, env)
    z, check = forest.causal_inputs(x, env, state['preprocess'])
    np.testing.assert_array_equal(support, check)
    n, ntree = len(x), len(state['model'].estimators_)
    f = np.zeros((n, 3))
    mean, std = np.zeros(n), np.zeros(n)
    for j, tree in enumerate(state['model'].estimators_):
        at = fitted['offsets'][j]+tree.apply(z)
        assert (fitted['stats'][at, 0] > 0).all(), 'Inference reached a leaf without known training support'
        f += fitted['values'][at]/ntree
        mean += fitted['stats'][at, 1]/ntree
        std += fitted['stats'][at, 2]/ntree
    if (f[:, :2].sum(1) > 1+1e-10).any() or (f[:, 2] > f[:, 1]+1e-10).any():
        raise ValueError('Refitted moments violate feasible fraction support')
    new = original.copy()
    new[:, 0] = np.minimum(env*f[:, 0], new[:, 2])
    new[:, 1] = env*f[:, 1]
    new[:, 4] = env*f[:, 2]
    np.testing.assert_array_equal(new[:, 2:4], original[:, 2:4])
    return original, new, support, dict(weighted_leaf_envelope_mean=mean, mean_leaf_envelope_std=std)


def evaluate(state, old, new, y, env, moving, support, sites, rec, frames, ids, diagnostic):
    actions = {k:eligible(p, moving, support) for k, p in [('original', old), ('refit', new)]}
    actions['original_matched'], actions['refit_matched'] = matched(actions['original'], actions['refit'],
        forest.core.signed(old)[:, 0], forest.core.signed(new)[:, 0], rec, frames, ids)
    _, inv = np.unique(np.rec.fromarrays([rec, frames]), return_inverse=True)
    np.testing.assert_array_equal(np.bincount(inv, weights=actions['original_matched']),
                                  np.bincount(inv, weights=actions['refit_matched']))
    bounds = {name:completion_bounds(y, a, env) for name, a in actions.items()}
    known = np.isfinite(y).all(1)
    denom = float(y[known, 2].sum())
    scores = {k:forest.signed_score(p, y, state['preprocess'], sites, rec, frames)
              for k, p in [('original', old), ('refit', new)]}
    contrasts = {}
    for suffix in ('', '_matched'):
        a, b = bounds['original'+suffix], bounds['refit'+suffix]
        contrasts['utility'+suffix+'_percent_full_known_reference'] = (
            100*(b['selected_net_gain_lower_mass']-a['selected_net_gain_lower_mass'])/denom if denom > 0 else None)
    diag = {}
    for k, mask in dict(all=np.ones(len(y), bool), original_selected=actions['original'],
                        original_known_unsafe=actions['original'] & known &
                            ((y[:, 1] > .02*y[:, 2]) | (y[:, 4] > .02*y[:, 3])),
                        removed=actions['original'] & ~actions['refit'],
                        added=actions['refit'] & ~actions['original']).items():
        em = diagnostic['weighted_leaf_envelope_mean']
        es = diagnostic['mean_leaf_envelope_std']
        m = mask & known
        diag[k] = dict(rows=int(mask.sum()), known=int(m.sum()),
            known_harmful=int((m & (y[:, 1] > 0)).sum()),
            query_envelope_sum=float(env[mask].sum()), leaf_mean_envelope_sum=float(em[mask].sum()),
            leaf_std_sum=float(es[mask].sum()),
            rows_leaf_mean_exceeds_query_envelope=int((mask & (em > env)).sum()))
    return dict(scores=scores, signed_MSE_change=scores['refit']-scores['original'],
        policies=bounds, contrasts=contrasts, leaf_scale_diagnostic=diag,
        full_known_reference_mass=denom, rows=len(y)), actions


def summarize(groups, cfg):
    out = dict(groups=len(groups), source_localities=len({g['source'] for g in groups}),
        leaf_refits=len(groups), new_tree_splits=0, new_neural_updates=0,
        independent_confirmation=False, transfer_evaluated=False, deployment_changed=False)
    for name in ('original','refit','original_matched','refit_matched'):
        r = [g['result']['policies'][name] for g in groups]
        risk = [v['easy_selected_risk_upper'] for v in r if v['easy_selected_risk_upper'] is not None]
        out[name] = dict(selected=sum(v['selected_count'] for v in r),
            unknown_selected=sum(v['selected_unknown'] for v in r),
            complete_support=sum(v['finite_completion_supported'] for v in r),
            defined_easy_risk=len(risk), violations=sum(v > .02+1e-12 for v in risk),
            worst_easy_upper=max(risk) if risk else None)
    for key in ('signed_MSE_change','utility_percent_full_known_reference','utility_matched_percent_full_known_reference'):
        values = [(g['source'], g['result'][key] if key == 'signed_MSE_change' else g['result']['contrasts'][key]) for g in groups]
        out[key] = interval(values, cfg['bootstrap_resamples'], cfg['bootstrap_seed'])
    out['advance_to_transfer'] = bool(out['signed_MSE_change']['CI95'][1] < 0
        and out['utility_percent_full_known_reference']['CI95'][0] > 0
        and out['utility_matched_percent_full_known_reference']['CI95'][0] > 0
        and out['refit']['complete_support'] >= out['original']['complete_support']
        and out['refit']['violations'] <= out['original']['violations']
        and out['refit']['worst_easy_upper'] is not None
        and out['refit']['worst_easy_upper'] <= out['original']['worst_easy_upper'])
    return out
