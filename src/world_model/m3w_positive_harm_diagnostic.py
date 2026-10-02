"""Frozen causal inference and offline algebraic cost-error attribution."""
import numpy as np
from src.world_model import m3w_positive_harm as model

forest = model.base.forest
MOMENTS = ('benefit', 'harm', 'reference', 'easy_reference', 'easy_harm')
SCORES = ('utility', 'all_risk_excess', 'easy_risk_excess')
LINEAR = np.array([[1., -1., 0., 0., 0.], [0., 1., -.02, 0., 0.], [0., 0., 0., -.02, 1.]])


def attribution(old, new, target, scale):
    a, b, y, scale = [np.asarray(v, float) for v in (old, new, target, scale)]
    if a.shape != b.shape or a.shape != y.shape or a.ndim != 2 or a.shape[1] != 5:
        raise ValueError('Five aligned finite known moments required')
    if not all(np.isfinite(v).all() for v in (a, b, y, scale)) or scale.shape != (3,) or (scale <= 0).any():
        raise ValueError('Known finite targets and fixed positive source score scales required')
    jacobian = LINEAR/scale[:, None]
    ea, eb = (a-y)@jacobian.T, (b-y)@jacobian.T
    score = (eb*eb-ea*ea)/3
    moment = (b-a)*((ea+eb)@jacobian)/3
    np.testing.assert_allclose(score.sum(1), moment.sum(1), atol=1e-9, rtol=1e-9)
    return dict(scores=score, moments=moment, old_MSE=np.mean(ea*ea, 1), new_MSE=np.mean(eb*eb, 1))


def leaf_support(leaves, weight, harm):
    leaves, w, y = np.asarray(leaves), np.asarray(weight, float), np.asarray(harm, float)
    if y.shape != (len(w), 2) or len(leaves) != len(w) or not len(w) or (w <= 0).any() or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError('Known positive-weight TRAIN rows required')
    nodes, inv = np.unique(leaves, return_inverse=True); count = len(nodes)
    totals, eff = [], []
    for j in range(2):
        event = y[:, j] > 0
        mass = np.bincount(inv, weights=w*event, minlength=count)
        square = np.bincount(inv, weights=w*w*event, minlength=count)
        totals.append(np.bincount(inv, weights=event.astype(int), minlength=count))
        eff.append(np.divide(mass*mass, square, out=np.zeros(count), where=square > 0))
    return dict(nodes=nodes, harm_count=np.column_stack(totals), effective_harm=np.column_stack(eff))


def training_support(state, x, envelope, target, sites, recordings, frames):
    pr = state['preprocess']; z, _ = forest.causal_inputs(x, envelope, pr)
    known = np.isfinite(target).all(1); w, _ = forest.core.weights(sites, recordings, frames, known)
    inputs = dict(features=forest.fingerprint(z[known]),
        targets=forest.fingerprint(forest.transformed_targets(target[known], pr)),
        weights=forest.fingerprint(w[known]), known_mask=forest.fingerprint(known))
    assert inputs == state['input_hashes']
    return [leaf_support(t.apply(z[known]), w[known], target[known][:, model.HARM])
            for t in state['model'].estimators_]


def raw_positive(state, fitted, x, envelope, past_quality, train_support):
    pr = state['preprocess']; z, support = forest.causal_inputs(x, envelope, pr)
    q = model.base.standardize(past_quality, fitted['quality_mean'], fitted['quality_std'])
    state['model'].set_params(n_jobs=1)
    old = state['model'].predict(z)/forest.FACTORS*pr['rms']*pr['scale']
    positive = np.zeros((len(x), 8)); descriptor = np.zeros((len(x), 6))
    assert len(train_support) == len(state['model'].estimators_)
    for j, tree in enumerate(state['model'].estimators_):
        start, end = fitted['offsets'][j:j+2]; nodes = fitted['nodes'][start:end]
        leaves = tree.apply(z); at = np.searchsorted(nodes, leaves)
        np.testing.assert_array_equal(nodes[at], leaves)
        np.testing.assert_array_equal(nodes, train_support[j]['nodes'])
        descriptor[:, :2] += train_support[j]['effective_harm'][at]
        descriptor[:, 2:4] += train_support[j]['effective_harm'][at] < 5
        values = tree.tree_.value[leaves, :, 0].copy()
        descriptor[:, 4:6] += values[:, model.HARM]/forest.FACTORS[list(model.HARM)]
        at += start
        score = np.einsum('ni,nji->nj', q-fitted['means'][at], fitted['coef'][at])-fitted['log_normalizer'][at]
        values[:, model.HARM] *= np.exp(np.clip(score, -60, 60))
        positive += values
    trees = len(state['model'].estimators_); descriptor /= trees
    positive = (positive/trees)/forest.FACTORS*pr['rms']*pr['scale']
    np.testing.assert_array_equal(positive[:, (0, 2, 3)], old[:, (0, 2, 3)])
    return old[:, :5], positive[:, :5], support, descriptor


def error_slice(old, new, target, weights, preprocess, mask):
    y, w, mask = np.asarray(target), np.asarray(weights), np.asarray(mask)
    known = np.isfinite(y).all(1); at = mask & known; mass = float(w[at].sum())
    d = attribution(old[at], new[at], y[at], preprocess['scale']*preprocess['rms'][5:])
    row = dict(rows=int(mask.sum()), known_rows=int(at.sum()), unknown_rows=int((mask & ~known).sum()),
        known_query_weight_mass=mass, global_weighted_old_MSE=float(w[at]@d['old_MSE']),
        global_weighted_new_MSE=float(w[at]@d['new_MSE']),
        score_contributions=(w[at]@d['scores']).tolist(),
        moment_contributions=(w[at]@d['moments']).tolist())
    row['global_weighted_MSE_change'] = row['global_weighted_new_MSE']-row['global_weighted_old_MSE']
    np.testing.assert_allclose(sum(row['moment_contributions']), row['global_weighted_MSE_change'], atol=1e-9, rtol=1e-9)
    row['conditional_MSE_change'] = row['global_weighted_MSE_change']/mass if mass > 0 else None
    return row


def diagnose(state, raw, projected, y, envelope, actions, sites, recordings, frames, descriptor):
    pr = state['preprocess']; known = np.isfinite(y).all(1)
    weights, _ = forest.core.weights(sites, recordings, frames, known)
    old, new = actions['original'], actions['positive']
    masks = dict(all=np.ones(len(y), bool), original_selected=old, original_unselected=~old,
        positive_selected=new, positive_unselected=~new, added=new & ~old, removed=old & ~new,
        retained=new & old, original_matched=actions['original_matched_positive'],
        positive_matched=actions['positive_matched_original'])
    strata = {}
    for channel, index in (('harm', 1), ('easy_harm', 4)):
        j = 0 if index == 1 else 1
        ratio = np.divide(raw['positive'][:, index], raw['original'][:, index],
                          out=np.full(len(y), np.nan), where=raw['original'][:, index] > 0)
        edges = [-np.inf, .5, 1., 2., 4., np.inf]
        for low, high, label in zip(edges[:-1], edges[1:], ('le_half', 'half_to_one', 'one_to_two', 'two_to_four', 'over_four')):
            strata[channel+'_rate_'+label] = (ratio > low) & (ratio <= high)
        strata[channel+'_rate_undefined'] = ~np.isfinite(ratio)
        strata[channel+'_majority_train_eff_lt5'] = descriptor[:, j+2] >= .5
        strata[channel+'_majority_train_eff_ge5'] = descriptor[:, j+2] < .5
        strata[channel+'_mean_train_magnitude_le_0p1'] = descriptor[:, j+4] <= .1
        strata[channel+'_mean_train_magnitude_gt_0p1'] = descriptor[:, j+4] > .1
    compare = lambda a, b, mask: error_slice(a, b, y, weights, pr, mask)
    out = dict(cohorts={k: compare(projected['original'], projected['positive'], m) for k, m in masks.items()},
        strata={k: compare(projected['original'], projected['positive'], m) for k, m in strata.items()},
        raw_error_change=compare(raw['original'], raw['positive'], masks['all']),
        additive_error_change=compare(projected['original'], projected['additive'], masks['all']),
        projection_effects={k: compare(raw[k], projected[k], masks['all']) for k in raw})
    # Hold the learned delta fixed; contrast known-label over/underprediction offline.
    error = (projected['positive']-y)/(pr['scale']*pr['rms'][:5])
    for label, index in (('harm', 1), ('easy_harm', 4)):
        for direction, mask in (('over', error[:, index] > 0), ('under_or_equal', error[:, index] <= 0)):
            out['strata'][label+'_label_'+direction] = compare(projected['original'], projected['positive'], known & mask)
    full = out['cohorts']['all']
    for prefix in ('original', 'positive'):
        np.testing.assert_allclose(full['global_weighted_MSE_change'],
            out['cohorts'][prefix+'_selected']['global_weighted_MSE_change']+
            out['cohorts'][prefix+'_unselected']['global_weighted_MSE_change'], atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(full['global_weighted_MSE_change'], out['raw_error_change']['global_weighted_MSE_change']+
        out['projection_effects']['positive']['global_weighted_MSE_change']-
        out['projection_effects']['original']['global_weighted_MSE_change'], atol=1e-9, rtol=1e-9)
    out['descriptor_mean'] = np.mean(descriptor, 0).tolist()
    return out
