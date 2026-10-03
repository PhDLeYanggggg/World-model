"""Offline support and tree/ensemble diagnosis of immutable cost heads."""
import numpy as np
from src.world_model import m3w_positive_harm_diagnostic_v2 as prior
from src.world_model import m3w_cost_harm_newton as model

forest = prior.forest
DESCRIPTORS = ('zero_train_harm_fraction', 'zero_train_easy_harm_fraction',
               'outside_train_quality_box_fraction')


def quality_ranges(leaves, quality):
    nodes, inv = np.unique(leaves, return_inverse=True)
    q = np.asarray(quality, float)
    if q.shape != (len(leaves), 7) or not len(q) or not np.isfinite(q).all():
        raise ValueError('Seven finite known-TRAIN quality features required')
    lower = np.full((len(nodes), 7), np.inf)
    upper = np.full_like(lower, -np.inf)
    for j in range(7):
        np.minimum.at(lower[:, j], inv, q[:, j])
        np.maximum.at(upper[:, j], inv, q[:, j])
    return dict(nodes=nodes, lower=lower, upper=upper)


def training_support(state, fitted, x, envelope, target, quality, sites, recordings, frames):
    tables = prior.training_support(state, x, envelope, target, sites, recordings, frames)
    z, _ = forest.causal_inputs(x, envelope, state['preprocess'])
    known = np.isfinite(target).all(1)
    q = model.positive.base.standardize(quality, fitted['quality_mean'], fitted['quality_std'])
    for tree, table in zip(state['model'].estimators_, tables):
        bounds = quality_ranges(tree.apply(z[known]), q[known])
        np.testing.assert_array_equal(table['nodes'], bounds['nodes'])
        table.update(bounds)
    return tables


def raw_predictions(state, fitted, x, envelope, quality, tables):
    old, new, supported, previous = prior.raw_positive(state, fitted, x, envelope, quality, tables)
    z, _ = forest.causal_inputs(x, envelope, state['preprocess'])
    q = model.positive.base.standardize(quality, fitted['quality_mean'], fitted['quality_std'])
    extra = np.zeros((len(x), 3))
    for tree, table in zip(state['model'].estimators_, tables):
        leaves = tree.apply(z)
        at = np.searchsorted(table['nodes'], leaves)
        np.testing.assert_array_equal(table['nodes'][at], leaves)
        extra[:, :2] += table['harm_count'][at] == 0
        extra[:, 2] += np.any((q < table['lower'][at]-1e-12) | (q > table['upper'][at]+1e-12), 1)
    extra /= len(tables)
    return old, new, supported, previous, extra


def tree_score_diagnostic(state, fitted, x, envelope, quality, targets, sites, recordings, frames):
    """Labels occur only in this offline diagnostic, never in raw_predictions."""
    pr = state['preprocess']
    known = np.isfinite(targets).all(1)
    weights, _ = forest.core.weights(sites, recordings, frames, known)
    w, y = weights[known], targets[known]
    z, _ = forest.causal_inputs(x, envelope, pr)
    z = z[known]
    q = model.positive.base.standardize(quality, fitted['quality_mean'], fitted['quality_std'])[known]
    sum_predictions = [np.zeros((len(y), 3)), np.zeros((len(y), 3))]
    tree_mse = np.zeros(2)
    squared_predictions = np.zeros(2)
    penalty = 0.
    score_target = prior.signed(y)/float(pr['scale'])/pr['rms'][5:].astype(float)
    for j, tree in enumerate(state['model'].estimators_):
        start, end = fitted['offsets'][j:j+2]
        leaves = tree.apply(z)
        at = np.searchsorted(fitted['nodes'][start:end], leaves)+start
        np.testing.assert_array_equal(fitted['nodes'][at], leaves)
        old = tree.tree_.value[leaves, :, 0].copy()
        new = old.copy()
        eta = np.einsum('ni,nji->nj', q-fitted['means'][at], fitted['coef'][at])-fitted['log_normalizer'][at]
        new[:, model.HARM] *= np.exp(np.clip(eta, -60, 60))
        for arm, values in enumerate((old, new)):
            moments = values/forest.FACTORS*pr['rms']*pr['scale']
            scores = prior.signed(moments[:, :5])/float(pr['scale'])/pr['rms'][5:].astype(float)
            sum_predictions[arm] += scores
            tree_mse[arm] += w@np.mean((scores-score_target)**2, 1)
            squared_predictions[arm] += w@np.mean(scores*scores, 1)
        mass = tree.tree_.weighted_n_node_samples[fitted['nodes'][start:end]]
        norm = np.sum(fitted['coef'][start:end]**2, (1, 2))
        penalty += .5*fitted.get('settings', {}).get('penalty', 1.)*(mass@norm)/mass.sum()
    n = len(state['model'].estimators_)
    tree_mse /= n
    squared_predictions /= n
    ensemble_mse, dispersion = [], []
    for arm in range(2):
        mean = sum_predictions[arm]/n
        ensemble_mse.append(float(w@np.mean((mean-score_target)**2, 1)))
        dispersion.append(float(squared_predictions[arm]-w@np.mean(mean*mean, 1)))
        np.testing.assert_allclose(tree_mse[arm]-ensemble_mse[arm], dispersion[-1], atol=1e-9, rtol=1e-9)
    return dict(known_rows=int(known.sum()), unknown_rows=int((~known).sum()),
        mean_tree_MSE=tree_mse.tolist(), raw_ensemble_MSE=ensemble_mse, tree_dispersion=dispersion,
        mean_tree_MSE_change=float(tree_mse[1]-tree_mse[0]),
        raw_ensemble_MSE_change=ensemble_mse[1]-ensemble_mse[0],
        dispersion_change=dispersion[1]-dispersion[0], fixed_penalty=float(penalty/n))


def diagnose(state, raw, projected, targets, actions, sites, recordings, frames, previous, extra):
    y = targets
    known = np.isfinite(y).all(1)
    w, _ = forest.core.weights(sites, recordings, frames, known)
    old, new = actions['original'], actions['cost']
    masks = dict(all=np.ones(len(y), bool), original_selected=old, original_unselected=~old,
        cost_selected=new, cost_unselected=~new, added=new & ~old, removed=old & ~new,
        retained=new & old, original_matched=actions['original_matched_cost'],
        cost_matched=actions['cost_matched_original'])
    strata = {}
    for label, index, j in (('harm', 1, 0), ('easy_harm', 4, 1)):
        ratio = np.divide(raw['cost'][:, index], raw['original'][:, index],
                          out=np.full(len(y), np.nan), where=raw['original'][:, index] > 0)
        edges = [-np.inf, .5, 1., 2., 4., np.inf]
        for lo, hi, name in zip(edges[:-1], edges[1:], ('le_half', 'half_to_one', 'one_to_two', 'two_to_four', 'over_four')):
            strata[label+'_rate_'+name] = (ratio > lo) & (ratio <= hi)
        strata[label+'_rate_undefined'] = ~np.isfinite(ratio)
        strata[label+'_majority_train_eff_lt5'] = previous[:, j+2] >= .5
        strata[label+'_majority_train_eff_ge5'] = previous[:, j+2] < .5
        strata[label+'_all_train_zero'] = extra[:, j] == 1
        strata[label+'_any_train_positive'] = extra[:, j] < 1
    strata['outside_majority_leaf_quality_box'] = extra[:, 2] >= .5
    strata['inside_majority_leaf_quality_box'] = extra[:, 2] < .5
    compare = lambda a, b, mask: prior.error_slice(a, b, y, w, state['preprocess'], mask)
    out = dict(cohorts={k: compare(projected['original'], projected['cost'], m) for k, m in masks.items()},
        strata={k: compare(projected['original'], projected['cost'], m) for k, m in strata.items()},
        raw_error_change=compare(raw['original'], raw['cost'], masks['all']),
        projection_effects={k: compare(raw[k], projected[k], masks['all']) for k in raw})
    desc_masks = dict(all=masks['all'], cost_selected=new, retained=new & old,
        selected_known_harm=new & known & (y[:, 1] > 0),
        selected_known_easy_harm=new & known & (y[:, 4] > 0), selected_unknown=new & ~known)
    out['support_cohorts'] = {k: dict(rows=int(m.sum()), unknown_rows=int((m & ~known).sum()),
        descriptor_mean={name: float(extra[m, j].mean()) if m.any() else None for j, name in enumerate(DESCRIPTORS)},
        train_effective_harm_mean=np.mean(previous[m, :2], 0).tolist() if m.any() else None)
        for k, m in desc_masks.items()}
    return out
