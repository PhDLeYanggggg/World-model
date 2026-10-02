"""Versioned diagnosis retaining the registered reader's label precision."""
import numpy as np
from src.world_model import m3w_positive_harm_diagnostic as v1

forest, MOMENTS, SCORES, LINEAR = v1.forest, v1.MOMENTS, v1.SCORES, v1.LINEAR
training_support, raw_positive = v1.training_support, v1.raw_positive


def signed(y):
    # Do not promote float32 labels before their historical score transform.
    return np.stack((y[:, 0]-y[:, 1], y[:, 1]-.02*y[:, 2], y[:, 4]-.02*y[:, 3]), 1)


def attribution(old, new, target, scale, rms):
    a, b, y = np.asarray(old, float), np.asarray(new, float), np.asarray(target)
    rms = np.asarray(rms, float); scale = float(scale)
    if a.shape != b.shape or a.shape != y.shape or a.ndim != 2 or a.shape[1] != 5:
        raise ValueError('Five aligned finite known moments required')
    if not all(np.isfinite(v).all() for v in (a, b, y, rms)) or rms.shape != (3,) or (rms <= 0).any() or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Finite targets and fixed positive source score scales required')
    ea, eb = (signed(a)-signed(y))/scale/rms, (signed(b)-signed(y))/scale/rms
    jacobian = LINEAR/scale/rms[:, None]
    scores = (eb*eb-ea*ea)/3
    moments = (b-a)*((ea+eb)@jacobian)/3
    np.testing.assert_allclose(scores.sum(1), moments.sum(1), atol=1e-9, rtol=1e-9)
    return dict(scores=scores, moments=moments, old_MSE=np.mean(ea*ea, 1), new_MSE=np.mean(eb*eb, 1))


def error_slice(old, new, target, weights, preprocess, mask):
    y, w, mask = np.asarray(target), np.asarray(weights), np.asarray(mask)
    known = np.isfinite(y).all(1); at = mask & known; mass = float(w[at].sum())
    d = attribution(old[at], new[at], y[at], preprocess['scale'], preprocess['rms'][5:])
    row = dict(rows=int(mask.sum()), known_rows=int(at.sum()), unknown_rows=int((mask & ~known).sum()),
        known_query_weight_mass=mass, global_weighted_old_MSE=float(w[at]@d['old_MSE']),
        global_weighted_new_MSE=float(w[at]@d['new_MSE']),
        score_contributions=(w[at]@d['scores']).tolist(), moment_contributions=(w[at]@d['moments']).tolist())
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
    a, b, yy, w = projected['original'][known], projected['positive'][known], y[known], weights[known]
    def delta(d): return float(w@(d['new_MSE']-d['old_MSE']))
    fp64 = v1.attribution(a, b, yy.astype(float), pr['scale']*pr['rms'][5:].astype(float))
    bad_scale = v1.attribution(a, b, yy.astype(float), pr['scale']*pr['rms'][5:])
    precise = delta(fp64); previous = delta(bad_scale)
    out['numerical_reconciliation'] = dict(target_dtype=str(y.dtype), rms_dtype=str(pr['rms'].dtype),
        v1_scale_product_dtype=str((pr['scale']*pr['rms'][5:]).dtype),
        fully_float64_algebraic_delta=precise, v1_diagnostic_delta=previous,
        registered_readout_delta=full['global_weighted_MSE_change'],
        target_transform_rounding_effect=full['global_weighted_MSE_change']-precise,
        scale_product_rounding_effect=previous-precise)
    return out
