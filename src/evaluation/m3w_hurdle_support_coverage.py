"""Separate support differences before comparing frozen risk orderings."""
import numpy as np
from src.evaluation.m3w_hurdle_coverage import matched_choices, decompose
from src.evaluation.m3w_native_matched_coverage import paired_scene_contrast
from src.world_model.m3w_european_conditional_risk import pointwise_rule

ARMS = ('product_original', 'hurdle_original', 'product_common', 'hurdle_common',
        'hurdle_at_product', 'product_at_hurdle')


def support_choices(utility, product, hurdle, moving, sites, ids, *, budget):
    original = {
        'product_original': pointwise_rule(utility, product, moving, budget=budget, support_available=True),
        'hurdle_original': pointwise_rule(utility, hurdle, moving, budget=budget, support_available=True),
    }
    common = moving & (np.asarray(product)[:, 0] > 0) & (np.asarray(hurdle)[:, 0] > 0)
    choices, counts = matched_choices(utility, product, hurdle, common, sites, ids, budget=budget)
    choices['product_common'] = choices.pop('product_original')
    choices['hurdle_common'] = choices.pop('hurdle_original')
    choices.update(original)
    for site, row in counts.items():
        ix = np.asarray(sites) == site
        row['product_original_count'] = int((ix & original['product_original']).sum())
        row['hurdle_original_count'] = int((ix & original['hurdle_original']).sum())
        row['product_outside_common'] = int((ix & original['product_original'] & ~common).sum())
        row['hurdle_outside_common'] = int((ix & original['hurdle_original'] & ~common).sum())
    return choices, counts


def support_decomposition(errors, reference, sites, *, mask, resamples, seed):
    if set(errors) != set(ARMS):
        raise ValueError('Both full and common-support anchors required')
    common = dict(product_original=errors['product_common'], hurdle_original=errors['hurdle_common'],
                  hurdle_at_product=errors['hurdle_at_product'], product_at_hurdle=errors['product_at_hurdle'])
    result = decompose(common, reference, sites, mask=mask, resamples=resamples, seed=seed)
    r, s, m = np.asarray(reference, float), np.asarray(sites), np.asarray(mask)
    full = {name: np.asarray(errors[name], float) for name in ('product_original', 'hurdle_original')}
    for a in full.values():
        if (a.shape != r.shape or not np.array_equal(np.isnan(a), np.isnan(r))
                or np.isinf(a).any() or np.any(a[np.isfinite(a)] < 0)):
            raise ValueError('Full anchors require identical nonnegative label support')
    for site, row in result['by_locality'].items():
        if row['status'] != 'defined':
            continue
        use = m & (s == site) & np.isfinite(r)
        row['full_total'] = float(100 * (full['product_original'][use] - full['hurdle_original'][use]).sum() / r[use].sum())
        row['support_difference'] = row['full_total'] - row['total']
    for name in ('full_total', 'support_difference'):
        values = [row[name] for row in result['by_locality'].values()] if result['complete_locality_support'] else []
        result[name] = paired_scene_contrast(values, np.zeros(len(values)), resamples=resamples, seed=seed) if values else None
    return result
