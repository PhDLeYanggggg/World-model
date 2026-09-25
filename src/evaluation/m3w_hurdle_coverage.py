"""Outcome-blind locality-matched risk rankings and additive error accounting."""
import numpy as np

from src.evaluation.m3w_native_matched_coverage import top_count, paired_scene_contrast
from src.world_model.m3w_european_conditional_risk import pointwise_rule


ARMS = ('product_original', 'hurdle_original', 'hurdle_at_product', 'product_at_hurdle')
COMPONENTS = {
    'total': ('product_original', 'hurdle_original'),
    'ranking_at_product_count': ('product_original', 'hurdle_at_product'),
    'coverage_with_hurdle_ranking': ('hurdle_at_product', 'hurdle_original'),
    'coverage_with_product_ranking': ('product_original', 'product_at_hurdle'),
    'ranking_at_hurdle_count': ('product_at_hurdle', 'hurdle_original'),
}


def matched_choices(utility, product, hurdle, moving, sites, ids, *, budget):
    """Offline count controls, not new policies satisfying the risk constraint."""
    u, p, h = np.asarray(utility), np.asarray(product), np.asarray(hurdle)
    sites, ids = np.asarray(sites), np.asarray(ids)
    if (u.ndim != 1 or sites.shape != u.shape or ids.shape != u.shape
            or not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids)):
        raise ValueError('Unique integer IDs and aligned locality inputs required')
    original = {
        'product': pointwise_rule(u, p, moving, budget=budget, support_available=True),
        'hurdle': pointwise_rule(u, h, moving, budget=budget, support_available=True),
    }
    common = np.asarray(moving) & (u > 0) & (p[:, 0] > 0) & (h[:, 0] > 0)
    if any(np.any(use & ~common) for use in original.values()):
        raise ValueError('Anchors must lie in common causal support; no silent filtering')
    scores = {}
    for name, risk in (('product', p), ('hurdle', h)):
        risk = risk.astype(float)
        scores[name] = -np.divide(risk[:, 1], risk[:, 0], out=np.zeros(len(u)), where=risk[:, 0] > 0)
    choices = {name + '_original': use for name, use in original.items()}
    choices.update(hurdle_at_product=np.zeros(len(u), bool), product_at_hurdle=np.zeros(len(u), bool))
    records = {}
    for site in sorted(set(sites)):
        rows = np.flatnonzero(sites == site)
        counts = {name: int(use[rows].sum()) for name, use in original.items()}
        for anchor, other in (('product', 'hurdle'), ('hurdle', 'product')):
            reconstructed = top_count(scores[anchor][rows], common[rows], ids[rows], counts[anchor])
            np.testing.assert_array_equal(reconstructed, original[anchor][rows])
            choices[other + '_at_' + anchor][rows] = top_count(
                scores[other][rows], common[rows], ids[rows], counts[anchor])
        records[str(site)] = dict(rows=len(rows), eligible=int(common[rows].sum()),
                                 product_count=counts['product'], hurdle_count=counts['hurdle'])
    return choices, records


def decompose(errors, reference, sites, *, mask, resamples, seed):
    """Both paths use the same reference denominator; effects add in pp."""
    r, sites, mask = np.asarray(reference, float), np.asarray(sites), np.asarray(mask)
    if (r.ndim != 1 or sites.shape != r.shape or mask.shape != r.shape or mask.dtype != bool
            or set(errors) != set(ARMS) or np.isinf(r).any() or np.any(r[np.isfinite(r)] < 0)):
        raise ValueError('Aligned fixed error arms, reference and Boolean subset required')
    values = {name: np.asarray(x, float) for name, x in errors.items()}
    for x in values.values():
        if (x.shape != r.shape or not np.array_equal(np.isnan(x), np.isnan(r))
                or np.isinf(x).any() or np.any(x[np.isfinite(x)] < 0)):
            raise ValueError('All arms need identical finite nonnegative outcome support')
    known = np.isfinite(r)
    by_site = {}
    for site in sorted(set(sites)):
        use = (sites == site) & mask & known
        denominator = float(r[use].sum())
        row = dict(rows=int(use.sum()), reference_sum=denominator)
        if not use.any() or denominator == 0:
            row.update(status='no_supported_labels' if not use.any() else 'zero_reference_percentage_undefined')
        else:
            row['status'] = 'defined'
            for name, (a, b) in COMPONENTS.items():
                row[name] = float(100 * (values[a][use] - values[b][use]).sum() / denominator)
            np.testing.assert_allclose(row['total'], row['ranking_at_product_count'] + row['coverage_with_hurdle_ranking'], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(row['total'], row['ranking_at_hurdle_count'] + row['coverage_with_product_ranking'], atol=1e-10, rtol=1e-10)
        by_site[str(site)] = row
    complete = len(by_site) >= 2 and all(x['status'] == 'defined' for x in by_site.values())
    result = dict(by_locality=by_site, indexed_rows=int(mask.sum()),
                  unknown_rows=int((mask & ~known).sum()), complete_locality_support=complete)
    for name in COMPONENTS:
        vals = [x[name] for x in by_site.values()] if complete else []
        result[name] = paired_scene_contrast(vals, np.zeros(len(vals)), resamples=resamples, seed=seed) if complete else None
    return result
