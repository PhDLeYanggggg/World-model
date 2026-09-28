"""Label-only signed utility; never an inference feature or policy input."""
import numpy as np


def gain_labels(cv, reference, candidate, *, easy_cut, scale):
    cv, reference, candidate = (np.asarray(v, dtype=np.float64) for v in (cv, reference, candidate))
    if (cv.ndim != 1 or reference.shape != cv.shape or candidate.shape != cv.shape
            or not np.isfinite(scale) or scale <= 0 or not np.isfinite(easy_cut) or easy_cut < 0):
        raise ValueError('Aligned costs and finite fitting-only scale/easy cut required')
    known = np.isfinite(cv)
    for a in (cv, reference, candidate):
        if np.isinf(a).any() or not np.array_equal(known, np.isfinite(a)) or (a[known] < 0).any():
            raise ValueError('Missing labels must match; finite costs must be nonnegative')
    signed = (reference-candidate)/scale
    easy = ((cv > 0) & (cv <= easy_cut)).astype(float)
    easy[~known] = np.nan
    return dict(known=known, easy=easy, reference=reference/scale, candidate=candidate/scale,
                signed_gain=signed, positive_benefit=np.maximum(signed, 0),
                positive_harm=np.maximum(candidate-reference, 0)/scale)


def verify_alignment(ids, sites, labels, old_targets, roles):
    ids, sites, old = np.asarray(ids), np.asarray(sites), np.asarray(old_targets)
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or (ids < 0).any()
            or (np.diff(ids) <= 0).any() or sites.shape != ids.shape or old.shape != (len(ids), 3)
            or any(np.asarray(v).shape != ids.shape for v in labels.values())):
        raise ValueError('Unique ordered row IDs and exactly aligned label arrays required')
    fitting = set(roles['training_sites'])
    if set(sites) != fitting or any(fitting & set(roles[k]) for k in
                                  ('producer_sites', 'controller_sites', 'held_sites')):
        raise ValueError('Only source-excluded fitting rows allowed')
    np.testing.assert_array_equal(labels['known'], np.isfinite(old).all(1))
    np.testing.assert_array_equal(np.column_stack([labels[k] for k in ('easy', 'reference', 'positive_harm')]), old)
    np.testing.assert_allclose(labels['signed_gain'], labels['positive_benefit']-labels['positive_harm'],
                               rtol=1e-14, atol=1e-14, equal_nan=True)
