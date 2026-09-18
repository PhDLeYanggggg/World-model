"""Post-hoc error attribution for fixed training-side OOF candidates."""
import numpy as np


def decompose_costs(cv, errors, sites, native_scale):
    cv, errors, sites, native_scale = map(np.asarray, (cv, errors, sites, native_scale))
    if (cv.ndim != 1 or errors.ndim != 2 or errors.shape[1:] != cv.shape
            or len(errors) == 0 or sites.shape != cv.shape or native_scale.shape != cv.shape
            or not len(cv) or not all(np.isfinite(x).all() for x in (cv, errors, native_scale))
            or (cv < 0).any() or (errors < 0).any() or (native_scale <= 0).any()):
        raise ValueError('Aligned nonnegative costs and positive native scales required')
    names = np.unique(sites)
    if any(cv[sites == name].mean() <= 0 for name in names):
        raise ValueError('Each site needs positive baseline mean cost')
    mean = errors.mean(0)
    oracle = np.minimum(errors, cv[None, :]).mean(0)
    # Oracle decisions are made inside each seed, never after forecast averaging.
    easy = cv == 0
    denominator = np.mean([cv[sites == name].mean() for name in names])
    out = dict(seed_count=len(errors), rows=len(cv), easy_rows=int(easy.sum()),
               equal_site_cv_cost=float(denominator),
               equal_site_oracle_gain_percent=float(100 * (1 - np.mean([
                   oracle[sites == name].mean() for name in names]) / denominator)),
               equal_site_excess_decomposition_pp={}, subsets={})
    for label, mask in (('zero_target', easy), ('nonzero_target', ~easy)):
        component = np.mean([np.where(mask[sites == name],
            mean[sites == name] - cv[sites == name], 0).mean() for name in names])
        out['equal_site_excess_decomposition_pp'][label] = float(100 * component / denominator)
        base = float(cv[mask].sum())
        out['subsets'][label] = dict(rows=int(mask.sum()),
            gain_percent=float(100 * (1 - mean[mask].sum() / base)) if base else None,
            binary_oracle_gain_percent=float(100 * (1 - oracle[mask].sum() / base)) if base else None,
            native_pixel_mean_excess=float(((mean[mask] - cv[mask]) * native_scale[mask]).mean()) if mask.any() else None,
            beneficial_fraction_per_seed=[float((e[mask] < cv[mask]).mean()) if mask.any() else None for e in errors])
    total = float(100 * (np.mean([mean[sites == name].mean() for name in names]) / denominator - 1))
    assert np.isclose(sum(out['equal_site_excess_decomposition_pp'].values()), total)
    out['equal_site_excess_total_pp'] = total
    return out
