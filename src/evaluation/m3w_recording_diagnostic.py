"""Conditional recording-block uncertainty; never independent scene evidence."""
import numpy as np


def recording_resamples(groups, repetitions=2000, seed=38113):
    groups = np.asarray(groups)
    if groups.ndim != 1 or not len(groups) or repetitions < 1:
        raise ValueError('Nonempty aligned recording groups required')
    labels, inverse = np.unique(groups, return_inverse=True)
    if len(labels) < 2:
        raise ValueError('At least two recording blocks required')
    sampled = np.random.default_rng(seed).integers(len(labels), size=(repetitions, len(labels)))
    counts = np.zeros((repetitions, len(labels)), dtype=int)
    np.add.at(counts, (np.arange(repetitions)[:, None], sampled), 1)
    return labels, inverse, counts


def paired_gain_interval(reference, candidate, baseline, inverse, counts, mask=None):
    """Window-weighted gain in baseline percentage points; shared block draws."""
    a, b, cv = [np.asarray(x, dtype=float) for x in (reference, candidate, baseline)]
    inverse, counts = np.asarray(inverse), np.asarray(counts)
    if (a.ndim != 1 or not len(a) or a.shape != b.shape or a.shape != cv.shape
            or inverse.shape != a.shape or counts.ndim != 2
            or not all(np.isfinite(x).all() for x in (a, b, cv))
            or np.any(cv < 0) or np.any(inverse < 0)
            or np.any(inverse >= counts.shape[1]) or np.any(counts < 0)):
        raise ValueError('Finite aligned losses and legal block resamples required')
    mask = np.ones(len(a), dtype=bool) if mask is None else np.asarray(mask)
    if mask.shape != a.shape or mask.dtype != bool:
        raise ValueError('Aligned boolean diagnostic subset required')
    n = counts.shape[1]
    numerator = np.bincount(inverse[mask], weights=(a-b)[mask], minlength=n)
    denominator = np.bincount(inverse[mask], weights=cv[mask], minlength=n)
    sums, denoms = counts @ numerator, counts @ denominator
    valid = denoms > 0
    point = float(100*numerator.sum()/denominator.sum()) if denominator.sum() > 0 else None
    values = 100*sums[valid]/denoms[valid]
    return dict(point_percent=point,
        conditional_recording_ci95=np.quantile(values, [.025, .975]).tolist() if len(values) else None,
        valid_resamples=int(valid.sum()), invalid_zero_denominator_resamples=int((~valid).sum()),
        rows=int(mask.sum()), records=int(len(np.unique(inverse[mask]))),
        uncertainty_scope='recordings_within_one_previously_explored_physical_site')


def error_summary(ade, fde, baseline, native_scale, changed, hard):
    ade, fde, cv, native, changed, hard = map(np.asarray, (ade, fde, baseline, native_scale, changed, hard))
    if not len(ade) or any(x.shape != ade.shape for x in (fde, cv, native, changed, hard)):
        raise ValueError('Aligned nonempty per-query losses required')
    if not all(np.isfinite(x).all() for x in (ade, fde, cv, native, changed)) or np.any(native <= 0):
        raise ValueError('Finite losses and positive source scales required')
    zero, moving = cv == 0, cv > 0
    mean = lambda x, m: float(x[m].mean()) if m.any() else None
    gain = lambda m: float(100*(1-ade[m].sum()/cv[m].sum())) if m.any() and cv[m].sum() > 0 else None
    oracle = np.minimum(ade, cv)
    return dict(rows=len(ade), ade=float(ade.mean()), fde=float(fde.mean()), cv_ade=float(cv.mean()),
        gain_percent=gain(np.ones(len(ade), bool)), native_pixel_ade=float((ade*native).mean()),
        native_pixel_cv_ade=float((cv*native).mean()), harm_over_cv=float((ade-cv).mean()),
        easy_rows=int(zero.sum()), moving_rows=int(moving.sum()), hard_rows=int(hard.sum()),
        easy_absolute_harm=mean(ade, zero), easy_pixel_harm=mean(ade*native, zero),
        easy_percentage_degradation=None, easy_percentage_reason='zero_error_CV_denominator',
        moving_gain_percent=gain(moving), hard_gain_percent=gain(hard),
        actual_changed_rate=float(changed.mean()), tail_ade95=float(np.quantile(ade, .95)),
        tail_ade99=float(np.quantile(ade, .99)), rows_helped=int((ade < cv).sum()),
        rows_harmed=int((ade > cv).sum()), binary_oracle_ade=float(oracle.mean()),
        binary_oracle_gain_percent=float(100*(1-oracle.sum()/cv.sum())) if cv.sum() > 0 else None,
        regret_to_binary_future_oracle=float((ade-oracle).mean()), oracle_is_deployable=False)
