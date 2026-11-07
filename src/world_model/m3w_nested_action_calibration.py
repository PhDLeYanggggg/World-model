"""Calibrate selected moment errors from source-excluded policy predictions."""
import numpy as np

from src.evaluation.m3w_selection_forecast_readout import errors

ARMS = ('raw', 'harm_only', 'reference_only', 'joint')


def predicted_components(x, coef, scale, radius):
    _, harm, easy_harm, ref, easy_ref, unknown = np.einsum('nad,adc->nac', x, coef).transpose(2, 0, 1)
    missing = unknown.clip(0., 1.) * radius
    harm = np.minimum(harm.clip(0.) * scale[:, None], radius - missing)
    easy_harm = np.minimum(easy_harm.clip(0.) * scale[:, None], harm)
    ref = ref.clip(0.) * scale[:, None]
    easy_ref = np.minimum(easy_ref.clip(0.) * scale[:, None], ref)
    return np.stack((harm + missing, ref, easy_harm + missing, easy_ref), axis=-1)


def observed_components(paths, floor, target, valid, radius, easy_cut):
    ref, _ = errors(floor, target, valid)
    known = valid.any(1)
    ref = np.where(known, ref, 0.)
    easy = known & (ref > 0.) & (ref <= easy_cut)
    out = []
    for k in range(paths.shape[1]):
        e, _ = errors(paths[:, k], target, valid)
        harm = np.where(known, np.maximum(e-ref, 0.), 0.)
        missing = ~known * radius[:, k]
        out.append(np.column_stack((harm + missing, ref, harm * easy + missing, ref * easy)))
    return np.stack(out, axis=1)


def residuals(pred, actual, radius, choices, recordings):
    """Zero actual easy reference is retained as calibration evidence."""
    result = []
    for r in np.unique(recordings):
        ix = np.flatnonzero((recordings == r) & (choices >= 0)); take = (ix, choices[ix])
        if not len(ix):
            continue
        normalizer = float(radius[take].sum())
        if normalizer <= 0.:
            raise ValueError('Selected actions must be distinct from the floor')
        p, y = pred[take].sum(0), actual[take].sum(0)
        result.append(dict(recording=str(r), selected=len(ix), radius=normalizer,
            prediction=p.tolist(), observed=y.tolist(),
            residual=((y-p) * [1., -1., 1., -1.] / normalizer).tolist()))
    return result


def calibrate(donors, held):
    """Empirical source-tail adjustment, not a coverage certificate."""
    if held in donors:
        raise ValueError('Target source may not calibrate its own decisions')
    active = [rows for rows in donors.values() if rows]
    if len(active) < 3:
        return None
    maxima = np.stack([np.max([r['residual'] for r in rows], axis=0) for rows in active])
    return np.maximum(np.quantile(maxima, .90, axis=0, method='higher'), 0.)


def adjusted_excess(arm, all_excess, easy_excess, radius, margin):
    if arm not in ARMS:
        raise ValueError('Registered component required')
    if arm == 'raw':
        return all_excess, easy_excess
    if margin is None:
        return np.full_like(radius, np.inf), np.full_like(radius, np.inf)
    h, r, eh, er = margin
    harm = arm in ('harm_only', 'joint'); reference = arm in ('reference_only', 'joint')
    return (all_excess + radius*(harm*h + .02*reference*r),
            easy_excess + radius*(harm*eh + .02*reference*er))
