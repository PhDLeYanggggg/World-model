"""Change forecast amplitude without changing the original intervention roster."""
import numpy as np

from src.evaluation.m3w_recording_forecast_comparison import errors, fraction


def shrink(floor, alternative, alpha):
    floor, alternative, alpha = map(np.asarray, (floor, alternative, alpha))
    if (floor.shape != alternative.shape or floor.ndim != 3 or floor.shape[-1] != 2
            or alpha.shape not in ((), (len(floor),)) or not np.isfinite(alpha).all()
            or (alpha <= 0.).any() or (alpha > 1.).any()):
        raise ValueError('Aligned paths and strictly positive convex coefficients required')
    delta = alternative-floor
    out = floor+alpha.reshape(-1, 1, 1)*delta
    if not np.array_equal((delta != 0.).any((1, 2)), (out != floor).any((1, 2))):
        raise ValueError('Amplitude control changed selected support')
    return out


def trust_alpha(floor, alternative, scale):
    radius = np.linalg.norm(alternative-floor, axis=-1).max(1)
    if scale.shape != radius.shape or not np.isfinite(scale).all() or (scale <= 0.).any():
        raise ValueError('Require positive past-motion scales')
    return scale / np.maximum(scale, radius)


def bounds(floor, alternative, target, valid, easy_cut, alpha):
    """Upper limits implied by convex ADE for fixed observed support."""
    base, _ = errors(floor, target, valid)
    err, _ = errors(alternative, target, valid)
    known = valid.any(1)
    easy = known & (base > 0.) & (base <= easy_cut)
    radius = np.linalg.norm(alternative-floor, axis=-1).max(1)
    selected = radius > 0.
    alpha = np.broadcast_to(alpha, radius.shape)
    signed = np.where(known, err-base, 0.)*alpha
    harm = signed.clip(0.)
    unknown = float((alpha[~known]*radius[~known]).sum())
    return dict(completion_all_risk_upper=fraction(float(harm.sum())+unknown, float(base[known & selected].sum())),
        completion_easy_risk_upper=fraction(float(harm[easy].sum())+unknown, float(base[easy & selected].sum())),
        known_easy_net_degradation=fraction(float(signed[easy].sum()), float(base[easy].sum())))


def total_risk_cause(risk):
    known, upper = risk['known_all_risk'], risk['completion_all_risk_upper']
    if known is None or upper is None:
        return 'undefined_total_reference'
    if known > .02:
        return 'observed_harm_already_over_2'
    if upper > .02:
        return 'missing_radius_crosses_2'
    return 'total_risk_within_2'
