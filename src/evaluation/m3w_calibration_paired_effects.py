"""Paired completion effects from unchanged calibration sufficient statistics."""
import numpy as np


def unknown_distance(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 3 or a.shape[1:] != (12, 2):
        raise ValueError('Aligned frozen twelve-step paths required')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Finite prediction paths required')
    return float(np.linalg.norm(a-b, axis=-1).max(1).sum()) if len(a) else 0.


def paired_values(reference_mass, control_utility, candidate_utility, unknown_pair_distance):
    if not all(np.isfinite(v) for v in (reference_mass, control_utility, candidate_utility, unknown_pair_distance)):
        raise ValueError('Finite calibration sufficient statistics required')
    if reference_mass < 0 or unknown_pair_distance < 0:
        raise ValueError('Nonnegative reference and actual paired disagreement required')
    if reference_mass == 0:
        return dict(known_pp=None, completion_lower_pp=None, completion_upper_pp=None)
    gain = candidate_utility-control_utility
    return dict(known_pp=100*gain/reference_mass,
        completion_lower_pp=100*(gain-unknown_pair_distance)/reference_mass,
        completion_upper_pp=100*(gain+unknown_pair_distance)/reference_mass)


def interval(values, *, seed=20261102, resamples=3000):
    if not values or any(v is None for v in values.values()):
        return dict(point=None, CI95=None, status='required_locality_not_estimable')
    x = np.array([values[k] for k in sorted(values)])
    if not np.isfinite(x).all():
        raise ValueError('Finite locality effects required')
    draws = np.random.default_rng(seed).choice(x, (resamples, len(x))).mean(1)
    return dict(point=float(x.mean()), CI95=np.quantile(draws, [.025, .975]).tolist(),
        by_locality=values, locality_units=len(x), bootstrap_resamples=resamples,
        status='conditional_calibration_not_independent_confirmation')
