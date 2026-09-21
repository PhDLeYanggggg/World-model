"""Deterministic forecast-disagreement bounds, not statistical risk calibration."""
import numpy as np


def disagreement(candidate, baseline, scale):
    p, b, s = np.asarray(candidate, float), np.asarray(baseline, float), np.asarray(scale, float)
    if (p.shape != b.shape or p.ndim != 3 or p.shape[-1] != 2 or p.shape[1] < 1
            or s.shape != (len(p),) or not np.isfinite(p).all() or not np.isfinite(b).all()
            or not np.isfinite(s).all() or np.any(s <= 0)):
        raise ValueError('Finite past-conditioned rollouts and positive causal scales required')
    return np.linalg.norm(p-b, axis=-1)*s[:, None]


def partial_gain_bounds(candidate, baseline, target, valid, scale):
    """Sharp full-grid ADE-gain interval with observed labels held fixed.

    Missing targets range freely in the same Euclidean coordinate space; no
    smoothness/physical constraint is assumed. This is evaluation, never gating.
    """
    d = disagreement(candidate, baseline, scale)
    y, mask = np.asarray(target, float), np.asarray(valid)
    if (y.shape != np.shape(candidate) or mask.shape != d.shape or mask.dtype != bool
            or not np.isfinite(y[mask]).all()):
        raise ValueError('Aligned finite observed labels and explicit Boolean support required')
    observed = np.where(mask[..., None], y, 0)
    b_error = np.linalg.norm(np.asarray(baseline, float)-observed, axis=-1)*np.asarray(scale)[:, None]
    p_error = np.linalg.norm(np.asarray(candidate, float)-observed, axis=-1)*np.asarray(scale)[:, None]
    observed_gain = np.where(mask, b_error-p_error, 0).mean(1)
    missing_radius = np.where(mask, 0, d).mean(1)
    full_disagreement = d.mean(1)
    cv_exact_still_possible = ~np.any(mask & (b_error != 0), axis=1)
    return dict(lower=observed_gain-missing_radius, upper=observed_gain+missing_radius,
                known_gain_contribution=observed_gain, unknown_radius=missing_radius,
                full_disagreement=full_disagreement, complete=mask.all(1),
                exact_CV_still_possible=cv_exact_still_possible,
                exact_CV_harm_upper=np.where(cv_exact_still_possible, full_disagreement, 0.))


def bounded_fractions(costs, distance, complete):
    y, d, mask = np.asarray(costs, float), np.asarray(distance, float), np.asarray(complete)
    if y.shape != (len(d), 2) or mask.shape != d.shape or mask.dtype != bool:
        raise ValueError('Paired benefit/harm costs and complete-grid mask required')
    if np.any(~np.isfinite(y[mask])) or np.any(y[mask] < 0) or np.any(d < 0) or not np.isfinite(d).all():
        raise ValueError('Finite nonnegative complete costs and known disagreement required')
    if np.any(y[mask].sum(1)-d[mask] > 1e-10*(1+d[mask])):
        raise ValueError('Cost exceeds Euclidean disagreement bound')
    if np.any(y[mask & (d == 0)] != 0):
        raise ValueError('Identical predictions must have exactly zero cost')
    z = np.full_like(y, np.nan)
    z[mask] = np.divide(y[mask], d[mask, None], out=np.zeros_like(y[mask]), where=d[mask, None] > 0)
    return z


def energy_concentration(values, fraction=.01):
    x = np.asarray(values, float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or np.any(x < 0) or not 0 < fraction <= 1:
        raise ValueError('Finite nonnegative supported labels required')
    energy = np.square(x)
    k = max(1, int(np.ceil(len(x)*fraction)))
    return dict(rows=len(x), top_rows=k, positive_rows=int((x > 0).sum()),
                energy=float(energy.sum()), top_fraction_energy_share=None if energy.sum() == 0 else float(np.sort(energy)[-k:].sum()/energy.sum()))
