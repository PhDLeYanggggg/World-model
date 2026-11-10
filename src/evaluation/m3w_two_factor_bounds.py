"""Paired attribution of forecast replacements with a shared missing future."""

import numpy as np


def contrast(p00, p01, p10, p11, y, valid, weight, reference_mass):
    """Average the two intervention orders; bound missing gains by distance."""
    arrays = (p00, p01, p10, p11)
    if (y.ndim != 3 or y.shape[-1] != 2 or any(p.shape != y.shape or not np.isfinite(p).all() for p in arrays)
            or valid.shape != y.shape[:2] or valid.dtype != bool or weight.shape != (len(y),)
            or not np.isfinite(y[valid]).all() or not np.all(np.isfinite(weight) & (weight > 0))
            or not np.isfinite(reference_mass) or reference_mass < 0):
        raise ValueError('Expected finite matched forecast grids and a common reference denominator')
    if reference_mass == 0:
        return None
    target = np.where(valid[..., None], y, 0.)
    e00, e01, e10, e11 = (np.linalg.norm(p-target, axis=-1) for p in arrays)
    diag = e00-e11
    path = .5*((e00-e01)+(e10-e11))
    recipient = .5*((e00-e10)+(e01-e11))

    def distance(a, b):
        return np.linalg.norm(a-b, axis=-1)

    cross = distance(p00, p11)+distance(p01, p10)
    path_naive = .5*(distance(p00, p01)+distance(p10, p11))
    recipient_naive = .5*(distance(p00, p10)+distance(p01, p11))

    def effect(delta, radius, naive):
        mass = 100./reference_mass
        point = float(mass*(weight @ np.where(valid, delta, 0.).sum(1)))
        debit = float(mass*(weight @ np.where(valid, 0., radius).sum(1)))
        untightened = float(mass*(weight @ np.where(valid, 0., naive).sum(1)))
        return dict(observed_pp=point, lower_pp=point-debit, upper_pp=point+debit,
                    missing_radius_pp=debit, separate_pair_radius_pp=untightened)

    diag_radius = distance(p00, p11)
    return dict(total=effect(diag, diag_radius, diag_radius),
                path=effect(path, np.minimum(path_naive, .5*cross), path_naive),
                recipient=effect(recipient, np.minimum(recipient_naive, .5*cross), recipient_naive))
