"""Shift residual timing without changing the endpoint or peak radius."""

import numpy as np


SHAPES = ('original', 'early', 'late')


def phase_paths(cv, forecast):
    cv, forecast = np.asarray(cv, dtype=np.float64), np.asarray(forecast, dtype=np.float64)
    if (cv.shape != forecast.shape or cv.ndim != 3 or cv.shape[-1] != 2 or cv.shape[1] < 2
            or not np.isfinite(cv).all() or not np.isfinite(forecast).all()):
        raise ValueError('Expected finite paired forecast paths')
    d = forecast-cv
    n, steps = d.shape[:2]
    grid = np.arange(steps+1, dtype=float)
    peak = np.linalg.norm(d, axis=-1).argmax(1)+1
    residual = np.concatenate((np.zeros((n, 1, 2)), d), axis=1)
    result = {'original': forecast}
    for name, exponent in (('early', .5), ('late', 2.)):
        out = np.empty_like(d)
        for i, p in enumerate(peak):
            # Convex interpolation between existing residuals cannot grow the radius.
            u = np.r_[p*(grid[1:p+1]/p)**exponent,
                      p+(steps-p)*((grid[p+1:]-p)/max(steps-p, 1))**exponent]
            out[i] = cv[i]+np.column_stack([np.interp(u, grid, residual[i, :, j]) for j in range(2)])
            out[i, p-1] = forecast[i, p-1]
            out[i, -1] = forecast[i, -1]
        result[name] = out
    return result


def prefix_choices(paths, ols, steps=3):
    """Choose the shape closest to a causal OLS4 continuation at the boundary."""
    if not 1 <= steps <= ols.shape[1]:
        raise ValueError('Invalid boundary length')
    cost = np.column_stack([((paths[k][:, :steps]-ols[:, :steps])**2).sum((1, 2)) for k in SHAPES])
    return cost.argmin(1)
