"""Velocity fits at fixed forecast anchors using only observed positions."""

import numpy as np


SCHEDULES = {
    "recent": (-3, -2, -1, 0),
    "wide_uniform": (-6, -4, -2, 0),
    "wide_endpoints": (-6, -5, -1, 0),
    "wide_dense": (-6, -5, -4, -3, -2, -1, 0),
}


def fit_velocity(history, offsets):
    """OLS slope in native index units; forecast origin stays at the last point."""
    h = np.asarray(history, dtype=float)
    t = np.asarray(offsets)
    if (h.ndim != 3 or h.shape[1:] != (8, 2) or not np.isfinite(h).all()
            or t.ndim != 1 or len(t) < 2 or not np.issubdtype(t.dtype, np.integer)
            or (np.diff(t) <= 0).any() or t[0] < -7 or t[-1] != 0):
        raise ValueError("Eight observed points and increasing integer past offsets ending at zero required")
    centered = t-t.mean()
    rel = h[:, t+7]-h[:, -1:]
    return np.einsum("t,ntd->nd", centered, rel)/(centered@centered)


def candidates(history, cv, ols, frozen_recipients):
    cv, ols, on = map(np.asarray, (cv, ols, frozen_recipients))
    if (cv.shape != (len(history), 12, 2) or ols.shape != cv.shape
            or on.shape != (len(cv),) or on.dtype != bool):
        raise ValueError("Require aligned native reference paths and fixed recipients")
    lead = np.arange(1., 13.)[None, :, None]
    result = {"recent": cv+on[:, None, None]*.5*(ols-cv)}
    for name, offset in SCHEDULES.items():
        if name != "recent":
            alternative = fit_velocity(history, offset)[:, None]*lead
            result[name] = cv+on[:, None, None]*.5*(alternative-cv)
    return result
