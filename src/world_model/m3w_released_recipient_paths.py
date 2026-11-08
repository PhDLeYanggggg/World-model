"""Nested causal trajectory menus for exact-budget recipient comparisons."""

import numpy as np

from src.world_model.m3w_past_sampling import SCHEDULES, fit_velocity


MENUS = ("half_exact", "full_exact", "curved_exact", "curved_free")
WIDTHS = (4, 8, 10, 10)


def paths(history, cv, ols):
    """Capped curvature changes direction over lead while retaining the anchor."""
    lead = np.arange(1., 13.)[None, :, None]
    full = [ols]+[fit_velocity(history, offsets)[:, None]*lead
                  for name, offsets in SCHEDULES.items() if name != "recent"]
    half = [cv+.5*(p-cv) for p in full]
    dv = full[0][:, :1]-full[1][:, :1]
    correction = .5*(dv/1.5)*lead**2
    size = np.linalg.norm(correction, axis=-1)
    radius = np.linalg.norm(ols-cv, axis=-1)
    cap = np.minimum(size, radius)
    correction *= np.divide(cap, size, out=np.zeros_like(cap), where=size > 0.)[..., None]
    return np.stack(half+full+[ols+correction, ols-.5*correction], axis=1)
