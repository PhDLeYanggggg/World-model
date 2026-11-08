"""Past-only alternatives at the exact realized OLS4 edit radius."""

import numpy as np

from src.world_model.m3w_ols_replacement import rollout, velocity

NAMES = ("linear", "early", "late", "damped_speed", "long_fit", "tapered_turn")


def peak(offset):
    return np.linalg.norm(offset, axis=-1).max(axis=-1)


def turn_path(history, steps):
    dv = np.diff(history, axis=1)
    left, right = dv[:, :-1], dv[:, 1:]
    good = (np.linalg.norm(left, axis=2) > 1e-8) & (np.linalg.norm(right, axis=2) > 1e-8)
    cross = left[..., 0]*right[..., 1]-left[..., 1]*right[..., 0]
    dot = (left*right).sum(2)
    angle = np.where(good, np.arctan2(cross, dot).clip(-.35, .35), 0.)
    recent = angle[:, -3:]
    yaw = np.median(recent, axis=1).clip(-.20, .20)
    initial = velocity(history, 4)
    # Decay curvature to avoid cycling on a single uncertain turn estimate.
    phase = yaw[:, None]*np.cumsum(.8**np.arange(steps))[None]
    u, v = initial[:, :1], initial[:, 1:]
    c, s = np.cos(phase), np.sin(phase)
    return np.stack((u*c-v*s, u*s+v*c), axis=-1).cumsum(1)


def make_menu(history, steps, amplitude):
    cv, ols, legacy, _ = rollout(history, steps)
    a = np.asarray(amplitude, float)
    if a.shape != (len(cv),) or not np.isfinite(a).all() or np.any((a < 0.) | (a > 1.)):
        raise ValueError("Require a registered per-target OLS bridge amplitude")
    anchor = cv+a[:, None, None]*(ols-cv)
    anchor[a == 1.] = ols[a == 1.]
    off = anchor-cv
    r = peak(off)
    t = np.arange(1, steps+1, dtype=float)/steps
    bases = (off, off[:, -1:]*np.sqrt(t)[None, :, None], off[:, -1:]*t[None, :, None]**2,
             legacy[:, 2]-cv, legacy[:, 1]-cv, turn_path(history, steps)-cv)
    out = [anchor]
    for z in bases[1:]:
        z0 = peak(z)
        factor = np.divide(r, z0, out=np.zeros_like(r), where=z0 > 1e-12)
        y = cv+z*factor[:, None, None]
        y[z0 <= 1e-12] = anchor[z0 <= 1e-12]
        out.append(y)
    menu = np.stack(out, axis=1)
    if not np.isfinite(menu).all():
        raise ValueError("Nonfinite trajectory proposal")
    np.testing.assert_allclose(peak(menu-cv[:, None]), np.broadcast_to(r[:, None], menu.shape[:2]), atol=1e-10, rtol=1e-8)
    return cv, ols, menu


def prefix_rank(history, amplitude, active):
    """Rank using only held-out suffixes of the observed eight-step window."""
    values = []
    for k in (4, 5, 6):
        c, _, b = make_menu(history[:, :k], 8-k, amplitude)
        y = history[:, k:]-history[:, k-1:k]
        c_err = np.linalg.norm(c-y, axis=-1).mean(1)
        e = np.linalg.norm(b-y[:, None], axis=-1).mean(2)
        step = np.linalg.norm(np.diff(history[:, :k], axis=1), axis=-1).mean(1).clip(.05)
        values.append((c_err[:, None]-e)/step[:, None]/(8-k))
    scores = np.asarray(values)
    scores[:, ~active] = 0.
    return scores
