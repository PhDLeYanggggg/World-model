"""Standalone causal CV-to-OLS4 interpolation selected on fitting groups."""

import numpy as np

from src.world_model.m3w_dense_motion_moments import LEVELS, slope


GUARDS = ("all", "mean_positive", "three_origins", "four_origins")
MENU = tuple(f"{guard}_{level}" for guard in GUARDS for level in range(1, 5))


def backtests(history):
    h = np.asarray(history, dtype=float)
    if h.ndim != 3 or h.shape[1:] != (8, 2) or not np.isfinite(h).all():
        raise ValueError("Require eight finite causal positions")
    gains = []
    for origin in range(4, 8):
        prefix = h[:, origin-4:origin]
        cv = prefix[:, -1]-prefix[:, -2]
        ols = slope(prefix)
        velocities = cv[:, None]+LEVELS[None, :, None]*(ols-cv)[:, None]
        lead = np.arange(1., 9-origin)
        actual = h[:, origin:]-prefix[:, -1:]
        error = np.linalg.norm(velocities[:, :, None]*lead[None, None, :, None]
                               -actual[:, None], axis=-1).mean(2)
        gains.append((error[:, :1]-error[:, 1:])/lead.mean())
    return np.stack(gains, axis=1)


def menu(gains, legal):
    """Four amplitudes crossed with fixed evidence requirements."""
    gain = gains.mean(1)
    agreement = (gains > 0.).sum(1)
    arms = (np.ones_like(gain, dtype=bool), gain > 0.,
            (gain > 0.) & (agreement >= 3), agreement == 4)
    return {f"{guard}_{i}": np.where(keep[:, i-1] & legal, i, 0).astype(np.uint8)
            for guard, keep in zip(GUARDS, arms, strict=True) for i in range(1, 5)}


def select(scores, groups):
    """Prioritize worst-group risk pass fraction, then mean pass and utility."""
    if not groups or set(scores) != set(MENU):
        raise ValueError("Fixed menu and nonempty fitting groups required")

    def key(arm):
        cells = [scores[arm][g] for g in groups]
        rates = [v["passed"]/v["recordings"] for v in cells]
        gain = [v["gain_pp"] for v in cells]
        return (min(rates), np.mean(rates), np.mean(gain) if None not in gain else -np.inf)

    return max(MENU, key=key)
