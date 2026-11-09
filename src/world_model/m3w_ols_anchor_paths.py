"""Past-only OLS paths with gradually applied position and velocity corrections."""
import numpy as np

from src.world_model.m3w_prefix_expert_mixture import slope


NAMES = ('ols4_center_fast', 'ols4_center_slow', 'ols8_center_fast',
         'ols8_center_slow', 'ols4_longitudinal_half', 'ols4_longitudinal_full',
         'ols4_lateral_half', 'ols4_lateral_full', 'ols4_center_bend')


def candidates(history):
    h = np.asarray(history, dtype=float)
    if h.ndim != 3 or h.shape[1:] != (8, 2) or not np.isfinite(h).all():
        raise ValueError('Require eight finite history positions')
    h = h - h[:, -1:]
    t = np.arange(1., 13.)[None, :, None]
    v4, v8 = slope(h, 4), slope(h, 8)
    paths = []
    for w, v in ((4, v4), (8, v8)):
        fitted_position = h[:, -w:].mean(1) + ((w-1)/2)*v
        paths.extend(v[:, None]*t + fitted_position[:, None]*(-np.expm1(-t/tau))
                     for tau in (2., 5.))

    # Integrate a saturating velocity change inferred from disjoint past blocks.
    velocity = np.diff(h, axis=1)
    dv = velocity[:, -3:].mean(1) - velocity[:, :3].mean(1)
    speed = np.linalg.norm(v4, axis=1)
    direction = v4 / speed.clip(1e-10)[:, None]
    longitudinal = (dv*direction).sum(1).clip(-.5*speed, .5*speed)[:, None]*direction
    lateral = dv - (dv*direction).sum(1)[:, None]*direction
    lateral *= np.minimum(1., .5*speed/np.linalg.norm(lateral, axis=1).clip(1e-10))[:, None]
    bend = (1.-np.exp(-t/3.)).cumsum(1)
    for component in (longitudinal, lateral):
        paths.extend(v4[:, None]*t + a*component[:, None]*bend for a in (.5, 1.))
    paths.append(paths[1] + .5*(longitudinal+lateral)[:, None]*bend)
    return np.stack(paths, axis=1)
