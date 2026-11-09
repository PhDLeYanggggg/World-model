"""Causal compatibility of fixed forecast shapes with the observed history."""

import numpy as np

from src.world_model import m3w_ols_replacement as motion

PREFIXES = (4, 5, 6, 7)
WIDTH = len(PREFIXES)*8


def align(delta, current_velocity, prefix_velocity):
    """Rotate heading and rescale speed; keep orientation at stationary endpoints."""
    c = np.linalg.norm(current_velocity, axis=-1)
    p = np.linalg.norm(prefix_velocity, axis=-1)
    moving = (c > 1e-8) & (p > 1e-8)
    den = np.where(moving, c*p, 1.)
    cs = np.where(moving, (current_velocity*prefix_velocity).sum(-1)/den, 1.)
    sn = np.where(moving, (current_velocity[..., 0]*prefix_velocity[..., 1]
                          - current_velocity[..., 1]*prefix_velocity[..., 0])/den, 0.)
    ratio = np.where(moving, p/np.maximum(c, 1e-8), 1.).clip(.25, 4.)
    x, y = np.moveaxis(delta, -1, 0)
    return np.stack((cs[..., None]*x-sn[..., None]*y,
                     sn[..., None]*x+cs[..., None]*y), axis=-1)*ratio[..., None, None]


def features(history, cv, paths, recipients, cutoff):
    """The final-origin template is known now, not claimed to be known at a prefix."""
    h = np.asarray(history, dtype=float)
    if h.shape[1:] != (8, 2) or paths.shape[1:] != cv.shape or recipients.shape != (len(paths), 2):
        raise ValueError("Expected exact two-edit forecasts on the 8/12 grid")
    i = recipients
    d = paths[np.arange(len(paths))[:, None], i]-cv[i]
    final_v = h[:, -1]-h[:, -2]
    norm = lambda x: np.linalg.norm(x, axis=-1)
    out = []
    for k in PREFIXES:
        n = 8-k
        c, o, _, scale = motion.rollout(h[:, :k], n)
        y = h[:, k:]-h[:, k-1:k]
        v = h[:, k-1]-h[:, k-2]
        change = align(d[:, :, :n], final_v[i], v[i])
        replay = c[i]+change
        base = norm(c[i]-y[i]).mean(2)
        err = norm(replay-y[i]).mean(2)
        other = norm(o[i]-y[i]).mean(2)
        harm = np.maximum(err-base, 0.)
        easy = (base > 0.) & (base <= cutoff*n/12.)
        u, w = change[:, 0], change[:, 1]
        z = c[i[:, 0]]-c[i[:, 1]]-(y[i[:, 0]]-y[i[:, 1]])
        z0, z1 = norm(z), norm(z+u-w)
        scale_pair = np.mean(scale[i], axis=1)
        p = np.column_stack(((base-err).mean(1), (base-err).min(1), (other-err).mean(1),
            harm.mean(1), (harm*easy).max(1), (z0-z1).mean(1),
            (norm(z+u)+norm(z-w)-z1-z0).mean(1))) / scale_pair[:, None]
        out.append(np.column_stack((p, easy.mean(1))))
    x = np.column_stack(out).clip(-16., 16.).astype(np.float32)
    if x.shape != (len(paths), WIDTH) or not np.isfinite(x).all():
        raise ValueError("Invalid causal history replay")
    return x
