"""Causal vehicle primitives and a convex, completion-aware blend."""

import numpy as np
from scipy.optimize import minimize

from src.evaluation.m3w_camera_reference_capacity import linear_menu
from src.world_model.m3w_radius_matched_turn_menu import turn_path

NAMES = ('median4', 'smooth_velocity', 'decaying_acceleration', 'decaying_turn')
EPS = 1e-4


def paths(history, cv, scale):
    h = np.asarray(history, dtype=float)
    if (h.shape != (len(h), 8, 2) or cv.shape != (len(h), 12, 2)
            or scale.shape != (len(h),) or not np.isfinite(h).all()
            or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError('Expected eight finite past positions and causal scales')
    d = np.diff(h, axis=1)
    last = d[:, -1]
    a = (d[:, -2:].mean(1)-d[:, -4:-2].mean(1))/2.
    t = np.arange(12)
    acceleration = cv+a[:, None]*np.cumsum(np.cumsum(.72**t))[None, :, None]
    smooth = cv+(d[:, -6:].mean(1)-last)[:, None]*np.cumsum(1.-.8**(t+1))[None, :, None]
    p = np.stack((linear_menu(h, 12)[:, 2], smooth, acceleration, turn_path(h, 12)), axis=-1)
    delta = (p-cv[..., None])/scale[:, None, None, None]
    peak = np.linalg.norm(delta, axis=2).max(axis=1)
    return delta * (2./np.maximum(peak, 2.))[:, None, None, :]


def completion_terms(weights, basis, target):
    # target is zero at a missing step, which charges the correction from CV.
    r = np.einsum('ntdk,k->ntd', basis, weights)-target
    smooth = np.sqrt(np.sum(r*r, axis=-1)+EPS**2)
    g = np.einsum('ntdk,ntd->k', basis, r/smooth[..., None])/smooth.size
    return float(smooth.mean()-EPS), g


def fit(basis, cv, target, valid, scale):
    y = np.where(valid[..., None], (target-cv)/scale[:, None, None], 0.)
    result = minimize(completion_terms, np.zeros(len(NAMES)), jac=True, args=(basis, y), method='SLSQP',
                      bounds=[(0., 1.)]*len(NAMES),
                      constraints={'type': 'ineq', 'fun': lambda x: 1.-x.sum(), 'jac': lambda x: -np.ones_like(x)},
                      options={'maxiter': 100, 'ftol': 1e-11})
    if (not result.success or not np.isfinite(result.fun) or not np.isfinite(result.x).all()
            or np.any(result.x < -1e-8) or result.x.sum() > 1.+1e-8):
        raise ValueError(f'Completion blend optimization failed: {result.message}')
    w = result.x.clip(0., 1.)
    return dict(weights=(w/max(1., w.sum())).tolist(), iterations=int(result.nit),
                donor_loss=float(result.fun), cv_donor_loss=completion_terms(np.zeros_like(w), basis, y)[0])


def predict(basis, cv, scale, weights):
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(NAMES),) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() > 1.+1e-12:
        raise ValueError('Expected feasible nonnegative mixture weights')
    return cv+np.einsum('ntdk,k->ntd', basis, w)*scale[:, None, None]
