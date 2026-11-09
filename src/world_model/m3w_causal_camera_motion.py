"""Forecast in the last observed image plane and reproject with past-only motion."""

import numpy as np

from src.world_model.m3w_ols_replacement import rollout


def transport(path, matrices):
    return np.einsum("tij,ntj->nti", matrices[:, :2, :2], path)+matrices[None, :, :2, 2]


def camera_maps(steps, center):
    a = np.asarray(steps, dtype=float)
    center = np.asarray(center, dtype=float)
    if (a.shape != (7, 3, 3) or center.shape != (2,) or not np.isfinite(a).all()
            or not np.array_equal(a[:, 2], np.broadcast_to([0., 0., 1.], (7, 3)))
            or np.any(np.linalg.det(a[:, :2, :2]) <= 0.)):
        raise ValueError("Seven accepted, finite orientation-preserving similarities required")
    observed = np.repeat(np.eye(3)[None], 8, axis=0)
    for t in range(6, -1, -1):
        observed[t] = observed[t+1] @ a[t]
    tail = a[-3:]
    shift = np.median(tail[:, :2, :2] @ center+tail[:, :2, 2]-center, axis=0)
    yaw = np.median(np.arctan2(tail[:, 1, 0], tail[:, 0, 0]))
    log_scale = np.median(np.log(np.linalg.norm(tail[:, :2, 0], axis=1)))
    projected = np.repeat(np.eye(3)[None], 12, axis=0)
    pose = np.eye(3)
    for k in range(12):
        weight = .9**k
        c, s = np.cos(yaw*weight), np.sin(yaw*weight)
        linear = np.exp(log_scale*weight)*np.array([[c, -s], [s, c]])
        step = np.eye(3)
        step[:2, :2] = linear
        step[:2, 2] = center+shift*weight-linear@center
        pose = step@pose
        projected[k] = pose
    if not np.isfinite(observed).all() or not np.isfinite(projected).all():
        raise ValueError("Nonfinite camera composition")
    return observed, projected


ARMS = ("raw_cv", "raw_ols4", "stable_cv", "stable_ols4", "camera_cv", "camera_ols4")


def compare(history, steps, center, accepted):
    """All returns are displacements in the original future image coordinates."""
    cv, ols, _, _ = rollout(history, 12)
    if len(accepted) != 7:
        raise ValueError("Support is defined over exactly seven observed pairs")
    if not all(accepted):
        return dict(zip(ARMS, (cv, ols, cv, ols, cv, ols), strict=True))
    observed, projected = camera_maps(steps, center)
    stable = transport(history, observed)
    a, b, _, _ = rollout(stable, 12)
    origin = history[:, -1:]
    a0, b0 = (transport(origin+p, projected)-origin for p in (a, b))
    return dict(zip(ARMS, (cv, ols, a, b, a0, b0), strict=True))
