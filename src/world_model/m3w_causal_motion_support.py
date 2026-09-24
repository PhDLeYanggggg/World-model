"""Exact native-coordinate past-motion support, without outcome inputs."""
import numpy as np


def motion_profile(history, offsets):
    xy, t = np.asarray(history, dtype=np.float64), np.asarray(offsets)
    if (xy.ndim != 3 or xy.shape[1:] != (8, 2) or t.shape != xy.shape[:2]
            or not np.isfinite(xy).all() or not np.isfinite(t).all()
            or np.any(t > 0) or np.any(t[:, -1] != 0)
            or np.any(np.diff(t, axis=1) <= 0)
            or np.any(t != np.rint(t)) or np.any(np.abs(t) > 2**20)
            or np.any(2*xy != np.rint(2*xy)) or np.any(np.abs(xy) > 2**20)):
        raise ValueError('Eight complete finite past positions and increasing query-relative times required')
    dt = np.diff(t.astype(np.float64), axis=1)
    dx = np.diff(xy, axis=1)
    last = dx[:, -1]
    # Bounded half-pixel coordinates and integer annotation frames make these
    # cross-products exactly representable; arbitrary normalized floats fail.
    consistent = np.all(dx * dt[:, -1, None, None] == last[:, None] * dt[..., None], axis=(1, 2))
    backcast = xy[:, -1:] + t[..., None] * (last / dt[:, -1, None])[:, None]
    residual = np.linalg.norm(xy-backcast, axis=-1)
    path = np.linalg.norm(dx, axis=-1).sum(1)
    return dict(last_stop=np.all(last == 0, axis=1),
        stationary=np.all(dx == 0, axis=(1, 2)), exact_past_cv=consistent,
        last_step_distance=np.linalg.norm(last, axis=1), path_length=path,
        max_backcast_error=residual.max(1),
        relative_backcast_error=residual.max(1)/np.maximum(path, 1e-3))


def veto_eligibility(eligible, exact_past_cv):
    support, veto = np.asarray(eligible), np.asarray(exact_past_cv)
    if support.dtype != bool or veto.dtype != bool or support.ndim != 1 or veto.shape != support.shape:
        raise ValueError('Aligned boolean causal eligibility and past-CV flags required')
    return support & ~veto
