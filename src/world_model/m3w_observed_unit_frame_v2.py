"""Observed-unit frame with rollout diagnostics recomputed after conditioning."""
import numpy as np

from src.world_model.m3w_observed_unit_frame import observed_unit_frame as _frame_v1
from src.world_model.m3w_observed_unit_frame import SUMMARY_NAMES, restore_delta


def observed_unit_frame(geometry):
    x, radius, rotation, support = _frame_v1(geometry)
    history = x[:, :16].reshape(-1, 8, 2).astype(np.float64)
    times = x[:, 16:24].astype(np.float64)
    dt = np.diff(times, axis=1)
    velocity = np.diff(history, axis=1)/dt[..., None]
    v, position = velocity[:, -1], history[:, -1]
    accel = (v-velocity[:, -2])/(.5*(dt[:, -1]+dt[:, -2]))[:, None]
    offsets = dt[:, -1, None]*np.arange(1, 13)
    t = offsets[..., None]
    forecasts = [np.broadcast_to(position[:, None], (len(x), 12, 2)).copy(),
                 position[:, None]+t*v[:, None]]
    for decay in (.05, .10, .20):
        rate = (decay/dt[:, -1])[:, None, None]
        forecasts.append(position[:, None]-np.expm1(-rate*t)/rate*v[:, None])
    forecasts.append(position[:, None]+t*v[:, None]+.5*t*t*accel[:, None])
    heading = np.arctan2(velocity[:, -2:, 1], velocity[:, -2:, 0])
    turn = np.arctan2(np.sin(heading[:, 1]-heading[:, 0]), np.cos(heading[:, 1]-heading[:, 0]))
    omega = turn/(.5*(dt[:, -1]+dt[:, -2]))
    valid = (np.abs(omega) >= 1e-8) & (np.linalg.norm(velocity[:, -2], axis=1) >= 1e-8)
    phase = heading[:, -1, None]+omega[:, None]*offsets
    unit = np.stack((np.sin(phase)-np.sin(heading[:, -1, None]),
                     np.cos(heading[:, -1, None])-np.cos(phase)), axis=-1)
    magnitude = np.divide(np.linalg.norm(v, axis=1), omega, out=np.zeros(len(x)), where=valid)
    curved = position[:, None]+magnitude[:, None, None]*unit
    forecasts.append(np.where(valid[:, None, None], curved, forecasts[1]))
    x[:, 308:] = np.stack(forecasts, axis=1).reshape(len(x), -1)
    if not np.isfinite(x).all():
        raise FloatingPointError('Nonfinite internal rollout diagnostics')
    return x, radius, rotation, support
