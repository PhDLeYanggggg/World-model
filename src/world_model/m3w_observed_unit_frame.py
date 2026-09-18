"""Reversible internal conditioning; never modifies the evaluation coordinate scale."""
from __future__ import annotations

import numpy as np

DIM = 476
VECTOR_BLOCKS = ((0, 16), (24, 38), (38, 166), (308, 476))
SUMMARY_NAMES = (
    'last_speed_per_horizon_over_context', 'history_path_over_context',
    'spatial_anchor_present', 'last_acceleration_per_horizon2_over_context',
    'last_valid_motion_turn', 'turn_sum_over_normalized_path',
    'all_current_neighbor_count', 'nearest_selected_current_neighbor_over_context',
    'selected_neighbors_within_ego_motion_radius', 'minimum_tca_over_horizon',
    'maximum_closing_per_horizon_over_context', 'exact_stationary_history',
    'unit_horizon', 'last_observed_step_over_horizon',
)


def observed_unit_frame(geometry):
    """Return internal features, restoration radius/rotation and observed support.

    Only the frozen 476-column past payload is accepted. Its native-unit scalar
    summaries are reconstructed from typed positions/times, not rescaled blindly.
    Zero spatial context has no identifiable positive length; its delta is zero.
    """
    x = np.asarray(geometry, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != DIM or not np.isfinite(x).all():
        raise ValueError('Finite frozen 476-column geometry required')
    n = len(x)
    h = x[:, :16].reshape(n, 8, 2)
    ht = x[:, 16:24]
    pos = x[:, 38:166].reshape(n, 8, 8, 2)
    times = x[:, 166:230].reshape(n, 8, 8)
    raw_mask = x[:, 230:294].reshape(n, 8, 8)
    if not np.isin(raw_mask, [0, 1]).all():
        raise ValueError('Binary neighbor mask required')
    mask = raw_mask.astype(bool)
    if (np.any(ht > 0) or np.any(times[mask] > 0) or np.any(np.diff(ht, axis=1) <= 0)
            or not np.all(ht[:, -1] == 0)):
        raise ValueError('Increasing, query-aligned past times required')
    pos = np.where(mask[..., None], pos, 0.)
    dh = np.diff(h, axis=1)
    dt = np.diff(ht, axis=1)
    velocity = dh/dt[..., None]
    paired = mask[:, :, 1:] & mask[:, :, :-1]
    ndt = np.diff(times, axis=2)
    if np.any(ndt[paired] <= 0):
        raise ValueError('Increasing valid neighbor past times required')
    nv = np.divide(np.diff(pos, axis=2), ndt[..., None],
                   out=np.zeros((n, 8, 7, 2)), where=paired[..., None])

    radius = np.maximum(np.linalg.norm(h, axis=-1).max(1),
                        np.linalg.norm(pos, axis=-1).max((1, 2)))
    support = radius > 0
    divisor = np.where(support, radius, 1.)

    # Canonical direction uses last ego motion, then neighbor motion or position.
    direction = np.zeros((n, 2))
    moving = np.linalg.norm(dh, axis=-1) > 0
    last = 6-np.argmax(moving[:, ::-1], axis=1)
    ego = moving.any(1)
    direction[ego] = dh[np.flatnonzero(ego), last[ego]]
    neigh_moving = np.linalg.norm(nv, axis=-1) > 0
    last_n = 6-np.argmax(neigh_moving[:, :, ::-1], axis=2)
    has_n = neigh_moving.any(2)
    first_n = np.argmax(has_n, axis=1)
    use_n = ~ego & has_n.any(1)
    ii = np.flatnonzero(use_n); jj = first_n[use_n]
    direction[ii] = nv[ii, jj, last_n[ii, jj]]
    flat_pos = pos.reshape(n, 64, 2)
    has_pos = np.linalg.norm(flat_pos, axis=-1) > 0
    use_pos = ~ego & ~use_n & has_pos.any(1)
    ii = np.flatnonzero(use_pos)
    direction[ii] = flat_pos[ii, np.argmax(has_pos[ii], axis=1)]
    norm = np.linalg.norm(direction, axis=-1)
    unit = np.divide(direction, norm[:, None], out=np.zeros_like(direction), where=norm[:, None] > 0)
    unit[norm == 0] = [1., 0.]
    q = np.stack((unit, np.stack((-unit[:, 1], unit[:, 0]), axis=1)), axis=2)
    def local(v):
        shape = v.shape
        return (np.einsum('nvd,ndk->nvk', v.reshape(n, -1, 2), q)/divisor[:, None, None]).reshape(shape)

    lh, ln, lv = local(h), local(pos), local(velocity)
    out = np.zeros_like(x)
    out[:, :16], out[:, 16:24], out[:, 24:38] = lh.reshape(n, -1), ht, lv.reshape(n, -1)
    out[:, 38:166] = ln.reshape(n, -1)
    out[:, 166:230] = np.where(mask, times, 0.).reshape(n, -1)
    out[:, 230:294] = raw_mask.reshape(n, -1)
    out[:, 308:] = local(x[:, 308:].reshape(n, 7, 12, 2)).reshape(n, -1)
    path = np.linalg.norm(np.diff(lh, axis=1), axis=-1).sum(1)
    headings = np.arctan2(lv[:, :, 1], lv[:, :, 0])
    da = np.diff(headings, axis=1)
    angle = np.where(moving[:, 1:] & moving[:, :-1], np.arctan2(np.sin(da), np.cos(da)), 0.)
    acceleration = np.linalg.norm((lv[:, -1]-lv[:, -2])/(.5*(dt[:, -1]+dt[:, -2]))[:, None], axis=1)
    current = mask[:, :, -1] & (times[:, :, -1] == 0)
    distance = np.linalg.norm(ln[:, :, -1], axis=-1)
    nearest = np.min(np.where(current, distance, np.inf), axis=1)
    nearest[~current.any(1)] = 0.
    last_pair = paired[:, :, -1] & current
    rel_v = local(nv[:, :, -1])-lv[:, None, -1]
    dot = (ln[:, :, -1]*rel_v).sum(-1)
    vv = (rel_v*rel_v).sum(-1)
    tca = np.divide(-dot, vv, out=np.full((n, 8), 10.), where=last_pair & (vv > 0) & (dot < 0))
    closing = np.divide(-dot, distance, out=np.zeros((n, 8)), where=last_pair & (distance > 0))
    motion_radius = np.maximum(path, np.linalg.norm(lv[:, -1], axis=1))
    out[:, 294:308] = np.column_stack((
        np.linalg.norm(lv[:, -1], axis=1), path, support, acceleration, angle[:, -1],
        np.divide(np.abs(angle).sum(1), path, out=np.zeros(n), where=path > 0),
        x[:, 300], nearest, (current & (distance <= motion_radius[:, None])).sum(1),
        np.minimum(tca, 10.).min(1), np.maximum(closing, 0.).max(1),
        ~moving.any(1), np.ones(n), dt[:, -1],
    ))
    if not np.isfinite(out).all():
        raise FloatingPointError('Internal conditioning produced nonfinite features')
    return out.astype(np.float32), radius.astype(np.float32), q.astype(np.float32), support


def restore_delta(delta, radius, rotation, support):
    delta, radius, rotation, support = map(np.asarray, (delta, radius, rotation, support))
    if (delta.ndim != 3 or delta.shape[1:] != (12, 2) or radius.shape != (len(delta),)
            or rotation.shape != (len(delta), 2, 2) or support.shape != (len(delta),)):
        raise ValueError('Aligned twelve-step delta and past restoration frame required')
    return np.where(support[:, None, None],
        np.einsum('nvd,nkd->nvk', delta, rotation)*radius[:, None, None], 0.)
