"""Past-only observation diagnostics and a versioned partial-neighbor repair."""
import numpy as np


def checked_history(history, boxes):
    h, b = np.asarray(history, float), np.asarray(boxes, float)
    if h.ndim != 3 or h.shape[1:] != (8, 2) or b.shape != (len(h), 8, 4):
        raise ValueError('Complete eight-step histories and boxes required')
    if not np.isfinite(h).all() or not np.isfinite(b).all() or (b[..., 2:] < b[..., :2]).any():
        raise ValueError('Finite ordered boxes required')
    np.testing.assert_allclose(h, (b[..., :2]+b[..., 2:])/2, rtol=0, atol=0)
    return h, b


def ols_velocity(h):
    t = np.arange(h.shape[1], dtype=float)
    t -= t.mean()
    return np.einsum('ntd,t->nd', h, t)/(t@t)


def history_diagnostics(history, boxes):
    h, b = checked_history(history, boxes)
    size = b[..., 2:]-b[..., :2]
    scale = np.maximum(size[:, -1, 0], 1.)
    d = np.diff(h, axis=1)
    speed = np.linalg.norm(d, axis=2)
    v = ols_velocity(h)
    fitted = h.mean(1)[:, None]+(np.arange(8)-3.5)[None, :, None]*v[:, None]
    moving_pairs = (speed[:, :-1] > 0) & (speed[:, 1:] > 0)
    reversal = moving_pairs & ((d[:, :-1]*d[:, 1:]).sum(2) < 0)
    result = dict(
        path_over_width=speed.sum(1)/scale,
        line_residual_over_width=np.sqrt(((h-fitted)**2).sum(2).mean(1))/scale,
        last_fd_ols8_disagreement_over_width=np.linalg.norm(d[:, -1]-v, axis=1)/scale,
        width_range_over_width=np.ptp(size[..., 0], axis=1)/scale,
        height_range_over_height=np.ptp(size[..., 1], axis=1)/np.maximum(size[:, -1, 1], 1.),
        half_pixel_fraction=np.isclose(h*2, np.round(h*2), rtol=0, atol=1e-8).mean((1, 2)),
        stationary_step_fraction=(speed == 0).mean(1),
        reversal_fraction=reversal.sum(1)/np.maximum(moving_pairs.sum(1), 1),
        zero_current_width=size[:, -1, 0] == 0,
    )
    # Targets remain inside the observed prefix: this is not future ADE.
    for name, velocity in [('fd', h[:, 5]-h[:, 4]),
                           ('ols4', ols_velocity(h[:, 2:6])),
                           ('ols6', ols_velocity(h[:, :6]))]:
        pred = h[:, 5, None]+np.arange(1, 3)[None, :, None]*velocity[:, None]
        result['observed_prefix_'+name+'_error'] = np.linalg.norm(pred-h[:, 6:8], axis=2).mean(1)/scale
    return result


def masked_neighbors(inputs):
    allowed = {'query_frame', 'agent_id', 'class_id', 'history_xy', 'history_boxes',
               'history_valid', 'velocity_causal_fd', 'velocity_valid', 'target_eligible', 'baseline_cv'}
    if not set(inputs) <= allowed:
        raise ValueError('Only typed causal query inputs allowed')
    h = np.asarray(inputs['history_xy'], float)
    valid = np.asarray(inputs['history_valid'])
    ids = np.asarray(inputs['agent_id'])
    eligible = np.asarray(inputs['target_eligible'])
    if (h.shape != (len(ids), 8, 2) or valid.shape != (len(ids), 8)
            or valid.dtype != bool or eligible.dtype != bool
            or not np.array_equal(eligible, valid.all(1))
            or not valid[:, -1].all() or len(np.unique(ids)) != len(ids)
            or not np.isfinite(h[valid]).all()):
        raise ValueError('Unique current-visible agents and explicit past masks required')
    target = np.flatnonzero(eligible)
    xy = np.zeros((len(target), 8, 8, 2), dtype=np.float32)
    mask = np.zeros((len(target), 8, 8), dtype=bool)
    selected = np.full((len(target), 8), -1, dtype=np.int64)
    partial = np.zeros(len(target), dtype=np.int64)
    old_count = np.full(len(target), min(max(len(target)-1, 0), 8), dtype=np.int64)
    for j, i in enumerate(target):
        candidates = np.flatnonzero(np.arange(len(ids)) != i)
        distance = ((h[candidates, -1]-h[i, -1])**2).sum(1)
        near = candidates[np.lexsort((ids[candidates], distance))[:8]]
        m = valid[near]
        xy[j, :len(near)] = np.where(m[..., None], h[near]-h[i, -1], 0)
        mask[j, :len(near)] = m
        selected[j, :len(near)] = ids[near]
        partial[j] = (~eligible[near]).sum()
    return dict(target=target, xy=xy, valid=mask, agent_ids=selected,
                partial_neighbors=partial, legacy_neighbor_count=old_count)


def repair_geometry(legacy_geometry, neighbors):
    """Keep ego/rollout unchanged; use masked current-visible neighbor tokens."""
    g = np.asarray(legacy_geometry).copy()
    n = len(neighbors['target'])
    if g.shape != (n, 476) or g.dtype != np.float32:
        raise ValueError('Legacy float32 geometry schema required')
    mask = neighbors['valid']
    g[:, 38:166] = neighbors['xy'].reshape(n, 128)
    g[:, 166:230] = np.where(mask, np.arange(-7, 1, dtype=np.float32)/12, 0).reshape(n, 64)
    g[:, 230:294] = mask.reshape(n, 64)
    return g


def prefix_continuity(track, queries):
    """Dense raw-frame support inside [q-84,q], never after q."""
    frames = np.asarray(track['frame'])
    if (np.diff(frames) <= 0).any():
        raise ValueError('Ordered unique track frames required')
    rows = []
    for q in np.asarray(queries):
        start = np.searchsorted(frames, q-84)
        end = np.searchsorted(frames, q, side='right')
        f = frames[start:end]
        if not len(f) or f[-1] != q:
            raise ValueError('Query must be current-visible')
        rows.append((len(f)/85, int(np.max(np.diff(f), initial=0)),
                     float(np.mean(track['confidence'][start:end]))))
    return np.asarray(rows, float).reshape(-1, 3)


def summarize(values):
    x = np.asarray(values, float)
    if not len(x):
        return dict(n=0, mean=None, median=None, p90=None, p99=None)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite diagnostic')
    q = np.quantile(x, [.5, .9, .99])
    return dict(n=len(x), mean=float(x.mean()), median=float(q[0]), p90=float(q[1]), p99=float(q[2]))
