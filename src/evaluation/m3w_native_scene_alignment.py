"""Past-only common-frame reconstruction for the frozen native source cohort."""
import numpy as np


def restore(local, origin, rotation, scale):
    p, o, r, s = map(np.asarray, (local, origin, rotation, scale))
    n = len(p)
    if (p.ndim < 3 or p.shape[-1] != 2 or o.shape != (n, 2)
            or r.shape != (n, 2, 2) or s.shape != (n,)
            or not all(np.isfinite(x).all() for x in (p, o, r, s)) or np.any(s <= 0)):
        raise ValueError('Aligned finite local positions and causal transforms required')
    np.testing.assert_allclose(r @ r.transpose(0, 2, 1), np.broadcast_to(np.eye(2), r.shape), atol=1e-12)
    flat = p.reshape(n, -1, 2).astype(float)
    return (np.einsum('nti,nji->ntj', flat*s[:, None, None], r) + o[:, None]).reshape(p.shape)


def past_transforms(query_keys, image_rows, crop_keys, boxes, geometry, scale):
    """Resolve each query through its eight *past* annotation-box identities."""
    keys, rows, crop, boxes, g, supplied = map(np.asarray,
        (query_keys, image_rows, crop_keys, boxes, geometry, scale))
    n = len(keys)
    if (keys.shape != (n, 2) or rows.shape != (n, 8) or crop.ndim != 2 or crop.shape[1] != 2
            or boxes.shape != (len(crop), 4) or g.shape != (n, 476) or supplied.shape != (n,)
            or keys.dtype.kind not in 'iu' or rows.dtype.kind not in 'iu'
            or crop.dtype.kind not in 'iu' or np.any(rows < 0) or np.any(rows >= len(crop))
            or not np.isfinite(boxes).all() or np.any(boxes[:, 2:] < boxes[:, :2])
            or not np.isfinite(g).all() or not np.isfinite(supplied).all() or np.any(supplied <= 0)):
        raise ValueError('Complete frozen past geometry/provenance required')
    if len(np.unique(keys, axis=0)) != n or len(np.unique(crop, axis=0)) != len(crop):
        raise ValueError('Duplicate frame/agent key')
    times = keys[:, 0, None] - np.arange(7, -1, -1)*12
    np.testing.assert_array_equal(crop[rows, 0], times)
    np.testing.assert_array_equal(crop[rows, 1], np.broadcast_to(keys[:, 1, None], rows.shape))
    b = boxes[rows].astype(float)
    history = (b[:, :, :2]+b[:, :, 2:])/2
    velocity = (history[:, -1]-history[:, -2])/12
    speed = np.linalg.norm(velocity, axis=1)
    heading = np.where(speed > 1e-8, np.arctan2(velocity[:, 1], velocity[:, 0]), 0.)
    c, s = np.cos(heading), np.sin(heading)
    rotation = np.stack((c, -s, s, c), axis=1).reshape(n, 2, 2)
    path = np.linalg.norm(np.diff(history, axis=1), axis=-1).sum(1)
    scale = np.maximum(np.maximum(path, speed*144), 1e-3)
    # Existing source stores causal_features[2] as float32, then casts to float64.
    np.testing.assert_array_equal(supplied, scale.astype(np.float32).astype(float))
    origin = history[:, -1]
    expected_local = np.einsum('nti,nij->ntj', history-origin[:, None], rotation)/scale[:, None, None]
    np.testing.assert_allclose(g[:, :16], expected_local.astype(np.float32).reshape(n, 16), rtol=2e-7, atol=1e-7)
    cv = velocity[:, None]*np.arange(1, 13)[None, :, None]*12
    cv_local = np.einsum('nti,nij->ntj', cv, rotation)/scale[:, None, None]
    np.testing.assert_allclose(g[:, 332:356], cv_local.astype(np.float32).reshape(n, 24), rtol=2e-7, atol=1e-7)
    np.testing.assert_array_equal(g[:, 16:24], np.broadcast_to((-np.arange(7, -1, -1)/12).astype(np.float32), (n, 8)))
    local = g[:, :16].reshape(n, 8, 2)
    actual = restore(local, origin, rotation, supplied)
    exact_scale_reconstruction = restore(local, origin, rotation, scale)
    return dict(origin=origin, rotation=rotation, annotation_scale=scale,
        stored_metric_scale=supplied, history_native=history,
        current_box=b[:, -1],
        max_history_error_stored_scale=float(np.max(np.abs(actual-history))),
        max_history_error_annotation_scale=float(np.max(np.abs(exact_scale_reconstruction-history))),
        max_scale_relative_rounding=float(np.max(np.abs(supplied/scale-1))))


def scene_index(recordings, frames, tracks):
    rec, frames, tracks = map(np.asarray, (recordings, frames, tracks))
    if (rec.ndim != 1 or frames.shape != rec.shape or tracks.shape != rec.shape
            or frames.dtype.kind not in 'iu' or np.any(frames < 0)):
        raise ValueError('Aligned recording-scoped query identities required')
    order = np.lexsort((tracks, frames, rec))
    duplicate = ((rec[order][1:] == rec[order][:-1]) & (frames[order][1:] == frames[order][:-1])
                 & (tracks[order][1:] == tracks[order][:-1]))
    if duplicate.any():
        raise ValueError('Duplicate scene query agent')
    change = (rec[order][1:] != rec[order][:-1]) | (frames[order][1:] != frames[order][:-1])
    offsets = np.r_[0, np.flatnonzero(change)+1, len(order)].astype(np.int64)
    group = np.empty(len(order), np.int64)
    group[order] = np.repeat(np.arange(len(offsets)-1), np.diff(offsets))
    return dict(order=order, offsets=offsets, group=group,
        recordings=rec[order[offsets[:-1]]], frames=frames[order[offsets[:-1]]])


def resolve_target_neighbors(geometry, origin, rotation, scale, index, tolerance=1e-3):
    """Map current neighbor slots to eligible forecast queries; never invent IDs.

    -1 denotes no eligible target match, -2 multiple matches. This is not an
    inventory of all visible agents. Pixel tolerance is float32 reconstruction
    QA only, not a prediction-error or safety tolerance.
    """
    g = np.asarray(geometry)
    n = len(g)
    if g.shape != (n, 476) or not np.isfinite(g).all() or tolerance <= 0:
        raise ValueError('Frozen finite past geometry required')
    mask = g[:, 230:294].reshape(n, 8, 8)
    if not np.isin(mask, [0, 1]).all():
        raise ValueError('Explicit binary neighbor observation mask required')
    last = mask[:, :, -1].astype(bool)
    nb = g[:, 38:166].reshape(n, 8, 8, 2)
    positions = restore(nb[:, :, -1], origin, rotation, scale)
    mapping = np.full((n, 8), -1, np.int64)
    max_error = 0.
    for start, end in zip(index['offsets'][:-1], index['offsets'][1:]):
        ids = index['order'][start:end]
        delta = np.linalg.norm(positions[ids, :, None]-origin[ids][None, None], axis=-1)
        # A neighbor is never its own target, even with coincident coordinates.
        delta[np.arange(len(ids)), :, np.arange(len(ids))] = np.inf
        matches = (delta <= tolerance) & last[ids, :, None]
        count = matches.sum(-1)
        which = np.argmin(delta, axis=-1)
        one = count == 1
        mapping[ids] = np.where(one, ids[which], np.where(count > 1, -2, -1))
        if one.any():
            max_error = max(max_error, float(np.take_along_axis(delta, which[..., None], -1)[..., 0][one].max()))
    return dict(target_row=mapping, observed=last, max_unique_match_error=max_error)


def support_counts(selected, valid, group, tracks):
    use, valid, group, tracks = map(np.asarray, (selected, valid, group, tracks))
    if (use.ndim != 1 or use.dtype != bool or valid.shape != (len(use), 12) or valid.dtype != bool
            or group.shape != use.shape or tracks.shape != use.shape):
        raise ValueError('Aligned selection and explicit label masks required')
    count = valid.sum(1)
    return dict(selected_rows=int(use.sum()), unique_selected_tracks=len(np.unique(tracks[use])),
        selected_scene_queries=len(np.unique(group[use])),
        complete_selected=int((use & (count == 12)).sum()),
        partial_selected=int((use & (count > 0) & (count < 12)).sum()),
        unknown_selected=int((use & (count == 0)).sum()),
        label_count_histogram={str(k):int((use & (count == k)).sum()) for k in range(13)})
