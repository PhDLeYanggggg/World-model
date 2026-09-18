"""Training-side annotation magnitude and past-context diagnostics, not filtering."""
import numpy as np


PIXEL_EDGES = (0., 1., 2., 5., 10.)
BOX_EDGES = (0., .01, .05, .1, .25, .5, 1.)


def past_box_features(history_boxes):
    boxes = np.asarray(history_boxes, dtype=float)
    if boxes.ndim != 3 or boxes.shape[1:] != (8, 4) or not np.isfinite(boxes).all():
        raise ValueError('Finite eight-step past boxes required')
    size = boxes[:, :, 2:] - boxes[:, :, :2]
    if np.any(size <= 0):
        raise ValueError('Positive observed box dimensions required')
    scale = np.median(np.linalg.norm(size, axis=-1), axis=1)
    relative = size / scale[:, None, None]
    aspect = np.log(size[:, :, 0] / size[:, :, 1])
    return np.concatenate((relative.reshape(len(boxes), -1),
                           np.diff(relative, axis=1).reshape(len(boxes), -1), aspect), axis=1).astype(np.float32)


def trajectory_quality(relative_future, past_box_scale):
    future, scale = np.asarray(relative_future, float), np.asarray(past_box_scale, float)
    if (future.ndim != 3 or future.shape[1:] != (12, 2) or scale.shape != (len(future),)
            or not np.isfinite(future).all() or not np.isfinite(scale).all() or np.any(scale <= 0)):
        raise ValueError('Complete finite labels and positive past-only scale required')
    distance = np.linalg.norm(future, axis=-1)
    steps = np.diff(np.concatenate((np.zeros((len(future), 1, 2)), future), axis=1), axis=1)
    path = np.linalg.norm(steps, axis=-1).sum(1)
    end = distance[:, -1]
    peak = distance.max(1)
    changed = distance > 0
    first = np.where(changed.any(1), np.argmax(changed, axis=1) + 1, 0)
    return dict(max_pixel_displacement=peak, endpoint_pixel_displacement=end, path_pixel_length=path,
                displacement_over_past_box=peak/scale, first_changed_step=first,
                changed_steps=changed.sum(1), nonzero=changed.any(1),
                half_box_excursion=peak/scale >= .5,
                endpoint_over_path=np.divide(end, path, out=np.zeros_like(end), where=path > 0),
                returned_to_origin=changed.any(1) & (end == 0),
                final_four_outside_half_box=(distance[:, -4:] / scale[:, None] >= .5).all(1),
                half_pixel_grid_fraction=np.isclose(future*2, np.rint(future*2), atol=1e-6, rtol=0).mean((1, 2)))


def bins(values, edges):
    x, edges = np.asarray(values), np.asarray(edges)
    if x.ndim != 1 or np.any(x < 0) or not np.isfinite(x).all() or edges[0] != 0 or np.any(np.diff(edges) <= 0):
        raise ValueError('Nonnegative finite diagnostic values and increasing edges starting zero required')
    result = {'zero': x == 0}
    for low, high in zip(edges[:-1], edges[1:]):
        result[f'({low:g},{high:g}]'] = (x > low) & (x <= high)
    result[f'>{edges[-1]:g}'] = x > edges[-1]
    if not np.all(np.stack(list(result.values())).sum(0) == 1):
        raise AssertionError('Every row must appear in one magnitude bin')
    return result


def neighbor_directions(geometry):
    g = np.asarray(geometry, float)
    if g.ndim != 2 or g.shape[1] != 476 or not np.isfinite(g).all():
        raise ValueError('Frozen finite past-only geometry required')
    p = g[:, 38:166].reshape(-1, 8, 8, 2)
    t = g[:, 166:230].reshape(-1, 8, 8)
    m = g[:, 230:294].reshape(-1, 8, 8).astype(bool)
    valid = m[:, :, -1] & m[:, :, -2] & (t[:, :, -1] == 0) & (t[:, :, -2] < 0)
    dt = t[:, :, -1]-t[:, :, -2]
    v = np.divide(p[:, :, -1]-p[:, :, -2], dt[:, :, None],
                  out=np.zeros((len(g), 8, 2)), where=valid[:, :, None])
    moving = valid & (np.linalg.norm(v, axis=-1) > 0)
    first = moving.argmax(1)
    near = v[np.arange(len(g)), first]
    near[~moving.any(1)] = 0
    mean = v.sum(1) / np.maximum(valid.sum(1), 1)[:, None]
    current = m[:, :, -1] & (t[:, :, -1] == 0)
    first = current.argmax(1)
    to_near = p[np.arange(len(g)), first, -1].copy()
    to_near[~current.any(1)] = 0
    return {'nearest_moving_neighbor': near, 'mean_selected_neighbor_velocity': mean,
            'toward_nearest_current_neighbor': to_near}


def direction_scores(vector, endpoint):
    v, y = np.asarray(vector, float), np.asarray(endpoint, float)
    if v.shape != y.shape or v.ndim != 2 or v.shape[1] != 2 or not np.isfinite(v).all() or not np.isfinite(y).all():
        raise ValueError('Aligned finite directions required')
    denom = np.linalg.norm(v, axis=1)*np.linalg.norm(y, axis=1)
    support = denom > 0
    cosine = np.divide((v*y).sum(1), denom, out=np.zeros(len(v)), where=support)
    return support, np.clip(cosine, -1, 1)
