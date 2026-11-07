"""Geometry-only visibility proxies from observed MOT head boxes."""
import numpy as np


def frame_features(boxes):
    lo, size = boxes[:, :2], boxes[:, 2:]
    hi, center = lo+size, lo+size/2.
    area = size.prod(1)
    if not len(boxes) or not np.isfinite(boxes).all() or (size <= 0).any():
        raise ValueError('Finite, positive observed boxes required')
    out = np.zeros((len(boxes), 6))
    out[:, 0] = np.log(area)
    out[:, 1] = np.log(size[:, 0]/size[:, 1])
    out[:, 2] = np.log1p(len(boxes)-1)
    for a in range(0, len(boxes), 128):
        ix = np.arange(a, min(a+128, len(boxes)))
        inter = np.minimum(hi[ix, None], hi[None])-np.maximum(lo[ix, None], lo[None])
        overlap = inter.clip(0.).prod(2)/area[ix, None]
        diagonal = np.arange(len(ix)), ix
        overlap[diagonal] = 0.
        dist = np.linalg.norm(center[ix, None]-center[None], axis=2)/np.sqrt(area[ix, None])
        dist[diagonal] = np.inf
        out[ix, 3] = overlap.max(1)
        out[ix, 4] = np.log1p((overlap > 0.).sum(1))
        out[ix, 5] = np.log1p(dist.min(1).clip(max=50.))
    return out


def summarize(history):
    if history.shape[1:] != (8, 6):
        raise ValueError('Exactly eight observed frames required')
    return np.column_stack((history[:, -1], history.mean(1), history.std(1),
        history[:, -1]-history[:, 0], history[:, :, 3:].max(1))).astype(np.float32)


def aligned(rows, agents, frames, centers):
    """Read only frame-local geometry at query and the previous seven steps."""
    per_frame = {}
    for f in np.unique(frames[:, None]-np.arange(8)):
        v = rows[rows[:, 0] == f]
        v = v[np.argsort(v[:, 1], kind='stable')]
        keys, boxes = v[:, 1].astype(np.int64), v[:, 2:6]
        if len(np.unique(keys)) != len(keys):
            raise ValueError('Duplicate released box identity')
        per_frame[f] = keys, boxes[:, :2]+boxes[:, 2:]/2., frame_features(boxes)
    x = np.empty((len(agents), 8, 6))
    for q in np.unique(frames):
        i = np.flatnonzero(frames == q)
        for j, f in enumerate(range(q-7, q+1)):
            keys, xy, feat = per_frame[f]
            at = np.searchsorted(keys, agents[i])
            if (at == len(keys)).any() or not np.array_equal(keys[at], agents[i]):
                raise ValueError('Box/trajectory identity mismatch')
            np.testing.assert_array_equal(xy[at], centers[i, j])
            x[i, j] = feat[at]
    return summarize(x)
