"""Causal image-region context, not a walkability or metric calibration map."""
import csv
import io
from pathlib import PurePosixPath
import zipfile

import numpy as np


def source_polygons(archive, records):
    """Bind camera IDs from exact raw members, never from locality grouping."""
    result = {}
    with zipfile.ZipFile(archive) as z:
        members = [n for n in z.namelist()
                   if PurePosixPath(n).parent.as_posix() == '02.Resources/02.Place_polygons'
                   and n.endswith('_place_polygon.csv')]
        for record in records:
            square = int(PurePosixPath(record).name.split('_', 1)[0])
            candidates = [n for n in members if int(PurePosixPath(n).name.split('_', 1)[0]) == square]
            if len(candidates) != 1:
                raise ValueError('One exact camera polygon required')
            rows = list(csv.DictReader(io.StringIO(z.read(candidates[0]).decode('utf-8-sig'))))
            xy = np.array([[float(r['x']), float(r['y'])] for r in rows])
            if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or not np.isfinite(xy).all():
                raise ValueError('Finite image-coordinate polygon required')
            if np.array_equal(xy[0], xy[-1]):
                xy = xy[:-1]
            if np.any(np.ptp(xy, axis=0) <= 0):
                raise ValueError('Nondegenerate image polygon required')
            result[record] = dict(member=candidates[0], vertices=xy)
    return result


def signed_distance(points, polygon):
    p, v = np.asarray(points, float), np.asarray(polygon, float)
    if (p.ndim != 2 or p.shape[1] != 2 or v.ndim != 2 or v.shape[1] != 2
            or len(v) < 3 or not np.isfinite(p).all() or not np.isfinite(v).all()):
        raise ValueError('Finite query points and polygon required')
    a, b = v, np.roll(v, -1, axis=0)
    edge = b-a
    length = np.square(edge).sum(1)
    if np.any(length == 0):
        raise ValueError('Repeated polygon vertex')
    delta = p[:, None]-a
    t = np.clip((delta*edge).sum(-1)/length, 0, 1)
    distance = np.linalg.norm(delta-t[..., None]*edge, axis=-1).min(1)
    # A strict y straddle excludes horizontal edges without a division by zero.
    straddle = (a[None, :, 1] > p[:, None, 1]) != (b[None, :, 1] > p[:, None, 1])
    ratio = np.divide(p[:, None, 1]-a[None, :, 1], edge[None, :, 1],
                      out=np.zeros((len(p), len(v))), where=edge[None, :, 1] != 0)
    cross_x = a[None, :, 0]+ratio*edge[None, :, 0]
    inside = np.logical_xor.reduce(straddle & (p[:, None, 0] < cross_x), axis=1)
    return np.where(inside, distance, -distance)


def scene_features(*, history, polygon, rollout):
    """18 static/past/predicted features; no future label or availability input."""
    h, v, r = np.asarray(history, float), np.asarray(polygon, float), np.asarray(rollout, float)
    if h.shape != (len(h), 8, 2) or r.shape != (len(h), 12, 2) or not np.isfinite(h).all() or not np.isfinite(r).all():
        raise ValueError('Complete observed history and model rollout required')
    lower, span = v.min(0), np.ptp(v, axis=0)
    if (span <= 0).any():
        raise ValueError('Nondegenerate static polygon bounds required')
    diagonal = np.linalg.norm(span)
    current = h[:, -1]
    history_distance = signed_distance(h.reshape(-1, 2), v).reshape(-1, 8)/diagonal
    rollout_distance = signed_distance(r.reshape(-1, 2), v).reshape(-1, 12)/diagonal
    current_normalized = (current-lower)/span
    start_normalized = (h[:, 0]-lower)/span
    velocity = (current-h[:, -2])/span
    mean_velocity = (current-h[:, 0])/(7*span)
    edge_time = history_distance[:, -1]-history_distance[:, -2]
    columns = [current_normalized, start_normalized, velocity, mean_velocity,
               history_distance[:, [0, -1]], edge_time[:, None],
               (history_distance >= 0).mean(1)[:, None],
               rollout_distance[:, [0, 5, 11]],
               (rollout_distance >= 0).mean(1)[:, None],
               rollout_distance.min(1)[:, None], rollout_distance.mean(1)[:, None]]
    x = np.column_stack(columns).astype(np.float32)
    if x.shape != (len(h), 18) or not np.isfinite(x).all():
        raise ValueError('Finite 18-field scene proxy required')
    return x
