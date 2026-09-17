"""Past-only moving controls and image correspondence, not forecast features."""
from __future__ import annotations

import hashlib
import numpy as np
from scipy.signal import correlate2d

from src.evaluation.m3w_past_video_alignment import native_to_image_xy, past_frame_indices


def select_moving_controls(points, homography, recording_id, *, max_agents=12, min_motion=12.0):
    """First qualifying observed history per agent; no future label/index API."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
        raise ValueError('Finite [frame, agent, x, y] rows required')
    candidates = []
    for agent in np.unique(points[:, 1]):
        if agent != int(agent):
            raise ValueError('Integer agent identifiers required')
        track = points[points[:, 1] == agent]
        track = track[np.argsort(track[:, 0], kind='stable')]
        for end in range(7, len(track)):
            past = track[end - 7:end + 1]
            dt = np.diff(past[:, 0])
            if np.any(dt <= 0) or not np.all(dt == dt[-1]):
                continue
            times = past_frame_indices(past[-1, 0], past[:, 0])
            xy = native_to_image_xy(past[:, 2:4], homography, projected_axes='row_col')
            motion = float(np.linalg.norm(xy[-1] - xy[0]))
            if motion < min_motion:
                continue
            candidates.append({'agent_id': int(agent), 'frame_id': int(times[-1]),
                               'history_frames': times.tolist(), 'image_xy': xy.tolist(),
                               'history_net_motion_px': motion})
            break
    candidates.sort(key=lambda r: hashlib.sha256(f'{recording_id}/{r["agent_id"]}'.encode()).hexdigest())
    return candidates[:max_agents], len(candidates)


def centered_patch(image, image_xy, size):
    """Return full fixed patch or None; never fill missing pixels as observations."""
    a = np.asarray(image)
    xy = np.asarray(image_xy, dtype=float)
    if a.ndim not in (2, 3) or xy.shape != (2,) or not np.isfinite(xy).all() or size < 1:
        raise ValueError('Image, finite xy and positive size required')
    x, y = (int(v) for v in np.rint(xy))
    left, top = x - size // 2, y - size // 2
    if left < 0 or top < 0 or left + size > a.shape[1] or top + size > a.shape[0]:
        return None
    return a[top:top + size, left:left + size].copy()


def match_past_patch(earlier, later, earlier_xy, *, template_size=15, search_radius=24):
    """ZNCC in an earlier-point-centered search, with no later annotation input."""
    if template_size < 3 or template_size % 2 != 1 or search_radius < 1:
        raise ValueError('Odd template and positive search radius required')
    earlier, later = np.asarray(earlier, float), np.asarray(later, float)
    if earlier.ndim != 2 or later.shape != earlier.shape or not np.isfinite(earlier).all() or not np.isfinite(later).all():
        raise ValueError('Finite matching grayscale frames required')
    template = centered_patch(earlier, earlier_xy, template_size)
    region = centered_patch(later, earlier_xy, template_size + 2 * search_radius)
    if template is None or region is None:
        return {'status': 'out_of_image_support'}
    template -= template.mean()
    template_norm = float(np.linalg.norm(template))
    if template_norm < 1e-8:
        return {'status': 'flat_template'}
    ones = np.ones_like(template)
    sums = correlate2d(region, ones, mode='valid')
    energy = np.maximum(correlate2d(region ** 2, ones, mode='valid') - sums ** 2 / template.size, 0)
    denominator = np.sqrt(energy) * template_norm
    scores = np.full_like(denominator, -np.inf)
    np.divide(correlate2d(region, template, mode='valid'), denominator,
              out=scores, where=denominator > 1e-8)
    if not np.isfinite(scores).any():
        return {'status': 'flat_search'}
    peak = float(np.max(scores))
    iy, ix = np.nonzero(np.isclose(scores, peak, rtol=0, atol=1e-12))
    nearest = np.lexsort((ix, iy, (ix-search_radius)**2 + (iy-search_radius)**2))[0]
    delta = np.array([ix[nearest] - search_radius, iy[nearest] - search_radius])
    location = np.rint(earlier_xy) + delta
    return {'status': 'matched', 'peak_zncc': peak, 'image_xy': location.tolist(),
            'shift_xy': delta.tolist(), 'zero_shift_zncc': float(scores[search_radius, search_radius])
            if np.isfinite(scores[search_radius, search_radius]) else None}
