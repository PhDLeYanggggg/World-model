"""Past crop-motion proxies, with crop translation and supplied-H axes restored."""
from __future__ import annotations

import numpy as np

VARIANTS = ('quality_control', 'magnitude', 'directed')
VECTOR_NAMES = ('center_dx', 'center_dy', 'ring_dx', 'ring_dy', 'contrast_dx', 'contrast_dy')
MAGNITUDE_NAMES = ('center_q50', 'center_q90', 'ring_q50', 'ring_q90')
QUALITY_NAMES = ('center_coverage', 'ring_coverage', 'center_consistent', 'ring_consistent', 'pair_available')


def full_image_correspondence(flow, first_center, second_center):
    flow = np.asarray(flow, np.float64)
    if flow.shape != (32, 32, 2) or not np.isfinite(flow).all():
        raise ValueError('Finite 32x32 crop flow required')
    centers = np.asarray([first_center, second_center], np.float64)
    if centers.shape != (2, 2) or not np.isfinite(centers).all():
        raise ValueError('Finite image xy centers required')
    yy, xx = np.mgrid[:32, :32]
    source = np.stack((xx, yy), -1) * 3 + 1 + np.rint(centers[0]) - 48
    destination = source + flow * 3 + np.rint(centers[1]) - np.rint(centers[0])
    return source, destination


def image_to_native(xy, matrix):
    """Same supplied row/column convention as source adapters; no metric claim."""
    xy, matrix = np.asarray(xy, float), np.asarray(matrix, float)
    if xy.shape[-1] != 2 or matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError('Image xy and finite 3x3 supplied H required')
    q = np.concatenate((xy[..., ::-1], np.ones(xy.shape[:-1] + (1,))), -1) @ matrix.T
    valid = np.isfinite(q).all(-1) & (np.abs(q[..., 2]) > 1e-9)
    result = np.zeros_like(xy, dtype=float)
    np.divide(q[..., :2], q[..., 2:], out=result, where=valid[..., None])
    return result, valid


def pair_motion(rgb_first, rgb_second, coverage_first, coverage_second,
                first_center, second_center, matrix, cv):
    """Summaries only from the two supplied observed frames; no target argument."""
    for rgb in (rgb_first, rgb_second):
        if rgb.shape != (3, 32, 32) or rgb.dtype != np.uint8:
            raise ValueError('Use the bound uint8 CHW image cache')
    for coverage in (coverage_first, coverage_second):
        if coverage.shape != (32, 32) or np.any(coverage > 9):
            raise ValueError('Expected 3x3 source-pixel coverage counts')
    yy, xx = np.mgrid[:32, :32].astype(np.float32)
    center = (xx >= 8) & (xx < 24) & (yy >= 8) & (yy < 24)
    ring = (xx >= 4) & (xx < 28) & (yy >= 4) & (yy < 28) & ~center
    quality = np.zeros(5, np.float32)
    quality[:2] = [(coverage_first[m] / 9).mean() for m in (center, ring)]
    vectors, magnitude = np.zeros((3, 2), np.float64), np.zeros(4, np.float64)
    if not coverage_first.any() or not coverage_second.any():
        return vectors, magnitude, quality
    first = cv.cvtColor(rgb_first.transpose(1, 2, 0), cv.COLOR_RGB2GRAY)
    second = cv.cvtColor(rgb_second.transpose(1, 2, 0), cv.COLOR_RGB2GRAY)
    params = (.5, 3, 15, 3, 5, 1.2, 0)
    forward = cv.calcOpticalFlowFarneback(first, second, None, *params)
    backward = cv.calcOpticalFlowFarneback(second, first, None, *params)
    mx, my = xx + forward[..., 0], yy + forward[..., 1]
    reverse = cv.remap(backward, mx, my, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    next_coverage = cv.remap(coverage_second.astype(np.float32), mx, my,
                             cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    source, destination = full_image_correspondence(forward, first_center, second_center)
    old, old_valid = image_to_native(source, matrix)
    new, new_valid = image_to_native(destination, matrix)
    displacement = new - old
    valid = ((coverage_first == 9) & (next_coverage >= 8.999) & (mx >= 1) & (mx <= 30)
             & (my >= 1) & (my <= 30) & (np.linalg.norm(forward + reverse, axis=-1) <= 1.5)
             & old_valid & new_valid & np.isfinite(displacement).all(-1))
    quality[2:4] = [valid[m].mean() for m in (center, ring)]
    quality[4] = 1.
    supported = []
    for i, region in enumerate((center, ring)):
        points = displacement[valid & region]
        enough = len(points) >= 16
        supported.append(enough)
        if enough:
            vectors[i] = np.median(points, axis=0)
            magnitude[i * 2:i * 2 + 2] = np.quantile(np.linalg.norm(points, axis=1), [.5, .9])
    if all(supported):
        vectors[2] = vectors[0] - vectors[1]
    return vectors, magnitude, quality


def normalized_motion(vectors, magnitudes, rotation, scale):
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError('Past-only positive scale required')
    return np.concatenate(((np.asarray(vectors) @ rotation / scale).reshape(-1),
                           np.asarray(magnitudes) / scale)).astype(np.float32)


def feature_variant(geometry, motion, quality, variant):
    if variant not in VARIANTS or motion.shape[1:] != (7, 10) or quality.shape[1:] != (7, 5):
        raise ValueError('Registered seven-pair feature schema required')
    selected = motion.copy()
    if variant == 'quality_control':
        selected[:] = 0
    elif variant == 'magnitude':
        selected[:, :, :6] = 0
    return np.concatenate((geometry, quality.reshape(len(geometry), -1),
                           selected.reshape(len(geometry), -1)), axis=1).astype(np.float32)


def validate_past_join(history, image_ids, query, expected_ids):
    history, image_ids = np.asarray(history), np.asarray(image_ids)
    if (history.shape != (8, 4) or image_ids.shape != (8,)
            or not np.array_equal(image_ids, expected_ids)
            or not np.all(history[:, 1] == query['agent'])
            or history[-1, 0] != query['frame'] or not np.all(np.diff(history[:, 0]) > 0)
            or np.any(history[:, 0] > query['frame'])):
        raise ValueError('Past source-row/image/query alignment failed')
    return True
