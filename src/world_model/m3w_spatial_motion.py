"""Observed flow-grid features with resolution and pooling controls."""
from __future__ import annotations

import numpy as np
from src.world_model.m3w_observed_motion import image_to_native

VARIANTS = ('quality_control', 'lowpass_pool', 'lowpass_grid', 'native_grid')


def reduce_patch(rgb, coverage):
    """Exactly reproduce the old visible-pixel 96-to-32 reduction."""
    if rgb.shape != (3, 96, 96) or coverage.shape != (96, 96):
        raise ValueError('Registered native patch shapes required')
    if rgb.dtype != np.uint8 or coverage.dtype != np.uint8 or np.any(coverage > 1):
        raise ValueError('Native RGB and binary source coverage required')
    pixels = rgb.transpose(1, 2, 0).astype(np.uint32) * coverage[..., None]
    sums = pixels.reshape(32, 3, 32, 3, 3).sum((1, 3))
    count = coverage.reshape(32, 3, 32, 3).sum((1, 3))
    means = np.zeros_like(sums, dtype=float)
    np.divide(sums, count[..., None], out=means, where=count[..., None] > 0)
    return np.rint(means).astype(np.uint8).transpose(2, 0, 1), count.astype(np.uint8)


def grid_pool(displacement, valid, coverage):
    """Cell means preserve sparse motion that a whole-center median can remove."""
    if displacement.shape != (96, 96, 2) or valid.shape != (96, 96) or coverage.shape != (96, 96):
        raise ValueError('Registered 96-pixel fields required')
    grid = np.zeros((16, 3), np.float64)
    quality = np.zeros((16, 2), np.float32)
    for row in range(4):
        for col in range(4):
            cell = np.s_[row*24:(row+1)*24, col*24:(col+1)*24]
            k, mask = row*4+col, valid[cell]
            quality[k] = [coverage[cell].mean(), mask.mean()]
            values = displacement[cell][mask]
            if len(values) >= 16:
                grid[k, :2] = values.mean(0)
                grid[k, 2] = np.linalg.norm(values, axis=1).mean()
    points = displacement[24:72, 24:72][valid[24:72, 24:72]]
    pooled = np.zeros(3)
    if len(points) >= 16:
        pooled[:2] = np.median(points, axis=0)
        pooled[2] = np.quantile(np.linalg.norm(points, axis=1), .9)
    return grid, quality, pooled


def dense_pair(first, second, coverage_first, coverage_second, centers, matrix, cv):
    """Use only two supplied observed crops, never labels or post-query frames."""
    if first.shape != (3, 96, 96) or second.shape != first.shape:
        raise ValueError('Two native-size observed crops required')
    centers = np.asarray(centers, float)
    if centers.shape != (2, 2) or not np.isfinite(centers).all():
        raise ValueError('Finite image centers required')
    if not coverage_first.any() or not coverage_second.any():
        return np.zeros((16, 3)), np.zeros((16, 2), np.float32), np.zeros(3)
    first_gray = cv.cvtColor(first.transpose(1, 2, 0), cv.COLOR_RGB2GRAY)
    second_gray = cv.cvtColor(second.transpose(1, 2, 0), cv.COLOR_RGB2GRAY)
    params = (.5, 3, 45, 3, 7, 1.5, 0)
    forward = cv.calcOpticalFlowFarneback(first_gray, second_gray, None, *params)
    backward = cv.calcOpticalFlowFarneback(second_gray, first_gray, None, *params)
    yy, xx = np.mgrid[:96, :96].astype(np.float32)
    mx, my = xx+forward[..., 0], yy+forward[..., 1]
    reverse = cv.remap(backward, mx, my, cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    next_cov = cv.remap(coverage_second.astype(np.float32), mx, my, cv.INTER_LINEAR,
                        borderMode=cv.BORDER_CONSTANT)
    source = np.stack((xx, yy), -1) + np.rint(centers[0]) - 48
    destination = source + forward + np.rint(centers[1])-np.rint(centers[0])
    old, v1 = image_to_native(source, matrix)
    new, v2 = image_to_native(destination, matrix)
    valid = ((coverage_first == 1) & (next_cov >= .999) & (mx >= 3) & (mx <= 92)
             & (my >= 3) & (my <= 92) & (np.linalg.norm(forward+reverse, axis=-1) <= 4.5) & v1 & v2)
    return grid_pool(new-old, valid, coverage_first)


def normalized_grid(grid, rotation, scale):
    if not np.isfinite(scale) or scale <= 0 or grid.shape != (16, 3):
        raise ValueError('Fixed grid and past-only positive scale required')
    result = grid.copy()
    result[:, :2] = result[:, :2] @ rotation
    return (result/scale).astype(np.float32)


def feature_variant(geometry, motion, quality, variant):
    if (variant not in VARIANTS or motion.shape[1:] != (3, 7, 16, 3)
            or quality.shape[1:] != (2, 7, 16, 2)):
        raise ValueError('Registered resolution/grid schema required')
    index = dict(lowpass_pool=2, lowpass_grid=0, native_grid=1).get(variant)
    chosen = np.zeros_like(motion[:, 0]) if index is None else motion[:, index]
    return np.concatenate((geometry, quality.reshape(len(geometry), -1), chosen.reshape(len(geometry), -1)), 1).astype(np.float32)
