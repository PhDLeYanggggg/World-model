"""Fixed observed-image resolution/window controls; no future-label interface."""
import json
from pathlib import Path

import numpy as np

from src.evaluation.m3w_experiment_contract import file_digest
from src.evaluation.m3w_source_temporal_information import box_masks

VARIANTS = {'lowpass_w45': (32, 15), 'lowpass_w15': (32, 5),
            'native_w45': (96, 45), 'native_w15': (96, 15)}
REGISTRATION = 'configs/m3w_source_motion_resolution_v1.json'


def registration(root):
    plan = json.loads((root / REGISTRATION).read_text())
    if (plan['variants'] != list(VARIANTS) or plan['selection'] or plan['new_deployment']
            or plan['stage5c_executed'] or plan['smc_enabled'] or not plan['bindings']):
        raise ValueError('Fixed source-only resolution comparison required')
    for name, digest in plan['bindings'].items():
        if file_digest(root / name) != digest:
            raise ValueError('Changed registered dependency: ' + name)
    return plan


def regions(boxes, size):
    inside = box_masks(boxes, size=size)
    outside = ~box_masks(np.asarray(boxes) + [-3, -3, 3, 3], size=size)
    rim = size // 16
    outside[:, :rim] = False; outside[:, -rim:] = False
    outside[:, :, :rim] = False; outside[:, :, -rim:] = False
    return inside, outside


def displacement(flow, boxes, scale_xy):
    size = flow.shape[0]
    boxes, scale_xy = np.asarray(boxes, float), np.asarray(scale_xy, float)
    if (size not in (32, 96) or flow.shape != (size, size, 2)
            or boxes.shape != (2, 4) or scale_xy.shape != (2,)
            or not all(np.isfinite(a).all() for a in (flow, boxes, scale_xy))
            or np.any(scale_xy <= 0) or np.any(boxes[:, 2:] < boxes[:, :2])):
        raise ValueError('Finite ordered boxes, flow and positive axis scale required')
    centers = np.rint((boxes[:, :2] + boxes[:, 2:]) / 2)
    return (flow.astype(float) * (96 / size) + centers[1] - centers[0]) / scale_xy


def pair_features(rgb, coverage, boxes, flags, scale_xy, cv, variant):
    size, window = VARIANTS[variant]
    rgb, coverage, boxes, flags = map(np.asarray, (rgb, coverage, boxes, flags))
    area = (96 // size) ** 2
    if (rgb.shape != (2, 3, size, size) or rgb.dtype != np.uint8
            or coverage.shape != (2, size, size) or coverage.dtype != np.uint8
            or np.any(coverage > area) or flags.shape != (2, 3)
            or not np.isin(flags, [0, 1]).all()):
        raise ValueError('Registered observed RGB, coverage and flags required')
    inside, outside = regions(boxes, size)
    feature = np.zeros(19, np.float32); feature[16:] = flags.max(0)
    for j, region in enumerate((inside[0], outside[0])):
        feature[10+j] = coverage[0][region].mean()/area if region.any() else 0
    if not coverage[0].any() or not coverage[1].any():
        return feature
    gray = [cv.cvtColor(a.transpose(1, 2, 0), cv.COLOR_RGB2GRAY) for a in rgb]
    params = (.5, 3, window, 3, 5, 1.2, 0)
    forward = cv.calcOpticalFlowFarneback(gray[0], gray[1], None, *params)
    reverse = cv.calcOpticalFlowFarneback(gray[1], gray[0], None, *params)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    mx, my = xx + forward[..., 0], yy + forward[..., 1]
    def remap(array):
        return cv.remap(array.astype(np.float32), mx, my, cv.INTER_LINEAR,
                        borderMode=cv.BORDER_CONSTANT)
    border = size // 32
    common = ((coverage[0] == area) & (remap(coverage[1]) >= area-.001)
              & (mx >= border) & (mx <= size-1-border)
              & (my >= border) & (my <= size-1-border)
              & (np.linalg.norm(forward+remap(reverse), axis=-1) <= 4.5/(96/size)))
    delta = displacement(forward, boxes, scale_xy)
    for j, region in enumerate((inside, outside)):
        valid = common & region[0] & (remap(region[1]) >= .999)
        feature[12+j] = valid.sum()/max(1, region[0].sum())
        if valid.sum() >= 144/area:
            feature[14+j] = 1
            feature[2*j:2*j+2] = np.median(delta[valid], axis=0)
            feature[6+2*j:8+2*j] = np.quantile(np.linalg.norm(delta[valid], axis=-1), [.5, .9])
    if feature[14] and feature[15]:
        feature[4:6] = feature[:2] - feature[2:4]
    return feature


def raw_labels(future_native, name):
    """Loss/evaluation only; exact raw coordinates avoid normalized float32 ties."""
    future = np.asarray(future_native, float)
    if future.ndim != 3 or future.shape[1:] != (12, 2) or not np.isfinite(future).all():
        raise ValueError('Aligned raw future labels required for supervision only')
    if name == 'any_nonzero':
        return np.any(future != 0, axis=(1, 2)).astype(int)
    if name == 'over10_annotation_pixels':
        return (np.max(np.sum(future**2, axis=-1), axis=1) > 100).astype(int)
    raise ValueError('Unknown registered supervision target')
