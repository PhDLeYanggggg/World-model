"""Observed SDD box/background flow proxies, not segmentation or intent labels."""
import numpy as np

from src.evaluation.m3w_source_temporal_information import box_masks

MOTION_NAMES = ('box_dx', 'box_dy', 'surround_dx', 'surround_dy', 'contrast_dx',
                'contrast_dy', 'box_q50', 'box_q90', 'surround_q50', 'surround_q90')
QUALITY_NAMES = ('box_coverage', 'surround_coverage', 'box_consistency',
                 'surround_consistency', 'box_supported', 'surround_supported',
                 'lost', 'occluded', 'generated')


def annotation_displacement(flow, boxes, scale_xy):
    flow, boxes, scale_xy = map(lambda v: np.asarray(v, float), (flow, boxes, scale_xy))
    if (flow.shape != (32, 32, 2) or boxes.shape != (2, 4) or scale_xy.shape != (2,)
            or not all(np.isfinite(a).all() for a in (flow, boxes, scale_xy))
            or np.any(scale_xy <= 0) or np.any(boxes[:, 2:] < boxes[:, :2])):
        raise ValueError('Finite flow, ordered boxes and positive video/annotation scale required')
    centers = np.rint((boxes[:, :2]+boxes[:, 2:])/2)
    return (flow*3 + centers[1]-centers[0])/scale_xy


def regions(boxes):
    boxes = np.asarray(boxes, float)
    inside = box_masks(boxes)
    # Exclude a one-low-resolution-pixel rim around the annotated box.
    expanded = boxes + np.array([-3, -3, 3, 3])
    outside = ~box_masks(expanded)
    outside[:, :2] = False; outside[:, -2:] = False
    outside[:, :, :2] = False; outside[:, :, -2:] = False
    return inside, outside


def pair_features(rgb, coverage, boxes, flags, scale_xy, cv):
    rgb, coverage, boxes, flags = map(np.asarray, (rgb, coverage, boxes, flags))
    if (rgb.shape != (2, 3, 32, 32) or rgb.dtype != np.uint8
            or coverage.shape != (2, 32, 32) or np.any(coverage > 9)
            or flags.shape != (2, 3) or not np.isin(flags, [0, 1]).all()):
        raise ValueError('Two bound past crops, coverage and source flags required')
    inside, outside = regions(boxes)
    feature = np.zeros(19, np.float32)
    feature[16:] = flags.max(0)
    for j, region in enumerate((inside[0], outside[0])):
        feature[10+j] = coverage[0][region].mean()/9 if region.any() else 0
    if not coverage[0].any() or not coverage[1].any():
        return feature
    gray = [cv.cvtColor(a.transpose(1, 2, 0), cv.COLOR_RGB2GRAY) for a in rgb]
    params = (.5, 3, 15, 3, 5, 1.2, 0)
    forward = cv.calcOpticalFlowFarneback(gray[0], gray[1], None, *params)
    reverse = cv.calcOpticalFlowFarneback(gray[1], gray[0], None, *params)
    yy, xx = np.mgrid[:32, :32].astype(np.float32)
    mx, my = xx+forward[..., 0], yy+forward[..., 1]
    def remap(array):
        return cv.remap(array.astype(np.float32), mx, my, cv.INTER_LINEAR,
                        borderMode=cv.BORDER_CONSTANT)
    common = ((coverage[0] == 9) & (remap(coverage[1]) >= 8.999)
              & (mx >= 1) & (mx <= 30) & (my >= 1) & (my <= 30)
              & (np.linalg.norm(forward+remap(reverse), axis=-1) <= 1.5))
    displacement = annotation_displacement(forward, boxes, scale_xy)
    for j, region in enumerate((inside, outside)):
        valid = common & region[0] & (remap(region[1]) >= .999)
        feature[12+j] = valid.sum()/max(1, region[0].sum())
        if valid.sum() >= 16:
            feature[14+j] = 1
            feature[2*j:2*j+2] = np.median(displacement[valid], axis=0)
            feature[6+2*j:8+2*j] = np.quantile(np.linalg.norm(displacement[valid], axis=-1), [.5, .9])
    if feature[14] and feature[15]:
        feature[4:6] = feature[:2]-feature[2:4]
    return feature


def motion_tokens(raw, rotation, native_scale, radius):
    raw = np.asarray(raw, np.float32).copy()
    n = len(raw)
    if (raw.shape != (n, 7, 19) or rotation.shape != (n, 2, 2)
            or native_scale.shape != (n,) or radius.shape != (n,)
            or not np.isfinite(raw).all() or np.any(native_scale <= 0) or np.any(radius <= 0)):
        raise ValueError('Aligned past-only motion and restoration frame required')
    denominator = native_scale*radius
    raw[..., :6] = (np.einsum('ntvd,ndk->ntvk', raw[..., :6].reshape(n, 7, 3, 2), rotation)
                       / denominator[:, None, None, None]).reshape(n, 7, 6)
    raw[..., 6:10] /= denominator[:, None, None]
    return raw


def fit_motion_normalizer(training_tokens):
    values = np.asarray(training_tokens, np.float64)
    if values.ndim != 3 or values.shape[1:] != (7, 19) or not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite training-only motion tokens required')
    mean = values.mean((0, 1)); std = values.std((0, 1))
    constant = std < 1e-8
    std[constant] = 1
    return mean, std, constant


def token_payload(tokens, normalizer, arm):
    if arm not in ('quality', 'motion'):
        raise ValueError('Fixed paired representation arm required')
    mean, std, constant = normalizer
    values = np.clip((tokens-mean)/std, -8, 8).astype(np.float32)
    values[..., constant] = 0
    if arm == 'quality':
        values[..., :10] = 0
    result = np.zeros((len(tokens), 8, 512), np.float32)
    result[:, 1:, :19] = values
    return result
