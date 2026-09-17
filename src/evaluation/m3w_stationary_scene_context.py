"""Unverified static-obstacle context for a fit-only stationary-start probe."""
from __future__ import annotations

import xml.etree.ElementTree as ET
import numpy as np

from src.evaluation.m3w_stationary_start_probe import past_features
from src.evaluation.m3w_stationary_pooled_context import pool_context, NAMES as POOLED_NAMES


SCENE_NAMES = ['nearest_surface_distance', 'second_surface_distance', 'distance_gap',
               'nearest_is_circle', 'nearest_radius', 'frame_defined', 'nearest_tie']
DIRECTION_NAMES = ['mean_neighbor_normal', 'mean_neighbor_tangent',
                   'nearest_neighbor_normal', 'nearest_neighbor_tangent',
                   'mean_velocity_horizon_normal', 'mean_velocity_horizon_tangent',
                   'velocity_coherence', 'directional_neighbor_support']
NAMES = POOLED_NAMES + SCENE_NAMES + DIRECTION_NAMES
FEATURE_SETS = {'pooled': list(range(13)), 'scene': list(range(20)),
                'scene_neighbor': list(range(28))}


def read_obstacles(path):
    root = ET.parse(path).getroot()
    lines, circles = [], []
    for node in root.iter():
        tag = node.tag.rsplit('}', 1)[-1]
        if tag == 'Line':
            lines.append([float(node.attrib[k]) for k in ('x1', 'y1', 'x2', 'y2')])
        elif tag == 'Circle':
            circles.append([float(node.attrib[k]) for k in ('x', 'y', 'radius')])
    return validate_obstacles({'lines': np.asarray(lines).reshape(-1, 2, 2),
                               'circles': np.asarray(circles).reshape(-1, 3)})


def validate_obstacles(obstacles):
    lines, circles = np.asarray(obstacles['lines']), np.asarray(obstacles['circles'])
    if (lines.ndim != 3 or lines.shape[1:] != (2, 2) or circles.ndim != 2
            or circles.shape[1] != 3 or not len(lines)+len(circles)
            or not np.isfinite(lines).all() or not np.isfinite(circles).all()
            or np.any(np.linalg.norm(np.diff(lines, axis=1), axis=2) <= 0)
            or np.any(circles[:, 2] <= 0)):
        raise ValueError('Finite nondegenerate static geometry required')
    return {'lines': lines, 'circles': circles}


def obstacle_frame(center, obstacles):
    """Deterministic closest-surface frame; no learned or trajectory statistics."""
    center = np.asarray(center, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError('Finite current position required')
    obstacles = validate_obstacles(obstacles)
    lines, circles = obstacles['lines'], obstacles['circles']
    extent = list(lines.reshape(-1, 2))
    # Pairwise Euclidean diameter is rotation invariant, unlike an axis-aligned box.
    extent += list(circles[:, :2])
    extent = np.asarray(extent)
    scale = float(np.linalg.norm(extent[:, None]-extent[None], axis=-1).max())
    if len(circles):
        scale = max(scale, float(2*circles[:, 2].max()))
    if not scale > 0:
        raise ValueError('Static geometry has no coordinate scale')
    surfaces, kinds, radii = [], [], []
    for a, b in lines:
        segment = b-a
        u = float(np.clip((center-a) @ segment/(segment @ segment), 0, 1))
        surfaces.append(a+u*segment); kinds.append(0); radii.append(0.)
    for x, y, radius in circles:
        offset = center-np.array([x, y])
        distance = np.linalg.norm(offset)
        surfaces.append(np.array([x, y])+radius*offset/distance if distance else center.copy())
        kinds.append(1); radii.append(radius)
    relative = center-np.asarray(surfaces)
    distances = np.linalg.norm(relative, axis=1)
    order = np.argsort(distances, kind='stable')
    closest = int(order[0])
    d1, d2 = float(distances[closest]), float(distances[order[1]]) if len(order)>1 else float(distances[closest])
    nearest_mask = np.abs(distances-d1) <= 1e-10*scale
    nearest_points = np.asarray(surfaces)[nearest_mask]
    # Adjacent segments can share one closest corner without ambiguous normals.
    tie = bool(np.any(np.linalg.norm(nearest_points-np.asarray(surfaces)[closest], axis=1) > 1e-10*scale))
    defined = bool(d1 > 1e-10*scale and not tie)
    normal = relative[closest]/d1 if defined else np.zeros(2)
    basis = np.column_stack([normal, [-normal[1], normal[0]]])
    features = np.array([d1/scale, d2/scale, (d2-d1)/scale, kinds[closest],
                         radii[closest]/scale, float(defined), float(tie)])
    return {'basis': basis, 'scale': scale, 'defined': defined, 'features': features}


def scene_context(inputs, center, obstacles):
    """Accept only the frozen past schema. Neither labels nor row metadata enters."""
    control = pool_context(past_features(inputs)[None])[0]
    frame = obstacle_frame(center, obstacles)
    direction = np.zeros(8)
    t = np.asarray(inputs['history_frame_offsets'])
    eligible = np.asarray(inputs['neighbor_mask']).all(1) & np.isclose(
        inputs['neighbor_frame_offsets'], t[None], atol=1e-6, rtol=1e-6).all(1)
    if eligible.any() and frame['defined']:
        # The exactly stationary ego frame has zero heading, hence identity rotation.
        xy = np.asarray(inputs['neighbor_xy'][eligible], dtype=float)
        xy = xy*float(inputs['causal_features'][2]) @ frame['basis']/frame['scale']
        positions = xy[:, -1]
        velocity = (xy[:, -1]-xy[:, -2])/(t[-1]-t[-2])*inputs['prediction_frame_offsets'][-1]
        nearest = int(np.argmin(np.linalg.norm(positions, axis=1)))
        speed = np.linalg.norm(velocity, axis=1)
        coherence = float(np.linalg.norm(velocity.sum(0))/speed.sum()) if speed.sum() else 0.
        direction = np.r_[positions.mean(0), positions[nearest], velocity.mean(0), coherence, len(xy)]
    out = np.r_[control, frame['features'], direction]
    if out.shape != (len(NAMES),) or not np.isfinite(out).all():
        raise ValueError('Invalid static-scene features')
    return out, frame


def encode_future_label(future, center, frame):
    """Label-side operation only. Unsupported orientation remains loss-masked."""
    future, center = np.asarray(future), np.asarray(center)
    if future.shape != (12, 2) or center.shape != (2,) or not np.isfinite(future).all():
        raise ValueError('Complete twelve-step supervision required')
    return (future-center) @ frame['basis']/frame['scale']


def forecast_metrics(prediction, target, parent_scale, easy_threshold, groups=None):
    prediction, target = np.asarray(prediction), np.asarray(target)
    scale = np.asarray(parent_scale)
    if prediction.shape != target.shape or prediction.shape[1:] != (12, 2) or scale.shape != (len(target),) or not np.isfinite(prediction).all() or np.any(scale<=0):
        raise ValueError('Aligned finite native displacement forecasts required')
    norm = np.linalg.norm(target, axis=-1)
    err = np.linalg.norm(prediction-target, axis=-1)
    ade, floor = err.mean(1)/scale, norm.mean(1)/scale
    easy = floor <= easy_threshold
    def gain(a, b):
        return float(100*(1-a/b)) if b > 0 else None
    easy_floor = float(floor[easy].mean()) if easy.any() else None
    easy_model = float(ade[easy].mean()) if easy.any() else None
    true_endpoint, pred_endpoint = target[:, -1], prediction[:, -1]
    tn, pn = np.linalg.norm(true_endpoint, axis=1), np.linalg.norm(pred_endpoint, axis=1)
    direction = (tn>0) & (pn>0)
    angle = np.degrees(np.arccos(np.clip(np.sum(true_endpoint[direction]*pred_endpoint[direction], axis=1)/(tn[direction]*pn[direction]), -1, 1)))
    result = {'rows': len(target), 'native_ade': float(err.mean()), 'native_fde': float(err[:, -1].mean()),
              'parent_normalized_ade': float(ade.mean()), 'cv_parent_normalized_ade': float(floor.mean()),
              'gain_vs_cv_pct': gain(ade.mean(), floor.mean()), 'harm_over_cv': float((ade-floor).mean()),
              'positive_harm_mean': float(np.maximum(ade-floor, 0).mean()),
              'easy_rows': int(easy.sum()), 'easy_floor_ade': easy_floor, 'easy_model_ade': easy_model,
              'easy_absolute_harm': easy_model-easy_floor if easy.any() else None,
              'easy_degradation_pct': -gain(easy_model, easy_floor) if easy.any() and easy_floor>0 else None,
              'easy_ratio_status': 'defined' if easy.any() and easy_floor>0 else 'undefined_zero_floor_or_empty',
              'direction_scored_rows': int(direction.sum()), 'true_nonzero_endpoint_rows': int((tn>0).sum()),
              'angular_error_degrees': float(angle.mean()) if len(angle) else None,
              'new_deployment': False}
    if groups is not None:
        indices = {}
        for i, key in enumerate(groups):
            indices.setdefault(tuple(key), []).append(i)
        a = float(np.mean([ade[v].mean() for v in indices.values()]))
        b = float(np.mean([floor[v].mean() for v in indices.values()]))
        result['group_balanced'] = {'groups': len(indices), 'gain_vs_cv_pct': gain(a, b), 'harm_over_cv': a-b,
                                    'independence_established': False}
    return result
