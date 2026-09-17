"""Source-anchored UCY media inputs with explicit annotation provenance."""
from __future__ import annotations

import numpy as np

from src.evaluation.m3w_zara_media_lineage import history_support, image_coordinates


def project_image_xy(image_xy, matrix):
    xy, matrix = np.asarray(image_xy, float), np.asarray(matrix, float)
    if (xy.ndim != 2 or xy.shape[1] != 2 or matrix.shape != (3, 3)
            or not np.isfinite(xy).all() or not np.isfinite(matrix).all()):
        raise ValueError('Finite image xy and 3x3 supplied matrix required')
    homogeneous = np.column_stack([xy[:, ::-1], np.ones(len(xy))]) @ matrix.T
    if np.any(np.abs(homogeneous[:, 2]) < 1e-12):
        raise ValueError('Projection at infinity')
    return homogeneous[:, :2] / homogeneous[:, 2:]


def source_origin_anchor(native_xy, image_xy, matrix):
    """One predesignated source anchor; no least-squares fit or future targets."""
    native, image = np.asarray(native_xy, float), np.asarray(image_xy, float)
    if native.shape != (2,) or image.shape != (2,) or not np.isfinite(native).all():
        raise ValueError('Exactly one finite source coordinate pair required')
    return native - project_image_xy(image[None], matrix)[0]


def native_to_image_xy(native_xy, matrix, origin):
    native, origin = np.asarray(native_xy, float), np.asarray(origin, float)
    if native.ndim != 2 or native.shape[1] != 2 or origin.shape != (2,):
        raise ValueError('Native Nx2 and origin 2-vector required')
    # project_image_xy consumes xy and swaps to row/column before H.
    inverse_xy = project_image_xy((native - origin)[:, ::-1], np.linalg.inv(matrix))
    return inverse_xy[:, ::-1]


def select_past_media_controls(points, trace, width, height, *, max_agents=12):
    """First complete past for each ID; no future-survival/motion filtering."""
    points = np.asarray(points)
    if max_agents < 1:
        raise ValueError('Positive agent budget required')
    windows, strict = history_support(points, trace['latest_control_frame'], trace['available'], 1)
    all_xy = image_coordinates(trace['centered_xy'], width, height)
    chosen, seen = [], set()
    for history, is_strict in zip(windows, strict):
        agent = int(points[history[-1], 1])
        if agent in seen:
            continue
        seen.add(agent)
        chosen.append({'agent_id': agent, 'query_stored_frame': int(points[history[-1], 0]),
            'query_source_frame': int(trace['source_frame'][history[-1]]),
            'history_source_frames': trace['source_frame'][history].tolist(),
            'history_image_xy': all_xy[history].tolist(),
            'latest_contributing_control_frames': trace['latest_control_frame'][history].tolist(),
            'source_controls_as_of_query': bool(is_strict),
            'past_net_displacement_px': float(np.linalg.norm(all_xy[history[-1]] - all_xy[history[0]]))})
        if len(chosen) == max_agents:
            break
    return chosen


def validate_past_request(control, *, require_control_as_of_query=False):
    frames = np.asarray(control['history_source_frames'])
    xy = np.asarray(control['history_image_xy'])
    latest = np.asarray(control['latest_contributing_control_frames'])
    query = control['query_source_frame']
    if (frames.shape != (8,) or xy.shape != (8, 2) or latest.shape != (8,)
            or not np.isfinite(xy).all() or not np.array_equal(frames, np.round(frames))
            or np.any(np.diff(frames) != 10) or frames[-1] != query or np.any(frames < 0)):
        raise ValueError('Only eight ordered native past frames ending at query permitted')
    if np.any(latest < frames):
        raise ValueError('Invalid interpolation provenance')
    if require_control_as_of_query and np.any(latest > query):
        raise ValueError('Offline annotations require a control after the query')
    return True
