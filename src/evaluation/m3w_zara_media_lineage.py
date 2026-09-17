"""Trace supplied UCY spline coordinates without claiming sensor-time causality."""
from __future__ import annotations

from pathlib import Path
import numpy as np


def read_vsp(path):
    lines = []
    for line in Path(path).read_text().splitlines():
        text = line.partition(' - ')[0].strip()
        if text:
            lines.append(text.split())
    if not lines or len(lines[0]) != 1:
        raise ValueError('Missing spline count')
    count, cursor, tracks = int(lines[0][0]), 1, []
    for _ in range(count):
        if cursor >= len(lines) or len(lines[cursor]) != 1:
            raise ValueError('Missing control-point count')
        n = int(lines[cursor][0])
        cursor += 1
        block = lines[cursor:cursor + n]
        if n < 2 or len(block) != n or any(len(row) != 4 for row in block):
            raise ValueError('Malformed spline control points')
        a = np.asarray(block, dtype=float)
        if (not np.isfinite(a).all() or not np.array_equal(a[:, 2], np.round(a[:, 2]))
                or np.any(np.diff(a[:, 2]) <= 0)):
            raise ValueError('Finite strictly increasing integer control frames required')
        tracks.append(a)
        cursor += n
    if cursor != len(lines):
        raise ValueError('Unexpected trailing spline rows')
    return tracks


def interpolate_controls(control, frames):
    """Return linear coordinates and latest source-control frame per query."""
    control, frames = np.asarray(control), np.asarray(frames)
    if (control.ndim != 2 or control.shape[1] != 4 or len(control) < 2
            or frames.ndim != 1 or not np.isfinite(frames).all()
            or not np.isfinite(control).all() or np.any(np.diff(control[:, 2]) <= 0)):
        raise ValueError('Finite controls and query frames required')
    if np.any(frames < control[0, 2]) or np.any(frames > control[-1, 2]):
        raise ValueError('No extrapolation beyond source controls')
    right = np.searchsorted(control[:, 2], frames, side='left')
    xy = np.column_stack([np.interp(frames, control[:, 2], control[:, j]) for j in (0, 1)])
    return xy, control[right, 2].astype(np.int64), control[right, 2] == frames


def image_coordinates(centered_xy, width, height):
    p = np.asarray(centered_xy, float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all() or width <= 0 or height <= 0:
        raise ValueError('Finite centered 2D coordinates and positive image size required')
    return np.column_stack([p[:, 0] + width / 2, height / 2 - p[:, 1]])


def trace_rows(points, tracks, frame_offset):
    """Stored IDs are checked against one-based spline order, never inferred."""
    points = np.asarray(points)
    if (points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all()
            or not np.array_equal(points[:, :2], np.round(points[:, :2]))):
        raise ValueError('Expected frame, agent, x, y source rows')
    n = len(points)
    xy, right, exact, available = np.zeros((n, 2)), np.zeros(n, np.int64), np.zeros(n, bool), np.zeros(n, bool)
    source_frames = points[:, 0].astype(np.int64) - frame_offset
    for agent in np.unique(points[:, 1].astype(int)):
        if agent < 1 or agent > len(tracks):
            raise ValueError('Stored identity has no one-based spline')
        ids = np.flatnonzero(points[:, 1] == agent)
        control = tracks[agent - 1]
        ids = ids[(source_frames[ids] >= control[0, 2]) & (source_frames[ids] <= control[-1, 2])]
        if len(ids):
            xy[ids], right[ids], exact[ids] = interpolate_controls(control, source_frames[ids])
            available[ids] = True
    return {'centered_xy': xy, 'latest_control_frame': right, 'exact_control': exact,
            'available': available, 'source_frame': source_frames}


def history_support(points, latest_control_frame, available, frame_offset, step=10, length=8):
    """Past-only queries; future track length is never used to admit a history."""
    points = np.asarray(points)
    supported, strict = [], []
    for agent in np.unique(points[:, 1]):
        ids = np.flatnonzero(points[:, 1] == agent)
        ids = ids[np.argsort(points[ids, 0])]
        for j in range(length - 1, len(ids)):
            history = ids[j - length + 1:j + 1]
            if not np.all(np.diff(points[history, 0]) == step) or not np.all(available[history]):
                continue
            supported.append(history)
            strict.append(np.max(latest_control_frame[history]) <= points[history[-1], 0] - frame_offset)
    return np.asarray(supported, dtype=np.int64).reshape(-1, length), np.asarray(strict, dtype=bool)
