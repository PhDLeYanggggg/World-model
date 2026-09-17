"""SDD geometry bridge to the frozen 8-to-12 schema; no training admission."""
from __future__ import annotations

import numpy as np

from src.data_unification.m3w_causal_recordings import (
    INDEX_DTYPE, RecordingWindows, causal_coordinate_transform, track_boundaries,
)
from src.world_model.m3w_offline_visual_forecast import geometry_features


class SDDStepAdapter(RecordingWindows):
    """One recording in memory, index plus lazy samples, separate label access.

    The original annotation source remains offline/silver. A future training
    experiment must separately register roles and sampling; this bridge cannot
    turn an old test recording into untouched confirmation data.
    """

    def __init__(self, rows, labels, recording_id, stride, *, data_role='diagnostic_only'):
        if data_role != 'diagnostic_only':
            raise ValueError('Sampling/source-role decision has not admitted training')
        if (not isinstance(stride, (int, np.integer)) or isinstance(stride, (bool, np.bool_))
                or stride <= 0):
            raise ValueError('Positive integer raw-frame stride required')
        a, labels = np.asarray(rows, float), np.asarray(labels)
        if (a.ndim != 2 or a.shape[1] != 9 or not len(a) or labels.shape != (len(a),)
                or not np.isfinite(a).all() or np.any(a[:, 3:5] < a[:, 1:3])
                or np.any(a[:, [0, 5]] < 0) or np.any(a[:, [0, 5]] != np.rint(a[:, [0, 5]]))
                or not np.isin(a[:, 6:9], [0, 1]).all()):
            raise ValueError('Finite annotation rows and source labels required')
        order = np.lexsort((a[:, 5], a[:, 0]))
        self.source, self.labels = a[order], labels[order]
        if np.any((np.diff(self.source[:, 0]) == 0) & (np.diff(self.source[:, 5]) == 0)):
            raise ValueError('Conflicting or duplicate source frame/agent key')
        self.stride = int(stride)
        self.metadata = dict(id=recording_id, physical_scene=recording_id.split('/')[0],
            data_role=data_role, coordinate_unit='annotation_pixel', metric_status='pixel_space',
            observed_steps=8, predicted_steps=12, raw_frame_stride=self.stride,
            observation_mode='offline_annotated', effective_seconds=None, training_admitted=False)
        selected = (self.source[:, 6] == 0) & (self.source[:, 5] % stride == 0)
        self.source_ids = np.flatnonzero(selected)
        s = self.source[selected]
        self.points = np.column_stack((s[:, 5], s[:, 0], (s[:, 1:3]+s[:, 3:5])/2))
        if not len(self.points):
            raise ValueError('No observed source states on requested grid')
        self.starts, self.ends = track_boundaries(self.points)
        self.frame_order = np.argsort(self.points[:, 0], kind='stable')
        self.frame_values = self.points[self.frame_order, 0]
        self.max_neighbors = 8
        indices = []
        for start, end in zip(self.starts, self.ends):
            if end-start < 8:
                continue
            times = self.points[start:end, 0]
            windows = np.lib.stride_tricks.sliding_window_view(times, 8)
            current = start+7+np.flatnonzero(np.all(np.diff(windows, axis=1) == stride, axis=1))
            current = current[self.labels[self.source_ids[current]] == 'Pedestrian']
            piece = np.zeros(len(current), dtype=INDEX_DTYPE)
            piece['history_start'], piece['current_row'] = current-7, current
            piece['future_end'] = -1  # Future support cannot determine index membership.
            piece['future_steps'], piece['horizon_raw'] = 12, 12*stride
            indices.append(piece)
        self.index = np.concatenate(indices) if indices else np.empty(0, dtype=INDEX_DTYPE)
        source_starts = np.r_[0, np.flatnonzero(np.diff(self.source[:, 0]))+1]
        source_ends = np.r_[source_starts[1:], len(self.source)]
        self._source_tracks = {int(self.source[a, 0]): (int(a), int(b)) for a, b in zip(source_starts, source_ends)}

    def identity(self, item):
        row = self.index[item]
        current = self.points[int(row['current_row'])]
        return dict(recording_id=self.metadata['id'], agent_id=int(current[1]),
                    frame_id=int(current[0]), raw_frame_stride=self.stride,
                    data_role='diagnostic_only')

    def get_geometry(self, item):
        return geometry_features(self.get_inputs(item))

    def get_past_provenance(self, item):
        row = self.index[item]
        ids = self.source_ids[int(row['history_start']):int(row['current_row'])+1]
        return dict(annotation_boxes=self.source[ids, 1:5].copy(),
                    lost=self.source[ids, 6].astype(bool), occluded=self.source[ids, 7].astype(bool),
                    generated=self.source[ids, 8].astype(bool), agent_type=self.labels[ids[-1]])

    def get_labels(self, item):
        row = self.index[item]
        history = self.points[int(row['history_start']):int(row['current_row'])+1]
        frame, agent = map(int, history[-1, :2])
        start, end = self._source_tracks[agent]
        track = self.source[start:end]
        frames = frame+np.arange(1, 13)*self.stride
        loc = np.searchsorted(track[:, 5], frames)
        safe = np.minimum(loc, len(track)-1)
        present = (loc < len(track)) & (track[safe, 5] == frames)
        flags = np.zeros((12, 3), bool)
        flags[present] = track[safe[present], 6:9].astype(bool)
        mask = present & ~flags[:, 0]
        xy = np.zeros((12, 2), float)
        xy[mask] = (track[safe[mask], 1:3]+track[safe[mask], 3:5])/2
        transform = causal_coordinate_transform(history[:, 2:], history[:, 0], 12*self.stride)
        normalized = np.zeros((12, 2), np.float32)
        normalized[mask] = ((xy[mask]-transform['origin_xy']) @ transform['rotation']/transform['scale']).astype(np.float32)
        return dict(future_frame_ids=frames, future_xy_dataset_local=xy,
                    future_xy_normalized=normalized, future_label_mask=mask,
                    future_annotation_present=present, future_lost=flags[:, 0],
                    future_occluded=flags[:, 1], future_generated=flags[:, 2])

    def get_scene_inputs(self, *args, **kwargs):
        raise ValueError('Use indexed 8-to-12 diagnostic requests; arbitrary scene requests are not registered')

    def get_scene_labels(self, *args, **kwargs):
        raise ValueError('Use separate per-query masked labels')


def masked_future_ade(prediction, target, valid):
    """Loss adapter: unsupported labels have zero gradient, not zero-error evidence."""
    import torch
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[1:] != (12, 2):
        raise ValueError('Matching twelve-step 2D trajectories required')
    if valid.shape != prediction.shape[:2] or valid.dtype != torch.bool:
        raise ValueError('Explicit boolean future-label mask required')
    if not torch.isfinite(prediction).all() or not torch.isfinite(target[valid]).all():
        raise ValueError('Finite predictions and supported labels required')
    target = torch.where(valid[..., None], target, prediction.detach())
    distance = torch.linalg.vector_norm(prediction-target, dim=-1)
    count = valid.sum(1)
    ade = (distance*valid).sum(1)/count.clamp_min(1)
    return ade, count > 0
