"""Input-only bridge from a raw annotation prefix to fixed 8/12 predictors.

No dataset files, future labels, learned statistics or predictive role are opened
here. Calling code must pass source admission before loading a reserved prefix.
"""
from __future__ import annotations

import numpy as np

from src.data_unification.m3w_causal_recordings import RecordingWindows, track_boundaries
from src.world_model.m3w_offline_visual_forecast import geometry_features


def dense_prefix(positions, valid, agent_ids, *, query_frame):
    """Slice admitted dense arrays without materializing later coordinates/masks.

    This numerical helper does not admit or open a source. The caller chooses an
    observable query independently of whether any future labels will be present.
    """
    shape = positions.shape
    ids = np.asarray(agent_ids)
    if (len(shape) != 3 or shape[-1] != 2 or valid.shape != shape[:2]
            or valid.dtype != np.dtype(bool) or ids.shape != (shape[0],)
            or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or np.any(ids >= 2**53)
            or len(np.unique(ids)) != len(ids)
            or type(query_frame) is not int or not 7 <= query_frame < shape[1]):
        raise ValueError("Aligned dense [agent, raw frame, xy] arrays and valid query required")
    start = query_frame - 7
    observed = np.asarray(valid[:, start:query_frame + 1], dtype=bool)
    coordinates = np.asarray(positions[:, start:query_frame + 1, :], dtype=np.float64)
    agent, step = np.nonzero(observed)
    xy = coordinates[agent, step]
    if not np.isfinite(xy).all():
        raise ValueError("Nonfinite coordinate marked observed in the requested prefix")
    return np.column_stack((step + start, ids[agent], xy))


class ExternalPrefixAdapter(RecordingWindows):
    """Explicitly native stride1; do not silently resample to an SDD time scale."""

    def __init__(self, prefix_points, *, query_frame, recording_id):
        if type(query_frame) is not int or query_frame < 7 or not isinstance(recording_id, str) or not recording_id:
            raise ValueError("Named recording and integer query supporting eight raw steps required")
        points = np.asarray(prefix_points, dtype=np.float64)
        if (points.ndim != 2 or points.shape[1] != 4 or not len(points)
                or not np.isfinite(points).all() or np.any(points[:, :2] != np.rint(points[:, :2]))
                or np.any(points[:, :2] < 0) or np.any(points[:, :2] >= 2**53)):
            raise ValueError("Finite raw [frame, agent, x, y] prefix required")
        if np.any(points[:, 0] > query_frame):
            raise ValueError("Future rows are forbidden in the input-only adapter")
        points = points[(points[:, 0] >= query_frame - 7)]
        if not len(points) or not np.any(points[:, 0] == query_frame):
            raise ValueError("No currently observed agent in the requested prefix")
        order = np.lexsort((points[:, 0], points[:, 1]))
        self.points = points[order].copy()
        if np.any((np.diff(self.points[:, 0]) == 0) & (np.diff(self.points[:, 1]) == 0)):
            raise ValueError("Duplicate raw frame/agent key")
        self.points.flags.writeable = False
        self.starts, self.ends = track_boundaries(self.points)
        self.frame_order = np.argsort(self.points[:, 0], kind="stable")
        self.frame_values = self.points[self.frame_order, 0]
        self.max_neighbors = 8
        self.query_frame = query_frame
        self.metadata = dict(id=recording_id, physical_scene="unassigned_not_independence_evidence",
            data_role="input_adapter_only", source_admission=False,
            coordinate_claim="dataset_local_unverified", observed_steps=8, predicted_steps=12,
            raw_annotation_stride=1, effective_seconds=None, labels_available=False,
            observation_mode="offline_annotation_prefix_not_sensor_time_certificate")

    def get_scene_inputs(self):
        scene = super().get_scene_inputs(self.query_frame, 12, history_steps=8)
        visible = self.points[self.points[:, 0] == self.query_frame]
        scene["observed_agent_ids"] = visible[:, 1].astype(np.int64)
        scene["data_role"] = "input_adapter_only"
        scene["source_admission"] = False
        for agent in scene["agents"]:
            inputs = agent["inputs"]
            center = agent["coordinate_transform"]["origin_xy"]
            scale = agent["coordinate_transform"]["scale"]
            other = visible[visible[:, 1] != agent["agent_id"]]
            distances = np.linalg.norm(other[:, 2:4] - center, axis=1)
            nearest = np.lexsort((other[:, 1], distances))[:8]
            own = self.points[self.points[:, 1] == agent["agent_id"]]
            velocity = own[-1, 2:4] - own[-2, 2:4]
            ttcs, closings = [], []
            for row in other[nearest]:
                past = self.points[self.points[:, 1] == row[1]]
                # A missing previous raw frame does not imply a stationary neighbor
                # and must not create a velocity estimate across an annotation gap.
                if len(past) < 2 or past[-2, 0] != self.query_frame - 1:
                    continue
                relative = row[2:4] - center
                relative_velocity = past[-1, 2:4] - past[-2, 2:4] - velocity
                squared = float(relative_velocity @ relative_velocity)
                dot = float(relative @ relative_velocity)
                ttc = -dot / squared if squared > 1e-12 and dot < 0 else 120.
                ttcs.append(min(ttc / 12., 10.))
                closings.append(max(0., -dot / max(float(np.linalg.norm(relative)), 1e-8)) / scale)
            inputs["causal_features"][9] = min(ttcs, default=10.)
            inputs["causal_features"][10] = max(closings, default=0.)
        return scene

    def geometry_batch(self):
        scene = self.get_scene_inputs()
        if not scene["agents"]:
            return np.empty((0, 476), dtype=np.float32), scene
        geometry = np.stack([geometry_features(agent["inputs"]) for agent in scene["agents"]])
        if geometry.shape != (len(scene["agents"]), 476) or not np.isfinite(geometry).all():
            raise ValueError("Causal geometry does not match the fixed predictor schema")
        return geometry, scene

    def get_labels(self, *args, **kwargs):
        raise PermissionError("Input-only adapter has no future-label access")

    def get_scene_labels(self, *args, **kwargs):
        raise PermissionError("Input-only adapter has no future-label access")

    def get_inputs(self, *args, **kwargs):
        raise ValueError("Use the explicit shared-scene query, not future-conditioned window indices")

    def __len__(self):
        return len(self.get_scene_inputs()["agents"])
