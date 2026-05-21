from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


@dataclass
class InteractionFeatures:
    nearest_neighbor_distance_min: float = 999.0
    nearest_neighbor_distance_mean: float = 999.0
    time_to_collision_min: float = 999.0
    closing_speed_max: float = 0.0
    local_density_mean: float = 0.0
    crossing_angle_mean: float = 0.0
    relative_speed_mean: float = 0.0
    neighbor_count_mean: float = 0.0
    graph_attention_proxy: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                min(self.nearest_neighbor_distance_min, 50.0) / 50.0,
                min(self.nearest_neighbor_distance_mean, 50.0) / 50.0,
                min(self.time_to_collision_min, 50.0) / 50.0,
                np.clip(self.closing_speed_max, -20.0, 20.0) / 20.0,
                np.clip(self.local_density_mean, 0.0, 1.0),
                np.clip(self.crossing_angle_mean, -np.pi, np.pi) / np.pi,
                min(self.relative_speed_mean, 20.0) / 20.0,
                min(self.neighbor_count_mean, 20.0) / 20.0,
                np.clip(self.graph_attention_proxy, 0.0, 1.0),
            ],
            dtype=np.float64,
        )


class Stage5B6InteractionEncoder:
    """Past-only kNN interaction features from the world-state table.

    This encoder is deliberately causal: it only reads frames listed in the
    episode history up to the decision frame. It does not use future endpoint
    labels, mined hard labels, or test statistics.
    """

    def __init__(self, dataset: str, root: str | Path = "data/stage5b_world_state"):
        self.dataset = dataset
        self.path = Path(root) / dataset / "world_state.csv"
        self.table = pd.read_csv(self.path) if self.path.exists() else pd.DataFrame()

    def encode_episode(self, meta: Dict, k: int = 5) -> InteractionFeatures:
        if self.table.empty:
            return InteractionFeatures()
        scene_id = str(meta.get("scene_id", ""))
        agent_id = str(meta.get("primary_agent_id", ""))
        frames = list(meta.get("frames", []))[: int(meta.get("past_horizon", 10))]
        if not frames:
            return InteractionFeatures()
        per_frame = [self._encode_frame(scene_id, frame, agent_id, k=k) for frame in frames]
        distances = [f["nearest_distance"] for f in per_frame if np.isfinite(f["nearest_distance"])]
        mean_distances = [f["mean_distance"] for f in per_frame if np.isfinite(f["mean_distance"])]
        ttcs = [f["time_to_collision"] for f in per_frame if np.isfinite(f["time_to_collision"])]
        closings = [f["closing_speed"] for f in per_frame]
        densities = [f["density"] for f in per_frame]
        angles = [f["crossing_angle"] for f in per_frame]
        rel_speeds = [f["relative_speed"] for f in per_frame]
        counts = [f["neighbor_count"] for f in per_frame]
        attention = [f["attention_proxy"] for f in per_frame]
        return InteractionFeatures(
            nearest_neighbor_distance_min=float(min(distances)) if distances else 999.0,
            nearest_neighbor_distance_mean=float(np.mean(mean_distances)) if mean_distances else 999.0,
            time_to_collision_min=float(min(ttcs)) if ttcs else 999.0,
            closing_speed_max=float(max(closings)) if closings else 0.0,
            local_density_mean=float(np.mean(densities)) if densities else 0.0,
            crossing_angle_mean=float(np.mean(angles)) if angles else 0.0,
            relative_speed_mean=float(np.mean(rel_speeds)) if rel_speeds else 0.0,
            neighbor_count_mean=float(np.mean(counts)) if counts else 0.0,
            graph_attention_proxy=float(np.mean(attention)) if attention else 0.0,
        )

    def _encode_frame(self, scene_id: str, frame_id, agent_id: str, k: int) -> Dict[str, float]:
        frame = self.table[(self.table["scene_id"].astype(str) == scene_id) & (self.table["frame_id"].astype(int) == int(frame_id))]
        own = frame[frame["agent_id"].astype(str) == agent_id]
        others = frame[frame["agent_id"].astype(str) != agent_id]
        if own.empty or others.empty:
            return self._empty_frame()
        p = own[["x", "y"]].iloc[0].to_numpy(float)
        v = own[["vx", "vy"]].iloc[0].to_numpy(float)
        op = others[["x", "y"]].to_numpy(float)
        ov = others[["vx", "vy"]].to_numpy(float)
        rel = op - p
        dist = np.linalg.norm(rel, axis=1)
        order = np.argsort(dist)[: max(1, k)]
        rel = rel[order]
        dist = dist[order]
        relv = ov[order] - v
        rel_speed = np.linalg.norm(relv, axis=1)
        closing = -np.sum(rel * relv, axis=1) / np.maximum(dist, 1e-6)
        ttc = dist / np.maximum(closing, 1e-6)
        ttc[closing <= 0] = 999.0
        own_heading = np.arctan2(v[1], v[0])
        other_heading = np.arctan2(ov[order, 1], ov[order, 0])
        crossing = np.angle(np.exp(1j * (other_heading - own_heading)))
        attention = np.exp(-dist / 5.0) * np.maximum(closing, 0.0) / (np.maximum(rel_speed, 1e-6) + 1.0)
        return {
            "nearest_distance": float(np.min(dist)),
            "mean_distance": float(np.mean(dist)),
            "time_to_collision": float(np.min(ttc)),
            "closing_speed": float(np.max(closing)),
            "density": float(np.mean(dist < 5.0)),
            "crossing_angle": float(np.mean(np.abs(crossing))),
            "relative_speed": float(np.mean(rel_speed)),
            "neighbor_count": float(len(dist)),
            "attention_proxy": float(np.clip(np.mean(attention), 0.0, 1.0)),
        }

    @staticmethod
    def _empty_frame() -> Dict[str, float]:
        return {
            "nearest_distance": 999.0,
            "mean_distance": 999.0,
            "time_to_collision": 999.0,
            "closing_speed": 0.0,
            "density": 0.0,
            "crossing_angle": 0.0,
            "relative_speed": 0.0,
            "neighbor_count": 0.0,
            "attention_proxy": 0.0,
        }


def interaction_feature_matrix(dataset: str, metas: Iterable[Dict], root: str | Path = "data/stage5b_world_state") -> List[np.ndarray]:
    encoder = Stage5B6InteractionEncoder(dataset, root=root)
    return [encoder.encode_episode(meta).as_array() for meta in metas]

