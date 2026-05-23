from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


TOKEN_NAMES = [
    "agent_state",
    "agent_history",
    "scene_patch",
    "scene_sdf",
    "goal_region",
    "interaction_edge",
    "baseline_rollout",
    "horizon",
    "dataset",
    "time",
    "mask",
]


@dataclass(frozen=True)
class TokenSchema:
    token_to_features: Dict[str, List[int]]
    feature_names: List[str]

    @property
    def token_names(self) -> List[str]:
        return TOKEN_NAMES


def build_token_schema(feature_names: List[str]) -> TokenSchema:
    groups = {name: [] for name in TOKEN_NAMES}
    for idx, name in enumerate(feature_names):
        lower = name.lower()
        if lower.startswith("agent_type") or lower in {"speed_now", "vx_now", "vy_now", "ax_now", "ay_now", "accel_mag_now"}:
            groups["agent_state"].append(idx)
        elif any(key in lower for key in ["past", "heading", "curvature", "speed_mean", "speed_std", "speed_delta", "straightness"]):
            groups["agent_history"].append(idx)
        elif any(key in lower for key in ["scene_image", "split_within_scene"]):
            groups["scene_patch"].append(idx)
        elif any(key in lower for key in ["scene_clamp", "goal_source_visual_prior"]):
            groups["scene_sdf"].append(idx)
        elif any(key in lower for key in ["goal", "nearest_goal"]):
            groups["goal_region"].append(idx)
        elif any(key in lower for key in ["density", "nearest_neighbor", "ttc", "closing"]):
            groups["interaction_edge"].append(idx)
        elif any(key in lower for key in ["rollout", "baseline", "damped", "cv_", "ca_"]):
            groups["baseline_rollout"].append(idx)
        elif lower.startswith("horizon"):
            groups["horizon"].append(idx)
        elif lower in {"horizon_norm", "split_within_scene"}:
            groups["dataset"].append(idx)
        elif "start_frame" in lower:
            groups["time"].append(idx)
        else:
            groups["mask"].append(idx)
    for name, ids in groups.items():
        if not ids:
            groups[name] = [0]
    return TokenSchema(token_to_features=groups, feature_names=feature_names)
