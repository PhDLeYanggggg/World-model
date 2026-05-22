from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class SceneEncoder:
    """Encodes scene-pack metadata without pretending inferred labels are gold."""

    def encode(self, scene_features: Dict[str, Any] | None, agent_count: int) -> Dict[str, np.ndarray]:
        scene_features = scene_features or {}
        quality = str(scene_features.get("annotation_quality", scene_features.get("quality", "unknown")))
        quality_score = {
            "gold_human": 1.0,
            "silver_human_confirmed": 0.8,
            "silver_rule_confirmed": 0.45,
            "ai_visual_silver": 0.35,
            "inferred_only": 0.15,
        }.get(quality, 0.1)
        metric_flag = 1.0 if str(scene_features.get("metric_status", "")).lower() == "metric" else 0.0
        has_scene = 1.0 if scene_features else 0.0
        return {
            "scene_quality": np.full(agent_count, quality_score, dtype=np.float64),
            "metric_flag": np.full(agent_count, metric_flag, dtype=np.float64),
            "has_scene": np.full(agent_count, has_scene, dtype=np.float64),
        }

