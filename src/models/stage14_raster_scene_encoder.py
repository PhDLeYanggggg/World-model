from __future__ import annotations

from typing import Dict


def encode_raster_features(scene_pack: Dict[str, object]) -> Dict[str, object]:
    return {
        "has_walkable": bool(scene_pack.get("walkable_suggestion")),
        "has_goals": bool(scene_pack.get("candidate_goals")),
        "metric_status": scene_pack.get("metric_status", "unknown"),
    }

