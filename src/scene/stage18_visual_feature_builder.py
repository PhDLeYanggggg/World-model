from __future__ import annotations


def visual_feature_placeholder(scene_pack: dict) -> list[float]:
    return [1.0 if scene_pack.get("scene_image") else 0.0, 1.0 if scene_pack.get("walkable_mask") else 0.0, 0.0]

