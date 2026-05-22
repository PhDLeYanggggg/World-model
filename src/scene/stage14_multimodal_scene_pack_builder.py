from __future__ import annotations

from typing import Any, Dict

from src.stage14_pipeline import build_multimodal_scene_packs


def build(limit: int = 64) -> Dict[str, Any]:
    return build_multimodal_scene_packs(limit=limit)

