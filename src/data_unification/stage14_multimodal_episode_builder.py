from __future__ import annotations

from typing import Any, Dict

from src.stage14_pipeline import build_multimodal_episodes


def build(limit: int = 256) -> Dict[str, Any]:
    return build_multimodal_episodes(limit=limit)

