from __future__ import annotations

from typing import Any, Dict

from src.stage14_pipeline import rebuild_ewap_t100_episodes


def rebuild(max_episodes: int = 64) -> Dict[str, Any]:
    return rebuild_ewap_t100_episodes(max_episodes=max_episodes)

