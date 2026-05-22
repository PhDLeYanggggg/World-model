from __future__ import annotations

from typing import Any, Dict

from src.stage15_pipeline import run_stage15_search


def search(max_trials: int = 12) -> Dict[str, Any]:
    return run_stage15_search(max_trials=max_trials)

