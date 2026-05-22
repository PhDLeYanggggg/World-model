from __future__ import annotations

from typing import Any, Dict

from src.stage15_pipeline import run_stage15_benchmark


def evaluate() -> Dict[str, Any]:
    return run_stage15_benchmark()

