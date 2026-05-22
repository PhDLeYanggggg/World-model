from __future__ import annotations

from typing import Any, Dict

from src.stage14_pipeline import run_stage14_benchmark


def run_benchmark() -> Dict[str, Any]:
    return run_stage14_benchmark()

