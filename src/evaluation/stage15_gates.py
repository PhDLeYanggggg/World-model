from __future__ import annotations

from typing import Any, Dict

from src.stage15_pipeline import evaluate_stage15_gates


def evaluate(loop_report: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return evaluate_stage15_gates(loop_report=loop_report)

