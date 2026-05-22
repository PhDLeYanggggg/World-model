from __future__ import annotations

from typing import Any, Dict

from src.stage14_pipeline import evaluate_stage14_gates


def evaluate_gates(loop_report: Dict[str, Any] | None = None) -> Dict[str, Any]:
    return evaluate_stage14_gates(loop_report=loop_report)

