from __future__ import annotations

from typing import Any, Dict

from src.stage15_pipeline import run_oracle_diagnostics


def evaluate() -> Dict[str, Any]:
    return run_oracle_diagnostics()

