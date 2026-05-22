from __future__ import annotations

from typing import Any, Dict

from src.stage15_pipeline import read_json


def compare() -> Dict[str, Any]:
    return read_json("outputs/reports/stage15_ewap_t100_expansion_report.json", {})

