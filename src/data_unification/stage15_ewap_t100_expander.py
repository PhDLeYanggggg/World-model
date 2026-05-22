from __future__ import annotations

from typing import Any, Dict

from src.stage15_pipeline import expand_ewap_rows


def expand(max_t100: int = 256, max_t50: int = 512) -> Dict[str, Any]:
    return expand_ewap_rows(max_t100=max_t100, max_t50=max_t50)

