from __future__ import annotations

from typing import Any, Dict

from src.stage14_pipeline import validate_stage14_t100_masks


def validate() -> Dict[str, Any]:
    return validate_stage14_t100_masks()

