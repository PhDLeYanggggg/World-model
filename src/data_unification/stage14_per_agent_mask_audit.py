from __future__ import annotations

from typing import Any, Dict

from src.stage14_pipeline import audit_ewap_t100_masks


def run_audit() -> Dict[str, Any]:
    return audit_ewap_t100_masks()

