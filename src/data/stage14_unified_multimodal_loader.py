from __future__ import annotations

from typing import Any, Dict

from src.stage14_pipeline import multimodal_data_audit


def audit_available_sources() -> Dict[str, Any]:
    return multimodal_data_audit()

