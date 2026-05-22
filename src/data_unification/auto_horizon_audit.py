from __future__ import annotations

from pathlib import Path
from typing import Dict

from src.orchestrator.research_state import read_json, write_md


def build_auto_horizon_audit() -> Dict:
    stage12 = read_json("outputs/reports/stage12_horizon_audit.json", default={}) or {}
    return {
        "source": "stage12_horizon_audit",
        "official_velocity": "causal_fd unless native velocity is proven causal",
        "pixel_metric_rule": "pixel-space remains pixel-space unless homography/scale exists",
        "datasets": stage12,
    }


def write_auto_horizon_audit() -> Dict:
    payload = build_auto_horizon_audit()
    lines = [
        "# Auto Horizon Audit",
        "",
        "- This report reuses the latest Stage 12 horizon audit.",
        "- t+100 produced by downsampling must report effective seconds.",
        "- fake t+100 construction is forbidden.",
    ]
    write_md("outputs/reports/auto_horizon_audit.md", lines)
    return payload

