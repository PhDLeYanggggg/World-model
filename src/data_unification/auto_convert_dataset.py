from __future__ import annotations

from typing import Dict

from src.orchestrator.research_state import read_json, write_json, write_md


def summarize_available_conversions() -> Dict:
    stage12 = read_json("outputs/reports/stage12_data_audit.json", default={}) or {}
    if isinstance(stage12, list):
        converted = [row.get("dataset_name", "unknown") for row in stage12 if row.get("loader_status") == "loaded"]
    elif isinstance(stage12, dict):
        converted = list(stage12.keys())
    else:
        converted = []
    return {
        "converted_from_latest_stage": converted,
        "official_velocity_policy": "causal_fd",
        "central_fd_policy": "diagnostic_only",
        "goal_policy": "train_endpoints_only_for_suggestions; no test endpoints",
    }


def write_auto_conversion_report() -> Dict:
    payload = summarize_available_conversions()
    write_json("outputs/reports/auto_conversion_report.json", payload)
    write_md(
        "outputs/reports/auto_conversion_report.md",
        [
            "# Auto Conversion Report",
            "",
            f"- converted_from_latest_stage: `{payload['converted_from_latest_stage']}`",
            f"- official_velocity_policy: `{payload['official_velocity_policy']}`",
            f"- goal_policy: `{payload['goal_policy']}`",
        ],
    )
    return payload
