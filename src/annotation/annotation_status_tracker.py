from __future__ import annotations

from src.orchestrator.research_state import read_json


def latest_annotation_status() -> dict:
    summary = read_json("outputs/reports/stage12_final_summary.json", default={}) or {}
    return {
        "gold_human": 0,
        "silver_human_confirmed": int(summary.get("human_confirmed_scenes", 0) or 0),
        "silver_rule_confirmed": int(summary.get("silver_rule_confirmed_scenes", 0) or 0),
        "inferred_only": int(summary.get("inferred_only_scenes", 0) or 0),
        "note": "Silver/human counts are inherited from Stage 12; rule-confirmed is not gold.",
    }

