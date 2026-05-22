from __future__ import annotations


QUALITY_ORDER = {
    "inferred_only": 0,
    "silver_rule_confirmed": 1,
    "ai_visual_silver": 2,
    "self_audited_silver": 3,
    "gold_human": 4,
}


def is_automatic_silver(quality: str) -> bool:
    return quality in {"silver_rule_confirmed", "ai_visual_silver", "self_audited_silver"}

