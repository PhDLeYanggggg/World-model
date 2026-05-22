from __future__ import annotations

from src.orchestrator.research_state import write_json, write_md


def write_annotation_upgrade_report() -> dict:
    payload = {
        "policy": {
            "inferred_only_to_silver_rule_confirmed": "allowed only for high-confidence rule confirmation",
            "ai_visual_silver": "allowed from visual inspection, not gold",
            "silver_human_confirmed": "requires human confirmation",
            "gold_human": "requires direct human annotation and validation",
        },
        "automatic_upgrades_performed": 0,
    }
    write_json("outputs/reports/annotation_upgrade_report.json", payload)
    write_md(
        "outputs/reports/annotation_upgrade_report.md",
        [
            "# Annotation Upgrade Report",
            "",
            "- automatic_upgrades_performed: `0`",
            "- No annotation is promoted to human/gold without explicit human confirmation.",
        ],
    )
    return payload

