from __future__ import annotations


def self_check_annotation(annotation):
    checks = annotation.get("self_checks", {})
    return {
        "passed": all(bool(v) for k, v in checks.items() if k not in {"goal_count_not_top3_saturated", "consensus_score", "endpoint_coverage"}),
        "gold_human": False,
    }

