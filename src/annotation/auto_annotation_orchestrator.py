from __future__ import annotations

from src.annotation.annotation_status_tracker import latest_annotation_status
from src.annotation.auto_annotation_task_builder import write_annotation_task_report
from src.annotation.human_review_queue import build_human_review_queue
from src.orchestrator.research_state import write_json, write_md


def run_annotation_orchestrator(limit: int = 10) -> dict:
    status = latest_annotation_status()
    tasks = write_annotation_task_report(limit=limit)
    queue = build_human_review_queue(limit=limit)
    payload = {"status": status, "tasks": tasks, "human_review_queue": queue}
    write_json("outputs/reports/annotation_orchestrator_report.json", payload)
    write_md(
        "outputs/reports/annotation_orchestrator_report.md",
        [
            "# Annotation Orchestrator Report",
            "",
            f"- gold_human: `{status['gold_human']}`",
            f"- silver_human_confirmed: `{status['silver_human_confirmed']}`",
            f"- silver_rule_confirmed: `{status['silver_rule_confirmed']}`",
            f"- review_queue: `{len(queue)}`",
            "",
            "AI/rule suggestions are not human gold. Test endpoints are forbidden for candidate goal construction.",
        ],
    )
    return payload

