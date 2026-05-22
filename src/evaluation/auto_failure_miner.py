from __future__ import annotations

from src.orchestrator.gate_reader import summarize_latest_reports
from src.orchestrator.failure_analyzer import analyze_failures
from src.orchestrator.research_state import write_json, write_md


def write_auto_failure_analysis() -> dict:
    latest = summarize_latest_reports()
    payload = analyze_failures(latest["report_text"], latest["gate_text"])
    write_json("outputs/reports/auto_failure_analysis.json", payload)
    lines = ["# Auto Failure Analysis", "", "## Top Failures", "", *[f"- {item}" for item in payload["top_failures"]]]
    lines += ["", "## Recommended Fixes", "", *[f"- {item}" for item in payload["recommended_fixes"]]]
    write_md("outputs/reports/auto_failure_analysis.md", lines)
    return payload

