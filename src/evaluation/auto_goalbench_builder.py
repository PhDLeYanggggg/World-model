from __future__ import annotations

from src.orchestrator.research_state import read_json, write_json, write_md


def write_auto_goalbench_report() -> dict:
    payload = read_json("outputs/reports/stage12_goalbench_v4_report.json", default={}) or {}
    result = {"source": "stage12_goalbench_v4_report", **payload}
    write_json("outputs/reports/auto_goalbench_report.json", result)
    write_md(
        "outputs/reports/auto_goalbench_report.md",
        [
            "# Auto GoalBench Report",
            "",
            f"- official_records: `{result.get('official_records', 'unknown')}`",
            "- Majority-beating goal prediction is not proven by this report; it only summarizes available records.",
        ],
    )
    return result

