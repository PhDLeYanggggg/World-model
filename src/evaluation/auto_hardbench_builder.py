from __future__ import annotations

from src.orchestrator.research_state import read_json, write_json, write_md


def write_auto_hardbench_report() -> dict:
    records = read_json("outputs/reports/stage12_hard_failure_report.json", default=[]) or []
    if isinstance(records, list):
        count = len(records)
    elif isinstance(records, dict) and isinstance(records.get("records"), list):
        count = len(records["records"])
    elif isinstance(records, dict):
        count = int(records.get("records", 0) or 0)
    else:
        count = 0
    result = {"source": "stage12_hard_failure_report", "records": count}
    write_json("outputs/reports/auto_hardbench_report.json", result)
    write_md("outputs/reports/auto_hardbench_report.md", ["# Auto HardBench Report", "", f"- records: `{count}`"])
    return result
