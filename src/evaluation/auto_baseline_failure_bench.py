from __future__ import annotations

from src.evaluation.auto_hardbench_builder import write_auto_hardbench_report
from src.orchestrator.research_state import write_json, write_md


def write_auto_baseline_failure_report() -> dict:
    hard = write_auto_hardbench_report()
    result = {
        "source": hard["source"],
        "baseline_failure_records": hard["records"],
        "sufficient_for_diagnostic": hard["records"] >= 100,
    }
    write_json("outputs/reports/auto_baseline_failure_report.json", result)
    write_md(
        "outputs/reports/auto_baseline_failure_report.md",
        [
            "# Auto BaselineFailureBench Report",
            "",
            f"- baseline_failure_records: `{result['baseline_failure_records']}`",
            f"- sufficient_for_diagnostic: `{result['sufficient_for_diagnostic']}`",
        ],
    )
    return result

