from __future__ import annotations

from src.orchestrator.failure_analyzer import strongest_baseline_summary
from src.orchestrator.research_state import write_json, write_md


def write_auto_benchmark_report() -> dict:
    baselines = strongest_baseline_summary()
    payload = {
        "source": "stage12_rebenchmark",
        "strongest_baseline_summary": baselines,
        "new_benchmark_run": False,
        "reason": "Quick auto loop reuses Stage 12 deterministic rebenchmark.",
    }
    write_json("outputs/reports/auto_benchmark_report.json", payload)
    lines = ["# Auto Benchmark Report", "", "- new_benchmark_run: `False`", "", "| dataset | best learned | baseline FDE | improvement |", "| --- | --- | --- | --- |"]
    for dataset, row in baselines.items():
        lines.append(f"| {dataset} | {row.get('best_learned_variant')} | {row.get('strongest_baseline_FDE')} | {row.get('improvement')} |")
    write_md("outputs/reports/auto_benchmark_report.md", lines)
    return payload

