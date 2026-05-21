from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src.data_discovery.data_cards import write_data_cards
from src.data_discovery.dataset_registry import built_in_records, write_registry_outputs
from src.data_discovery.license_audit import write_license_report
from src.data_discovery.source_health_check import health_check_rows


def run_discovery(output_dir: str | Path = "outputs/data_registry") -> Dict:
    records = built_in_records()
    rows = write_registry_outputs(records, output_dir)
    write_license_report(rows)
    write_data_cards(rows)
    health = health_check_rows(rows, network=False)
    health_path = Path("outputs/reports/stage5_data/source_health_check_stage5.md")
    health_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Stage 5 Source Health Check", "", "| dataset | complete | missing | network_checked | status |", "| --- | --- | --- | --- | --- |"]
    for row in health:
        lines.append(f"| {row['dataset_name']} | {row['registry_complete']} | {row['missing_fields']} | {row['network_checked']} | {row['health_status']} |")
    health_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "records": rows,
        "candidate_count": len(rows),
        "downloadable_count": sum(1 for row in rows if row["download_status"] in {"downloaded", "downloadable"}),
        "gated_count": sum(1 for row in rows if row["download_status"] in {"gated", "requires_application"}),
        "t100_count": sum(1 for row in rows if row["can_evaluate_t100"]),
    }
