from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_discovery.auto_dataset_ranker import rank_datasets
from src.data_discovery.auto_dataset_search import discover_candidates
from src.data_discovery.legal_download_planner import build_legal_download_plan
from src.orchestrator.research_state import write_json, write_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Legal auto dataset discovery/planning.")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute-download", action="store_true")
    args = parser.parse_args()

    rows = rank_datasets(discover_candidates())
    if args.dataset != "all":
        rows = [row for row in rows if row["dataset_name"].lower() == args.dataset.lower()]
    plan = build_legal_download_plan(rows)

    out_json = Path("outputs/reports/auto_data_discovery_report.json")
    write_json(out_json, {"dry_run": not args.execute_download, "datasets": rows, "download_plan": plan})
    lines = [
        "# Auto Data Discovery Report",
        "",
        f"- dry_run: `{not args.execute_download}`",
        f"- candidate_count: `{len(rows)}`",
        "",
        "| dataset | score | status | auto_download | legal notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows[:30]:
        lines.append(
            f"| {row['dataset_name']} | {row.get('auto_rank_score')} | {row.get('download_status')} | "
            f"{row.get('can_download_automatically')} | {row.get('legal_notes', '')} |"
        )
    lines += [
        "",
        "No gated or license-restricted data is downloaded by this dry-run planner.",
    ]
    write_md("outputs/reports/auto_data_discovery_report.md", lines)
    print({"candidate_count": len(rows), "report": str(out_json), "execute_download": bool(args.execute_download)})


if __name__ == "__main__":
    main()
