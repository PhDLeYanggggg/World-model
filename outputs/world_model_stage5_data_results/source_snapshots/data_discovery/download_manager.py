from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


def plan_downloads(rows: Iterable[Dict], dataset: str | None = None, priority_only: bool = False, max_gb: float | None = None, dry_run: bool = True) -> List[Dict]:
    plans = []
    for row in rows:
        key = slug(row["dataset_name"])
        if dataset and dataset not in {key, row["dataset_name"]}:
            continue
        if priority_only and int(row.get("priority_score", 0)) < 70:
            continue
        status = row.get("download_status")
        action = "skip"
        command = row.get("download_command") or ""
        reason = ""
        if status == "downloaded":
            action = "already_available"
        elif status == "downloadable":
            action = "dry_run_download" if dry_run else "download"
        elif status in {"gated", "requires_application"}:
            action = "placeholder_only"
            reason = "requires registration, license agreement, or application"
        else:
            action = "skip"
            reason = f"download_status={status}"
        plans.append(
            {
                "dataset_name": row["dataset_name"],
                "dataset_key": key,
                "download_status": status,
                "action": action,
                "dry_run": dry_run,
                "max_gb": max_gb,
                "command": command,
                "reason": reason,
            }
        )
    return plans


def write_download_plan(plans: List[Dict], path: str | Path = "outputs/reports/stage5_data/download_plan_stage5.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(plans, indent=2), encoding="utf-8")
    md = ["# Stage 5 Download Plan", "", "| dataset | status | action | reason | command |", "| --- | --- | --- | --- | --- |"]
    for plan in plans:
        md.append(f"| {plan['dataset_name']} | {plan['download_status']} | {plan['action']} | {plan['reason']} | `{plan['command']}` |")
    Path(str(path).replace(".json", ".md")).write_text("\n".join(md) + "\n", encoding="utf-8")


def slug(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")
