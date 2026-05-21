from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = {
    "sdd": {
        "name": "Stanford Drone Dataset",
        "status": "manual_or_user_path_required",
        "license_note": "CC BY-NC-SA 3.0 / non-commercial; user must verify terms before download/use.",
        "command": "Provide --sdd-root pointing to an already accepted local copy, or manually download from the official SDD site.",
    },
    "opentraj": {
        "name": "OpenTraj-supported pedestrian datasets",
        "status": "manual_or_user_path_required",
        "license_note": "Mixed licenses by source dataset; verify each dataset before use.",
        "command": "Provide --opentraj-root after preparing OpenTraj data locally.",
    },
    "trajnet_full": {
        "name": "full TrajNet++",
        "status": "partially_available_locally",
        "license_note": "Use official TrajNet++ terms; current project only has a small prepared fallback.",
        "command": "Provide --trajnet-root for a full prepared tree.",
    },
    "eth_ucy_full": {
        "name": "full ETH/UCY",
        "status": "partially_available_locally",
        "license_note": "Use original ETH/UCY dataset terms; current project only has a short fallback.",
        "command": "Provide --eth-ucy-root for full raw tracks.",
    },
    "aerialmpt_long": {
        "name": "AerialMPT longer sequences",
        "status": "not_found_locally",
        "license_note": "Use AerialMPT dataset terms.",
        "command": "Provide --aerialmpt-root if longer annotated sequences are available.",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute-download", action="store_true")
    parser.add_argument("--max-gb", type=float, default=0.0)
    args = parser.parse_args()
    selected = DATASETS if args.dataset == "all" else {args.dataset: DATASETS.get(args.dataset)}
    records = []
    for key, row in selected.items():
        if not row:
            records.append({"dataset": key, "status": "unknown_dataset"})
            continue
        status = "dry_run_only"
        note = "Stage 6 does not bypass licenses/logins. No automatic large download is attempted."
        if args.execute_download:
            status = "not_downloaded_requires_manual_license_or_user_path"
            note = row["command"]
        records.append({"dataset_key": key, **row, "run_status": status, "note": note, "max_gb": args.max_gb})
    out = Path("outputs/reports/stage6_download_records.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    lines = ["# Stage 6 Pedestrian/Drone Download Plan", "", "No license-gated or manual-terms dataset is downloaded automatically.", ""]
    for r in records:
        lines.append(f"- `{r['dataset_key']}`: {r['run_status']} - {r['note']}")
    Path("outputs/reports/stage6_download_failures.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"records": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

