#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = {
    "sdd": {
        "status": "manual_license",
        "message": "Stanford Drone Dataset requires license review/manual download. This script will not bypass terms.",
        "suggested_action": "Download from the official Stanford Drone Dataset page, then pass the local root to Stage 7 audit.",
    },
    "trajnet_full": {
        "status": "already_supported_if_local_repo_exists",
        "message": "Use existing data/stage5b_raw/trajnetplusplusdataset if present.",
        "suggested_action": "Run run_stage7_scene_data_audit.py.",
    },
    "eth_ucy_full": {
        "status": "already_supported_if_local_repo_exists",
        "message": "ETH/UCY fallback currently comes through the local TrajNet++ original-data tree.",
        "suggested_action": "Run run_stage7_scene_data_audit.py.",
    },
    "aerialmpt_long": {
        "status": "local_only",
        "message": "Existing local AerialMPT extraction can be audited, but no t+50/t+100 guarantee.",
        "suggested_action": "Add longer AerialMPT sequences manually if available.",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["all", *DATASETS.keys()])
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute-download", action="store_true")
    args = parser.parse_args()
    selected = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    records = []
    for name, info in selected.items():
        record = {"dataset": name, "download_executed": False, **info}
        if args.execute_download:
            record["download_executed"] = False
            record["message"] += " Automatic download is intentionally disabled unless the source is public and license-free."
        records.append(record)
    out = Path("outputs/reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage7_scene_data_download_plan.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

