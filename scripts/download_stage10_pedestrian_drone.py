from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = {
    "sdd": {
        "status": "requires_manual_license",
        "official_url": "https://cvgl.stanford.edu/projects/uav_data/",
        "note": "Stanford Drone Dataset is non-commercial and requires the user to follow the official terms. This script does not bypass license acceptance.",
    },
    "opentraj": {
        "status": "manual_or_git_prepare",
        "official_url": "https://github.com/crowdbotp/OpenTraj",
        "note": "OpenTraj is a toolkit and points to datasets with separate licenses; prepare source-specific data locally.",
    },
    "trajnet_full": {
        "status": "already_partially_available",
        "official_url": "https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge",
        "note": "A bundled TrajNet++ original-data subset is already converted locally through Stage 8.5.",
    },
    "eth_ucy_full": {
        "status": "already_partially_available",
        "official_url": "https://icu.ee.ethz.ch/research/datsets.html",
        "note": "A bundled ETH/UCY-style fallback is already converted locally through Stage 8.5.",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 10 legal pedestrian/drone dataset download planner.")
    parser.add_argument("--dataset", default="all", choices=["all", *DATASETS.keys()])
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--execute-download", action="store_true", default=False)
    parser.add_argument("--max-gb", type=float, default=5.0)
    args = parser.parse_args()
    selected = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    rows = []
    for name, info in selected.items():
        row = {"dataset": name, **info, "max_gb": args.max_gb, "download_attempted": False, "downloaded": False}
        if args.execute_download and info["status"] == "requires_manual_license":
            row["failure_reason"] = "Manual license/terms required. Download was intentionally not attempted."
        elif args.execute_download:
            row["failure_reason"] = "No direct safe download command is encoded for this source; provide local path via verify/prepare scripts."
        else:
            row["failure_reason"] = "dry_run_only"
        rows.append(row)
    out = Path("outputs/reports/stage10_download_plan.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    md = ["# Stage 10 Download Plan", "", "| dataset | status | official_url | downloaded | note | failure_reason |", "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        md.append(f"| {r['dataset']} | {r['status']} | {r['official_url']} | {r['downloaded']} | {r['note']} | {r['failure_reason']} |")
    Path("outputs/reports/stage10_download_plan.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"planned": len(rows), "downloaded": 0, "report": str(out)}, indent=2))


if __name__ == "__main__":
    main()
