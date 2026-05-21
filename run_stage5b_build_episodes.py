from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data_unification.build_stage5b_episodes import build_stage5b_dataset, write_all_summary


DEFAULT_PATHS = {
    "tgsim": "https://data.transportation.gov/resource/brzy-6zfh.csv?$limit=50000",
    "tgsim_i90": "https://data.transportation.gov/resource/9uas-hf8b.csv?$limit=50000",
    "trajnet": "data/stage5b_raw/trajnetplusplusdataset/data/trajnet_original/stanford",
    "eth_ucy": "data/stage5b_raw/trajnetplusplusdataset/data/trajnet_original/biwi",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Stage 5B real dataset world-state and episodes.")
    parser.add_argument("--datasets", default="tgsim,trajnet,eth_ucy")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--tgsim-data", default=None)
    parser.add_argument("--trajnet-root", default=None)
    parser.add_argument("--eth-ucy-root", default=None)
    parser.add_argument("--sdd-root", default=None)
    args = parser.parse_args()
    requested = [item.strip() for item in args.datasets.split(",") if item.strip() and item.strip() != "all_available"]
    if "all_available" in args.datasets:
        requested = ["tgsim", "trajnet", "eth_ucy"]
    path_overrides = {"tgsim": args.tgsim_data, "trajnet": args.trajnet_root, "eth_ucy": args.eth_ucy_root, "sdd": args.sdd_root}
    summaries = []
    failures = []
    for dataset in requested:
        path = path_overrides.get(dataset) or DEFAULT_PATHS.get(dataset)
        if not path:
            failures.append({"dataset": dataset, "reason": "no path provided"})
            continue
        try:
            summaries.append(build_stage5b_dataset(dataset, path, quick=args.quick))
        except Exception as exc:
            failures.append({"dataset": dataset, "path": path, "reason": str(exc)})
    write_all_summary(summaries)
    Path("outputs/reports/stage5b_build_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    if failures:
        lines = ["# Stage 5B Build Failures", "", "| dataset | path | reason |", "| --- | --- | --- |"]
        for row in failures:
            lines.append(f"| {row.get('dataset')} | {row.get('path','')} | {row.get('reason')} |")
        Path("outputs/reports/stage5b_build_failures.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"summaries": summaries, "failures": failures}, indent=2)[:4000])
    return 0 if summaries else 1


if __name__ == "__main__":
    raise SystemExit(main())
