#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = {
    "sdd": {
        "license": "Stanford Drone Dataset non-commercial research license; manual agreement required",
        "status": "requires_manual_download",
        "instruction": "Download SDD from the official Stanford Drone Dataset page, accept the license manually, then pass --sdd-root to the audit/build commands.",
    },
    "opentraj": {
        "license": "varies by underlying dataset",
        "status": "user_path_only",
        "instruction": "Provide a local OpenTraj-compatible root. Do not treat registry-only datasets as converted data.",
    },
    "trajnet_full": {
        "license": "dataset-specific original terms",
        "status": "local_if_present",
        "instruction": "The existing data/stage5b_world_state/trajnet conversion can be reused for Stage 8.5.",
    },
    "eth_ucy_full": {
        "license": "ETH/UCY academic terms; verify before redistribution",
        "status": "local_if_present",
        "instruction": "The existing data/stage5b_world_state/eth_ucy conversion can be reused for Stage 8.5.",
    },
    "aerialmpt_long": {
        "license": "verify source license",
        "status": "user_path_only",
        "instruction": "Provide a longer AerialMPT sequence; bauma3 remains too short for t+50/t+100 verified pedestrian claims.",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["all", *DATASETS.keys()])
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute-download", action="store_true")
    parser.add_argument("--max-gb", type=float, default=5.0)
    args = parser.parse_args()
    selected = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    records = []
    for name, meta in selected.items():
        records.append(
            {
                "dataset": name,
                **meta,
                "execute_download_requested": bool(args.execute_download),
                "download_executed": False,
                "reason": "Stage 8.5 does not bypass license/login/manual terms. Use local paths for gated datasets.",
            }
        )
    out = Path("outputs/reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage8p5_download_plan.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (out / "stage8p5_download_failures.md").write_text(
        "# Stage 8.5 Download Notes\n\nNo gated pedestrian/drone dataset was automatically downloaded. SDD requires manual license acceptance and a local path.\n",
        encoding="utf-8",
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
