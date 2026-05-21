#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = {
    "sdd": {
        "download_status": "requires_manual_license",
        "license": "Stanford Drone Dataset non-commercial research license; manual agreement required",
        "official_note": "Do not auto-download. User must download from the official SDD page and pass --sdd-root.",
    },
    "opentraj": {
        "download_status": "source_specific",
        "license": "varies by underlying dataset",
        "official_note": "Use an existing OpenTraj-compatible local root; do not treat registry-only datasets as converted.",
    },
    "trajnet_full": {
        "download_status": "local_if_present",
        "license": "dataset-specific original terms",
        "official_note": "The local TrajNet++ tree is already supported when data/stage5b_raw/trajnetplusplusdataset exists.",
    },
    "eth_ucy_full": {
        "download_status": "local_if_present",
        "license": "ETH/UCY academic terms; verify before redistribution",
        "official_note": "The current ETH/UCY fallback is t+10 only; it does not satisfy long-horizon pedestrian gate.",
    },
    "aerialmpt_long": {
        "download_status": "user_path_only",
        "license": "verify source license",
        "official_note": "Existing bauma sample is short; longer sequences must be provided by the user.",
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
                "reason": "Stage 8 does not bypass license/login/manual terms. Provide local paths for gated datasets.",
            }
        )
    out = Path("outputs/reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage8_download_plan.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (out / "stage8_download_failures.md").write_text(
        "# Stage 8 Download Notes\n\n"
        "No gated pedestrian/drone dataset was automatically downloaded. SDD requires manual license acceptance.\n",
        encoding="utf-8",
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()

