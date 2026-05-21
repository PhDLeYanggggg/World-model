from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 12 legal pedestrian/drone dataset download planner.")
    parser.add_argument("--dataset", default="all", choices=["all", "sdd", "opentraj", "trajnet", "eth_ucy", "aerialmpt"])
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--execute-download", action="store_true")
    args = parser.parse_args()
    plan = {
        "sdd": "Requires user to accept Stanford Drone Dataset non-commercial license and provide local path; no automatic bypass.",
        "opentraj": "Requires user-provided OpenTraj root or per-dataset legal download.",
        "trajnet": "Local TrajNet++ original-data tree already present under data/stage5b_raw/trajnetplusplusdataset.",
        "eth_ucy": "Local ewap_dataset_light.tgz already present and used for Stage 12 long-horizon audit.",
        "aerialmpt": "Local DLR_AerialMPT_Dataset.zip already present under data/aerialmpt.",
    }
    selected = plan if args.dataset == "all" else {args.dataset: plan[args.dataset]}
    print(json.dumps({"execute_download": bool(args.execute_download), "note": "No gated/license dataset is downloaded automatically.", "plan": selected}, indent=2))


if __name__ == "__main__":
    main()
