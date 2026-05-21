from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Stage 5 datasets after legal download.")
    parser.add_argument("--dataset", default="all_available")
    parser.add_argument("--raw-root", default="data/stage5_raw")
    parser.add_argument("--world-root", default="data/stage5_world_state")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for folder in [args.raw_root, args.world_root, "data/stage5_episodes", "outputs/reports/stage5_data"]:
        Path(folder).mkdir(parents=True, exist_ok=True)
    print(f"Prepared Stage 5 directory skeleton for {args.dataset}. dry_run={args.dry_run}")
    print("Use run_stage5_build_episodes.py after placing legally downloaded data under data/stage5_raw/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
