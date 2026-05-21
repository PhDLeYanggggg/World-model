from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_discovery.dataset_registry import built_in_records, registry_as_dicts
from src.data_discovery.download_manager import plan_downloads, write_download_plan


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 5 dataset download manager. Defaults to dry-run.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--max-gb", type=float, default=None)
    parser.add_argument("--priority-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    dry_run = True if args.dry_run or not args.dataset else False
    rows = registry_as_dicts(built_in_records())
    plans = plan_downloads(rows, dataset=args.dataset, priority_only=args.priority_only, max_gb=args.max_gb, dry_run=dry_run)
    write_download_plan(plans)
    for plan in plans:
        print(f"{plan['dataset_key']}: {plan['action']} ({plan['download_status']}) {plan['reason']}")
    if not dry_run:
        print("No bulk downloader runs by default. This script currently writes a legal download plan and placeholders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
