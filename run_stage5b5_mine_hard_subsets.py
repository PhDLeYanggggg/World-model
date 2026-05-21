from __future__ import annotations

import argparse
import json

from src.evaluation.hard_subset_miner import mine_dataset, write_hard_subset_report
from src.evaluation.leakage_audit_stage5b import available_stage5b_datasets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="all_available")
    args = parser.parse_args()
    datasets = available_stage5b_datasets() if args.datasets == "all_available" else [x.strip() for x in args.datasets.split(",") if x.strip()]
    rows = [mine_dataset(dataset) for dataset in datasets]
    write_hard_subset_report(rows)
    print(json.dumps({"hard_subsets": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
