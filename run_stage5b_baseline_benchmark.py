from __future__ import annotations

import argparse
import json

from src.evaluation.baseline_benchmark_stage5b import available_datasets, benchmark_dataset, write_outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="all_available")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    datasets = available_datasets() if args.datasets == "all_available" else [x.strip() for x in args.datasets.split(",") if x.strip()]
    rows = [benchmark_dataset(dataset, split="test") for dataset in datasets]
    payload = write_outputs(rows)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
