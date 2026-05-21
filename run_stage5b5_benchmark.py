from __future__ import annotations

import argparse
import json

from src.evaluation.leakage_audit_stage5b import available_stage5b_datasets
from src.evaluation.stage5b5_benchmark import run_benchmark, write_benchmark_outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="all_available")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    datasets = available_stage5b_datasets() if args.datasets == "all_available" else [x.strip() for x in args.datasets.split(",") if x.strip()]
    payload = run_benchmark(datasets)
    write_benchmark_outputs(payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
