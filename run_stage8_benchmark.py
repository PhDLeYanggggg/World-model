from __future__ import annotations

import argparse

from src.evaluation.stage8_benchmark import run_benchmark, write_outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="all_available")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    datasets = None if args.datasets == "all_available" else [x.strip() for x in args.datasets.split(",") if x.strip()]
    payload = write_outputs(run_benchmark(datasets=datasets, quick=args.quick))
    print("Wrote outputs/reports/metrics_stage8.json")
    print(f"variants={len(payload['variants'])}")


if __name__ == "__main__":
    main()
