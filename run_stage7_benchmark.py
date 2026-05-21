#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

from src.evaluation.stage7_benchmark import run_benchmark, write_outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="all_available")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    datasets = None if args.datasets == "all_available" else [x.strip() for x in args.datasets.split(",") if x.strip()]
    payload = write_outputs(run_benchmark(datasets))
    print(json.dumps({"variants": len(payload["variants"]), "datasets": payload["datasets"]}, indent=2))

