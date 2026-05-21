from __future__ import annotations

import argparse
import json

from src.evaluation.hardbench_builder import build_hardbench, write_outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="all_available")
    args = parser.parse_args()
    datasets = None if args.datasets == "all_available" else [x.strip() for x in args.datasets.split(",") if x.strip()]
    payload = write_outputs(build_hardbench(datasets))
    print(json.dumps({k: v for k, v in payload.items() if k != "records"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

