from __future__ import annotations

import argparse
import json

from src.data_unification.horizon_builder import available_stage5b_world_state, horizon_audit_dataset, write_horizon_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="all_available")
    args = parser.parse_args()
    datasets = available_stage5b_world_state() if args.datasets == "all_available" else [x.strip() for x in args.datasets.split(",") if x.strip()]
    rows = [horizon_audit_dataset(dataset) for dataset in datasets]
    write_horizon_report(rows)
    print(json.dumps({"horizon_audit": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
