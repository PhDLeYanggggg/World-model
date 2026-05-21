from __future__ import annotations

import argparse
import json

from src.evaluation.leakage_audit_stage5b import audit_dataset, available_stage5b_datasets, write_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="all_available")
    args = parser.parse_args()
    datasets = available_stage5b_datasets() if args.datasets == "all_available" else [x.strip() for x in args.datasets.split(",") if x.strip()]
    audits = [audit_dataset(dataset) for dataset in datasets]
    write_report(audits)
    print(json.dumps({"audits": audits}, indent=2))
    return 0 if all(row["passed"] for row in audits) else 1


if __name__ == "__main__":
    raise SystemExit(main())
