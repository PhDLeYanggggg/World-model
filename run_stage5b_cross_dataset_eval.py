from __future__ import annotations

import argparse
import json

from src.evaluation.cross_dataset_eval_stage5b import run_cross_dataset_eval, write_cross_dataset_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    result = run_cross_dataset_eval()
    write_cross_dataset_report(result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
