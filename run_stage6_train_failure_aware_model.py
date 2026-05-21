from __future__ import annotations

import argparse
import json

from src.training.train_stage6_failure_aware_model import train_failure_aware_models


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.parse_args()
    payload = train_failure_aware_models()
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

