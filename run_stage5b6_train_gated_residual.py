from __future__ import annotations

import argparse
import json

from src.training.train_stage5b6_gated_residual import train_all_variants


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.parse_args()
    payload = train_all_variants()
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

