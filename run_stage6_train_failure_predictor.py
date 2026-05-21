from __future__ import annotations

import json

from src.training.train_baseline_failure_predictor import train_predictor


def main() -> int:
    metrics = train_predictor()
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

