from __future__ import annotations

import argparse
import json

from src.training.train_stage15_baseline_preserving import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 15 baseline-preserving deterministic model.")
    parser.add_argument("--max-trials", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(train(max_trials=args.max_trials), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

