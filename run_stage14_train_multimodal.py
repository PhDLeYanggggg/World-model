from __future__ import annotations

import argparse
import json

from src.training.train_stage14_multimodal import train_stage14_multimodal


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stage 14 deterministic multimodal scaffold.")
    parser.add_argument("--max-trials-per-family", type=int, default=1)
    parser.add_argument("--max-iterations", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(train_stage14_multimodal(args.max_trials_per_family, args.max_iterations), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

