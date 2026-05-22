from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.search.stage13_deterministic_search import run_stage13_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 13 deterministic bounded residual search.")
    parser.add_argument("--max-trials-per-family", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--no-training", action="store_true")
    args = parser.parse_args()
    result = run_stage13_search(
        max_trials_per_family=args.max_trials_per_family,
        max_iterations=args.max_iterations,
        allow_training=not args.no_training,
    )
    print(json.dumps({k: result[k] for k in ["executed_training", "trial_count", "episode_count", "best_overall_gate_candidate"]}, indent=2, default=str))


if __name__ == "__main__":
    main()

