from __future__ import annotations

import argparse
import json

from src.data_unification.stage14_ewap_t100_rebuilder import rebuild
from src.evaluation.stage14_t100_mask_validator import validate


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild EWAP t+100 per-agent episodes for Stage 14.")
    parser.add_argument("--max-episodes", type=int, default=64)
    args = parser.parse_args()
    result = rebuild(max_episodes=args.max_episodes)
    validation = validate()
    print(json.dumps({"rebuild": result, "validation": validation}, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

