from __future__ import annotations

import argparse
import json

from src.search.stage15_deterministic_search import search


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 15 deterministic search.")
    parser.add_argument("--max-trials", type=int, default=12)
    args = parser.parse_args()
    print(json.dumps(search(max_trials=args.max_trials), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

