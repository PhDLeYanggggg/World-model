from __future__ import annotations

import argparse
import json

from src.data_unification.stage14_multimodal_episode_builder import build


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Stage 14 multimodal episodes.")
    parser.add_argument("--limit", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(build(limit=args.limit), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
