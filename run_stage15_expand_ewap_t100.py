from __future__ import annotations

import argparse
import json

from src.data_unification.stage15_ewap_t100_expander import expand


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand Stage 15 EWAP t+50/t+100 per-agent rows.")
    parser.add_argument("--max-t100", type=int, default=256)
    parser.add_argument("--max-t50", type=int, default=512)
    args = parser.parse_args()
    print(json.dumps(expand(max_t100=args.max_t100, max_t50=args.max_t50), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

