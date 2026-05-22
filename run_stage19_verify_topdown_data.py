from __future__ import annotations

import argparse

from src.stage19_pipeline import verify_topdown_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    verify_topdown_data(quick=args.quick)

