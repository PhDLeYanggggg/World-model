from __future__ import annotations

import argparse

from src.stage18_pipeline import eval_jepa


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--medium", action="store_true")
    args = parser.parse_args()
    eval_jepa(quick=args.quick, medium=args.medium)

