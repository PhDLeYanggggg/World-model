from __future__ import annotations

import argparse

from src.stage19_pipeline import evaluate_stage19


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--medium", action="store_true")
    args = parser.parse_args()
    evaluate_stage19(quick=args.quick, medium=args.medium)

