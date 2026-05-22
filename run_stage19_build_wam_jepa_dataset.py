from __future__ import annotations

import argparse

from src.stage19_pipeline import build_wam_jepa_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    build_wam_jepa_dataset(quick=args.quick)

