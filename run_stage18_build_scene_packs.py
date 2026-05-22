from __future__ import annotations

import argparse

from src.stage18_pipeline import build_scene_packs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    build_scene_packs(quick=args.quick)

