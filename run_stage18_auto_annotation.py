from __future__ import annotations

import argparse

from src.stage18_pipeline import auto_annotate, write_current_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    write_current_state()
    auto_annotate(quick=args.quick)

