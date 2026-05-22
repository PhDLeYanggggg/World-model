from __future__ import annotations

import argparse

from src.stage19_pipeline import auto_label_audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    auto_label_audit(quick=args.quick)

