from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.auto_train_deterministic import write_deterministic_training_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto deterministic training planner.")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    payload = write_deterministic_training_plan()
    if not args.execute:
        print({"executed_training": False, "reason": payload["reason"]})
    else:
        print({"executed_training": False, "reason": "Execution path is intentionally not implemented for full training in auto quick mode."})


if __name__ == "__main__":
    main()
