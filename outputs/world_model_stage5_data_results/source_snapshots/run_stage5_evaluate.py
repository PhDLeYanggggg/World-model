from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.training.evaluate_stage5_foundation import evaluate_stage5_foundation_placeholder


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Stage 5 foundation checkpoint.")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    result = evaluate_stage5_foundation_placeholder()
    result["checkpoint"] = args.checkpoint
    Path("outputs/reports/report_stage5_cross_dataset_eval.md").write_text("# Stage 5 Cross-Dataset Eval\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
