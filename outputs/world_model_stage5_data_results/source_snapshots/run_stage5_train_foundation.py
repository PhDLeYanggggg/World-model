from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.training.train_stage5_foundation import train_stage5_foundation_quick


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Stage 5 deterministic foundation world-model scaffold.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if cfg.get("mode") == "full" and not cfg.get("allow_full_run", False):
        print("Full training is disabled by default. Set allow_full_run only after data gates pass.")
        return 2
    result = train_stage5_foundation_quick(cfg.get("training", {}))
    Path("outputs/reports/report_stage5_foundation_training.md").write_text("# Stage 5 Foundation Training\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
