from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.training.train_stage5b_deterministic import train_and_evaluate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage5b_deterministic_quick.yaml")
    args = parser.parse_args()
    one = train_and_evaluate(mode="one_step")
    multi = train_and_evaluate(mode="multistep")
    payload = {"config": args.config, "one_step": one, "multistep": multi}
    out = Path("outputs/reports/stage5b_deterministic_metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
