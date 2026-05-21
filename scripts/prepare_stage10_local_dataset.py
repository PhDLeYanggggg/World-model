from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.sdd_loader import load_sdd_trajectories
from src.stage10_common import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a user-provided local pedestrian/drone dataset for Stage 10.")
    parser.add_argument("--dataset", required=True, choices=["sdd"])
    parser.add_argument("--data", required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    if args.dataset == "sdd":
        table, meta = load_sdd_trajectories(args.data, quick=args.quick, max_rows=args.max_rows)
        out_dir = Path("data/stage10_world_state/sdd")
        out_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(out_dir / "world_state.csv", index=False)
        write_json(out_dir / "metadata.json", meta)
        print(json.dumps({"dataset": "sdd", "rows": len(table), "output": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
