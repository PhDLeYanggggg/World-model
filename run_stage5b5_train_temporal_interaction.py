from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.training.train_stage5b5_temporal_numpy import train_numpy_temporal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stage5b5_temporal_quick.yaml")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    args = parser.parse_args()
    if args.backend == "torch":
        from src.training.train_stage5b5_temporal import train_temporal
        from src.training.evaluate_stage5b5_temporal import evaluate_checkpoint

        results = []
        for mode in ["direct_multi_horizon", "recurrent_rollout", "hybrid"]:
            train_mode = "recurrent_rollout" if mode == "recurrent_rollout" else ("hybrid" if mode == "hybrid" else "direct_multi_horizon")
            trained = train_temporal(mode=train_mode, epochs=args.epochs)
            metrics = evaluate_checkpoint(trained["checkpoint"], split="test")
            results.append({"mode": mode, **trained, "test_metrics": metrics})
    else:
        trained = train_numpy_temporal()
        results = [{"mode": "numpy_temporal_interaction_ridge", **trained}]
    out = Path("outputs/reports/stage5b5_temporal_training.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": args.config, "runs": results}, indent=2), encoding="utf-8")
    print(json.dumps({"runs": [{k: v for k, v in r.items() if k != "test_metrics"} for r in results]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
