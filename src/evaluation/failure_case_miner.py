from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.baseline_benchmark_stage5b import load_dataset_episodes, rollout
from src.evaluation.stage5b5_benchmark import load_model, predict_residual
from src.training.train_stage5b5_temporal_numpy import feature_vector, hard_lookup, load_json


def mine_failures() -> List[Dict]:
    metrics = load_json("outputs/reports/metrics_stage5b5.json", {"datasets": {}})
    baselines = load_json("outputs/reports/stage5b_baseline_metrics.json", {"datasets": {}})
    hard = hard_lookup()
    model = load_model()
    failures = []
    for dataset, row in metrics.get("datasets", {}).items():
        baseline = baselines["datasets"][dataset]["strongest_causal_baseline"]
        for ep in load_dataset_episodes(dataset, split="test"):
            states = ep["states"]
            meta = ep["meta"]
            past = int(meta.get("past_horizon", 10))
            future_len = states.shape[0] - past
            target = 100 if future_len >= 100 else min(10, future_len)
            dt = float(meta.get("dt_s", 1.0))
            base = rollout(states[:past], target, dt, baseline)[1:]
            true = states[past : past + target]
            hrow = hard.get((dataset, int(meta.get("episode_id", -1))), {})
            x = feature_vector(dataset, states[:past], hrow, target, future_len)
            residual, alpha = predict_residual(model, dataset, target, x)
            pred = base.copy()
            for step in range(target):
                pred[step, 0, 0:2] += residual * ((step + 1) / target)
            b_fde = float(np.linalg.norm(base[-1, 0, 0:2] - true[-1, 0, 0:2]))
            l_fde = float(np.linalg.norm(pred[-1, 0, 0:2] - true[-1, 0, 0:2]))
            if l_fde > 1.2 * max(b_fde, 1e-9) or np.linalg.norm(residual) > 3.5:
                failure = {
                    "dataset": dataset,
                    "scene_id": meta.get("scene_id"),
                    "episode_id": int(meta.get("episode_id", -1)),
                    "agent_id": meta.get("primary_agent_id"),
                    "event_type": ",".join(hrow.get("events", ["unknown"])),
                    "hardness": hrow.get("hardness", "unknown"),
                    "baseline_FDE": round(b_fde, 6),
                    "learned_FDE": round(l_fde, 6),
                    "ratio": round(l_fde / max(b_fde, 1e-9), 6),
                    "likely_cause": likely_cause(dataset, hrow, residual),
                    "recommended_fix": "Improve deterministic temporal model and validate residual gates per horizon/subset.",
                }
                plot_failure(failure, states[:past], true, base, pred)
                failures.append(failure)
    failures = sorted(failures, key=lambda x: x["ratio"], reverse=True)[:20]
    return failures


def likely_cause(dataset: str, hrow: Dict, residual: np.ndarray) -> str:
    if dataset.startswith("tgsim"):
        return "traffic route/intent missing; residual over-corrects a strong causal velocity baseline"
    if hrow.get("hardness") == "hard":
        return "short pedestrian snippet has nonlinear motion but no long context or scene map"
    if np.linalg.norm(residual) > 3.5:
        return "residual magnitude near clipping threshold"
    return "baseline already strong; residual adds noise"


def plot_failure(failure, hist, true, base, pred):
    out = Path("outputs/figures/stage5b5/failure_cases")
    out.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.plot(hist[:, 0, 0], hist[:, 0, 1], "ko-", label="history", markersize=2)
    plt.plot(true[:, 0, 0], true[:, 0, 1], "g-", label="ground truth")
    plt.plot(base[:, 0, 0], base[:, 0, 1], "b--", label="baseline")
    plt.plot(pred[:, 0, 0], pred[:, 0, 1], "r--", label="learned")
    plt.legend(fontsize=7)
    plt.title(f"{failure['dataset']} ep {failure['episode_id']}")
    plt.tight_layout()
    plt.savefig(out / f"{failure['dataset']}_episode_{failure['episode_id']}.png", dpi=150)
    plt.close()


def write_failure_report(failures: List[Dict]):
    out = Path("outputs/reports/failure_analysis_stage5b5.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    out.write_text("# Stage 5B.5 Failure Analysis\n\n" + markdown_table(failures), encoding="utf-8")


def markdown_table(rows):
    if not rows:
        return "_No failure cases matched the configured thresholds._\n"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"
