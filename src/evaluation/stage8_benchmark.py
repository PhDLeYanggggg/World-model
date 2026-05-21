from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from src.evaluation.evaluate_stage8_world_model import evaluate_checkpoint
from src.evaluation.stage8_goalbench_gold import available_stage8_datasets


REPORT_DIR = Path("outputs/reports")
CKPT_DIR = Path("outputs/checkpoints/stage8")


def checkpoint_paths() -> List[Path]:
    names = [
        "goal_only_v2.json",
        "scene_only_v2.json",
        "multiagent_only_v2.json",
        "scene_goal_v2.json",
        "scene_goal_multiagent_v2.json",
        "topk_goal_diagnostic_v2.json",
    ]
    return [CKPT_DIR / name for name in names if (CKPT_DIR / name).exists()]


def run_benchmark(datasets: List[str] | None = None, quick: bool = True) -> Dict:
    datasets = datasets or available_stage8_datasets()
    variants = [evaluate_checkpoint(path, datasets=datasets) for path in checkpoint_paths()]
    payload = {"stage": "8", "datasets": datasets, "variants": variants, "quick": bool(quick), "latent_enabled": False, "smc_enabled": False}
    return payload


def flatten_rows(payload: Dict) -> List[Dict]:
    rows = []
    for variant in payload.get("variants", []):
        name = variant.get("variant", "unknown")
        for dataset, drow in variant.get("datasets", {}).items():
            for subset, srow in drow.get("subsets", {}).items():
                for horizon, hrow in srow.get("horizons", {}).items():
                    rows.append(
                        {
                            "model": name,
                            "dataset": dataset,
                            "subset": subset,
                            "horizon": horizon,
                            "ADE": hrow["ADE"],
                            "FDE": hrow["FDE"],
                            "baseline_ADE": hrow["baseline_ADE"],
                            "baseline_FDE": hrow["baseline_FDE"],
                            "improvement": hrow["improvement_over_strongest"],
                            "ci_low": hrow["bootstrap_ci"]["ci_low"],
                            "ci_high": hrow["bootstrap_ci"]["ci_high"],
                            "episodes": srow["episodes"],
                            "alpha_mean": srow["alpha_mean"],
                            "failure_probability": srow["failure_probability_mean"],
                            "intervention_rate": srow["intervention_rate"],
                            "false_intervention_rate": srow["false_intervention_rate"],
                            "residual_magnitude": srow["residual_magnitude_mean"],
                        }
                    )
    return rows


def target_rows(payload: Dict) -> List[Dict]:
    rows = []
    for variant in payload.get("variants", []):
        name = variant.get("variant", "unknown")
        for dataset, drow in variant.get("datasets", {}).items():
            for subset in ["all", "hardbench", "baseline_failure", "easy", "scene_gold", "inferred_only", "multi_agent", "pedestrian_drone", "traffic", "verified_t50", "verified_t100"]:
                srow = drow.get("subsets", {}).get(subset)
                if not srow or not srow.get("horizons"):
                    continue
                target = "100" if "100" in srow["horizons"] else max(srow["horizons"], key=lambda h: int(h))
                h = srow["horizons"][target]
                rows.append(
                    {
                        "model": name,
                        "dataset": dataset,
                        "subset": subset,
                        "target_horizon": target,
                        "FDE": h["FDE"],
                        "baseline_FDE": h["baseline_FDE"],
                        "improvement": h["improvement_over_strongest"],
                        "episodes": srow["episodes"],
                        "alpha_mean": srow["alpha_mean"],
                        "intervention_rate": srow["intervention_rate"],
                        "false_intervention_rate": srow["false_intervention_rate"],
                    }
                )
    return rows


def write_outputs(payload: Dict) -> Dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = flatten_rows(payload)
    (REPORT_DIR / "metrics_stage8.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (REPORT_DIR / "metrics_stage8.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["model"])
        writer.writeheader()
        writer.writerows(rows)
    table = markdown_table(target_rows(payload))
    (REPORT_DIR / "metrics_table_stage8.md").write_text(table, encoding="utf-8")
    (REPORT_DIR / "report_stage8_benchmark.md").write_text(
        "# Stage 8 Benchmark\n\n"
        "Official deterministic comparison against each dataset's strongest causal baseline. Top-k goal rows are diagnostic only and are not latent generative modeling.\n\n"
        + table,
        encoding="utf-8",
    )
    return payload


def markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return "_No rows._\n"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"
