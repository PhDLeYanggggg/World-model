from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from src.evaluation.evaluate_stage7_goal_conditioned_world_model import evaluate_checkpoint
from src.evaluation.leakage_audit_stage5b import available_stage5b_datasets


REPORT_DIR = Path("outputs/reports")
CKPT_DIR = Path("outputs/checkpoints/stage7")


def checkpoint_paths() -> List[Path]:
    names = [
        "goal_only_residual.json",
        "scene_only_residual.json",
        "interaction_scalar_residual.json",
        "goal_interaction_residual.json",
        "goal_scene_interaction_residual.json",
        "topk_goal_mixture_diagnostic.json",
    ]
    return [CKPT_DIR / name for name in names if (CKPT_DIR / name).exists()]


def run_benchmark(datasets: List[str] | None = None) -> Dict:
    datasets = datasets or available_stage5b_datasets()
    variants = [evaluate_checkpoint(path, datasets) for path in checkpoint_paths()]
    return {"stage": "7", "datasets": datasets, "variants": variants, "latent_enabled": False, "smc_enabled": False}


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
                            "failure_prob": srow["failure_probability_mean"],
                            "intervention_rate": srow["intervention_rate"],
                            "false_intervention_rate": srow["false_intervention_rate"],
                            "residual_mag": srow["residual_magnitude_mean"],
                        }
                    )
    return rows


def target_rows(payload: Dict) -> List[Dict]:
    rows = []
    for variant in payload.get("variants", []):
        name = variant.get("variant", "unknown")
        for dataset, drow in variant.get("datasets", {}).items():
            for subset in ["all", "easy", "hard", "baseline_failure", "scene_grounded", "pedestrian_drone", "traffic", "verified_t50", "verified_t100"]:
                srow = drow.get("subsets", {}).get(subset)
                if not srow:
                    continue
                horizons = srow.get("horizons", {})
                if not horizons:
                    continue
                target = "100" if "100" in horizons else max(horizons, key=lambda h: int(h))
                h = horizons[target]
                rows.append(
                    {
                        "model": name,
                        "dataset": dataset,
                        "subset": subset,
                        "target": target,
                        "FDE": h["FDE"],
                        "baseline_FDE": h["baseline_FDE"],
                        "improvement": h["improvement_over_strongest"],
                        "episodes": srow["episodes"],
                        "alpha": srow["alpha_mean"],
                        "intervention": srow["intervention_rate"],
                        "false_intervention": srow["false_intervention_rate"],
                    }
                )
    return rows


def write_outputs(payload: Dict) -> Dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = flatten_rows(payload)
    (REPORT_DIR / "metrics_stage7.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (REPORT_DIR / "metrics_stage7.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["model"])
        writer.writeheader()
        writer.writerows(rows)
    (REPORT_DIR / "metrics_table_stage7.md").write_text(markdown_table(target_rows(payload)), encoding="utf-8")
    (REPORT_DIR / "report_stage7_benchmark.md").write_text("# Stage 7 Benchmark\n\n" + markdown_table(target_rows(payload)), encoding="utf-8")
    return payload


def markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return "_No rows._\n"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"

