from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from src.evaluation.evaluate_stage9_per_agent import evaluate_checkpoint


REPORT_DIR = Path("outputs/reports")
CKPT_DIR = Path("outputs/checkpoints/stage9")


def checkpoint_paths() -> List[Path]:
    names = [
        "per_agent_no_scene.json",
        "per_agent_scene_only.json",
        "per_agent_goal_only.json",
        "per_agent_interaction_only.json",
        "per_agent_scene_goal.json",
        "per_agent_full_scene_goal_interaction.json",
    ]
    return [CKPT_DIR / name for name in names if (CKPT_DIR / name).exists()]


def run_stage9_benchmark() -> Dict:
    variants = [evaluate_checkpoint(path) for path in checkpoint_paths()]
    return {"stage": "9", "variants": variants, "latent_enabled": False, "smc_enabled": False}


def flatten(payload: Dict) -> List[Dict]:
    rows = []
    for variant in payload.get("variants", []):
        for dataset, drow in variant.get("datasets", {}).items():
            for subset, srow in drow.get("subsets", {}).items():
                for horizon, hrow in srow.get("horizons", {}).items():
                    rows.append(
                        {
                            "model": variant.get("variant"),
                            "dataset": dataset,
                            "subset": subset,
                            "horizon": horizon,
                            "FDE": hrow["FDE"],
                            "baseline_FDE": hrow["baseline_FDE"],
                            "improvement": hrow["improvement_over_strongest"],
                            "ADE": hrow["ADE"],
                            "episodes": srow["episodes"],
                            "alpha": srow["alpha_mean"],
                            "intervention": srow["intervention_rate"],
                            "residual": srow["residual_magnitude_mean"],
                            "physical_validity": hrow["physical_validity"],
                        }
                    )
    return rows


def target_rows(payload: Dict) -> List[Dict]:
    rows = []
    for variant in payload.get("variants", []):
        for dataset, drow in variant.get("datasets", {}).items():
            for subset in ["all", "easy", "hard", "baseline_failure", "goalbench_official", "ge5", "pedestrian_drone", "silver"]:
                srow = drow.get("subsets", {}).get(subset)
                if not srow or not srow.get("horizons"):
                    continue
                target = "10" if "10" in srow["horizons"] else max(srow["horizons"], key=lambda x: int(x))
                h = srow["horizons"][target]
                rows.append(
                    {
                        "model": variant.get("variant"),
                        "dataset": dataset,
                        "subset": subset,
                        "horizon": target,
                        "FDE": h["FDE"],
                        "baseline_FDE": h["baseline_FDE"],
                        "improvement": h["improvement_over_strongest"],
                        "episodes": srow["episodes"],
                        "alpha": srow["alpha_mean"],
                        "physical_validity": h["physical_validity"],
                    }
                )
    return rows


def write_stage9_benchmark(payload: Dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = flatten(payload)
    (REPORT_DIR / "metrics_stage9.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (REPORT_DIR / "metrics_stage9.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["model"])
        writer.writeheader()
        writer.writerows(rows)
    table = markdown_table(target_rows(payload))
    (REPORT_DIR / "metrics_table_stage9.md").write_text(table, encoding="utf-8")
    (REPORT_DIR / "report_stage9_benchmark.md").write_text("# Stage 9 Benchmark\n\n" + table, encoding="utf-8")


def markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return "_No rows._\n"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"
