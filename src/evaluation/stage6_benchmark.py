from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

from src.evaluation.evaluate_stage6_failure_aware_model import evaluate_checkpoint
from src.evaluation.leakage_audit_stage5b import available_stage5b_datasets


REPORT_DIR = Path("outputs/reports")
CKPT_DIR = Path("outputs/checkpoints/stage6")


def checkpoint_paths() -> List[Path]:
    names = [
        "failure_predictor_only_gate.json",
        "learned_alpha_gate.json",
        "hybrid_failure_predictor_plus_learned_gate.json",
        "no_interaction_ablation.json",
        "scalar_interaction_ablation.json",
        "graph_interaction_ablation.json",
    ]
    return [CKPT_DIR / name for name in names if (CKPT_DIR / name).exists()]


def run_benchmark(datasets: List[str] | None = None) -> Dict:
    datasets = datasets or available_stage5b_datasets()
    variants = [evaluate_checkpoint(path, datasets) for path in checkpoint_paths()]
    return {"stage": "6", "datasets": datasets, "variants": variants}


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
            for subset in ["all", "easy", "hard", "baseline_failure", "verified_t50", "verified_t100"]:
                srow = drow.get("subsets", {}).get(subset)
                if not srow:
                    continue
                horizons = srow.get("horizons", {})
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


def interaction_ablation(payload: Dict) -> List[Dict]:
    rows = []
    for variant in payload.get("variants", []):
        name = variant.get("variant", "")
        if "interaction" not in name:
            continue
        hard = []
        failure = []
        for drow in variant.get("datasets", {}).values():
            for subset_name, bucket in [("hard", hard), ("baseline_failure", failure)]:
                srow = drow.get("subsets", {}).get(subset_name)
                if not srow:
                    continue
                horizons = srow.get("horizons", {})
                target = "100" if "100" in horizons else max(horizons, key=lambda h: int(h))
                bucket.append(float(horizons[target]["improvement_over_strongest"]))
        rows.append(
            {
                "model": name,
                "mean_hard_improvement": round(sum(hard) / max(len(hard), 1), 6),
                "mean_failure_improvement": round(sum(failure) / max(len(failure), 1), 6),
            }
        )
    return rows


def write_outputs(payload: Dict) -> Dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = flatten_rows(payload)
    (REPORT_DIR / "metrics_stage6.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (REPORT_DIR / "metrics_stage6.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["model"])
        writer.writeheader()
        writer.writerows(rows)
    (REPORT_DIR / "metrics_table_stage6.md").write_text(markdown_table(target_rows(payload)), encoding="utf-8")
    ia = interaction_ablation(payload)
    (REPORT_DIR / "stage6_interaction_ablation.json").write_text(json.dumps(ia, indent=2), encoding="utf-8")
    (REPORT_DIR / "stage6_interaction_ablation.md").write_text("# Stage 6 Interaction Ablation\n\n" + markdown_table(ia), encoding="utf-8")
    (REPORT_DIR / "report_stage6_benchmark.md").write_text("# Stage 6 Benchmark\n\n" + markdown_table(target_rows(payload)), encoding="utf-8")
    return payload


def markdown_table(rows: List[Dict]) -> str:
    if not rows:
        return "_No rows._\n"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"

