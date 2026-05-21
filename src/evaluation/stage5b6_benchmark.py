from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

from src.evaluation.baseline_failure_oracle import baseline_failure_rows, write_oracle_report
from src.evaluation.evaluate_stage5b6_gated_residual import evaluate_checkpoint
from src.evaluation.leakage_audit_stage5b import available_stage5b_datasets


REPORT_DIR = Path("outputs/reports")
CKPT_DIR = Path("outputs/checkpoints/stage5b6")
OFFICIAL_GATED_VARIANTS = {
    "gated_residual_all_data",
    "gated_residual_hard_weighted",
    "gated_residual_failure_classifier_aux",
}


def checkpoint_paths() -> List[Path]:
    return sorted(CKPT_DIR.glob("*.json"))


def run_benchmark(datasets: List[str] | None = None) -> Dict:
    datasets = datasets or available_stage5b_datasets()
    variants = []
    for ckpt in checkpoint_paths():
        result = evaluate_checkpoint(ckpt, datasets)
        variants.append(result)
    best_alpha_rows = best_variant_alpha_rows(variants)
    write_oracle_report(baseline_failure_rows(), alpha_rows=best_alpha_rows)
    return {"stage": "5B.6", "datasets": datasets, "variants": variants}


def best_variant_alpha_rows(variants: List[Dict]) -> List[Dict]:
    # Pick the lowest mean all-test target FDE variant for oracle calibration diagnostics.
    best = None
    best_score = float("inf")
    for variant in variants:
        if variant.get("variant") not in OFFICIAL_GATED_VARIANTS:
            continue
        scores = []
        for drow in variant.get("datasets", {}).values():
            all_subset = drow.get("subsets", {}).get("all", {})
            horizons = all_subset.get("horizons", {})
            if not horizons:
                continue
            target = "100" if "100" in horizons else max(horizons, key=lambda h: int(h))
            scores.append(horizons[target]["FDE"])
        if scores and sum(scores) / len(scores) < best_score:
            best = variant
            best_score = sum(scores) / len(scores)
    return best.get("alpha_rows", []) if best else []


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
                            "residual_mag": srow["residual_magnitude_mean"],
                            "physical_validity": srow["physical_validity_rate"],
                        }
                    )
    return rows


def target_rows(payload: Dict) -> List[Dict]:
    rows = []
    for variant in payload.get("variants", []):
        name = variant.get("variant", "unknown")
        for dataset, drow in variant.get("datasets", {}).items():
            for subset in ["all", "hard", "easy"]:
                srow = drow.get("subsets", {}).get(subset)
                if not srow:
                    continue
                horizons = srow.get("horizons", {})
                if not horizons:
                    continue
                target = "100" if "100" in horizons else max(horizons, key=lambda h: int(h))
                hrow = horizons[target]
                rows.append(
                    {
                        "model": name,
                        "dataset": dataset,
                        "subset": subset,
                        "target_horizon": target,
                        "FDE": hrow["FDE"],
                        "baseline_FDE": hrow["baseline_FDE"],
                        "improvement": hrow["improvement_over_strongest"],
                        "ci": f"[{hrow['bootstrap_ci']['ci_low']}, {hrow['bootstrap_ci']['ci_high']}]",
                        "episodes": srow["episodes"],
                        "alpha_mean": srow["alpha_mean"],
                    }
                )
    return rows


def interaction_ablation_rows(payload: Dict) -> List[Dict]:
    mapping = {
        "no_interaction_ablation": "no interaction",
        "nearest_neighbor_scalar_ablation": "nearest-neighbor scalar features only",
        "graph_attention_interaction_ablation": "graph attention interaction",
        "graph_attention_temporal_history_ablation": "graph attention + temporal neighbor history",
    }
    rows = []
    for variant in payload.get("variants", []):
        name = variant.get("variant", "unknown")
        if name not in mapping:
            continue
        hard_imps = []
        all_imps = []
        for drow in variant.get("datasets", {}).values():
            for subset_name, bucket in [("all", all_imps), ("hard", hard_imps)]:
                srow = drow.get("subsets", {}).get(subset_name)
                if not srow:
                    continue
                horizons = srow.get("horizons", {})
                if not horizons:
                    continue
                target = "100" if "100" in horizons else max(horizons, key=lambda h: int(h))
                bucket.append(float(horizons[target]["improvement_over_strongest"]))
        rows.append(
            {
                "ablation": mapping[name],
                "mean_all_target_improvement": round(sum(all_imps) / max(len(all_imps), 1), 6),
                "mean_hard_target_improvement": round(sum(hard_imps) / max(len(hard_imps), 1), 6),
                "note": "quick deterministic ablation; interaction is past-only from kNN world-state table",
            }
        )
    return rows


def write_outputs(payload: Dict) -> Dict:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = flatten_rows(payload)
    (REPORT_DIR / "metrics_stage5b6.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (REPORT_DIR / "metrics_stage5b6.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["model"])
        writer.writeheader()
        writer.writerows(rows)
    (REPORT_DIR / "metrics_table_stage5b6.md").write_text(markdown_table(target_rows(payload)), encoding="utf-8")
    ablations = interaction_ablation_rows(payload)
    (REPORT_DIR / "stage5b6_interaction_ablation.json").write_text(json.dumps(ablations, indent=2), encoding="utf-8")
    (REPORT_DIR / "stage5b6_interaction_ablation.md").write_text("# Stage 5B.6 Interaction Ablation\n\n" + markdown_table(ablations), encoding="utf-8")
    (REPORT_DIR / "report_stage5b6_benchmark.md").write_text(
        "# Stage 5B.6 Benchmark\n\n"
        "All-test, hard-test, and verified long-horizon metrics are separated. Hard subsets below reliability thresholds remain diagnostic.\n\n"
        + markdown_table(target_rows(payload)),
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
