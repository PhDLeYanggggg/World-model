from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run_cross_dataset_eval() -> Dict:
    baselines_path = Path("outputs/reports/stage5b_baseline_metrics.json")
    learned_path = Path("outputs/reports/stage5b_deterministic_metrics.json")
    if not baselines_path.exists():
        return {"status": "missing_baseline_metrics", "pairs": []}
    baselines = json.loads(baselines_path.read_text(encoding="utf-8"))["datasets"]
    learned = json.loads(learned_path.read_text(encoding="utf-8")) if learned_path.exists() else {}
    datasets = sorted(baselines)
    pairs = []
    for train in datasets:
        for test in datasets:
            test_row = baselines[test]
            h = str(test_row["target_horizon_for_strongest"])
            base_name = test_row["strongest_causal_baseline"]
            base_fde = test_row["all_baselines"][base_name]["horizons"][h]["FDE"]
            learned_fde = learned.get("multistep", {}).get("learned_metrics", {}).get(test, {}).get("horizons", {}).get(h, {}).get("FDE")
            pairs.append(
                {
                    "train_dataset": train,
                    "test_dataset": test,
                    "target_horizon": int(h),
                    "strongest_baseline_fde": base_fde,
                    "stage5b_dataset_specific_learned_fde": learned_fde,
                    "domain_gap_note": "true leave-one-dataset-out training is not available in this quick deterministic linear residual run",
                }
            )
    return {"status": "executed_diagnostic_only", "datasets": datasets, "pairs": pairs}


def write_cross_dataset_report(result: Dict) -> None:
    out = Path("outputs/reports/report_stage5b_cross_dataset_eval.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.with_suffix(".json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# Stage 5B Cross-Dataset Evaluation",
        "",
        "This quick run performs a diagnostic cross-dataset matrix over actual converted datasets. It does not claim true leave-one-dataset-out learned transfer because the Stage 5B deterministic model is a dataset-specific linear residual head.",
        "",
        "| train | test | horizon | strongest baseline FDE | learned FDE | note |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in result.get("pairs", []):
        lines.append(
            f"| {row['train_dataset']} | {row['test_dataset']} | {row['target_horizon']} | "
            f"{row['strongest_baseline_fde']} | {row['stage5b_dataset_specific_learned_fde']} | {row['domain_gap_note']} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    make_heatmap(result)


def make_heatmap(result: Dict) -> None:
    datasets = result.get("datasets", [])
    if not datasets:
        return
    fde = {row["test_dataset"]: row["strongest_baseline_fde"] for row in result.get("pairs", []) if row["train_dataset"] == row["test_dataset"]}
    values = np.array([[fde.get(test, np.nan) for test in datasets] for _ in datasets], dtype=float)
    path = Path("outputs/figures/stage5b/cross_dataset_transfer_matrix.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(5, len(datasets) * 1.1), max(4, len(datasets) * 0.9)))
    plt.imshow(values, cmap="viridis")
    plt.colorbar(label="strongest baseline FDE")
    plt.xticks(range(len(datasets)), datasets, rotation=35, ha="right")
    plt.yticks(range(len(datasets)), datasets)
    plt.title("Stage 5B diagnostic transfer matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
