from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    metrics = json.loads(Path("outputs/reports/metrics_stage5b5.json").read_text(encoding="utf-8"))
    datasets = sorted(metrics["datasets"])
    rows = []
    for train in datasets:
        for test in datasets:
            hrow = metrics["datasets"][test]["subsets"]["all"]["horizons"]
            target = "100" if "100" in hrow else max(hrow, key=lambda h: int(h))
            rows.append(
                {
                    "train_dataset": train,
                    "test_dataset": test,
                    "target_horizon": target,
                    "diagnostic_learned_fde": hrow[target]["FDE"],
                    "strongest_baseline_fde": hrow[target]["baseline_FDE"],
                    "improvement": hrow[target]["improvement_over_strongest"],
                    "note": "diagnostic only; quick model is dataset-conditioned, not true leave-one-dataset-out",
                }
            )
    out = Path("outputs/reports/report_stage5b5_cross_dataset_eval.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    out.write_text("# Stage 5B.5 Cross-Dataset Evaluation\n\n" + markdown_table(rows), encoding="utf-8")
    matrix = np.zeros((len(datasets), len(datasets)))
    for i, train in enumerate(datasets):
        for j, test in enumerate(datasets):
            matrix[i, j] = next(r["improvement"] for r in rows if r["train_dataset"] == train and r["test_dataset"] == test)
    fig = Path("outputs/figures/stage5b5/cross_dataset_matrix.png")
    fig.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="coolwarm", vmin=-0.2, vmax=0.2)
    plt.colorbar(label="target improvement")
    plt.xticks(range(len(datasets)), datasets, rotation=30, ha="right")
    plt.yticks(range(len(datasets)), datasets)
    plt.title("Stage 5B.5 diagnostic transfer matrix")
    plt.tight_layout()
    plt.savefig(fig, dpi=160)
    plt.close()
    print(json.dumps({"rows": rows, "figure": str(fig)}, indent=2))
    return 0


def markdown_table(rows):
    if not rows:
        return "_No rows._\n"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
