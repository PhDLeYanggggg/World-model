from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    # Stage 5-Data quick benchmark reuses the repaired Stage 4.5 TGSIM causal baseline results
    # until more datasets are legally downloaded and converted.
    source = Path("outputs/reports/metrics_stage4p5.json")
    out_json = Path("outputs/reports/stage5_baseline_metrics.json")
    out_csv = Path("outputs/reports/stage5_baseline_metrics.csv")
    out_md = Path("outputs/reports/stage5_baseline_table.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        metrics = {"status": "missing_stage4p5_metrics", "datasets": {}}
    else:
        stage4p5 = json.loads(source.read_text(encoding="utf-8"))
        metrics = {
            "status": "quick_partial",
            "official_note": "Only TGSIM is converted in this quick run; registry-only datasets are not benchmarked.",
            "datasets": {
                "TGSIM Foggy Bottom": {
                    "strongest_causal_baseline": "constant_turn_rate_velocity",
                    "metrics": stage4p5.get("constant_turn_rate_velocity", {}),
                    "all_baselines": {k: v for k, v in stage4p5.items() if k in {"constant_velocity_causal_fd", "damped_velocity", "constant_acceleration_causal", "constant_turn_rate_velocity", "identity_hand_physics", "tuned_hand_physics"}},
                }
            },
        }
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    rows = []
    for dataset, payload in metrics.get("datasets", {}).items():
        model = payload["strongest_causal_baseline"]
        h = payload["metrics"].get("horizons", {})
        rows.append({"dataset": dataset, "strongest_causal_baseline": model, "FDE@100": h.get("100", {}).get("FDE"), "ADE@100": h.get("100", {}).get("ADE")})
    out_csv.write_text("dataset,strongest_causal_baseline,FDE@100,ADE@100\n" + "\n".join(f"{r['dataset']},{r['strongest_causal_baseline']},{r['FDE@100']},{r['ADE@100']}" for r in rows) + "\n", encoding="utf-8")
    out_md.write_text(markdown_table(rows), encoding="utf-8")
    print(f"Wrote {out_json}")
    return 0


def markdown_table(rows):
    if not rows:
        return "_No baseline rows._\n"
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
