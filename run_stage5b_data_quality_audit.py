from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.leakage_audit_stage5b import available_stage5b_datasets


def read_world_state(dataset: str) -> pd.DataFrame:
    return pd.read_csv(Path("data/stage5b_world_state") / dataset / "world_state.csv")


def summarize(dataset: str) -> dict:
    table = read_world_state(dataset)
    track_lengths = table.groupby(["scene_id", "agent_id"])["frame_id"].nunique()
    dt = table["dt_s"].replace([float("inf"), -float("inf")], pd.NA).dropna()
    speed = table["speed"].replace([float("inf"), -float("inf")], pd.NA).dropna()
    accel = table["acceleration_norm"].replace([float("inf"), -float("inf")], pd.NA).dropna()
    gaps = table.sort_values(["scene_id", "agent_id", "frame_id"]).groupby(["scene_id", "agent_id"])["frame_id"].diff().fillna(1.0)
    return {
        "dataset_name": dataset,
        "total_rows": int(len(table)),
        "total_scenes": int(table["scene_id"].nunique()),
        "total_agents": int(table["agent_id"].nunique()),
        "total_tracks": int(track_lengths.shape[0]),
        "total_frames": int(table["frame_id"].nunique()),
        "mean_track_length": round(float(track_lengths.mean()), 3) if len(track_lengths) else 0.0,
        "p50_track_length": round(float(track_lengths.quantile(0.5)), 3) if len(track_lengths) else 0.0,
        "p95_track_length": round(float(track_lengths.quantile(0.95)), 3) if len(track_lengths) else 0.0,
        "dt_median": round(float(dt.median()), 6) if len(dt) else 0.0,
        "speed_p50": round(float(speed.quantile(0.5)), 4) if len(speed) else 0.0,
        "speed_p95": round(float(speed.quantile(0.95)), 4) if len(speed) else 0.0,
        "accel_p50": round(float(accel.quantile(0.5)), 4) if len(accel) else 0.0,
        "accel_p95": round(float(accel.quantile(0.95)), 4) if len(accel) else 0.0,
        "missing_frame_gap_count": int((gaps > 1).sum()),
        "abnormal_jump_count": int((speed > max(50.0, speed.quantile(0.999) if len(speed) else 50.0)).sum()),
        "coordinate_unit": str(table["coordinate_unit"].iloc[0]) if len(table) else "unknown",
        "is_metric": str(table["coordinate_unit"].iloc[0]) == "meter" if len(table) else False,
        "agent_type_distribution": table["agent_type"].value_counts().to_dict(),
    }


def simple_bar(path: Path, labels, values, title: str, ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(6, len(labels) * 1.4), 4))
    plt.bar(labels, values, color="#4C78A8")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def histogram(path: Path, series_by_dataset: dict, title: str, xlabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    for name, values in series_by_dataset.items():
        if len(values):
            plt.hist(values, bins=40, alpha=0.45, label=name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="all_available")
    args = parser.parse_args()
    datasets = available_stage5b_datasets() if args.datasets == "all_available" else [x.strip() for x in args.datasets.split(",") if x.strip()]
    summaries = [summarize(dataset) for dataset in datasets]
    out = Path("outputs/reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_quality_audit_stage5b.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    lines = ["# Stage 5B Data Quality Audit", "", "| dataset | rows | tracks | frames | mean_track | dt_median | speed_p95 | accel_p95 | coordinate_unit | missing_gaps |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |"]
    for row in summaries:
        lines.append(
            f"| {row['dataset_name']} | {row['total_rows']} | {row['total_tracks']} | {row['total_frames']} | "
            f"{row['mean_track_length']} | {row['dt_median']} | {row['speed_p95']} | {row['accel_p95']} | {row['coordinate_unit']} | {row['missing_frame_gap_count']} |"
        )
    (out / "data_quality_audit_stage5b.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig = Path("outputs/figures/stage5b/data_quality")
    simple_bar(fig / "dataset_size_bar.png", [r["dataset_name"] for r in summaries], [r["total_rows"] for r in summaries], "Stage 5B converted rows", "rows")
    simple_bar(fig / "t100_samples_by_dataset.png", [r["dataset_name"] for r in summaries], [json.loads((Path("outputs/reports") / f"stage5b_episode_summary_{r['dataset_name']}.json").read_text()).get("samples_t100", 0) for r in summaries], "Actual t+100 candidate samples", "samples")
    simple_bar(fig / "coordinate_unit_by_dataset.png", [r["dataset_name"] for r in summaries], [1 if r["is_metric"] else 0 for r in summaries], "Metric coordinate availability", "metric=1")
    simple_bar(fig / "missing_frame_gaps.png", [r["dataset_name"] for r in summaries], [r["missing_frame_gap_count"] for r in summaries], "Missing frame gaps", "count")

    tracks = {}
    speeds = {}
    accels = {}
    for dataset in datasets:
        table = read_world_state(dataset)
        tracks[dataset] = table.groupby(["scene_id", "agent_id"])["frame_id"].nunique().to_numpy()
        speeds[dataset] = table["speed"].clip(upper=table["speed"].quantile(0.99)).to_numpy()
        accels[dataset] = table["acceleration_norm"].clip(upper=table["acceleration_norm"].quantile(0.99)).to_numpy()
    histogram(fig / "track_length_histogram.png", tracks, "Track length distribution", "frames")
    histogram(fig / "speed_distribution_by_dataset.png", speeds, "Speed distribution by dataset", "source units / s")
    histogram(fig / "acceleration_distribution_by_dataset.png", accels, "Acceleration distribution by dataset", "source units / s^2")
    type_counts = {r["dataset_name"]: sum(r["agent_type_distribution"].values()) for r in summaries}
    simple_bar(fig / "agent_type_distribution_by_dataset.png", list(type_counts), list(type_counts.values()), "Agent rows by dataset", "rows")
    print(json.dumps({"summaries": summaries, "figures": str(fig)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
