from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from src.data_discovery.download_manager import plan_downloads, write_download_plan
from src.data_discovery.search_public_datasets import run_discovery


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 5-Data public dataset discovery.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    payload = run_discovery()
    plans = plan_downloads(payload["records"], dry_run=True)
    write_download_plan(plans)
    write_discovery_report(payload)
    write_placeholder_figures(payload["records"])
    print(f"Discovered {payload['candidate_count']} candidate data sources.")
    print("Dry-run only: no gated or large dataset was downloaded.")
    return 0


def write_discovery_report(payload: dict) -> None:
    rows = payload["records"]
    report = f"""# Stage 5 Data Discovery Report

## Required Current-State Admission

The current model is a pseudo-3D / 2.5D physics-informed learned residual state-space world model. It is not true 3D and not an exceptional world model. Stage 4.5 score is 64/100, verdict is `prototype_with_repaired_baselines_but_failed_learned_dynamics_gate`, and passed gates are 4/8. TGSIM t+100 is verified; official Stage 4.5 benchmark uses causal_fd velocity; strongest causal baseline is constant_turn_rate_velocity with FDE@100 about 0.04820m; best learned residual FDE@100 is about 1.14498m. Learned residual did not beat the strongest causal baseline, and SMC remains premature.

## Discovery Summary

- Candidate data sources: {payload['candidate_count']}
- Downloaded or legally downloadable according to registry: {payload['downloadable_count']}
- Gated / requires application: {payload['gated_count']}
- Sources that likely support t+100: {payload['t100_count']}

This is a registry and dry-run discovery stage. Gated datasets are not downloaded, and registry-only datasets are not counted as trained or converted.

## Official Source Notes

- Stanford Drone Dataset: official Stanford UAV data page, non-commercial CC BY-NC-SA 3.0.
- TGSIM: official data.gov / data.transportation.gov public resource.
- Argoverse / Waymo / nuScenes / INTERACTION / leveLX datasets: strong candidates but require terms, application, or large downloads.
- OpenDD: useful map-aware traffic data, but CC BY-ND requires caution around derivative redistribution.

## Top Priority Records

{markdown_top(rows)}
"""
    Path("outputs/reports/data_discovery_report_stage5.md").write_text(report, encoding="utf-8")
    Path("outputs/reports/report_stage5_data_discovery.md").write_text(report, encoding="utf-8")


def markdown_top(rows):
    selected = sorted(rows, key=lambda row: -int(row["priority_score"]))[:12]
    lines = ["| dataset | domain | status | license | priority | loader |", "| --- | --- | --- | --- | --- | --- |"]
    for row in selected:
        lines.append(f"| {row['dataset_name']} | {row['domain']} | {row['download_status']} | {row['license']} | {row['priority_score']} | {row['loader_status']} |")
    return "\n".join(lines)


def write_placeholder_figures(rows):
    out = Path("outputs/figures/stage5")
    out.mkdir(parents=True, exist_ok=True)
    names = [
        "dataset_registry_summary.png",
        "data_lake_size_by_dataset.png",
        "t100_samples_by_dataset.png",
        "agent_type_distribution_by_dataset.png",
        "speed_distribution_by_dataset.png",
        "track_length_distribution.png",
        "domain_gap_heatmap.png",
        "baseline_vs_learned_by_dataset.png",
        "ADE_FDE_curves_by_horizon.png",
        "physical_validity_by_dataset.png",
        "cross_dataset_transfer_matrix.png",
        "rollout_examples_real.png",
        "rollout_examples_synthetic.png",
        "failure_cases_top20.png",
        "maps_and_scene_geometry_examples.png",
    ]
    for name in names:
        image = Image.new("RGB", (1100, 620), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 20, 1080, 600), outline="black", width=2)
        draw.text((50, 50), name.replace("_", " ").replace(".png", ""), fill="black")
        draw.text((50, 95), "Stage 5-Data dry-run figure. Full plot requires downloaded/converted datasets.", fill="black")
        draw.text((50, 140), f"Registered datasets: {len(rows)}", fill="black")
        image.save(out / name)


if __name__ == "__main__":
    raise SystemExit(main())
