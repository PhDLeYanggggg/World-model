from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.annotation.stage8p5_annotation_schema import make_stage8p5_annotation
from src.annotation.stage8p5_auto_suggestions import suggestions_from_world_state, write_preview_png
from src.annotation.stage8p5_export import write_stage8p5_annotation


def prepare_stage8p5_annotations(world_root: str | Path = "data/stage8p5_world_state") -> Dict:
    records = []
    for csv_path in sorted(Path(world_root).glob("*/world_state.csv")):
        dataset = csv_path.parent.name
        df = pd.read_csv(csv_path)
        for scene_id, scene_df in df.groupby("scene_id"):
            suggestion = suggestions_from_world_state(dataset, str(scene_id), scene_df)
            preview = write_preview_png(dataset, str(scene_id), scene_df, suggestion)
            annotation = make_stage8p5_annotation(
                dataset_name=dataset,
                scene_id=str(scene_id),
                coordinate_unit=suggestion["coordinate_unit"],
                boundary_polygon=suggestion["boundary_polygon"],
                goal_regions=suggestion["goal_regions"],
                annotation_quality=suggestion["annotation_quality"],
                notes=(
                    f"{suggestion['rule_confirmation_reason']}. "
                    "Silver here means high-confidence rule-confirmed train-only endpoint regions, not true human-labelled goals."
                ),
            )
            path = write_stage8p5_annotation(annotation)
            records.append(
                {
                    "dataset_name": dataset,
                    "scene_id": str(scene_id),
                    "annotation_quality": annotation["annotation_quality"],
                    "goal_count": len(annotation["goal_regions"]),
                    "train_endpoint_count": suggestion["train_endpoint_count"],
                    "preview_image": preview,
                    "path": str(path),
                    "test_endpoints_used": False,
                }
            )
    return {"stage": "8.5", "annotations": records}


def write_annotation_report(payload: Dict) -> None:
    out = Path("outputs/reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage8p5_annotation_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = payload["annotations"]
    keys = ["dataset_name", "scene_id", "annotation_quality", "goal_count", "train_endpoint_count", "preview_image"]
    lines = ["# Stage 8.5 Annotation Report", "", "| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    lines += [
        "",
        f"gold scenes: {sum(r['annotation_quality'] == 'gold' for r in rows)}",
        f"silver scenes: {sum(r['annotation_quality'] == 'silver' for r in rows)}",
        f"inferred-only scenes: {sum(r['annotation_quality'] == 'inferred_only' for r in rows)}",
        "Silver annotations are rule-confirmed from train split endpoints only; they are not true human-labelled goals.",
    ]
    (out / "stage8p5_annotation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
