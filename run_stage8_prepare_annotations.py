#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

from src.annotation.scene_annotation_tool import prepare_annotations_from_multiagent_episodes, prepare_annotations_from_scene_packs


if __name__ == "__main__":
    records = prepare_annotations_from_scene_packs()
    records += prepare_annotations_from_multiagent_episodes()
    out = Path("outputs/reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage8_annotation_prepare_report.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    rows = [
        {"dataset": r["dataset_name"], "scene": r["scene_id"], "quality": r["annotation_quality"], "goals": len(r["goal_regions"])}
        for r in records
    ]
    lines = ["# Stage 8 Annotation Prepare Report", "", "| dataset | scene | quality | goals |", "| --- | --- | --- | --- |"]
    lines += [f"| {r['dataset']} | {r['scene']} | {r['quality']} | {r['goals']} |" for r in rows]
    out.joinpath("stage8_annotation_prepare_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"annotations": len(records)}, indent=2))
