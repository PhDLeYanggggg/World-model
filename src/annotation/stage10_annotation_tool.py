from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.annotation.stage10_annotation_schema import make_stage10_annotation
from src.annotation.stage10_auto_suggestions import build_annotation_task, write_annotation_task
from src.stage10_common import REPORT_DIR, ensure_dir, write_json, write_markdown_table


def prepare_stage10_annotation_tasks(previous_root: str | Path = "data/stage8p5_annotations") -> Dict:
    rows: List[Dict] = []
    for path in sorted(Path(previous_root).glob("*/*/scene_annotation.json")):
        previous = json.loads(path.read_text(encoding="utf-8"))
        ann = make_stage10_annotation(previous)
        ann_path = write_stage10_annotation(ann)
        preview = f"outputs/figures/stage8p5_annotation_previews/{ann['dataset_name']}_{ann['scene_id']}.png"
        task = build_annotation_task(ann, ann_path, preview if Path(preview).exists() else "")
        task_path = write_annotation_task(task)
        rows.append(
            {
                "dataset_name": ann["dataset_name"],
                "scene_id": ann["scene_id"],
                "annotation_quality": ann["annotation_quality"],
                "goal_count": len(ann.get("goal_regions", [])),
                "requires_human_review": ann["requires_human_review"],
                "task_path": str(task_path),
                "annotation_path": str(ann_path),
                "test_endpoints_used": False,
            }
        )
    payload = {"stage": "10", "annotations": rows}
    write_annotation_report(payload)
    return payload


def write_stage10_annotation(annotation: Dict, out_root: str | Path = "data/stage10_annotations") -> Path:
    out_dir = ensure_dir(Path(out_root) / annotation["dataset_name"] / annotation["scene_id"])
    path = out_dir / "scene_annotation.json"
    path.write_text(json.dumps(annotation, indent=2), encoding="utf-8")
    return path


def write_annotation_report(payload: Dict) -> None:
    write_json(REPORT_DIR / "stage10_annotation_report.json", payload)
    rows = payload["annotations"]
    table_rows = [
        {
            "dataset_name": r["dataset_name"],
            "scene_id": r["scene_id"],
            "annotation_quality": r["annotation_quality"],
            "goal_count": r["goal_count"],
            "requires_human_review": r["requires_human_review"],
        }
        for r in rows
    ]
    extra = [
        f"gold_human scenes: {sum(r['annotation_quality'] == 'gold_human' for r in rows)}",
        f"silver_human_confirmed scenes: {sum(r['annotation_quality'] == 'silver_human_confirmed' for r in rows)}",
        f"silver_rule_confirmed scenes: {sum(r['annotation_quality'] == 'silver_rule_confirmed' for r in rows)}",
        f"inferred_only scenes: {sum(r['annotation_quality'] == 'inferred_only' for r in rows)}",
        "Annotation tasks are not counted as human-confirmed labels until reviewed.",
    ]
    write_markdown_table(REPORT_DIR / "stage10_annotation_report.md", "Stage 10 Annotation Task Report", table_rows, extra)


def main() -> None:
    payload = prepare_stage10_annotation_tasks()
    print(json.dumps({"annotations": len(payload["annotations"]), "human_confirmed": sum(r["annotation_quality"] in {"gold_human", "silver_human_confirmed"} for r in payload["annotations"])}, indent=2))


if __name__ == "__main__":
    main()
