#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

from src.annotation.scene_annotation_validator import validate_annotation


def main() -> None:
    records = []
    for path in Path("data/stage8_annotations").glob("*/*/scene_annotation.json"):
        annotation = json.loads(path.read_text(encoding="utf-8"))
        errors = validate_annotation(annotation)
        records.append({"path": str(path), "dataset": annotation.get("dataset_name"), "scene": annotation.get("scene_id"), "quality": annotation.get("annotation_quality"), "valid": not errors, "errors": errors})
    out = Path("outputs/reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage8_annotation_validation.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    lines = ["# Stage 8 Annotation Validation", "", "| dataset | scene | quality | valid | errors |", "| --- | --- | --- | --- | --- |"]
    for r in records:
        lines.append(f"| {r['dataset']} | {r['scene']} | {r['quality']} | {r['valid']} | {r['errors']} |")
    out.joinpath("stage8_annotation_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"annotations": len(records), "valid": sum(r["valid"] for r in records)}, indent=2))


if __name__ == "__main__":
    main()

