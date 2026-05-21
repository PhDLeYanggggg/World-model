from __future__ import annotations

import json
from pathlib import Path

from src.annotation.stage8p5_annotation_validator import validate_stage8p5_annotation


def main() -> None:
    rows = []
    for path in sorted(Path("data/stage8p5_annotations").glob("*/*/scene_annotation.json")):
        ann = json.loads(path.read_text(encoding="utf-8"))
        errors = validate_stage8p5_annotation(ann)
        rows.append({"path": str(path), "dataset": ann.get("dataset_name"), "scene": ann.get("scene_id"), "quality": ann.get("annotation_quality"), "valid": not errors, "errors": errors})
    out = Path("outputs/reports")
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage8p5_annotation_validation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = ["# Stage 8.5 Annotation Validation", "", "| dataset | scene | quality | valid | errors |", "| --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['scene']} | {r['quality']} | {r['valid']} | {r['errors']} |")
    (out / "stage8p5_annotation_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"annotations": len(rows), "valid": sum(r["valid"] for r in rows)}, indent=2))


if __name__ == "__main__":
    main()
