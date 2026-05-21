from __future__ import annotations

from src.annotation.stage8p5_annotation_tool import prepare_stage8p5_annotations, write_annotation_report


if __name__ == "__main__":
    payload = prepare_stage8p5_annotations()
    write_annotation_report(payload)
    print(f"annotations={len(payload['annotations'])}")
