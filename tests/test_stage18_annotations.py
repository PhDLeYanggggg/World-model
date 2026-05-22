from __future__ import annotations

from src.stage14_pipeline import read_json
from src.stage18_pipeline import auto_annotate


def test_stage18_auto_annotation_never_creates_human_gold():
    report = auto_annotate(quick=True)
    assert report["gold_human_count"] == 0
    annotations = list(__import__("pathlib").Path("data/stage18_annotations").glob("*.json"))
    assert annotations
    for path in annotations[:5]:
        ann = read_json(path, {})
        assert ann["annotation_quality"] != "gold_human"
        assert ann["gold_human"] is False
        assert ann["self_checks"]["test_endpoints_used_for_goal_construction"] is False

