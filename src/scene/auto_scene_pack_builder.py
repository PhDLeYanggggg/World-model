from __future__ import annotations

from src.orchestrator.research_state import read_json, write_json, write_md


def write_auto_scene_pack_report() -> dict:
    payload = read_json("outputs/reports/stage12_scene_pack_report.json", default={}) or {}
    result = {"source": "stage12_scene_pack_report", **payload}
    write_json("outputs/reports/auto_scene_pack_report.json", result)
    write_md(
        "outputs/reports/auto_scene_pack_report.md",
        [
            "# Auto Scene Pack Report",
            "",
            f"- source: `{result['source']}`",
            f"- scene_packs_with_goals: `{result.get('scenes_with_goals', result.get('scene_packs_with_goals', 'unknown'))}`",
            "- Scene packs are reused from the latest trusted stage until new annotations are validated.",
        ],
    )
    return result

