from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src.annotation.auto_scene_suggestion import suggested_scene_task
from src.orchestrator.research_state import read_json, write_json, write_md


def build_annotation_tasks(limit: int = 10) -> List[Dict]:
    priority = read_json("outputs/reports/stage12_annotation_priority_list.json", default=[]) or []
    if isinstance(priority, dict):
        priority = priority.get("scenes", [])
    tasks = []
    for row in priority[:limit]:
        tasks.append(suggested_scene_task(row.get("scene_id", "unknown"), row.get("dataset_name", "unknown"), row.get("annotation_quality", "inferred_only")))
    return tasks


def write_annotation_task_report(limit: int = 10) -> List[Dict]:
    tasks = build_annotation_tasks(limit=limit)
    write_json("outputs/reports/annotation_task_report.json", {"tasks": tasks})
    lines = ["# Annotation Task Report", "", f"- task_count: `{len(tasks)}`", ""]
    for task in tasks:
        lines.append(f"- {task['dataset_name']} / {task['scene_id']}: `{task['annotation_quality']}`")
    write_md("outputs/reports/annotation_task_report.md", lines)
    return tasks
