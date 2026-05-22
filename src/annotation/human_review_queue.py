from __future__ import annotations

from src.annotation.auto_annotation_task_builder import build_annotation_tasks


def build_human_review_queue(limit: int = 10) -> list[dict]:
    queue = []
    for task in build_annotation_tasks(limit=limit):
        item = dict(task)
        item["review_status"] = "needs_human_review"
        item["allowed_upgrade"] = "silver_human_confirmed after human confirmation; gold_human only after direct human annotation + validation"
        queue.append(item)
    return queue

