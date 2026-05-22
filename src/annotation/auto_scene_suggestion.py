from __future__ import annotations

from typing import Dict, List


def suggested_scene_task(scene_id: str, dataset_name: str, quality: str = "inferred_only") -> Dict:
    return {
        "scene_id": scene_id,
        "dataset_name": dataset_name,
        "annotation_quality": quality,
        "suggestions": [
            "walkable area from visible paths/roads/sidewalks",
            "candidate exits/goals from train-split endpoint clusters",
            "boundary polygon from observed trajectory extent",
            "obstacles/no-go zones only when visually obvious",
        ],
        "guardrails": [
            "do not use test endpoints",
            "do not mark inferred suggestions as gold",
            "pixel annotation remains pixel-space unless homography/scale exists",
        ],
    }

