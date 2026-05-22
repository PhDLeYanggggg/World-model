from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


REGISTRY_PATHS = [
    Path("outputs/world_model_stage5_data_results/data_registry/dataset_registry_stage5.json"),
    Path("outputs/data_registry/dataset_registry_stage5.json"),
]


CURATED_PEDESTRIAN_DRONE = [
    {
        "dataset_name": "Stanford Drone Dataset",
        "official_url": "https://cvgl.stanford.edu/projects/uav_data/",
        "license": "Non-commercial; user must accept terms",
        "requires_login": False,
        "requires_application": False,
        "can_download_automatically": False,
        "scene_image_available": True,
        "trajectory_annotation_available": True,
        "homography_available": False,
        "metric_coordinates_available": False,
        "t50_possible": True,
        "t100_possible": True,
        "priority_score": 95,
        "legal_notes": "Do not download automatically; ask user for accepted local copy.",
    },
    {
        "dataset_name": "OpenTraj compatible pedestrian datasets",
        "official_url": "https://github.com/crowdbotp/OpenTraj",
        "license": "Mixed per source dataset",
        "requires_login": False,
        "requires_application": False,
        "can_download_automatically": False,
        "scene_image_available": "varies",
        "trajectory_annotation_available": True,
        "homography_available": "varies",
        "metric_coordinates_available": "varies",
        "t50_possible": True,
        "t100_possible": "varies",
        "priority_score": 90,
        "legal_notes": "Use per-dataset official terms; OpenTraj is a loader ecosystem, not a universal license.",
    },
    {
        "dataset_name": "ETH/UCY EWAP",
        "official_url": "https://icu.ee.ethz.ch/research/datsets.html",
        "license": "Academic/citation required; verify redistribution before sharing",
        "requires_login": False,
        "requires_application": False,
        "can_download_automatically": False,
        "scene_image_available": True,
        "trajectory_annotation_available": True,
        "homography_available": True,
        "metric_coordinates_available": True,
        "t50_possible": True,
        "t100_possible": True,
        "priority_score": 88,
        "legal_notes": "Already locally available via Stage 12; keep raw bundle local.",
    },
    {
        "dataset_name": "full TrajNet++ original datasets",
        "official_url": "https://www.aicrowd.com/challenges/trajnet-a-trajectory-forecasting-challenge",
        "license": "Mixed/benchmark terms",
        "requires_login": True,
        "requires_application": False,
        "can_download_automatically": False,
        "scene_image_available": False,
        "trajectory_annotation_available": True,
        "homography_available": "varies",
        "metric_coordinates_available": "varies",
        "t50_possible": "depends on source sequence",
        "t100_possible": "depends on source sequence",
        "priority_score": 82,
        "legal_notes": "Use local user-provided copy or official challenge access.",
    },
]


def load_existing_registry() -> List[Dict]:
    for path in REGISTRY_PATHS:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, list) else payload.get("datasets", [])
    return []


def discover_candidates() -> List[Dict]:
    rows = []
    seen = set()
    for row in CURATED_PEDESTRIAN_DRONE + load_existing_registry():
        name = row.get("dataset_name") or row.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        rows.append(
            {
                "dataset_name": name,
                "official_url": row.get("official_url", ""),
                "license": row.get("license", "unknown"),
                "download_status": row.get("download_status", "discovered"),
                "requires_login": row.get("requires_login", "unknown"),
                "requires_application": row.get("requires_application", "unknown"),
                "can_download_automatically": row.get("can_download_automatically", False),
                "raw_video_available": row.get("has_raw_video", row.get("raw_video_available", "unknown")),
                "scene_image_available": row.get("has_images", row.get("scene_image_available", "unknown")),
                "trajectory_annotation_available": row.get("has_trajectories", row.get("trajectory_annotation_available", "unknown")),
                "homography_available": row.get("has_homography", row.get("homography_available", "unknown")),
                "metric_coordinates_available": row.get("has_metric_coordinates", row.get("metric_coordinates_available", "unknown")),
                "frame_rate": row.get("frame_rate", "unknown"),
                "track_length_estimate": row.get("average_track_length", "unknown"),
                "t50_possible": row.get("estimated_samples_t50", row.get("t50_possible", "unknown")),
                "t100_possible": row.get("can_evaluate_t100", row.get("t100_possible", "unknown")),
                "scene_context_available": row.get("has_scene_map", row.get("scene_context_available", "unknown")),
                "agent_types": row.get("agent_type", row.get("agent_types", "unknown")),
                "priority_score": int(row.get("priority_score", 0) or 0),
                "reason_for_priority": row.get("reason_for_priority", "pedestrian/drone long-horizon or scene context candidate"),
                "legal_notes": row.get("legal_notes", row.get("notes", "")),
                "next_user_action_if_needed": row.get("next_user_action_if_needed", ""),
            }
        )
    rows.sort(key=lambda item: item["priority_score"], reverse=True)
    return rows

