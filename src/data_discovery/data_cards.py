from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List


def write_data_cards(rows: Iterable[Dict], output_dir: str | Path = "outputs/reports/stage5_data/data_cards") -> List[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for row in rows:
        path = out / f"{slug(row['dataset_name'])}.md"
        text = f"""# Data Card: {row['dataset_name']}

## Access

- Domain: {row['domain']}
- Official URL: {row['official_url']}
- Download status: {row['download_status']}
- License: {row['license']}
- Commercial use allowed: {row['commercial_use_allowed']}
- Redistribution allowed: {row['redistribution_allowed']}
- Citation required: {row['citation_required']}

## World-State Value

- Trajectories: {row['has_trajectories']}
- Metric coordinates: {row['has_metric_coordinates']}
- Scene map / geometry: {row['has_scene_map']} / {row['has_obstacle_geometry']} / {row['has_walkable_area']}
- Agent type: {row['has_agent_type']}
- Heading / velocity / acceleration: {row['has_heading']} / {row['has_velocity']} / {row['has_acceleration']}
- Can evaluate t+100: {row['can_evaluate_t100']}

## Use In This Project

- Loader status: {row['loader_status']}
- Download command: `{row['download_command']}`
- Preprocessing command: `{row['preprocessing_command']}`
- Priority score: {row['priority_score']}
- Priority reason: {row['reason_for_priority']}

## Notes

{row['notes']}
"""
        path.write_text(text, encoding="utf-8")
        paths.append(path)
    return paths


def slug(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")
