from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


REPORT_DIR = Path("outputs/reports")
STAGE10_RESULTS_DIR = Path("outputs/world_model_stage10_results")
PEDESTRIAN_DATASETS = {"trajnet", "eth_ucy", "sdd", "opentraj", "full_trajnet", "full_eth_ucy", "aerialmpt_long", "ucy"}


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path, default):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def write_json(path: str | Path, payload) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown_table(path: str | Path, title: str, rows: List[Dict], extra_lines: Iterable[str] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if rows:
        keys = list(rows[0].keys())
        lines += ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    else:
        lines.append("No records.")
    if extra_lines:
        lines += ["", *extra_lines]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def stage10_quality_from_previous(value: str) -> str:
    value = str(value or "inferred_only")
    if value in {"gold_human", "silver_human_confirmed", "silver_rule_confirmed", "inferred_only"}:
        return value
    if value == "silver":
        return "silver_rule_confirmed"
    if value == "gold":
        return "silver_rule_confirmed"
    return "inferred_only"


def is_official_annotation_quality(quality: str) -> bool:
    return quality in {"gold_human", "silver_human_confirmed", "silver_rule_confirmed"}


def is_human_annotation_quality(quality: str) -> bool:
    return quality in {"gold_human", "silver_human_confirmed"}


def available_world_state_sources(root: str | Path = "data/stage8p5_world_state") -> List[str]:
    base = Path(root)
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir() and (p / "world_state.csv").exists())


def copy_report_to_stage10_package(path: str | Path) -> None:
    src = Path(path)
    if not src.exists():
        return
    dst = STAGE10_RESULTS_DIR / "reports" / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
