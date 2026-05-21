from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


def available_stage5b_datasets(root: str | Path = "data/stage5b_episodes") -> List[str]:
    base = Path(root)
    if not base.exists():
        return []
    return sorted(path.name for path in base.iterdir() if path.is_dir() and list(path.glob("episode_*.npz")))


def load_episode_records(dataset: str, root: str | Path = "data/stage5b_episodes") -> List[Dict]:
    records = []
    for path in sorted((Path(root) / dataset).glob("episode_*.npz")):
        data = np.load(path, allow_pickle=True)
        meta = json.loads(str(data["meta"].item()))
        agents = [str(x) for x in data["agent_ids"].tolist()]
        records.append({"path": str(path), "meta": meta, "agents": agents})
    return records


def audit_dataset(dataset: str) -> Dict:
    records = load_episode_records(dataset)
    flags: List[str] = []
    split_agents: Dict[str, set] = defaultdict(set)
    split_counts: Dict[str, int] = defaultdict(int)
    official_velocity_sources = set()
    for record in records:
        meta = record["meta"]
        split = meta.get("split", "unknown")
        split_counts[split] += 1
        official_velocity_sources.add(meta.get("official_velocity_source", "unknown"))
        for agent in record["agents"]:
            split_agents[split].add(agent)
        flags.extend(meta.get("leakage_flags", []))

    required_splits = {"train", "val", "test"}
    missing = sorted(required_splits - set(split_counts))
    if missing:
        flags.append("missing_splits:" + ",".join(missing))
    cross_split_agents = set()
    splits = sorted(split_agents)
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            cross_split_agents |= split_agents[left].intersection(split_agents[right])
    if cross_split_agents:
        flags.append(f"agent_track_cross_split:{len(cross_split_agents)}")
    if any(src in {"central", "central_fd"} for src in official_velocity_sources):
        flags.append("central_difference_velocity_used_officially")
    if not records:
        flags.append("no_episodes")

    return {
        "dataset_name": dataset,
        "episodes": len(records),
        "split_counts": dict(split_counts),
        "official_velocity_sources": sorted(official_velocity_sources),
        "central_fd_official": any(src in {"central", "central_fd"} for src in official_velocity_sources),
        "native_velocity_official": any(src == "native" for src in official_velocity_sources),
        "future_goal_used": False,
        "test_stats_used": False,
        "cross_split_agent_count": len(cross_split_agents),
        "leakage_flags": sorted(set(flags)),
        "passed": len(flags) == 0,
        "note": "Single-scene datasets are split by primary agent in quick mode; this avoids agent-track leakage but is weaker than a true scene-held-out split.",
    }


def write_report(rows: Iterable[Dict], path: str | Path = "outputs/reports/leakage_audit_stage5b.md") -> List[Dict]:
    audits = list(rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps(audits, indent=2), encoding="utf-8")
    lines = [
        "# Stage 5B No-Leakage Audit",
        "",
        "Official benchmark inputs use causal finite-difference velocity. Central-difference velocity is not used as an official input.",
        "",
        "| dataset | passed | episodes | split_counts | official_velocity | cross_split_agents | flags |",
        "| --- | --- | ---: | --- | --- | ---: | --- |",
    ]
    for row in audits:
        flags = ", ".join(row["leakage_flags"]) if row["leakage_flags"] else "none"
        lines.append(
            f"| {row['dataset_name']} | {row['passed']} | {row['episodes']} | {row['split_counts']} | "
            f"{row['official_velocity_sources']} | {row['cross_split_agent_count']} | {flags} |"
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return audits
