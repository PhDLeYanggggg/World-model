from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.data.synthetic_physical_crowd import save_episode, simulate_episode
from src.physics.scene_geometry import make_scene_templates, scene_from_dict, scene_to_dict


def generate_dataset(cfg: Dict, quick: bool = True, force: bool = False) -> List[Dict]:
    root = Path(cfg["synthetic_dir"])
    manifest_path = root / "manifest_stage2.json"
    if manifest_path.exists() and not force:
        return load_dataset(root)

    rng = np.random.default_rng(int(cfg["seed"]))
    scenes = make_scene_templates()
    counts = {
        "train": int(cfg["dataset"]["quick_train_episodes" if quick else "train_episodes"]),
        "val": int(cfg["dataset"]["quick_val_episodes" if quick else "val_episodes"]),
        "test": int(cfg["dataset"]["quick_test_episodes" if quick else "test_episodes"]),
    }
    manifest = {"name": "SyntheticPhysicalCrowd2.5D", "quick": quick, "episodes": []}
    episodes: List[Dict] = []
    episode_id = 0
    for split, count in counts.items():
        for index in range(count):
            scene = scenes[index % len(scenes)]
            ep = simulate_episode(scene, episode_id, split, cfg, rng)
            filename = f"{split}/episode_{episode_id:04d}.npz"
            save_episode(root / filename, ep)
            item = {**ep["meta"], "npz": filename, "scene": scene_to_dict(scene)}
            manifest["episodes"].append(item)
            episodes.append({"meta": ep["meta"], "scene": scene, "states": ep["states"]})
            episode_id += 1
    root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return episodes


def load_dataset(root: str | Path) -> List[Dict]:
    root = Path(root)
    manifest = json.loads((root / "manifest_stage2.json").read_text(encoding="utf-8"))
    episodes = []
    for item in manifest["episodes"]:
        arrays = np.load(root / item["npz"])
        meta = json.loads(str(arrays["meta"]))
        episodes.append({"meta": meta, "scene": scene_from_dict(item["scene"]), "states": arrays["states"]})
    return episodes


def split_episodes(episodes: List[Dict], split: str) -> List[Dict]:
    return [episode for episode in episodes if episode["meta"]["split"] == split]


def dataset_summary(episodes: List[Dict], root: str | Path) -> Dict:
    out = {
        "episodes": len(episodes),
        "frames_min": int(min(e["meta"]["frames"] for e in episodes)),
        "frames_max": int(max(e["meta"]["frames"] for e in episodes)),
        "agents_min": int(min(e["meta"]["agents"] for e in episodes)),
        "agents_max": int(max(e["meta"]["agents"] for e in episodes)),
        "storage": str(root),
    }
    for split in ["train", "val", "test"]:
        subset = split_episodes(episodes, split)
        out[f"{split}_episodes"] = len(subset)
        out[f"{split}_agents_mean"] = round(float(np.mean([e["meta"]["agents"] for e in subset])), 2) if subset else 0
    return out
