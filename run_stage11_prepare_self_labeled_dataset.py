from __future__ import annotations

import json
import shutil
from pathlib import Path


REPORT_DIR = Path("outputs/reports")


def copy_tree(src: Path, dst: Path) -> int:
    if not src.exists():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    for child in src.iterdir():
        target = dst / child.name
        if child.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(child, target)
            count += sum(1 for _ in target.rglob("*") if _.is_file())
        else:
            shutil.copy2(child, target)
            count += 1
    return count


def main() -> None:
    episode_root = Path("data/stage11_multiagent_episodes")
    scene_root = Path("data/stage11_scene_packs")
    episode_root.mkdir(parents=True, exist_ok=True)
    scene_root.mkdir(parents=True, exist_ok=True)
    stage10_episode_files = copy_tree(Path("data/stage10_multiagent_episodes"), episode_root)
    stage10_scene_files = copy_tree(Path("data/stage10_scene_packs"), scene_root)
    # AerialMPT builders write directly into the same Stage 11 roots.
    aerial_episodes = sum(1 for _ in (episode_root / "aerialmpt").glob("episode_*.npz")) if (episode_root / "aerialmpt").exists() else 0
    payload = {
        "stage": "11",
        "episode_root": str(episode_root),
        "scene_pack_root": str(scene_root),
        "copied_stage10_episode_files": stage10_episode_files,
        "copied_stage10_scene_files": stage10_scene_files,
        "aerialmpt_episodes": aerial_episodes,
        "datasets": sorted(p.name for p in episode_root.iterdir() if p.is_dir()),
        "notes": [
            "Stage 11 uses Stage 10 self/silver scene labels plus AerialMPT ai_visual_silver labels when present.",
            "AerialMPT remains pixel-space and short-horizon.",
        ],
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "stage11_self_labeled_dataset_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Stage 11 Self-Labeled Dataset Report",
        "",
        f"Episode root: `{payload['episode_root']}`",
        f"Scene pack root: `{payload['scene_pack_root']}`",
        f"Datasets: `{payload['datasets']}`",
        f"AerialMPT episodes: `{payload['aerialmpt_episodes']}`",
        "",
        "Notes:",
        *[f"- {n}" for n in payload["notes"]],
    ]
    (REPORT_DIR / "stage11_self_labeled_dataset_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
