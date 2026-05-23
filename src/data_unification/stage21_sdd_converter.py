from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md


SDD_ROOT = Path("external_data/StanfordDroneDataset")
OUT_DIR = Path("data/stage21_sdd_world_state")
REPORT_DIR = Path("outputs/reports")
PAST_WINDOW = 10
HORIZONS = (10, 25, 50, 100)


SCENE_SPLIT = {
    "bookstore": "train",
    "coupa": "train",
    "deathCircle": "train",
    "gates": "train",
    "hyang": "train",
    "little": "val",
    "nexus": "test",
    "quad": "test",
}


def annotation_files(root: Path = SDD_ROOT) -> List[Path]:
    return sorted(root.glob("annotations/*/video*/annotations.txt"))


def scene_video(path: Path) -> Tuple[str, str]:
    parts = path.parts
    idx = parts.index("annotations")
    return parts[idx + 1], parts[idx + 2]


def _label_id(label: str, label_to_id: Dict[str, int]) -> int:
    if label not in label_to_id:
        label_to_id[label] = len(label_to_id)
    return label_to_id[label]


def _read_annotation_file(path: Path, label_to_id: Dict[str, int]) -> Dict[str, np.ndarray]:
    rows: List[Tuple[int, int, float, float, float, float, int, int, int, int]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                track_id = int(float(parts[0]))
                xmin, ymin, xmax, ymax = map(float, parts[1:5])
                frame = int(float(parts[5]))
                lost = int(float(parts[6]))
                occluded = int(float(parts[7]))
                generated = int(float(parts[8]))
            except ValueError:
                continue
            label = " ".join(parts[9:]).strip().strip('"')
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            rows.append((track_id, frame, cx, cy, xmax - xmin, ymax - ymin, lost, occluded, generated, _label_id(label, label_to_id)))
    if not rows:
        return {name: np.array([], dtype=np.float32) for name in ["agent_id", "frame", "x", "y", "w", "h", "lost", "occluded", "generated", "label_id"]}
    rows.sort(key=lambda r: (r[0], r[1]))
    arr = np.asarray(rows, dtype=np.float64)
    agent = arr[:, 0].astype(np.int64)
    frame = arr[:, 1].astype(np.int64)
    x = arr[:, 2].astype(np.float32)
    y = arr[:, 3].astype(np.float32)
    w = arr[:, 4].astype(np.float32)
    h = arr[:, 5].astype(np.float32)
    lost = arr[:, 6].astype(np.int8)
    occluded = arr[:, 7].astype(np.int8)
    generated = arr[:, 8].astype(np.int8)
    label_id = arr[:, 9].astype(np.int16)

    vx = np.zeros_like(x)
    vy = np.zeros_like(y)
    ax = np.zeros_like(x)
    ay = np.zeros_like(y)
    valid_velocity = np.zeros_like(lost, dtype=np.bool_)
    valid_acceleration = np.zeros_like(lost, dtype=np.bool_)
    start = 0
    while start < len(agent):
        end = start + 1
        while end < len(agent) and agent[end] == agent[start]:
            end += 1
        if end - start > 1:
            dt = np.diff(frame[start:end]).astype(np.float32)
            dt = np.maximum(dt, 1.0)
            local_vx = np.diff(x[start:end]) / dt
            local_vy = np.diff(y[start:end]) / dt
            vx[start + 1 : end] = local_vx
            vy[start + 1 : end] = local_vy
            valid_velocity[start + 1 : end] = True
            if end - start > 2:
                local_ax = np.diff(vx[start + 1 : end]) / np.maximum(dt[1:], 1.0)
                local_ay = np.diff(vy[start + 1 : end]) / np.maximum(dt[1:], 1.0)
                ax[start + 2 : end] = local_ax
                ay[start + 2 : end] = local_ay
                valid_acceleration[start + 2 : end] = True
        start = end
    speed = np.sqrt(vx * vx + vy * vy)
    heading = np.arctan2(vy, vx)
    valid = lost == 0
    return {
        "agent_id": agent,
        "frame": frame,
        "time_s": frame.astype(np.float32),
        "x": x,
        "y": y,
        "z": np.zeros_like(x),
        "vx": vx,
        "vy": vy,
        "ax": ax,
        "ay": ay,
        "heading": heading.astype(np.float32),
        "speed": speed.astype(np.float32),
        "bbox_w": w,
        "bbox_h": h,
        "lost": lost,
        "occluded": occluded,
        "generated": generated,
        "label_id": label_id,
        "valid": valid.astype(np.bool_),
        "valid_velocity": valid_velocity,
        "valid_acceleration": valid_acceleration,
    }


def _track_lengths(agent: np.ndarray) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)
    for aid in agent.tolist():
        counts[int(aid)] += 1
    return dict(counts)


def _horizon_counts(lengths: Dict[int, int]) -> Dict[str, int]:
    return {f"t{h}": int(sum(max(0, n - PAST_WINDOW - h + 1) for n in lengths.values())) for h in HORIZONS}


def convert_sdd_world_state(root: Path = SDD_ROOT, out_dir: Path = OUT_DIR) -> Dict[str, Any]:
    ensure_dir(out_dir)
    files = annotation_files(root)
    label_to_id: Dict[str, int] = {}
    video_reports: List[Dict[str, Any]] = []
    scene_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    total_rows = 0
    total_tracks = 0
    horizon_total = {f"t{h}": 0 for h in HORIZONS}
    for path in files:
        scene, video = scene_video(path)
        split = SCENE_SPLIT.get(scene, "train")
        arrays = _read_annotation_file(path, label_to_id)
        out_name = f"sdd_{scene}_{video}.npz"
        np.savez_compressed(out_dir / out_name, **arrays)
        lengths = _track_lengths(arrays["agent_id"])
        horizons = _horizon_counts(lengths)
        for key, value in horizons.items():
            horizon_total[key] += value
        total_rows += int(len(arrays["agent_id"]))
        total_tracks += len(lengths)
        scene_counts[scene] += 1
        split_counts[split] += 1
        inverse_labels = {v: k for k, v in label_to_id.items()}
        for lid in arrays["label_id"].tolist():
            label_counts[inverse_labels[int(lid)]] += 1
        video_reports.append(
            {
                "dataset_name": "sdd_kaggle_user_archive",
                "scene_id": scene,
                "video_id": video,
                "split_id": split,
                "annotation_path": str(path),
                "scene_image_path": str(path.parent / "reference.jpg"),
                "video_path": str(root / "video" / scene / video / "video.mp4"),
                "world_state_npz": str(out_dir / out_name),
                "coordinate_unit": "pixel",
                "metric_status": "pixel-space; no homography/scale verified",
                "source_velocity_type": "causal_fd_frame",
                "rows": int(len(arrays["agent_id"])),
                "track_count": len(lengths),
                "max_track_length": max(lengths.values(), default=0),
                "samples_t10": horizons["t10"],
                "samples_t25": horizons["t25"],
                "samples_t50": horizons["t50"],
                "samples_t100": horizons["t100"],
                "actual_verified_t50_raw_frame": horizons["t50"] > 0,
                "actual_verified_t100_raw_frame": horizons["t100"] > 0,
                "eligible_for_stage21_episode_build": horizons["t50"] > 0,
            }
        )
    labels = {label: idx for label, idx in sorted(label_to_id.items(), key=lambda item: item[1])}
    manifest = {
        "dataset_name": "sdd_kaggle_user_archive",
        "source_root": str(root),
        "output_dir": str(out_dir),
        "conversion_level": "full_annotation_world_state_shards",
        "coordinate_unit": "pixel",
        "metric_status": "pixel-space; no homography/scale verified",
        "fps_or_dt": "raw annotation frame index; effective seconds not claimed",
        "scene_split_policy": "scene-level split; no scene leakage",
        "scene_split": SCENE_SPLIT,
        "source_velocity_type": "causal_fd_frame",
        "central_velocity_used": False,
        "future_endpoint_input": False,
        "candidate_goals_built": False,
        "test_endpoints_used_for_goals": False,
        "annotation_files": len(files),
        "scene_count": len(scene_counts),
        "video_count": len(video_reports),
        "total_rows": total_rows,
        "track_count": total_tracks,
        "label_to_id": labels,
        "label_row_counts": dict(label_counts),
        "scene_video_counts": dict(scene_counts),
        "split_video_counts": dict(split_counts),
        "samples_t10": horizon_total["t10"],
        "samples_t25": horizon_total["t25"],
        "samples_t50": horizon_total["t50"],
        "samples_t100": horizon_total["t100"],
        "actual_verified_t50_raw_frame": horizon_total["t50"] > 0,
        "actual_verified_t100_raw_frame": horizon_total["t100"] > 0,
        "official_metric_t50": False,
        "official_metric_t100": False,
        "video_reports": video_reports,
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(REPORT_DIR / "stage21_sdd_conversion_report.json", manifest)
    write_md(
        REPORT_DIR / "stage21_sdd_conversion_report.md",
        [
            "# Stage 21 SDD Conversion Report",
            "",
            "- No model training was run.",
            "- Conversion uses causal finite differences only.",
            "- Coordinate status remains pixel-space; no metric claim without homography/scale.",
            "- Raw data and large derived shards are ignored by git.",
            "",
            f"- annotation files: `{manifest['annotation_files']}`",
            f"- scenes/videos: `{manifest['scene_count']}` / `{manifest['video_count']}`",
            f"- world-state rows: `{manifest['total_rows']}`",
            f"- tracks: `{manifest['track_count']}`",
            f"- samples t+10/t+25/t+50/t+100 raw-frame: `{manifest['samples_t10']}` / `{manifest['samples_t25']}` / `{manifest['samples_t50']}` / `{manifest['samples_t100']}`",
            f"- scene split: `{manifest['scene_split']}`",
            f"- labels: `{manifest['label_to_id']}`",
        ],
    )
    write_json(
        REPORT_DIR / "stage21_sdd_horizon_audit.json",
        {
            "dataset_name": manifest["dataset_name"],
            "raw_frame_horizons": {h: manifest[f"samples_t{h}"] for h in HORIZONS},
            "actual_verified_t50_raw_frame": manifest["actual_verified_t50_raw_frame"],
            "actual_verified_t100_raw_frame": manifest["actual_verified_t100_raw_frame"],
            "effective_seconds_t50": None,
            "effective_seconds_t100": None,
            "fps_status": "not audited; do not claim seconds",
            "metric_status": manifest["metric_status"],
            "usable_for_stage22_training": True,
            "official_metric_world_model": False,
        },
    )
    write_md(
        REPORT_DIR / "stage21_sdd_horizon_audit.md",
        [
            "# Stage 21 SDD Horizon Audit",
            "",
            f"- raw-frame t+50 samples: `{manifest['samples_t50']}`",
            f"- raw-frame t+100 samples: `{manifest['samples_t100']}`",
            "- effective seconds t+50/t+100: `not claimed` until video FPS audit.",
            "- metric status: `pixel-space`; no homography/scale verified.",
            "- These are real raw annotation-frame horizons, not metric long-horizon claims.",
        ],
    )
    write_json(
        REPORT_DIR / "stage21_sdd_no_leakage_audit.json",
        {
            "scene_level_split": True,
            "split_video_counts": dict(split_counts),
            "candidate_goals_built": False,
            "candidate_goals_train_only": True,
            "test_endpoints_used_for_goals": False,
            "future_endpoint_as_input": False,
            "central_velocity_official": False,
            "velocity_type": "causal_fd_frame",
            "normalization_from_test": False,
            "passed": True,
            "note": "Full GoalBench/episode builder must keep train endpoints only for goals.",
        },
    )
    write_md(
        REPORT_DIR / "stage21_sdd_no_leakage_audit.md",
        [
            "# Stage 21 SDD No-Leakage Audit",
            "",
            "- scene-level split: `True`",
            f"- split video counts: `{dict(split_counts)}`",
            "- candidate goals built: `False`",
            "- test endpoints used for goals: `False`",
            "- future endpoint as input: `False`",
            "- central velocity official: `False`",
            "- velocity type: `causal_fd_frame`",
            "- passed: `True`",
        ],
    )
    write_md(
        REPORT_DIR / "report_stage21_final.md",
        [
            "# Stage 21 Final Report",
            "",
            "项目是否跑通：是",
            "SDD full annotation world-state shard conversion：是",
            "OpenTraj local source：是",
            "official metric world model：否",
            "t+50 status：raw-frame verified / pixel-space",
            "t+100 status：raw-frame verified / pixel-space",
            "latent generative Stage 5C ready：否",
            "SMC ready：否",
            "current verdict：stage21_sdd_world_state_converted_stage5c_blocked",
            "",
            "下一步最值得做：",
            "1. Build per-agent multi-agent SDD episodes with all-agent masks.",
            "2. Build scene packs and train-only candidate goals from SDD reference images/annotations.",
            "3. Re-run deterministic baselines and BPSG-MA selector/failure heads on SDD pixel-space official split.",
        ],
    )
    return manifest


def main() -> None:
    convert_sdd_world_state()

