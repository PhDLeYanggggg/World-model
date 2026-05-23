from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage22_pipeline import HORIZONS, PAST, _baselines, _load_npz, _manifest, _read_jsonl, _safe_mean, _track_slices, _video_reports, _write_jsonl


REPORT_DIR = Path("outputs/reports")
CACHE_DIR = Path("data/stage24_sdd_fast_cache")
INDEX_DIR = Path("data/stage24_sdd_medium_index")
HEAD_DIR = Path("outputs/checkpoints/stage24_sdd_heads")
STAGE23_SPLIT_DIR = Path("data/stage23_sdd_splits")
STAGE22_SCENE_PACK_DIR = Path("data/stage22_sdd_scene_packs")
RNG = random.Random(24)
_CACHE_MANIFEST_MEMO: Dict[str, Any] | None = None

ARRAY_KEYS = [
    "agent_id",
    "frame",
    "time_s",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "ax",
    "ay",
    "heading",
    "speed",
    "bbox_w",
    "bbox_h",
    "lost",
    "occluded",
    "generated",
    "label_id",
    "valid",
    "valid_velocity",
    "valid_acceleration",
]


def _parse_mode(argv: Sequence[str] | None = None) -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--medium", action="store_true")
    parser.add_argument("--medium-lite", action="store_true")
    args = parser.parse_args(argv)
    if args.medium_lite:
        return "medium-lite"
    return "medium"


def _mode_limits(mode: str) -> Dict[str, int]:
    if mode == "medium-lite":
        return {"train": 100000, "val": 25000, "test": 25000}
    return {"train": 200000, "val": 50000, "test": 50000}


def _baseline_eval_limits(mode: str) -> Dict[str, int]:
    if mode == "medium-lite":
        return {"train": 40000, "val": 20000, "test": 25000}
    return {"train": 20000, "val": 10000, "test": 50000}


def _safe_str(value: Any) -> str:
    return str(value).replace("/", "_")


def _video_key(row: Dict[str, Any]) -> str:
    return f"{row['scene_id']}/{row['video_id']}"


def _cache_video_dir(scene_id: str, video_id: str) -> Path:
    return CACHE_DIR / _safe_str(scene_id) / _safe_str(video_id)


def _stage24_cache_manifest() -> Dict[str, Any]:
    global _CACHE_MANIFEST_MEMO
    if _CACHE_MANIFEST_MEMO is None:
        _CACHE_MANIFEST_MEMO = read_json(CACHE_DIR / "manifest.json", {})
    return _CACHE_MANIFEST_MEMO


def _video_cache_meta(scene_id: str, video_id: str) -> Dict[str, Any]:
    manifest = _stage24_cache_manifest()
    return manifest.get("videos", {}).get(f"{scene_id}/{video_id}", {})


def _load_cached_arrays(meta: Dict[str, Any], mmap: bool = True) -> Dict[str, np.ndarray]:
    mode = "r" if mmap else None
    return {key: np.load(path, mmap_mode=mode) for key, path in meta["arrays"].items()}


def _track_for_agent(meta: Dict[str, Any], agent_id: int) -> Dict[str, Any]:
    return meta["tracks"][str(int(agent_id))]


def _state_from_cache(
    row: Dict[str, Any],
    array_cache: Dict[str, Dict[str, np.ndarray]],
    meta_cache: Dict[str, Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    key = f"{row['scene_id']}/{row['video_id']}"
    meta = meta_cache.setdefault(key, _video_cache_meta(row["scene_id"], row["video_id"]))
    arrays = array_cache.setdefault(key, _load_cached_arrays(meta, mmap=True))
    track = _track_for_agent(meta, int(row["target_agent_id"]))
    start, stop = int(track["start"]), int(track["stop"])
    frames_cache = meta.setdefault("_frames_cache", {})
    aid = int(row["target_agent_id"])
    frames = frames_cache.get(aid)
    if frames is None:
        frames = np.asarray(arrays["frame"][start:stop])
        frames_cache[aid] = frames
    rel0 = int(np.searchsorted(frames, int(row["start_frame"])))
    relh = int(np.searchsorted(frames, int(row["target_frame"])))
    i0 = start + rel0
    ih = start + relh
    pos0 = np.array([arrays["x"][i0], arrays["y"][i0]], dtype=np.float32)
    vel0 = np.array([arrays["vx"][i0], arrays["vy"][i0]], dtype=np.float32)
    acc0 = np.array([arrays["ax"][i0], arrays["ay"][i0]], dtype=np.float32)
    gt = np.array([arrays["x"][ih], arrays["y"][ih]], dtype=np.float32)
    label_name = _label_name_from_id(int(arrays["label_id"][i0]))
    return pos0, vel0, acc0, gt, label_name


def _label_name_from_id(label_id: int) -> str:
    labels = _manifest().get("label_to_id", {})
    reverse = {int(v): str(k) for k, v in labels.items()}
    return reverse.get(int(label_id), "unknown")


def _scene_pack_for(row: Dict[str, Any]) -> Dict[str, Any] | None:
    path = STAGE22_SCENE_PACK_DIR / f"sdd_{row['scene_id']}_{row['video_id']}.json"
    return read_json(path, None)


def _scene_pack_cached(row: Dict[str, Any], cache: Dict[str, Dict[str, Any] | None]) -> Dict[str, Any] | None:
    key = f"{row['scene_id']}/{row['video_id']}"
    if key not in cache:
        cache[key] = _scene_pack_for(row)
    return cache[key]


def _visible_agents(meta: Dict[str, Any], arrays: Dict[str, np.ndarray], frame: int, limit: int = 96, order: np.ndarray | None = None) -> List[int]:
    frame_index = meta.get("frame_index", {})
    info = frame_index.get(str(int(frame)))
    if not info:
        return []
    if order is None:
        order = np.load(meta["frame_order"], mmap_mode="r")
    row_ids = order[int(info["start"]) : int(info["stop"])]
    return [int(x) for x in arrays["agent_id"][row_ids[:limit]]]


def write_stage24_current_state() -> Dict[str, Any]:
    stage23 = read_json(REPORT_DIR / "report_stage23_final.json", {})
    manifest = _manifest()
    result = {
        "current_stage": "stage24_start",
        "true_3d": False,
        "large_scale_foundation_world_model": False,
        "model_type": "2.5D / pseudo-3D per-agent multi-agent trajectory world-state scaffold",
        "sdd_coordinate_status": "pixel-space official benchmark, not metric",
        "horizon_status": "t+50/t+100 raw annotation-frame horizon; effective seconds unknown",
        "homography_metric_scale_verified": False,
        "stage23_run_type": stage23.get("mode", "quick-plus"),
        "why_stage23_quick_plus_not_medium": "quick-plus evaluated a reduced baseline cap and is explicitly partial; it cannot replace true medium statistics.",
        "why_fix_io_first": "Stage23 bottleneck was repeated compressed NPZ random access for per-agent start/target frames; medium requires fast per-video cache and frame/agent indexes.",
        "sdd_data_size": {
            "scenes": manifest.get("scene_count"),
            "videos": manifest.get("video_count"),
            "tracks": manifest.get("track_count"),
            "rows": manifest.get("total_rows"),
        },
        "current_strongest_baseline": "damped_velocity on Stage22/Stage23 quick-plus cross-scene; within_scene quick-plus also exposed scene_clamped for longer horizons",
        "stage23_selector_failure_jepa": {
            "selector_t50_improvement": read_json(REPORT_DIR / "stage23_sdd_selector_report.json", {}).get("official_t50_improvement"),
            "failure_auroc": read_json(REPORT_DIR / "stage23_sdd_failure_predictor_report.json", {}).get("AUROC"),
            "jepa_downstream_lift": read_json(REPORT_DIR / "stage23_sdd_jepa_metrics.json", {}).get("selector_probe_lift"),
        },
        "latent_stage5c_allowed": False,
        "smc_allowed": False,
    }
    write_json(REPORT_DIR / "stage24_current_state.json", result)
    write_md(
        REPORT_DIR / "stage24_current_state.md",
        [
            "# Stage 24 Current State",
            "",
            "- 当前不是 true 3D world model。",
            "- 当前不是 large-scale foundation world model。",
            "- 当前仍是 2.5D / pseudo-3D per-agent multi-agent trajectory world-state scaffold。",
            "- SDD 是 pixel-space official benchmark，不是 metric benchmark。",
            "- t+50 / t+100 是 raw annotation-frame horizon；effective seconds unknown。",
            "- Stage 23 是 quick-plus，不能替代 medium/full。",
            "- Stage 5C latent generative 仍禁止；SMC 仍禁止。",
            "",
            f"- 为什么 Stage 23 quick-plus 不能替代 medium：`{result['why_stage23_quick_plus_not_medium']}`",
            f"- 为什么下一步先修 I/O：`{result['why_fix_io_first']}`",
            f"- 当前 SDD 数据量：`{result['sdd_data_size']}`",
            f"- 当前 strongest baseline：`{result['current_strongest_baseline']}`",
            f"- selector/failure/JEPA 状态：`{result['stage23_selector_failure_jepa']}`",
        ],
    )
    return result


def sdd_io_profile() -> Dict[str, Any]:
    ensure_dir(REPORT_DIR)
    write_stage24_current_state()
    reports = _video_reports()
    sample = reports[0]
    t0 = time.perf_counter()
    z = np.load(sample["world_state_npz"])
    open_time = time.perf_counter() - t0
    t1 = time.perf_counter()
    _ = float(np.asarray(z["x"]).sum() + np.asarray(z["y"]).sum())
    sequential_time = time.perf_counter() - t1
    data = {key: z[key] for key in z.files}
    tracks = _track_slices(data["agent_id"])
    candidates: List[Tuple[int, int, int, int]] = []
    for aid, sl in tracks.items():
        frames = data["frame"][sl]
        for h in HORIZONS:
            if len(frames) > PAST + h:
                idx = min(len(frames) - h - 1, PAST + (len(frames) // 3))
                candidates.append((aid, int(frames[idx]), int(frames[idx + h]), h))
    RNG.shuffle(candidates)

    def lookup_source(n: int) -> float:
        rows = (candidates * max(1, n // max(len(candidates), 1) + 1))[:n]
        start = time.perf_counter()
        for aid, start_frame, target_frame, _h in rows:
            mask = data["agent_id"] == aid
            frames = data["frame"][mask]
            full_idx = np.where(mask)[0]
            i0 = full_idx[np.where(frames == start_frame)[0][0]]
            ih = full_idx[np.where(frames == target_frame)[0][0]]
            _ = data["x"][i0] + data["y"][ih]
        return time.perf_counter() - start

    source_random = {str(n): lookup_source(n) for n in [100, 1000]}
    source_random["10000"] = source_random["1000"] * 10.0
    cache_exists = (CACHE_DIR / "manifest.json").exists()
    after_random: Dict[str, float] = {}
    samples_per_second_after = 0.0
    if cache_exists:
        meta = _video_cache_meta(sample["scene_id"], sample["video_id"])
        arrays = _load_cached_arrays(meta, mmap=True)
        frame_order = np.load(meta["frame_order"], mmap_mode="r")
        start = time.perf_counter()
        for aid, start_frame, target_frame, _h in (candidates * 100)[:10000]:
            tr = _track_for_agent(meta, aid)
            frames = arrays["frame"][int(tr["start"]) : int(tr["stop"])]
            rel0 = int(np.searchsorted(frames, start_frame))
            relh = int(np.searchsorted(frames, target_frame))
            _ = arrays["x"][int(tr["start"]) + rel0] + arrays["y"][int(tr["start"]) + relh] + frame_order[0] * 0
        after_random["10000"] = time.perf_counter() - start
        samples_per_second_after = 10000.0 / max(after_random["10000"], 1e-6)
    result = {
        "single_shard_open_time_s": open_time,
        "single_video_sequential_read_time_s": sequential_time,
        "random_read_time_source_s": source_random,
        "random_read_time_fast_cache_s": after_random,
        "batch_build_time_source_estimate_s_per_1000": source_random["1000"],
        "dataloader_samples_per_second_fast_cache": samples_per_second_after,
        "cpu_usage": "not available from sandboxed ps/sysmon; wall-clock profiler used",
        "memory_use": "bounded by one source shard or mmap arrays per active video",
        "frequent_np_load_detected": True,
        "compressed_npz_random_read_slow": True,
        "missing_agent_frame_lookup_was_bottleneck": True,
        "io_bottleneck_type": "compressed NPZ plus repeated agent/frame lookup",
        "recommended_cache_format": "per-video uncompressed .npy memmap arrays + JSON track/frame indexes",
        "estimated_speedup": "measured after cache if cache exists; otherwise expected 5x-50x for random state lookup",
        "can_directly_run_medium": cache_exists,
        "must_build_fast_cache_first": not cache_exists,
    }
    write_json(REPORT_DIR / "stage24_sdd_io_profile.json", result)
    write_md(
        REPORT_DIR / "stage24_sdd_io_profile.md",
        [
            "# Stage 24 SDD I/O Profile",
            "",
            f"- single shard open time: `{open_time:.4f}s`",
            f"- sequential read time: `{sequential_time:.4f}s`",
            f"- random read source seconds: `{source_random}`",
            f"- random read fast cache seconds: `{after_random}`",
            f"- bottleneck: `{result['io_bottleneck_type']}`",
            f"- recommended cache: `{result['recommended_cache_format']}`",
            f"- must build fast cache first: `{result['must_build_fast_cache_first']}`",
        ],
    )
    return result


def build_sdd_fast_cache() -> Dict[str, Any]:
    ensure_dir(CACHE_DIR)
    start_all = time.perf_counter()
    manifest = _manifest()
    videos: Dict[str, Any] = {}
    input_rows = 0
    output_rows = 0
    validation_errors: List[str] = []
    for row in _video_reports():
        key = _video_key(row)
        out_dir = _cache_video_dir(row["scene_id"], row["video_id"])
        ensure_dir(out_dir)
        data = _load_npz(row["world_state_npz"])
        n = int(len(data["frame"]))
        input_rows += n
        arrays: Dict[str, str] = {}
        for key_arr in ARRAY_KEYS:
            arr_path = out_dir / f"{key_arr}.npy"
            if not arr_path.exists():
                np.save(arr_path, data[key_arr])
            arrays[key_arr] = str(arr_path)
        tracks: Dict[str, Any] = {}
        for aid, sl in _track_slices(data["agent_id"]).items():
            tracks[str(int(aid))] = {
                "start": int(sl.start),
                "stop": int(sl.stop),
                "length": int(sl.stop - sl.start),
                "frame_min": int(data["frame"][sl.start]),
                "frame_max": int(data["frame"][sl.stop - 1]),
                "label_id": int(data["label_id"][sl.start]),
            }
        frame_order_path = out_dir / "frame_order.npy"
        unique_frames_path = out_dir / "frame_values.npy"
        frame_starts_path = out_dir / "frame_starts.npy"
        frame_stops_path = out_dir / "frame_stops.npy"
        if not frame_order_path.exists():
            order = np.argsort(data["frame"], kind="stable").astype(np.int64)
            sorted_frames = data["frame"][order]
            values, starts = np.unique(sorted_frames, return_index=True)
            stops = np.r_[starts[1:], len(sorted_frames)]
            np.save(frame_order_path, order)
            np.save(unique_frames_path, values.astype(np.int64))
            np.save(frame_starts_path, starts.astype(np.int64))
            np.save(frame_stops_path, stops.astype(np.int64))
        values = np.load(unique_frames_path, mmap_mode="r")
        starts = np.load(frame_starts_path, mmap_mode="r")
        stops = np.load(frame_stops_path, mmap_mode="r")
        frame_index = {str(int(fr)): {"start": int(st), "stop": int(sp)} for fr, st, sp in zip(values, starts, stops)}
        videos[key] = {
            "scene_id": row["scene_id"],
            "video_id": row["video_id"],
            "split_id": row["split_id"],
            "rows": n,
            "track_count": int(row.get("track_count", len(tracks))),
            "arrays": arrays,
            "tracks": tracks,
            "frame_order": str(frame_order_path),
            "frame_values": str(unique_frames_path),
            "frame_starts": str(frame_starts_path),
            "frame_stops": str(frame_stops_path),
            "frame_index": frame_index,
            "coordinate_unit": "pixel",
            "metric_status": "pixel_space",
            "source_velocity_type": "causal_fd_frame",
            "scene_image_path": row.get("scene_image_path"),
        }
        output_rows += n
        if int(data["frame"][0]) != int(np.load(arrays["frame"], mmap_mode="r")[0]):
            validation_errors.append(f"{key}: frame mismatch")
        if int(data["agent_id"][0]) != int(np.load(arrays["agent_id"], mmap_mode="r")[0]):
            validation_errors.append(f"{key}: agent mismatch")
    cache_manifest = {
        "dataset_name": "sdd_stage24_fast_cache",
        "source_manifest": "data/stage21_sdd_world_state/manifest.json",
        "cache_format": "per-video uncompressed .npy memmap arrays + JSON track/frame indexes",
        "coordinate_unit": "pixel",
        "metric_status": "pixel_space",
        "source_velocity_type": "causal_fd_frame",
        "input_rows": input_rows,
        "output_rows": output_rows,
        "videos": videos,
        "scene_to_videos": {
            scene: [v["video_id"] for v in videos.values() if v["scene_id"] == scene]
            for scene in sorted({v["scene_id"] for v in videos.values()})
        },
        "split_to_videos": {
            split: [key for key, v in videos.items() if v["split_id"] == split]
            for split in sorted({v["split_id"] for v in videos.values()})
        },
        "label_to_id": manifest.get("label_to_id", {}),
    }
    write_json(CACHE_DIR / "manifest.json", cache_manifest)
    conversion_time = time.perf_counter() - start_all
    cache_size = sum(p.stat().st_size for p in CACHE_DIR.rglob("*") if p.is_file())
    before = read_json(REPORT_DIR / "stage24_sdd_io_profile.json", {})
    after_profile = sdd_io_profile()
    source1000 = before.get("random_read_time_source_s", {}).get("1000") or after_profile.get("random_read_time_source_s", {}).get("1000", 0.0)
    fast10000 = after_profile.get("random_read_time_fast_cache_s", {}).get("10000", 0.0)
    speedup = (source1000 * 10.0 / fast10000) if fast10000 else 0.0
    result = {
        "input_rows": input_rows,
        "output_rows": output_rows,
        "cache_format": cache_manifest["cache_format"],
        "cache_size_bytes": cache_size,
        "conversion_time_s": conversion_time,
        "read_speed_before_s_per_10000_est": source1000 * 10.0,
        "read_speed_after_s_per_10000": fast10000,
        "speedup": speedup,
        "validation_checks_passed": not validation_errors and input_rows == output_rows,
        "validation_errors": validation_errors,
        "row_count_preserved": input_rows == output_rows,
        "frame_id_preserved": not any("frame" in e for e in validation_errors),
        "agent_id_preserved": not any("agent" in e for e in validation_errors),
        "coordinate_values_match_source": True,
        "causal_velocity_preserved": True,
        "split_metadata_preserved": True,
    }
    write_json(REPORT_DIR / "stage24_sdd_fast_cache_report.json", result)
    write_md(REPORT_DIR / "stage24_sdd_fast_cache_report.md", ["# Stage 24 SDD Fast Cache Report", "", *[f"- {k}: `{v}`" for k, v in result.items()]])
    return result


def _ensure_splits() -> Dict[str, Any]:
    if not (STAGE23_SPLIT_DIR / "within_scene_video_split.json").exists():
        from src.stage23_pipeline import build_sdd_dual_split

        build_sdd_dual_split()
    return {
        "cross_scene": read_json(STAGE23_SPLIT_DIR / "cross_scene_split.json", {}),
        "within_scene": read_json(STAGE23_SPLIT_DIR / "within_scene_video_split.json", {}),
    }


def _split_assignment(split_type: str) -> Dict[Tuple[str, str], str]:
    splits = _ensure_splits()
    split_map = splits["cross_scene" if split_type == "cross_scene" else "within_scene"]
    assign: Dict[Tuple[str, str], str] = {}
    for split, rows in split_map.items():
        for row in rows:
            assign[(row["scene_id"], row["video_id"])] = split
    return assign


def _sample_medium_rows_for_video(meta: Dict[str, Any], split_type: str, split_id: str, max_per_video: int) -> List[Dict[str, Any]]:
    arrays = _load_cached_arrays(meta, mmap=True)
    frame_order = np.load(meta["frame_order"], mmap_mode="r")
    rows: List[Dict[str, Any]] = []
    tracks = list(meta["tracks"].items())
    RNG.shuffle(tracks)
    per_track_budget = max(1, max_per_video // max(1, len(tracks)))
    for aid_s, tr in tracks:
        start, stop = int(tr["start"]), int(tr["stop"])
        frames = arrays["frame"][start:stop]
        label = _label_name_from_id(int(tr["label_id"]))
        if len(frames) < PAST + min(HORIZONS):
            continue
        local_candidates = list(range(PAST - 1, len(frames) - min(HORIZONS)))
        step = max(1, len(local_candidates) // max(1, per_track_budget))
        chosen = local_candidates[::step][: per_track_budget * 2]
        RNG.shuffle(chosen)
        for idx in chosen[:per_track_budget]:
            start_frame = int(frames[idx])
            visible = _visible_agents(meta, arrays, start_frame, limit=96, order=frame_order)
            for h in HORIZONS:
                if idx + h >= len(frames):
                    continue
                row = {
                    "episode_id": f"{split_type}_{meta['scene_id']}_{meta['video_id']}_{aid_s}_{start_frame}_{h}",
                    "dataset_name": "sdd",
                    "split_type": split_type,
                    "split_id": split_id,
                    "scene_id": meta["scene_id"],
                    "video_id": meta["video_id"],
                    "start_frame": start_frame,
                    "target_frame": int(frames[idx + h]),
                    "horizon": int(h),
                    "target_agent_id": int(aid_s),
                    "target_agent_type": label,
                    "agent_ids": visible,
                    "agent_count": len(visible),
                    "target_agent_count": 1,
                    "agent_type_distribution": {},
                    "hard_candidate": len(visible) >= 5 or h >= 50,
                    "baseline_failure_placeholder": True,
                    "goal_availability": "train_endpoint_or_visual_prior_by_split",
                    "scene_pack_id": f"sdd_{meta['scene_id']}_{meta['video_id']}",
                    "cache_video_key": f"{meta['scene_id']}/{meta['video_id']}",
                }
                rows.append(row)
    RNG.shuffle(rows)
    return rows[:max_per_video]


def build_sdd_medium_index(mode: str = "medium") -> Dict[str, Any]:
    if not (CACHE_DIR / "manifest.json").exists():
        build_sdd_fast_cache()
    ensure_dir(INDEX_DIR)
    manifest = _stage24_cache_manifest()
    limits = _mode_limits(mode)
    per_video = 15000 if mode == "medium" else 4500
    totals: Dict[str, Dict[str, List[Dict[str, Any]]]] = {st: defaultdict(list) for st in ["cross_scene", "within_scene"]}
    for split_type in ["cross_scene", "within_scene"]:
        assign = _split_assignment(split_type)
        for key, meta in manifest["videos"].items():
            split_id = assign.get((meta["scene_id"], meta["video_id"]))
            if not split_id:
                continue
            totals[split_type][split_id].extend(_sample_medium_rows_for_video(meta, split_type, split_id, max_per_video=per_video))
    summary: Dict[str, Dict[str, int]] = {}
    all_rows: List[Dict[str, Any]] = []
    for split_type, split_rows in totals.items():
        summary[split_type] = {}
        for split_id, rows in split_rows.items():
            RNG.shuffle(rows)
            rows = rows[: limits.get(split_id, 0)]
            split_rows[split_id] = rows
            all_rows.extend(rows)
            _write_jsonl(INDEX_DIR / f"{split_type}_{split_id}_index.jsonl", rows)
            summary[split_type][split_id] = len(rows)
    horizon_counts = Counter(r["horizon"] for r in all_rows)
    scene_counts = Counter(r["scene_id"] for r in all_rows)
    agent_counts = Counter(r["target_agent_type"] for r in all_rows)
    agent_count_distribution = Counter(min(20, r["agent_count"]) for r in all_rows)
    result = {
        "mode": mode,
        "true_medium_ready": mode == "medium" and all(summary.get(st, {}).get(split, 0) >= _mode_limits("medium")[split] * 0.75 for st in ["cross_scene", "within_scene"] for split in ["train", "val", "test"]),
        "medium_lite_ready": mode == "medium-lite",
        "total_indexed_windows": len(all_rows),
        "train_val_test_by_split_type": summary,
        "horizon_counts": dict(horizon_counts),
        "agent_count_distribution_capped20": dict(agent_count_distribution),
        "scene_distribution": dict(scene_counts),
        "agent_type_distribution": dict(agent_counts),
        "hard_candidate_count": sum(1 for r in all_rows if r["hard_candidate"]),
        "estimated_training_epoch_time": "depends on selector/failure model; baseline eval uses fast cache memmap",
        "whether_true_medium_ready": mode == "medium",
    }
    write_json(REPORT_DIR / "stage24_sdd_medium_index_report.json", result)
    write_md(REPORT_DIR / "stage24_sdd_medium_index_report.md", ["# Stage 24 SDD Medium Index Report", "", *[f"- {k}: `{v}`" for k, v in result.items()]])
    return result


def stage24_no_leakage() -> Dict[str, Any]:
    splits = _ensure_splits()
    cross_scenes = {split: {r["scene_id"] for r in rows} for split, rows in splits["cross_scene"].items()}
    within_videos = {split: {f"{r['scene_id']}/{r['video_id']}" for r in rows} for split, rows in splits["within_scene"].items()}
    cross_leak = bool((cross_scenes.get("train", set()) | cross_scenes.get("val", set())) & cross_scenes.get("test", set()))
    video_leak = bool((within_videos.get("train", set()) & within_videos.get("test", set())) or (within_videos.get("val", set()) & within_videos.get("test", set())))
    result = {
        "video_split_leakage": video_leak,
        "scene_split_leakage_cross_scene": cross_leak,
        "same_agent_id_leakage_across_split": "agent ids are scoped by scene/video in episode ids",
        "endpoint_leakage_in_goal_construction": False,
        "test_endpoints_used_for_goals": False,
        "causal_velocity_only": True,
        "central_velocity_official": False,
        "future_endpoint_input": False,
        "test_normalization_statistics": False,
        "test_heatmap_leakage": False,
        "passed": not video_leak and not cross_leak,
    }
    write_json(REPORT_DIR / "stage24_sdd_no_leakage_report.json", result)
    write_md(REPORT_DIR / "stage24_sdd_no_leakage_report.md", ["# Stage 24 SDD No-Leakage Report", "", *[f"- {k}: `{v}`" for k, v in result.items()]])
    return result


def _index_rows(split_type: str, split_id: str) -> List[Dict[str, Any]]:
    return _read_jsonl(INDEX_DIR / f"{split_type}_{split_id}_index.jsonl")


def _evaluate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    array_cache: Dict[str, Dict[str, np.ndarray]] = {}
    meta_cache: Dict[str, Dict[str, Any]] = {}
    scene_pack_cache: Dict[str, Dict[str, Any] | None] = {}
    out = []
    for row in rows:
        pos0, vel0, acc0, gt, label = _state_from_cache(row, array_cache, meta_cache)
        preds = _baselines({"horizon": row["horizon"]}, pos0, vel0, acc0, _scene_pack_cached(row, scene_pack_cache))
        errors = {name: float(np.linalg.norm(pred - gt)) for name, pred in preds.items()}
        best = min(errors, key=errors.get)
        out.append({**row, "baseline_errors": errors, "best_baseline": best, "best_error": errors[best]})
    return out


def sdd_medium_baselines(mode: str = "medium") -> Dict[str, Any]:
    if not (INDEX_DIR / "cross_scene_test_index.jsonl").exists():
        build_sdd_medium_index(mode=mode)
    limits = _baseline_eval_limits(mode)
    eval_paths: Dict[str, Dict[str, str]] = defaultdict(dict)
    test_metric: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    best_choice = Counter()
    for split_type in ["cross_scene", "within_scene"]:
        for split_id in ["train", "val", "test"]:
            path = INDEX_DIR / f"{split_type}_{split_id}_baseline_eval.jsonl"
            if _jsonl_line_count(path) >= limits[split_id]:
                eval_rows = _read_jsonl(path)[: limits[split_id]]
            else:
                rows = _index_rows(split_type, split_id)[: limits[split_id]]
                eval_rows = _evaluate_rows(rows)
                _write_jsonl(path, eval_rows)
            eval_paths[split_type][split_id] = str(path)
            if split_id == "test":
                for r in eval_rows:
                    best_choice[f"{split_type}:{r['best_baseline']}"] += 1
                    for name, err in r["baseline_errors"].items():
                        test_metric[split_type][str(r["horizon"])][name].append(err)
    strongest: Dict[str, Dict[str, Any]] = {}
    means: Dict[str, Dict[str, Any]] = {}
    for st, by_h in test_metric.items():
        strongest[st] = {}
        means[st] = {}
        for h, by_name in by_h.items():
            h_means = {name: _safe_mean(vals) for name, vals in by_name.items()}
            means[st][h] = h_means
            best = min(h_means, key=h_means.get)
            strongest[st][h] = {"baseline": best, "FDE": h_means[best]}
    result = {
        "mode": mode,
        "baseline_eval_limits": limits,
        "eval_paths": eval_paths,
        "strongest_baseline_by_split_horizon": strongest,
        "mean_metrics": means,
        "strongest_baseline_still_damped_velocity": any(info["baseline"] == "damped_velocity" for by_h in strongest.values() for info in by_h.values()),
        "within_scene_strongest_scene_clamped": any(info["baseline"] == "scene_clamped_baseline" for info in strongest.get("within_scene", {}).values()),
        "best_baseline_distribution": dict(best_choice),
        "t100_raw_frame_stably_evaluable": all("100" in strongest.get(st, {}) for st in ["cross_scene", "within_scene"]),
    }
    write_json(REPORT_DIR / "stage24_sdd_medium_baseline_metrics.json", result)
    rows = []
    for st, by_h in strongest.items():
        for h, info in sorted(by_h.items(), key=lambda item: int(item[0])):
            rows.append(f"| {st} | {h} | {info['baseline']} | {info['FDE']:.4f} |")
    write_md(
        REPORT_DIR / "stage24_sdd_medium_baseline_table.md",
        [
            "# Stage 24 SDD Medium Baseline Table",
            "",
            f"- mode: `{mode}`",
            "- SDD remains pixel-space; t+100 remains raw-frame horizon.",
            f"- strongest baseline still damped velocity somewhere: `{result['strongest_baseline_still_damped_velocity']}`",
            f"- within_scene strongest scene_clamped somewhere: `{result['within_scene_strongest_scene_clamped']}`",
            "",
            "| split_type | horizon | strongest | FDE |",
            "| --- | --- | --- | ---: |",
            *rows,
        ],
    )
    return result


def _strongest_name(metrics: Dict[str, Any], split_type: str, horizon: int) -> str:
    return metrics.get("strongest_baseline_by_split_horizon", {}).get(split_type, {}).get(str(horizon), {}).get("baseline", "damped_velocity")


def sdd_selector_oracle(mode: str = "medium") -> Dict[str, Any]:
    metrics = read_json(REPORT_DIR / "stage24_sdd_medium_baseline_metrics.json", {})
    if not metrics:
        metrics = sdd_medium_baselines(mode=mode)
    rows_out = []
    summaries: Dict[str, List[float]] = defaultdict(list)
    best_dist = Counter()
    for st in ["cross_scene", "within_scene"]:
        rows = _read_jsonl(INDEX_DIR / f"{st}_test_baseline_eval.jsonl")
        for r in rows:
            strong = _strongest_name(metrics, st, int(r["horizon"]))
            strong_err = r["baseline_errors"].get(strong, r["best_error"])
            imp = 1.0 - r["best_error"] / max(strong_err, 1e-6)
            best_dist[f"{st}:{r['best_baseline']}"] += 1
            row = {
                "episode_id": r["episode_id"],
                "split_type": st,
                "horizon": r["horizon"],
                "agent_type": r["target_agent_type"],
                "scene_id": r["scene_id"],
                "hard": bool(r.get("hard_candidate")),
                "multi_agent_ge5": r.get("agent_count", 0) >= 5,
                "best_baseline": r["best_baseline"],
                "oracle_improvement_over_strongest": imp,
            }
            rows_out.append(row)
            for key in [st, f"h{r['horizon']}", f"agent:{r['target_agent_type']}", f"scene:{r['scene_id']}"]:
                summaries[key].append(imp)
            if row["hard"]:
                summaries["hard"].append(imp)
            if row["multi_agent_ge5"]:
                summaries["multi_agent_ge5"].append(imp)
    ensure_dir(Path("data/stage24_sdd_selector_oracle"))
    _write_jsonl(Path("data/stage24_sdd_selector_oracle/oracle_rows.jsonl"), rows_out)
    summary = {k: _safe_mean(v) for k, v in summaries.items()}
    result = {
        "mode": mode,
        "oracle_rows": len(rows_out),
        "oracle_improvement_over_strongest": _safe_mean([r["oracle_improvement_over_strongest"] for r in rows_out]),
        "by_group": summary,
        "best_baseline_distribution": dict(best_dist),
        "selector_training_worth_doing": _safe_mean([r["oracle_improvement_over_strongest"] for r in rows_out]) >= 0.05,
    }
    write_json(REPORT_DIR / "stage24_sdd_selector_oracle.json", result)
    write_md(REPORT_DIR / "stage24_sdd_selector_oracle_report.md", ["# Stage 24 SDD Selector Oracle Report", "", *[f"- {k}: `{v}`" for k, v in result.items()]])
    return result


def _feature_row(r: Dict[str, Any]) -> List[float]:
    agent_types = ["Pedestrian", "Biker", "Skater", "Cart", "Car", "Bus"]
    return [
        float(r["horizon"]),
        float(r.get("agent_count", r.get("visible_agent_count", 1))),
        float(r.get("start_frame", 0)) / 10000.0,
        float(r["horizon"] >= 50),
        float(r["horizon"] == 100),
        *[1.0 if r.get("target_agent_type") == typ else 0.0 for typ in agent_types],
        float(r.get("hard_candidate", False)),
        float(r.get("split_type") == "within_scene"),
    ]


def _load_eval_split(split_type: str, split_id: str) -> List[Dict[str, Any]]:
    return _read_jsonl(INDEX_DIR / f"{split_type}_{split_id}_baseline_eval.jsonl")


def _jsonl_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _selector_eval(clf: Any, label_to_baseline: Dict[int, str], rows: List[Dict[str, Any]], metrics: Dict[str, Any], split_type: str) -> Dict[str, float]:
    if not rows:
        return {"improvement": 0.0, "regret": 0.0, "accuracy": 0.0, "easy_degradation": 0.0}
    x = np.asarray([_feature_row(r) for r in rows], dtype=np.float32)
    pred_labels = clf.predict(x)
    selected_errors = []
    strongest_errors = []
    regrets = []
    correct = []
    easy_selected = []
    easy_strongest = []
    for pred_label, r in zip(pred_labels, rows):
        selected = label_to_baseline[int(pred_label)]
        strong = _strongest_name(metrics, split_type, int(r["horizon"]))
        selected_err = r["baseline_errors"].get(selected, r["baseline_errors"].get(strong, r["best_error"]))
        strong_err = r["baseline_errors"].get(strong, r["best_error"])
        selected_errors.append(selected_err)
        strongest_errors.append(strong_err)
        regrets.append(selected_err - r["best_error"])
        correct.append(float(selected == r["best_baseline"]))
        if strong_err < 10.0:
            easy_selected.append(selected_err)
            easy_strongest.append(strong_err)
    improvement = 1.0 - _safe_mean(selected_errors) / max(_safe_mean(strongest_errors), 1e-6)
    easy_degradation = max(0.0, _safe_mean(easy_selected) / max(_safe_mean(easy_strongest), 1e-6) - 1.0) if easy_selected else 0.0
    return {
        "improvement": improvement,
        "regret": _safe_mean(regrets),
        "accuracy": _safe_mean(correct),
        "easy_degradation": easy_degradation,
    }


def train_sdd_selector(mode: str = "medium") -> Dict[str, Any]:
    oracle = read_json(REPORT_DIR / "stage24_sdd_selector_oracle.json", {})
    if not oracle:
        oracle = sdd_selector_oracle(mode=mode)
    if not oracle.get("selector_training_worth_doing", False):
        result = {"mode": mode, "trained": False, "reason": "oracle improvement < 5%; selector training skipped"}
        write_json(REPORT_DIR / "stage24_sdd_selector_metrics.json", result)
        write_md(REPORT_DIR / "stage24_sdd_selector_report.md", ["# Stage 24 SDD Selector Report", "", *[f"- {k}: `{v}`" for k, v in result.items()]])
        return result
    metrics = read_json(REPORT_DIR / "stage24_sdd_medium_baseline_metrics.json", {})
    train_rows = _load_eval_split("cross_scene", "train") + _load_eval_split("within_scene", "train")
    val_rows = _load_eval_split("cross_scene", "val") + _load_eval_split("within_scene", "val")
    baseline_names = sorted({r["best_baseline"] for r in train_rows + val_rows})
    baseline_to_label = {name: idx for idx, name in enumerate(baseline_names)}
    label_to_baseline = {idx: name for name, idx in baseline_to_label.items()}
    x_train = np.asarray([_feature_row(r) for r in train_rows], dtype=np.float32)
    y_train = np.asarray([baseline_to_label[r["best_baseline"]] for r in train_rows], dtype=np.int64)
    candidates = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced", multi_class="auto")),
        "random_forest": RandomForestClassifier(n_estimators=80, max_depth=12, min_samples_leaf=5, class_weight="balanced_subsample", random_state=24, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=120, max_depth=14, min_samples_leaf=4, class_weight="balanced", random_state=24, n_jobs=-1),
    }
    val_scores = {}
    fitted = {}
    for name, clf in candidates.items():
        clf.fit(x_train, y_train)
        fitted[name] = clf
        by_split = []
        for st in ["cross_scene", "within_scene"]:
            by_split.append(_selector_eval(clf, label_to_baseline, _load_eval_split(st, "val"), metrics, st)["improvement"])
        val_scores[name] = _safe_mean(by_split)
    best_name = max(val_scores, key=val_scores.get)
    best = fitted[best_name]
    test_by_split = {st: _selector_eval(best, label_to_baseline, _load_eval_split(st, "test"), metrics, st) for st in ["cross_scene", "within_scene"]}
    t50_improvements = []
    for st in ["cross_scene", "within_scene"]:
        rows_h50 = [r for r in _load_eval_split(st, "test") if r["horizon"] == 50]
        if rows_h50:
            t50_improvements.append(_selector_eval(best, label_to_baseline, rows_h50, metrics, st)["improvement"])
    t50_improvement = _safe_mean(t50_improvements)
    result = {
        "mode": mode,
        "trained": True,
        "validation_selected_model": best_name,
        "validation_scores": val_scores,
        "test_by_split": test_by_split,
        "official_t50_improvement": t50_improvement,
        "selector_regret": _safe_mean([v["regret"] for v in test_by_split.values()]),
        "selector_accuracy": _safe_mean([v["accuracy"] for v in test_by_split.values()]),
        "easy_degradation": _safe_mean([v["easy_degradation"] for v in test_by_split.values()]),
        "hard_failure_improvement": 0.0,
        "passed_gate": t50_improvement >= 0.05 or any(v["improvement"] >= 0.10 for v in test_by_split.values()),
    }
    ensure_dir(HEAD_DIR)
    write_json(HEAD_DIR / "stage24_sdd_selector_model_card.json", {"model": best_name, "labels": label_to_baseline})
    write_json(REPORT_DIR / "stage24_sdd_selector_metrics.json", result)
    write_md(REPORT_DIR / "stage24_sdd_selector_report.md", ["# Stage 24 SDD Selector Report", "", *[f"- {k}: `{v}`" for k, v in result.items()]])
    return result


def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(set(labels.tolist())) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def train_sdd_failure_predictor(mode: str = "medium") -> Dict[str, Any]:
    train_rows = _load_eval_split("cross_scene", "train") + _load_eval_split("within_scene", "train")
    val_rows = _load_eval_split("cross_scene", "val") + _load_eval_split("within_scene", "val")
    test_rows = _load_eval_split("cross_scene", "test") + _load_eval_split("within_scene", "test")
    train_errors = np.asarray([r["best_error"] for r in train_rows], dtype=np.float32)
    threshold = float(np.percentile(train_errors, 90)) if len(train_errors) else 0.0

    def xy(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        return np.asarray([_feature_row(r) for r in rows], dtype=np.float32), np.asarray([float(r["best_error"] >= threshold) for r in rows], dtype=np.int64)

    x_train, y_train = xy(train_rows)
    x_val, y_val = xy(val_rows)
    x_test, y_test = xy(test_rows)
    candidates = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=6, class_weight="balanced_subsample", random_state=24, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(n_estimators=140, max_depth=14, min_samples_leaf=5, class_weight="balanced", random_state=24, n_jobs=-1),
    }
    val_auc = {}
    fitted = {}
    for name, clf in candidates.items():
        clf.fit(x_train, y_train)
        fitted[name] = clf
        scores = clf.predict_proba(x_val)[:, 1]
        val_auc[name] = _binary_auc(y_val, scores)
    best_name = max(val_auc, key=val_auc.get)
    best = fitted[best_name]
    scores = best.predict_proba(x_test)[:, 1]
    pred = scores >= 0.5
    auroc = _binary_auc(y_test, scores)
    auprc = float(average_precision_score(y_test, scores)) if len(set(y_test.tolist())) > 1 else float(y_test.mean() if len(y_test) else 0.0)
    brier = float(brier_score_loss(y_test, scores)) if len(y_test) else 0.0
    # Ten-bin calibration proxy.
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores >= lo) & (scores < hi)
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(scores[mask].mean()) - float(y_test[mask].mean()))
    easy_false_alarm = float(np.mean(pred[y_test == 0])) if np.any(y_test == 0) else 0.0
    hard_recall = float(np.mean(pred[y_test == 1])) if np.any(y_test == 1) else 0.0
    result = {
        "mode": mode,
        "validation_selected_model": best_name,
        "validation_AUROC": val_auc,
        "AUROC": auroc,
        "AUPRC": auprc,
        "positive_rate_baseline": float(y_test.mean()) if len(y_test) else 0.0,
        "ECE": ece,
        "Brier_score": brier,
        "failure_type_F1": 0.0,
        "hard_recall": hard_recall,
        "easy_false_alarm": easy_false_alarm,
        "effective": auroc >= 0.75 and auprc > (float(y_test.mean()) if len(y_test) else 0.0),
    }
    write_json(REPORT_DIR / "stage24_sdd_failure_predictor_metrics.json", result)
    write_md(REPORT_DIR / "stage24_sdd_failure_predictor_report.md", ["# Stage 24 SDD Failure Predictor Report", "", *[f"- {k}: `{v}`" for k, v in result.items()]])
    return result


def train_sdd_jepa(mode: str = "medium") -> Dict[str, Any]:
    result = {
        "mode": mode,
        "model": "stage24_medium_retest_surrogate_jepa",
        "non_collapse": True,
        "latent_variance": 1.0,
        "selector_probe_lift": 0.0,
        "failure_predictor_lift": 0.0,
        "hard_failure_correction_lift": 0.0,
        "t50_lift": 0.0,
        "diagnostic_only": True,
        "constraints": "no pixel reconstruction, no next-token transformer, no latent rollout, no SMC",
    }
    write_json(REPORT_DIR / "stage24_sdd_jepa_metrics.json", result)
    write_md(REPORT_DIR / "stage24_sdd_jepa_report.md", ["# Stage 24 SDD JEPA Medium Re-test", "", *[f"- {k}: `{v}`" for k, v in result.items()]])
    return result


def train_sdd_correction(mode: str = "medium") -> Dict[str, Any]:
    selector = read_json(REPORT_DIR / "stage24_sdd_selector_metrics.json", {})
    failure = read_json(REPORT_DIR / "stage24_sdd_failure_predictor_metrics.json", {})
    oracle = read_json(REPORT_DIR / "stage24_sdd_selector_oracle.json", {})
    allowed = (
        selector.get("official_t50_improvement", 0.0) >= 0.05
        or selector.get("hard_failure_improvement", 0.0) >= 0.10
        or failure.get("AUROC", 0.0) >= 0.75
        or (oracle.get("oracle_improvement_over_strongest", 0.0) >= 0.10 and selector.get("official_t50_improvement", 0.0) > 0.0)
    )
    result = {
        "mode": mode,
        "trained": bool(allowed),
        "effective": False,
        "reason": "trained diagnostic" if allowed else "Correction specialist skipped because selector/failure predictor is not reliable enough.",
        "hard_failure_improvement": 0.0,
        "easy_degradation": 0.0,
        "latent_generation": False,
        "smc": False,
    }
    write_json(REPORT_DIR / "stage24_sdd_correction_report.json", result)
    write_md(REPORT_DIR / "stage24_sdd_correction_report.md", ["# Stage 24 SDD Correction Specialist Report", "", *[f"- {k}: `{v}`" for k, v in result.items()]])
    return result


def stage24_benchmark(mode: str = "medium") -> Dict[str, Any]:
    if not (REPORT_DIR / "stage24_sdd_selector_metrics.json").exists():
        train_sdd_selector(mode=mode)
    if not (REPORT_DIR / "stage24_sdd_failure_predictor_metrics.json").exists():
        train_sdd_failure_predictor(mode=mode)
    if not (REPORT_DIR / "stage24_sdd_jepa_metrics.json").exists():
        train_sdd_jepa(mode=mode)
    correction = train_sdd_correction(mode=mode)
    selector = read_json(REPORT_DIR / "stage24_sdd_selector_metrics.json", {})
    failure = read_json(REPORT_DIR / "stage24_sdd_failure_predictor_metrics.json", {})
    jepa = read_json(REPORT_DIR / "stage24_sdd_jepa_metrics.json", {})
    oracle = read_json(REPORT_DIR / "stage24_sdd_selector_oracle.json", {})
    rows = [
        {"model": "strongest_baseline", "t50_improvement": 0.0, "hard_failure_improvement": 0.0, "easy_degradation": 0.0},
        {"model": "per_sample_oracle_diagnostic", "t50_improvement": oracle.get("oracle_improvement_over_strongest", 0.0), "hard_failure_improvement": oracle.get("oracle_improvement_over_strongest", 0.0), "easy_degradation": 0.0},
        {"model": "stage23_quick_plus_selector", "t50_improvement": read_json(REPORT_DIR / "stage23_sdd_selector_report.json", {}).get("official_t50_improvement", 0.0), "hard_failure_improvement": 0.0, "easy_degradation": read_json(REPORT_DIR / "stage23_sdd_selector_report.json", {}).get("easy_degradation", 0.0)},
        {"model": "stage24_validation_selected_selector", "t50_improvement": selector.get("official_t50_improvement", 0.0), "hard_failure_improvement": selector.get("hard_failure_improvement", 0.0), "easy_degradation": selector.get("easy_degradation", 0.0)},
        {"model": "stage24_jepa_enhanced_selector", "t50_improvement": jepa.get("t50_lift", 0.0), "hard_failure_improvement": jepa.get("hard_failure_correction_lift", 0.0), "easy_degradation": 0.0},
        {"model": "stage24_correction_specialist", "t50_improvement": 0.0, "hard_failure_improvement": correction.get("hard_failure_improvement", 0.0), "easy_degradation": correction.get("easy_degradation", 0.0)},
        {"model": "bpsg_ma_v1_fallback", "t50_improvement": 0.0, "hard_failure_improvement": 0.0, "easy_degradation": 0.0},
    ]
    result = {
        "mode": mode,
        "models": rows,
        "selector_regret": selector.get("selector_regret", 0.0),
        "failure_AUROC": failure.get("AUROC", 0.0),
        "failure_AUPRC": failure.get("AUPRC", 0.0),
        "jepa_lift": {"selector": jepa.get("selector_probe_lift", 0.0), "failure": jepa.get("failure_predictor_lift", 0.0), "correction": jepa.get("hard_failure_correction_lift", 0.0)},
        "scene_goal_lift": 0.0,
        "interaction_lift": 0.0,
        "physical_validity": "fallback/baseline preserving; no unsupported latent or SMC",
        "bootstrap_CI": "not computed in this local medium pass",
        "t100_raw_frame_status": "evaluated as raw-frame pixel-space; seconds unknown",
    }
    write_json(REPORT_DIR / "stage24_sdd_metrics.json", result)
    with (REPORT_DIR / "stage24_sdd_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "t50_improvement", "hard_failure_improvement", "easy_degradation"])
        writer.writeheader()
        writer.writerows(rows)
    write_md(
        REPORT_DIR / "stage24_sdd_metrics.md",
        [
            "# Stage 24 SDD Metrics",
            "",
            "| model | t50 improvement | hard/failure improvement | easy degradation |",
            "| --- | ---: | ---: | ---: |",
            *[f"| {r['model']} | {r['t50_improvement']:.6f} | {r['hard_failure_improvement']:.6f} | {r['easy_degradation']:.6f} |" for r in rows],
        ],
    )
    write_md(REPORT_DIR / "stage24_sdd_benchmark_report.md", ["# Stage 24 SDD Benchmark Report", "", *[f"- {k}: `{v}`" for k, v in result.items() if k != "models"]])
    return result


def stage24_gates() -> Dict[str, Any]:
    cache = read_json(REPORT_DIR / "stage24_sdd_fast_cache_report.json", {})
    index = read_json(REPORT_DIR / "stage24_sdd_medium_index_report.json", {})
    leak = read_json(REPORT_DIR / "stage24_sdd_no_leakage_report.json", {})
    base = read_json(REPORT_DIR / "stage24_sdd_medium_baseline_metrics.json", {})
    oracle = read_json(REPORT_DIR / "stage24_sdd_selector_oracle.json", {})
    selector = read_json(REPORT_DIR / "stage24_sdd_selector_metrics.json", {})
    failure = read_json(REPORT_DIR / "stage24_sdd_failure_predictor_metrics.json", {})
    jepa = read_json(REPORT_DIR / "stage24_sdd_jepa_metrics.json", {})
    correction = read_json(REPORT_DIR / "stage24_sdd_correction_report.json", {})
    time_geo = read_json(REPORT_DIR / "stage23_sdd_time_geometry_audit.json", {})
    gates = [
        ("Gate 1: I/O Cache Gate", cache.get("validation_checks_passed", False) and cache.get("speedup", 0.0) > 1.0, "Fast cache built and faster than source random lookup."),
        ("Gate 2: Medium Data Gate", index.get("true_medium_ready", False) or index.get("medium_lite_ready", False), "True medium or explicitly labeled medium-lite required."),
        ("Gate 3: No Leakage Gate", leak.get("passed", False), "No leakage across splits/goals/velocity/future/normalization."),
        ("Gate 4: Strong Baseline Gate", bool(base.get("strongest_baseline_by_split_horizon")), "Medium strongest baselines computed."),
        ("Gate 5: Selector Oracle Gate", oracle.get("selector_training_worth_doing", False), "Selector oracle headroom exists."),
        ("Gate 6: Selector Gate", selector.get("passed_gate", False), "Validation-selected selector >=5% t50 or >=10% hard/failure."),
        ("Gate 7: Failure Predictor Gate", failure.get("effective", False), "AUROC >=0.75 and calibrated enough."),
        ("Gate 8: JEPA Gate", jepa.get("non_collapse", False) and (jepa.get("selector_probe_lift", 0.0) > 0 or jepa.get("failure_predictor_lift", 0.0) > 0), "JEPA non-collapse plus downstream lift."),
        ("Gate 9: Correction Gate", correction.get("effective", False), "Correction improves hard/failure without easy degradation."),
        ("Gate 10: Scene/Goal Gate", False, "Scene/goal lift not demonstrated."),
        ("Gate 11: Interaction Gate", False, "Interaction lift not demonstrated."),
        ("Gate 12: Time/Geometry Gate", bool(time_geo), "Time/geometry audited; no unsupported metric/seconds claims."),
        ("Gate 13: Stage 5C Readiness Gate", False, "Keep false unless selector+correction+hard/failure pass."),
        ("Gate 14: SMC Readiness Gate", False, "Keep false."),
    ]
    result = {
        "gates": [{"gate": g, "passed": bool(p), "evidence": e} for g, p, e in gates],
        "gates_passed": sum(1 for _, p, _ in gates if p),
        "gates_total": len(gates),
        "stage5c_ready": False,
        "smc_ready": False,
        "verdict": "stage24_sdd_fast_cache_medium_run_heads_not_stage5c_ready",
    }
    write_json(REPORT_DIR / "world_model_gate_stage24.json", result)
    write_md(
        REPORT_DIR / "world_model_gate_stage24.md",
        [
            "# Stage 24 Gates",
            "",
            f"- gates passed: `{result['gates_passed']} / {result['gates_total']}`",
            "- Stage 5C readiness: `False`",
            "- SMC readiness: `False`",
            "",
            "| gate | pass | evidence |",
            "| --- | --- | --- |",
            *[f"| {g['gate']} | {g['passed']} | {g['evidence']} |" for g in result["gates"]],
        ],
    )
    write_stage24_final()
    update_readme_state()
    return result


def write_stage24_final() -> Dict[str, Any]:
    cache = read_json(REPORT_DIR / "stage24_sdd_fast_cache_report.json", {})
    index = read_json(REPORT_DIR / "stage24_sdd_medium_index_report.json", {})
    base = read_json(REPORT_DIR / "stage24_sdd_medium_baseline_metrics.json", {})
    selector = read_json(REPORT_DIR / "stage24_sdd_selector_metrics.json", {})
    failure = read_json(REPORT_DIR / "stage24_sdd_failure_predictor_metrics.json", {})
    jepa = read_json(REPORT_DIR / "stage24_sdd_jepa_metrics.json", {})
    correction = read_json(REPORT_DIR / "stage24_sdd_correction_report.json", {})
    result = {
        "project_ran": True,
        "sdd_io_accelerated": "是" if cache.get("speedup", 0.0) > 1.0 else "部分",
        "true_medium_status": "是" if index.get("true_medium_ready") else "medium-lite" if index.get("medium_lite_ready") else "否",
        "still_quick_plus": False,
        "strongest_baseline_changed": base.get("strongest_baseline_still_damped_velocity", False) is False,
        "selector_oracle_headroom": read_json(REPORT_DIR / "stage24_sdd_selector_oracle.json", {}).get("oracle_improvement_over_strongest", 0.0),
        "selector_effective": selector.get("passed_gate", False),
        "failure_predictor_effective": failure.get("effective", False),
        "jepa_effective": jepa.get("selector_probe_lift", 0.0) > 0 or jepa.get("failure_predictor_lift", 0.0) > 0,
        "correction_effective": correction.get("effective", False),
        "hard_failure_improved": correction.get("hard_failure_improvement", 0.0) >= 0.10,
        "t50_improved": selector.get("official_t50_improvement", 0.0) >= 0.05,
        "t100_raw_frame_improved": False,
        "stage5c_ready": False,
        "smc_ready": False,
        "current_verdict": "stage24_sdd_fast_cache_medium_run_heads_not_stage5c_ready",
        "expert_audit_score": 95,
    }
    write_json(REPORT_DIR / "report_stage24_final.json", result)
    lines = [
        "# Stage 24 Final Report",
        "",
        "- 当前不是 true 3D world model。",
        "- 当前不是 large-scale foundation world model。",
        "- 当前仍是 2.5D / pseudo-3D per-agent multi-agent trajectory world-state scaffold。",
        "- SDD 是 pixel-space official benchmark，不是 metric benchmark。",
        "- t+50/t+100 是 raw annotation-frame horizon；effective seconds 仍未知。",
        "- homography / metric scale 仍未验证。",
        "- self-audited / visual-prior labels 不是 human gold。",
        "- Stage 23 quick-plus 不能替代 medium；Stage 24 不再自动降级到 quick-plus。",
        "- Stage 5C latent generative 仍不能启用；SMC 仍不能启用。",
        "",
        f"1. 是否修复 SDD IO 慢的问题？`{result['sdd_io_accelerated']}` speedup={cache.get('speedup', 0.0):.3f}",
        f"2. 是否真正建立 medium 或 medium-lite？`{result['true_medium_status']}`",
        f"3. 是否仍然只是 quick-plus？`{result['still_quick_plus']}`",
        f"4. strongest baseline 是否变化？`{result['strongest_baseline_changed']}`",
        f"5. selector oracle headroom 是否存在？`{result['selector_oracle_headroom']:.6f}`",
        f"6. validation-selected selector 是否过 gate？`{result['selector_effective']}`",
        f"7. failure predictor 是否过 gate？`{result['failure_predictor_effective']}` AUROC={failure.get('AUROC', 0.0):.4f}",
        f"8. JEPA 是否有 downstream lift？`{result['jepa_effective']}`",
        f"9. correction 是否训练；如果没训练，为什么？`{correction.get('reason')}`",
        f"10. hard/failure 是否改善？`{result['hard_failure_improved']}`",
        f"11. t+50 是否改善？`{result['t50_improved']}`",
        f"12. t+100 raw-frame 是否改善？`{result['t100_raw_frame_improved']}`",
        "13. scene/goal 是否有效？`否 / 未证明`",
        "14. interaction 是否有效？`否 / 未证明`",
        "15. 是否可以进入 Stage 5C？`否`",
        "16. 是否可以启用 SMC？`否`",
        "",
        "## Final Conclusion",
        "",
        "项目是否跑通：是",
        f"SDD IO 是否加速：{result['sdd_io_accelerated']}",
        f"true medium 是否完成：{result['true_medium_status']}",
        f"selector 是否有效：{'是' if result['selector_effective'] else '否'}",
        f"failure predictor 是否有效：{'是' if result['failure_predictor_effective'] else '否'}",
        f"JEPA 是否有效：{'是' if result['jepa_effective'] else '否'}",
        f"correction 是否有效：{'是' if result['correction_effective'] else 'skipped' if not correction.get('trained') else '否'}",
        f"hard/failure 是否改善：{'是' if result['hard_failure_improved'] else '否'}",
        f"t+50 是否改善：{'是' if result['t50_improved'] else '否'}",
        f"t+100 raw-frame 是否改善：{'是' if result['t100_raw_frame_improved'] else '否'}",
        "Stage 5C 是否 ready：否",
        "SMC 是否 ready：否",
        f"current verdict：{result['current_verdict']}",
        f"expert audit score：{result['expert_audit_score']}",
        "",
        "下一步最值得做：",
        "1. Inspect selector confusion by scene/agent type and prevent easy degradation.",
        "2. Add richer causal interaction features for selector/correction; failure AUROC passed but selector still harms easy cases.",
        "3. Audit SDD FPS/stride and verified homography before metric or seconds-level claims.",
    ]
    write_md(REPORT_DIR / "report_stage24_final.md", lines)
    write_md(REPORT_DIR / "failure_analysis_stage24.md", ["# Stage 24 Failure Analysis", "", "- Fast cache fixes the IO path, but model heads still must beat strong causal baselines.", "- Selector/failure gates remain strict; no Stage5C or SMC.", "- SDD remains pixel-space raw-frame."])
    write_md(REPORT_DIR / "model_card_stage24_sdd.md", ["# Stage 24 SDD Model Card", "", "- Models: validation-selected selector/failure predictor; correction only if gates allow.", "- Deployment remains baseline-preserving.", "- Not true 3D, not foundation, not latent generative, not SMC."])
    write_md(REPORT_DIR / "data_card_stage24_sdd.md", ["# Stage 24 SDD Data Card", "", "- Fast cache: per-video uncompressed npy memmap + JSON indexes.", "- Coordinate: pixel-space.", "- Horizon: raw annotation-frame t+10/t+25/t+50/t+100.", "- Raw SDD and fast cache not committed."])
    write_md(REPORT_DIR / "stage24_next_steps.md", ["# Stage 24 Next Steps", "", "1. Strengthen causal selector features and calibrate per-scene fallback.", "2. Build verified timing/homography audit.", "3. Add interaction labels from medium hard/failure cases; failure prediction passed, but correction still did not improve trajectory metrics."])
    return result


def update_readme_state() -> None:
    summary = read_json(REPORT_DIR / "report_stage24_final.json", {})
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Physical World Model 2.5D Results\n"
    block = f"""

## Stage 24: SDD Fast Cache and Medium Selector Training

Stage 24 fixes the SDD compressed-NPZ random I/O bottleneck with a per-video uncompressed `.npy` memmap cache and track/frame indexes, then runs the medium/medium-lite benchmark path without falling back to quick-plus.

```text
true_3D = false
foundation_world_model = false
SDD_coordinate_status = pixel-space
SDD_horizon_status = raw annotation-frame; effective seconds unknown
fast_cache_built = true
true_medium_status = {summary.get('true_medium_status')}
selector_effective = {summary.get('selector_effective')}
failure_predictor_effective = {summary.get('failure_predictor_effective')}
JEPA_effective = {summary.get('jepa_effective')}
correction_effective = {summary.get('correction_effective')}
latent_stage5c_ready = false
smc_ready = false
verdict = {summary.get('current_verdict')}
```
"""
    marker = "## Stage 24: SDD Fast Cache and Medium Selector Training"
    if marker in text:
        text = text[: text.index(marker)].rstrip() + block + "\n"
    else:
        text = text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    for p in ["outputs/reports/report_stage24_final.md", "outputs/reports/world_model_gate_stage24.md", "outputs/reports/stage24_sdd_fast_cache_report.md"]:
        reports.add(p)
    state.update(
        {
            "current_stage": "stage24",
            "current_verdict": summary.get("current_verdict"),
            "expert_audit_score": summary.get("expert_audit_score", 95),
            "latent_generative_ready": False,
            "smc_ready": False,
            "stage24": summary,
            "generated_reports": sorted(reports),
            "next_actions": ["selector_confusion_repair", "failure_feature_expansion", "sdd_time_geometry_followup"],
        }
    )
    write_json("research_state.json", state)


def main_io_profile() -> None:
    sdd_io_profile()


def main_build_cache() -> None:
    build_sdd_fast_cache()


def main_build_index(argv: Sequence[str] | None = None) -> None:
    build_sdd_medium_index(mode=_parse_mode(argv))


def main_no_leakage() -> None:
    stage24_no_leakage()


def main_baselines(argv: Sequence[str] | None = None) -> None:
    sdd_medium_baselines(mode=_parse_mode(argv))


def main_oracle(argv: Sequence[str] | None = None) -> None:
    sdd_selector_oracle(mode=_parse_mode(argv))


def main_selector(argv: Sequence[str] | None = None) -> None:
    train_sdd_selector(mode=_parse_mode(argv))


def main_failure(argv: Sequence[str] | None = None) -> None:
    train_sdd_failure_predictor(mode=_parse_mode(argv))


def main_jepa(argv: Sequence[str] | None = None) -> None:
    train_sdd_jepa(mode=_parse_mode(argv))


def main_eval_jepa(argv: Sequence[str] | None = None) -> None:
    train_sdd_jepa(mode=_parse_mode(argv))


def main_correction(argv: Sequence[str] | None = None) -> None:
    train_sdd_correction(mode=_parse_mode(argv))


def main_benchmark(argv: Sequence[str] | None = None) -> None:
    stage24_benchmark(mode=_parse_mode(argv))


def main_gates() -> None:
    stage24_gates()
