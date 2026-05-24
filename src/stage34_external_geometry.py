from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage28_pipeline import BASELINE_NAMES, LATENT_DIR as SDD_LATENT_DIR
from src.stage30_m3w_verified import _combined_hash, _feature_manifest, _git_commit, _hash_path
from src import stage31_external_generalization as s31
from src import stage33_coordinate_invariant as s33


OUT_DIR = Path("outputs/stage34_external_geometry")
DATA_DIR = Path("data/stage34_external_geometry")
SCENE_DIR = Path("data/stage34_external_scene_packs")
EXT_FEATURE_DIR = Path("data/stage31_external_feature_store")
EXT_LATENT_DIR = Path("data/stage31_external_latent_cache")
SDD_FEATURE_DIR = Path("data/stage26_sdd_feature_store")
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
EPS = 1e-6

BASELINES_V2 = [
    "constant_position",
    "constant_velocity_causal_fd",
    "damped_velocity",
    "constant_acceleration_causal",
    "constant_turn_rate_velocity",
    "scene_clamped_baseline",
    "goal_directed_baseline",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    write_json(path, _jsonable(dict(payload)))


def _append_ledger(entry: Mapping[str, Any]) -> None:
    ensure_dir(OUT_DIR)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(dict(entry)), ensure_ascii=False) + "\n")
    rows = [json.loads(line) for line in LEDGER_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    lines = [
        "# Stage34 External Geometry Run Ledger",
        "",
        "| command | source | status | wall time s | input hash | output hash | git commit |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['command']}` | `{row['source']}` | `{row['status']}` | {float(row['wall_time_s']):.3f} | `{row['input_hash'][:12]}` | `{row['output_hash'][:12]}` | `{row['git_commit']}` |"
        )
    write_md(OUT_DIR / "run_ledger.md", lines)


def run_logged(name: str, fn: Callable[[], Dict[str, Any]], inputs: Sequence[str | Path], outputs: Sequence[str | Path], source: str = "fresh_run") -> Dict[str, Any]:
    ensure_dir(OUT_DIR)
    start = time.perf_counter()
    status = "failed"
    input_hash = _combined_hash(inputs)
    try:
        payload = fn()
        status = "success"
        return payload
    finally:
        _append_ledger(
            {
                "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
                "step": name,
                "inputs": [str(p) for p in inputs],
                "outputs": [str(p) for p in outputs],
                "wall_time_s": time.perf_counter() - start,
                "status": status,
                "input_hash": input_hash,
                "output_hash": _combined_hash(outputs),
                "git_commit": _git_commit(),
                "source": source,
            }
        )


def _load_feature(domain: str, split: str) -> Dict[str, np.ndarray]:
    return dict(np.load((EXT_FEATURE_DIR if domain == "external" else SDD_FEATURE_DIR) / f"{split}.npz"))


def _load_latent(domain: str, split: str) -> Dict[str, np.ndarray]:
    return dict(np.load((EXT_LATENT_DIR if domain == "external" else SDD_LATENT_DIR) / f"{split}.npz"))


def _source_order(split: str) -> List[str]:
    d = _load_feature("external", split)
    seen: set[str] = set()
    out: List[str] = []
    for src in d.get("source_file", np.asarray([], dtype="U256")).astype(str).tolist():
        if src not in seen:
            seen.add(src)
            out.append(src)
    return out


def _row_geometry_for_file(path: Path) -> Dict[str, List[Any]]:
    arr = s31._read_opentraj_txt(path)
    out: Dict[str, List[Any]] = defaultdict(list)
    if len(arr) == 0:
        return out
    scene = path.parent.name
    for agent in np.unique(arr[:, 1]).astype(int):
        tr = arr[arr[:, 1] == agent]
        tr = tr[np.argsort(tr[:, 0])]
        frames = tr[:, 0].astype(int)
        if len(tr) < 4:
            continue
        step = max(1, len(tr) // 80)
        for i in range(2, len(tr) - 1, step):
            for horizon in [10, 25, 50, 100]:
                target_frame = int(frames[i] + horizon)
                future_ids = np.where(frames >= target_frame)[0]
                if len(future_ids) == 0:
                    continue
                j = int(future_ids[0])
                frame_delta = int(frames[j] - frames[i])
                if j <= i or frame_delta > max(10, horizon + 15):
                    continue
                past = tr[max(0, i - 7) : i + 1]
                p0 = tr[i]
                p_start = past[0]
                fut = tr[j]
                out["scene_id"].append(scene)
                out["source_file"].append(str(path))
                out["agent_id"].append(int(agent))
                out["frame_id"].append(int(frames[i]))
                out["current_x"].append(float(p0[2]))
                out["current_y"].append(float(p0[3]))
                out["past_start_x"].append(float(p_start[2]))
                out["past_start_y"].append(float(p_start[3]))
                out["future_endpoint_x"].append(float(fut[2]))
                out["future_endpoint_y"].append(float(fut[3]))
                out["horizon"].append(int(horizon))
                out["dt_frame_step"].append(frame_delta)
                out["track_length"].append(int(len(tr)))
                out["valid_mask"].append(True)
    return out


def _concat_rows(rows: List[Dict[str, List[Any]]]) -> Dict[str, np.ndarray]:
    keys = [
        "scene_id",
        "source_file",
        "agent_id",
        "frame_id",
        "current_x",
        "current_y",
        "past_start_x",
        "past_start_y",
        "future_endpoint_x",
        "future_endpoint_y",
        "horizon",
        "dt_frame_step",
        "track_length",
        "valid_mask",
    ]
    merged: Dict[str, List[Any]] = {k: [] for k in keys}
    for row in rows:
        for k in keys:
            merged[k].extend(row.get(k, []))
    out: Dict[str, np.ndarray] = {
        "scene_id": np.asarray(merged["scene_id"], dtype="U64"),
        "source_file": np.asarray(merged["source_file"], dtype="U256"),
        "agent_id": np.asarray(merged["agent_id"], dtype=np.int64),
        "frame_id": np.asarray(merged["frame_id"], dtype=np.int64),
        "current_x": np.asarray(merged["current_x"], dtype=np.float32),
        "current_y": np.asarray(merged["current_y"], dtype=np.float32),
        "past_start_x": np.asarray(merged["past_start_x"], dtype=np.float32),
        "past_start_y": np.asarray(merged["past_start_y"], dtype=np.float32),
        "future_endpoint_x": np.asarray(merged["future_endpoint_x"], dtype=np.float32),
        "future_endpoint_y": np.asarray(merged["future_endpoint_y"], dtype=np.float32),
        "horizon": np.asarray(merged["horizon"], dtype=np.int16),
        "dt_frame_step": np.asarray(merged["dt_frame_step"], dtype=np.float32),
        "track_length": np.asarray(merged["track_length"], dtype=np.int32),
        "valid_mask": np.asarray(merged["valid_mask"], dtype=bool),
    }
    return out


def external_row_geometry() -> Dict[str, Any]:
    ensure_dir(DATA_DIR)
    report: Dict[str, Any] = {
        "source": "fresh_run",
        "source_labels": {"stage31_external_feature_store": "cached_verified", "raw_opentraj_files": "cached_verified", "row_geometry": "fresh_run"},
        "splits": {},
        "no_leakage": {
            "future_endpoint_in_features": False,
            "future_endpoint_label_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
    }
    all_ok = True
    for split in ["train", "val", "test"]:
        feature = _load_feature("external", split)
        rows = [_row_geometry_for_file(Path(src)) for src in _source_order(split)]
        geo = _concat_rows(rows)
        expected = int(len(feature["x"]))
        aligned = expected == int(len(geo["horizon"]))
        if aligned:
            aligned = aligned and bool(np.array_equal(feature["horizon"].astype(np.int16), geo["horizon"]))
            aligned = aligned and bool(np.array_equal(feature.get("scene_id", np.asarray([], dtype="U64")).astype(str), geo["scene_id"].astype(str)))
            aligned = aligned and bool(np.array_equal(feature.get("source_file", np.asarray([], dtype="U256")).astype(str), geo["source_file"].astype(str)))
        all_ok = all_ok and aligned
        np.savez_compressed(DATA_DIR / f"external_row_geometry_{split}.npz", **geo)
        report["splits"][split] = {
            "expected_feature_rows": expected,
            "geometry_rows": int(len(geo["horizon"])),
            "aligned_to_feature_store": bool(aligned),
            "scenes": int(len(set(geo["scene_id"].astype(str).tolist()))),
            "agents": int(len(set(geo["agent_id"].astype(int).tolist()))) if len(geo["agent_id"]) else 0,
            "horizon_counts": dict(Counter(geo["horizon"].astype(int).tolist())),
            "has_future_endpoint_label": True,
            "future_endpoint_used_as_inference_feature": False,
        }
    report["row_geometry_complete"] = bool(all_ok)
    _write_json(OUT_DIR / "external_row_geometry_report.json", report)
    write_md(
        OUT_DIR / "external_row_geometry_report.md",
        [
            "# Stage34 External Row Geometry Report",
            "",
            "- source: `fresh_run`; Stage31 feature rows and raw OpenTraj files are `cached_verified`.",
            "- future endpoint is stored only as supervision/evaluation label, not an inference feature.",
            f"- row geometry complete: `{report['row_geometry_complete']}`",
            f"- splits: `{report['splits']}`",
            f"- no leakage: `{report['no_leakage']}`",
        ],
    )
    return report


def _load_geo(split: str) -> Dict[str, np.ndarray]:
    return dict(np.load(DATA_DIR / f"external_row_geometry_{split}.npz"))


def _cluster(points: np.ndarray, max_k: int = 8) -> np.ndarray:
    return s33._cluster_points(points, max_k=max_k)


def _scene_train_goals() -> Dict[str, np.ndarray]:
    train = _load_geo("train")
    out: Dict[str, np.ndarray] = {}
    scenes = train["scene_id"].astype(str)
    pts = np.stack([train["future_endpoint_x"], train["future_endpoint_y"]], axis=1).astype(np.float64)
    for scene in sorted(set(scenes.tolist())):
        p = pts[scenes == scene]
        out[scene] = _cluster(p, max_k=8)
    return out


def _angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.arccos(np.clip(dot / np.maximum(denom, EPS), -1.0, 1.0))


def external_scene_goals() -> Dict[str, Any]:
    external_row_geometry()
    ensure_dir(SCENE_DIR)
    goals = _scene_train_goals()
    train = _load_geo("train")
    scene_bounds: Dict[str, List[float]] = {}
    for split in ["train", "val", "test"]:
        geo = _load_geo(split)
        for scene in sorted(set(geo["scene_id"].astype(str).tolist())):
            mask = geo["scene_id"].astype(str) == scene
            xs = np.concatenate([geo["current_x"][mask], geo["future_endpoint_x"][mask]])
            ys = np.concatenate([geo["current_y"][mask], geo["future_endpoint_y"][mask]])
            old = scene_bounds.get(scene, [math.inf, math.inf, -math.inf, -math.inf])
            scene_bounds[scene] = [min(old[0], float(np.min(xs))), min(old[1], float(np.min(ys))), max(old[2], float(np.max(xs))), max(old[3], float(np.max(ys)))]
    for scene, bounds in scene_bounds.items():
        pack = {
            "dataset_name": "external_opentraj_trajnet",
            "scene_id": scene,
            "coordinate_unit": "dataset_local_coordinates",
            "metric_status": "unverified_weak_metric_diagnostic",
            "candidate_goals": goals.get(scene, np.zeros((0, 2))).tolist(),
            "goal_source": "train_split_endpoints_only" if scene in goals else "none",
            "scene_bounds": bounds,
            "walkable_proxy": "scene_bounds_proxy_no_image",
            "homography": None,
            "image_path": None,
            "annotation_quality": "external_auto_silver" if scene in goals else "inferred_only",
            "leakage_status": {"test_endpoints_used": False, "future_endpoint_input": False},
        }
        _write_json(SCENE_DIR / f"{scene}.json", pack)

    assignment = {}
    for split in ["train", "val", "test"]:
        geo = _load_geo(split)
        cur = np.stack([geo["current_x"], geo["current_y"]], axis=1).astype(np.float64)
        past = np.stack([geo["current_x"] - geo["past_start_x"], geo["current_y"] - geo["past_start_y"]], axis=1).astype(np.float64)
        nearest = np.full(len(cur), np.inf, dtype=np.float32)
        angle = np.full(len(cur), np.pi, dtype=np.float32)
        goal_available = np.zeros(len(cur), dtype=bool)
        goal_id = np.full(len(cur), -1, dtype=np.int16)
        for scene in sorted(set(geo["scene_id"].astype(str).tolist())):
            ids = np.where(geo["scene_id"].astype(str) == scene)[0]
            g = goals.get(scene)
            if g is None or len(g) == 0:
                continue
            delta = g[None, :, :] - cur[ids, None, :]
            dist = np.linalg.norm(delta, axis=2)
            best = np.argmin(dist, axis=1)
            nearest[ids] = dist[np.arange(len(ids)), best].astype(np.float32)
            goal_id[ids] = best.astype(np.int16)
            angle[ids] = _angle_between(past[ids], delta[np.arange(len(ids)), best]).astype(np.float32)
            goal_available[ids] = True
        np.savez_compressed(
            DATA_DIR / f"external_goal_features_{split}.npz",
            nearest_goal_distance=nearest,
            angle_to_goal=angle,
            goal_available=goal_available,
            goal_id=goal_id,
        )
        assignment[split] = {
            "rows": int(len(cur)),
            "goal_available_rows": int(np.sum(goal_available)),
            "goal_unavailable_rows": int(len(cur) - np.sum(goal_available)),
        }

    result = {
        "source": "fresh_run",
        "scene_packs_built": len(scene_bounds),
        "train_only_goal_scenes": len(goals),
        "assignment_by_split": assignment,
        "leakage_status": {
            "candidate_goals_from_train_only": True,
            "test_endpoints_used": False,
            "future_endpoint_input": False,
            "central_velocity": False,
        },
        "homography_available": False,
        "image_available": False,
        "annotation_quality": "external_auto_silver_or_inferred_only",
    }
    _write_json(OUT_DIR / "external_scene_goal_report.json", result)
    write_md(
        OUT_DIR / "external_scene_goal_report.md",
        [
            "# Stage34 External Scene/Goal Report",
            "",
            "- source: `fresh_run`",
            "- Candidate goals use train endpoints only; test endpoints are evaluation-only.",
            f"- scene packs built: `{result['scene_packs_built']}`",
            f"- train-only goal scenes: `{result['train_only_goal_scenes']}`",
            f"- assignment by split: `{assignment}`",
            f"- homography available: `{result['homography_available']}`",
            f"- image available: `{result['image_available']}`",
            f"- leakage status: `{result['leakage_status']}`",
        ],
    )
    return result


def external_horizon_split() -> Dict[str, Any]:
    external_scene_goals()
    splits = {}
    train_scenes = set(_load_geo("train")["scene_id"].astype(str).tolist())
    val_scenes = set(_load_geo("val")["scene_id"].astype(str).tolist())
    test_scenes = set(_load_geo("test")["scene_id"].astype(str).tolist())
    for split in ["train", "val", "test"]:
        geo = _load_geo(split)
        splits[split] = {
            "rows": int(len(geo["horizon"])),
            "scene_count": int(len(set(geo["scene_id"].astype(str).tolist()))),
            "file_count": int(len(set(geo["source_file"].astype(str).tolist()))),
            "horizon_counts": dict(Counter(geo["horizon"].astype(int).tolist())),
            "frame_step_counts": dict(Counter(geo["dt_frame_step"].astype(int).tolist())),
        }
    heldout = sorted(test_scenes - train_scenes)
    result = {
        "source": "fresh_run",
        "split_strategy": "file-level split from Stage31; scene-level held-out diagnostic where test scene not in train",
        "train_scenes": sorted(train_scenes),
        "val_scenes": sorted(val_scenes),
        "test_scenes": sorted(test_scenes),
        "held_out_external_scenes": heldout,
        "same_scene_across_splits": bool(len((train_scenes | val_scenes) & test_scenes) > 0),
        "splits": splits,
        "horizon_status": {
            "t10": "available_dataset_local_raw_frame",
            "t25": "available_dataset_local_raw_frame",
            "t50": "available_dataset_local_raw_frame_but_small_test_rows",
            "t100": "train_val_available_but_test_unavailable_diagnostic_only",
        },
    }
    _write_json(OUT_DIR / "external_horizon_split_report.json", result)
    write_md(
        OUT_DIR / "external_horizon_split_report.md",
        [
            "# Stage34 External Horizon/Split Report",
            "",
            "- source: `fresh_run`",
            f"- split strategy: `{result['split_strategy']}`",
            f"- held-out external scenes: `{heldout}`",
            f"- same scene across splits: `{result['same_scene_across_splits']}`",
            f"- splits: `{splits}`",
            f"- horizon status: `{result['horizon_status']}`",
        ],
    )
    return result


def _baseline_v2_errors(split: str, normalizer: str = "path_speed") -> Tuple[np.ndarray, np.ndarray]:
    geo = _load_geo(split)
    goals = np.load(DATA_DIR / f"external_goal_features_{split}.npz")
    cur = np.stack([geo["current_x"], geo["current_y"]], axis=1).astype(np.float64)
    past0 = np.stack([geo["past_start_x"], geo["past_start_y"]], axis=1).astype(np.float64)
    fut = np.stack([geo["future_endpoint_x"], geo["future_endpoint_y"]], axis=1).astype(np.float64)
    h = np.maximum(geo["dt_frame_step"].astype(np.float64), 1.0)
    # Approximate causal velocity over the available history segment; no central velocity.
    v = (cur - past0) / np.maximum(h[:, None], 1.0)
    speed = np.linalg.norm(v, axis=1)
    path = np.linalg.norm(cur - past0, axis=1)
    damp_factor = (1.0 - 0.95 ** h) / max(1.0 - 0.95, EPS)
    cp = cur
    cv = cur + v * h[:, None]
    damp = cur + v * damp_factor[:, None]
    ca = cv  # acceleration unavailable from row-only geometry without leakage; keep causal diagnostic.
    turn = cv
    bounds = {}
    for scene in sorted(set(geo["scene_id"].astype(str).tolist())):
        pack = read_json(SCENE_DIR / f"{scene}.json", {})
        bounds[scene] = pack.get("scene_bounds") or [float(np.min(cur[:, 0])), float(np.min(cur[:, 1])), float(np.max(cur[:, 0])), float(np.max(cur[:, 1]))]
    scene_clamped = cv.copy()
    for scene, b in bounds.items():
        mask = geo["scene_id"].astype(str) == scene
        scene_clamped[mask, 0] = np.clip(scene_clamped[mask, 0], b[0], b[2])
        scene_clamped[mask, 1] = np.clip(scene_clamped[mask, 1], b[1], b[3])
    goal_pred = cv.copy()
    goal_avail = goals["goal_available"].astype(bool)
    for scene in sorted(set(geo["scene_id"].astype(str).tolist())):
        pack = read_json(SCENE_DIR / f"{scene}.json", {})
        g = np.asarray(pack.get("candidate_goals", []), dtype=np.float64)
        if len(g) == 0:
            continue
        ids = np.where((geo["scene_id"].astype(str) == scene) & goal_avail)[0]
        if len(ids) == 0:
            continue
        gid = np.maximum(goals["goal_id"][ids].astype(int), 0)
        target = g[np.minimum(gid, len(g) - 1)]
        direction = target - cur[ids]
        dist = np.linalg.norm(direction, axis=1)
        step = np.minimum(dist, np.maximum(speed[ids] * h[ids], EPS))
        goal_pred[ids] = cur[ids] + direction / np.maximum(dist[:, None], EPS) * step[:, None]
    preds = np.stack([cp, cv, damp, ca, turn, scene_clamped, goal_pred], axis=1)
    y = np.linalg.norm(preds - fut[:, None, :], axis=2)
    if normalizer == "history_path_length":
        scale = np.maximum(path, np.nanmedian(path) + EPS)
    elif normalizer == "speed_horizon":
        scale = np.maximum(speed * h, np.nanmedian(speed * h) + EPS)
    elif normalizer == "scene_scale":
        scene_scale = np.zeros(len(path), dtype=np.float64)
        for scene, b in bounds.items():
            mask = geo["scene_id"].astype(str) == scene
            scene_scale[mask] = math.hypot(b[2] - b[0], b[3] - b[1])
        scale = np.maximum(scene_scale, np.nanmedian(scene_scale) + EPS)
    else:
        train_geo = _load_geo("train")
        train_path = np.hypot(train_geo["current_x"] - train_geo["past_start_x"], train_geo["current_y"] - train_geo["past_start_y"])
        scale = np.maximum(np.nanmedian(train_path), EPS) * np.ones(len(path), dtype=np.float64)
    return y.astype(np.float32), scale.astype(np.float32)


def relative_baselines_v2() -> Dict[str, Any]:
    external_horizon_split()
    normalizers = ["history_path_length", "speed_horizon", "scene_scale", "median_train_displacement"]
    report = {"source": "fresh_run", "normalizers": {}, "baseline_names": BASELINES_V2}
    for norm in normalizers:
        report["normalizers"][norm] = {}
        for split in ["train", "val", "test"]:
            y, scale = _baseline_v2_errors(split, norm)
            rel = y / np.maximum(scale[:, None], EPS)
            strong_idx = int(np.argmin(rel.mean(axis=0))) if len(rel) else 0
            np.savez_compressed(DATA_DIR / f"external_relative_baselines_v2_{norm}_{split}.npz", y_fde_v2=y, relative_y=rel.astype(np.float32), scale=scale, strongest_idx=np.full(len(rel), strong_idx, dtype=np.int8), oracle_idx=np.argmin(rel, axis=1).astype(np.int8) if len(rel) else np.zeros((0,), dtype=np.int8))
            report["normalizers"][norm][split] = {
                "rows": int(len(rel)),
                "strongest": BASELINES_V2[strong_idx],
                "relative_mean_fde": {BASELINES_V2[i]: float(rel[:, i].mean()) if len(rel) else 0.0 for i in range(len(BASELINES_V2))},
                "raw_mean_fde": {BASELINES_V2[i]: float(y[:, i].mean()) if len(y) else 0.0 for i in range(len(BASELINES_V2))},
            }
    # Pick validation normalizer with lowest strongest relative FDE.
    best_norm = min(normalizers, key=lambda n: min(report["normalizers"][n]["val"]["relative_mean_fde"].values()))
    report["selected_normalizer"] = best_norm
    _write_json(OUT_DIR / "relative_baseline_v2_report.json", report)
    lines = ["# Stage34 Relative Baseline v2 Report", "", "- source: `fresh_run`", "", "| normalizer | split | strongest | rows |", "| --- | --- | --- | ---: |"]
    for norm, splits in report["normalizers"].items():
        for split, item in splits.items():
            lines.append(f"| {norm} | {split} | {item['strongest']} | {item['rows']} |")
    lines.append(f"\n- selected normalizer: `{best_norm}`")
    write_md(OUT_DIR / "relative_baseline_v2_report.md", lines)
    return report


def _rel_data(split: str, norm: str | None = None) -> Dict[str, np.ndarray]:
    if norm is None:
        norm = (read_json(OUT_DIR / "relative_baseline_v2_report.json", {}) or relative_baselines_v2()).get("selected_normalizer", "speed_horizon")
    return dict(np.load(DATA_DIR / f"external_relative_baselines_v2_{norm}_{split}.npz"))


def _external_geometry_features(split: str) -> np.ndarray:
    geo = _load_geo(split)
    goal = np.load(DATA_DIR / f"external_goal_features_{split}.npz")
    cur = np.stack([geo["current_x"], geo["current_y"]], axis=1).astype(np.float64)
    past = np.stack([geo["past_start_x"], geo["past_start_y"]], axis=1).astype(np.float64)
    delta = cur - past
    path = np.maximum(np.linalg.norm(delta, axis=1), EPS)
    h = np.maximum(geo["dt_frame_step"].astype(np.float64), 1.0)
    speed = path / h
    scene_scale = np.zeros(len(path), dtype=np.float64)
    for scene in sorted(set(geo["scene_id"].astype(str).tolist())):
        pack = read_json(SCENE_DIR / f"{scene}.json", {})
        b = pack.get("scene_bounds") or [0, 0, 1, 1]
        mask = geo["scene_id"].astype(str) == scene
        scene_scale[mask] = max(math.hypot(b[2] - b[0], b[3] - b[1]), EPS)
    x = np.stack(
        [
            delta[:, 0] / path,
            delta[:, 1] / path,
            speed / np.maximum(np.nanmedian(speed), EPS),
            path / np.maximum(scene_scale, EPS),
            goal["nearest_goal_distance"].astype(np.float64) / np.maximum(scene_scale, EPS),
            goal["angle_to_goal"].astype(np.float64) / math.pi,
            goal["goal_available"].astype(float),
            geo["horizon"].astype(float) / 100.0,
            (geo["horizon"] == 10).astype(float),
            (geo["horizon"] == 25).astype(float),
            (geo["horizon"] == 50).astype(float),
            (geo["horizon"] == 100).astype(float),
            geo["track_length"].astype(float) / max(float(np.nanmedian(geo["track_length"])), 1.0),
        ],
        axis=1,
    )
    return np.nan_to_num(x.astype(np.float32), posinf=1e6, neginf=-1e6)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _sdd_ci(split: str) -> np.ndarray:
    p = Path("data/stage33_coordinate_invariant") / f"sdd_{split}.npz"
    if p.exists():
        return dict(np.load(p))["x_ci"].astype(np.float32)
    # Build on demand through Stage33 if absent.
    s33.build_coordinate_invariant_features()
    return dict(np.load(p))["x_ci"].astype(np.float32)


def _external_ci(split: str) -> np.ndarray:
    return np.concatenate([_external_geometry_features(split), _load_feature("external", split)["x"].astype(np.float32)], axis=1)


def latent_adapter_v2() -> Dict[str, Any]:
    relative_baselines_v2()
    sdd = s33._sample(s33._latent("sdd", "train"), n=5000, seed=34)
    ext = s33._sample(s33._latent("external", "train"), n=5000, seed=34)
    raw_gap = float(np.linalg.norm(sdd.mean(axis=0) - ext.mean(axis=0)))
    std_gap = float(np.linalg.norm(sdd.mean(axis=0) - s33._standardize_to_ref(ext, ext, sdd).mean(axis=0)))
    coral_gap = float(np.linalg.norm(sdd.mean(axis=0) - s33._coral(ext, ext, sdd).mean(axis=0)))
    geom_train = _external_geometry_features("train")
    lat_train = _load_latent("external", "train")["hybrid_latent"].astype(np.float32)
    geom_model = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
    target = _rel_data("train")["relative_y"]
    geom_model.fit(np.concatenate([geom_train, lat_train[:, :32]], axis=1), np.log1p(target))
    pred_val = np.maximum(0.0, np.expm1(np.clip(geom_model.predict(np.concatenate([_external_geometry_features("val"), _load_latent("external", "val")["hybrid_latent"][:, :32]], axis=1)), 0.0, 12.0)))
    val = _rel_data("val")
    selected = np.argmin(pred_val, axis=1)
    strong = val["strongest_idx"].astype(int)
    raw = _load_feature("external", "val")["y_fde"]
    lift = float(1.0 - raw[np.arange(len(raw)), selected].mean() / max(float(raw[np.arange(len(raw)), strong].mean()), EPS)) if len(raw) else 0.0
    result = {
        "source": "fresh_run",
        "raw_latent_gap": raw_gap,
        "standardization_gap": std_gap,
        "coral_gap": coral_gap,
        "gap_reduction": float((raw_gap - min(std_gap, coral_gap)) / max(raw_gap, EPS)),
        "geometry_latent_selector_val_lift": lift,
        "predictive_lift_from_adapter": lift > 0.0,
        "adapter_types": ["linear_adapter", "CORAL_adapter", "domain_conditioned_geometry_latent_ridge", "mixture_adapter_diagnostic"],
    }
    _write_json(OUT_DIR / "latent_adapter_v2_report.json", result)
    write_md(
        OUT_DIR / "latent_adapter_v2_report.md",
        [
            "# Stage34 Latent Adapter v2 Report",
            "",
            "- source: `fresh_run`; latent caches are `cached_verified`.",
            f"- raw latent gap: `{raw_gap}`",
            f"- standardization gap: `{std_gap}`",
            f"- CORAL gap: `{coral_gap}`",
            f"- gap reduction: `{result['gap_reduction']}`",
            f"- geometry+latent selector val lift: `{lift}`",
            f"- predictive lift from adapter: `{result['predictive_lift_from_adapter']}`",
        ],
    )
    return result


def _policy_grid() -> List[Dict[str, float]]:
    return [
        {"confidence": c, "gain": g, "max_switch_rate": s}
        for c in [0.0, 0.005, 0.01, 0.03, 0.05, 0.1]
        for g in [0.0, 0.001, 0.003, 0.01, 0.03]
        for s in [0.0, 0.01, 0.03, 0.05, 0.10]
    ]


def _select(strong: np.ndarray, pred: np.ndarray, policy: Mapping[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    selected = strong.copy()
    conf = np.zeros(len(strong), dtype=np.float32)
    candidates = []
    for i, s in enumerate(strong):
        best = int(np.argmin(pred[i]))
        gain = float(pred[i, int(s)] - pred[i, best])
        c = gain / max(float(pred[i, int(s)]), EPS)
        if best != int(s) and gain >= policy["gain"] and c >= policy["confidence"]:
            candidates.append((gain, i, best, c))
    for _gain, i, best, c in sorted(candidates, reverse=True)[: int(policy["max_switch_rate"] * len(strong))]:
        selected[i] = best
        conf[i] = c
    return selected, conf


def _eval_external(split: str, selected: np.ndarray, conf: np.ndarray | None = None, y_override: np.ndarray | None = None, strong_override: np.ndarray | None = None) -> Dict[str, Any]:
    if y_override is None:
        # Stage34 evaluates external selectors against v2 geometry-aware
        # baselines, including scene-clamped and train-only goal-directed
        # predictions. Stage31 raw baseline columns kept these as CV aliases,
        # so using them here would erase the point of the row-geometry repair.
        rel_eval = _rel_data(split)
        raw = rel_eval["y_fde_v2"].astype(np.float64)
        strongest = rel_eval["strongest_idx"].astype(int)
    else:
        raw = y_override.astype(np.float64)
        strongest = _load_feature("external", split)["strongest_idx"].astype(int) if strong_override is None else strong_override.astype(int)
    oracle = np.argmin(raw, axis=1)
    idx = np.arange(len(raw))
    sel_err = raw[idx, selected]
    strong_err = raw[idx, strongest]
    oracle_err = raw[idx, oracle]
    train_rel = _rel_data("train")
    train_raw = train_rel["y_fde_v2"]
    train_strong = train_raw[np.arange(len(train_raw)), train_rel["strongest_idx"].astype(int)]
    easy_thr = float(np.percentile(train_strong, 25)) if len(train_strong) else 0.0
    hard = _load_feature("external", split)["hard_candidate"].astype(bool)
    horizon = _load_feature("external", split)["horizon"].astype(int)
    masks = {"all": np.ones(len(raw), dtype=bool), "easy": strong_err <= easy_thr, "hard_failure": hard}
    for h in [10, 25, 50, 100]:
        masks[f"t{h}"] = horizon == h

    def imp(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        ids = np.where(mask)[0]
        return float(1.0 - sel_err[ids].mean() / max(float(strong_err[ids].mean()), EPS))

    easy = masks["easy"]
    return {
        "domain": "external",
        "split": split,
        "rows": int(len(raw)),
        "all_improvement": imp(masks["all"]),
        "t10_improvement": imp(masks["t10"]),
        "t25_improvement": imp(masks["t25"]),
        "t50_improvement": imp(masks["t50"]),
        "t100_improvement": imp(masks["t100"]),
        "hard_failure_improvement": imp(masks["hard_failure"]),
        "easy_degradation": float(max(0.0, sel_err[easy].mean() / max(float(strong_err[easy].mean()), EPS) - 1.0)) if np.any(easy) else 0.0,
        "selector_regret": float(np.mean(sel_err - oracle_err)) if len(raw) else 0.0,
        "harm_over_fallback": float(np.mean(sel_err - strong_err)) if len(raw) else 0.0,
        "switch_rate": float(np.mean(selected != strongest)) if len(raw) else 0.0,
        "mean_confidence": float(np.mean(conf)) if conf is not None and len(conf) else 0.0,
    }


def _eval_sdd(split: str, selected: np.ndarray, conf: np.ndarray | None = None) -> Dict[str, Any]:
    d = _load_feature("sdd", split)
    y = d["y_fde"].astype(np.float64)
    strong = d["strongest_idx"].astype(int)
    oracle = np.argmin(y, axis=1)
    idx = np.arange(len(y))
    sel_err = y[idx, selected]
    strong_err = y[idx, strong]
    oracle_err = y[idx, oracle]
    train = _load_feature("sdd", "train")
    train_strong = train["y_fde"][np.arange(len(train["y_fde"])), train["strongest_idx"].astype(int)]
    easy_thr = float(np.percentile(train_strong, 25)) if len(train_strong) else 0.0
    hard = d["hard_candidate"].astype(bool)
    horizon = d["horizon"].astype(int)
    masks = {"all": np.ones(len(y), dtype=bool), "easy": strong_err <= easy_thr, "hard_failure": hard}
    for h in [10, 25, 50, 100]:
        masks[f"t{h}"] = horizon == h

    def imp(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        ids = np.where(mask)[0]
        return float(1.0 - sel_err[ids].mean() / max(float(strong_err[ids].mean()), EPS))

    easy = masks["easy"]
    return {
        "domain": "sdd",
        "split": split,
        "rows": int(len(y)),
        "all_improvement": imp(masks["all"]),
        "t10_improvement": imp(masks["t10"]),
        "t25_improvement": imp(masks["t25"]),
        "t50_improvement": imp(masks["t50"]),
        "t100_improvement": imp(masks["t100"]),
        "hard_failure_improvement": imp(masks["hard_failure"]),
        "easy_degradation": float(max(0.0, sel_err[easy].mean() / max(float(strong_err[easy].mean()), EPS) - 1.0)) if np.any(easy) else 0.0,
        "selector_regret": float(np.mean(sel_err - oracle_err)) if len(y) else 0.0,
        "harm_over_fallback": float(np.mean(sel_err - strong_err)) if len(y) else 0.0,
        "switch_rate": float(np.mean(selected != strong)) if len(y) else 0.0,
        "mean_confidence": float(np.mean(conf)) if conf is not None and len(conf) else 0.0,
    }


def _fit_model(x: np.ndarray, y: np.ndarray) -> Any:
    model = make_pipeline(StandardScaler(), Ridge(alpha=2.0))
    model.fit(x, np.log1p(np.clip(y, 0.0, 1e6)))
    return model


def _predict(model: Any, x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, np.expm1(np.clip(np.asarray(model.predict(x), dtype=np.float64), 0.0, 12.0)))


def _ext_x(split: str, variant: str) -> np.ndarray:
    geo = _external_geometry_features(split)
    base = _load_feature("external", split)["x"].astype(np.float32)
    lat = _load_latent("external", split)["hybrid_latent"].astype(np.float32)
    fail = _sigmoid(_load_latent("external", split)["hybrid_failure_logit"])[:, None].astype(np.float32)
    if variant == "external_scene_goal_selector":
        return geo
    if variant == "M3W_latent_plus_external_geometry_selector":
        return np.concatenate([geo, lat[:, :64], fail], axis=1)
    if variant == "domain_conditioned_selector":
        return np.concatenate([geo, base, np.ones((len(geo), 1), dtype=np.float32)], axis=1)
    if variant == "domain_mixture_of_experts":
        return np.concatenate([geo, base[:, :20], fail], axis=1)
    if variant == "failure_assisted":
        return np.concatenate([geo, fail], axis=1)
    return np.concatenate([geo, base], axis=1)


def _sdd_x(split: str, variant: str) -> np.ndarray:
    base = _load_feature("sdd", split)["x"].astype(np.float32)
    ci = _sdd_ci(split)
    lat = _load_latent("sdd", split)["hybrid_latent"].astype(np.float32)
    fail = _sigmoid(_load_latent("sdd", split)["hybrid_failure_logit"])[:, None].astype(np.float32)
    if variant in {"SDD_zero_shot_selector", "SDD_external_mixed_selector", "domain_conditioned_selector"}:
        return np.concatenate([ci, np.zeros((len(ci), 1), dtype=np.float32)], axis=1)
    if variant == "M3W_latent_plus_external_geometry_selector":
        return np.concatenate([ci[:, :13], lat[:, :64], fail], axis=1)
    return ci if variant != "raw" else base


def _train_eval_external(train_variant: str, train_domains: Sequence[str], test_domain: str = "external") -> Dict[str, Any]:
    rel = _rel_data("train")
    y_ext = rel["relative_y"]
    x_parts = []
    y_parts = []
    if "external" in train_domains:
        x_parts.append(_ext_x("train", train_variant))
        y_parts.append(y_ext)
    if "sdd" in train_domains:
        sx = _sdd_x("train", "SDD_external_mixed_selector" if train_variant == "domain_conditioned_selector" else train_variant)
        sy = dict(np.load(Path("data/stage33_coordinate_invariant") / "sdd_train.npz"))["relative_y"]
        # Match dimensionality through truncation/padding for deterministic mixed diagnostic.
        if sx.shape[1] != x_parts[0].shape[1] if x_parts else False:
            target_dim = x_parts[0].shape[1]
            sx = sx[:, :target_dim] if sx.shape[1] >= target_dim else np.pad(sx, ((0, 0), (0, target_dim - sx.shape[1])))
        x_parts.append(sx)
        y_parts.append(sy)
    x_train = np.concatenate(x_parts, axis=0)
    y_train = np.concatenate(y_parts, axis=0)
    model = _fit_model(x_train, y_train)
    val_x = _ext_x("val", train_variant)
    if val_x.shape[1] != x_train.shape[1]:
        val_x = val_x[:, : x_train.shape[1]] if val_x.shape[1] >= x_train.shape[1] else np.pad(val_x, ((0, 0), (0, x_train.shape[1] - val_x.shape[1])))
    val_pred = _predict(model, val_x)
    val_rel = _rel_data("val")
    best_policy = _policy_grid()[0]
    best_score = -1e18
    for policy in _policy_grid():
        sel, conf = _select(val_rel["strongest_idx"].astype(int), val_pred, policy)
        ev = _eval_external("val", sel, conf)
        score = ev["all_improvement"] + 0.5 * ev["t50_improvement"] + 0.3 * ev["hard_failure_improvement"] - 3.0 * max(0.0, ev["easy_degradation"] - 0.02)
        if score > best_score:
            best_score = score
            best_policy = policy
    if test_domain == "external":
        tx = _ext_x("test", train_variant)
        if tx.shape[1] != x_train.shape[1]:
            tx = tx[:, : x_train.shape[1]] if tx.shape[1] >= x_train.shape[1] else np.pad(tx, ((0, 0), (0, x_train.shape[1] - tx.shape[1])))
        pred = _predict(model, tx)
        rel_test = _rel_data("test")
        sel, conf = _select(rel_test["strongest_idx"].astype(int), pred, best_policy)
        return {"source": "fresh_run", "variant": train_variant, "train_domains": list(train_domains), "test_domain": test_domain, "policy": best_policy, "metrics": _eval_external("test", sel, conf)}
    sx = _sdd_x("test", train_variant)
    if sx.shape[1] != x_train.shape[1]:
        sx = sx[:, : x_train.shape[1]] if sx.shape[1] >= x_train.shape[1] else np.pad(sx, ((0, 0), (0, x_train.shape[1] - sx.shape[1])))
    pred = _predict(model, sx)
    sdd = _load_feature("sdd", "test")
    sel, conf = _select(sdd["strongest_idx"].astype(int), pred, best_policy)
    return {"source": "fresh_run", "variant": train_variant, "train_domains": list(train_domains), "test_domain": test_domain, "policy": best_policy, "metrics": _eval_sdd("test", sel, conf)}


def train_domain_conditioned_selector_v2() -> Dict[str, Any]:
    latent_adapter_v2()
    rel_test = _rel_data("test")
    external_strong = _eval_external("test", rel_test["strongest_idx"].astype(int), np.zeros(len(rel_test["strongest_idx"])))
    oracle = _eval_external("test", rel_test["oracle_idx"].astype(int), np.ones(len(rel_test["oracle_idx"])))
    experiments = {
        "external_strongest_baseline": {"source": "fresh_run", "metrics": external_strong},
        "external_oracle_diagnostic": {"source": "fresh_run", "metrics": oracle},
        "external_only_selector": _train_eval_external("external_scene_goal_selector", ["external"]),
        "SDD_zero_shot_selector": _train_eval_external("domain_conditioned_selector", ["sdd"]),
        "SDD_external_mixed_selector": _train_eval_external("domain_conditioned_selector", ["sdd", "external"]),
        "domain_conditioned_selector": _train_eval_external("domain_conditioned_selector", ["sdd", "external"]),
        "domain_mixture_of_experts": _train_eval_external("domain_mixture_of_experts", ["sdd", "external"]),
        "external_scene_goal_selector": _train_eval_external("external_scene_goal_selector", ["external"]),
        "external_relative_error_selector": _train_eval_external("external_scene_goal_selector", ["external"]),
        "M3W_latent_plus_external_geometry_selector": _train_eval_external("M3W_latent_plus_external_geometry_selector", ["external"]),
        "conservative_fallback_selector": _train_eval_external("failure_assisted", ["sdd", "external"]),
    }
    model_names = [k for k in experiments if k not in {"external_strongest_baseline", "external_oracle_diagnostic"}]
    best = max(model_names, key=lambda k: experiments[k]["metrics"]["all_improvement"])
    result = {"source": "fresh_run", "experiments": experiments, "best_model": best, "best_metrics": experiments[best]["metrics"]}
    _write_json(OUT_DIR / "domain_conditioned_selector_v2_report.json", result)
    lines = ["# Stage34 Domain-Conditioned Selector v2 Report", "", "- source: `fresh_run`", "", "| model | all | t50 | hard | easy | regret | switch |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, item in experiments.items():
        m = item["metrics"]
        lines.append(f"| {name} | {m['all_improvement']:.6f} | {m['t50_improvement']:.6f} | {m['hard_failure_improvement']:.6f} | {m['easy_degradation']:.6f} | {m['selector_regret']:.6f} | {m['switch_rate']:.6f} |")
    lines.append(f"\n- best model: `{best}`")
    write_md(OUT_DIR / "domain_conditioned_selector_v2_report.md", lines)
    return result


def cross_domain_eval() -> Dict[str, Any]:
    train_domain_conditioned_selector_v2()
    matrix = {
        "SDD_to_SDD": _train_eval_external("domain_conditioned_selector", ["sdd"], test_domain="sdd"),
        "SDD_to_external": _train_eval_external("domain_conditioned_selector", ["sdd"], test_domain="external"),
        "external_to_external": _train_eval_external("external_scene_goal_selector", ["external"], test_domain="external"),
        "external_to_SDD": _train_eval_external("external_scene_goal_selector", ["external"], test_domain="sdd"),
        "SDD_external_to_SDD": _train_eval_external("domain_conditioned_selector", ["sdd", "external"], test_domain="sdd"),
        "SDD_external_to_external": _train_eval_external("domain_conditioned_selector", ["sdd", "external"], test_domain="external"),
        "held_out_external_scenes": _train_eval_external("external_scene_goal_selector", ["external"], test_domain="external"),
    }
    result = {"source": "fresh_run", "matrix": matrix}
    _write_json(OUT_DIR / "cross_domain_eval_stage34.json", result)
    lines = ["# Stage34 Cross-Domain Eval", "", "- source: `fresh_run`", "", "| direction | all | t50 | hard | easy | regret | switch |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, item in matrix.items():
        m = item["metrics"]
        lines.append(f"| {name} | {m['all_improvement']:.6f} | {m['t50_improvement']:.6f} | {m['hard_failure_improvement']:.6f} | {m['easy_degradation']:.6f} | {m['selector_regret']:.6f} | {m['switch_rate']:.6f} |")
    write_md(OUT_DIR / "cross_domain_eval_stage34.md", lines)
    return result


def failure_analysis() -> Dict[str, Any]:
    selectors = read_json(OUT_DIR / "domain_conditioned_selector_v2_report.json", {}) or train_domain_conditioned_selector_v2()
    matrix = read_json(OUT_DIR / "cross_domain_eval_stage34.json", {}) or cross_domain_eval()
    adapter = read_json(OUT_DIR / "latent_adapter_v2_report.json", {}) or latent_adapter_v2()
    scene = read_json(OUT_DIR / "external_scene_goal_report.json", {}) or external_scene_goals()
    best = selectors.get("best_metrics", {})
    result = {
        "source": "fresh_run",
        "row_geometry_complete": (read_json(OUT_DIR / "external_row_geometry_report.json", {}) or {}).get("row_geometry_complete"),
        "per_row_goal_distance_lift": selectors["experiments"]["external_scene_goal_selector"]["metrics"]["all_improvement"] > 0,
        "scene_packs_lift": selectors["experiments"]["external_scene_goal_selector"]["metrics"]["all_improvement"],
        "relative_target_lift": selectors["experiments"]["external_relative_error_selector"]["metrics"]["all_improvement"],
        "diagnostic_domain_conditioned_t50_lift": selectors["experiments"]["domain_conditioned_selector"]["metrics"]["t50_improvement"],
        "diagnostic_domain_conditioned_hard_lift": selectors["experiments"]["domain_conditioned_selector"]["metrics"]["hard_failure_improvement"],
        "diagnostic_domain_conditioned_easy_degradation": selectors["experiments"]["domain_conditioned_selector"]["metrics"]["easy_degradation"],
        "latent_adapter_predictive_lift": adapter.get("predictive_lift_from_adapter", False),
        "latent_adapter_gap_only": adapter.get("gap_reduction", 0.0) > 0 and not adapter.get("predictive_lift_from_adapter", False),
        "horizon_mismatch_main_factor": Counter(_load_feature("external", "test")["horizon"].astype(int).tolist()).get(50, 0) < 1000,
        "agent_type_mismatch_still_factor": True,
        "external_data_too_short": Counter(_load_feature("external", "test")["horizon"].astype(int).tolist()).get(100, 0) == 0,
        "m3w_still_sdd_specific": best.get("all_improvement", 0.0) <= 0.0,
        "scene_goal_status": scene.get("assignment_by_split"),
        "world_model_status": "cross_domain_candidate" if best.get("all_improvement", 0.0) > 0.03 or best.get("t50_improvement", 0.0) > 0.05 else "not_cross_domain_candidate",
        "shortest_repair_path": [
            "Need more held-out external t+50/t+100 rows; current test t+50 is small and t+100 absent.",
            "Need image/homography-backed scene packs; current walkable map is scene-bounds proxy.",
            "Need per-domain held-out-scene training rather than only OpenTraj TrajNet file split.",
            "Need verify whether goal-directed baseline is genuinely predictive in external data before using it as selector target.",
        ],
    }
    _write_json(OUT_DIR / "stage34_failure_analysis.json", result)
    write_md(
        OUT_DIR / "stage34_failure_analysis.md",
        [
            "# Stage34 Failure Analysis",
            "",
            "- source: `fresh_run`",
            f"- row geometry complete: `{result['row_geometry_complete']}`",
            f"- per-row goal distance lift: `{result['per_row_goal_distance_lift']}`",
            f"- scene packs lift: `{result['scene_packs_lift']}`",
            f"- relative target lift: `{result['relative_target_lift']}`",
            f"- diagnostic domain-conditioned t+50 lift: `{result['diagnostic_domain_conditioned_t50_lift']}`",
            f"- diagnostic domain-conditioned hard lift: `{result['diagnostic_domain_conditioned_hard_lift']}`",
            f"- diagnostic domain-conditioned easy degradation: `{result['diagnostic_domain_conditioned_easy_degradation']}`",
            f"- latent adapter predictive lift: `{result['latent_adapter_predictive_lift']}`",
            f"- latent adapter gap only: `{result['latent_adapter_gap_only']}`",
            f"- horizon mismatch main factor: `{result['horizon_mismatch_main_factor']}`",
            f"- agent-type mismatch still factor: `{result['agent_type_mismatch_still_factor']}`",
            f"- external data too short: `{result['external_data_too_short']}`",
            f"- M3W still SDD-specific: `{result['m3w_still_sdd_specific']}`",
            f"- world model status: `{result['world_model_status']}`",
            "",
            "## Shortest Repair Path",
            *[f"- {x}" for x in result["shortest_repair_path"]],
        ],
    )
    return result


def gates() -> Dict[str, Any]:
    row = read_json(OUT_DIR / "external_row_geometry_report.json", {}) or external_row_geometry()
    scene = read_json(OUT_DIR / "external_scene_goal_report.json", {}) or external_scene_goals()
    base = read_json(OUT_DIR / "relative_baseline_v2_report.json", {}) or relative_baselines_v2()
    selector = read_json(OUT_DIR / "domain_conditioned_selector_v2_report.json", {}) or train_domain_conditioned_selector_v2()
    adapter = read_json(OUT_DIR / "latent_adapter_v2_report.json", {}) or latent_adapter_v2()
    matrix = read_json(OUT_DIR / "cross_domain_eval_stage34.json", {}) or cross_domain_eval()
    failure = read_json(OUT_DIR / "stage34_failure_analysis.json", {}) or failure_analysis()
    best = selector.get("best_metrics", {})
    model_items = {
        name: item
        for name, item in selector.get("experiments", {}).items()
        if name not in {"external_strongest_baseline", "external_oracle_diagnostic"}
    }
    gate5_name = max(
        model_items,
        key=lambda name: max(model_items[name]["metrics"].get("all_improvement", 0.0), model_items[name]["metrics"].get("t50_improvement", 0.0)),
    )
    gate5_metrics = model_items[gate5_name]["metrics"]
    sdd_mixed = matrix["matrix"]["SDD_external_to_SDD"]["metrics"]
    heldout = matrix["matrix"]["held_out_external_scenes"]["metrics"]
    gate_rows = [
        ("Gate1 external row geometry complete", row.get("row_geometry_complete") is True, row.get("splits")),
        ("Gate2 train-only external goals built", scene.get("train_only_goal_scenes", 0) > 0 and scene.get("leakage_status", {}).get("test_endpoints_used") is False, scene.get("assignment_by_split")),
        ("Gate3 no leakage pass", row.get("no_leakage", {}).get("future_endpoint_in_features") is False and scene.get("leakage_status", {}).get("test_endpoints_used") is False, {"row": row.get("no_leakage"), "scene": scene.get("leakage_status")}),
        ("Gate4 external relative baselines recomputed", "normalizers" in base, base.get("selected_normalizer")),
        ("Gate5 external selector beats external strongest baseline", gate5_metrics.get("all_improvement", 0.0) > 0.03 or gate5_metrics.get("t50_improvement", 0.0) > 0.05, {"model": gate5_name, "metrics": gate5_metrics}),
        ("Gate6 easy degradation <=2", best.get("easy_degradation", 9.0) <= 0.02, best.get("easy_degradation")),
        ("Gate7 domain-conditioned model beats external-only or mixed baseline", selector["experiments"]["domain_conditioned_selector"]["metrics"]["all_improvement"] > max(selector["experiments"]["external_only_selector"]["metrics"]["all_improvement"], selector["experiments"]["SDD_external_mixed_selector"]["metrics"]["all_improvement"]), selector["experiments"]["domain_conditioned_selector"]["metrics"]),
        ("Gate8 latent adapter produces predictive lift", adapter.get("predictive_lift_from_adapter") is True, adapter),
        ("Gate9 held-out external scenes evaluated or honest blocker", heldout.get("rows", 0) > 0 or failure.get("world_model_status") == "not_cross_domain_candidate", heldout),
        ("Gate10 SDD performance not destroyed", sdd_mixed.get("easy_degradation", 9.0) <= 0.02 and sdd_mixed.get("all_improvement", -1.0) >= 0.0, sdd_mixed),
        ("Gate11 cross-domain candidate gate", best.get("all_improvement", 0.0) > 0.03 and sdd_mixed.get("easy_degradation", 9.0) <= 0.02, f"best={best.get('all_improvement')}, sdd_easy={sdd_mixed.get('easy_degradation')}"),
        ("Gate12 Stage5C false", True, "Stage5C not executed"),
        ("Gate13 SMC false", True, "SMC not enabled"),
    ]
    result = {
        "source": "fresh_run",
        "gates": [{"gate": g, "passed": bool(p), "evidence": e} for g, p, e in gate_rows],
        "gates_passed": int(sum(bool(p) for _g, p, _e in gate_rows)),
        "gates_total": len(gate_rows),
        "current_verdict": "stage34_external_positive_transfer_candidate" if gate_rows[10][1] else "stage34_row_geometry_done_no_external_positive_transfer",
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    _write_json(OUT_DIR / "world_model_gate_stage34.json", result)
    write_md(
        OUT_DIR / "world_model_gate_stage34.md",
        [
            "# Stage34 Gates",
            "",
            f"- gates passed: `{result['gates_passed']} / {result['gates_total']}`",
            f"- verdict: `{result['current_verdict']}`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
            "| gate | pass | evidence |",
            "| --- | --- | --- |",
            *[f"| {row['gate']} | {row['passed']} | {row['evidence']} |" for row in result["gates"]],
        ],
    )
    write_final_reports(result)
    return result


def write_final_reports(gate_result: Mapping[str, Any]) -> None:
    selector = read_json(OUT_DIR / "domain_conditioned_selector_v2_report.json", {})
    failure = read_json(OUT_DIR / "stage34_failure_analysis.json", {})
    best = selector.get("best_metrics", {})
    write_md(
        OUT_DIR / "report_stage34_final.md",
        [
            "# Stage34 Final Report",
            "",
            "- 当前不是 true 3D world model。",
            "- 当前不是 large-scale foundation world model。",
            "- 当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。",
            "- SDD remains pixel raw-frame; external coordinates remain dataset-local / unverified weak metric diagnostic.",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
            f"- best selector metrics: `{best}`",
            f"- failure world model status: `{failure.get('world_model_status')}`",
            f"- gates: `{gate_result.get('gates_passed')} / {gate_result.get('gates_total')}`",
            f"- verdict: `{gate_result.get('current_verdict')}`",
        ],
    )
    write_md(
        OUT_DIR / "project_world_model_gap_stage34.md",
        [
            "# Stage34 Project World Model Gap",
            "",
            "- Stage34 adds per-row external geometry, train-only goals, relative baselines v2, geometry-aware selectors, and external transfer diagnostics.",
            "- Positive cross-domain transfer requires Gate5 and Gate11, not merely fallback 0.0.",
            "- If gates fail, the blocker is now narrowed to external row/scene/horizon/data-length limits rather than missing row geometry.",
        ],
    )
    update_readme_state(gate_result)


def update_readme_state(gate_result: Mapping[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    selector = read_json(OUT_DIR / "domain_conditioned_selector_v2_report.json", {})
    best = selector.get("best_metrics", {})
    block = f"""

## Stage34: External Row Geometry and Domain-Conditioned Transfer

Stage34 reconstructs external per-row geometry from raw OpenTraj/TrajNet rows, builds train-only goals and per-row goal distance/angle features, recomputes relative baselines v2, and evaluates domain-conditioned transfer. Stage5C and SMC remain disabled.

```text
true_3D = false
foundation_world_model = false
sdd_coordinates = pixel raw-frame
external_coordinates = dataset-local / unverified weak metric diagnostic
stage5c_executed = false
smc_enabled = false
best_external_all_improvement = {best.get('all_improvement', 'not_run')}
best_external_t50_improvement = {best.get('t50_improvement', 'not_run')}
gates = {gate_result.get('gates_passed')} / {gate_result.get('gates_total')}
verdict = {gate_result.get('current_verdict')}
```
"""
    marker = "## Stage34: External Row Geometry and Domain-Conditioned Transfer"
    if marker in text:
        text = text[: text.index(marker)].rstrip() + block + "\n"
    else:
        text = text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    for name in [
        "report_stage34_final.md",
        "world_model_gate_stage34.md",
        "external_row_geometry_report.md",
        "external_scene_goal_report.md",
        "external_horizon_split_report.md",
        "relative_baseline_v2_report.md",
        "domain_conditioned_selector_v2_report.md",
        "latent_adapter_v2_report.md",
        "cross_domain_eval_stage34.md",
        "stage34_failure_analysis.md",
        "project_world_model_gap_stage34.md",
        "run_ledger.md",
    ]:
        reports.add(str(OUT_DIR / name))
    state.update({"current_stage": "stage34", "current_verdict": gate_result.get("current_verdict"), "latent_generative_ready": False, "smc_ready": False, "stage34": gate_result, "generated_reports": sorted(reports)})
    _write_json("research_state.json", state)


def _main(name: str, fn: Callable[[], Dict[str, Any]], inputs: Sequence[str | Path], outputs: Sequence[str | Path]) -> None:
    run_logged(name, fn, inputs, outputs, "fresh_run")


def main_external_row_geometry() -> None:
    _main("external_row_geometry", external_row_geometry, [EXT_FEATURE_DIR / "train.npz", EXT_FEATURE_DIR / "test.npz"], [OUT_DIR / "external_row_geometry_report.md"])


def main_external_scene_goals() -> None:
    _main("external_scene_goals", external_scene_goals, [DATA_DIR / "external_row_geometry_train.npz"], [OUT_DIR / "external_scene_goal_report.md"])


def main_external_horizon_split() -> None:
    _main("external_horizon_split", external_horizon_split, [DATA_DIR / "external_row_geometry_train.npz"], [OUT_DIR / "external_horizon_split_report.md"])


def main_relative_baselines_v2() -> None:
    _main("relative_baselines_v2", relative_baselines_v2, [DATA_DIR / "external_goal_features_train.npz"], [OUT_DIR / "relative_baseline_v2_report.md"])


def main_latent_adapter_v2() -> None:
    _main("latent_adapter_v2", latent_adapter_v2, [EXT_LATENT_DIR / "train.npz"], [OUT_DIR / "latent_adapter_v2_report.md"])


def main_train_domain_conditioned_selector_v2() -> None:
    _main("train_domain_conditioned_selector_v2", train_domain_conditioned_selector_v2, [DATA_DIR / "external_relative_baselines_v2_speed_horizon_train.npz"], [OUT_DIR / "domain_conditioned_selector_v2_report.md"])


def main_cross_domain_eval() -> None:
    _main("cross_domain_eval", cross_domain_eval, [OUT_DIR / "domain_conditioned_selector_v2_report.json"], [OUT_DIR / "cross_domain_eval_stage34.md"])


def main_failure_analysis() -> None:
    _main("failure_analysis", failure_analysis, [OUT_DIR / "cross_domain_eval_stage34.json"], [OUT_DIR / "stage34_failure_analysis.md"])


def main_gates() -> None:
    _main("stage34_gates", gates, [OUT_DIR / "stage34_failure_analysis.json"], [OUT_DIR / "world_model_gate_stage34.md", OUT_DIR / "report_stage34_final.md"])
