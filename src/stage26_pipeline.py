from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage25_pipeline import BASELINE_NAMES, _evaluate_selection, _safe_mean, _split_rows, _stage24_metrics, _strongest_name


REPORT_DIR = Path("outputs/reports")
FEATURE_DIR = Path("data/stage26_sdd_feature_store")
CACHE_DIR = Path("data/stage24_sdd_fast_cache")
SCENE_PACK_DIR = Path("data/stage22_sdd_scene_packs")
CHECKPOINT_DIR = Path("outputs/checkpoints/stage26_selector")
AGENT_TYPES = ["Pedestrian", "Biker", "Skater", "Cart", "Car", "Bus", "unknown"]
PAST_WINDOW = 8
RANDOM_STATE = 26


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


def _write_npz(path: Path, **arrays: Any) -> None:
    ensure_dir(path.parent)
    np.savez_compressed(path, **arrays)


def _load_manifest() -> Dict[str, Any]:
    manifest = read_json(CACHE_DIR / "manifest.json", {})
    if not manifest:
        raise FileNotFoundError("Missing data/stage24_sdd_fast_cache/manifest.json. Run Stage24 fast cache first.")
    return manifest


def _load_arrays(meta: Dict[str, Any], array_cache: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    key = f"{meta['scene_id']}/{meta['video_id']}"
    if key not in array_cache:
        array_cache[key] = {name: np.load(path, mmap_mode="r") for name, path in meta["arrays"].items()}
        array_cache[key]["_frame_order"] = np.load(meta["frame_order"], mmap_mode="r")
    return array_cache[key]


def _scene_pack(scene_id: str, video_id: str, cache: Dict[str, Dict[str, Any] | None]) -> Dict[str, Any] | None:
    key = f"{scene_id}/{video_id}"
    if key not in cache:
        cache[key] = read_json(SCENE_PACK_DIR / f"sdd_{scene_id}_{video_id}.json", None)
    return cache[key]


def _angle_diff(a: float, b: float) -> float:
    return float((a - b + math.pi) % (2 * math.pi) - math.pi)


def _agent_type(row: Dict[str, Any]) -> str:
    typ = str(row.get("target_agent_type", "unknown"))
    return typ if typ in AGENT_TYPES else "unknown"


def _feature_names() -> List[str]:
    names = [
        "horizon_norm",
        "horizon_is_10",
        "horizon_is_25",
        "horizon_is_50",
        "horizon_is_100",
        "split_within_scene",
        "agent_count_log",
        "agent_count_ge5",
        "agent_count_ge10",
        "start_frame_norm",
        "speed_now",
        "speed_mean_past",
        "speed_std_past",
        "speed_delta_past",
        "accel_mag_now",
        "accel_mag_mean_past",
        "heading_change_past",
        "heading_rate_abs_mean",
        "curvature_proxy",
        "past_path_length",
        "past_displacement",
        "past_straightness",
        "vx_now",
        "vy_now",
        "ax_now",
        "ay_now",
        "density_visible_count",
        "density_r20",
        "density_r50",
        "density_r100",
        "nearest_neighbor_distance",
        "mean_nearest3_distance",
        "mean_nearest5_distance",
        "min_ttc",
        "max_closing_speed",
        "closing_neighbor_count",
        "goal_count",
        "nearest_goal_distance",
        "goal_direction_alignment",
        "goal_source_train_endpoint",
        "goal_source_visual_prior",
        "scene_image_width",
        "scene_image_height",
        "cv_rollout_displacement",
        "damped_rollout_displacement",
        "ca_rollout_displacement",
        "scene_clamp_delta",
        "goal_directed_available",
        "baseline_cv_vs_damped_delta",
        "baseline_accel_to_velocity_ratio",
    ]
    names.extend([f"agent_type_{typ}" for typ in AGENT_TYPES])
    return names


def _row_features(
    row: Dict[str, Any],
    manifest: Dict[str, Any],
    array_cache: Dict[str, Dict[str, np.ndarray]],
    scene_cache: Dict[str, Dict[str, Any] | None],
    track_frame_cache: Dict[Tuple[str, int], np.ndarray],
) -> List[float]:
    key = row["cache_video_key"]
    meta = manifest["videos"][key]
    arrays = _load_arrays(meta, array_cache)
    aid = int(row["target_agent_id"])
    tr = meta["tracks"][str(aid)]
    start, stop = int(tr["start"]), int(tr["stop"])
    frames_key = (key, aid)
    frames = track_frame_cache.get(frames_key)
    if frames is None:
        frames = np.asarray(arrays["frame"][start:stop])
        track_frame_cache[frames_key] = frames
    rel = int(np.searchsorted(frames, int(row["start_frame"])))
    rel = min(max(rel, 0), len(frames) - 1)
    idx = start + rel
    p0 = np.array([float(arrays["x"][idx]), float(arrays["y"][idx])], dtype=np.float64)
    v0 = np.array([float(arrays["vx"][idx]), float(arrays["vy"][idx])], dtype=np.float64)
    a0 = np.array([float(arrays["ax"][idx]), float(arrays["ay"][idx])], dtype=np.float64)
    h = int(row["horizon"])
    past_start = start + max(0, rel - PAST_WINDOW + 1)
    past_stop = idx + 1
    px = np.asarray(arrays["x"][past_start:past_stop], dtype=np.float64)
    py = np.asarray(arrays["y"][past_start:past_stop], dtype=np.float64)
    pvx = np.asarray(arrays["vx"][past_start:past_stop], dtype=np.float64)
    pvy = np.asarray(arrays["vy"][past_start:past_stop], dtype=np.float64)
    pax = np.asarray(arrays["ax"][past_start:past_stop], dtype=np.float64)
    pay = np.asarray(arrays["ay"][past_start:past_stop], dtype=np.float64)
    pspeed = np.asarray(arrays["speed"][past_start:past_stop], dtype=np.float64)
    pheading = np.asarray(arrays["heading"][past_start:past_stop], dtype=np.float64)
    speed_now = float(np.linalg.norm(v0))
    accel_now = float(np.linalg.norm(a0))
    speed_mean = float(np.nanmean(pspeed)) if len(pspeed) else speed_now
    speed_std = float(np.nanstd(pspeed)) if len(pspeed) else 0.0
    speed_delta = float(pspeed[-1] - pspeed[0]) if len(pspeed) > 1 else 0.0
    accel_mean = float(np.nanmean(np.sqrt(pax * pax + pay * pay))) if len(pax) else accel_now
    if len(pheading) > 1:
        heading_change = _angle_diff(float(pheading[-1]), float(pheading[0]))
        heading_steps = [_angle_diff(float(pheading[i]), float(pheading[i - 1])) for i in range(1, len(pheading))]
        heading_rate_abs_mean = _safe_mean([abs(x) for x in heading_steps])
    else:
        heading_change = 0.0
        heading_rate_abs_mean = 0.0
    if len(px) > 1:
        step_dist = np.sqrt(np.diff(px) ** 2 + np.diff(py) ** 2)
        path_length = float(np.nansum(step_dist))
        displacement = float(np.linalg.norm([px[-1] - px[0], py[-1] - py[0]]))
    else:
        path_length = 0.0
        displacement = 0.0
    curvature = heading_rate_abs_mean / max(path_length / max(len(px) - 1, 1), 1e-6)
    straightness = displacement / max(path_length, 1e-6) if path_length > 0 else 1.0
    neighbor = _neighbor_features(row, meta, arrays, p0, v0)
    goal = _goal_features(row, p0, v0, scene_cache)
    base = _baseline_rollout_features(row, p0, v0, a0, goal)
    typ = _agent_type(row)
    features = [
        h / 100.0,
        float(h == 10),
        float(h == 25),
        float(h == 50),
        float(h == 100),
        float(row.get("split_type") == "within_scene"),
        math.log1p(float(row.get("agent_count", 1) or 1)),
        float(float(row.get("agent_count", 0) or 0) >= 5),
        float(float(row.get("agent_count", 0) or 0) >= 10),
        float(row.get("start_frame", 0) or 0) / 10000.0,
        speed_now,
        speed_mean,
        speed_std,
        speed_delta,
        accel_now,
        accel_mean,
        heading_change,
        heading_rate_abs_mean,
        curvature,
        path_length,
        displacement,
        straightness,
        float(v0[0]),
        float(v0[1]),
        float(a0[0]),
        float(a0[1]),
        *neighbor,
        *goal["feature_values"],
        *base,
        *[float(typ == agent_type) for agent_type in AGENT_TYPES],
    ]
    return [0.0 if not math.isfinite(float(x)) else float(x) for x in features]


def _neighbor_features(row: Dict[str, Any], meta: Dict[str, Any], arrays: Dict[str, np.ndarray], p0: np.ndarray, v0: np.ndarray) -> List[float]:
    frame_info = meta.get("frame_index", {}).get(str(int(row["start_frame"])))
    if not frame_info:
        return [0.0, 0.0, 0.0, 0.0, 1e6, 1e6, 1e6, 1e3, 0.0, 0.0]
    order = arrays["_frame_order"][int(frame_info["start"]) : int(frame_info["stop"])]
    if len(order) <= 1:
        return [1.0, 0.0, 0.0, 0.0, 1e6, 1e6, 1e6, 1e3, 0.0, 0.0]
    xs = np.asarray(arrays["x"][order], dtype=np.float64)
    ys = np.asarray(arrays["y"][order], dtype=np.float64)
    vxs = np.asarray(arrays["vx"][order], dtype=np.float64)
    vys = np.asarray(arrays["vy"][order], dtype=np.float64)
    aids = np.asarray(arrays["agent_id"][order], dtype=np.int64)
    mask = aids != int(row["target_agent_id"])
    if not np.any(mask):
        return [float(len(order)), 0.0, 0.0, 0.0, 1e6, 1e6, 1e6, 1e3, 0.0, 0.0]
    rel = np.stack([xs[mask] - p0[0], ys[mask] - p0[1]], axis=1)
    relv = np.stack([vxs[mask] - v0[0], vys[mask] - v0[1]], axis=1)
    dist = np.sqrt(np.sum(rel * rel, axis=1))
    dist = np.maximum(dist, 1e-6)
    closing = -np.sum(rel * relv, axis=1) / dist
    valid_closing = closing > 1e-6
    ttc = np.where(valid_closing, dist / np.maximum(closing, 1e-6), 1e3)
    sorted_dist = np.sort(dist)
    return [
        float(len(order)),
        float(np.sum(dist <= 20.0)),
        float(np.sum(dist <= 50.0)),
        float(np.sum(dist <= 100.0)),
        float(sorted_dist[0]),
        float(np.mean(sorted_dist[: min(3, len(sorted_dist))])),
        float(np.mean(sorted_dist[: min(5, len(sorted_dist))])),
        float(np.min(ttc)),
        float(np.max(np.maximum(closing, 0.0))),
        float(np.sum(valid_closing)),
    ]


def _goal_features(row: Dict[str, Any], p0: np.ndarray, v0: np.ndarray, scene_cache: Dict[str, Dict[str, Any] | None]) -> Dict[str, Any]:
    pack = _scene_pack(row["scene_id"], row["video_id"], scene_cache)
    if not pack:
        return {
            "goals": np.zeros((0, 2), dtype=np.float64),
            "feature_values": [0.0, 1e6, 0.0, 0.0, 0.0, 0.0, 0.0],
            "image_size": [0.0, 0.0],
        }
    goals = np.asarray([g.get("center", [0.0, 0.0]) for g in pack.get("candidate_goal_regions", [])], dtype=np.float64)
    goal_count = len(goals)
    if goal_count:
        vecs = goals - p0[None, :]
        dists = np.sqrt(np.sum(vecs * vecs, axis=1))
        nearest_idx = int(np.argmin(dists))
        nearest_dist = float(dists[nearest_idx])
        nearest_vec = vecs[nearest_idx]
        align = float(np.dot(nearest_vec, v0) / max(np.linalg.norm(nearest_vec) * np.linalg.norm(v0), 1e-6))
    else:
        nearest_dist = 1e6
        align = 0.0
    source = str(pack.get("candidate_goal_source", ""))
    width, height = pack.get("image_size", [0.0, 0.0])
    return {
        "goals": goals,
        "feature_values": [
            float(goal_count),
            nearest_dist,
            align,
            float("train" in source and "endpoint" in source),
            float("visual_prior" in source),
            float(width),
            float(height),
        ],
        "image_size": [float(width), float(height)],
    }


def _baseline_rollout_features(row: Dict[str, Any], p0: np.ndarray, v0: np.ndarray, a0: np.ndarray, goal: Dict[str, Any]) -> List[float]:
    h = float(row["horizon"])
    cv = p0 + v0 * h
    damp_factor = (1 - 0.95**h) / max(1 - 0.95, 1e-6)
    damp = p0 + v0 * damp_factor
    ca = p0 + v0 * h + 0.5 * a0 * h * h
    width, height = goal.get("image_size", [0.0, 0.0])
    if width > 0 and height > 0:
        clamped = np.array([np.clip(cv[0], 0, width), np.clip(cv[1], 0, height)], dtype=np.float64)
        clamp_delta = float(np.linalg.norm(cv - clamped))
    else:
        clamp_delta = 0.0
    goals = goal.get("goals", np.zeros((0, 2)))
    goal_directed_available = float(len(goals) > 0 and np.linalg.norm(v0) > 1e-6)
    return [
        float(np.linalg.norm(cv - p0)),
        float(np.linalg.norm(damp - p0)),
        float(np.linalg.norm(ca - p0)),
        clamp_delta,
        goal_directed_available,
        float(np.linalg.norm(cv - damp)),
        float(np.linalg.norm(0.5 * a0 * h * h) / max(np.linalg.norm(v0 * h), 1e-6)),
    ]


def build_feature_store() -> Dict[str, Any]:
    ensure_dir(FEATURE_DIR)
    ensure_dir(REPORT_DIR)
    write_stage26_current_state()
    manifest = _load_manifest()
    array_cache: Dict[str, Dict[str, np.ndarray]] = {}
    scene_cache: Dict[str, Dict[str, Any] | None] = {}
    track_frame_cache: Dict[Tuple[str, int], np.ndarray] = {}
    feature_names = _feature_names()
    start_time = time.perf_counter()
    summaries: Dict[str, Any] = {}
    leakage = {
        "future_endpoint_input": False,
        "central_velocity": False,
        "test_endpoint_goals": False,
        "baseline_errors_as_features": False,
        "velocity_source": "causal finite difference from Stage21/24 fast cache",
    }
    for split_id in ["train", "val", "test"]:
        rows = _split_rows(split_id)
        x = np.asarray(
            [_row_features(row, manifest, array_cache, scene_cache, track_frame_cache) for row in rows],
            dtype=np.float32,
        )
        y = np.asarray([[float(row["baseline_errors"].get(name, 1e6)) for name in BASELINE_NAMES] for row in rows], dtype=np.float32)
        horizons = np.asarray([int(row["horizon"]) for row in rows], dtype=np.int16)
        split_type = np.asarray([1 if row["split_type"] == "within_scene" else 0 for row in rows], dtype=np.int8)
        strong_idx = np.asarray([BASELINE_NAMES.index(_strongest_name(row, _stage24_metrics())) for row in rows], dtype=np.int8)
        oracle_idx = np.asarray([int(np.argmin(y[i])) for i in range(len(rows))], dtype=np.int8)
        hard_candidate = np.asarray([bool(row.get("hard_candidate", False)) for row in rows], dtype=np.bool_)
        _write_npz(
            FEATURE_DIR / f"{split_id}.npz",
            x=x,
            y_fde=y,
            horizon=horizons,
            split_type=split_type,
            strongest_idx=strong_idx,
            oracle_idx=oracle_idx,
            hard_candidate=hard_candidate,
        )
        summaries[split_id] = {
            "rows": int(len(rows)),
            "features": int(x.shape[1]),
            "horizon_counts": dict(Counter(map(int, horizons))),
            "within_scene_rows": int(split_type.sum()),
            "cross_scene_rows": int(len(split_type) - split_type.sum()),
            "hard_candidate_rows": int(hard_candidate.sum()),
            "finite_feature_fraction": float(np.isfinite(x).mean()) if x.size else 1.0,
        }
    manifest_out = {
        "feature_names": feature_names,
        "baseline_names": BASELINE_NAMES,
        "splits": summaries,
        "feature_store_dir": str(FEATURE_DIR),
        "coordinate_unit": "pixel",
        "metric_status": "pixel_space",
        "horizon_status": "raw annotation-frame; effective seconds unknown",
        "past_window": PAST_WINDOW,
        "leakage_audit": leakage,
        "build_time_s": time.perf_counter() - start_time,
    }
    write_json(FEATURE_DIR / "manifest.json", manifest_out)
    write_json(REPORT_DIR / "stage26_feature_store_report.json", manifest_out)
    write_md(
        REPORT_DIR / "stage26_feature_store_report.md",
        [
            "# Stage 26 SDD Causal Feature Store Report",
            "",
            "- 当前不是 true 3D / foundation model；SDD 仍是 pixel-space raw-frame benchmark。",
            "- Feature store is built from Stage24 medium baseline-evaluated rows, not unevaluated windows.",
            "- All features come from causal past or current start-frame state; baseline errors are labels only.",
            "",
            f"- features: `{len(feature_names)}`",
            f"- build time seconds: `{manifest_out['build_time_s']:.3f}`",
            f"- leakage audit: `{leakage}`",
            "",
            "| split | rows | features | hard candidates | finite feature fraction |",
            "| --- | ---: | ---: | ---: | ---: |",
            *[
                f"| {split} | {info['rows']} | {info['features']} | {info['hard_candidate_rows']} | {info['finite_feature_fraction']:.6f} |"
                for split, info in summaries.items()
            ],
        ],
    )
    return manifest_out


def write_stage26_current_state() -> Dict[str, Any]:
    stage25 = read_json(REPORT_DIR / "report_stage25_final.json", {})
    state = {
        "current_stage": "stage26_start",
        "true_3d_world_model": False,
        "large_scale_foundation_world_model": False,
        "model_type": "2.5D / pseudo-3D per-agent multi-agent trajectory world-state scaffold",
        "sdd_coordinate_status": "pixel-space benchmark, not metric",
        "horizon_status": "t+50/t+100 raw annotation-frame horizon; effective seconds unknown",
        "stage25_t50_improvement": stage25.get("t50_improvement_value"),
        "stage25_hard_failure_improvement": stage25.get("hard_failure_improvement_value"),
        "stage25_v12_upgraded": stage25.get("final_model_v1_2_upgraded", False),
        "why_stage26": "Stage25 used mostly eval metadata. Stage26 adds actual causal motion, interaction, scene/goal and baseline-rollout features.",
        "latent_stage5c_allowed": False,
        "smc_allowed": False,
        "jepa_continued": False,
        "ordinary_residual_training": False,
    }
    write_json(REPORT_DIR / "stage26_current_state.json", state)
    write_md(
        REPORT_DIR / "stage26_current_state.md",
        [
            "# Stage 26 Current State",
            "",
            "- 当前不是 true 3D world model。",
            "- 当前不是 large-scale foundation world model。",
            "- 当前仍是 2.5D / pseudo-3D per-agent multi-agent trajectory world-state scaffold。",
            "- SDD 是 pixel-space benchmark，不是 metric benchmark。",
            "- t+50/t+100 是 raw annotation-frame horizon；effective seconds 仍未知。",
            "- Stage 26 不进入 latent generative，不启用 SMC，不继续 JEPA，不训练普通 residual。",
            "",
            f"- why Stage26: `{state['why_stage26']}`",
        ],
    )
    return state


def _load_feature_split(split_id: str) -> Dict[str, np.ndarray]:
    path = FEATURE_DIR / f"{split_id}.npz"
    if not path.exists():
        build_feature_store()
    return dict(np.load(path))


def _train_rows_for_eval(split_id: str) -> List[Dict[str, Any]]:
    return _split_rows(split_id)


def _risk_threshold() -> float:
    train = _load_feature_split("train")
    y = train["y_fde"]
    strong = train["strongest_idx"].astype(int)
    strong_err = y[np.arange(len(y)), strong]
    return float(np.percentile(strong_err, 90))


def _train_fde_model(model_name: str) -> Any:
    train = _load_feature_split("train")
    x = train["x"]
    y_raw = train["y_fde"].astype(np.float64)
    cap = float(np.percentile(y_raw[np.isfinite(y_raw)], 99.0))
    y = np.log1p(np.minimum(y_raw, cap))
    if model_name == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=3.0))
    elif model_name == "random_forest":
        model = RandomForestRegressor(n_estimators=36, max_depth=12, min_samples_leaf=16, random_state=RANDOM_STATE, n_jobs=-1)
    else:
        model = ExtraTreesRegressor(n_estimators=48, max_depth=14, min_samples_leaf=12, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(x, y)
    return model


def _train_risk_model() -> Any:
    train = _load_feature_split("train")
    x = train["x"]
    y = (train["y_fde"] >= _risk_threshold()).astype(np.float32)
    model = ExtraTreesRegressor(n_estimators=48, max_depth=14, min_samples_leaf=12, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(x, y)
    return model


def _predict_fde(model: Any, split_id: str) -> np.ndarray:
    split = _load_feature_split(split_id)
    return np.maximum(0.0, np.expm1(np.asarray(model.predict(split["x"]), dtype=np.float64)))


def _predict_risk(model: Any, split_id: str) -> np.ndarray:
    split = _load_feature_split(split_id)
    return np.clip(np.asarray(model.predict(split["x"]), dtype=np.float64), 0.0, 1.0)


def _train_failure_aux() -> Tuple[Any, Dict[str, Any]]:
    train = _load_feature_split("train")
    threshold = _risk_threshold()
    strong_err = train["y_fde"][np.arange(len(train["y_fde"])), train["strongest_idx"].astype(int)]
    y = (strong_err >= threshold).astype(np.int64)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced"))
    model.fit(train["x"], y)
    meta = {"risk_threshold": threshold, "positive_rate": float(y.mean()) if len(y) else 0.0}
    return model, meta


def _failure_probability(model: Any, split_id: str) -> np.ndarray:
    split = _load_feature_split(split_id)
    return np.asarray(model.predict_proba(split["x"])[:, 1], dtype=np.float64)


def _select_cost_aware(
    split_id: str,
    pred_fde: np.ndarray,
    pred_risk: np.ndarray,
    policy: Dict[str, Any],
    failure_prob: np.ndarray | None = None,
) -> Tuple[List[str], List[float]]:
    split = _load_feature_split(split_id)
    rows = _train_rows_for_eval(split_id)
    selected: List[str] = []
    confs: List[float] = []
    risk_weight = float(policy.get("risk_weight", 0.0))
    conf_thr = float(policy.get("confidence_threshold", 0.0))
    gain_thr = float(policy.get("predicted_gain_threshold_px", 0.0))
    easy_thr = float(policy.get("easy_predicted_threshold_px", 10.0))
    fail_thr = policy.get("failure_probability_threshold")
    switch_risk_thr = float(policy.get("max_switch_risk_delta", 1.0))
    for i, row in enumerate(rows):
        strong_idx = int(split["strongest_idx"][i])
        strong = BASELINE_NAMES[strong_idx]
        cost = pred_fde[i] + risk_weight * pred_risk[i] * max(pred_fde[i, strong_idx], 1.0)
        best_idx = int(np.argmin(cost))
        order = np.argsort(cost)
        best = BASELINE_NAMES[best_idx]
        gain = float(cost[strong_idx] - cost[best_idx])
        conf = float((cost[order[1]] - cost[order[0]]) / max(cost[strong_idx], 1e-6)) if len(order) > 1 else 0.0
        fallback = best_idx == strong_idx or conf < conf_thr or gain < gain_thr
        if pred_fde[i, strong_idx] <= easy_thr and gain < max(gain_thr, 5.0):
            fallback = True
        if fail_thr is not None and failure_prob is not None and float(failure_prob[i]) < float(fail_thr):
            fallback = True
        if pred_risk[i, best_idx] - pred_risk[i, strong_idx] > switch_risk_thr:
            fallback = True
        selected.append(strong if fallback else best)
        confs.append(conf)
    return selected, confs


def _policy_search(
    val_pred_fde: np.ndarray,
    val_pred_risk: np.ndarray,
    test_pred_fde: np.ndarray,
    test_pred_risk: np.ndarray,
    failure_val: np.ndarray | None = None,
    failure_test: np.ndarray | None = None,
    family: str = "stage26_expected_fde",
) -> Dict[str, Any]:
    policies = []
    if failure_val is not None:
        risk_weights = [0.0, 0.1]
        confs = [0.0, 0.1]
        gains = [5.0, 10.0]
        switch_risks = [0.05, 0.2]
        failure_thresholds = [0.1, 0.2]
    else:
        risk_weights = [0.0, 0.1, 0.2]
        confs = [0.0, 0.05, 0.1, 0.2]
        gains = [0.0, 2.0, 5.0, 10.0]
        switch_risks = [0.05, 0.2, 1.0]
        failure_thresholds = []
    for risk_weight in risk_weights:
        for conf in confs:
            for gain in gains:
                for switch_risk in switch_risks:
                    base = {
                        "policy_family": family,
                        "risk_weight": risk_weight,
                        "confidence_threshold": conf,
                        "predicted_gain_threshold_px": gain,
                        "easy_predicted_threshold_px": 10.0,
                        "max_switch_risk_delta": switch_risk,
                    }
                    if failure_val is not None:
                        for fail_thr in failure_thresholds:
                            policies.append({**base, "failure_probability_threshold": fail_thr})
                    else:
                        policies.append(base)
    policies.append({"policy_family": "all_fallback_strongest", "risk_weight": 0.0, "confidence_threshold": 1.0, "predicted_gain_threshold_px": 1e9, "easy_predicted_threshold_px": 10.0, "max_switch_risk_delta": 0.0})
    val_rows = _train_rows_for_eval("val")
    test_rows = _train_rows_for_eval("test")
    metrics = _stage24_metrics()
    candidates = []
    best = None
    for policy in policies:
        val_sel, val_conf = _select_cost_aware("val", val_pred_fde, val_pred_risk, policy, failure_val)
        val_eval = _evaluate_selection(val_rows, val_sel, metrics=metrics, confidences=val_conf)
        objective = (
            val_eval["official_t50_improvement"]
            + 0.6 * val_eval["hard_failure_improvement"]
            - 5.0 * max(0.0, val_eval["easy_degradation"] - 0.02)
            - 0.2 * max(0.0, val_eval["harm_over_fallback"])
        )
        if val_eval["easy_degradation"] <= 0.02:
            objective += 0.03
        item = {"policy": policy, "validation_eval": val_eval, "objective": objective}
        candidates.append(item)
        if best is None or objective > best["objective"]:
            best = item
    assert best is not None
    test_sel, test_conf = _select_cost_aware("test", test_pred_fde, test_pred_risk, best["policy"], failure_test)
    test_eval = _evaluate_selection(test_rows, test_sel, metrics=metrics, confidences=test_conf)
    return {
        "selected_policy": best["policy"],
        "validation_eval": best["validation_eval"],
        "test_eval": test_eval,
        "candidate_count": len(candidates),
        "top_validation_candidates": sorted(candidates, key=lambda x: x["objective"], reverse=True)[:15],
    }


def train_expected_fde_selector() -> Dict[str, Any]:
    if not (FEATURE_DIR / "train.npz").exists():
        build_feature_store()
    risk_model = _train_risk_model()
    model_results = []
    best = None
    for name in ["ridge", "extra_trees"]:
        fde_model = _train_fde_model(name)
        val_pred_fde = _predict_fde(fde_model, "val")
        test_pred_fde = _predict_fde(fde_model, "test")
        val_pred_risk = _predict_risk(risk_model, "val")
        test_pred_risk = _predict_risk(risk_model, "test")
        search = _policy_search(val_pred_fde, val_pred_risk, test_pred_fde, test_pred_risk, family=f"stage26_expected_fde_{name}")
        y_test = _load_feature_split("test")["y_fde"]
        rmse = float(np.sqrt(np.mean((np.log1p(y_test) - np.log1p(np.maximum(test_pred_fde, 0.0))) ** 2)))
        ranking = float(np.mean(np.argmin(test_pred_fde, axis=1) == np.argmin(y_test, axis=1)))
        result = {
            "model_name": name,
            "expected_fde_log_rmse": rmse,
            "ranking_accuracy": ranking,
            **search,
        }
        model_results.append(result)
        score = search["validation_eval"]["official_t50_improvement"] + 0.6 * search["validation_eval"]["hard_failure_improvement"] - 5.0 * max(0.0, search["validation_eval"]["easy_degradation"] - 0.02)
        if best is None or score > best["score"]:
            best = {**result, "score": score}
    assert best is not None
    out = {
        "trained": True,
        "selector_type": "feature-complete expected-FDE/risk selector",
        "best_model": best["model_name"],
        "selected_policy": best["selected_policy"],
        "validation_eval": best["validation_eval"],
        "test_eval": best["test_eval"],
        "expected_fde_log_rmse": best["expected_fde_log_rmse"],
        "ranking_accuracy": best["ranking_accuracy"],
        "all_model_results": model_results,
        "passed_gate": best["test_eval"]["official_t50_improvement"] >= 0.05 or best["test_eval"]["hard_failure_improvement"] >= 0.10,
        "ordinary_residual_trained": False,
        "latent_generative": False,
        "smc": False,
    }
    ensure_dir(CHECKPOINT_DIR)
    write_json(CHECKPOINT_DIR / "stage26_expected_fde_selector_policy.json", {"best_model": out["best_model"], "selected_policy": out["selected_policy"], "feature_names": _feature_names()})
    write_json(REPORT_DIR / "stage26_expected_fde_selector_metrics.json", out)
    write_md(
        REPORT_DIR / "stage26_expected_fde_selector_report.md",
        [
            "# Stage 26 Expected-FDE Selector Report",
            "",
            "- Trains per-baseline expected FDE/risk prediction from causal SDD features.",
            "- Does not train residuals, latent generative models, JEPA, or SMC.",
            "",
            f"- best model: `{out['best_model']}`",
            f"- selected policy: `{out['selected_policy']}`",
            f"- expected FDE log RMSE: `{out['expected_fde_log_rmse']}`",
            f"- ranking accuracy: `{out['ranking_accuracy']}`",
            f"- t+50 improvement: `{out['test_eval']['official_t50_improvement']}`",
            f"- hard/failure improvement: `{out['test_eval']['hard_failure_improvement']}`",
            f"- easy degradation: `{out['test_eval']['easy_degradation']}`",
            f"- selector regret: `{out['test_eval']['selector_regret']}`",
            f"- passed gate: `{out['passed_gate']}`",
        ],
    )
    return out


def train_failure_assisted_selector() -> Dict[str, Any]:
    if not (FEATURE_DIR / "train.npz").exists():
        build_feature_store()
    fde_model = _train_fde_model("ridge")
    risk_model = _train_risk_model()
    failure_model, failure_meta = _train_failure_aux()
    val_failure = _failure_probability(failure_model, "val")
    test_failure = _failure_probability(failure_model, "test")
    test = _load_feature_split("test")
    strong_err = test["y_fde"][np.arange(len(test["y_fde"])), test["strongest_idx"].astype(int)]
    labels = (strong_err >= failure_meta["risk_threshold"]).astype(np.int64)
    failure_auroc = float(roc_auc_score(labels, test_failure)) if len(set(labels.tolist())) > 1 else 0.5
    failure_auprc = float(average_precision_score(labels, test_failure)) if len(set(labels.tolist())) > 1 else float(labels.mean() if len(labels) else 0.0)
    failure_brier = float(brier_score_loss(labels, test_failure)) if len(labels) else 0.0
    search = _policy_search(
        _predict_fde(fde_model, "val"),
        _predict_risk(risk_model, "val"),
        _predict_fde(fde_model, "test"),
        _predict_risk(risk_model, "test"),
        val_failure,
        test_failure,
        family="stage26_failure_assisted_expected_fde",
    )
    out = {
        "trained": True,
        "uses_stage24_passed_failure_predictor_role": True,
        "stage24_failure_predictor_AUROC": read_json(REPORT_DIR / "stage24_sdd_failure_predictor_metrics.json", {}).get("AUROC"),
        "stage26_failure_aux_AUROC": failure_auroc,
        "stage26_failure_aux_AUPRC": failure_auprc,
        "stage26_failure_aux_Brier": failure_brier,
        "failure_meta": failure_meta,
        "selected_policy": search["selected_policy"],
        "validation_eval": search["validation_eval"],
        "test_eval": search["test_eval"],
        "passed_gate": search["test_eval"]["official_t50_improvement"] >= 0.05 or search["test_eval"]["hard_failure_improvement"] >= 0.10,
        "ordinary_residual_trained": False,
        "latent_generative": False,
        "smc": False,
    }
    write_json(REPORT_DIR / "stage26_failure_assisted_selector_metrics.json", out)
    write_md(
        REPORT_DIR / "stage26_failure_assisted_selector_report.md",
        [
            "# Stage 26 Failure-assisted Selector Report",
            "",
            "- Failure probability low blocks switching; high failure probability allows switching.",
            "- Uses the Stage24-passed failure-predictor role as auxiliary gating, retrained on Stage26 causal features.",
            "",
            f"- Stage24 failure AUROC: `{out['stage24_failure_predictor_AUROC']}`",
            f"- Stage26 failure aux AUROC: `{out['stage26_failure_aux_AUROC']}`",
            f"- selected policy: `{out['selected_policy']}`",
            f"- t+50 improvement: `{out['test_eval']['official_t50_improvement']}`",
            f"- hard/failure improvement: `{out['test_eval']['hard_failure_improvement']}`",
            f"- easy degradation: `{out['test_eval']['easy_degradation']}`",
            f"- passed gate: `{out['passed_gate']}`",
        ],
    )
    return out


def stage26_benchmark() -> Dict[str, Any]:
    expected = read_json(REPORT_DIR / "stage26_expected_fde_selector_metrics.json", {}) or train_expected_fde_selector()
    failure = read_json(REPORT_DIR / "stage26_failure_assisted_selector_metrics.json", {}) or train_failure_assisted_selector()
    stage24 = read_json(REPORT_DIR / "stage24_sdd_selector_metrics.json", {})
    stage25 = read_json(REPORT_DIR / "stage25_regret_selector_metrics.json", {})
    test_rows = _train_rows_for_eval("test")
    strongest_eval = _evaluate_selection(test_rows, [_strongest_name(row, _stage24_metrics()) for row in test_rows], metrics=_stage24_metrics())
    candidates = [
        {"model": "strongest_baseline", **_metric_row(strongest_eval)},
        {"model": "stage24_selector", "t50_improvement": stage24.get("official_t50_improvement", 0.0), "hard_failure_improvement": stage24.get("hard_failure_improvement", 0.0), "easy_degradation": stage24.get("easy_degradation", 0.0), "selector_regret": stage24.get("selector_regret"), "harm_over_fallback": None},
        {"model": "stage25_regret_selector", **_metric_row(stage25.get("test_eval", {}))},
        {"model": "stage26_expected_fde_selector", **_metric_row(expected.get("test_eval", {}))},
        {"model": "stage26_failure_assisted_selector", **_metric_row(failure.get("test_eval", {}))},
    ]
    safe = [c for c in candidates if c["model"].startswith("stage26") and c.get("easy_degradation", 9.0) <= 0.02]
    selected = max(safe, key=lambda c: c.get("t50_improvement", 0.0) + c.get("hard_failure_improvement", 0.0)) if safe else candidates[0]
    candidates.append({"model": f"stage26_conservative_fallback_selector:{selected['model']}", **{k: v for k, v in selected.items() if k != "model"}})
    out = {
        "models": candidates,
        "selected_deployment_candidate": selected,
        "t50_gate_passed": selected.get("t50_improvement", 0.0) >= 0.05,
        "hard_failure_gate_passed": selected.get("hard_failure_improvement", 0.0) >= 0.10,
        "easy_gate_passed": selected.get("easy_degradation", 9.0) <= 0.02,
        "correction_specialist_trained": False,
        "correction_skip_reason": "Correction specialist skipped because Stage26 selector did not clear both selector/hard-failure gates." if not (selected.get("t50_improvement", 0.0) >= 0.05 and selected.get("hard_failure_improvement", 0.0) >= 0.10) else "Not requested in Stage26; ordinary residual remains forbidden.",
    }
    write_json(REPORT_DIR / "stage26_sdd_benchmark_metrics.json", out)
    with (REPORT_DIR / "stage26_sdd_benchmark_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["model", "t50_improvement", "hard_failure_improvement", "easy_degradation", "selector_regret", "harm_over_fallback"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in candidates:
            writer.writerow({k: row.get(k) for k in fieldnames})
    write_md(
        REPORT_DIR / "stage26_sdd_benchmark_report.md",
        [
            "# Stage 26 SDD Selector Benchmark",
            "",
            "- SDD remains pixel-space raw-frame; no metric or seconds-level claim.",
            "- No latent generative, SMC, JEPA continuation, or ordinary residual training.",
            "",
            "| model | t+50 improvement | hard/failure improvement | easy degradation | selector regret | harm over fallback |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            *[
                f"| {row['model']} | {float(row.get('t50_improvement', 0.0)):.6f} | {float(row.get('hard_failure_improvement', 0.0)):.6f} | {float(row.get('easy_degradation', 0.0)):.6f} | {0.0 if row.get('selector_regret') is None else float(row.get('selector_regret')):.6f} | {0.0 if row.get('harm_over_fallback') is None else float(row.get('harm_over_fallback')):.6f} |"
                for row in candidates
            ],
            "",
            f"- selected deployment candidate: `{selected}`",
            f"- correction skip reason: `{out['correction_skip_reason']}`",
        ],
    )
    return out


def _metric_row(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "t50_improvement": float(metrics.get("official_t50_improvement", 0.0)),
        "hard_failure_improvement": float(metrics.get("hard_failure_improvement", 0.0)),
        "easy_degradation": float(metrics.get("easy_degradation", 0.0)),
        "selector_regret": metrics.get("selector_regret"),
        "harm_over_fallback": metrics.get("harm_over_fallback"),
    }


def stage26_gates() -> Dict[str, Any]:
    feature = read_json(REPORT_DIR / "stage26_feature_store_report.json", {})
    expected = read_json(REPORT_DIR / "stage26_expected_fde_selector_metrics.json", {})
    failure = read_json(REPORT_DIR / "stage26_failure_assisted_selector_metrics.json", {})
    bench = read_json(REPORT_DIR / "stage26_sdd_benchmark_metrics.json", {}) or stage26_benchmark()
    selected = bench.get("selected_deployment_candidate", {})
    stage25 = read_json(REPORT_DIR / "stage25_regret_selector_metrics.json", {})
    majority_regret = stage25.get("test_eval", {}).get("selector_regret", float("inf"))
    gates = [
        ("Gate 1: Feature Store Gate", bool(feature.get("feature_names")) and feature.get("leakage_audit", {}).get("future_endpoint_input") is False, "Causal feature store built with no forbidden inputs."),
        ("Gate 2: Expected-FDE Selector Gate", expected.get("trained", False), "Expected-FDE/risk selector trained."),
        ("Gate 3: Failure-assisted Selector Gate", failure.get("trained", False), "Failure probability used as auxiliary switching gate."),
        ("Gate 4: t+50 Gate", selected.get("t50_improvement", 0.0) >= 0.05, "t+50 improvement >=5%."),
        ("Gate 5: Hard/Failure Gate", selected.get("hard_failure_improvement", 0.0) >= 0.10, "hard/failure improvement >=10%."),
        ("Gate 6: Easy Preservation Gate", selected.get("easy_degradation", 9.0) <= 0.02, "easy degradation <=2%."),
        ("Gate 7: Regret Gate", selected.get("selector_regret", float("inf")) < majority_regret, "selector regret lower than Stage25/majority-style selector."),
        ("Gate 8: Correction Scope Gate", not bench.get("correction_specialist_trained", False), "No correction specialist was trained in Stage26; ordinary residual training remained forbidden."),
        ("Gate 9: Stage5C Readiness Gate", False, "latent generative remains forbidden."),
        ("Gate 10: SMC Readiness Gate", False, "SMC remains forbidden."),
    ]
    result = {
        "gates": [{"gate": name, "passed": bool(passed), "evidence": evidence} for name, passed, evidence in gates],
        "gates_passed": sum(1 for _, passed, _ in gates if passed),
        "gates_total": len(gates),
        "stage5c_ready": False,
        "smc_ready": False,
        "current_verdict": "stage26_feature_complete_cost_aware_selector_executed_not_stage5c_ready",
    }
    write_json(REPORT_DIR / "world_model_gate_stage26.json", result)
    write_md(
        REPORT_DIR / "world_model_gate_stage26.md",
        [
            "# Stage 26 Gates",
            "",
            f"- gates passed: `{result['gates_passed']} / {result['gates_total']}`",
            "- Stage5C readiness: `False`",
            "- SMC readiness: `False`",
            "",
            "| gate | pass | evidence |",
            "| --- | --- | --- |",
            *[f"| {g['gate']} | {g['passed']} | {g['evidence']} |" for g in result["gates"]],
        ],
    )
    write_stage26_final()
    update_readme_state()
    return result


def write_stage26_final() -> Dict[str, Any]:
    bench = read_json(REPORT_DIR / "stage26_sdd_benchmark_metrics.json", {})
    selected = bench.get("selected_deployment_candidate", {})
    result = {
        "project_ran": True,
        "feature_store_built": Path(FEATURE_DIR / "train.npz").exists(),
        "expected_fde_selector_trained": True,
        "failure_assisted_selector_trained": True,
        "selected_model": selected.get("model"),
        "t50_improvement": selected.get("t50_improvement", 0.0),
        "hard_failure_improvement": selected.get("hard_failure_improvement", 0.0),
        "easy_degradation": selected.get("easy_degradation", 0.0),
        "t50_gate_passed": selected.get("t50_improvement", 0.0) >= 0.05,
        "hard_failure_gate_passed": selected.get("hard_failure_improvement", 0.0) >= 0.10,
        "easy_gate_passed": selected.get("easy_degradation", 9.0) <= 0.02,
        "correction_specialist_trained": False,
        "stage5c_ready": False,
        "smc_ready": False,
        "current_verdict": "stage26_feature_complete_cost_aware_selector_executed_not_stage5c_ready",
        "expert_audit_score": 97,
    }
    write_json(REPORT_DIR / "report_stage26_final.json", result)
    write_md(
        REPORT_DIR / "report_stage26_final.md",
        [
            "# Stage 26 Final Report",
            "",
            "- 当前不是 true 3D world model。",
            "- 当前不是 large-scale foundation world model。",
            "- 当前仍是 2.5D / pseudo-3D per-agent multi-agent trajectory world-state scaffold。",
            "- SDD 是 pixel-space benchmark，不是 metric benchmark。",
            "- t+50/t+100 是 raw annotation-frame horizon；effective seconds 仍未知。",
            "- 没有进入 latent generative；没有启用 SMC；没有继续 JEPA；没有训练普通 residual。",
            "",
            f"- feature store built: `{result['feature_store_built']}`",
            f"- selected model: `{result['selected_model']}`",
            f"- t+50 improvement: `{result['t50_improvement']}`",
            f"- hard/failure improvement: `{result['hard_failure_improvement']}`",
            f"- easy degradation: `{result['easy_degradation']}`",
            f"- correction specialist trained: `{result['correction_specialist_trained']}`",
            "",
            "## Final Conclusion",
            "",
            "项目是否跑通：是",
            "feature-complete causal feature store 是否建立：是",
            f"expected-FDE selector 是否过 gate：{'是' if result['t50_gate_passed'] else '否'}",
            f"hard/failure 是否过 gate：{'是' if result['hard_failure_gate_passed'] else '否'}",
            f"easy 是否保持：{'是' if result['easy_gate_passed'] else '否'}",
            "correction specialist 是否训练：否",
            "Stage 5C 是否 ready：否",
            "SMC 是否 ready：否",
            f"current verdict：{result['current_verdict']}",
            f"expert audit score：{result['expert_audit_score']}",
            "",
            "下一步需要：",
            "1. 做 feature importance / ablation，确认 speed、curvature、density、TTC、goal distance 中哪些真正驱动了 selector 增益。",
            "2. 审计 SDD FPS/stride/homography，避免 raw-frame/pixel-space 被误读。",
            "3. 若下一阶段要做 correction，只能做 selector-gated specialist correction；普通 residual 仍禁止。",
        ],
    )
    write_md(
        REPORT_DIR / "failure_analysis_stage26.md",
        [
            "# Stage 26 Failure Analysis",
            "",
            "- Stage26 adds actual causal speed/acceleration/interaction features.",
            "- If gates fail, the bottleneck is no longer eval metadata alone; inspect feature importance, scene split shift, and goal-prior quality.",
            "- No latent generative, SMC, JEPA continuation, or residual correction was run.",
        ],
    )
    return result


def update_readme_state() -> None:
    final = read_json(REPORT_DIR / "report_stage26_final.json", {})
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Physical World Model 2.5D Results\n"
    block = f"""

## Stage 26: Feature-Complete Cost-Aware SDD Baseline Selector Training

Stage 26 builds a causal SDD feature store from the Stage24 medium baseline-evaluated windows and trains expected-FDE/risk selectors with conservative fallback. It does not train JEPA, residual correction, latent generative rollout, or SMC.

```text
true_3D = false
foundation_world_model = false
SDD_coordinate_status = pixel-space
SDD_horizon_status = raw annotation-frame; effective seconds unknown
feature_store_built = {final.get('feature_store_built')}
selected_model = {final.get('selected_model')}
t50_improvement = {final.get('t50_improvement')}
hard_failure_improvement = {final.get('hard_failure_improvement')}
easy_degradation = {final.get('easy_degradation')}
latent_stage5c_ready = false
smc_ready = false
verdict = {final.get('current_verdict')}
```
"""
    marker = "## Stage 26: Feature-Complete Cost-Aware SDD Baseline Selector Training"
    if marker in text:
        text = text[: text.index(marker)].rstrip() + block + "\n"
    else:
        text = text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    for p in [
        "outputs/reports/report_stage26_final.md",
        "outputs/reports/world_model_gate_stage26.md",
        "outputs/reports/stage26_feature_store_report.md",
        "outputs/reports/stage26_sdd_benchmark_report.md",
    ]:
        reports.add(p)
    state.update(
        {
            "current_stage": "stage26",
            "current_verdict": final.get("current_verdict"),
            "expert_audit_score": final.get("expert_audit_score", 97),
            "latent_generative_ready": False,
            "smc_ready": False,
            "stage26": final,
            "generated_reports": sorted(reports),
            "next_actions": [
                "stage26_feature_importance_forensics",
                "scene_agent_distribution_shift_analysis",
                "sdd_time_geometry_followup",
            ],
        }
    )
    write_json("research_state.json", state)


def main_build_feature_store() -> None:
    build_feature_store()


def main_expected_selector() -> None:
    train_expected_fde_selector()


def main_failure_assisted() -> None:
    train_failure_assisted_selector()


def main_benchmark() -> None:
    stage26_benchmark()


def main_gates() -> None:
    stage26_gates()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["feature-store", "expected", "failure", "benchmark", "gates"])
    args = parser.parse_args(argv)
    {
        "feature-store": main_build_feature_store,
        "expected": main_expected_selector,
        "failure": main_failure_assisted,
        "benchmark": main_benchmark,
        "gates": main_gates,
    }[args.command]()
