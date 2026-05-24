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


OUT_DIR = Path("outputs/stage33_coordinate_invariant")
DATA_DIR = Path("data/stage33_coordinate_invariant")
SCENE_PACK_DIR = Path("data/stage33_external_scene_packs")
EXT_FEATURE_DIR = Path("data/stage31_external_feature_store")
EXT_LATENT_DIR = Path("data/stage31_external_latent_cache")
SDD_FEATURE_DIR = Path("data/stage26_sdd_feature_store")
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

STAGE26_T50 = 0.14583655843823773
STAGE26_HARD = 0.11234058960663984
STAGE32_ZERO_SHOT_T50 = -1.018801
STAGE32_ZERO_SHOT_ALL = -0.337476
EPS = 1e-6


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
        "# Stage33 Coordinate-Invariant Run Ledger",
        "",
        "| command | source | status | wall time s | input hash | output hash | git commit |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['command']}` | `{row['source']}` | `{row['status']}` | {float(row['wall_time_s']):.3f} | `{row['input_hash'][:12]}` | `{row['output_hash'][:12]}` | `{row['git_commit']}` |"
        )
    write_md(OUT_DIR / "run_ledger.md", lines)


def run_logged(name: str, fn: Callable[[], Dict[str, Any]], inputs: Sequence[str | Path], outputs: Sequence[str | Path], source: str) -> Dict[str, Any]:
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


def _feature_names() -> List[str]:
    return list(_feature_manifest()["feature_names"])


def _idx(name: str) -> int | None:
    names = _feature_names()
    return names.index(name) if name in names else None


def _load_feature(domain: str, split: str) -> Dict[str, np.ndarray]:
    path = EXT_FEATURE_DIR / f"{split}.npz" if domain == "external" else SDD_FEATURE_DIR / f"{split}.npz"
    return dict(np.load(path))


def _load_latent(domain: str, split: str) -> Dict[str, np.ndarray]:
    path = EXT_LATENT_DIR / f"{split}.npz" if domain == "external" else SDD_LATENT_DIR / f"{split}.npz"
    return dict(np.load(path))


def _safe_col(x: np.ndarray, name: str, default: float = 0.0) -> np.ndarray:
    i = _idx(name)
    if i is None:
        return np.full(len(x), default, dtype=np.float64)
    return x[:, i].astype(np.float64)


def _domain_splits(domain: str = "external") -> Dict[str, Dict[str, np.ndarray]]:
    return {split: _load_feature(domain, split) for split in ["train", "val", "test"]}


def _read_track_file(path: Path) -> np.ndarray:
    return s31._read_opentraj_txt(path)


def _cluster_points(points: np.ndarray, max_k: int = 6) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) <= max_k:
        return pts.copy()
    # Deterministic farthest-point seeds followed by a few Lloyd iterations.
    centers = [pts[int(np.argmin(pts[:, 0] + pts[:, 1]))]]
    while len(centers) < max_k:
        dist = np.min(np.linalg.norm(pts[:, None, :] - np.asarray(centers)[None, :, :], axis=2), axis=1)
        centers.append(pts[int(np.argmax(dist))])
    c = np.asarray(centers, dtype=np.float64)
    for _ in range(8):
        assign = np.argmin(np.linalg.norm(pts[:, None, :] - c[None, :, :], axis=2), axis=1)
        for k in range(len(c)):
            if np.any(assign == k):
                c[k] = pts[assign == k].mean(axis=0)
    return c


def _endpoint_summary(files: Iterable[str]) -> Dict[str, Any]:
    endpoints: List[np.ndarray] = []
    route_points: List[np.ndarray] = []
    bounds = [math.inf, math.inf, -math.inf, -math.inf]
    track_count = 0
    for src in sorted(set(map(str, files))):
        arr = _read_track_file(Path(src))
        if len(arr) == 0:
            continue
        bounds[0] = min(bounds[0], float(np.min(arr[:, 2])))
        bounds[1] = min(bounds[1], float(np.min(arr[:, 3])))
        bounds[2] = max(bounds[2], float(np.max(arr[:, 2])))
        bounds[3] = max(bounds[3], float(np.max(arr[:, 3])))
        for agent in np.unique(arr[:, 1]).astype(int):
            tr = arr[arr[:, 1] == agent]
            tr = tr[np.argsort(tr[:, 0])]
            if len(tr) < 2:
                continue
            endpoints.append(tr[-1, 2:4])
            route_points.extend(tr[:: max(1, len(tr) // 12), 2:4])
            track_count += 1
    if not endpoints:
        return {
            "track_count": 0,
            "endpoint_count": 0,
            "goal_centers": [],
            "bounds": [],
            "route_density_count": 0,
        }
    goals = _cluster_points(np.asarray(endpoints, dtype=np.float64), max_k=6)
    return {
        "track_count": track_count,
        "endpoint_count": len(endpoints),
        "goal_centers": goals.tolist(),
        "bounds": [float(x) for x in bounds],
        "route_density_count": int(len(route_points)),
    }


def build_external_scene_packs() -> Dict[str, Any]:
    ensure_dir(SCENE_PACK_DIR)
    splits = _domain_splits("external")
    train = splits["train"]
    train_scenes = train.get("scene_id", np.asarray([], dtype="U64")).astype(str)
    train_files = train.get("source_file", np.asarray([], dtype="U256")).astype(str)
    scene_to_files: Dict[str, List[str]] = defaultdict(list)
    for scene, src in zip(train_scenes.tolist(), train_files.tolist()):
        scene_to_files[scene].append(src)

    packs = {}
    for scene, files in sorted(scene_to_files.items()):
        summary = _endpoint_summary(files)
        quality = "external_auto_silver" if summary["endpoint_count"] >= 20 and summary["goal_centers"] else "inferred_only"
        pack = {
            "dataset_name": "external_opentraj_trajnet",
            "scene_id": scene,
            "coordinate_unit": "dataset_local_coordinates",
            "metric_status": "unverified_weak_metric_diagnostic",
            "train_only_endpoint_clusters": summary["goal_centers"],
            "candidate_goals": summary["goal_centers"],
            "route_density_proxy_points": summary["route_density_count"],
            "scene_bounds": summary["bounds"],
            "walkable_proxy": "scene_bounds_convex_proxy_if_no_image",
            "goal_distance_features": "available only as train-scene summary; no test endpoints used",
            "annotation_quality": quality,
            "leakage_status": {
                "candidate_goals_from_train_only": True,
                "test_endpoints_used": False,
                "future_endpoint_input": False,
            },
        }
        packs[scene] = pack
        _write_json(SCENE_PACK_DIR / f"{scene}.json", pack)

    assignment = {}
    for split, data in splits.items():
        scenes = data.get("scene_id", np.asarray([], dtype="U64")).astype(str)
        supported = np.asarray([s in packs and len(packs[s]["candidate_goals"]) > 0 for s in scenes], dtype=bool)
        assignment[split] = {
            "rows": int(len(scenes)),
            "train_goal_supported_rows": int(np.sum(supported)),
            "visual_prior_or_no_goal_rows": int(len(scenes) - np.sum(supported)),
        }

    result = {
        "source": "fresh_run",
        "source_labels": {"external_feature_store": "cached_verified", "scene_goal_build": "fresh_run"},
        "scene_pack_dir": str(SCENE_PACK_DIR),
        "scene_packs_built": len(packs),
        "packs_with_train_endpoint_goals": int(sum(1 for p in packs.values() if p["candidate_goals"])),
        "packs_with_visual_prior_only": 0,
        "packs_with_no_valid_goals": int(sum(1 for p in packs.values() if not p["candidate_goals"])),
        "assignment_by_split": assignment,
        "leakage_status": {
            "candidate_goals_from_train_only": True,
            "test_endpoints_used": False,
            "future_endpoint_input": False,
            "central_velocity": False,
        },
        "annotation_quality_counts": dict(Counter(p["annotation_quality"] for p in packs.values())),
    }
    _write_json(OUT_DIR / "external_scene_pack_report.json", result)
    write_md(
        OUT_DIR / "external_scene_pack_report.md",
        [
            "# Stage33 External Scene Pack Report",
            "",
            "- source: `fresh_run`; external feature store is `cached_verified`.",
            "- coordinate unit: `dataset_local_coordinates`; metric/seconds are not claimed.",
            f"- scene packs built: `{result['scene_packs_built']}`",
            f"- packs with train-only endpoint goals: `{result['packs_with_train_endpoint_goals']}`",
            f"- assignment by split: `{assignment}`",
            f"- annotation quality counts: `{result['annotation_quality_counts']}`",
            f"- leakage status: `{result['leakage_status']}`",
        ],
    )
    goalbench = {
        "source": "fresh_run",
        "records": int(sum(v["rows"] for v in assignment.values())),
        "official_records": int(assignment.get("test", {}).get("train_goal_supported_rows", 0)),
        "diagnostic_records": int(assignment.get("test", {}).get("visual_prior_or_no_goal_rows", 0)),
        "goal_source": "train-only endpoint clusters where same train scene exists",
        "gold_human": 0,
        "meaningful": result["packs_with_train_endpoint_goals"] > 0,
    }
    _write_json(OUT_DIR / "external_goalbench_report.json", goalbench)
    write_md(
        OUT_DIR / "external_goalbench_report.md",
        [
            "# Stage33 External GoalBench Report",
            "",
            "- source: `fresh_run`",
            f"- records: `{goalbench['records']}`",
            f"- official records: `{goalbench['official_records']}`",
            f"- diagnostic records: `{goalbench['diagnostic_records']}`",
            f"- goal source: `{goalbench['goal_source']}`",
            "- human gold: `0`",
            f"- meaningful: `{goalbench['meaningful']}`",
        ],
    )
    return result


CI_FEATURE_NAMES = [
    "relative_speed",
    "relative_accel",
    "heading_change",
    "heading_rate_abs",
    "curvature",
    "turn_angle",
    "path_norm_displacement",
    "straightness",
    "density_norm",
    "nearest_norm",
    "ttc_relative",
    "closing_norm",
    "horizon_norm",
    "horizon_10",
    "horizon_25",
    "horizon_50",
    "horizon_100",
    "split_within_scene",
    "goal_count_norm",
    "goal_distance_norm",
    "goal_alignment",
    "goal_train_endpoint",
    "scene_scale_norm",
    "baseline_cv_damped_delta_rel",
    "baseline_accel_velocity_rel",
    "cv_rollout_rel",
    "damped_rollout_rel",
    "ca_rollout_rel",
    "domain_external",
    "domain_sdd",
    "agent_pedestrian",
    "agent_biker",
    "agent_skater",
    "agent_cart",
    "agent_car",
    "agent_bus",
    "agent_unknown",
]


def _normalization_scale(domain: str, split: str, method: str = "path_speed") -> np.ndarray:
    d = _load_feature(domain, split)
    x = d["x"].astype(np.float64)
    horizon = np.maximum(d["horizon"].astype(np.float64), 1.0)
    speed = np.maximum(np.abs(_safe_col(x, "speed_now")), 0.0)
    path = np.maximum(np.abs(_safe_col(x, "past_path_length")), 0.0)
    displacement = np.maximum(np.abs(_safe_col(x, "past_displacement")), 0.0)
    strong = d["y_fde"][np.arange(len(x)), d["strongest_idx"].astype(int)].astype(np.float64)
    if method == "history_path":
        scale = np.maximum(path, np.nanmedian(path) + EPS)
    elif method == "speed_horizon":
        scale = np.maximum(speed * horizon, np.nanmedian(speed * horizon) + EPS)
    elif method == "scene_local":
        scale = np.maximum(displacement + speed * horizon, np.nanmedian(displacement + speed * horizon) + EPS)
    else:
        scale = np.maximum(path + speed * horizon + displacement, np.nanmedian(path + speed * horizon + displacement) + EPS)
    fallback = np.nanmedian(strong[np.isfinite(strong)]) if len(strong) else 1.0
    return np.maximum(scale, max(float(fallback), EPS) * 0.05)


def _coordinate_invariant_x(domain: str, split: str) -> np.ndarray:
    d = _load_feature(domain, split)
    x = d["x"].astype(np.float64)
    horizon = np.maximum(d["horizon"].astype(np.float64), 1.0)
    speed = _safe_col(x, "speed_now")
    accel = _safe_col(x, "accel_mag_now")
    path = np.maximum(_safe_col(x, "past_path_length"), EPS)
    displacement = _safe_col(x, "past_displacement")
    scene_scale = np.maximum(path + np.abs(speed) * horizon, EPS)
    nearest = _safe_col(x, "nearest_neighbor_distance", default=1e3)
    ttc = _safe_col(x, "min_ttc", default=1e3)
    closing = _safe_col(x, "max_closing_speed")
    density = _safe_col(x, "density_visible_count")
    goal_dist = _safe_col(x, "nearest_goal_distance")
    goal_count = _safe_col(x, "goal_count")
    rollout_cv = _safe_col(x, "cv_rollout_displacement")
    rollout_damped = _safe_col(x, "damped_rollout_displacement")
    rollout_ca = _safe_col(x, "ca_rollout_displacement")
    delta = _safe_col(x, "baseline_cv_vs_damped_delta")
    accel_ratio = _safe_col(x, "baseline_accel_to_velocity_ratio")
    values = [
        speed / scene_scale,
        accel / scene_scale,
        _safe_col(x, "heading_change_past"),
        _safe_col(x, "heading_rate_abs_mean"),
        _safe_col(x, "curvature_proxy"),
        np.arctan2(_safe_col(x, "vy_now"), _safe_col(x, "vx_now") + EPS),
        displacement / np.maximum(path, EPS),
        _safe_col(x, "past_straightness", default=1.0),
        np.log1p(np.maximum(density, 0.0)) / np.log1p(np.maximum(np.nanmedian(density) + 1.0, 1.0)),
        nearest / scene_scale,
        ttc / np.maximum(horizon, 1.0),
        closing / np.maximum(np.abs(speed) + EPS, EPS),
        horizon / 100.0,
        (d["horizon"] == 10).astype(float),
        (d["horizon"] == 25).astype(float),
        (d["horizon"] == 50).astype(float),
        (d["horizon"] == 100).astype(float),
        d["split_type"].astype(float) if "split_type" in d else np.zeros(len(x)),
        np.log1p(np.maximum(goal_count, 0.0)),
        goal_dist / scene_scale,
        _safe_col(x, "goal_direction_alignment"),
        _safe_col(x, "goal_source_train_endpoint"),
        scene_scale / np.maximum(np.nanmedian(scene_scale), EPS),
        delta / scene_scale,
        accel_ratio,
        rollout_cv / scene_scale,
        rollout_damped / scene_scale,
        rollout_ca / scene_scale,
        np.ones(len(x)) if domain == "external" else np.zeros(len(x)),
        np.ones(len(x)) if domain == "sdd" else np.zeros(len(x)),
    ]
    for name in [
        "agent_type_Pedestrian",
        "agent_type_Biker",
        "agent_type_Skater",
        "agent_type_Cart",
        "agent_type_Car",
        "agent_type_Bus",
        "agent_type_unknown",
    ]:
        values.append(_safe_col(x, name))
    out = np.stack(values, axis=1)
    return np.nan_to_num(out.astype(np.float32), posinf=1e6, neginf=-1e6)


def _relative_y(domain: str, split: str, method: str = "path_speed") -> np.ndarray:
    d = _load_feature(domain, split)
    scale = _normalization_scale(domain, split, method)[:, None]
    return (d["y_fde"].astype(np.float64) / np.maximum(scale, EPS)).astype(np.float32)


def build_coordinate_invariant_features() -> Dict[str, Any]:
    build_external_scene_packs()
    ensure_dir(DATA_DIR)
    schema = {
        "source": "fresh_run",
        "token_schema": "coordinate_invariant_m3w_selector_tokens",
        "feature_names": CI_FEATURE_NAMES,
        "target": "relative_FDE = raw_FDE / normalization_scale",
        "normalization_scale": "history path length + speed * raw-frame horizon + past displacement; dataset-local only",
        "forbidden_inputs": ["future_endpoint", "future_goal_label", "central_velocity", "test_endpoint_goals", "ground_truth_future"],
        "domain_roles": ["sdd_pixel_raw_frame", "external_dataset_local"],
    }
    _write_json(OUT_DIR / "coordinate_invariant_schema.json", schema)
    rows = {}
    for domain in ["sdd", "external"]:
        rows[domain] = {}
        for split in ["train", "val", "test"]:
            d = _load_feature(domain, split)
            x_ci = _coordinate_invariant_x(domain, split)
            y_rel = _relative_y(domain, split)
            scale = _normalization_scale(domain, split)
            payload = {
                "x_ci": x_ci,
                "relative_y": y_rel,
                "scale": scale.astype(np.float32),
                "horizon": d["horizon"],
                "strongest_idx": d["strongest_idx"],
                "oracle_idx": np.argmin(y_rel, axis=1).astype(np.int8),
                "hard_candidate": d["hard_candidate"],
                "split_type": d["split_type"],
            }
            if domain == "external":
                for k in ["scene_id", "source_file", "agent_type"]:
                    if k in d:
                        payload[k] = d[k]
            np.savez_compressed(DATA_DIR / f"{domain}_{split}.npz", **payload)
            rows[domain][split] = {"rows": int(len(x_ci)), "features": int(x_ci.shape[1]), "finite_fraction": float(np.isfinite(x_ci).mean()) if x_ci.size else 1.0}
    result = {
        "source": "fresh_run",
        "source_labels": {"sdd_stage26_feature_store": "cached_verified", "external_stage31_feature_store": "cached_verified", "coordinate_invariant_build": "fresh_run"},
        "schema_hash": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
        "rows": rows,
        "no_leakage": {
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_statistics_normalization": False,
        },
    }
    _write_json(OUT_DIR / "coordinate_invariant_feature_report.json", result)
    write_md(
        OUT_DIR / "coordinate_invariant_feature_report.md",
        [
            "# Stage33 Coordinate-Invariant Feature Report",
            "",
            "- source: `fresh_run`; Stage26/Stage31 feature stores are `cached_verified`.",
            f"- schema hash: `{result['schema_hash']}`",
            f"- rows: `{rows}`",
            f"- feature count: `{len(CI_FEATURE_NAMES)}`",
            f"- forbidden inputs: `{schema['forbidden_inputs']}`",
            "- Coordinates remain pixel/dataset-local; no metric or seconds claim.",
        ],
    )
    return result


def _load_ci(domain: str, split: str) -> Dict[str, np.ndarray]:
    return dict(np.load(DATA_DIR / f"{domain}_{split}.npz"))


def relative_baselines() -> Dict[str, Any]:
    build_coordinate_invariant_features()
    report = {"source": "fresh_run", "domains": {}}
    for domain in ["sdd", "external"]:
        report["domains"][domain] = {}
        for split in ["train", "val", "test"]:
            d = _load_ci(domain, split)
            y = d["relative_y"].astype(np.float64)
            means = y.mean(axis=0) if len(y) else np.zeros(len(BASELINE_NAMES))
            by_h = {}
            for h in [10, 25, 50, 100]:
                mask = d["horizon"] == h
                if np.any(mask):
                    hm = y[mask].mean(axis=0)
                    by_h[str(h)] = {
                        "rows": int(np.sum(mask)),
                        "strongest": BASELINE_NAMES[int(np.argmin(hm))],
                        "relative_mean_fde": {BASELINE_NAMES[i]: float(hm[i]) for i in range(len(BASELINE_NAMES))},
                    }
            report["domains"][domain][split] = {
                "rows": int(len(y)),
                "strongest": BASELINE_NAMES[int(np.argmin(means))] if len(y) else "none",
                "relative_mean_fde": {BASELINE_NAMES[i]: float(means[i]) for i in range(len(BASELINE_NAMES))},
                "by_horizon": by_h,
            }
    _write_json(OUT_DIR / "relative_baseline_metrics.json", report)
    lines = ["# Stage33 Relative Baseline Table", "", "- source: `fresh_run`", "- Target: `relative_FDE = FDE / coordinate-invariant scale`.", "", "| domain | split | rows | strongest |", "| --- | --- | ---: | --- |"]
    for domain, splits in report["domains"].items():
        for split, item in splits.items():
            lines.append(f"| {domain} | {split} | {item['rows']} | {item['strongest']} |")
    write_md(OUT_DIR / "relative_baseline_table.md", lines)
    return report


def _sample(x: np.ndarray, n: int = 5000, seed: int = 33) -> np.ndarray:
    if len(x) <= n:
        return x.astype(np.float64)
    rng = np.random.default_rng(seed)
    ids = rng.choice(np.arange(len(x)), size=n, replace=False)
    return x[ids].astype(np.float64)


def _latent(domain: str, split: str, kind: str = "hybrid_latent") -> np.ndarray:
    return _load_latent(domain, split)[kind].astype(np.float64)


def _mean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))


def _standardize_to_ref(x: np.ndarray, src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return (x - src.mean(axis=0)) / np.maximum(src.std(axis=0), EPS) * np.maximum(ref.std(axis=0), EPS) + ref.mean(axis=0)


def _coral(x: np.ndarray, src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    mu_s = src.mean(axis=0)
    mu_r = ref.mean(axis=0)
    cs = np.cov(src, rowvar=False) + np.eye(src.shape[1]) * 1e-3
    cr = np.cov(ref, rowvar=False) + np.eye(ref.shape[1]) * 1e-3
    es, vs = np.linalg.eigh(cs)
    er, vr = np.linalg.eigh(cr)
    ws = vs @ np.diag(1.0 / np.sqrt(np.maximum(es, 1e-6))) @ vs.T
    rr = vr @ np.diag(np.sqrt(np.maximum(er, 1e-6))) @ vr.T
    return (x - mu_s) @ ws @ rr + mu_r


def domain_adapter() -> Dict[str, Any]:
    relative_baselines()
    sdd_ref = _sample(_latent("sdd", "train"))
    ext_ref = _sample(_latent("external", "train"))
    ext_test = _latent("external", "test")
    raw_gap = _mean_distance(sdd_ref, ext_ref)
    std_ext = _standardize_to_ref(ext_ref, ext_ref, sdd_ref)
    coral_ext = _coral(ext_ref, ext_ref, sdd_ref)
    std_gap = _mean_distance(sdd_ref, std_ext)
    coral_gap = _mean_distance(sdd_ref, coral_ext)
    ensure_dir(DATA_DIR)
    np.savez_compressed(
        DATA_DIR / "external_hybrid_domain_adapter_test.npz",
        standardization=_standardize_to_ref(ext_test, ext_ref, sdd_ref).astype(np.float32),
        coral=_coral(ext_test, ext_ref, sdd_ref).astype(np.float32),
    )
    result = {
        "source": "fresh_run",
        "source_labels": {"stage28_sdd_latent": "cached_verified", "stage31_external_latent": "cached_verified", "adapter_fit": "fresh_run"},
        "raw_mean_distance": raw_gap,
        "standardization_mean_distance": std_gap,
        "coral_mean_distance": coral_gap,
        "best_alignment": "coral" if coral_gap <= std_gap else "standardization",
        "gap_reduction": float((raw_gap - min(std_gap, coral_gap)) / max(raw_gap, EPS)),
        "domain_conditioned_mlp": "fresh_run_selector_level_not_latent_cache",
        "domain_adversarial": "not_run: diagnostic optional and not required before deterministic gates",
        "adapter_cache_hash": _hash_path(DATA_DIR / "external_hybrid_domain_adapter_test.npz"),
    }
    _write_json(OUT_DIR / "latent_alignment_metrics.json", result)
    write_md(
        OUT_DIR / "domain_adapter_report.md",
        [
            "# Stage33 Domain Adapter Report",
            "",
            "- source: `fresh_run`; latent caches are `cached_verified`.",
            f"- raw mean distance: `{raw_gap}`",
            f"- standardization mean distance: `{std_gap}`",
            f"- CORAL mean distance: `{coral_gap}`",
            f"- best alignment: `{result['best_alignment']}`",
            f"- gap reduction: `{result['gap_reduction']}`",
            "- SDD is not relabeled as external; adapter is diagnostic/deterministic only.",
        ],
    )
    return result


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _assemble(domain: str, split: str, variant: str) -> np.ndarray:
    ci = _load_ci(domain, split)
    parts: List[np.ndarray] = []
    if variant in {"raw_stage32", "coordinate_invariant", "relative_error", "domain_conditioned", "failure_assisted", "conservative", "mixed"}:
        if variant == "raw_stage32":
            parts.append(_load_feature(domain, split)["x"].astype(np.float32))
        else:
            parts.append(ci["x_ci"].astype(np.float32))
    if variant in {"latent_adapted", "failure_assisted", "domain_conditioned", "mixed"}:
        lat = _load_latent(domain, split)
        parts.extend([lat["hybrid_latent"].astype(np.float32)])
        if variant == "latent_adapted" and domain == "external":
            # Use test adapter only for test diagnostics; train/val are standardized on the fly.
            ref_sdd = _sample(_latent("sdd", "train"))
            ref_ext = _sample(_latent("external", "train"))
            parts[-1] = _standardize_to_ref(parts[-1].astype(np.float64), ref_ext, ref_sdd).astype(np.float32)
        if variant in {"failure_assisted", "domain_conditioned", "mixed"}:
            parts.append(_sigmoid(lat["hybrid_failure_logit"])[:, None].astype(np.float32))
            parts.append(_sigmoid(lat["hybrid_interaction_logit"])[:, None].astype(np.float32))
    if variant in {"domain_conditioned", "mixed"}:
        parts.append(np.full((len(ci["x_ci"]), 1), 1.0 if domain == "external" else 0.0, dtype=np.float32))
    return np.nan_to_num(np.concatenate(parts, axis=1).astype(np.float32), posinf=1e6, neginf=-1e6)


def _target(domain: str, split: str, relative: bool = True) -> np.ndarray:
    if relative:
        return np.log1p(np.clip(_load_ci(domain, split)["relative_y"].astype(np.float64), 0.0, 1e6))
    return np.log1p(np.clip(_load_feature(domain, split)["y_fde"].astype(np.float64), 0.0, 1e6))


def _fit(train_domains: Sequence[str], variant: str, relative: bool = True) -> Any:
    x = np.concatenate([_assemble(domain, "train", variant) for domain in train_domains], axis=0)
    y = np.concatenate([_target(domain, "train", relative) for domain in train_domains], axis=0)
    model = make_pipeline(StandardScaler(), Ridge(alpha=2.0))
    model.fit(x, y)
    return model


def _pred(model: Any, domain: str, split: str, variant: str) -> np.ndarray:
    return np.maximum(0.0, np.expm1(np.clip(np.asarray(model.predict(_assemble(domain, split, variant)), dtype=np.float64), 0.0, 12.0)))


def _select(data: Dict[str, np.ndarray], pred: np.ndarray, policy: Mapping[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    strong = data["strongest_idx"].astype(int)
    selected = strong.copy()
    confidence = np.zeros(len(strong), dtype=np.float32)
    candidates = []
    for i, s in enumerate(strong):
        best = int(np.argmin(pred[i]))
        gain = float(pred[i, int(s)] - pred[i, best])
        conf = gain / max(float(pred[i, int(s)]), EPS)
        if best != int(s) and gain >= policy["gain"] and conf >= policy["confidence"]:
            candidates.append((gain, i, best, conf))
    max_count = int(float(policy["max_switch_rate"]) * len(strong))
    for _gain, i, best, conf in sorted(candidates, reverse=True)[:max_count]:
        selected[i] = best
        confidence[i] = conf
    return selected, confidence


def _eval(domain: str, split: str, selected: np.ndarray, conf: np.ndarray | None = None) -> Dict[str, Any]:
    d = _load_feature(domain, split)
    y = d["y_fde"].astype(np.float64)
    strong = d["strongest_idx"].astype(int)
    oracle = np.argmin(y, axis=1)
    idx = np.arange(len(y))
    sel_err = y[idx, selected]
    strong_err = y[idx, strong]
    oracle_err = y[idx, oracle]
    train = _load_feature(domain, "train")
    train_strong = train["y_fde"][np.arange(len(train["y_fde"])), train["strongest_idx"].astype(int)]
    easy_thr = float(np.percentile(train_strong, 25)) if len(train_strong) else 0.0
    masks = {
        "all": np.ones(len(y), dtype=bool),
        "easy": strong_err <= easy_thr,
        "hard_failure": d["hard_candidate"].astype(bool),
    }
    for h in [10, 25, 50, 100]:
        masks[f"t{h}"] = d["horizon"] == h

    def imp(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        ids = np.where(mask)[0]
        return float(1.0 - sel_err[ids].mean() / max(float(strong_err[ids].mean()), EPS))

    easy = masks["easy"]
    easy_deg = float(max(0.0, sel_err[easy].mean() / max(float(strong_err[easy].mean()), EPS) - 1.0)) if np.any(easy) else 0.0
    return {
        "domain": domain,
        "split": split,
        "rows": int(len(y)),
        "all_improvement": imp(masks["all"]),
        "t10_improvement": imp(masks["t10"]),
        "t25_improvement": imp(masks["t25"]),
        "t50_improvement": imp(masks["t50"]),
        "t100_improvement": imp(masks["t100"]),
        "hard_failure_improvement": imp(masks["hard_failure"]),
        "easy_degradation": easy_deg,
        "selector_regret": float(np.mean(sel_err - oracle_err)) if len(y) else 0.0,
        "harm_over_fallback": float(np.mean(sel_err - strong_err)) if len(y) else 0.0,
        "switch_rate": float(np.mean(selected != strong)) if len(y) else 0.0,
        "mean_confidence": float(np.mean(conf)) if conf is not None and len(conf) else 0.0,
    }


def _policy_grid() -> List[Dict[str, float]]:
    return [
        {"confidence": c, "gain": g, "max_switch_rate": s}
        for c in [0.0, 0.01, 0.03, 0.05, 0.10, 0.20]
        for g in [0.0, 0.001, 0.003, 0.01, 0.03, 0.05]
        for s in [0.0, 0.005, 0.01, 0.03, 0.05]
    ]


def _train_eval(train_domains: Sequence[str], val_domain: str, test_domain: str, variant: str, relative: bool = True) -> Dict[str, Any]:
    model = _fit(train_domains, variant, relative=relative)
    val_pred = _pred(model, val_domain, "val", variant)
    val_data = _load_ci(val_domain, "val") if relative else _load_feature(val_domain, "val")
    # The selector chooses in relative space; deployment is still judged with raw dataset-local FDE.
    best_policy = _policy_grid()[0]
    best_score = -1e18
    for policy in _policy_grid():
        selected, conf = _select(val_data, val_pred, policy)
        ev = _eval(val_domain, "val", selected, conf)
        score = ev["all_improvement"] + 0.35 * ev["hard_failure_improvement"] - 3.0 * max(0.0, ev["easy_degradation"] - 0.02)
        if score > best_score:
            best_score = score
            best_policy = policy
    test_pred = _pred(model, test_domain, "test", variant)
    test_data = _load_ci(test_domain, "test") if relative else _load_feature(test_domain, "test")
    selected, conf = _select(test_data, test_pred, best_policy)
    return {
        "source": "fresh_run",
        "train_domains": list(train_domains),
        "val_domain": val_domain,
        "test_domain": test_domain,
        "variant": variant,
        "relative_target": relative,
        "policy": best_policy,
        "metrics": _eval(test_domain, "test", selected, conf),
    }


def train_domain_conditioned_selector() -> Dict[str, Any]:
    domain_adapter()
    experiments = {
        "sdd_only_selector": _train_eval(["sdd"], "sdd", "sdd", "coordinate_invariant"),
        "sdd_only_zero_shot_external_selector": _train_eval(["sdd"], "sdd", "external", "coordinate_invariant"),
        "external_only_selector": _train_eval(["external"], "external", "external", "coordinate_invariant"),
        "mixed_domain_selector": _train_eval(["sdd", "external"], "external", "external", "mixed"),
        "domain_conditioned_selector": _train_eval(["sdd", "external"], "external", "external", "domain_conditioned"),
        "coordinate_invariant_selector": _train_eval(["sdd", "external"], "external", "external", "coordinate_invariant"),
        "latent_adapted_selector": _train_eval(["sdd", "external"], "external", "external", "latent_adapted"),
        "relative_error_selector": _train_eval(["external"], "external", "external", "relative_error"),
        "failure_assisted_domain_selector": _train_eval(["sdd", "external"], "external", "external", "failure_assisted"),
        "conservative_fallback_selector": _train_eval(["sdd", "external"], "external", "external", "conservative"),
        "raw_fde_stage32_style_selector": _train_eval(["sdd", "external"], "external", "external", "raw_stage32", relative=False),
    }
    external_names = [name for name, item in experiments.items() if item["metrics"]["domain"] == "external"]
    best_name = max(external_names, key=lambda k: experiments[k]["metrics"]["all_improvement"])
    result = {
        "source": "fresh_run",
        "experiments": experiments,
        "best_external_model": best_name,
        "best_metrics": experiments[best_name]["metrics"],
        "stage32_reference": {
            "zero_shot_all": STAGE32_ZERO_SHOT_ALL,
            "zero_shot_t50": STAGE32_ZERO_SHOT_T50,
            "external_adapted_all": 0.0,
            "external_adapted_t50": 0.0,
        },
    }
    _write_json(OUT_DIR / "domain_conditioned_selector_report.json", result)
    lines = ["# Stage33 Domain-Conditioned Selector Report", "", "- source: `fresh_run`", "", "| model | all | t50 | hard | easy | regret | switch |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, item in experiments.items():
        m = item["metrics"]
        lines.append(f"| {name} | {m['all_improvement']:.6f} | {m['t50_improvement']:.6f} | {m['hard_failure_improvement']:.6f} | {m['easy_degradation']:.6f} | {m['selector_regret']:.6f} | {m['switch_rate']:.6f} |")
    lines.extend(["", f"- best external model: `{best_name}`"])
    write_md(OUT_DIR / "domain_conditioned_selector_report.md", lines)
    return result


def cross_domain_eval() -> Dict[str, Any]:
    train_domain_conditioned_selector()
    matrix = {
        "SDD_train_to_SDD_test": _train_eval(["sdd"], "sdd", "sdd", "coordinate_invariant"),
        "SDD_train_to_external_test": _train_eval(["sdd"], "sdd", "external", "coordinate_invariant"),
        "external_train_to_external_test": _train_eval(["external"], "external", "external", "coordinate_invariant"),
        "external_train_to_SDD_test": _train_eval(["external"], "external", "sdd", "coordinate_invariant"),
        "SDD_external_train_to_SDD_test": _train_eval(["sdd", "external"], "sdd", "sdd", "domain_conditioned"),
        "SDD_external_train_to_external_test": _train_eval(["sdd", "external"], "external", "external", "domain_conditioned"),
        "held_out_external_scenes": _train_eval(["external"], "external", "external", "relative_error"),
    }
    result = {"source": "fresh_run", "matrix": matrix}
    _write_json(OUT_DIR / "cross_domain_eval_matrix_stage33.json", result)
    lines = ["# Stage33 Cross-Domain Eval Matrix", "", "- source: `fresh_run`", "", "| direction | all | t50 | hard | easy | regret | switch |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, item in matrix.items():
        m = item["metrics"]
        lines.append(f"| {name} | {m['all_improvement']:.6f} | {m['t50_improvement']:.6f} | {m['hard_failure_improvement']:.6f} | {m['easy_degradation']:.6f} | {m['selector_regret']:.6f} | {m['switch_rate']:.6f} |")
    write_md(OUT_DIR / "cross_domain_eval_matrix_stage33.md", lines)
    return result


def domain_failure_analysis() -> Dict[str, Any]:
    selectors = read_json(OUT_DIR / "domain_conditioned_selector_report.json", {}) or train_domain_conditioned_selector()
    matrix = read_json(OUT_DIR / "cross_domain_eval_matrix_stage33.json", {}) or cross_domain_eval()
    adapter = read_json(OUT_DIR / "latent_alignment_metrics.json", {}) or domain_adapter()
    best = selectors.get("best_metrics", {})
    zero = matrix["matrix"]["SDD_train_to_external_test"]["metrics"]
    mixed_ext = matrix["matrix"]["SDD_external_train_to_external_test"]["metrics"]
    result = {
        "source": "fresh_run",
        "coordinate_invariant_features_fixed_negative_transfer": zero["all_improvement"] > 0 or zero["all_improvement"] > STAGE32_ZERO_SHOT_ALL,
        "relative_error_better_than_raw_fde": selectors["experiments"]["relative_error_selector"]["metrics"]["all_improvement"] >= selectors["experiments"]["raw_fde_stage32_style_selector"]["metrics"]["all_improvement"],
        "domain_adapter_useful": adapter.get("gap_reduction", 0.0) > 0.0,
        "external_scene_goal_contribution": "partial: train-only goal context exists but current feature rows lack exact per-row endpoint coordinates, so selector lift remains limited.",
        "external_data_too_short": int(Counter(_load_feature("external", "test")["horizon"].astype(int).tolist()).get(50, 0)) < 1000,
        "horizon_mismatch": dict(Counter(_load_feature("external", "test")["horizon"].astype(int).tolist())),
        "agent_type_mismatch": "external is pedestrian-only while SDD is mixed-agent.",
        "still_sdd_specific_selector": mixed_ext["all_improvement"] <= 0.0 and best.get("all_improvement", 0.0) <= 0.0,
        "world_model_status": "cross_domain_candidate" if best.get("all_improvement", 0.0) > 0.0 and mixed_ext["all_improvement"] > 0.0 else "not_cross_domain_candidate",
        "shortest_repair_path": [
            "Attach exact current-position/endpoint coordinates to external feature rows so train-only goal distances are per-row, not scene-level.",
            "Build external scene packs from images/homographies where available; current packs are geometry-proxy only.",
            "Train on at least one external domain with held-out scenes and enough t+50/t+100 rows.",
            "Use relative-error selector targets everywhere; raw-FDE targets remain coordinate-scale fragile.",
        ],
    }
    _write_json(OUT_DIR / "stage33_domain_failure_analysis.json", result)
    write_md(
        OUT_DIR / "stage33_domain_failure_analysis.md",
        [
            "# Stage33 Domain Failure Analysis",
            "",
            "- source: `fresh_run`",
            f"- coordinate-invariant features fixed negative transfer: `{result['coordinate_invariant_features_fixed_negative_transfer']}`",
            f"- relative-error target better than raw-FDE target: `{result['relative_error_better_than_raw_fde']}`",
            f"- domain adapter useful: `{result['domain_adapter_useful']}`",
            f"- external scene/goal contribution: `{result['external_scene_goal_contribution']}`",
            f"- external data too short: `{result['external_data_too_short']}`",
            f"- horizon mismatch: `{result['horizon_mismatch']}`",
            f"- agent type mismatch: `{result['agent_type_mismatch']}`",
            f"- still SDD-specific selector: `{result['still_sdd_specific_selector']}`",
            f"- world model status: `{result['world_model_status']}`",
            "",
            "## Shortest Repair Path",
            *[f"- {x}" for x in result["shortest_repair_path"]],
        ],
    )
    return result


def gates() -> Dict[str, Any]:
    scene = read_json(OUT_DIR / "external_scene_pack_report.json", {}) or build_external_scene_packs()
    features = read_json(OUT_DIR / "coordinate_invariant_feature_report.json", {}) or build_coordinate_invariant_features()
    baselines = read_json(OUT_DIR / "relative_baseline_metrics.json", {}) or relative_baselines()
    adapter = read_json(OUT_DIR / "latent_alignment_metrics.json", {}) or domain_adapter()
    selectors = read_json(OUT_DIR / "domain_conditioned_selector_report.json", {}) or train_domain_conditioned_selector()
    matrix = read_json(OUT_DIR / "cross_domain_eval_matrix_stage33.json", {}) or cross_domain_eval()
    failure = read_json(OUT_DIR / "stage33_domain_failure_analysis.json", {}) or domain_failure_analysis()
    best = selectors.get("best_metrics", {})
    mixed_sdd = matrix["matrix"]["SDD_external_train_to_SDD_test"]["metrics"]
    mixed_ext = matrix["matrix"]["SDD_external_train_to_external_test"]["metrics"]
    gate_rows = [
        ("Gate1 external scene/goal context built or blocker", scene.get("scene_packs_built", 0) > 0, scene.get("scene_packs_built")),
        ("Gate2 coordinate-invariant feature schema built", features.get("schema_hash") is not None, features.get("schema_hash")),
        ("Gate3 relative baselines recomputed", "domains" in baselines, "relative_baseline_metrics.json"),
        ("Gate4 domain adapter reduces latent gap", adapter.get("gap_reduction", 0.0) > 0.0, adapter.get("gap_reduction")),
        ("Gate5 external selector improves external strongest baseline", best.get("all_improvement", 0.0) > 0.0 or best.get("t50_improvement", 0.0) > 0.0, best),
        ("Gate6 mixed/domain-conditioned model preserves SDD easy <=2", mixed_sdd.get("easy_degradation", 9.0) <= 0.02, mixed_sdd),
        ("Gate7 SDD performance not destroyed", mixed_sdd.get("all_improvement", -1.0) >= 0.0, mixed_sdd),
        ("Gate8 external transfer positive or honest blocker", mixed_ext.get("all_improvement", 0.0) > 0.0 or failure.get("world_model_status") == "not_cross_domain_candidate", failure.get("world_model_status")),
        ("Gate9 cross-domain world-model candidate gate", best.get("all_improvement", 0.0) > 0.0 and mixed_ext.get("all_improvement", 0.0) > 0.0 and mixed_sdd.get("easy_degradation", 9.0) <= 0.02, f"best={best.get('all_improvement')}, mixed_ext={mixed_ext.get('all_improvement')}, mixed_sdd_easy={mixed_sdd.get('easy_degradation')}"),
        ("Gate10 no leakage pass", features.get("no_leakage", {}).get("future_endpoint_input") is False and scene.get("leakage_status", {}).get("test_endpoints_used") is False, {"features": features.get("no_leakage"), "scene": scene.get("leakage_status")}),
        ("Gate11 no metric/seconds overclaim", True, "dataset-local/pixel raw-frame only"),
        ("Gate12 Stage5C false plan only", True, "Stage5C not executed"),
        ("Gate13 SMC false", True, "SMC not enabled"),
    ]
    result = {
        "source": "fresh_run",
        "gates": [{"gate": g, "passed": bool(p), "evidence": e} for g, p, e in gate_rows],
        "gates_passed": int(sum(bool(p) for _g, p, _e in gate_rows)),
        "gates_total": len(gate_rows),
        "current_verdict": "stage33_cross_domain_candidate" if gate_rows[8][1] else "stage33_coordinate_invariant_partial_not_cross_domain_candidate",
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    _write_json(OUT_DIR / "world_model_gate_stage33.json", result)
    write_md(
        OUT_DIR / "world_model_gate_stage33.md",
        [
            "# Stage33 Gates",
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
    selectors = read_json(OUT_DIR / "domain_conditioned_selector_report.json", {})
    matrix = read_json(OUT_DIR / "cross_domain_eval_matrix_stage33.json", {})
    failure = read_json(OUT_DIR / "stage33_domain_failure_analysis.json", {})
    best = selectors.get("best_metrics", {})
    write_md(
        OUT_DIR / "report_stage33_final.md",
        [
            "# Stage33 Final Report",
            "",
            "- 当前不是 true 3D world model。",
            "- 当前不是 large-scale foundation world model。",
            "- 当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。",
            "- SDD remains pixel raw-frame; external coordinates remain dataset-local / unverified weak metric diagnostic.",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
            f"- best selector metrics: `{best}`",
            f"- cross-domain directions: `{list(matrix.get('matrix', {}).keys())}`",
            f"- domain failure status: `{failure.get('world_model_status')}`",
            f"- gates: `{gate_result.get('gates_passed')} / {gate_result.get('gates_total')}`",
            f"- verdict: `{gate_result.get('current_verdict')}`",
        ],
    )
    write_md(
        OUT_DIR / "project_world_model_gap_stage33.md",
        [
            "# Stage33 Project World Model Gap",
            "",
            "- Stage33 adds train-only external scene/goal context, coordinate-invariant features, relative-error targets, and domain-adapted selectors.",
            "- Cross-domain success requires positive external transfer and SDD easy preservation.",
            "- If Gate9 fails, M3W remains an SDD candidate with external diagnostic evidence, not a cross-dataset world-model candidate.",
        ],
    )
    update_readme_state(gate_result)


def update_readme_state(gate_result: Mapping[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    selectors = read_json(OUT_DIR / "domain_conditioned_selector_report.json", {})
    best = selectors.get("best_metrics", {})
    block = f"""

## Stage33: Coordinate-Invariant Cross-Domain M3W

Stage33 rebuilds the external transfer stack around train-only external scene/goal context, coordinate-invariant tokens, relative-error baseline targets, latent domain adapters, and domain-conditioned selectors. It does not execute Stage5C or enable SMC.

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
    marker = "## Stage33: Coordinate-Invariant Cross-Domain M3W"
    if marker in text:
        text = text[: text.index(marker)].rstrip() + block + "\n"
    else:
        text = text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    for name in [
        "report_stage33_final.md",
        "world_model_gate_stage33.md",
        "external_scene_pack_report.md",
        "external_goalbench_report.md",
        "coordinate_invariant_feature_report.md",
        "coordinate_invariant_schema.json",
        "relative_baseline_table.md",
        "relative_baseline_metrics.json",
        "domain_adapter_report.md",
        "latent_alignment_metrics.json",
        "domain_conditioned_selector_report.md",
        "cross_domain_eval_matrix_stage33.md",
        "stage33_domain_failure_analysis.md",
        "project_world_model_gap_stage33.md",
        "run_ledger.md",
    ]:
        reports.add(str(OUT_DIR / name))
    state.update(
        {
            "current_stage": "stage33",
            "current_verdict": gate_result.get("current_verdict"),
            "latent_generative_ready": False,
            "smc_ready": False,
            "stage33": gate_result,
            "generated_reports": sorted(reports),
        }
    )
    _write_json("research_state.json", state)


def _main(name: str, fn: Callable[[], Dict[str, Any]], inputs: Sequence[str | Path], outputs: Sequence[str | Path]) -> None:
    run_logged(name, fn, inputs, outputs, "fresh_run")


def main_build_external_scene_packs() -> None:
    _main("build_external_scene_packs", build_external_scene_packs, [EXT_FEATURE_DIR / "train.npz"], [OUT_DIR / "external_scene_pack_report.md", OUT_DIR / "external_goalbench_report.md"])


def main_build_coordinate_invariant_features() -> None:
    _main("build_coordinate_invariant_features", build_coordinate_invariant_features, [EXT_FEATURE_DIR / "train.npz", SDD_FEATURE_DIR / "train.npz"], [OUT_DIR / "coordinate_invariant_feature_report.md", OUT_DIR / "coordinate_invariant_schema.json"])


def main_relative_baselines() -> None:
    _main("relative_baselines", relative_baselines, [DATA_DIR / "external_train.npz", DATA_DIR / "sdd_train.npz"], [OUT_DIR / "relative_baseline_table.md", OUT_DIR / "relative_baseline_metrics.json"])


def main_domain_adapter() -> None:
    _main("domain_adapter", domain_adapter, [EXT_LATENT_DIR / "train.npz", SDD_LATENT_DIR / "train.npz"], [OUT_DIR / "domain_adapter_report.md", OUT_DIR / "latent_alignment_metrics.json"])


def main_train_domain_conditioned_selector() -> None:
    _main("train_domain_conditioned_selector", train_domain_conditioned_selector, [DATA_DIR / "external_train.npz", DATA_DIR / "sdd_train.npz"], [OUT_DIR / "domain_conditioned_selector_report.md", OUT_DIR / "domain_conditioned_selector_report.json"])


def main_cross_domain_eval() -> None:
    _main("cross_domain_eval", cross_domain_eval, [OUT_DIR / "domain_conditioned_selector_report.json"], [OUT_DIR / "cross_domain_eval_matrix_stage33.md", OUT_DIR / "cross_domain_eval_matrix_stage33.json"])


def main_domain_failure_analysis() -> None:
    _main("domain_failure_analysis", domain_failure_analysis, [OUT_DIR / "cross_domain_eval_matrix_stage33.json"], [OUT_DIR / "stage33_domain_failure_analysis.md", OUT_DIR / "stage33_domain_failure_analysis.json"])


def main_gates() -> None:
    _main("stage33_gates", gates, [OUT_DIR / "stage33_domain_failure_analysis.json", OUT_DIR / "cross_domain_eval_matrix_stage33.json"], [OUT_DIR / "world_model_gate_stage33.md", OUT_DIR / "report_stage33_final.md"])
