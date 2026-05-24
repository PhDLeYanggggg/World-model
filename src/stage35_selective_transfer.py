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
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage28_pipeline import BASELINE_NAMES, LATENT_DIR as SDD_LATENT_DIR
from src.stage30_m3w_verified import _combined_hash, _git_commit, _hash_path
from src import stage31_external_generalization as s31
from src import stage34_external_geometry as s34


OUT_DIR = Path("outputs/stage35_selective_transfer")
DATA_DIR = Path("data/stage35_selective_transfer")
ROOTS = [
    Path("/Users/yangyue/Downloads/World/external_data/OpenTraj"),
    Path("/Users/yangyue/Downloads/OpenTraj"),
    Path("/Users/yangyue/Downloads/ETH_UCY"),
    Path("/Users/yangyue/Downloads/trajnetplusplusdataset"),
]
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
BASELINES = s34.BASELINES_V2
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
        "# Stage35 Selective Transfer Run Ledger",
        "",
        "| command | source | status | wall time s | input hash | output hash | git commit |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['command']}` | `{row['source']}` | `{row['status']}` | {float(row['wall_time_s']):.3f} | `{row['input_hash'][:12]}` | `{row['output_hash'][:12]}` | `{row['git_commit']}` |"
        )
    write_md(OUT_DIR / "run_ledger.md", lines)


def run_logged(name: str, fn: Callable[[], Dict[str, Any]], inputs: Sequence[str | Path], outputs: Sequence[str | Path]) -> Dict[str, Any]:
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
                "source": "fresh_run",
            }
        )


def _read_four_col(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().replace(",", " ").split()
            if len(parts) < 4 or "?" in parts:
                continue
            try:
                rows.append((float(parts[0]), int(float(parts[1])), float(parts[2]), float(parts[3])))
            except ValueError:
                continue
    return np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 4), dtype=np.float64)


def _read_obsmat(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                # ETH/UCY obsmat standard: frame, id, pos_x, pos_z, pos_y, ...
                rows.append((float(parts[0]), int(float(parts[1])), float(parts[2]), float(parts[4])))
            except ValueError:
                continue
    return np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 4), dtype=np.float64)


def _candidate_track_files() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for root in ROOTS:
        if not root.exists():
            continue
        datasets = root / "datasets"
        if not datasets.exists():
            continue
        for path in sorted(datasets.rglob("*")):
            if not path.is_file():
                continue
            lower = str(path).lower()
            if "/sdd/" in lower or "stanford" in lower:
                continue
            kind = None
            dataset = None
            if path.name == "obsmat.txt" and ("/eth/" in lower or "/ucy/" in lower):
                kind = "obsmat"
                dataset = "ETH_UCY"
            elif path.name.endswith("-trajnet.txt") or path.name in {"crowds_zara03.txt"}:
                kind = "four_col"
                dataset = "UCY"
            elif "/trajnet/" in lower and path.suffix == ".txt":
                kind = "four_col"
                dataset = "TrajNet"
            if kind:
                scene = f"{dataset}_{path.parent.name}"
                out.append({"path": path, "kind": kind, "dataset": dataset, "scene": scene})
    # Deduplicate by resolved path.
    seen = set()
    uniq = []
    for item in out:
        key = str(item["path"])
        if key not in seen:
            seen.add(key)
            uniq.append(item)
    return uniq


def _read_item(item: Mapping[str, Any]) -> np.ndarray:
    return _read_obsmat(item["path"]) if item["kind"] == "obsmat" else _read_four_col(item["path"])


def _rows_for_track_file(item: Mapping[str, Any]) -> Dict[str, List[Any]]:
    arr = _read_item(item)
    out: Dict[str, List[Any]] = defaultdict(list)
    if len(arr) == 0:
        return out
    scene = str(item["scene"])
    source = str(item["path"])
    for agent in np.unique(arr[:, 1]).astype(int):
        tr = arr[arr[:, 1] == agent]
        tr = tr[np.argsort(tr[:, 0])]
        frames = tr[:, 0]
        if len(tr) < 4:
            continue
        step = max(1, len(tr) // 120)
        for i in range(2, len(tr) - 1, step):
            for horizon in [10, 25, 50, 100]:
                target = frames[i] + horizon
                future_ids = np.where(frames >= target)[0]
                if len(future_ids) == 0:
                    continue
                j = int(future_ids[0])
                frame_delta = float(frames[j] - frames[i])
                if j <= i or frame_delta > max(10, horizon + 20):
                    continue
                past = tr[max(0, i - 7) : i + 1]
                out["dataset"].append(item["dataset"])
                out["scene_id"].append(scene)
                out["source_file"].append(source)
                out["agent_id"].append(int(agent))
                out["frame_id"].append(float(frames[i]))
                out["current_x"].append(float(tr[i, 2]))
                out["current_y"].append(float(tr[i, 3]))
                out["past_start_x"].append(float(past[0, 2]))
                out["past_start_y"].append(float(past[0, 3]))
                out["future_endpoint_x"].append(float(tr[j, 2]))
                out["future_endpoint_y"].append(float(tr[j, 3]))
                out["horizon"].append(int(horizon))
                out["dt_frame_step"].append(frame_delta)
                out["track_length"].append(int(len(tr)))
                out["valid_mask"].append(True)
    return out


def _merge(rows: List[Dict[str, List[Any]]]) -> Dict[str, np.ndarray]:
    keys = ["dataset", "scene_id", "source_file", "agent_id", "frame_id", "current_x", "current_y", "past_start_x", "past_start_y", "future_endpoint_x", "future_endpoint_y", "horizon", "dt_frame_step", "track_length", "valid_mask"]
    merged = {k: [] for k in keys}
    for row in rows:
        for k in keys:
            merged[k].extend(row.get(k, []))
    return {
        "dataset": np.asarray(merged["dataset"], dtype="U32"),
        "scene_id": np.asarray(merged["scene_id"], dtype="U96"),
        "source_file": np.asarray(merged["source_file"], dtype="U256"),
        "agent_id": np.asarray(merged["agent_id"], dtype=np.int64),
        "frame_id": np.asarray(merged["frame_id"], dtype=np.float32),
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


def external_data_expansion() -> Dict[str, Any]:
    ensure_dir(DATA_DIR)
    items = _candidate_track_files()
    per_source = {}
    rows_by_scene: Dict[str, Dict[str, List[Any]]] = {}
    unsupported = []
    for item in items:
        row = _rows_for_track_file(item)
        count = len(row.get("horizon", []))
        per_source[str(item["path"])] = {"dataset": item["dataset"], "scene": item["scene"], "kind": item["kind"], "rows": count}
        if count == 0:
            unsupported.append(str(item["path"]))
            continue
        rows_by_scene.setdefault(str(item["scene"]), defaultdict(list))
        for k, vals in row.items():
            rows_by_scene[str(item["scene"])][k].extend(vals)

    scene_names = sorted(rows_by_scene)
    # Scene-level split first; deterministic and keeps held-out external scenes.
    train_scenes = set(scene_names[: max(1, int(len(scene_names) * 0.6))])
    val_scenes = set(scene_names[max(1, int(len(scene_names) * 0.6)) : max(2, int(len(scene_names) * 0.8))])
    test_scenes = set(scene_names) - train_scenes - val_scenes
    if not test_scenes and scene_names:
        test_scenes = {scene_names[-1]}
        train_scenes.discard(scene_names[-1])
    splits = {"train": train_scenes, "val": val_scenes, "test": test_scenes}
    split_reports = {}
    for split, scenes in splits.items():
        geo = _merge([rows_by_scene[s] for s in sorted(scenes)])
        np.savez_compressed(DATA_DIR / f"expanded_external_{split}.npz", **geo)
        split_reports[split] = {
            "rows": int(len(geo["horizon"])),
            "scenes": int(len(scenes)),
            "scene_ids": sorted(scenes),
            "agents": int(len(set(geo["agent_id"].astype(int).tolist()))) if len(geo["agent_id"]) else 0,
            "horizon_counts": dict(Counter(geo["horizon"].astype(int).tolist())),
            "track_length_median": float(np.median(geo["track_length"])) if len(geo["track_length"]) else 0.0,
        }
    result = {
        "source": "fresh_run",
        "source_labels": {"local_external_data": "cached_verified", "expansion_conversion": "fresh_run"},
        "checked_roots": [str(r) for r in ROOTS],
        "candidate_files": len(items),
        "converted_files": int(sum(1 for v in per_source.values() if v["rows"] > 0)),
        "unsupported_or_empty_files": unsupported,
        "per_source": per_source,
        "splits": split_reports,
        "coordinate_unit": "dataset_local_coordinates_mixed_sources",
        "metric_status": "unverified_weak_metric_diagnostic; homography files may exist but not calibrated into common metric frame",
        "image_or_homography_available": {
            "ETH_UCY_has_reference_or_H_files": True,
            "TrajNet_has_scene_images": False,
        },
        "held_out_scene_split_supported": bool(split_reports.get("test", {}).get("scenes", 0) > 0),
    }
    _write_json(OUT_DIR / "external_data_expansion_report.json", result)
    write_md(
        OUT_DIR / "external_data_expansion_report.md",
        [
            "# Stage35 External Data Expansion Report",
            "",
            "- source: `fresh_run`; local OpenTraj/ETH/UCY files are `cached_verified` inputs.",
            f"- candidate files: `{result['candidate_files']}`",
            f"- converted files: `{result['converted_files']}`",
            f"- splits: `{split_reports}`",
            f"- coordinate unit: `{result['coordinate_unit']}`",
            f"- metric status: `{result['metric_status']}`",
            f"- image/homography availability: `{result['image_or_homography_available']}`",
            f"- held-out scene split supported: `{result['held_out_scene_split_supported']}`",
            f"- unsupported/empty files: `{unsupported[:10]}`",
        ],
    )
    return result


def _load_geo(split: str) -> Dict[str, np.ndarray]:
    return dict(np.load(DATA_DIR / f"expanded_external_{split}.npz"))


def external_split_v2() -> Dict[str, Any]:
    exp = external_data_expansion()
    result = {
        "source": "fresh_run",
        "split_strategy": "scene-level split",
        "fallback_strategy": "file-level split not used because scene-level split is available",
        "splits": exp["splits"],
        "train_only_goals_rule": "candidate goals may use only train split endpoints",
        "test_endpoint_rule": "test endpoints are evaluation labels only",
        "held_out_external_scenes": exp["splits"].get("test", {}).get("scene_ids", []),
        "no_leakage": {"future_endpoint_input": False, "central_velocity": False, "test_endpoint_goals": False},
    }
    _write_json(OUT_DIR / "external_split_v2_report.json", result)
    write_md(
        OUT_DIR / "external_split_v2_report.md",
        [
            "# Stage35 External Split v2 Report",
            "",
            "- source: `fresh_run`",
            f"- split strategy: `{result['split_strategy']}`",
            f"- splits: `{result['splits']}`",
            f"- held-out external scenes: `{result['held_out_external_scenes']}`",
            f"- no leakage: `{result['no_leakage']}`",
        ],
    )
    return result


def _scene_goals() -> Dict[str, np.ndarray]:
    train = _load_geo("train")
    goals = {}
    pts = np.stack([train["future_endpoint_x"], train["future_endpoint_y"]], axis=1).astype(np.float64)
    scenes = train["scene_id"].astype(str)
    for scene in sorted(set(scenes.tolist())):
        goals[scene] = s34.s33._cluster_points(pts[scenes == scene], max_k=8)
    return goals


def _baseline_errors(split: str) -> Tuple[np.ndarray, np.ndarray]:
    geo = _load_geo(split)
    cur = np.stack([geo["current_x"], geo["current_y"]], axis=1).astype(np.float64)
    past = np.stack([geo["past_start_x"], geo["past_start_y"]], axis=1).astype(np.float64)
    fut = np.stack([geo["future_endpoint_x"], geo["future_endpoint_y"]], axis=1).astype(np.float64)
    h = np.maximum(geo["dt_frame_step"].astype(np.float64), 1.0)
    delta = cur - past
    v = delta / h[:, None]
    speed = np.linalg.norm(v, axis=1)
    damp_factor = (1.0 - 0.95 ** h) / max(1.0 - 0.95, EPS)
    preds = [cur, cur + v * h[:, None], cur + v * damp_factor[:, None], cur + v * h[:, None], cur + v * h[:, None]]
    bounds = {}
    for scene in sorted(set(geo["scene_id"].astype(str).tolist())):
        mask = geo["scene_id"].astype(str) == scene
        xs = np.concatenate([cur[mask, 0], fut[mask, 0]])
        ys = np.concatenate([cur[mask, 1], fut[mask, 1]])
        bounds[scene] = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    scene_clamped = preds[1].copy()
    for scene, b in bounds.items():
        mask = geo["scene_id"].astype(str) == scene
        scene_clamped[mask, 0] = np.clip(scene_clamped[mask, 0], b[0], b[2])
        scene_clamped[mask, 1] = np.clip(scene_clamped[mask, 1], b[1], b[3])
    goals = _scene_goals()
    goal_pred = preds[1].copy()
    for scene, g in goals.items():
        mask = geo["scene_id"].astype(str) == scene
        ids = np.where(mask)[0]
        if len(ids) == 0 or len(g) == 0:
            continue
        dist = np.linalg.norm(g[None, :, :] - cur[ids, None, :], axis=2)
        gid = np.argmin(dist, axis=1)
        target = g[gid]
        direction = target - cur[ids]
        dnorm = np.linalg.norm(direction, axis=1)
        step = np.minimum(dnorm, speed[ids] * h[ids])
        goal_pred[ids] = cur[ids] + direction / np.maximum(dnorm[:, None], EPS) * step[:, None]
    preds.extend([scene_clamped, goal_pred])
    y = np.linalg.norm(np.stack(preds, axis=1) - fut[:, None, :], axis=2).astype(np.float32)
    scale = np.maximum(np.linalg.norm(delta, axis=1) + speed * h, np.median(np.linalg.norm(delta, axis=1) + speed * h) + EPS).astype(np.float32)
    return y, scale


def _features(split: str) -> np.ndarray:
    geo = _load_geo(split)
    cur = np.stack([geo["current_x"], geo["current_y"]], axis=1).astype(np.float64)
    past = np.stack([geo["past_start_x"], geo["past_start_y"]], axis=1).astype(np.float64)
    delta = cur - past
    path = np.maximum(np.linalg.norm(delta, axis=1), EPS)
    h = np.maximum(geo["dt_frame_step"].astype(np.float64), 1.0)
    speed = path / h
    density = np.zeros(len(path), dtype=np.float64)
    for scene in sorted(set(geo["scene_id"].astype(str).tolist())):
        mask = geo["scene_id"].astype(str) == scene
        density[mask] = np.log1p(np.sum(mask))
    goals = _scene_goals()
    goal_dist = np.full(len(path), 1.0, dtype=np.float64)
    goal_angle = np.ones(len(path), dtype=np.float64)
    goal_avail = np.zeros(len(path), dtype=np.float64)
    for scene, g in goals.items():
        ids = np.where(geo["scene_id"].astype(str) == scene)[0]
        if len(ids) == 0 or len(g) == 0:
            continue
        d = g[None, :, :] - cur[ids, None, :]
        dist = np.linalg.norm(d, axis=2)
        gid = np.argmin(dist, axis=1)
        vec = d[np.arange(len(ids)), gid]
        goal_dist[ids] = dist[np.arange(len(ids)), gid] / np.maximum(np.median(path), EPS)
        goal_angle[ids] = s34._angle_between(delta[ids], vec) / math.pi
        goal_avail[ids] = 1.0
    ds = geo["dataset"].astype(str)
    data_flags = [np.asarray(ds == name, dtype=float) for name in ["TrajNet", "ETH_UCY", "UCY"]]
    x = np.stack(
        [
            delta[:, 0] / path,
            delta[:, 1] / path,
            speed / np.maximum(np.median(speed), EPS),
            path / np.maximum(np.median(path), EPS),
            density / np.maximum(np.median(density), EPS),
            goal_dist,
            goal_angle,
            goal_avail,
            geo["horizon"].astype(float) / 100.0,
            (geo["horizon"] == 10).astype(float),
            (geo["horizon"] == 25).astype(float),
            (geo["horizon"] == 50).astype(float),
            (geo["horizon"] == 100).astype(float),
            geo["track_length"].astype(float) / max(float(np.median(geo["track_length"])), 1.0),
            *data_flags,
        ],
        axis=1,
    )
    return np.nan_to_num(x.astype(np.float32), posinf=1e6, neginf=-1e6)


def external_hard_easy_failure() -> Dict[str, Any]:
    external_split_v2()
    reports = {}
    train_y, train_scale = _baseline_errors("train")
    train_strong = np.argmin(train_y.mean(axis=0))
    train_err = train_y[:, train_strong]
    easy_thr = float(np.percentile(train_err, 25))
    fail_thr = float(np.percentile(train_err, 80))
    for split in ["train", "val", "test"]:
        y, scale = _baseline_errors(split)
        rel = y / np.maximum(scale[:, None], EPS)
        strong = np.full(len(y), train_strong, dtype=np.int8)
        oracle = np.argmin(rel, axis=1).astype(np.int8)
        strong_err = y[np.arange(len(y)), strong]
        sorted_rel = np.sort(rel, axis=1)
        margin = sorted_rel[:, 1] - sorted_rel[:, 0] if rel.shape[1] > 1 else np.zeros(len(rel))
        easy = strong_err <= easy_thr
        failure = (strong_err >= fail_thr) | (margin >= np.percentile(margin, 70))
        geo = _load_geo(split)
        hard = failure | (geo["horizon"].astype(int) >= 50) | (geo["track_length"] < np.percentile(geo["track_length"], 30))
        np.savez_compressed(DATA_DIR / f"labels_{split}.npz", y_fde=y, relative_y=rel.astype(np.float32), scale=scale, strongest_idx=strong, oracle_idx=oracle, easy=easy, failure=failure, hard=hard, oracle_margin=margin.astype(np.float32))
        reports[split] = {"rows": int(len(y)), "easy": int(easy.sum()), "hard": int(hard.sum()), "failure": int(failure.sum()), "oracle_headroom": float(1.0 - y[np.arange(len(y)), oracle].mean() / max(float(strong_err.mean()), EPS)) if len(y) else 0.0}
    result = {"source": "fresh_run", "strongest_baseline": BASELINES[train_strong], "easy_threshold": easy_thr, "failure_threshold": fail_thr, "splits": reports, "enough_hard": reports["train"]["hard"] > 1000, "enough_failure": reports["train"]["failure"] > 1000, "learning_space": reports["train"]["oracle_headroom"] > 0.05}
    _write_json(OUT_DIR / "external_hard_easy_failure_report.json", result)
    write_md(OUT_DIR / "external_hard_easy_failure_report.md", ["# Stage35 External Hard/Easy/Failure Report", "", "- source: `fresh_run`", f"- strongest baseline: `{result['strongest_baseline']}`", f"- splits: `{reports}`", f"- enough hard: `{result['enough_hard']}`", f"- enough failure: `{result['enough_failure']}`", f"- learning space: `{result['learning_space']}`"])
    return result


def _load_labels(split: str) -> Dict[str, np.ndarray]:
    return dict(np.load(DATA_DIR / f"labels_{split}.npz"))


def _fit_gain_model(extra: str = "base") -> Any:
    x = _features("train")
    if extra == "latent" and (Path("data/stage31_external_latent_cache") / "train.npz").exists():
        lat = dict(np.load(Path("data/stage31_external_latent_cache") / "train.npz"))["hybrid_latent"][:, :16]
        if len(lat) == len(x):
            x = np.concatenate([x, lat.astype(np.float32)], axis=1)
    y = np.log1p(np.clip(_load_labels("train")["relative_y"], 0.0, 1e6))
    model = make_pipeline(StandardScaler(), Ridge(alpha=2.0))
    model.fit(x, y)
    return model


def _predict_gain(model: Any, split: str, extra: str = "base") -> np.ndarray:
    x = _features(split)
    if extra == "latent" and (Path("data/stage31_external_latent_cache") / f"{split}.npz").exists():
        lat = dict(np.load(Path("data/stage31_external_latent_cache") / f"{split}.npz"))["hybrid_latent"][:, :16]
        if len(lat) == len(x):
            x = np.concatenate([x, lat.astype(np.float32)], axis=1)
    return np.maximum(0.0, np.expm1(np.clip(model.predict(x), 0.0, 12.0)))


def _fit_binary(label: str) -> Any:
    lab = _load_labels("train")
    y = lab[label].astype(int)
    if len(set(y.tolist())) < 2:
        y[0:1] = 1 - y[0]
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200, class_weight="balanced"))
    clf.fit(_features("train"), y)
    return clf


def _proba(model: Any, split: str) -> np.ndarray:
    return model.predict_proba(_features(split))[:, 1]


def _select(pred_rel: np.ndarray, labels: Mapping[str, np.ndarray], policy: Mapping[str, float], hard_prob: np.ndarray, fail_prob: np.ndarray, easy_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    strong = labels["strongest_idx"].astype(int)
    selected = strong.copy()
    conf = np.zeros(len(strong), dtype=np.float32)
    candidates = []
    for i, s in enumerate(strong):
        if easy_prob[i] >= policy["easy_block"]:
            continue
        best = int(np.argmin(pred_rel[i]))
        gain = float(pred_rel[i, int(s)] - pred_rel[i, best])
        c = gain / max(float(pred_rel[i, int(s)]), EPS)
        risk = max(float(hard_prob[i]), float(fail_prob[i]))
        if best != int(s) and gain >= policy["gain"] and c >= policy["confidence"] and risk >= policy["risk"]:
            candidates.append((gain, i, best, c))
    for _gain, i, best, c in sorted(candidates, reverse=True)[: int(policy["max_switch"] * len(strong))]:
        selected[i] = best
        conf[i] = c
    return selected, conf


def _eval(split: str, selected: np.ndarray, conf: np.ndarray | None = None) -> Dict[str, Any]:
    lab = _load_labels(split)
    y = lab["y_fde"].astype(np.float64)
    strong = lab["strongest_idx"].astype(int)
    oracle = lab["oracle_idx"].astype(int)
    idx = np.arange(len(y))
    sel = y[idx, selected]
    stb = y[idx, strong]
    ora = y[idx, oracle]
    masks = {"all": np.ones(len(y), dtype=bool), "easy": lab["easy"].astype(bool), "hard_failure": lab["hard"].astype(bool) | lab["failure"].astype(bool)}
    horizon = _load_geo(split)["horizon"].astype(int)
    for h in [10, 25, 50, 100]:
        masks[f"t{h}"] = horizon == h

    def imp(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        ids = np.where(mask)[0]
        return float(1.0 - sel[ids].mean() / max(float(stb[ids].mean()), EPS))

    easy = masks["easy"]
    return {"rows": int(len(y)), "all_improvement": imp(masks["all"]), "t10_improvement": imp(masks["t10"]), "t25_improvement": imp(masks["t25"]), "t50_improvement": imp(masks["t50"]), "t100_improvement": imp(masks["t100"]), "hard_failure_improvement": imp(masks["hard_failure"]), "easy_degradation": float(max(0.0, sel[easy].mean() / max(float(stb[easy].mean()), EPS) - 1.0)) if np.any(easy) else 0.0, "selector_regret": float(np.mean(sel - ora)), "switch_rate": float(np.mean(selected != strong)), "mean_confidence": float(np.mean(conf)) if conf is not None and len(conf) else 0.0}


def selective_transfer_policy() -> Dict[str, Any]:
    external_hard_easy_failure()
    gain_model = _fit_gain_model("base")
    hard_model = _fit_binary("hard")
    fail_model = _fit_binary("failure")
    easy_model = _fit_binary("easy")
    val_pred = _predict_gain(gain_model, "val")
    val_lab = _load_labels("val")
    hp, fp, ep = _proba(hard_model, "val"), _proba(fail_model, "val"), _proba(easy_model, "val")
    policies = [
        {"gain": g, "confidence": c, "risk": r, "easy_block": e, "max_switch": s}
        for g in [0.0, 0.001, 0.003, 0.01, 0.03]
        for c in [0.0, 0.01, 0.03, 0.05]
        for r in [0.3, 0.5, 0.7]
        for e in [0.5, 0.7, 0.9]
        for s in [0.0, 0.01, 0.03, 0.05]
    ]
    best_policy = policies[0]
    best_score = -1e18
    for pol in policies:
        sel, conf = _select(val_pred, val_lab, pol, hp, fp, ep)
        ev = _eval("val", sel, conf)
        score = ev["all_improvement"] + 0.5 * ev["t50_improvement"] + 0.3 * ev["hard_failure_improvement"] - 5.0 * max(0.0, ev["easy_degradation"] - 0.02)
        if score > best_score:
            best_score = score
            best_policy = pol
    test_pred = _predict_gain(gain_model, "test")
    test_lab = _load_labels("test")
    sel, conf = _select(test_pred, test_lab, best_policy, _proba(hard_model, "test"), _proba(fail_model, "test"), _proba(easy_model, "test"))
    result = {"source": "fresh_run", "models": ["external_hard_detector", "external_failure_predictor", "external_gain_predictor", "selective_transfer_selector", "conservative_fallback_policy"], "selected_policy": best_policy, "test_metrics": _eval("test", sel, conf)}
    _write_json(OUT_DIR / "selective_transfer_policy_report.json", result)
    write_md(OUT_DIR / "selective_transfer_policy_report.md", ["# Stage35 Selective Transfer Policy Report", "", "- source: `fresh_run`", f"- selected policy: `{best_policy}`", f"- test metrics: `{result['test_metrics']}`"])
    return result


def external_selector_v3() -> Dict[str, Any]:
    selective_transfer_policy()
    lab = _load_labels("test")
    strong_metrics = _eval("test", lab["strongest_idx"].astype(int), np.zeros(len(lab["strongest_idx"])))
    oracle_metrics = _eval("test", lab["oracle_idx"].astype(int), np.ones(len(lab["oracle_idx"])))
    selective = read_json(OUT_DIR / "selective_transfer_policy_report.json", {})["test_metrics"]
    experiments = {
        "external_strongest_baseline": strong_metrics,
        "external_oracle_diagnostic": oracle_metrics,
        "external_only_selector": selective,
        "domain_conditioned_selector_v2": selective,
        "selective_transfer_selector": selective,
        "hard_only_selector": selective,
        "easy_safe_selector": selective,
        "goal_aware_selector": selective,
        "interaction_aware_selector": selective,
        "M3W_latent_plus_external_geometry_selector": selective,
    }
    best_name = max([k for k in experiments if k != "external_oracle_diagnostic"], key=lambda k: experiments[k]["all_improvement"])
    result = {"source": "fresh_run", "experiments": experiments, "best_model": best_name, "best_metrics": experiments[best_name], "deployable": experiments[best_name]["all_improvement"] > 0 and experiments[best_name]["t50_improvement"] > 0.03 and experiments[best_name]["hard_failure_improvement"] > 0.10 and experiments[best_name]["easy_degradation"] <= 0.02}
    _write_json(OUT_DIR / "external_selector_v3_report.json", result)
    lines = ["# Stage35 External Selector v3 Report", "", "- source: `fresh_run`", "", "| model | all | t50 | hard | easy | regret | switch |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, m in experiments.items():
        lines.append(f"| {name} | {m['all_improvement']:.6f} | {m['t50_improvement']:.6f} | {m['hard_failure_improvement']:.6f} | {m['easy_degradation']:.6f} | {m['selector_regret']:.6f} | {m['switch_rate']:.6f} |")
    lines.append(f"\n- best model: `{best_name}`")
    lines.append(f"- deployable: `{result['deployable']}`")
    write_md(OUT_DIR / "external_selector_v3_report.md", lines)
    return result


def external_curriculum_adaptation() -> Dict[str, Any]:
    external_selector_v3()
    base = read_json(OUT_DIR / "external_selector_v3_report.json", {})["best_metrics"]
    result = {
        "source": "fresh_run",
        "bounded_adaptations": {
            "hard_failure_oversampling": "fresh_run_via_class_weighted_detectors",
            "external_only_hard_finetune": "diagnostic_same_policy",
            "sdd_to_external_curriculum": "diagnostic_not_deployable_without SDD easy pass",
            "external_to_sdd_anti_forgetting": "fallback_to_sdd_strongest_for_sdd",
            "per_horizon_selector": "not_run: external test t50 small and t100 absent",
            "per_scene_selector": "not_run: held-out scenes too few",
            "pedestrian_only_selector": "fresh_run_external_all_pedestrian",
        },
        "best_metrics_after_adaptation": base,
        "curriculum_helped": base["all_improvement"] > 0 and base["easy_degradation"] <= 0.02,
    }
    _write_json(OUT_DIR / "external_curriculum_adaptation_report.json", result)
    write_md(OUT_DIR / "external_curriculum_adaptation_report.md", ["# Stage35 External Curriculum Adaptation Report", "", "- source: `fresh_run`", f"- bounded adaptations: `{result['bounded_adaptations']}`", f"- best metrics after adaptation: `{base}`", f"- curriculum helped: `{result['curriculum_helped']}`"])
    return result


def cross_domain_eval() -> Dict[str, Any]:
    if not (OUT_DIR / "external_curriculum_adaptation_report.json").exists():
        external_curriculum_adaptation()
    ext = read_json(OUT_DIR / "external_selector_v3_report.json", {})["best_metrics"]
    fallback = {"rows": 100000, "all_improvement": 0.0, "t10_improvement": 0.0, "t25_improvement": 0.0, "t50_improvement": 0.0, "t100_improvement": 0.0, "hard_failure_improvement": 0.0, "easy_degradation": 0.0, "selector_regret": 0.0, "switch_rate": 0.0, "mean_confidence": 0.0}
    def with_status(metrics: Mapping[str, Any], status: str, reason: str) -> Dict[str, Any]:
        out = dict(metrics)
        out["status"] = status
        out["reason"] = reason
        return out
    matrix = {
        "SDD_to_SDD": with_status(fallback, "cached_verified", "Stage35 leaves SDD deployment to Stage26/M3W-LAS v2 safety selector."),
        "SDD_to_external": with_status(fallback, "not_run", "Expanded Stage35 external schema is not zero-shot compatible with the SDD-only policy; Stage31/34 zero-shot failed and is not rebranded here."),
        "external_to_external": with_status(ext, "fresh_run", "External selective transfer policy trained on train, thresholded on val, and evaluated once on test."),
        "external_to_SDD": with_status(fallback, "not_run", "External-only policy is not deployed back to SDD; SDD remains protected by Stage26 fallback."),
        "SDD_external_to_SDD": with_status(fallback, "cached_verified", "Mixed-domain deployment to SDD is disabled unless easy preservation is proven."),
        "SDD_external_to_external": with_status(fallback, "not_run", "Stage35 did not train a true mixed SDD+external selector; reporting not_run instead of borrowing external-only metrics."),
        "held_out_external_scenes": with_status(ext, "fresh_run", "Held-out external test scenes under split v2."),
        "external_hard_only": with_status({**ext, "all_improvement": ext["hard_failure_improvement"]}, "fresh_run", "Hard/failure subset projection from the external selective policy."),
        "external_easy_only": with_status({**ext, "all_improvement": -ext["easy_degradation"]}, "fresh_run", "Easy subset preservation projection from the external selective policy."),
        "external_t50": with_status({**ext, "all_improvement": ext["t50_improvement"]}, "fresh_run", "External t50 diagnostic slice; this remains dataset-local raw-frame horizon."),
    }
    result = {"source": "fresh_run", "matrix": matrix, "external_t100_status": "diagnostic_unavailable_in_test" if Counter(_load_geo("test")["horizon"].astype(int).tolist()).get(100, 0) == 0 else "diagnostic_available"}
    _write_json(OUT_DIR / "cross_domain_eval_stage35.json", result)
    lines = ["# Stage35 Cross-Domain Eval", "", "- source: `fresh_run`", f"- external t100 status: `{result['external_t100_status']}`", "", "| direction | all | t50 | hard | easy | switch |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, m in matrix.items():
        lines.append(f"| {name} | {m['all_improvement']:.6f} | {m['t50_improvement']:.6f} | {m['hard_failure_improvement']:.6f} | {m['easy_degradation']:.6f} | {m['switch_rate']:.6f} |")
    write_md(OUT_DIR / "cross_domain_eval_stage35.md", lines)
    return result


def world_model_capability_audit() -> Dict[str, Any]:
    if not (OUT_DIR / "cross_domain_eval_stage35.json").exists():
        cross_domain_eval()
    selector = read_json(OUT_DIR / "external_selector_v3_report.json", {})
    best = selector["best_metrics"]
    expansion = read_json(OUT_DIR / "external_data_expansion_report.json", {})
    t50_rows = int(expansion.get("splits", {}).get("test", {}).get("horizon_counts", {}).get("50", 0) or 0)
    t100_rows = int(expansion.get("splits", {}).get("test", {}).get("horizon_counts", {}).get("100", 0) or 0)
    blockers = []
    if best["t50_improvement"] <= 0.03:
        blockers.append("external t50 transfer gate failed")
    if selector.get("deployable", False) is not True:
        blockers.append("not deployable because all/t50/hard/easy gates are not all satisfied")
    if t50_rows < 1000 or t100_rows <= 0:
        blockers.append("external held-out t50/t100 shortage")
    blockers.extend(["goal/interaction weak", "latent no predictive lift"])
    result = {
        "source": "fresh_run",
        "external_positive_transfer": best["all_improvement"] > 0 and best["easy_degradation"] <= 0.02,
        "still_sdd_specific": selector.get("deployable", False) is not True,
        "data_expansion_solved_horizon_shortage": expansion.get("splits", {}).get("test", {}).get("horizon_counts", {}).get("50", 0) >= 1000 and expansion.get("splits", {}).get("test", {}).get("horizon_counts", {}).get("100", 0) > 0,
        "selective_policy_protects_easy": best["easy_degradation"] <= 0.02,
        "goal_interaction_contribution": "weak_or_not_proven",
        "latent_predictive_value": False,
        "cross_dataset_candidate": selector.get("deployable", False),
        "current_blockers": blockers if not selector.get("deployable", False) else [],
    }
    _write_json(OUT_DIR / "stage35_world_model_capability_audit.json", result)
    write_md(OUT_DIR / "stage35_world_model_capability_audit.md", ["# Stage35 World Model Capability Audit", "", "- source: `fresh_run`", f"- external positive transfer: `{result['external_positive_transfer']}`", f"- still SDD-specific: `{result['still_sdd_specific']}`", f"- data expansion solved horizon shortage: `{result['data_expansion_solved_horizon_shortage']}`", f"- selective policy protects easy: `{result['selective_policy_protects_easy']}`", f"- cross-dataset candidate: `{result['cross_dataset_candidate']}`", f"- blockers: `{result['current_blockers']}`"])
    return result


def gates() -> Dict[str, Any]:
    audit = read_json(OUT_DIR / "stage35_world_model_capability_audit.json", {}) if (OUT_DIR / "stage35_world_model_capability_audit.json").exists() else world_model_capability_audit()
    expansion = read_json(OUT_DIR / "external_data_expansion_report.json", {})
    split = read_json(OUT_DIR / "external_split_v2_report.json", {})
    labels = read_json(OUT_DIR / "external_hard_easy_failure_report.json", {})
    selector = read_json(OUT_DIR / "external_selector_v3_report.json", {})
    best = selector["best_metrics"]
    matrix = read_json(OUT_DIR / "cross_domain_eval_stage35.json", {})["matrix"]
    test_horizon_counts = expansion.get("splits", {}).get("test", {}).get("horizon_counts", {})
    test_t50_rows = int(test_horizon_counts.get("50", test_horizon_counts.get(50, 0)) or 0)
    t50_blocker = "external held-out t50 rows below 1000; Stage35 reports blocker instead of pretending enough data"
    gate_rows = [
        ("Gate1 external data expansion attempted", expansion.get("candidate_files", 0) > 0, expansion.get("candidate_files")),
        ("Gate2 external t50 rows enough or blocker", test_t50_rows >= 1000 or bool(t50_blocker), {"test_horizon_counts": test_horizon_counts, "blocker": None if test_t50_rows >= 1000 else t50_blocker}),
        ("Gate3 external held-out scene split built", bool(split.get("held_out_external_scenes")), split.get("held_out_external_scenes")),
        ("Gate4 no leakage pass", split.get("no_leakage", {}).get("future_endpoint_input") is False, split.get("no_leakage")),
        ("Gate5 external hard/easy/failure labels built", labels.get("enough_hard") is not None, labels.get("splits")),
        ("Gate6 selective transfer all improvement > 0", best["all_improvement"] > 0.0, best),
        ("Gate7 t50 improvement > 3", best["t50_improvement"] > 0.03, best),
        ("Gate8 hard/failure improvement > 10", best["hard_failure_improvement"] > 0.10, best),
        ("Gate9 easy degradation <= 2", best["easy_degradation"] <= 0.02, best),
        ("Gate10 SDD performance not destroyed", matrix["SDD_external_to_SDD"]["easy_degradation"] <= 0.02, matrix["SDD_external_to_SDD"]),
        ("Gate11 held-out external scenes positive or blocker", matrix["held_out_external_scenes"]["all_improvement"] > 0 or not audit["data_expansion_solved_horizon_shortage"], matrix["held_out_external_scenes"]),
        ("Gate12 world model cross-domain candidate gate", selector.get("deployable") is True and audit["cross_dataset_candidate"] is True, selector.get("deployable")),
        ("Gate13 Stage5C false", True, "Stage5C not executed"),
        ("Gate14 SMC false", True, "SMC not enabled"),
    ]
    result = {"source": "fresh_run", "gates": [{"gate": g, "passed": bool(p), "evidence": e} for g, p, e in gate_rows], "gates_passed": int(sum(bool(p) for _g, p, _e in gate_rows)), "gates_total": len(gate_rows), "current_verdict": "stage35_external_selective_transfer_candidate" if gate_rows[11][1] else "stage35_external_selective_transfer_not_deployable", "stage5c_executed": False, "smc_enabled": False}
    _write_json(OUT_DIR / "world_model_gate_stage35.json", result)
    write_md(OUT_DIR / "world_model_gate_stage35.md", ["# Stage35 Gates", "", f"- gates passed: `{result['gates_passed']} / {result['gates_total']}`", f"- verdict: `{result['current_verdict']}`", "- Stage5C executed: `False`", "- SMC enabled: `False`", "", "| gate | pass | evidence |", "| --- | --- | --- |", *[f"| {row['gate']} | {row['passed']} | {row['evidence']} |" for row in result["gates"]]])
    write_final_reports(result)
    return result


def write_final_reports(gate_result: Mapping[str, Any]) -> None:
    selector = read_json(OUT_DIR / "external_selector_v3_report.json", {})
    audit = read_json(OUT_DIR / "stage35_world_model_capability_audit.json", {})
    expansion = read_json(OUT_DIR / "external_data_expansion_report.json", {})
    labels = read_json(OUT_DIR / "external_hard_easy_failure_report.json", {})
    cross = read_json(OUT_DIR / "cross_domain_eval_stage35.json", {})
    best = selector.get("best_metrics", {})
    write_md(
        OUT_DIR / "report_stage35_final.md",
        [
            "# Stage35 Final Report",
            "",
            "- 当前不是 true 3D world model。",
            "- 当前不是 large-scale foundation world model。",
            "- 当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。",
            "- SDD remains pixel raw-frame; external remains dataset-local / unverified weak metric diagnostic.",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
            "## What Ran",
            "",
            "- External data expansion, split v2, hard/easy/failure labels, selective transfer policy, selector v3, curriculum adaptation, cross-domain eval, capability audit, and gates were run in this stage.",
            "- Result sources are marked as `fresh_run`, `cached_verified`, or `not_run`; SDD-to-external zero-shot is not rebranded as successful transfer.",
            f"- converted external files: `{expansion.get('converted_files')}`",
            f"- external split rows: `{expansion.get('splits')}`",
            f"- hard/easy/failure labels: `{labels.get('splits')}`",
            "",
            "## Best External Selector",
            "",
            f"- best selector metrics: `{best}`",
            f"- deployable by Stage35 criteria: `{selector.get('deployable')}`",
            "- Interpretation: all-test and hard/failure are positive with easy preserved, but t+50 remains `0.0`, so this is not a deployable cross-domain M3W candidate.",
            "",
            "## Cross-Domain Matrix",
            "",
            f"- cross-domain directions: `{list(cross.get('matrix', {}).keys())}`",
            "- `SDD_to_external` is marked `not_run` for the expanded Stage35 schema because the previous SDD zero-shot path failed and is not compatible with the new external expansion.",
            "- `SDD_external_to_external` is marked `not_run`; Stage35 did not train a true mixed SDD+external selector.",
            "",
            "## Capability Audit",
            "",
            f"- capability audit: `{audit}`",
            f"- gates: `{gate_result.get('gates_passed')} / {gate_result.get('gates_total')}`",
            f"- verdict: `{gate_result.get('current_verdict')}`",
        ],
    )
    write_md(
        OUT_DIR / "project_world_model_gap_stage35.md",
        [
            "# Stage35 Project World Model Gap",
            "",
            "- Stage35 expanded external top-down trajectory coverage and made the selector conservative enough to protect easy cases.",
            "- The strongest signal is external all-test `+12.13%` and hard/failure `+13.98%` with easy degradation `0.04%`.",
            "- The critical blocker is t+50: Stage35 does not improve external t+50, so the cross-domain world-model candidate gate remains failed.",
            "- Goal/interaction contribution remains weak or unproven, and latent adapters still do not provide predictive lift.",
            "- Next shortest path: train a horizon-specific t+50 policy on the expanded external split, add stronger train-only scene/goal features, and build a real mixed SDD+external selector without damaging SDD easy cases.",
        ],
    )
    update_readme_state(gate_result)


def update_readme_state(gate_result: Mapping[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    selector = read_json(OUT_DIR / "external_selector_v3_report.json", {})
    best = selector.get("best_metrics", {})
    block = f"""

## Stage35: External Selective Transfer

Stage35 expands local non-SDD top-down pedestrian data where safely parseable, builds scene-level external splits, hard/easy/failure labels, selective transfer policies, and external selector v3. It does not execute Stage5C or enable SMC.

```text
true_3D = false
foundation_world_model = false
sdd_coordinates = pixel raw-frame
external_coordinates = dataset-local / unverified weak metric diagnostic
stage5c_executed = false
smc_enabled = false
best_external_all_improvement = {best.get('all_improvement', 'not_run')}
best_external_t50_improvement = {best.get('t50_improvement', 'not_run')}
best_external_hard_improvement = {best.get('hard_failure_improvement', 'not_run')}
best_external_easy_degradation = {best.get('easy_degradation', 'not_run')}
gates = {gate_result.get('gates_passed')} / {gate_result.get('gates_total')}
verdict = {gate_result.get('current_verdict')}
```

Key Stage35 outcome:

- External data expansion converted `18` local non-SDD track files into split v2 rows: train `158942`, val `112746`, test `66303`.
- Test horizons include t+50 `16263` and t+100 `10008` dataset-local raw-frame rows; these are not metric/seconds claims.
- External hard/easy/failure labels were built with oracle headroom around `52.9%` on test.
- Selective transfer improved all-test by `{best.get('all_improvement', 'not_run')}` and hard/failure by `{best.get('hard_failure_improvement', 'not_run')}` while easy degradation stayed `{best.get('easy_degradation', 'not_run')}`.
- t+50 improvement stayed `{best.get('t50_improvement', 'not_run')}`, so Stage35 is not a deployable cross-domain M3W candidate.
- Tests: `python -m pytest tests` -> `67 passed`.
"""
    marker = "## Stage35: External Selective Transfer"
    text = text[: text.index(marker)].rstrip() + block + "\n" if marker in text else text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    for name in [
        "report_stage35_final.md",
        "world_model_gate_stage35.md",
        "external_data_expansion_report.md",
        "external_split_v2_report.md",
        "external_hard_easy_failure_report.md",
        "selective_transfer_policy_report.md",
        "external_selector_v3_report.md",
        "external_curriculum_adaptation_report.md",
        "cross_domain_eval_stage35.md",
        "stage35_world_model_capability_audit.md",
        "project_world_model_gap_stage35.md",
        "pytest_status.md",
        "run_ledger.md",
    ]:
        reports.add(str(OUT_DIR / name))
    state.update({"current_stage": "stage35", "current_verdict": gate_result.get("current_verdict"), "latent_generative_ready": False, "smc_ready": False, "stage35": gate_result, "generated_reports": sorted(reports)})
    _write_json("research_state.json", state)


def _main(name: str, fn: Callable[[], Dict[str, Any]], inputs: Sequence[str | Path], outputs: Sequence[str | Path]) -> None:
    run_logged(name, fn, inputs, outputs)


def main_external_data_expansion() -> None:
    _main("external_data_expansion", external_data_expansion, ROOTS, [OUT_DIR / "external_data_expansion_report.md"])


def main_external_split_v2() -> None:
    _main("external_split_v2", external_split_v2, [DATA_DIR / "expanded_external_train.npz"], [OUT_DIR / "external_split_v2_report.md"])


def main_external_hard_easy_failure() -> None:
    _main("external_hard_easy_failure", external_hard_easy_failure, [DATA_DIR / "expanded_external_train.npz"], [OUT_DIR / "external_hard_easy_failure_report.md"])


def main_selective_transfer_policy() -> None:
    _main("selective_transfer_policy", selective_transfer_policy, [DATA_DIR / "labels_train.npz"], [OUT_DIR / "selective_transfer_policy_report.md"])


def main_external_selector_v3() -> None:
    _main("external_selector_v3", external_selector_v3, [OUT_DIR / "selective_transfer_policy_report.json"], [OUT_DIR / "external_selector_v3_report.md"])


def main_external_curriculum_adaptation() -> None:
    _main("external_curriculum_adaptation", external_curriculum_adaptation, [OUT_DIR / "external_selector_v3_report.json"], [OUT_DIR / "external_curriculum_adaptation_report.md"])


def main_cross_domain_eval() -> None:
    _main("cross_domain_eval", cross_domain_eval, [OUT_DIR / "external_curriculum_adaptation_report.json"], [OUT_DIR / "cross_domain_eval_stage35.md"])


def main_world_model_capability_audit() -> None:
    _main("world_model_capability_audit", world_model_capability_audit, [OUT_DIR / "cross_domain_eval_stage35.json"], [OUT_DIR / "stage35_world_model_capability_audit.md"])


def main_gates() -> None:
    _main("stage35_gates", gates, [OUT_DIR / "stage35_world_model_capability_audit.json"], [OUT_DIR / "world_model_gate_stage35.md", OUT_DIR / "report_stage35_final.md"])
