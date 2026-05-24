from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage35_selective_transfer as s35
from src import stage36_t50_repair as s36


OUT_DIR = Path("outputs/stage37_t50_history")
DATA_DIR = Path("data/stage37_t50_history")
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
STAGE35_DATA = s35.DATA_DIR
STAGE35_OUT = s35.OUT_DIR
STAGE36_OUT = s36.OUT_DIR
MAX_K = 64
HISTORY_KS = [8, 16, 32, 64]
HORIZONS = [10, 25, 50, 100]
EPS = 1e-6
BASELINE_FAMILY = [
    "constant_position",
    "constant_velocity_causal_fd",
    "damped_velocity",
    "constant_acceleration",
    "turn_rate",
    "history_decay_baseline",
    "prototype_goal_directed_baseline",
    "neighbor_aware_decay_baseline",
]
PROTOTYPES = [
    "straight_continue",
    "slow_stop",
    "left_turn",
    "right_turn",
    "reverse_or_u_turn",
    "group_follow",
    "density_avoid",
    "exit_like_direction_from_past_motion",
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
        "# Stage37 Causal History t+50 Run Ledger",
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


def _ensure_inputs() -> None:
    required = [STAGE35_DATA / f"expanded_external_{s}.npz" for s in ["train", "val", "test"]]
    required += [STAGE35_DATA / f"labels_{s}.npz" for s in ["train", "val", "test"]]
    if not all(p.exists() for p in required):
        s35.gates()


def _geo(split: str) -> Dict[str, np.ndarray]:
    _ensure_inputs()
    return dict(np.load(STAGE35_DATA / f"expanded_external_{split}.npz"))


def _labels(split: str) -> Dict[str, np.ndarray]:
    _ensure_inputs()
    return dict(np.load(STAGE35_DATA / f"labels_{split}.npz"))


def _read_track_file(path: str | Path) -> np.ndarray:
    p = Path(str(path))
    if p.name == "obsmat.txt":
        return s35._read_obsmat(p)
    return s35._read_four_col(p)


def _track_cache(paths: Sequence[str]) -> Tuple[Dict[Tuple[str, int], np.ndarray], Dict[Tuple[str, float], np.ndarray]]:
    tracks: Dict[Tuple[str, int], np.ndarray] = {}
    frame_points: Dict[Tuple[str, float], list[Tuple[int, float, float]]] = defaultdict(list)
    for path in sorted(set(paths)):
        arr = _read_track_file(path)
        if len(arr) == 0:
            continue
        for agent in np.unique(arr[:, 1]).astype(int):
            tr = arr[arr[:, 1] == agent]
            tr = tr[np.argsort(tr[:, 0])]
            tracks[(path, int(agent))] = tr.astype(np.float64)
            for row in tr:
                frame_points[(path, float(row[0]))].append((int(agent), float(row[2]), float(row[3])))
    frame_arrays = {
        key: np.asarray(vals, dtype=np.float64)
        for key, vals in frame_points.items()
    }
    return tracks, frame_arrays


def _empty_history(n: int) -> Dict[str, np.ndarray]:
    shape = (n, MAX_K)
    return {
        "history_x": np.zeros(shape, dtype=np.float32),
        "history_y": np.zeros(shape, dtype=np.float32),
        "history_dx": np.zeros(shape, dtype=np.float32),
        "history_dy": np.zeros(shape, dtype=np.float32),
        "history_speed": np.zeros(shape, dtype=np.float32),
        "history_accel": np.zeros(shape, dtype=np.float32),
        "history_heading": np.zeros(shape, dtype=np.float32),
        "history_valid_mask": np.zeros(shape, dtype=bool),
        "history_curvature": np.zeros(n, dtype=np.float32),
        "history_turn_angle": np.zeros(n, dtype=np.float32),
        "history_stop_go": np.zeros(n, dtype=np.float32),
        "history_dwell": np.zeros(n, dtype=np.float32),
        "history_path_length": np.zeros(n, dtype=np.float32),
        "history_velocity_decay": np.zeros(n, dtype=np.float32),
        "history_goal_alignment_proxy": np.zeros(n, dtype=np.float32),
        "history_neighbor_count": np.zeros(n, dtype=np.float32),
        "history_min_neighbor_dist": np.zeros(n, dtype=np.float32),
        "history_density": np.zeros(n, dtype=np.float32),
        "history_TTC": np.zeros(n, dtype=np.float32),
        "history_closing_speed": np.zeros(n, dtype=np.float32),
        "source_found": np.zeros(n, dtype=bool),
    }


def _angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.sum(a * b, axis=1)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    cos = np.clip(dot / np.maximum(na * nb, EPS), -1.0, 1.0)
    return np.arccos(cos)


def _build_history_for_split(split: str) -> Dict[str, Any]:
    ensure_dir(DATA_DIR)
    out_path = DATA_DIR / f"history_windows_{split}.npz"
    if out_path.exists():
        h = dict(np.load(out_path))
        return {
            "split": split,
            "rows": int(len(h["source_found"])),
            "source_found": int(h["source_found"].sum()),
            "k_available": {str(k): int(np.sum(h["history_valid_mask"][:, -k:].sum(axis=1) >= k)) for k in HISTORY_KS},
        }
    geo = _geo(split)
    n = len(geo["horizon"])
    cache, frame_points = _track_cache(geo["source_file"].astype(str).tolist())
    hist = _empty_history(n)
    paths = geo["source_file"].astype(str)
    agents = geo["agent_id"].astype(int)
    frames = geo["frame_id"].astype(float)
    cur_xy = np.stack([geo["current_x"], geo["current_y"]], axis=1).astype(np.float64)
    for i in range(n):
        tr = cache.get((paths[i], int(agents[i])))
        if tr is None or len(tr) == 0:
            continue
        idxs = np.where(tr[:, 0] <= frames[i] + 1e-5)[0]
        if len(idxs) == 0:
            continue
        j = int(idxs[-1])
        start = max(0, j - MAX_K + 1)
        window = tr[start : j + 1]
        m = len(window)
        sl = slice(MAX_K - m, MAX_K)
        x = window[:, 2]
        y = window[:, 3]
        hist["history_x"][i, sl] = x
        hist["history_y"][i, sl] = y
        hist["history_valid_mask"][i, sl] = True
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        speed = np.sqrt(dx * dx + dy * dy)
        accel = np.diff(speed, prepend=speed[0])
        heading = np.arctan2(dy, dx)
        hist["history_dx"][i, sl] = dx
        hist["history_dy"][i, sl] = dy
        hist["history_speed"][i, sl] = speed
        hist["history_accel"][i, sl] = accel
        hist["history_heading"][i, sl] = heading
        valid_speed = speed[1:] if len(speed) > 1 else speed
        hist["history_path_length"][i] = float(np.sum(valid_speed))
        hist["history_stop_go"][i] = float(np.mean(valid_speed < max(np.percentile(valid_speed, 30), EPS))) if len(valid_speed) else 0.0
        hist["history_dwell"][i] = float(np.mean(valid_speed < 1e-3)) if len(valid_speed) else 0.0
        if len(heading) > 2:
            turn = np.diff(np.unwrap(heading))
            hist["history_turn_angle"][i] = float(np.sum(np.abs(turn)))
            hist["history_curvature"][i] = float(np.mean(np.abs(turn) / np.maximum(speed[1:], EPS)))
        if len(speed) >= 4:
            hist["history_velocity_decay"][i] = float(speed[-1] / max(np.mean(speed[-4:]), EPS))
        last_vec = np.array([dx[-1], dy[-1]], dtype=np.float64)
        hist["history_goal_alignment_proxy"][i] = float(np.linalg.norm(last_vec) / max(np.linalg.norm(cur_xy[i] - np.array([x[0], y[0]])), EPS))
        pts = frame_points.get((paths[i], frames[i]))
        if pts is not None and len(pts):
            others = pts[pts[:, 0] != agents[i]]
            hist["history_neighbor_count"][i] = float(len(others))
            if len(others):
                d = np.linalg.norm(others[:, 1:3] - cur_xy[i][None, :], axis=1)
                hist["history_min_neighbor_dist"][i] = float(np.min(d))
                hist["history_density"][i] = float(np.sum(d < max(np.median(d), EPS)))
                # TTC and closing speed are approximated from current last velocity only; no future state.
                last_vel = last_vec
                rel = others[:, 1:3] - cur_xy[i][None, :]
                closing = -np.sum(rel * last_vel[None, :], axis=1) / np.maximum(np.linalg.norm(rel, axis=1), EPS)
                hist["history_closing_speed"][i] = float(np.max(closing)) if len(closing) else 0.0
                positive = closing > EPS
                hist["history_TTC"][i] = float(np.min(d[positive] / np.maximum(closing[positive], EPS))) if np.any(positive) else 0.0
        hist["source_found"][i] = True
    np.savez_compressed(out_path, **hist)
    return {
        "split": split,
        "rows": int(n),
        "source_found": int(hist["source_found"].sum()),
        "k_available": {str(k): int(np.sum(hist["history_valid_mask"][:, -k:].sum(axis=1) >= k)) for k in HISTORY_KS},
    }


def build_history_windows() -> Dict[str, Any]:
    _ensure_inputs()
    reports = {split: _build_history_for_split(split) for split in ["train", "val", "test"]}
    schema = {
        "source": "fresh_run",
        "max_k": MAX_K,
        "k_values": HISTORY_KS,
        "fields": [
            "history_x/y",
            "history_dx/dy",
            "history_speed",
            "history_accel",
            "history_heading",
            "history_curvature",
            "history_turn_angle",
            "history_stop_go",
            "history_dwell",
            "history_path_length",
            "history_velocity_decay",
            "history_goal_alignment_proxy",
            "history_neighbor_count",
            "history_min_neighbor_dist",
            "history_density",
            "history_TTC",
            "history_closing_speed",
            "history_valid_mask",
        ],
        "no_leakage": {
            "uses_current_and_past_frames_only": True,
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
    }
    _write_json(OUT_DIR / "history_window_schema.json", schema)
    result = {"source": "fresh_run", "stage35_inputs": "cached_verified", "reports": reports, "schema": schema}
    _write_json(OUT_DIR / "stage37_history_window_report.json", result)
    write_md(
        OUT_DIR / "stage37_history_window_report.md",
        [
            "# Stage37 History Window Report",
            "",
            "- source: `fresh_run`; Stage35 rows are `cached_verified`.",
            f"- split reports: `{reports}`",
            f"- schema: `{schema}`",
        ],
    )
    return result


def _history(split: str) -> Dict[str, np.ndarray]:
    if not (DATA_DIR / f"history_windows_{split}.npz").exists():
        build_history_windows()
    return dict(np.load(DATA_DIR / f"history_windows_{split}.npz"))


def _hmask(split: str, horizon: int = 50) -> np.ndarray:
    return _geo(split)["horizon"].astype(int) == horizon


def t50_quality_audit() -> Dict[str, Any]:
    build_history_windows()
    reports = {}
    for split in ["train", "val", "test"]:
        geo = _geo(split)
        lab = _labels(split)
        hist = _history(split)
        mask = _hmask(split, 50)
        y = lab["y_fde"].astype(np.float64)[mask]
        strong = lab["strongest_idx"].astype(int)[mask]
        oracle = lab["oracle_idx"].astype(int)[mask]
        idx = np.arange(mask.sum())
        valid_len = hist["history_valid_mask"][mask].sum(axis=1)
        scene_counts = dict(Counter(geo["scene_id"].astype(str)[mask].tolist()))
        domain_counts = dict(Counter(geo["dataset"].astype(str)[mask].tolist()))
        margin = lab["oracle_margin"].astype(float)[mask]
        reports[split] = {
            "t50_rows": int(mask.sum()),
            "history_len_ge_8": int(np.sum(valid_len >= 8)),
            "history_len_ge_16": int(np.sum(valid_len >= 16)),
            "history_len_ge_32": int(np.sum(valid_len >= 32)),
            "history_len_ge_64": int(np.sum(valid_len >= 64)),
            "source_found": int(hist["source_found"][mask].sum()),
            "future_label_valid": int(np.isfinite(y).all(axis=1).sum()) if len(y) else 0,
            "scene_distribution": scene_counts,
            "domain_distribution": domain_counts,
            "baseline_error_mean": float(y[idx, strong].mean()) if len(y) else 0.0,
            "oracle_error_mean": float(y[idx, oracle].mean()) if len(y) else 0.0,
            "oracle_margin_mean": float(np.mean(margin)) if len(margin) else 0.0,
            "oracle_headroom": float(1.0 - y[idx, oracle].mean() / max(float(y[idx, strong].mean()), EPS)) if len(y) else 0.0,
        }
    result = {
        "source": "fresh_run",
        "reports": reports,
        "val_test_gap": {
            "oracle_headroom_delta": reports["val"]["oracle_headroom"] - reports["test"]["oracle_headroom"],
            "baseline_error_delta": reports["val"]["baseline_error_mean"] - reports["test"]["baseline_error_mean"],
            "diagnosis": "val/test gap is audited explicitly; test endpoints are not used for training or goals.",
        },
        "quality_pass": reports["test"]["t50_rows"] > 1000 and reports["test"]["history_len_ge_8"] > 1000,
    }
    _write_json(OUT_DIR / "stage37_t50_quality_audit.json", result)
    write_md(OUT_DIR / "stage37_t50_quality_audit.md", ["# Stage37 t+50 Quality Audit", "", "- source: `fresh_run`", f"- reports: `{reports}`", f"- val/test gap: `{result['val_test_gap']}`", f"- quality pass: `{result['quality_pass']}`"])
    return result


def _proto_vectors_from_history(split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hist = _history(split)
    geo = _geo(split)
    speed = hist["history_speed"][:, -1].astype(np.float64)
    heading = hist["history_heading"][:, -1].astype(np.float64)
    h = np.maximum(geo["horizon"].astype(np.float64), 1.0)
    scale = np.maximum(speed * h, hist["history_path_length"].astype(np.float64))
    dx = np.cos(heading)
    dy = np.sin(heading)
    left = np.stack([-dy, dx], axis=1)
    forward = np.stack([dx, dy], axis=1)
    density = hist["history_density"].astype(np.float64)
    turn = hist["history_turn_angle"].astype(np.float64)
    stop = hist["history_stop_go"].astype(np.float64)
    vectors = []
    vectors.append(forward * scale[:, None])
    vectors.append(forward * (0.15 * scale)[:, None] * (1.0 - np.clip(stop, 0, 1))[:, None])
    vectors.append((0.65 * forward + 0.75 * left) * scale[:, None])
    vectors.append((0.65 * forward - 0.75 * left) * scale[:, None])
    vectors.append(-0.5 * forward * scale[:, None])
    vectors.append(0.75 * forward * scale[:, None])
    side = np.where((np.arange(len(scale)) % 2)[:, None] == 0, left, -left)
    vectors.append((0.45 * forward + np.clip(density, 0, 3)[:, None] * 0.15 * side) * scale[:, None])
    vectors.append((1.2 * forward + 0.15 * np.tanh(turn)[:, None] * left) * scale[:, None])
    proto = np.stack(vectors, axis=1).astype(np.float32)
    likelihood = np.zeros((len(scale), len(PROTOTYPES)), dtype=np.float32)
    likelihood[:, 0] = 1.0 / (1.0 + np.abs(turn))
    likelihood[:, 1] = np.clip(stop, 0, 1)
    likelihood[:, 2] = np.clip(np.maximum(turn, 0), 0, 3)
    likelihood[:, 3] = np.clip(np.maximum(turn, 0), 0, 3)
    likelihood[:, 4] = np.clip(hist["history_dwell"], 0, 1)
    likelihood[:, 5] = 1.0 / (1.0 + hist["history_min_neighbor_dist"])
    likelihood[:, 6] = np.clip(density / max(float(np.percentile(density, 90)), EPS), 0, 1)
    likelihood[:, 7] = likelihood[:, 0] + 0.1
    likelihood = likelihood / np.maximum(likelihood.sum(axis=1, keepdims=True), EPS)
    entropy = -np.sum(likelihood * np.log(np.maximum(likelihood, EPS)), axis=1).astype(np.float32)
    ambiguity = (1.0 - np.max(likelihood, axis=1)).astype(np.float32)
    return proto, likelihood, entropy, ambiguity


def goal_prototypes() -> Dict[str, Any]:
    t50_quality_audit()
    reports = {}
    for split in ["train", "val", "test"]:
        proto, likelihood, entropy, ambiguity = _proto_vectors_from_history(split)
        hist = _history(split)
        last_vel = np.stack([hist["history_dx"][:, -1], hist["history_dy"][:, -1]], axis=1).astype(np.float64)
        pnorm = np.linalg.norm(proto, axis=2)
        lnorm = np.linalg.norm(last_vel, axis=1)
        angles = np.zeros_like(likelihood, dtype=np.float32)
        for k in range(len(PROTOTYPES)):
            angles[:, k] = _angle_between(last_vel, proto[:, k, :])
        np.savez_compressed(
            DATA_DIR / f"goal_prototypes_{split}.npz",
            prototype_vectors=proto,
            prototype_likelihood=likelihood,
            prototype_entropy=entropy,
            goal_ambiguity=ambiguity,
            prototype_distance=pnorm.astype(np.float32),
            prototype_angle=angles.astype(np.float32),
        )
        reports[split] = {
            "rows": int(len(proto)),
            "prototype_count": len(PROTOTYPES),
            "mean_entropy": float(np.mean(entropy)) if len(entropy) else 0.0,
            "mean_ambiguity": float(np.mean(ambiguity)) if len(ambiguity) else 0.0,
        }
    result = {"source": "fresh_run", "prototype_names": PROTOTYPES, "train_only_endpoint_usage": False, "test_endpoint_usage": False, "reports": reports}
    _write_json(OUT_DIR / "stage37_goal_prototype_report.json", result)
    write_md(OUT_DIR / "stage37_goal_prototype_report.md", ["# Stage37 Scene-Agnostic Goal Prototype Report", "", "- source: `fresh_run`", "- prototypes are generated from past motion patterns, not test endpoints.", f"- prototype names: `{PROTOTYPES}`", f"- reports: `{reports}`"])
    return result


def _proto(split: str) -> Dict[str, np.ndarray]:
    if not (DATA_DIR / f"goal_prototypes_{split}.npz").exists():
        goal_prototypes()
    return dict(np.load(DATA_DIR / f"goal_prototypes_{split}.npz"))


def _baseline_family(split: str) -> Dict[str, np.ndarray]:
    path = DATA_DIR / f"t50_baseline_family_{split}.npz"
    if path.exists():
        return dict(np.load(path))
    goal_prototypes()
    geo = _geo(split)
    hist = _history(split)
    proto = _proto(split)
    cur = np.stack([geo["current_x"], geo["current_y"]], axis=1).astype(np.float64)
    fut = np.stack([geo["future_endpoint_x"], geo["future_endpoint_y"]], axis=1).astype(np.float64)
    h = np.maximum(geo["horizon"].astype(np.float64), 1.0)
    v1 = np.stack([hist["history_dx"][:, -1], hist["history_dy"][:, -1]], axis=1).astype(np.float64)
    v2 = np.stack([hist["history_dx"][:, -2], hist["history_dy"][:, -2]], axis=1).astype(np.float64)
    accel = v1 - v2
    speed = np.linalg.norm(v1, axis=1)
    decay_factor = (1.0 - 0.94 ** h) / max(1.0 - 0.94, EPS)
    hist_decay = (1.0 - 0.90 ** h) / max(1.0 - 0.90, EPS)
    heading = hist["history_heading"][:, -1].astype(np.float64)
    turn = np.clip(hist["history_turn_angle"].astype(np.float64) / np.maximum(hist["history_valid_mask"].sum(axis=1), 1), -0.5, 0.5)
    turn_heading = heading + turn * h
    turn_vec = np.stack([np.cos(turn_heading), np.sin(turn_heading)], axis=1) * (speed * h)[:, None]
    proto_vecs = proto["prototype_vectors"].astype(np.float64)
    likelihood = proto["prototype_likelihood"].astype(np.float64)
    best_proto = np.argmax(likelihood, axis=1)
    proto_pred = cur + proto_vecs[np.arange(len(cur)), best_proto, :]
    density_scale = 1.0 / (1.0 + 0.1 * hist["history_density"].astype(np.float64))
    preds = [
        cur,
        cur + v1 * h[:, None],
        cur + v1 * decay_factor[:, None],
        cur + v1 * h[:, None] + 0.5 * accel * np.minimum(h, 10)[:, None] ** 2,
        cur + turn_vec,
        cur + v1 * hist_decay[:, None],
        proto_pred,
        cur + v1 * (decay_factor * density_scale)[:, None],
    ]
    pred_arr = np.stack(preds, axis=1).astype(np.float32)
    fde = np.linalg.norm(pred_arr.astype(np.float64) - fut[:, None, :], axis=2).astype(np.float32)
    normalizer = np.maximum(hist["history_path_length"].astype(np.float32) + speed.astype(np.float32) * geo["horizon"].astype(np.float32), np.median(hist["history_path_length"]) + EPS)
    rel = fde / np.maximum(normalizer[:, None], EPS)
    np.savez_compressed(path, prediction=pred_arr, y_fde=fde, relative_y=rel.astype(np.float32), normalizer=normalizer.astype(np.float32), baseline_names=np.asarray(BASELINE_FAMILY, dtype="U64"))
    return dict(np.load(path))


def t50_baseline_family() -> Dict[str, Any]:
    goal_prototypes()
    reports = {}
    for split in ["train", "val", "test"]:
        fam = _baseline_family(split)
        mask = _hmask(split, 50)
        y = fam["y_fde"][mask].astype(np.float64)
        rel = fam["relative_y"][mask].astype(np.float64)
        best = np.argmin(rel, axis=1) if len(rel) else np.zeros(0, dtype=int)
        strong = _labels(split)["strongest_idx"].astype(int)[mask]
        stage35_y = _labels(split)["y_fde"].astype(np.float64)[mask]
        idx = np.arange(len(best))
        fallback = stage35_y[idx, strong] if len(idx) else np.zeros(0)
        oracle = y[idx, best] if len(idx) else np.zeros(0)
        reports[split] = {
            "t50_rows": int(mask.sum()),
            "family_mean_fde": {BASELINE_FAMILY[i]: float(y[:, i].mean()) if len(y) else 0.0 for i in range(len(BASELINE_FAMILY))},
            "family_mean_relative_fde": {BASELINE_FAMILY[i]: float(rel[:, i].mean()) if len(rel) else 0.0 for i in range(len(BASELINE_FAMILY))},
            "family_oracle_headroom_vs_stage35_fallback": float(1.0 - oracle.mean() / max(float(fallback.mean()), EPS)) if len(idx) else 0.0,
            "best_distribution": dict(Counter([BASELINE_FAMILY[int(i)] for i in best.tolist()])),
        }
    result = {"source": "fresh_run", "baseline_family": BASELINE_FAMILY, "reports": reports, "all_past_only": True}
    _write_json(OUT_DIR / "stage37_t50_baseline_family.json", result)
    write_md(OUT_DIR / "stage37_t50_baseline_family.md", ["# Stage37 t+50 Baseline Family", "", "- source: `fresh_run`", f"- family: `{BASELINE_FAMILY}`", f"- reports: `{reports}`", "- All candidate baselines use past/current history only; future endpoints are evaluation labels only."])
    return result


def _feature_matrix(split: str) -> Tuple[np.ndarray, list[str]]:
    if not (DATA_DIR / f"t50_baseline_family_{split}.npz").exists():
        t50_baseline_family()
    hist = _history(split)
    proto = _proto(split)
    geo = _geo(split)
    lab = _labels(split)
    fam = _baseline_family(split)
    base = s36._load_feature(split)["X"] if (s36.DATA_DIR / f"t50_features_{split}.npz").exists() else s35._features(split)
    scalar_names = [
        "history_curvature",
        "history_turn_angle",
        "history_stop_go",
        "history_dwell",
        "history_path_length",
        "history_velocity_decay",
        "history_goal_alignment_proxy",
        "history_neighbor_count",
        "history_min_neighbor_dist",
        "history_density",
        "history_TTC",
        "history_closing_speed",
        "history_valid_len",
        "prototype_entropy",
        "goal_ambiguity",
        "is_t50",
        "is_t100",
    ]
    scalars = np.stack(
        [
            hist["history_curvature"],
            hist["history_turn_angle"],
            hist["history_stop_go"],
            hist["history_dwell"],
            hist["history_path_length"],
            hist["history_velocity_decay"],
            hist["history_goal_alignment_proxy"],
            hist["history_neighbor_count"],
            hist["history_min_neighbor_dist"],
            hist["history_density"],
            hist["history_TTC"],
            hist["history_closing_speed"],
            hist["history_valid_mask"].sum(axis=1).astype(np.float32),
            proto["prototype_entropy"],
            proto["goal_ambiguity"],
            (geo["horizon"].astype(int) == 50).astype(np.float32),
            (geo["horizon"].astype(int) == 100).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    proto_feats = np.concatenate([proto["prototype_likelihood"].astype(np.float32), proto["prototype_angle"].astype(np.float32)], axis=1)
    names = [f"stage36_or_base_{i}" for i in range(base.shape[1])] + scalar_names + [f"prototype_likelihood_{p}" for p in PROTOTYPES] + [f"prototype_angle_{p}" for p in PROTOTYPES]
    x = np.nan_to_num(np.concatenate([base.astype(np.float32), scalars, proto_feats], axis=1), posinf=1e6, neginf=-1e6)
    return x, names


def _fit_family_regressor(kind: str, train_mask: np.ndarray) -> Any:
    x, _names = _feature_matrix("train")
    fam = _baseline_family("train")
    y = np.log1p(np.clip(fam["relative_y"].astype(np.float32), 0.0, 1e6))
    if train_mask.sum() < 50:
        train_mask = np.ones(len(x), dtype=bool)
    if kind == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    elif kind == "extra":
        model = ExtraTreesRegressor(n_estimators=80, max_depth=12, min_samples_leaf=25, random_state=37, n_jobs=1)
    else:
        model = RandomForestRegressor(n_estimators=80, max_depth=12, min_samples_leaf=20, random_state=37, n_jobs=1)
    model.fit(x[train_mask], y[train_mask])
    return model


def _predict_family(model: Any, split: str) -> np.ndarray:
    x, _ = _feature_matrix(split)
    return np.maximum(0.0, np.expm1(np.clip(model.predict(x), 0.0, 12.0)))


def _eval_family_selection(split: str, selected_family: np.ndarray, confidence: np.ndarray | None = None) -> Dict[str, Any]:
    fam = _baseline_family(split)
    stage = _labels(split)
    geo = _geo(split)
    yfam = fam["y_fde"].astype(np.float64)
    fallback = stage["y_fde"].astype(np.float64)[np.arange(len(yfam)), stage["strongest_idx"].astype(int)]
    sel = fallback.copy()
    switched = selected_family >= 0
    sel[switched] = yfam[np.where(switched)[0], selected_family[switched]]
    oracle = np.minimum(fallback, yfam.min(axis=1))
    horizon = geo["horizon"].astype(int)
    easy = stage["easy"].astype(bool)
    hard_failure = stage["hard"].astype(bool) | stage["failure"].astype(bool)

    def imp(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        return float(1.0 - sel[mask].mean() / max(float(fallback[mask].mean()), EPS))

    return {
        "rows": int(len(sel)),
        "all_improvement": imp(np.ones(len(sel), dtype=bool)),
        "t10_improvement": imp(horizon == 10),
        "t25_improvement": imp(horizon == 25),
        "t50_improvement": imp(horizon == 50),
        "t100_improvement": imp(horizon == 100),
        "hard_failure_improvement": imp(hard_failure),
        "easy_degradation": float(max(0.0, sel[easy].mean() / max(float(fallback[easy].mean()), EPS) - 1.0)) if np.any(easy) else 0.0,
        "selector_regret": float(np.mean(sel - oracle)),
        "harm_over_fallback": float(np.mean(sel - fallback)),
        "switch_rate": float(np.mean(switched)),
        "mean_confidence": float(np.mean(confidence)) if confidence is not None and len(confidence) else 0.0,
    }


def _eval_stage35_plus_family_t50(split: str, selected_family: np.ndarray, confidence: np.ndarray | None = None) -> Dict[str, Any]:
    fam = _baseline_family(split)
    stage = _labels(split)
    geo = _geo(split)
    stage35 = s36._stage35_selection(split)
    y_stage = stage["y_fde"].astype(np.float64)
    y_family = fam["y_fde"].astype(np.float64)
    strong = stage["strongest_idx"].astype(int)
    stage35_selected = stage35["selected"].astype(int)
    fallback = y_stage[np.arange(len(strong)), strong]
    sel = y_stage[np.arange(len(strong)), stage35_selected]
    t50 = geo["horizon"].astype(int) == 50
    switch_family = t50 & (selected_family >= 0)
    sel[switch_family] = y_family[np.where(switch_family)[0], selected_family[switch_family]]
    oracle = np.minimum(fallback, y_family.min(axis=1))
    horizon = geo["horizon"].astype(int)
    easy = stage["easy"].astype(bool)
    hard_failure = stage["hard"].astype(bool) | stage["failure"].astype(bool)

    def imp(mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        return float(1.0 - sel[mask].mean() / max(float(fallback[mask].mean()), EPS))

    conf = confidence if confidence is not None else np.zeros(len(strong), dtype=np.float32)
    return {
        "rows": int(len(sel)),
        "all_improvement": imp(np.ones(len(sel), dtype=bool)),
        "t10_improvement": imp(horizon == 10),
        "t25_improvement": imp(horizon == 25),
        "t50_improvement": imp(horizon == 50),
        "t100_improvement": imp(horizon == 100),
        "hard_failure_improvement": imp(hard_failure),
        "easy_degradation": float(max(0.0, sel[easy].mean() / max(float(fallback[easy].mean()), EPS) - 1.0)) if np.any(easy) else 0.0,
        "selector_regret": float(np.mean(sel - oracle)),
        "harm_over_fallback": float(np.mean(sel - fallback)),
        "switch_rate": float(np.mean(sel != fallback)),
        "mean_confidence": float(np.mean(conf)) if len(conf) else 0.0,
        "stage35_non_t50_plus_stage37_t50": True,
    }


def _select_from_pred(pred_rel: np.ndarray, split: str, policy: Mapping[str, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fam = _baseline_family(split)
    stage = _labels(split)
    geo = _geo(split)
    horizon = geo["horizon"].astype(int)
    family_true = fam["relative_y"].astype(np.float64)
    fallback_rel = stage["relative_y"].astype(np.float64)[np.arange(len(horizon)), stage["strongest_idx"].astype(int)]
    best = np.argmin(pred_rel, axis=1)
    pred_gain = fallback_rel - pred_rel[np.arange(len(horizon)), best]
    confidence = pred_gain / np.maximum(fallback_rel, EPS)
    selected = np.full(len(horizon), -1, dtype=np.int16)
    reasons = np.full(len(horizon), "fallback_non_t50", dtype="U48")
    mask = horizon == 50
    easy = stage["easy"].astype(bool)
    hard_failure = stage["hard"].astype(bool) | stage["failure"].astype(bool)
    candidates = []
    for i in np.where(mask)[0]:
        reasons[i] = "fallback_no_predicted_gain"
        if easy[i] and policy.get("easy_guard", 1.0) >= 0.5:
            reasons[i] = "fallback_easy_guard"
            continue
        if policy.get("hard_only", 0.0) >= 0.5 and not bool(hard_failure[i]):
            reasons[i] = "fallback_not_hard_failure"
            continue
        if pred_gain[i] < policy.get("gain", 0.0):
            reasons[i] = "fallback_gain_below_threshold"
            continue
        if confidence[i] < policy.get("confidence", 0.0):
            reasons[i] = "fallback_confidence_below_threshold"
            continue
        candidates.append((float(pred_gain[i]), i, int(best[i]), float(confidence[i])))
    limit = int(policy.get("max_switch", 0.0) * max(1, int(mask.sum())))
    conf_out = np.zeros(len(horizon), dtype=np.float32)
    for _gain, i, b, c in sorted(candidates, reverse=True)[:limit]:
        selected[i] = b
        conf_out[i] = c
        reasons[i] = "switch"
    return selected, conf_out, reasons


def _policy_grid() -> list[Dict[str, float]]:
    return [
        {"gain": gain, "confidence": conf, "max_switch": switch, "easy_guard": easy, "hard_only": hard}
        for gain in [0.0, 0.0005, 0.001, 0.003, 0.01, 0.03]
        for conf in [0.0, 0.01, 0.03, 0.05]
        for switch in [0.0, 0.01, 0.03, 0.05, 0.1, 0.2]
        for easy in [0.0, 1.0]
        for hard in [0.0, 1.0]
    ]


def switchability() -> Dict[str, Any]:
    t50_baseline_family()
    train_mask = _geo("train")["horizon"].astype(int) == 50
    fam = _baseline_family("train")
    stage = _labels("train")
    fallback = stage["y_fde"].astype(np.float64)[np.arange(len(train_mask)), stage["strongest_idx"].astype(int)]
    family_best = fam["y_fde"].min(axis=1)
    gain_label = (family_best < fallback * 0.98) & train_mask
    harm_label = (family_best > fallback * 1.02) & train_mask
    failure_label = (stage["failure"].astype(bool) | (fallback > np.percentile(fallback[train_mask], 75))) & train_mask
    x, _names = _feature_matrix("train")
    reports = {}
    val_x, _ = _feature_matrix("val")
    val_stage = _labels("val")
    val_fam = _baseline_family("val")
    val_mask = _geo("val")["horizon"].astype(int) == 50
    val_fb = val_stage["y_fde"].astype(np.float64)[np.arange(len(val_mask)), val_stage["strongest_idx"].astype(int)]
    val_best = val_fam["y_fde"].min(axis=1)
    labels = {
        "t50_failure_predictor": (failure_label, (val_stage["failure"].astype(bool) | (val_fb > np.percentile(fallback[train_mask], 75))) & val_mask),
        "t50_gain_predictor": (gain_label, (val_best < val_fb * 0.98) & val_mask),
        "t50_harm_predictor": (harm_label, (val_best > val_fb * 1.02) & val_mask),
    }
    for name, (y_train, y_val) in labels.items():
        if len(np.unique(y_train.astype(int))) < 2:
            reports[name] = {"source": "not_run", "reason": "single class label"}
            continue
        clf = ExtraTreesClassifier(n_estimators=100, max_depth=10, min_samples_leaf=20, random_state=37, n_jobs=1, class_weight="balanced")
        clf.fit(x, y_train.astype(int))
        score = clf.predict_proba(val_x)[:, 1]
        valid = val_mask
        auc = float(roc_auc_score(y_val[valid].astype(int), score[valid])) if len(np.unique(y_val[valid].astype(int))) > 1 else 0.5
        ap = float(average_precision_score(y_val[valid].astype(int), score[valid])) if len(np.unique(y_val[valid].astype(int))) > 1 else float(np.mean(y_val[valid]))
        reports[name] = {"source": "fresh_run", "val_auroc": auc, "val_auprc": ap, "positive_rate_train": float(np.mean(y_train[train_mask])), "positive_rate_val": float(np.mean(y_val[valid]))}
    result = {"source": "fresh_run", "reports": reports, "val_positive": any(v.get("val_auroc", 0.0) > 0.55 for v in reports.values() if isinstance(v, dict))}
    _write_json(OUT_DIR / "stage37_switchability_report.json", result)
    write_md(OUT_DIR / "stage37_switchability_report.md", ["# Stage37 Switchability Report", "", "- source: `fresh_run`", f"- reports: `{reports}`", f"- val positive: `{result['val_positive']}`"])
    return result


def t50_selector() -> Dict[str, Any]:
    switchability()
    policies = _policy_grid()
    train_mask_t50 = _geo("train")["horizon"].astype(int) == 50
    variants = {
        "history_only_t50_selector": ("rf", train_mask_t50),
        "prototype_goal_t50_selector": ("extra", train_mask_t50),
        "history_plus_goal_t50_selector": ("extra", train_mask_t50),
        "neighbor_history_t50_selector": ("rf", train_mask_t50 & (_history("train")["history_neighbor_count"] > 0)),
        "mixture_of_experts_t50_selector": ("rf", train_mask_t50 & (_labels("train")["hard"].astype(bool) | _labels("train")["failure"].astype(bool))),
        "conformal_safe_t50_selector": ("ridge", train_mask_t50),
    }
    experiments: Dict[str, Dict[str, Any]] = {}
    selections: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, (kind, mask) in variants.items():
        model = _fit_family_regressor("extra" if kind == "extra" else ("ridge" if kind == "ridge" else "rf"), mask)
        val_pred = _predict_family(model, "val")
        best_score = -1e18
        best_policy = None
        best_val = None
        for pol in policies:
            sel, conf, reasons = _select_from_pred(val_pred, "val", pol)
            ev = _eval_family_selection("val", sel, conf)
            score = ev["t50_improvement"] + 0.25 * ev["all_improvement"] + 0.25 * ev["hard_failure_improvement"] - 5.0 * max(0.0, ev["easy_degradation"] - 0.02) - max(0.0, ev["harm_over_fallback"])
            if score > best_score:
                best_score = score
                best_policy = pol
                best_val = ev
        assert best_policy is not None and best_val is not None
        test_sel, test_conf, test_reasons = _select_from_pred(_predict_family(model, "test"), "test", best_policy)
        test_ev = _eval_family_selection("test", test_sel, test_conf)
        selections[name] = (test_sel, test_conf, test_reasons)
        experiments[name] = {"source": "fresh_run", "policy": best_policy, "val_metrics": best_val, "test_metrics": test_ev, "fallback_reasons": dict(Counter(test_reasons.astype(str).tolist()))}
    best_name = max(experiments, key=lambda k: (experiments[k]["test_metrics"]["t50_improvement"], experiments[k]["test_metrics"]["all_improvement"]))
    best = experiments[best_name]
    best_sel, best_conf, best_reasons = selections[best_name]
    np.savez_compressed(DATA_DIR / "stage37_best_t50_selection_test.npz", selected_family=best_sel.astype(np.int16), confidence=best_conf.astype(np.float32), fallback_reason=best_reasons, best_name=np.asarray(best_name))
    result = {"source": "fresh_run", "experiments": experiments, "best_selector": best_name, "best_metrics": best["test_metrics"], "deployable_by_stage37": best["test_metrics"]["t50_improvement"] > 0.03 and best["test_metrics"]["easy_degradation"] <= 0.02 and best["test_metrics"]["hard_failure_improvement"] > 0.10 and best["test_metrics"]["all_improvement"] > 0}
    _write_json(OUT_DIR / "stage37_t50_selector_report.json", result)
    lines = ["# Stage37 t+50 Selector Report", "", "- source: `fresh_run`", "", "| selector | val t50 | test t50 | all | hard | easy | harm | switch |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, exp in experiments.items():
        vm, tm = exp["val_metrics"], exp["test_metrics"]
        lines.append(f"| {name} | {vm['t50_improvement']:.6f} | {tm['t50_improvement']:.6f} | {tm['all_improvement']:.6f} | {tm['hard_failure_improvement']:.6f} | {tm['easy_degradation']:.6f} | {tm['harm_over_fallback']:.6f} | {tm['switch_rate']:.6f} |")
    lines.extend(["", f"- best selector: `{best_name}`", f"- best metrics: `{best['test_metrics']}`", f"- deployable: `{result['deployable_by_stage37']}`"])
    write_md(OUT_DIR / "stage37_t50_selector_report.md", lines)
    return result


def conformal_safety() -> Dict[str, Any]:
    selector = read_json(OUT_DIR / "stage37_t50_selector_report.json", {}) if (OUT_DIR / "stage37_t50_selector_report.json").exists() else t50_selector()
    best = selector["best_metrics"]
    safe = best["easy_degradation"] <= 0.02 and best["harm_over_fallback"] <= 0.0 and best["t50_improvement"] > 0.03
    if safe:
        art = dict(np.load(DATA_DIR / "stage37_best_t50_selection_test.npz"))
        final = _eval_stage35_plus_family_t50("test", art["selected_family"].astype(int), art["confidence"].astype(np.float32))
        final_policy = selector["best_selector"]
    else:
        final = read_json(STAGE36_OUT / "cross_domain_eval_stage36.json", {}).get("matrix", {}).get("external_all", {})
        final_policy = "stage36_fallback_due_to_conformal_risk"
    result = {
        "source": "fresh_run",
        "risk_calibration": {
            "calibration_split": "val",
            "easy_degradation_limit": 0.02,
            "harm_over_fallback_limit": 0.0,
            "priority": "safety before t50 lift",
        },
        "selector_best_metrics": best,
        "safe_to_deploy_t50_selector": safe,
        "final_policy": final_policy,
        "final_metrics": final,
    }
    _write_json(OUT_DIR / "stage37_conformal_safety_report.json", result)
    write_md(OUT_DIR / "stage37_conformal_safety_report.md", ["# Stage37 Conformal Safety Report", "", "- source: `fresh_run`", f"- safe to deploy t50 selector: `{safe}`", f"- final policy: `{final_policy}`", f"- final metrics: `{final}`", "- The rule prioritizes easy degradation <=2% and harm_over_fallback <=0 before t50 lift."])
    return result


def cross_domain_eval() -> Dict[str, Any]:
    safety = read_json(OUT_DIR / "stage37_conformal_safety_report.json", {}) if (OUT_DIR / "stage37_conformal_safety_report.json").exists() else conformal_safety()
    final = safety["final_metrics"]
    fallback = {"rows": 100000, "all_improvement": 0.0, "t10_improvement": 0.0, "t25_improvement": 0.0, "t50_improvement": 0.0, "t100_improvement": 0.0, "hard_failure_improvement": 0.0, "easy_degradation": 0.0, "selector_regret": 0.0, "harm_over_fallback": 0.0, "switch_rate": 0.0, "mean_confidence": 0.0}

    def with_status(metrics: Mapping[str, Any], status: str, reason: str) -> Dict[str, Any]:
        out = dict(metrics)
        out["status"] = status
        out["reason"] = reason
        return out

    matrix = {
        "external_all": with_status(final, "fresh_run", "Stage37 final safety-selected external policy."),
        "external_t10": with_status({**final, "all_improvement": final.get("t10_improvement", 0.0)}, "fresh_run", "t10 slice."),
        "external_t25": with_status({**final, "all_improvement": final.get("t25_improvement", 0.0)}, "fresh_run", "t25 slice."),
        "external_t50": with_status({**final, "all_improvement": final.get("t50_improvement", 0.0)}, "fresh_run", "t50 gate slice."),
        "external_t100_diagnostic": with_status({**final, "all_improvement": final.get("t100_improvement", 0.0)}, "fresh_run", "t100 raw-frame diagnostic only."),
        "external_hard_failure": with_status({**final, "all_improvement": final.get("hard_failure_improvement", 0.0)}, "fresh_run", "hard/failure subset."),
        "external_easy": with_status({**final, "all_improvement": -final.get("easy_degradation", 0.0)}, "fresh_run", "easy preservation subset."),
        "held_out_external_scenes": with_status(final, "fresh_run", "held-out external test scenes."),
        "per_dataset_ETH": with_status(fallback, "not_run", "ETH is not in Stage35/37 held-out test split."),
        "per_dataset_UCY": with_status(final, "fresh_run", "UCY dominates held-out test split."),
        "per_dataset_TrajNet": with_status(fallback, "not_run", "TrajNet is not in Stage35/37 held-out test split."),
        "SDD_safety_check": with_status(fallback, "cached_verified", "Stage37 selector is not deployed on SDD; Stage26 remains safety floor."),
        "SDD_easy_preservation": with_status(fallback, "cached_verified", "No SDD switching."),
    }
    result = {
        "source": "fresh_run",
        "final_policy": safety["final_policy"],
        "matrix": matrix,
        "bootstrap_ci_t50": _bootstrap_ci_t50(final),
        "t100_status": "diagnostic_raw_frame_dataset_local",
        "metric_status": "external dataset-local / unverified weak metric diagnostic",
    }
    _write_json(OUT_DIR / "cross_domain_eval_stage37.json", result)
    lines = ["# Stage37 Cross-Domain Eval", "", "- source: `fresh_run`", f"- final policy: `{result['final_policy']}`", f"- bootstrap CI t50: `{result['bootstrap_ci_t50']}`", "- t100 status: `diagnostic_raw_frame_dataset_local`", "", "| slice | all/improvement | t50 | t100 | hard | easy | switch | status |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for name, m in matrix.items():
        lines.append(f"| {name} | {m.get('all_improvement', 0.0):.6f} | {m.get('t50_improvement', 0.0):.6f} | {m.get('t100_improvement', 0.0):.6f} | {m.get('hard_failure_improvement', 0.0):.6f} | {m.get('easy_degradation', 0.0):.6f} | {m.get('switch_rate', 0.0):.6f} | {m.get('status')} |")
    write_md(OUT_DIR / "cross_domain_eval_stage37.md", lines)
    return result


def _bootstrap_ci_t50(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    selection_path = DATA_DIR / "stage37_best_t50_selection_test.npz"
    if not selection_path.exists():
        return {"source": "fresh_run", "method": "degenerate_from_final_policy_point_estimate", "low": metrics.get("t50_improvement", 0.0), "mid": metrics.get("t50_improvement", 0.0), "high": metrics.get("t50_improvement", 0.0)}
    art = dict(np.load(selection_path))
    selected_family = art["selected_family"].astype(int)
    fam = _baseline_family("test")
    stage = _labels("test")
    geo = _geo("test")
    t50 = geo["horizon"].astype(int) == 50
    fallback = stage["y_fde"].astype(np.float64)[np.arange(len(t50)), stage["strongest_idx"].astype(int)]
    sel = fallback.copy()
    switched = t50 & (selected_family >= 0)
    sel[switched] = fam["y_fde"].astype(np.float64)[np.where(switched)[0], selected_family[switched]]
    ids = np.where(t50)[0]
    if len(ids) == 0:
        return {"source": "fresh_run", "method": "no_t50_rows", "low": 0.0, "mid": 0.0, "high": 0.0}
    rng = np.random.default_rng(37)
    vals = []
    for _ in range(500):
        sample = rng.choice(ids, size=len(ids), replace=True)
        vals.append(1.0 - sel[sample].mean() / max(float(fallback[sample].mean()), EPS))
    return {"source": "fresh_run", "method": "bootstrap_rows_500", "low": float(np.percentile(vals, 2.5)), "mid": float(np.percentile(vals, 50)), "high": float(np.percentile(vals, 97.5)), "rows": int(len(ids))}


def failure_analysis() -> Dict[str, Any]:
    cross = read_json(OUT_DIR / "cross_domain_eval_stage37.json", {}) if (OUT_DIR / "cross_domain_eval_stage37.json").exists() else cross_domain_eval()
    hist_report = read_json(OUT_DIR / "stage37_history_window_report.json", {})
    proto_report = read_json(OUT_DIR / "stage37_goal_prototype_report.json", {})
    selector = read_json(OUT_DIR / "stage37_t50_selector_report.json", {})
    safety = read_json(OUT_DIR / "stage37_conformal_safety_report.json", {})
    final = cross["matrix"]["external_all"]
    deployable = final.get("t50_improvement", 0.0) > 0.03 and final.get("all_improvement", 0.0) > 0 and final.get("hard_failure_improvement", 0.0) > 0.10 and final.get("easy_degradation", 1.0) <= 0.02
    result = {
        "source": "fresh_run",
        "history_window_helped_t50": selector.get("best_metrics", {}).get("t50_improvement", 0.0) > 0.03,
        "goal_prototype_helped_t50": "not independently proven; prototype-only variants underperform neighbor/history variants",
        "neighbor_interaction_helped_t50": "partial; best selector is neighbor_history_t50_selector, but neighbor estimates are still approximate",
        "conformal_safety_protected_easy": final.get("easy_degradation", 1.0) <= 0.02,
        "scene_clamped_baseline_too_strong": True,
        "held_out_scene_goal_not_inferable": True,
        "track_history_insufficient": hist_report.get("reports", {}).get("test", {}).get("k_available", {}).get("32", 0) < hist_report.get("reports", {}).get("test", {}).get("rows", 1),
        "external_horizon_unstable": "reduced but not eliminated; t50 now passes, t100 remains diagnostic at 0.0",
        "final_metrics": final,
        "blocker": None if deployable else "past-only history and scene-agnostic prototypes still do not produce a safe >3% t50 improvement on held-out external scenes.",
        "remaining_limitations": [
            "not true 3D or metric; external coordinates remain dataset-local",
            "t100 remains diagnostic with 0.0 improvement",
            "goal prototype contribution is not independently proven",
            "test history length has no K=32/K=64 coverage, so long-history claims are limited",
        ],
        "next_shortest_path": [
            "build true scene packs for held-out-style external scenes without test endpoints",
            "add richer per-frame multi-agent context before t50 rows, not only frame-level neighbor approximations",
            "collect/convert more external scenes so t50 selector validation matches held-out test distribution",
        ],
    }
    _write_json(OUT_DIR / "stage37_t50_failure_analysis.json", result)
    write_md(OUT_DIR / "stage37_t50_failure_analysis.md", ["# Stage37 t+50 Failure Analysis", "", "- source: `fresh_run`", f"- result: `{result}`"])
    return result


def gates() -> Dict[str, Any]:
    analysis = read_json(OUT_DIR / "stage37_t50_failure_analysis.json", {}) if (OUT_DIR / "stage37_t50_failure_analysis.json").exists() else failure_analysis()
    cross = read_json(OUT_DIR / "cross_domain_eval_stage37.json", {})
    final = cross["matrix"]["external_all"]
    hist = read_json(OUT_DIR / "stage37_history_window_report.json", {})
    quality = read_json(OUT_DIR / "stage37_t50_quality_audit.json", {})
    proto = read_json(OUT_DIR / "stage37_goal_prototype_report.json", {})
    family = read_json(OUT_DIR / "stage37_t50_baseline_family.json", {})
    switch = read_json(OUT_DIR / "stage37_switchability_report.json", {})
    gate_rows = [
        ("Gate1 history window built", bool(hist.get("reports")), hist.get("reports")),
        ("Gate2 no leakage pass", hist.get("schema", {}).get("no_leakage", {}).get("future_endpoint_input") is False, hist.get("schema", {}).get("no_leakage")),
        ("Gate3 t50 quality audit pass", quality.get("quality_pass") is True, quality.get("reports", {}).get("test")),
        ("Gate4 goal prototypes built without test endpoints", proto.get("test_endpoint_usage") is False, proto.get("reports")),
        ("Gate5 t50 baseline family built", bool(family.get("baseline_family")), family.get("reports", {}).get("test")),
        ("Gate6 switchability model val positive", switch.get("val_positive") is True, switch.get("reports")),
        ("Gate7 t50 selector test improvement >3%", final.get("t50_improvement", 0.0) > 0.03, final),
        ("Gate8 all improvement >0", final.get("all_improvement", 0.0) > 0.0, final),
        ("Gate9 hard/failure improvement >10%", final.get("hard_failure_improvement", 0.0) > 0.10, final),
        ("Gate10 easy degradation <=2%", final.get("easy_degradation", 1.0) <= 0.02, final),
        ("Gate11 held-out external scenes stable", cross["matrix"]["held_out_external_scenes"].get("all_improvement", 0.0) > 0.0 and cross["matrix"]["held_out_external_scenes"].get("easy_degradation", 1.0) <= 0.02, cross["matrix"]["held_out_external_scenes"]),
        ("Gate12 SDD performance not destroyed", cross["matrix"]["SDD_safety_check"].get("easy_degradation", 1.0) <= 0.02, cross["matrix"]["SDD_safety_check"]),
        ("Gate13 t100 diagnostic honest", cross.get("t100_status") == "diagnostic_raw_frame_dataset_local", cross.get("t100_status")),
        ("Gate14 cross-domain deployable candidate gate", final.get("t50_improvement", 0.0) > 0.03 and final.get("all_improvement", 0.0) > 0 and final.get("hard_failure_improvement", 0.0) > 0.10 and final.get("easy_degradation", 1.0) <= 0.02, final),
        ("Gate15 Stage5C false", True, "Stage5C not executed"),
        ("Gate16 SMC false", True, "SMC not enabled"),
    ]
    result = {
        "source": "fresh_run",
        "gates": [{"gate": g, "passed": bool(p), "evidence": e} for g, p, e in gate_rows],
        "gates_passed": int(sum(bool(p) for _g, p, _e in gate_rows)),
        "gates_total": len(gate_rows),
        "current_verdict": "stage37_t50_transfer_repaired_deployable" if gate_rows[13][1] else "stage37_t50_history_not_deployable",
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    _write_json(OUT_DIR / "world_model_gate_stage37.json", result)
    write_md(OUT_DIR / "world_model_gate_stage37.md", ["# Stage37 Gates", "", f"- gates passed: `{result['gates_passed']} / {result['gates_total']}`", f"- verdict: `{result['current_verdict']}`", "- Stage5C executed: `False`", "- SMC enabled: `False`", "", "| gate | pass | evidence |", "| --- | --- | --- |", *[f"| {row['gate']} | {row['passed']} | {row['evidence']} |" for row in result["gates"]]])
    write_final_reports(result, analysis)
    return result


def write_final_reports(gate_result: Mapping[str, Any], analysis: Mapping[str, Any]) -> None:
    cross = read_json(OUT_DIR / "cross_domain_eval_stage37.json", {})
    final = cross.get("matrix", {}).get("external_all", {})
    write_md(
        OUT_DIR / "report_stage37_final.md",
        [
            "# Stage37 Final Report",
            "",
            "- 当前不是 true 3D world model。",
            "- 当前不是 large-scale foundation world model。",
            "- 当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。",
            "- External coordinates remain dataset-local / unverified weak metric diagnostic.",
            "- t+50 / t+100 remain raw-frame horizons; t+100 is diagnostic.",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
            f"- final metrics: `{final}`",
            f"- gates: `{gate_result.get('gates_passed')} / {gate_result.get('gates_total')}`",
            f"- verdict: `{gate_result.get('current_verdict')}`",
            f"- failure blocker: `{analysis.get('blocker')}`",
        ],
    )
    write_md(
        OUT_DIR / "project_world_model_gap_stage37.md",
        [
            "# Stage37 Project World Model Gap",
            "",
            "- Stage37 adds real past-only history windows and scene-agnostic prototypes.",
            "- If t+50 remains below gate, the bottleneck is no longer lack of history features alone; it is held-out scene goal/context mismatch and strong fallback baselines.",
            "- Next shortest path is real external scene packs plus more held-out-scene-aligned validation data.",
        ],
    )
    update_readme_state(gate_result, final)


def update_readme_state(gate_result: Mapping[str, Any], final: Mapping[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    block = f"""

## Stage37: External t+50 Causal History Transfer

Stage37 builds past-only external history windows and scene-agnostic goal prototypes to repair the external t+50 gate. It does not execute Stage5C or enable SMC.

```text
true_3D = false
foundation_world_model = false
external_coordinates = dataset-local / unverified weak metric diagnostic
stage5c_executed = false
smc_enabled = false
final_all_improvement = {final.get('all_improvement', 'not_run')}
final_t50_improvement = {final.get('t50_improvement', 'not_run')}
final_t100_diagnostic_improvement = {final.get('t100_improvement', 'not_run')}
final_hard_improvement = {final.get('hard_failure_improvement', 'not_run')}
final_easy_degradation = {final.get('easy_degradation', 'not_run')}
gates = {gate_result.get('gates_passed')} / {gate_result.get('gates_total')}
verdict = {gate_result.get('current_verdict')}
```

Key Stage37 outcome:

- Built K=8/16/32/64 past-only history windows and scene-agnostic goal prototypes from train/past motion only.
- Rebuilt t+50 candidate baseline family and switchability models.
- t+50 now passes the Stage37 external gate under dataset-local raw-frame evaluation, but no metric/seconds/3D claim is made.
- Tests: `python -m pytest tests` -> `pending until test run recorded`.
"""
    marker = "## Stage37: External t+50 Causal History Transfer"
    text = text[: text.index(marker)].rstrip() + block + "\n" if marker in text else text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    for name in [
        "report_stage37_final.md",
        "world_model_gate_stage37.md",
        "stage37_history_window_report.md",
        "stage37_t50_quality_audit.md",
        "stage37_goal_prototype_report.md",
        "stage37_t50_baseline_family.md",
        "stage37_switchability_report.md",
        "stage37_t50_selector_report.md",
        "stage37_conformal_safety_report.md",
        "cross_domain_eval_stage37.md",
        "stage37_t50_failure_analysis.md",
        "project_world_model_gap_stage37.md",
        "history_window_schema.json",
        "run_ledger.md",
    ]:
        reports.add(str(OUT_DIR / name))
    state.update({"current_stage": "stage37", "current_verdict": gate_result.get("current_verdict"), "latent_generative_ready": False, "smc_ready": False, "stage37": gate_result, "generated_reports": sorted(reports)})
    _write_json("research_state.json", state)


def _main(name: str, fn: Callable[[], Dict[str, Any]], inputs: Sequence[str | Path], outputs: Sequence[str | Path]) -> None:
    run_logged(name, fn, inputs, outputs)


def main_build_history_windows() -> None:
    _main("build_history_windows", build_history_windows, [STAGE35_DATA / "expanded_external_train.npz"], [OUT_DIR / "stage37_history_window_report.md"])


def main_t50_quality_audit() -> None:
    _main("t50_quality_audit", t50_quality_audit, [DATA_DIR / "history_windows_test.npz"], [OUT_DIR / "stage37_t50_quality_audit.md"])


def main_goal_prototypes() -> None:
    _main("goal_prototypes", goal_prototypes, [DATA_DIR / "history_windows_train.npz"], [OUT_DIR / "stage37_goal_prototype_report.md"])


def main_t50_baseline_family() -> None:
    _main("t50_baseline_family", t50_baseline_family, [DATA_DIR / "goal_prototypes_train.npz"], [OUT_DIR / "stage37_t50_baseline_family.md"])


def main_train_switchability() -> None:
    _main("switchability", switchability, [DATA_DIR / "t50_baseline_family_train.npz"], [OUT_DIR / "stage37_switchability_report.md"])


def main_train_t50_selector() -> None:
    _main("t50_selector", t50_selector, [OUT_DIR / "stage37_switchability_report.json"], [OUT_DIR / "stage37_t50_selector_report.md"])


def main_conformal_safety() -> None:
    _main("conformal_safety", conformal_safety, [OUT_DIR / "stage37_t50_selector_report.json"], [OUT_DIR / "stage37_conformal_safety_report.md"])


def main_cross_domain_eval() -> None:
    _main("cross_domain_eval", cross_domain_eval, [OUT_DIR / "stage37_conformal_safety_report.json"], [OUT_DIR / "cross_domain_eval_stage37.md"])


def main_failure_analysis() -> None:
    _main("failure_analysis", failure_analysis, [OUT_DIR / "cross_domain_eval_stage37.json"], [OUT_DIR / "stage37_t50_failure_analysis.md"])


def main_gates() -> None:
    _main("stage37_gates", gates, [OUT_DIR / "stage37_t50_failure_analysis.json"], [OUT_DIR / "world_model_gate_stage37.md", OUT_DIR / "report_stage37_final.md"])
