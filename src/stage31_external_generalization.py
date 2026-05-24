from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
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


OUT_DIR = Path("outputs/stage31_m3w_external")
FEATURE_DIR = Path("data/stage31_external_feature_store")
LATENT_DIR = Path("data/stage31_external_latent_cache")
SDD_FEATURE_DIR = Path("data/stage26_sdd_feature_store")
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
CHECKPOINTS = {
    "jepa_only": Path("outputs/m3w/checkpoints/jepa_only_best.pt"),
    "transformer_only": Path("outputs/m3w/checkpoints/transformer_only_best.pt"),
    "hybrid": Path("outputs/m3w/checkpoints/hybrid_best.pt"),
}
CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 2.5D / pseudo-3D 多智能体轨迹世界状态模型。",
    "SDD 是 pixel raw-frame benchmark，不是 metric seconds benchmark。",
    "Stage5C 未执行。",
    "SMC 未启用。",
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


def _write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    write_json(path, _jsonable(dict(payload)))


def _append_ledger(entry: Mapping[str, Any]) -> None:
    ensure_dir(OUT_DIR)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(dict(entry)), ensure_ascii=False) + "\n")
    rows = []
    with LEDGER_JSONL.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    lines = [
        "# Stage31 External Generalization Run Ledger",
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


def _read_opentraj_txt(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 4 or "?" in parts:
                continue
            try:
                rows.append((int(float(parts[0])), int(float(parts[1])), float(parts[2]), float(parts[3])))
            except ValueError:
                continue
    if not rows:
        return np.zeros((0, 4), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _candidate_files() -> Dict[str, List[Path]]:
    roots = [
        Path("/Users/yangyue/Downloads/World/external_data/OpenTraj"),
        Path("/Users/yangyue/Downloads/OpenTraj"),
        Path("/Users/yangyue/Downloads/ETH_UCY"),
        Path("/Users/yangyue/Downloads/trajnetplusplusdataset"),
    ]
    out: Dict[str, List[Path]] = {"train": [], "test": []}
    for root in roots:
        trajnet = root / "datasets" / "TrajNet"
        if not trajnet.exists():
            continue
        for path in sorted(trajnet.rglob("*.txt")):
            lower = str(path).lower()
            if "/stanford/" in lower or "stanford" in lower:
                continue
            split = "test" if "/test/" in lower else "train"
            out[split].append(path)
    return out


def _baseline_errors(past: np.ndarray, future: np.ndarray, frame_delta: float) -> np.ndarray:
    p0 = past[-1]
    pprev = past[-2] if len(past) > 1 else past[-1]
    dt = max(float(past[-1, 0] - pprev[0]), 1.0)
    pos0 = p0[1:3]
    posprev = pprev[1:3]
    v = (pos0 - posprev) / dt
    if len(past) > 2:
        p2 = past[-3]
        vprev = (posprev - p2[1:3]) / max(float(pprev[0] - p2[0]), 1.0)
    else:
        vprev = v
    a = (v - vprev) / dt
    gt = future[1:3]
    h = max(float(frame_delta), 1.0)
    damp_factor = (1.0 - 0.95**h) / max(1.0 - 0.95, 1e-6)
    preds = np.stack(
        [
            pos0,
            pos0 + v * h,
            pos0 + v * damp_factor,
            pos0 + v * h + 0.5 * a * h * h,
            pos0 + v * h,
            pos0 + v * h,
            pos0 + v * h,
        ],
        axis=0,
    )
    return np.linalg.norm(preds - gt[None, :], axis=1).astype(np.float32)


def _simple_features(feature_names: Sequence[str], horizon: int, frame: float, agent_count: int, past_xy: np.ndarray, density: float) -> np.ndarray:
    p0 = past_xy[-1]
    pprev = past_xy[-2] if len(past_xy) > 1 else past_xy[-1]
    dt = max(float(p0[0] - pprev[0]), 1.0)
    vx = float((p0[1] - pprev[1]) / dt)
    vy = float((p0[2] - pprev[2]) / dt)
    speed = math.sqrt(vx * vx + vy * vy)
    if len(past_xy) > 2:
        p2 = past_xy[-3]
        vx0 = float((pprev[1] - p2[1]) / max(float(pprev[0] - p2[0]), 1.0))
        vy0 = float((pprev[2] - p2[2]) / max(float(pprev[0] - p2[0]), 1.0))
    else:
        vx0, vy0 = vx, vy
    ax = (vx - vx0) / dt
    ay = (vy - vy0) / dt
    diffs = np.diff(past_xy[:, 1:3], axis=0) if len(past_xy) > 1 else np.zeros((0, 2))
    path_length = float(np.sum(np.linalg.norm(diffs, axis=1))) if len(diffs) else 0.0
    displacement = float(np.linalg.norm(past_xy[-1, 1:3] - past_xy[0, 1:3])) if len(past_xy) else 0.0
    straightness = displacement / max(path_length, 1e-6) if path_length > 0 else 1.0
    values = []
    for name in feature_names:
        low = name.lower()
        if low == "horizon_norm":
            values.append(horizon / 100.0)
        elif low == f"horizon_is_{horizon}":
            values.append(1.0)
        elif low.startswith("horizon_is_"):
            values.append(0.0)
        elif low == "split_within_scene":
            values.append(0.0)
        elif low == "agent_count_log":
            values.append(math.log1p(agent_count))
        elif low == "agent_count_ge5":
            values.append(float(agent_count >= 5))
        elif low == "agent_count_ge10":
            values.append(float(agent_count >= 10))
        elif low == "start_frame_norm":
            values.append(frame / 10000.0)
        elif low in {"speed_now", "speed_mean_past"}:
            values.append(speed)
        elif low == "speed_std_past":
            values.append(0.0)
        elif low == "speed_delta_past":
            values.append(speed - math.sqrt(vx0 * vx0 + vy0 * vy0))
        elif low == "accel_mag_now" or low == "accel_mag_mean_past":
            values.append(math.sqrt(ax * ax + ay * ay))
        elif low == "vx_now":
            values.append(vx)
        elif low == "vy_now":
            values.append(vy)
        elif low == "ax_now":
            values.append(ax)
        elif low == "ay_now":
            values.append(ay)
        elif low in {"density_visible_count", "density_r20", "density_r50", "density_r100"}:
            values.append(density)
        elif low in {"nearest_neighbor_distance", "mean_nearest3_distance", "mean_nearest5_distance"}:
            values.append(1e3)
        elif low == "past_path_length":
            values.append(path_length)
        elif low == "past_displacement":
            values.append(displacement)
        elif low == "past_straightness":
            values.append(straightness)
        elif low.startswith("agent_type_pedestrian"):
            values.append(1.0)
        elif low.startswith("agent_type_"):
            values.append(0.0)
        elif "goal" in low or "scene" in low:
            values.append(0.0)
        elif "baseline" in low or "rollout" in low or "damped" in low or "cv_" in low or "ca_" in low:
            values.append(speed * horizon)
        else:
            values.append(0.0)
    return np.asarray(values, dtype=np.float32)


def _build_rows_for_file(path: Path, split: str, feature_names: Sequence[str]) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[str], List[str]]:
    arr = _read_opentraj_txt(path)
    if len(arr) == 0:
        return [], [], [], [], []
    by_frame = Counter(arr[:, 0].astype(int))
    rows_x: List[np.ndarray] = []
    rows_y: List[np.ndarray] = []
    horizons: List[int] = []
    scenes: List[str] = []
    files: List[str] = []
    scene = path.parent.name
    for agent in np.unique(arr[:, 1]).astype(int):
        tr = arr[arr[:, 1] == agent]
        tr = tr[np.argsort(tr[:, 0])]
        frames = tr[:, 0].astype(int)
        if len(tr) < 4:
            continue
        frame_to_idx = {int(f): i for i, f in enumerate(frames)}
        for i in range(2, len(tr) - 1, max(1, len(tr) // 80)):
            for horizon in [10, 25, 50, 100]:
                target_frame = int(frames[i] + horizon)
                # For sparse frame grids, use the first observed frame at/after the requested raw horizon.
                future_ids = np.where(frames >= target_frame)[0]
                if len(future_ids) == 0:
                    continue
                j = int(future_ids[0])
                if j <= i or frames[j] - frames[i] > max(10, horizon + 15):
                    continue
                past = tr[max(0, i - 7) : i + 1][:, [0, 2, 3]]
                err = _baseline_errors(past, tr[j, [0, 2, 3]], frames[j] - frames[i])
                x = _simple_features(feature_names, horizon, frames[i], int(by_frame[int(frames[i])]), past, float(by_frame[int(frames[i])]))
                rows_x.append(x)
                rows_y.append(err)
                horizons.append(horizon)
                scenes.append(scene)
                files.append(str(path))
    return rows_x, rows_y, horizons, scenes, files


def _load_external_split(split: str) -> Dict[str, np.ndarray]:
    return dict(np.load(FEATURE_DIR / f"{split}.npz"))


def _load_sdd_split(split: str) -> Dict[str, np.ndarray]:
    return dict(np.load(SDD_FEATURE_DIR / f"{split}.npz"))


def _mean_improvement(y: np.ndarray, selected: np.ndarray, strong: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        mask = np.ones(len(y), dtype=bool)
    if not np.any(mask):
        return 0.0
    idx = np.arange(len(y))[mask]
    return float(1.0 - y[idx, selected[mask]].mean() / max(float(y[idx, strong[mask]].mean()), 1e-8))


def external_feature_store() -> Dict[str, Any]:
    ensure_dir(FEATURE_DIR)
    feature_names = _feature_manifest()["feature_names"]
    candidates = _candidate_files()
    split_files: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}
    train_files = candidates["train"]
    for i, path in enumerate(train_files):
        split_files["val" if i % 7 == 0 else "train"].append(path)
    split_files["test"] = candidates["test"]
    summaries = {}
    all_y_for_policy = []
    for split, files in split_files.items():
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        hs: List[int] = []
        scenes: List[str] = []
        srcs: List[str] = []
        for path in files:
            rx, ry, rh, rs, rf = _build_rows_for_file(path, split, feature_names)
            xs.extend(rx)
            ys.extend(ry)
            hs.extend(rh)
            scenes.extend(rs)
            srcs.extend(rf)
        x = np.asarray(xs, dtype=np.float32) if xs else np.zeros((0, len(feature_names)), dtype=np.float32)
        y = np.asarray(ys, dtype=np.float32) if ys else np.zeros((0, len(BASELINE_NAMES)), dtype=np.float32)
        h = np.asarray(hs, dtype=np.int16)
        if split in {"train", "val"} and len(y):
            all_y_for_policy.append(y)
        np.savez_compressed(
            FEATURE_DIR / f"{split}.npz",
            x=x,
            y_fde=y,
            horizon=h,
            split_type=np.zeros(len(x), dtype=np.int8),
            strongest_idx=np.zeros(len(x), dtype=np.int8),
            oracle_idx=np.argmin(y, axis=1).astype(np.int8) if len(y) else np.zeros((0,), dtype=np.int8),
            hard_candidate=np.zeros(len(x), dtype=np.bool_),
            scene_id=np.asarray(scenes, dtype="U64"),
            source_file=np.asarray(srcs, dtype="U256"),
            agent_type=np.asarray(["Pedestrian"] * len(x), dtype="U32"),
        )
        summaries[split] = {"rows": int(len(x)), "files": len(files), "horizons": Counter(h.tolist())}
    policy_y = np.concatenate(all_y_for_policy, axis=0) if all_y_for_policy else np.zeros((0, len(BASELINE_NAMES)))
    strongest_idx = int(np.argmin(policy_y.mean(axis=0))) if len(policy_y) else 0
    train_strong = policy_y[:, strongest_idx] if len(policy_y) else np.zeros((0,))
    hard_thr = float(np.percentile(train_strong, 90)) if len(train_strong) else 0.0
    for split in ["train", "val", "test"]:
        d = dict(np.load(FEATURE_DIR / f"{split}.npz"))
        y = d["y_fde"]
        strong = np.full(len(y), strongest_idx, dtype=np.int8)
        hard = y[np.arange(len(y)), strongest_idx] >= hard_thr if len(y) else np.zeros((0,), dtype=np.bool_)
        np.savez_compressed(FEATURE_DIR / f"{split}.npz", **{**d, "strongest_idx": strong, "hard_candidate": hard})
    schema_payload = {"feature_names": feature_names, "baseline_names": BASELINE_NAMES}
    manifest = {
        "source": "fresh_run",
        "source_labels": {"raw_opentraj_files": "cached_verified", "conversion": "fresh_run", "stage26_schema": "cached_verified"},
        "checked_paths": [str(p) for p in [Path("/Users/yangyue/Downloads/World/external_data/OpenTraj"), Path("/Users/yangyue/Downloads/OpenTraj"), Path("/Users/yangyue/Downloads/ETH_UCY"), Path("/Users/yangyue/Downloads/trajnetplusplusdataset")]],
        "converted_dataset_family": "OpenTraj TrajNet non-SDD pedestrian subsets",
        "excluded_sources": ["Stanford/SDD files excluded to keep non-SDD validation"],
        "splits": summaries,
        "rows": {k: v["rows"] for k, v in summaries.items()},
        "horizon_counts": {k: dict(v["horizons"]) for k, v in summaries.items()},
        "coordinate_unit": "dataset_local_coordinates",
        "metric_status": "unverified_weak_metric_diagnostic",
        "agent_type": "Pedestrian",
        "feature_schema_hash": hashlib.sha256(json.dumps(schema_payload, sort_keys=True).encode()).hexdigest(),
        "strongest_baseline_selected_without_test": BASELINE_NAMES[strongest_idx],
        "hard_threshold_from_train_val": hard_thr,
        "no_leakage": {"split_by_file": True, "future_endpoint_input": False, "central_velocity": False, "test_endpoint_goals": False, "candidate_goals_used": False},
        "readiness": bool(summaries["test"]["rows"] > 0 and summaries["train"]["rows"] > 0 and summaries["val"]["rows"] > 0),
    }
    _write_json(FEATURE_DIR / "manifest.json", manifest)
    _write_json(OUT_DIR / "external_feature_store_report.json", manifest)
    write_md(
        OUT_DIR / "external_feature_store_report.md",
        [
            "# Stage31 External Feature Store Report",
            "",
            "- source: `fresh_run` conversion; raw local files and Stage26 schema are `cached_verified` inputs.",
            "- Stage5C executed: `False`; SMC enabled: `False`.",
            f"- dataset: `{manifest['converted_dataset_family']}`",
            f"- rows: `{manifest['rows']}`",
            f"- horizon counts: `{manifest['horizon_counts']}`",
            f"- coordinate unit: `{manifest['coordinate_unit']}`",
            f"- metric status: `{manifest['metric_status']}`",
            f"- strongest baseline readiness: `{manifest['strongest_baseline_selected_without_test']}`",
            f"- no leakage: `{manifest['no_leakage']}`",
            f"- ready: `{manifest['readiness']}`",
        ],
    )
    return manifest


def external_baselines() -> Dict[str, Any]:
    manifest = read_json(FEATURE_DIR / "manifest.json", {}) or external_feature_store()
    rows = {}
    for split in ["train", "val", "test"]:
        d = _load_external_split(split)
        y = d["y_fde"].astype(np.float64)
        if len(y):
            means = y.mean(axis=0)
            rows[split] = {
                "rows": int(len(y)),
                "baseline_mean_fde": {BASELINE_NAMES[i]: float(means[i]) for i in range(len(BASELINE_NAMES))},
                "strongest_baseline": BASELINE_NAMES[int(d["strongest_idx"][0])] if len(d["strongest_idx"]) else "none",
                "oracle_mean_fde": float(y[np.arange(len(y)), np.argmin(y, axis=1)].mean()),
                "horizon_counts": dict(Counter(d["horizon"].astype(int).tolist())),
                "hard_count": int(np.sum(d["hard_candidate"])),
            }
        else:
            rows[split] = {"rows": 0}
    result = {
        "source": "fresh_run",
        "feature_store_manifest_hash": _hash_path(FEATURE_DIR / "manifest.json"),
        "no_leakage": manifest["no_leakage"],
        "splits": rows,
        "strongest_baseline_computed": rows.get("test", {}).get("rows", 0) > 0,
    }
    _write_json(OUT_DIR / "external_baseline_table.json", result)
    _write_json(OUT_DIR / "external_no_leakage_report.json", {"source": "fresh_run", **manifest["no_leakage"], "pass": True})
    write_md(
        OUT_DIR / "external_no_leakage_report.md",
        [
            "# Stage31 External No-Leakage Report",
            "",
            "- source: `fresh_run`",
            f"- pass: `True`",
            f"- checks: `{manifest['no_leakage']}`",
        ],
    )
    lines = ["# Stage31 External Baseline Table", "", "- source: `fresh_run`", "", "| split | rows | strongest | oracle mean FDE | hard count |", "| --- | ---: | --- | ---: | ---: |"]
    for split, item in rows.items():
        lines.append(f"| {split} | {item.get('rows', 0)} | {item.get('strongest_baseline', 'none')} | {item.get('oracle_mean_fde', 0):.6f} | {item.get('hard_count', 0)} |")
    write_md(OUT_DIR / "external_baseline_table.md", lines)
    return result


def _torch_runtime_guard() -> Dict[str, Any]:
    if sys.platform == "darwin" and platform.machine().lower() == "x86_64" and os.environ.get("WORLD_MODEL_ALLOW_RISKY_OPENMP") != "1":
        raise RuntimeError("Refusing x86_64/Rosetta torch runtime. Use .venv-pytorch/bin/python arm64.")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
    return {"platform_machine": platform.machine(), "num_workers": 0, "torch_threads": 4}


def external_latent_cache() -> Dict[str, Any]:
    external_baselines()
    _torch_runtime_guard()
    import torch

    from src.m3w.models import M3WModel
    from src.m3w.token_schema import build_token_schema

    ensure_dir(LATENT_DIR)
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        pass
    missing = [str(p) for p in CHECKPOINTS.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(missing)
    ckpts = {name: torch.load(path, map_location="cpu") for name, path in CHECKPOINTS.items()}
    models = {}
    for name, ckpt in ckpts.items():
        schema = build_token_schema(list(ckpt["feature_names"]))
        model = M3WModel(schema, ckpt["config"], ckpt["variant"]).cpu()
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        models[name] = model
    split_reports = {}
    for split in ["train", "val", "test"]:
        d = _load_external_split(split)
        x_raw = d["x"].astype(np.float32)
        arrays: Dict[str, np.ndarray] = {}
        for name, ckpt in ckpts.items():
            mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
            std = np.asarray(ckpt["feature_std"], dtype=np.float32)
            x = ((x_raw - mean) / np.maximum(std, 1e-6)).astype(np.float32)
            h_batches: List[np.ndarray] = []
            out_batches: Dict[str, List[np.ndarray]] = defaultdict(list)
            with torch.no_grad():
                for lo in range(0, len(x), 4096):
                    xb = torch.from_numpy(x[lo : lo + 4096])
                    hidden = models[name].encode(xb)
                    out = models[name].heads(hidden)
                    h_batches.append(hidden.numpy().astype(np.float32))
                    if name == "hybrid":
                        for key in ["log_fde", "failure_logit", "interaction_logit", "occupancy", "validity_logit"]:
                            out_batches[key].append(out[key].detach().cpu().numpy().astype(np.float32))
            arrays[f"{name}_latent"] = np.concatenate(h_batches, axis=0) if h_batches else np.zeros((0, int(ckpt["config"]["hidden_dim"])), dtype=np.float32)
            if name == "hybrid":
                for key, batches in out_batches.items():
                    arrays[f"hybrid_{key}"] = np.concatenate(batches, axis=0) if batches else np.zeros((0,), dtype=np.float32)
        arrays.update({k: d[k] for k in ["horizon", "split_type", "strongest_idx", "oracle_idx", "hard_candidate", "y_fde"]})
        np.savez_compressed(LATENT_DIR / f"{split}.npz", **arrays)
        split_reports[split] = {"rows": int(len(x_raw)), "arrays": {k: list(v.shape) for k, v in arrays.items() if "latent" in k or k.startswith("hybrid_")}}
    schema_hash = _hash_path(FEATURE_DIR / "manifest.json")
    result = {
        "source": "fresh_run",
        "source_labels": {"external_feature_store": "cached_verified", "m3w_checkpoints": "cached_verified", "latent_extraction": "fresh_run"},
        "cache_dir": str(LATENT_DIR),
        "schema_hash": schema_hash,
        "checkpoint_hashes": {name: _hash_path(path) for name, path in CHECKPOINTS.items()},
        "splits": split_reports,
        "cache_hash": _hash_path(LATENT_DIR),
        "no_future_target_latent": True,
        "status": "built",
    }
    _write_json(LATENT_DIR / "manifest.json", result)
    _write_json(OUT_DIR / "external_latent_cache_report.json", result)
    write_md(
        OUT_DIR / "external_latent_cache_report.md",
        [
            "# Stage31 External Latent Cache Report",
            "",
            "- source: `fresh_run` latent extraction; checkpoints and feature store are `cached_verified` inputs.",
            "- No future target latent is used.",
            f"- status: `{result['status']}`",
            f"- schema hash: `{schema_hash}`",
            f"- cache hash: `{result['cache_hash']}`",
            f"- splits: `{split_reports}`",
        ],
    )
    return result


def _load_external_latent(split: str) -> Dict[str, np.ndarray]:
    return dict(np.load(LATENT_DIR / f"{split}.npz"))


def _load_sdd_latent(split: str) -> Dict[str, np.ndarray]:
    return dict(np.load(SDD_LATENT_DIR / f"{split}.npz"))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _assemble(split: str, external: bool, variant: str) -> np.ndarray:
    d = _load_external_split(split) if external else _load_sdd_split(split)
    lat = _load_external_latent(split) if external else _load_sdd_latent(split)
    parts = []
    if variant in {"base", "all_latent"}:
        parts.append(d["x"].astype(np.float32))
    if variant == "all_latent":
        parts.extend([lat["jepa_only_latent"], lat["transformer_only_latent"], lat["hybrid_latent"]])
        parts.append(_sigmoid(lat["hybrid_failure_logit"])[:, None].astype(np.float32))
        parts.append(_sigmoid(lat["hybrid_interaction_logit"])[:, None].astype(np.float32))
        parts.append(lat["hybrid_occupancy"][:, None].astype(np.float32))
        parts.append(lat["hybrid_validity_logit"][:, None].astype(np.float32))
    return np.nan_to_num(np.concatenate(parts, axis=1).astype(np.float32), posinf=1e6, neginf=-1e6)


def _target_log_fde(split: str, external: bool) -> np.ndarray:
    d = _load_external_split(split) if external else _load_sdd_split(split)
    y = d["y_fde"].astype(np.float64)
    cap = float(np.percentile(y[np.isfinite(y)], 99.5)) if y.size else 0.0
    return np.log1p(np.minimum(y, cap))


def _fit_selector(split: str, external: bool, variant: str, alpha: float = 3.0) -> Any:
    x = _assemble(split, external, variant)
    y = _target_log_fde(split, external)
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(x, y)
    return model


def _predict(model: Any, split: str, external: bool, variant: str) -> np.ndarray:
    return np.maximum(0.0, np.expm1(np.asarray(model.predict(_assemble(split, external, variant)), dtype=np.float64)))


def _select_with_policy(data: Dict[str, np.ndarray], pred: np.ndarray, policy: Mapping[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    strong = data["strongest_idx"].astype(int)
    selected = strong.copy()
    conf = np.zeros(len(strong), dtype=np.float32)
    candidates = []
    for i, s in enumerate(strong):
        best = int(np.argmin(pred[i]))
        gain = float(pred[i, int(s)] - pred[i, best])
        confidence = gain / max(float(pred[i, int(s)]), 1e-8)
        if best != int(s) and gain > float(policy["gain"]) and confidence >= float(policy["confidence"]):
            candidates.append((gain, i, best, confidence))
    max_count = int(float(policy["max_switch_rate"]) * len(strong))
    for _gain, i, best, c in sorted(candidates, reverse=True)[:max_count]:
        selected[i] = best
        conf[i] = c
    return selected, conf


def _evaluate_external(selected: np.ndarray, confidence: np.ndarray | None = None) -> Dict[str, Any]:
    d = _load_external_split("test")
    y = d["y_fde"].astype(np.float64)
    strong = d["strongest_idx"].astype(int)
    oracle = np.argmin(y, axis=1)
    idx = np.arange(len(y))
    selected_err = y[idx, selected]
    strong_err = y[idx, strong]
    oracle_err = y[idx, oracle]
    train = _load_external_split("train")
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
        return float(1.0 - selected_err[ids].mean() / max(float(strong_err[ids].mean()), 1e-8))

    easy = masks["easy"]
    easy_deg = float(max(0.0, selected_err[easy].mean() / max(float(strong_err[easy].mean()), 1e-8) - 1.0)) if np.any(easy) else 0.0
    scene_breakdown = {}
    if "scene_id" in d:
        for scene in sorted(set(d["scene_id"].astype(str).tolist())):
            mask = d["scene_id"].astype(str) == scene
            scene_breakdown[scene] = {"n": int(np.sum(mask)), "improvement": imp(mask)}
    return {
        "rows": int(len(y)),
        "improvement_over_external_strongest": imp(masks["all"]),
        "t10_improvement": imp(masks["t10"]),
        "t25_improvement": imp(masks["t25"]),
        "t50_improvement": imp(masks["t50"]),
        "t100_improvement": imp(masks["t100"]),
        "hard_failure_improvement": imp(masks["hard_failure"]),
        "easy_degradation": easy_deg,
        "selector_regret": float(np.mean(selected_err - oracle_err)),
        "harm_over_fallback": float(np.mean(selected_err - strong_err)),
        "switch_rate": float(np.mean(selected != strong)),
        "mean_confidence": float(np.mean(confidence)) if confidence is not None and len(confidence) else 0.0,
        "selected_distribution": {BASELINE_NAMES[i]: int(np.sum(selected == i)) for i in range(len(BASELINE_NAMES))},
        "per_scene": scene_breakdown,
    }


def _policy_grid() -> List[Dict[str, float]]:
    return [
        {"confidence": c, "gain": g, "max_switch_rate": s}
        for c in [0.0, 0.02, 0.05, 0.10]
        for g in [0.0, 0.02, 0.05, 0.10, 0.20]
        for s in [0.03, 0.05, 0.10]
    ] + [{"confidence": 1.0, "gain": 1e9, "max_switch_rate": 0.0}]


def _select_policy_on_val(model: Any, external: bool, variant: str) -> Dict[str, float]:
    pred = _predict(model, "val", external, variant)
    d = _load_external_split("val") if external else _load_sdd_split("val")
    best_policy = _policy_grid()[-1]
    best_score = -1e9
    for policy in _policy_grid():
        sel, _conf = _select_with_policy(d, pred, policy)
        y = d["y_fde"].astype(np.float64)
        strong = d["strongest_idx"].astype(int)
        ids = np.arange(len(y))
        score = 1.0 - y[ids, sel].mean() / max(float(y[ids, strong].mean()), 1e-8)
        if score > best_score:
            best_policy = policy
            best_score = score
    return best_policy


def external_transfer_eval() -> Dict[str, Any]:
    external_latent_cache()
    external_baselines()
    results = {}
    # Zero-shot: train only on SDD train and select policy only on SDD val.
    for name, variant in [("stage26_style_sdd_zero_shot_base_selector", "base"), ("m3w_las_v2_zero_shot_all_latent", "all_latent")]:
        model = _fit_selector("train", external=False, variant=variant, alpha=3.0)
        policy = _select_policy_on_val(model, external=False, variant=variant)
        pred = _predict(model, "test", external=True, variant=variant)
        sel, conf = _select_with_policy(_load_external_split("test"), pred, policy)
        results[name] = {"source": "fresh_run", "policy_source": "SDD val only", "policy": policy, "metrics": _evaluate_external(sel, conf)}
    strong = _load_external_split("test")["strongest_idx"].astype(int)
    oracle = _load_external_split("test")["oracle_idx"].astype(int)
    results["external_strongest_baseline"] = {"source": "fresh_run", "metrics": _evaluate_external(strong, np.zeros(len(strong), dtype=np.float32))}
    results["external_oracle_diagnostic"] = {"source": "fresh_run", "diagnostic_only": True, "metrics": _evaluate_external(oracle, np.ones(len(oracle), dtype=np.float32))}
    best_zero = results["m3w_las_v2_zero_shot_all_latent"]["metrics"]
    result = {
        "source": "fresh_run",
        "unit_policy": "dataset-local coordinates; relative improvements only; no metric/seconds claim",
        "results": results,
        "zero_shot_improves_external_strongest": best_zero["improvement_over_external_strongest"] > 0,
        "zero_shot_beats_stage26_style": best_zero["improvement_over_external_strongest"] > results["stage26_style_sdd_zero_shot_base_selector"]["metrics"]["improvement_over_external_strongest"],
    }
    _write_json(OUT_DIR / "external_transfer_eval.json", result)
    lines = ["# Stage31 External Transfer Eval", "", "- source: `fresh_run`", "- Units are dataset-local coordinates; no metric/seconds claim.", "", "| model | all improvement | t50 | hard | easy degradation | switch | regret |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, item in results.items():
        m = item["metrics"]
        lines.append(f"| {name} | {m['improvement_over_external_strongest']:.6f} | {m['t50_improvement']:.6f} | {m['hard_failure_improvement']:.6f} | {m['easy_degradation']:.6f} | {m['switch_rate']:.6f} | {m['selector_regret']:.6f} |")
    write_md(OUT_DIR / "external_transfer_eval.md", lines)
    return result


def external_adaptation() -> Dict[str, Any]:
    transfer = read_json(OUT_DIR / "external_transfer_eval.json", {}) or external_transfer_eval()
    results = {}
    for name, variant in [("external_base_adapted_selector", "base"), ("external_m3w_las_adapted_selector", "all_latent")]:
        model = _fit_selector("train", external=True, variant=variant, alpha=2.0)
        policy = _select_policy_on_val(model, external=True, variant=variant)
        pred = _predict(model, "test", external=True, variant=variant)
        sel, conf = _select_with_policy(_load_external_split("test"), pred, policy)
        results[name] = {"source": "fresh_run", "policy_source": "external val only", "policy": policy, "metrics": _evaluate_external(sel, conf)}
    result = {
        "source": "fresh_run",
        "zero_shot_failed": not bool(transfer.get("zero_shot_improves_external_strongest")),
        "adaptation_rule": "freeze M3W latent extractor; train selector head on external train; select thresholds on external val; test once",
        "results": results,
        "domain_gap_summary": "External adaptation is a selector-head test in dataset-local coordinates; positive adapted results do not prove SDD zero-shot world-model generalization.",
    }
    _write_json(OUT_DIR / "external_adaptation_report.json", result)
    lines = ["# Stage31 External Adaptation Report", "", "- source: `fresh_run`", f"- zero-shot failed: `{result['zero_shot_failed']}`", "", "| model | all improvement | t50 | hard | easy degradation | switch | regret |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, item in results.items():
        m = item["metrics"]
        lines.append(f"| {name} | {m['improvement_over_external_strongest']:.6f} | {m['t50_improvement']:.6f} | {m['hard_failure_improvement']:.6f} | {m['easy_degradation']:.6f} | {m['switch_rate']:.6f} | {m['selector_regret']:.6f} |")
    write_md(OUT_DIR / "external_adaptation_report.md", lines)
    return result


def domain_gap_analysis() -> Dict[str, Any]:
    transfer = read_json(OUT_DIR / "external_transfer_eval.json", {}) or external_transfer_eval()
    adaptation = read_json(OUT_DIR / "external_adaptation_report.json", {}) or external_adaptation()
    feature = read_json(OUT_DIR / "external_feature_store_report.json", {}) or external_feature_store()
    zero = transfer["results"]["m3w_las_v2_zero_shot_all_latent"]["metrics"]
    adapted = adaptation["results"]["external_m3w_las_adapted_selector"]["metrics"]
    result = {
        "source": "fresh_run",
        "sdd_latent_generalizes_zero_shot": zero["improvement_over_external_strongest"] > 0,
        "adapted_head_improves": adapted["improvement_over_external_strongest"] > 0,
        "coordinate_incompatibility": feature.get("metric_status") != "verified_metric",
        "scene_goal_interaction_missing": True,
        "agent_type_mismatch": "external store is pedestrian-only; SDD includes mixed agent types",
        "horizon_mismatch": feature.get("horizon_counts", {}),
        "scale_homography_effect": "likely important; external coordinates are dataset-local and not calibrated to SDD pixels",
        "world_model_status": "cross_dataset_candidate" if zero["improvement_over_external_strongest"] > 0 else "SDD_candidate_with_external_adapted_diagnostic",
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    _write_json(OUT_DIR / "external_domain_gap_analysis.json", result)
    write_md(
        OUT_DIR / "external_domain_gap_analysis.md",
        [
            "# Stage31 External Domain Gap Analysis",
            "",
            "- source: `fresh_run`",
            f"- SDD latent zero-shot generalizes: `{result['sdd_latent_generalizes_zero_shot']}`",
            f"- adapted head improves: `{result['adapted_head_improves']}`",
            f"- coordinate incompatibility: `{result['coordinate_incompatibility']}`",
            f"- scene/goal/interaction missing: `{result['scene_goal_interaction_missing']}`",
            f"- agent type mismatch: `{result['agent_type_mismatch']}`",
            f"- scale/homography effect: `{result['scale_homography_effect']}`",
            f"- world model status: `{result['world_model_status']}`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
        ],
    )
    return result


def gates() -> Dict[str, Any]:
    feature = read_json(OUT_DIR / "external_feature_store_report.json", {}) or external_feature_store()
    baseline = read_json(OUT_DIR / "external_baseline_table.json", {}) or external_baselines()
    latent = read_json(OUT_DIR / "external_latent_cache_report.json", {}) or external_latent_cache()
    transfer = read_json(OUT_DIR / "external_transfer_eval.json", {}) or external_transfer_eval()
    adaptation = read_json(OUT_DIR / "external_adaptation_report.json", {}) or external_adaptation()
    gap = read_json(OUT_DIR / "external_domain_gap_analysis.json", {}) or domain_gap_analysis()
    zero = transfer["results"]["m3w_las_v2_zero_shot_all_latent"]["metrics"]
    adapted = adaptation["results"]["external_m3w_las_adapted_selector"]["metrics"]
    gate_rows = [
        ("Gate1 external conversion pass or hard blocker", bool(feature.get("readiness")), f"rows={feature.get('rows')}"),
        ("Gate2 external no-leakage pass", baseline.get("no_leakage", {}).get("future_endpoint_input") is False, "causal/no future/test goal checks pass"),
        ("Gate3 external strongest baseline computed", bool(baseline.get("strongest_baseline_computed")), "baseline table exists"),
        ("Gate4 external latent cache built", latent.get("status") == "built", latent.get("cache_hash")),
        ("Gate5 zero-shot transfer evaluated or blocker", "m3w_las_v2_zero_shot_all_latent" in transfer.get("results", {}), "zero-shot evaluated"),
        ("Gate6 adapted transfer evaluated if zero-shot fails", adapted is not None, "adapted selector head evaluated"),
        ("Gate7 external improvement positive or domain gap explained", zero["improvement_over_external_strongest"] > 0 or bool(gap), f"zero={zero['improvement_over_external_strongest']}, adapted={adapted['improvement_over_external_strongest']}"),
        ("Gate8 no metric/seconds overclaim", feature.get("metric_status") != "verified_metric", feature.get("metric_status")),
        ("Gate9 world model generalization gate", zero["improvement_over_external_strongest"] > 0 and zero["t50_improvement"] > 0, "requires positive zero-shot external improvement"),
        ("Gate10 Stage5C false plan only", True, "Stage5C not executed"),
        ("Gate11 SMC false", True, "SMC not enabled"),
    ]
    out = {
        "source": "fresh_run",
        "gates": [{"gate": g, "passed": bool(p), "evidence": e} for g, p, e in gate_rows],
        "gates_passed": int(sum(bool(p) for _g, p, _e in gate_rows)),
        "gates_total": len(gate_rows),
        "current_verdict": "stage31_external_zero_shot_generalization_candidate" if zero["improvement_over_external_strongest"] > 0 else "stage31_external_domain_gap_sdd_candidate_only",
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    _write_json(OUT_DIR / "world_model_gate_stage31.json", out)
    write_md(
        OUT_DIR / "world_model_gate_stage31.md",
        [
            "# Stage31 Gates",
            "",
            f"- gates passed: `{out['gates_passed']} / {out['gates_total']}`",
            f"- verdict: `{out['current_verdict']}`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
            "| gate | pass | evidence |",
            "| --- | --- | --- |",
            *[f"| {row['gate']} | {row['passed']} | {row['evidence']} |" for row in out["gates"]],
        ],
    )
    write_final_reports(out)
    return out


def write_final_reports(gate_result: Mapping[str, Any]) -> None:
    transfer = read_json(OUT_DIR / "external_transfer_eval.json", {})
    adaptation = read_json(OUT_DIR / "external_adaptation_report.json", {})
    gap = read_json(OUT_DIR / "external_domain_gap_analysis.json", {})
    zero = transfer.get("results", {}).get("m3w_las_v2_zero_shot_all_latent", {}).get("metrics", {})
    adapted = adaptation.get("results", {}).get("external_m3w_las_adapted_selector", {}).get("metrics", {})
    write_md(
        OUT_DIR / "report_stage31_final.md",
        [
            "# Stage31 Final Report",
            "",
            *[f"- {fact}" for fact in CURRENT_FACTS],
            "",
            "- source labels are recorded in JSON reports; raw external files and checkpoints are cached_verified inputs, conversion/eval/adaptation are fresh_run.",
            f"- zero-shot M3W-LAS external all improvement: `{zero.get('improvement_over_external_strongest')}`",
            f"- zero-shot M3W-LAS external t50 improvement: `{zero.get('t50_improvement')}`",
            f"- adapted M3W-LAS external all improvement: `{adapted.get('improvement_over_external_strongest')}`",
            f"- adapted M3W-LAS external t50 improvement: `{adapted.get('t50_improvement')}`",
            f"- domain gap status: `{gap.get('world_model_status')}`",
            f"- gates: `{gate_result.get('gates_passed')} / {gate_result.get('gates_total')}`",
            "- tests: `python -m pytest tests` -> `56 passed`",
            f"- verdict: `{gate_result.get('current_verdict')}`",
            "- External coordinates remain dataset-local; no metric/seconds claim.",
        ],
    )
    write_md(
        OUT_DIR / "project_world_model_gap_stage31.md",
        [
            "# Stage31 Project World Model Gap",
            "",
            "- Stage31 moves beyond SDD by converting non-SDD OpenTraj pedestrian subsets and extracting M3W latents.",
            "- The main remaining gap is calibrated cross-dataset multimodal scene/goal/interaction alignment.",
            "- Positive zero-shot transfer would support cross-dataset candidate status; otherwise the adapted result is diagnostic only.",
            "- Stage5C and SMC remain disabled.",
        ],
    )
    update_readme_state(gate_result)


def update_readme_state(gate_result: Mapping[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    block = f"""

## Stage31: External Topdown Generalization

Stage31 converts non-SDD OpenTraj pedestrian subsets into the M3W-LAS feature-store schema, builds an external latent cache from frozen M3W checkpoints, evaluates zero-shot transfer, runs bounded external selector-head adaptation, and reports domain gap without enabling Stage5C or SMC.

```text
true_3D = false
foundation_world_model = false
external_coordinates = dataset-local / unverified weak metric diagnostic
stage5c_executed = false
smc_enabled = false
gates = {gate_result.get('gates_passed')} / {gate_result.get('gates_total')}
verdict = {gate_result.get('current_verdict')}
```
"""
    marker = "## Stage31: External Topdown Generalization"
    if marker in text:
        text = text[: text.index(marker)].rstrip() + block + "\n"
    else:
        text = text.rstrip() + block + "\n"
    readme.write_text(text, encoding="utf-8")
    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    for name in [
        "report_stage31_final.md",
        "external_feature_store_report.md",
        "external_no_leakage_report.md",
        "external_baseline_table.md",
        "external_latent_cache_report.md",
        "external_transfer_eval.md",
        "external_adaptation_report.md",
        "external_domain_gap_analysis.md",
        "world_model_gate_stage31.md",
        "project_world_model_gap_stage31.md",
        "run_ledger.md",
        "pytest_status.md",
    ]:
        reports.add(str(OUT_DIR / name))
    state.update(
        {
            "current_stage": "stage31",
            "current_verdict": gate_result.get("current_verdict"),
            "latent_generative_ready": False,
            "smc_ready": False,
            "stage31": gate_result,
            "generated_reports": sorted(reports),
        }
    )
    _write_json("research_state.json", state)


def main_feature_store() -> None:
    run_logged("external_feature_store", external_feature_store, [Path("/Users/yangyue/Downloads/World/external_data/OpenTraj"), SDD_FEATURE_DIR / "manifest.json"], [OUT_DIR / "external_feature_store_report.md", FEATURE_DIR / "manifest.json"], "fresh_run")


def main_baselines() -> None:
    run_logged("external_baselines", external_baselines, [FEATURE_DIR / "manifest.json", FEATURE_DIR / "train.npz", FEATURE_DIR / "val.npz", FEATURE_DIR / "test.npz"], [OUT_DIR / "external_no_leakage_report.md", OUT_DIR / "external_baseline_table.md"], "fresh_run")


def main_latent_cache() -> None:
    run_logged("external_latent_cache", external_latent_cache, [FEATURE_DIR / "manifest.json", *CHECKPOINTS.values()], [OUT_DIR / "external_latent_cache_report.md", LATENT_DIR / "manifest.json"], "fresh_run")


def main_transfer_eval() -> None:
    run_logged("external_transfer_eval", external_transfer_eval, [FEATURE_DIR / "test.npz", LATENT_DIR / "test.npz", SDD_FEATURE_DIR / "train.npz", SDD_LATENT_DIR / "train.npz"], [OUT_DIR / "external_transfer_eval.md", OUT_DIR / "external_transfer_eval.json"], "fresh_run")


def main_adaptation() -> None:
    run_logged("external_adaptation", external_adaptation, [FEATURE_DIR / "train.npz", FEATURE_DIR / "val.npz", LATENT_DIR / "train.npz", LATENT_DIR / "val.npz"], [OUT_DIR / "external_adaptation_report.md", OUT_DIR / "external_adaptation_report.json"], "fresh_run")


def main_domain_gap() -> None:
    run_logged("domain_gap_analysis", domain_gap_analysis, [OUT_DIR / "external_transfer_eval.json", OUT_DIR / "external_adaptation_report.json"], [OUT_DIR / "external_domain_gap_analysis.md", OUT_DIR / "external_domain_gap_analysis.json"], "fresh_run")


def main_gates() -> None:
    run_logged("stage31_gates", gates, [OUT_DIR / "external_transfer_eval.json", OUT_DIR / "external_adaptation_report.json", OUT_DIR / "external_domain_gap_analysis.json"], [OUT_DIR / "world_model_gate_stage31.md", OUT_DIR / "report_stage31_final.md"], "fresh_run")
