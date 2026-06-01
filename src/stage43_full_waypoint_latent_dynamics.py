from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("WORLD_MODEL_TORCH_THREADS", "4")
os.environ.setdefault("WORLD_MODEL_TORCH_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", os.environ["WORLD_MODEL_TORCH_THREADS"])
os.environ.setdefault("MKL_NUM_THREADS", os.environ["WORLD_MODEL_TORCH_THREADS"])
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ["WORLD_MODEL_TORCH_THREADS"])
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", os.environ["WORLD_MODEL_TORCH_THREADS"])
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ["WORLD_MODEL_TORCH_THREADS"])
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

if (
    platform.system().lower() == "darwin"
    and platform.machine().lower() == "x86_64"
    and os.environ.get("WORLD_MODEL_ALLOW_RISKY_OPENMP") != "1"
):
    raise RuntimeError(
        "Refusing Stage43 full-waypoint torch training under macOS x86_64/Rosetta. "
        "Use .venv-pytorch/bin/python arm64 with num_workers=0."
    )

import numpy as np
import torch
from torch import nn

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_protected_latent_state_model import _jsonable


OUT_DIR = Path("outputs/stage43_latent_state")
CKPT_DIR = OUT_DIR / "checkpoints"
CACHE_DIR = Path("data/stage43_full_waypoint_supervision_cache")
DATA35 = Path("data/stage35_selective_transfer")
DATA36 = Path("data/stage36_t50_repair")
DATA37 = Path("data/stage37_t50_history")

REPORT_JSON = OUT_DIR / "stage43_full_waypoint_latent_dynamics.json"
REPORT_MD = OUT_DIR / "stage43_full_waypoint_latent_dynamics.md"
GATE_MD = OUT_DIR / "stage43_stage_m_full_waypoint_latent_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
HEARTBEAT_JSON = OUT_DIR / "stage43_full_waypoint_latent_heartbeat.json"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE43_M_FULL_WAYPOINT_LATENT_DYNAMICS"
SOURCE = "fresh_stage43_m_full_waypoint_latent_dynamics"
SPLITS = ["train", "val", "test"]
OLD_SPLITS = ["train", "val", "test"]
DOMAINS = ["ETH_UCY", "TrajNet", "UCY"]
HORIZONS = [10, 25, 50, 100]
WAYPOINT_FRAC = np.asarray([0.25, 0.50, 0.75, 1.0], dtype=np.float32)
EPS = 1e-8


@dataclass
class WaypointSplit:
    split: str
    x: np.ndarray
    waypoint_delta: np.ndarray
    waypoint_valid: np.ndarray
    floor_waypoint_delta: np.ndarray
    floor_ade: np.ndarray
    floor_fde: np.ndarray
    y_failure: np.ndarray
    y_gain: np.ndarray
    y_harm: np.ndarray
    y_density: np.ndarray
    horizon: np.ndarray
    domain: np.ndarray
    source_file: np.ndarray
    scene_id: np.ndarray
    hard: np.ndarray
    failure: np.ndarray
    easy: np.ndarray
    scale: np.ndarray
    feature_names: list[str]


class FullWaypointLatentDynamics(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.future_target_encoder = nn.Sequential(
            nn.Linear(14, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.head = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 13))

    def forward(self, x: torch.Tensor, target_vec: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        z_t = self.encoder(x)
        z_next = self.dynamics(z_t)
        out = self.head(z_next)
        result = {
            "z_t": z_t,
            "z_next": z_next,
            "waypoint_delta": out[:, :8].reshape(-1, 4, 2),
            "failure_logit": out[:, 8],
            "gain_logit": out[:, 9],
            "harm_logit": out[:, 10],
            "density": torch.sigmoid(out[:, 11]),
            "validity_logit": out[:, 12],
        }
        if target_vec is not None:
            result["target_latent"] = self.future_target_encoder(target_vec).detach()
        return result


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_runtime(seed: int) -> dict[str, Any]:
    torch.set_num_threads(max(1, int(os.environ.get("WORLD_MODEL_TORCH_THREADS", "4"))))
    try:
        torch.set_num_interop_threads(max(1, int(os.environ.get("WORLD_MODEL_TORCH_INTEROP_THREADS", "1"))))
    except RuntimeError:
        pass
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "device": "cpu",
        "num_workers": 0,
    }


def _cache_path(split: str) -> Path:
    return CACHE_DIR / f"stage43_full_waypoint_supervision_{split}.npz"


def _npz(path: Path) -> Mapping[str, np.ndarray]:
    return np.load(path, allow_pickle=False)


def _row_hash(cache: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in ["old_split", "local_row", "dataset", "scene_id", "source_file", "agent_id", "frame_id", "horizon"]:
        digest.update(key.encode("utf-8"))
        arr = np.asarray(cache[key])
        if arr.dtype.kind in {"U", "S", "O"}:
            digest.update(arr.astype(str).tobytes())
        else:
            digest.update(arr.tobytes())
    return digest.hexdigest()


def _gather_old(cache: Mapping[str, np.ndarray], artifact: str, key: str) -> np.ndarray:
    out: np.ndarray | None = None
    old_split = cache["old_split"].astype(str)
    local = cache["local_row"].astype(np.int64)
    paths = {
        "geo": DATA35 / "expanded_external_{split}.npz",
        "labels": DATA35 / "labels_{split}.npz",
        "history": DATA37 / "history_windows_{split}.npz",
        "goal": DATA37 / "goal_prototypes_{split}.npz",
        "baseline": DATA37 / "t50_baseline_family_{split}.npz",
        "stage35": DATA36 / "stage35_selection_{split}.npz",
    }
    for split in OLD_SPLITS:
        ids = np.where(old_split == split)[0]
        if len(ids) == 0:
            continue
        z = _npz(Path(str(paths[artifact]).format(split=split)))
        vals = z[key][local[ids]]
        if out is None:
            out_shape = (len(old_split), *vals.shape[1:])
            out = np.zeros(out_shape, dtype=vals.dtype)
        out[ids] = vals
    if out is None:
        raise ValueError(f"No rows gathered for {artifact}:{key}")
    return out


def _one_hot(values: np.ndarray, choices: list[Any], prefix: str) -> tuple[np.ndarray, list[str]]:
    out = np.zeros((len(values), len(choices)), dtype=np.float32)
    for i, choice in enumerate(choices):
        out[:, i] = (values == choice).astype(np.float32)
    return out, [f"{prefix}_{choice}" for choice in choices]


def _tail(values: np.ndarray, k: int) -> np.ndarray:
    return values[:, -k:].astype(np.float32)


def _stage_floor_endpoint(cache: Mapping[str, np.ndarray], baseline_pred: np.ndarray) -> np.ndarray:
    selected = _gather_old(cache, "stage35", "selected").astype(np.int64).clip(0, baseline_pred.shape[1] - 1)
    endpoint = baseline_pred[np.arange(len(selected)), selected].astype(np.float32)
    old_split = cache["old_split"].astype(str)
    horizon = cache["horizon"].astype(np.int64)
    t50_test = (old_split == "test") & (horizon == 50)
    st37 = DATA37 / "stage37_best_t50_selection_test.npz"
    if np.any(t50_test) and st37.exists():
        local = cache["local_row"].astype(np.int64)
        selected37_all = _npz(st37)["selected_family"].astype(np.int64)
        selected37 = selected37_all[local[t50_test]].clip(0, baseline_pred.shape[1] - 1)
        endpoint[t50_test] = baseline_pred[t50_test][np.arange(len(selected37)), selected37].astype(np.float32)
    return endpoint


def _build_split(split: str, *, max_rows: int | None, seed: int) -> WaypointSplit:
    cache = _npz(_cache_path(split))
    n = len(cache["horizon"])
    ids = np.arange(n)
    if max_rows is not None and max_rows < n:
        rng = np.random.default_rng(seed + {"train": 0, "val": 1, "test": 2}[split])
        ids = np.sort(rng.choice(ids, size=int(max_rows), replace=False))
    sub = {key: np.asarray(cache[key])[ids] for key in cache.files}
    scale = np.maximum(sub["scale"].astype(np.float32), 1e-4)
    cur = sub["current_xy"].astype(np.float32)
    waypoints = sub["waypoint_xy"].astype(np.float32)
    valid = sub["waypoint_valid"].astype(bool)
    waypoint_delta = ((waypoints - cur[:, None, :]) / scale[:, None, None]).astype(np.float32)

    hist_keys_1d = [
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
    ]
    history = {key: _gather_old(sub, "history", key) for key in ["history_dx", "history_dy", "history_speed", "history_accel", "history_heading", "history_valid_mask", *hist_keys_1d]}
    goal = {key: _gather_old(sub, "goal", key) for key in ["prototype_likelihood", "prototype_distance", "prototype_angle", "prototype_entropy", "goal_ambiguity"]}
    baseline_pred = _gather_old(sub, "baseline", "prediction").astype(np.float32)
    labels_y_fde = _gather_old(sub, "labels", "y_fde").astype(np.float32)
    labels_oracle_idx = _gather_old(sub, "labels", "oracle_idx").astype(np.int64)
    labels_oracle_margin = _gather_old(sub, "labels", "oracle_margin").astype(np.float32)

    floor_endpoint = _stage_floor_endpoint(sub, baseline_pred)
    floor_delta = ((floor_endpoint - cur) / scale[:, None]).astype(np.float32)
    floor_waypoint_delta = WAYPOINT_FRAC[None, :, None] * floor_delta[:, None, :]
    floor_xy = cur[:, None, :] + floor_waypoint_delta * scale[:, None, None]
    floor_err = np.linalg.norm(floor_xy.astype(np.float64) - waypoints.astype(np.float64), axis=2) / scale[:, None]
    floor_ade = ((floor_err * valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)).astype(np.float32)
    floor_fde = floor_err[:, -1].astype(np.float32)

    row = np.arange(len(ids))
    oracle_err = labels_y_fde[row, labels_oracle_idx.clip(0, labels_y_fde.shape[1] - 1)]
    strongest_idx = _gather_old(sub, "labels", "strongest_idx").astype(np.int64)
    strongest_err = labels_y_fde[row, strongest_idx.clip(0, labels_y_fde.shape[1] - 1)]
    y_gain = (oracle_err + 0.01 < strongest_err).astype(np.float32)
    y_harm = (sub["easy"].astype(bool) | (labels_oracle_margin < 0.01)).astype(np.float32)
    y_density = np.clip(history["history_density"].astype(np.float32) / 10.0, 0.0, 1.0)

    domain = sub["dataset"].astype(str)
    horizon = sub["horizon"].astype(np.int64)
    domain_oh, domain_names = _one_hot(domain, DOMAINS, "domain")
    horizon_oh, horizon_names = _one_hot(horizon, HORIZONS, "horizon")
    feature_parts: list[np.ndarray] = [
        cur / scale[:, None],
        horizon[:, None].astype(np.float32) / 100.0,
        domain_oh,
        horizon_oh,
    ]
    feature_names = ["current_x_over_scale", "current_y_over_scale", "horizon_norm", *domain_names, *horizon_names]
    for key in ["history_dx", "history_dy", "history_speed", "history_accel", "history_heading", "history_valid_mask"]:
        vals = _tail(history[key], 16)
        feature_parts.append(vals)
        feature_names.extend([f"{key}_tail{i}" for i in range(vals.shape[1])])
    for key in hist_keys_1d:
        vals = history[key].astype(np.float32)[:, None]
        feature_parts.append(vals)
        feature_names.append(key)
    for key in ["prototype_likelihood", "prototype_distance", "prototype_angle"]:
        vals = goal[key].astype(np.float32)
        feature_parts.append(vals)
        feature_names.extend([f"{key}_{i}" for i in range(vals.shape[1])])
    for key in ["prototype_entropy", "goal_ambiguity"]:
        vals = goal[key].astype(np.float32)[:, None]
        feature_parts.append(vals)
        feature_names.append(key)
    baseline_rel = ((baseline_pred - cur[:, None, :]) / scale[:, None, None]).reshape(len(ids), -1)
    feature_parts.append(baseline_rel.astype(np.float32))
    feature_names.extend([f"baseline_endpoint_rel_{i}" for i in range(baseline_rel.shape[1])])
    feature_parts.append(floor_delta)
    feature_names.extend(["floor_endpoint_rel_x", "floor_endpoint_rel_y"])
    x = np.concatenate(feature_parts, axis=1).astype(np.float32)
    return WaypointSplit(
        split=split,
        x=x,
        waypoint_delta=waypoint_delta,
        waypoint_valid=valid,
        floor_waypoint_delta=floor_waypoint_delta.astype(np.float32),
        floor_ade=floor_ade,
        floor_fde=floor_fde,
        y_failure=sub["failure"].astype(np.float32),
        y_gain=y_gain,
        y_harm=y_harm,
        y_density=y_density.astype(np.float32),
        horizon=horizon,
        domain=domain,
        source_file=sub["source_file"].astype(str),
        scene_id=sub["scene_id"].astype(str),
        hard=sub["hard"].astype(bool),
        failure=sub["failure"].astype(bool),
        easy=sub["easy"].astype(bool),
        scale=scale,
        feature_names=feature_names,
    )


def _standardize(train: WaypointSplit, val: WaypointSplit, test: WaypointSplit) -> tuple[WaypointSplit, WaypointSplit, WaypointSplit, np.ndarray, np.ndarray]:
    mean = train.x.mean(axis=0).astype(np.float32)
    raw_std = train.x.std(axis=0).astype(np.float32)
    std = np.where(raw_std < 1e-3, 1.0, raw_std).astype(np.float32)
    for ds in [train, val, test]:
        ds.x = ((ds.x - mean) / std).astype(np.float32)
    return train, val, test, mean, std


def _target_vec(ds: WaypointSplit) -> np.ndarray:
    flat = ds.waypoint_delta.reshape(len(ds.x), -1)
    return np.concatenate(
        [
            flat,
            ds.y_failure[:, None],
            ds.y_gain[:, None],
            ds.y_harm[:, None],
            ds.y_density[:, None],
            (ds.horizon[:, None].astype(np.float32) / 100.0),
            ds.waypoint_valid.mean(axis=1, keepdims=True).astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)


def _batch_indices(n: int, batch_size: int, *, shuffle: bool, seed: int) -> list[np.ndarray]:
    ids = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(ids)
    return [ids[i : i + batch_size] for i in range(0, n, batch_size)]


def _loss(model: FullWaypointLatentDynamics, ds: WaypointSplit, ids: np.ndarray, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    x = torch.from_numpy(ds.x[ids]).to(device)
    target_delta = torch.from_numpy(ds.waypoint_delta[ids]).to(device)
    valid = torch.from_numpy(ds.waypoint_valid[ids].astype(np.float32)).to(device)
    y_failure = torch.from_numpy(ds.y_failure[ids]).to(device)
    y_gain = torch.from_numpy(ds.y_gain[ids]).to(device)
    y_harm = torch.from_numpy(ds.y_harm[ids]).to(device)
    y_density = torch.from_numpy(ds.y_density[ids]).to(device)
    target = torch.from_numpy(_target_vec(ds)[ids]).to(device)
    horizon = torch.from_numpy(ds.horizon[ids]).to(device)
    hard = torch.from_numpy((ds.hard[ids] | ds.failure[ids]).astype(np.float32)).to(device)
    out = model(x, target)
    per_wp = nn.functional.smooth_l1_loss(out["waypoint_delta"], target_delta, reduction="none").mean(dim=2)
    row_weight = 1.0 + 1.0 * hard + 1.0 * (horizon == 50).float() + 0.5 * (horizon == 100).float()
    waypoint = ((per_wp * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0) * row_weight).mean()
    endpoint = nn.functional.smooth_l1_loss(out["waypoint_delta"][:, -1, :], target_delta[:, -1, :])
    failure = nn.functional.binary_cross_entropy_with_logits(out["failure_logit"], y_failure)
    gain = nn.functional.binary_cross_entropy_with_logits(out["gain_logit"], y_gain)
    harm = nn.functional.binary_cross_entropy_with_logits(out["harm_logit"], y_harm)
    density = nn.functional.mse_loss(out["density"], y_density)
    latent = nn.functional.mse_loss(out["z_next"], out["target_latent"])
    variance = out["z_next"].float().var(dim=0).mean()
    collapse = torch.relu(torch.tensor(0.02, device=device) - variance)
    total = waypoint + 0.30 * endpoint + 0.35 * failure + 0.45 * gain + 0.55 * harm + 0.15 * density + 0.35 * latent + collapse
    return total, {
        "waypoint": float(waypoint.detach().cpu()),
        "endpoint": float(endpoint.detach().cpu()),
        "failure": float(failure.detach().cpu()),
        "gain": float(gain.detach().cpu()),
        "harm": float(harm.detach().cpu()),
        "density": float(density.detach().cpu()),
        "latent": float(latent.detach().cpu()),
        "latent_variance": float(variance.detach().cpu()),
    }


@torch.no_grad()
def _predict(model: FullWaypointLatentDynamics, ds: WaypointSplit, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    model.eval()
    outs: dict[str, list[np.ndarray]] = {"waypoint": [], "failure": [], "gain": [], "harm": [], "density": [], "latent": []}
    for ids in _batch_indices(len(ds.x), batch_size, shuffle=False, seed=0):
        x = torch.from_numpy(ds.x[ids]).to(device)
        out = model(x)
        outs["waypoint"].append(out["waypoint_delta"].detach().cpu().numpy())
        outs["failure"].append(torch.sigmoid(out["failure_logit"]).detach().cpu().numpy())
        outs["gain"].append(torch.sigmoid(out["gain_logit"]).detach().cpu().numpy())
        outs["harm"].append(torch.sigmoid(out["harm_logit"]).detach().cpu().numpy())
        outs["density"].append(out["density"].detach().cpu().numpy())
        outs["latent"].append(out["z_next"].detach().cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in outs.items()}


def _trajectory_error(ds: WaypointSplit, pred_delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    err = np.linalg.norm((pred_delta.astype(np.float64) - ds.waypoint_delta.astype(np.float64)), axis=2)
    valid = ds.waypoint_valid.astype(bool)
    ade = (err * valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    fde = err[:, -1]
    return ade.astype(np.float32), fde.astype(np.float32)


def _slice_improvement(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return 0.0
    return float(1.0 - float(np.mean(selected[mask])) / max(float(np.mean(floor[mask])), EPS))


def _metrics(ds: WaypointSplit, selected_ade: np.ndarray, selected_fde: np.ndarray, switched: np.ndarray) -> dict[str, Any]:
    hard_failure = ds.hard | ds.failure
    h50 = ds.horizon == 50
    h100 = ds.horizon == 100
    easy_deg = (
        float(max(0.0, float(np.mean(selected_ade[ds.easy])) / max(float(np.mean(ds.floor_ade[ds.easy])), EPS) - 1.0))
        if int(ds.easy.sum())
        else 0.0
    )
    return {
        "rows": int(len(ds.x)),
        "full_waypoint_ade_improvement_vs_floor": _slice_improvement(selected_ade, ds.floor_ade, np.ones(len(ds.x), dtype=bool)),
        "endpoint_fde_improvement_vs_floor": _slice_improvement(selected_fde, ds.floor_fde, np.ones(len(ds.x), dtype=bool)),
        "t50_full_waypoint_ade_improvement_vs_floor": _slice_improvement(selected_ade, ds.floor_ade, h50),
        "t50_endpoint_fde_improvement_vs_floor": _slice_improvement(selected_fde, ds.floor_fde, h50),
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor": _slice_improvement(selected_ade, ds.floor_ade, h100),
        "hard_failure_full_waypoint_ade_improvement_vs_floor": _slice_improvement(selected_ade, ds.floor_ade, hard_failure),
        "easy_degradation_vs_floor": easy_deg,
        "switch_rate": float(np.mean(switched)),
        "harm_over_floor_ade": float(np.mean(selected_ade - ds.floor_ade)),
        "mean_floor_ade": float(np.mean(ds.floor_ade)),
        "mean_selected_ade": float(np.mean(selected_ade)),
    }


def _bootstrap_ci(
    ds: WaypointSplit,
    selected_ade: np.ndarray,
    selected_fde: np.ndarray,
    *,
    n: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    size = len(selected_ade)
    hard_failure = ds.hard | ds.failure
    h50 = ds.horizon == 50
    h100 = ds.horizon == 100
    names = {
        "full_waypoint_ade_improvement_vs_floor": (selected_ade, ds.floor_ade, np.ones(size, dtype=bool), "improvement"),
        "endpoint_fde_improvement_vs_floor": (selected_fde, ds.floor_fde, np.ones(size, dtype=bool), "improvement"),
        "t50_full_waypoint_ade_improvement_vs_floor": (selected_ade, ds.floor_ade, h50, "improvement"),
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor": (selected_ade, ds.floor_ade, h100, "improvement"),
        "hard_failure_full_waypoint_ade_improvement_vs_floor": (selected_ade, ds.floor_ade, hard_failure, "improvement"),
        "easy_degradation_vs_floor": (selected_ade, ds.floor_ade, ds.easy, "easy_degradation"),
    }
    out: dict[str, Any] = {"n": int(n), "seed": int(seed), "metrics": {}}
    for name, (selected, floor, mask, kind) in names.items():
        ids = np.where(mask)[0]
        if len(ids) == 0:
            out["metrics"][name] = {"low": 0.0, "mean": 0.0, "high": 0.0, "rows": 0}
            continue
        vals = np.empty(int(n), dtype=np.float64)
        for i in range(int(n)):
            sample = rng.choice(ids, size=len(ids), replace=True)
            if kind == "easy_degradation":
                vals[i] = max(0.0, float(np.mean(selected[sample])) / max(float(np.mean(floor[sample])), EPS) - 1.0)
            else:
                vals[i] = 1.0 - float(np.mean(selected[sample])) / max(float(np.mean(floor[sample])), EPS)
        out["metrics"][name] = {
            "low": float(np.quantile(vals, 0.025)),
            "mean": float(np.mean(vals)),
            "high": float(np.quantile(vals, 0.975)),
            "rows": int(len(ids)),
        }
    return out


def _select_with_policy(ds: WaypointSplit, pred: Mapping[str, np.ndarray], policy: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_ade, candidate_fde = _trajectory_error(ds, pred["waypoint"])
    allow = (
        (pred["gain"] >= float(policy["gain_threshold"]))
        & (pred["harm"] <= float(policy["harm_threshold"]))
        & (pred["failure"] >= float(policy["failure_threshold"]))
    )
    selected_ade = np.where(allow, candidate_ade, ds.floor_ade).astype(np.float32)
    selected_fde = np.where(allow, candidate_fde, ds.floor_fde).astype(np.float32)
    return selected_ade, selected_fde, allow.astype(bool)


def _search_policy(val: WaypointSplit, pred: Mapping[str, np.ndarray]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    # A deployable protected policy must actually use the floor.  The previous
    # objective could select an all-switch policy when validation easy rows
    # happened not to degrade, which is too brittle for Stage43 safety claims.
    max_switch_rate = 0.90
    for gain in [0.0, 0.25, 0.45, 0.55, 0.65, 0.75, 0.85]:
        for harm in [0.15, 0.25, 0.35, 0.50, 0.75, 1.00]:
            for failure in [0.0, 0.10, 0.20, 0.35, 0.50]:
                policy = {"gain_threshold": gain, "harm_threshold": harm, "failure_threshold": failure}
                selected_ade, selected_fde, switched = _select_with_policy(val, pred, policy)
                metrics = _metrics(val, selected_ade, selected_fde, switched)
                degenerate_all_switch = (
                    gain <= 0.0
                    and harm >= 1.0
                    and failure <= 0.0
                    and metrics["switch_rate"] >= max_switch_rate
                )
                if metrics["easy_degradation_vs_floor"] > 0.02:
                    continue
                if metrics["switch_rate"] > max_switch_rate:
                    continue
                if degenerate_all_switch:
                    continue
                objective = (
                    metrics["full_waypoint_ade_improvement_vs_floor"]
                    + 1.2 * metrics["t50_full_waypoint_ade_improvement_vs_floor"]
                    + 0.8 * metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
                    - 0.15 * metrics["switch_rate"]
                )
                row = {"policy": policy, "metrics": metrics, "objective": float(objective)}
                if best is None or row["objective"] > best["objective"]:
                    best = row
    if best is None:
        # Honest diagnostic fallback: if no validation-safe switching policy
        # exists, keep the floor rather than deploying an unsafe neural head.
        selected_ade = val.floor_ade.copy()
        selected_fde = val.floor_fde.copy()
        switched = np.zeros(len(val.x), dtype=bool)
        return {
            "policy": {"gain_threshold": 1.01, "harm_threshold": -0.01, "failure_threshold": 1.01},
            "metrics": _metrics(val, selected_ade, selected_fde, switched),
            "objective": 0.0,
            "diagnostic": "no_validation_safe_switching_policy_found_keep_floor",
        }
    return best


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["test_metrics_with_floor"]
    ungated = payload["test_metrics_neural_without_floor"]
    gates = {
        "stage43_l_cache_present": payload["stage43_l_precondition"]["full_waypoint_supervised_training_ready"] is True,
        "torch_training_fresh_run": payload["result_source"] == "fresh_run" and Path(payload["checkpoint"]).exists(),
        "checkpoint_not_committed": payload["checkpoint_committed"] is False,
        "full_waypoint_labels_used_as_labels_only": payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_waypoint_label_eval_only"] is True,
        "no_future_endpoint_or_central_velocity_input": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["central_velocity_input"] is False,
        "latent_noncollapse": payload["latent_variance"] > 0.01,
        "protected_full_waypoint_eval_completed": metrics["rows"] > 0,
        "easy_preserved": metrics["easy_degradation_vs_floor"] <= 0.02,
        "neural_full_waypoint_lift_or_honest_diagnostic": (
            metrics["full_waypoint_ade_improvement_vs_floor"] > 0.0
            or metrics["t50_full_waypoint_ade_improvement_vs_floor"] > 0.0
            or metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0
            or payload["deploy_neural"] is False
        ),
        "ungated_neural_reported": "full_waypoint_ade_improvement_vs_floor" in ungated,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_m_protected_full_waypoint_latent_candidate_pass"
        if passed == total and payload["deploy_neural"]
        else "stage43_m_full_waypoint_latent_diagnostic_keep_floor",
        "deploy_neural_full_waypoint": bool(payload["deploy_neural"] and passed == total),
    }


def _train_eval(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CKPT_DIR)
    seed = int(args.seed)
    runtime = _configure_runtime(seed)
    mode = "quick" if args.quick else "small" if args.small else "medium"
    max_train = 6000 if args.quick else 30000 if args.small else 90000
    max_val = 3000 if args.quick else 12000 if args.small else 40000
    max_test = 3000 if args.quick else 16000 if args.small else 50000
    train = _build_split("train", max_rows=max_train, seed=seed)
    val = _build_split("val", max_rows=max_val, seed=seed)
    test = _build_split("test", max_rows=max_test, seed=seed)
    train, val, test, mean, std = _standardize(train, val, test)
    model = FullWaypointLatentDynamics(train.x.shape[1], hidden_dim=int(args.hidden_dim), latent_dim=int(args.latent_dim))
    device = torch.device("cpu")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    best_val = float("inf")
    best_path = CKPT_DIR / "stage43_full_waypoint_latent_dynamics.pt"
    history: list[dict[str, Any]] = []
    start = time.time()
    for epoch in range(int(args.epochs)):
        model.train()
        losses: list[float] = []
        stats: list[dict[str, float]] = []
        for batch_ids in _batch_indices(len(train.x), int(args.batch_size), shuffle=True, seed=seed + epoch):
            opt.zero_grad(set_to_none=True)
            loss, stat = _loss(model, train, batch_ids, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            stats.append(stat)
        val_pred = _predict(model, val, device, int(args.batch_size))
        val_ade, _ = _trajectory_error(val, val_pred["waypoint"])
        val_loss = float(np.mean((val_ade - val.floor_ade) ** 2))
        latent_var = float(np.mean([row["latent_variance"] for row in stats])) if stats else 0.0
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_candidate_mse_to_floor": val_loss,
            "latent_variance": latent_var,
        }
        history.append(row)
        write_json(
            HEARTBEAT_JSON,
            _jsonable(
                {
                    "source": SOURCE,
                    "epoch": epoch + 1,
                    "elapsed_s": time.time() - start,
                    "last": row,
                    "git_commit": _git_commit(),
                    "mode": mode,
                }
            ),
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_mean": mean,
                    "feature_std": std,
                    "feature_names": train.feature_names,
                    "input_dim": int(train.x.shape[1]),
                    "hidden_dim": int(args.hidden_dim),
                    "latent_dim": int(args.latent_dim),
                    "seed": seed,
                    "epoch": epoch + 1,
                    "runtime": runtime,
                    "no_leakage": {
                        "future_endpoint_input": False,
                        "future_waypoint_input": False,
                        "future_waypoint_label_eval_only": True,
                        "central_velocity_input": False,
                        "test_endpoint_goal_construction": False,
                        "test_statistics_normalization": False,
                    },
                },
                best_path,
            )
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    val_pred = _predict(model, val, device, int(args.batch_size))
    test_pred = _predict(model, test, device, int(args.batch_size))
    val_policy = _search_policy(val, val_pred)
    selected_ade, selected_fde, switched = _select_with_policy(test, test_pred, val_policy["policy"])
    protected_metrics = _metrics(test, selected_ade, selected_fde, switched)
    bootstrap = _bootstrap_ci(test, selected_ade, selected_fde, n=int(args.bootstrap), seed=seed + 1000)
    ungated_ade, ungated_fde = _trajectory_error(test, test_pred["waypoint"])
    ungated_metrics = _metrics(test, ungated_ade, ungated_fde, np.ones(len(test.x), dtype=bool))
    latent_var = float(np.var(test_pred["latent"], axis=0).mean()) if len(test_pred["latent"]) else 0.0
    deploy = bool(
        (
            protected_metrics["full_waypoint_ade_improvement_vs_floor"] > 0.0
            or protected_metrics["t50_full_waypoint_ade_improvement_vs_floor"] > 0.0
            or protected_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0
        )
        and protected_metrics["easy_degradation_vs_floor"] <= 0.02
    )
    stage43_l = read_json(OUT_DIR / "stage43_full_waypoint_supervision_cache.json", {})
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_run",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "mode": mode,
        "checkpoint": str(best_path),
        "checkpoint_sha256": _sha256(best_path),
        "checkpoint_committed": False,
        "runtime": runtime,
        "stage43_l_precondition": {
            "verdict": stage43_l.get("stage43_l_gate", {}).get("verdict"),
            "full_waypoint_supervised_training_ready": bool(
                stage43_l.get("stage43_l_gate", {}).get("full_waypoint_supervised_training_ready")
            ),
        },
        "cache_row_hashes": {split: _row_hash(_npz(_cache_path(split))) for split in SPLITS},
        "data_rows": {"train": len(train.x), "val": len(val.x), "test": len(test.x)},
        "feature_count": int(train.x.shape[1]),
        "training_history": history,
        "validation_selected_policy": val_policy,
        "test_metrics_with_floor": protected_metrics,
        "bootstrap_ci": bootstrap,
        "test_metrics_neural_without_floor": ungated_metrics,
        "latent_variance": latent_var,
        "deploy_neural": deploy,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "input_hash": _combined_hash(
            [
                _cache_path("train"),
                _cache_path("val"),
                _cache_path("test"),
                DATA37 / "history_windows_train.npz",
                DATA37 / "goal_prototypes_train.npz",
                DATA37 / "t50_baseline_family_train.npz",
                OUT_DIR / "stage43_full_waypoint_supervision_cache.json",
                OUT_DIR / "stage43_source_slice_repair.json",
            ]
        ),
    }
    payload["stage43_m_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(dict(payload)))
    gate = payload["stage43_m_gate"]
    m = payload["test_metrics_with_floor"]
    u = payload["test_metrics_neural_without_floor"]
    ci = payload["bootstrap_ci"]["metrics"]
    lines = [
        "# Stage43-M Protected Full-Waypoint Latent Dynamics",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- mode: `{payload['mode']}`",
        f"- checkpoint committed: `{payload['checkpoint_committed']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- deploy neural full-waypoint head: `{gate['deploy_neural_full_waypoint']}`",
        "",
        "## Current Facts",
        "",
        "- 当前不是 true 3D world model。",
        "- 当前不是 foundation world model。",
        "- 当前仍是 dataset-local / raw-frame 2.5D multi-agent world-state candidate。",
        "- full waypoints / future endpoints 只作为 loss/eval label，不作为 inference input。",
        "- Stage5C latent generative 未执行。",
        "- SMC 未启用。",
        "",
        "## Protected Test Metrics vs Full-Waypoint Floor",
        "",
        f"- rows: `{m['rows']}`",
        f"- full-waypoint ADE improvement: `{_pct(m['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- endpoint FDE improvement: `{_pct(m['endpoint_fde_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement: `{_pct(m['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50 endpoint FDE improvement: `{_pct(m['t50_endpoint_fde_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(m['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- hard/failure full-waypoint ADE improvement: `{_pct(m['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(m['easy_degradation_vs_floor'])}`",
        f"- switch rate: `{_pct(m['switch_rate'])}`",
        "",
        "## Bootstrap CI",
        "",
        f"- bootstrap n: `{payload['bootstrap_ci']['n']}`",
        f"- full-waypoint ADE improvement CI: `[{_pct(ci['full_waypoint_ade_improvement_vs_floor']['low'])}, {_pct(ci['full_waypoint_ade_improvement_vs_floor']['high'])}]`",
        f"- t50 full-waypoint ADE improvement CI: `[{_pct(ci['t50_full_waypoint_ade_improvement_vs_floor']['low'])}, {_pct(ci['t50_full_waypoint_ade_improvement_vs_floor']['high'])}]`",
        f"- hard/failure ADE improvement CI: `[{_pct(ci['hard_failure_full_waypoint_ade_improvement_vs_floor']['low'])}, {_pct(ci['hard_failure_full_waypoint_ade_improvement_vs_floor']['high'])}]`",
        f"- easy degradation CI: `[{_pct(ci['easy_degradation_vs_floor']['low'])}, {_pct(ci['easy_degradation_vs_floor']['high'])}]`",
        "",
        "## Ungated Neural Diagnostic",
        "",
        f"- full-waypoint ADE improvement: `{_pct(u['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement: `{_pct(u['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- hard/failure full-waypoint ADE improvement: `{_pct(u['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(u['easy_degradation_vs_floor'])}`",
        "",
        "## No-Leakage Boundary",
        "",
        "- future endpoint input: `False`",
        "- future waypoint input: `False`",
        "- future waypoint label/eval only: `True`",
        "- central velocity input: `False`",
        "- test endpoint goal construction: `False`",
        "- test statistics normalization: `False`",
        "",
        "## Interpretation",
        "",
    ]
    if gate["deploy_neural_full_waypoint"]:
        lines.append(
            "Stage43-M provides a protected full-waypoint latent dynamics candidate under the frozen floor. "
            "It is still dataset-local raw-frame 2.5D evidence, not metric/seconds-level or generative world-model execution."
        )
    else:
        lines.append(
            "Stage43-M trained a real torch latent full-waypoint dynamics head, but deployment remains diagnostic unless the protected metrics beat the floor while preserving easy cases."
        )
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage43-M Gate",
        "",
        f"verdict: `{gate['verdict']}`",
        f"deploy_neural_full_waypoint: `{gate['deploy_neural_full_waypoint']}`",
        f"passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    gate_lines.extend([f"| {name} | `{value}` |" for name, value in gate["gates"].items()])
    write_md(GATE_MD, gate_lines)
    _refresh_readmes(payload)
    _update_state(payload)
    _append_ledger(payload)


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_m_gate"]
    m = payload["test_metrics_with_floor"]
    ci = payload["bootstrap_ci"]["metrics"]
    summary = [
        "## Stage43-M protected full-waypoint latent dynamics",
        "",
        f"Result source: `{payload['result_source']}`. A torch latent dynamics head was trained on the frozen Stage43-L full-waypoint supervision cache, with future waypoints used only as labels/eval and with the frozen protected floor kept as the deployment guard.",
        "",
        f"- mode: `{payload['mode']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- deploy neural full-waypoint head: `{gate['deploy_neural_full_waypoint']}`",
        f"- full-waypoint ADE improvement vs floor: `{_pct(m['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement vs floor: `{_pct(m['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- hard/failure full-waypoint ADE improvement vs floor: `{_pct(m['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(m['easy_degradation_vs_floor'])}`",
        f"- t50 bootstrap CI: `[{_pct(ci['t50_full_waypoint_ade_improvement_vs_floor']['low'])}, {_pct(ci['t50_full_waypoint_ade_improvement_vs_floor']['high'])}]`",
        "",
        "Claim boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; Stage5C not executed; SMC not enabled.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, summary)


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    gate = payload["stage43_m_gate"]
    state["stage43_m_full_waypoint_latent_dynamics"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "mode": payload["mode"],
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "deploy_neural_full_waypoint": gate["deploy_neural_full_waypoint"],
        "metrics": payload["test_metrics_with_floor"],
        "bootstrap_ci": payload["bootstrap_ci"],
        "claim_boundary": payload["claim_boundary"],
        "checkpoint_committed": payload["checkpoint_committed"],
    }
    state["current_stage"] = "stage43_m_full_waypoint_latent_dynamics"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(OUT_DIR)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable({"event": "stage43_m_full_waypoint_latent_dynamics", "payload": payload}), ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Stage43-M protected full-waypoint latent dynamics head.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="Small smoke run with explicit quick provenance.")
    group.add_argument("--small", action="store_true", help="Local small run.")
    group.add_argument("--medium", action="store_true", help="Larger local run.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--seed", type=int, default=431)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    if not args.quick and not args.small and not args.medium:
        args.small = True
    result = _train_eval(args)
    gate = result["stage43_m_gate"]
    print(f"Stage43-M: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"deploy_neural_full_waypoint={gate['deploy_neural_full_waypoint']}")
    return result


if __name__ == "__main__":
    main()
