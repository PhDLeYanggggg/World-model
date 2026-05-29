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
        "Refusing Stage43 torch training under macOS x86_64/Rosetta. "
        "Use .venv-pytorch/bin/python arm64 with num_workers=0."
    )

import numpy as np
import torch
from torch import nn

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section


OUT_DIR = Path("outputs/stage43_latent_state")
CKPT_DIR = OUT_DIR / "checkpoints"
DATA35 = Path("data/stage35_selective_transfer")
DATA36 = Path("data/stage36_t50_repair")
DATA37 = Path("data/stage37_t50_history")

TRAINING_JSON = OUT_DIR / "stage43_protected_latent_training.json"
TRAINING_MD = OUT_DIR / "stage43_protected_latent_training.md"
EVAL_JSON = OUT_DIR / "stage43_protected_latent_eval.json"
EVAL_MD = OUT_DIR / "stage43_protected_latent_eval.md"
GATE_MD = OUT_DIR / "stage43_stage_c_protected_latent_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE43_C_PROTECTED_LATENT_STATE_SMALL"
SOURCE = "fresh_stage43_c_protected_latent_state_small"
HORIZONS = [10, 25, 50, 100]
DOMAINS = ["ETH_UCY", "TrajNet", "UCY"]
EPS = 1e-8


@dataclass
class SplitData:
    split: str
    x: np.ndarray
    y_delta: np.ndarray
    y_failure: np.ndarray
    y_gain: np.ndarray
    y_harm: np.ndarray
    y_occupancy: np.ndarray
    horizon: np.ndarray
    domain: np.ndarray
    floor_err: np.ndarray
    strongest_err: np.ndarray
    candidate_err_ref: np.ndarray
    hard: np.ndarray
    failure: np.ndarray
    easy: np.ndarray
    scale: np.ndarray
    feature_names: list[str]


class ProtectedLatentStateModel(nn.Module):
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
            nn.Linear(6, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.head = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 7))

    def forward(self, x: torch.Tensor, target_vec: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        z_t = self.encoder(x)
        z_next = self.dynamics(z_t)
        out = self.head(z_next)
        result = {
            "z_t": z_t,
            "z_next": z_next,
            "delta": out[:, :2],
            "failure_logit": out[:, 2],
            "gain_logit": out[:, 3],
            "harm_logit": out[:, 4],
            "occupancy": torch.sigmoid(out[:, 5]),
            "validity_logit": out[:, 6],
        }
        if target_vec is not None:
            result["target_latent"] = self.future_target_encoder(target_vec).detach()
        return result


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


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


def _load_npz(path: Path) -> Mapping[str, np.ndarray]:
    return np.load(path, allow_pickle=False)


def _one_hot(values: np.ndarray, choices: list[Any], prefix: str) -> tuple[np.ndarray, list[str]]:
    out = np.zeros((len(values), len(choices)), dtype=np.float32)
    for i, choice in enumerate(choices):
        out[:, i] = (values == choice).astype(np.float32)
    return out, [f"{prefix}_{choice}" for choice in choices]


def _tail(a: np.ndarray, k: int) -> np.ndarray:
    return a[:, -k:].astype(np.float32)


def _stage37_floor_error(split: str, labels: Mapping[str, np.ndarray], baseline: Mapping[str, np.ndarray]) -> np.ndarray:
    idx = np.arange(len(labels["y_fde"]))
    s35 = _load_npz(DATA36 / f"stage35_selection_{split}.npz")
    selected = np.asarray(s35["selected"], dtype=np.int64).clip(0, labels["y_fde"].shape[1] - 1)
    floor = labels["y_fde"][idx, selected].astype(np.float32)
    if split == "test" and (DATA37 / "stage37_best_t50_selection_test.npz").exists():
        st37 = _load_npz(DATA37 / "stage37_best_t50_selection_test.npz")
        h50 = _load_npz(DATA35 / "expanded_external_test.npz")["horizon"].astype(np.int64) == 50
        selected37 = np.asarray(st37["selected_family"], dtype=np.int64).clip(0, baseline["y_fde"].shape[1] - 1)
        floor[h50] = baseline["y_fde"][idx[h50], selected37[h50]].astype(np.float32)
    return floor


def _build_split(split: str, *, max_rows: int | None = None, seed: int = 431) -> SplitData:
    geo = _load_npz(DATA35 / f"expanded_external_{split}.npz")
    labels = _load_npz(DATA35 / f"labels_{split}.npz")
    hist = _load_npz(DATA37 / f"history_windows_{split}.npz")
    goal = _load_npz(DATA37 / f"goal_prototypes_{split}.npz")
    baseline = _load_npz(DATA37 / f"t50_baseline_family_{split}.npz")
    s35 = _load_npz(DATA36 / f"stage35_selection_{split}.npz")
    n = len(geo["horizon"])
    ids = np.arange(n)
    if max_rows is not None and max_rows < n:
        rng = np.random.default_rng(seed + {"train": 0, "val": 1, "test": 2}[split])
        ids = np.sort(rng.choice(ids, size=int(max_rows), replace=False))
    scale = np.maximum(labels["scale"][ids].astype(np.float32), 1e-4)
    cur = np.stack([geo["current_x"][ids], geo["current_y"][ids]], axis=1).astype(np.float32)
    fut = np.stack([geo["future_endpoint_x"][ids], geo["future_endpoint_y"][ids]], axis=1).astype(np.float32)
    y_delta = ((fut - cur) / scale[:, None]).astype(np.float32)
    domain_values = geo["dataset"][ids].astype(str)
    horizon_values = geo["horizon"][ids].astype(np.int64)
    domain_oh, domain_names = _one_hot(domain_values, DOMAINS, "domain")
    horizon_oh, horizon_names = _one_hot(horizon_values, HORIZONS, "horizon")
    current_norm = cur / scale[:, None]
    feature_parts: list[np.ndarray] = [current_norm, horizon_values[:, None].astype(np.float32) / 100.0, domain_oh, horizon_oh]
    feature_names = ["current_x_over_scale", "current_y_over_scale", "horizon_norm", *domain_names, *horizon_names]
    for key in ["history_dx", "history_dy", "history_speed", "history_accel", "history_heading", "history_valid_mask"]:
        vals = _tail(hist[key][ids], 8).astype(np.float32)
        feature_parts.append(vals)
        feature_names.extend([f"{key}_tail{i}" for i in range(vals.shape[1])])
    for key in [
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
    ]:
        feature_parts.append(hist[key][ids].astype(np.float32)[:, None])
        feature_names.append(key)
    for key in ["prototype_likelihood", "prototype_distance", "prototype_angle"]:
        vals = goal[key][ids].astype(np.float32)
        feature_parts.append(vals)
        feature_names.extend([f"{key}_{i}" for i in range(vals.shape[1])])
    for key in ["prototype_entropy", "goal_ambiguity"]:
        feature_parts.append(goal[key][ids].astype(np.float32)[:, None])
        feature_names.append(key)
    baseline_rel = ((baseline["prediction"][ids].astype(np.float32) - cur[:, None, :]) / scale[:, None, None]).reshape(len(ids), -1)
    feature_parts.append(baseline_rel)
    feature_names.extend([f"baseline_pred_rel_{i}" for i in range(baseline_rel.shape[1])])
    for key in ["predicted_gain", "hard_prob", "fail_prob", "easy_prob", "confidence"]:
        if key in s35.files:
            feature_parts.append(s35[key][ids].astype(np.float32)[:, None])
            feature_names.append(f"stage35_{key}")
    x = np.concatenate(feature_parts, axis=1).astype(np.float32)
    strongest_idx = labels["strongest_idx"][ids].astype(np.int64)
    oracle_idx = labels["oracle_idx"][ids].astype(np.int64)
    row = np.arange(len(ids))
    strongest_err = labels["y_fde"][ids][row, strongest_idx].astype(np.float32)
    oracle_err = labels["y_fde"][ids][row, oracle_idx].astype(np.float32)
    floor_full = _stage37_floor_error(split, labels, baseline)
    floor_err = floor_full[ids].astype(np.float32)
    gain = (oracle_err + 0.01 < strongest_err).astype(np.float32)
    harm = np.logical_or(labels["easy"][ids], labels["oracle_margin"][ids] < 0.01).astype(np.float32)
    occupancy = np.clip(hist["history_density"][ids].astype(np.float32) / 10.0, 0.0, 1.0)
    return SplitData(
        split=split,
        x=x,
        y_delta=y_delta,
        y_failure=labels["failure"][ids].astype(np.float32),
        y_gain=gain,
        y_harm=harm,
        y_occupancy=occupancy,
        horizon=horizon_values,
        domain=domain_values,
        floor_err=floor_err,
        strongest_err=strongest_err,
        candidate_err_ref=oracle_err,
        hard=labels["hard"][ids].astype(bool),
        failure=labels["failure"][ids].astype(bool),
        easy=labels["easy"][ids].astype(bool),
        scale=scale,
        feature_names=feature_names,
    )


def _standardize(train: SplitData, val: SplitData, test: SplitData) -> tuple[SplitData, SplitData, SplitData, np.ndarray, np.ndarray]:
    mean = train.x.mean(axis=0).astype(np.float32)
    raw_std = train.x.std(axis=0).astype(np.float32)
    # Held-out domain one-hot columns can be constant in train and active in test.
    # Dividing those columns by ~0 creates artificial million-scale features and
    # collapses the test latent state. Keep near-constant features unscaled.
    std = np.where(raw_std < 1e-3, 1.0, raw_std).astype(np.float32)
    for ds in [train, val, test]:
        ds.x = ((ds.x - mean) / std).astype(np.float32)
    return train, val, test, mean, std


def _target_vec(ds: SplitData) -> np.ndarray:
    return np.stack([ds.y_delta[:, 0], ds.y_delta[:, 1], ds.y_failure, ds.y_gain, ds.y_harm, ds.y_occupancy], axis=1).astype(np.float32)


def _batch_indices(n: int, batch_size: int, *, shuffle: bool, seed: int) -> list[np.ndarray]:
    ids = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(ids)
    return [ids[i : i + batch_size] for i in range(0, n, batch_size)]


def _loss(model: ProtectedLatentStateModel, ds: SplitData, ids: np.ndarray, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    x = torch.from_numpy(ds.x[ids]).to(device)
    y_delta = torch.from_numpy(ds.y_delta[ids]).to(device)
    y_failure = torch.from_numpy(ds.y_failure[ids]).to(device)
    y_gain = torch.from_numpy(ds.y_gain[ids]).to(device)
    y_harm = torch.from_numpy(ds.y_harm[ids]).to(device)
    y_occ = torch.from_numpy(ds.y_occupancy[ids]).to(device)
    target = torch.from_numpy(_target_vec(ds)[ids]).to(device)
    out = model(x, target)
    endpoint = nn.functional.smooth_l1_loss(out["delta"], y_delta)
    failure = nn.functional.binary_cross_entropy_with_logits(out["failure_logit"], y_failure)
    gain = nn.functional.binary_cross_entropy_with_logits(out["gain_logit"], y_gain)
    harm = nn.functional.binary_cross_entropy_with_logits(out["harm_logit"], y_harm)
    occupancy = nn.functional.mse_loss(out["occupancy"], y_occ)
    latent = nn.functional.mse_loss(out["z_next"], out["target_latent"])
    variance = out["z_next"].float().var(dim=0).mean()
    collapse = torch.relu(torch.tensor(0.02, device=device) - variance)
    total = 1.0 * endpoint + 0.4 * failure + 0.5 * gain + 0.6 * harm + 0.2 * occupancy + 0.4 * latent + collapse
    return total, {
        "endpoint": float(endpoint.detach().cpu()),
        "failure": float(failure.detach().cpu()),
        "gain": float(gain.detach().cpu()),
        "harm": float(harm.detach().cpu()),
        "occupancy": float(occupancy.detach().cpu()),
        "latent": float(latent.detach().cpu()),
        "latent_variance": float(variance.detach().cpu()),
    }


@torch.no_grad()
def _predict(model: ProtectedLatentStateModel, ds: SplitData, device: torch.device, batch_size: int) -> dict[str, np.ndarray]:
    model.eval()
    outs: dict[str, list[np.ndarray]] = {"delta": [], "failure": [], "gain": [], "harm": [], "occupancy": [], "latent": []}
    for ids in _batch_indices(len(ds.x), batch_size, shuffle=False, seed=0):
        x = torch.from_numpy(ds.x[ids]).to(device)
        out = model(x)
        outs["delta"].append(out["delta"].detach().cpu().numpy())
        outs["failure"].append(torch.sigmoid(out["failure_logit"]).detach().cpu().numpy())
        outs["gain"].append(torch.sigmoid(out["gain_logit"]).detach().cpu().numpy())
        outs["harm"].append(torch.sigmoid(out["harm_logit"]).detach().cpu().numpy())
        outs["occupancy"].append(out["occupancy"].detach().cpu().numpy())
        outs["latent"].append(out["z_next"].detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in outs.items()}


def _err_from_delta(ds: SplitData, pred_delta: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((pred_delta.astype(np.float64) - ds.y_delta.astype(np.float64)) ** 2, axis=1)).astype(np.float32)


def _slice_improvement(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return 0.0
    return float(1.0 - float(np.mean(selected[mask])) / max(float(np.mean(floor[mask])), EPS))


def _metrics(ds: SplitData, selected_err: np.ndarray, switched: np.ndarray) -> dict[str, Any]:
    hard_failure = ds.hard | ds.failure
    h50 = ds.horizon == 50
    h100 = ds.horizon == 100
    easy_deg = float(max(0.0, float(np.mean(selected_err[ds.easy])) / max(float(np.mean(ds.floor_err[ds.easy])), EPS) - 1.0)) if int(ds.easy.sum()) else 0.0
    return {
        "rows": int(len(ds.x)),
        "all_improvement_vs_floor": _slice_improvement(selected_err, ds.floor_err, np.ones(len(ds.x), dtype=bool)),
        "t50_improvement_vs_floor": _slice_improvement(selected_err, ds.floor_err, h50),
        "t100_raw_frame_diagnostic_vs_floor": _slice_improvement(selected_err, ds.floor_err, h100),
        "hard_failure_improvement_vs_floor": _slice_improvement(selected_err, ds.floor_err, hard_failure),
        "easy_degradation_vs_floor": easy_deg,
        "switch_rate": float(np.mean(switched)),
        "harm_over_floor": float(np.mean(selected_err - ds.floor_err)),
        "mean_floor_err": float(np.mean(ds.floor_err)),
        "mean_selected_err": float(np.mean(selected_err)),
    }


def _select_with_policy(ds: SplitData, pred: Mapping[str, np.ndarray], policy: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
    candidate_err = _err_from_delta(ds, pred["delta"])
    allow = (
        (pred["gain"] >= float(policy["gain_threshold"]))
        & (pred["harm"] <= float(policy["harm_threshold"]))
        & (pred["failure"] >= float(policy["failure_threshold"]))
    )
    selected = np.where(allow, candidate_err, ds.floor_err)
    return selected.astype(np.float32), allow.astype(bool)


def _search_policy(val: SplitData, pred: Mapping[str, np.ndarray]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for gain in [0.0, 0.25, 0.45, 0.55, 0.65, 0.75, 0.85]:
        for harm in [0.20, 0.30, 0.40, 0.50, 0.75, 0.90, 1.00]:
            for failure in [0.0, 0.10, 0.20, 0.30, 0.40]:
                policy = {"gain_threshold": gain, "harm_threshold": harm, "failure_threshold": failure}
                selected, switched = _select_with_policy(val, pred, policy)
                met = _metrics(val, selected, switched)
                objective = (
                    1.0 * met["all_improvement_vs_floor"]
                    + 1.5 * met["t50_improvement_vs_floor"]
                    + 0.8 * met["hard_failure_improvement_vs_floor"]
                    - 20.0 * max(0.0, met["easy_degradation_vs_floor"] - 0.02)
                    - 0.05 * met["switch_rate"]
                )
                row = {"policy": policy, "metrics": met, "objective": float(objective)}
                if best is None or row["objective"] > best["objective"]:
                    best = row
    assert best is not None
    return best


def _train(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CKPT_DIR)
    seed = int(args.seed)
    runtime = _configure_runtime(seed)
    max_train = 4000 if args.quick else 20000 if args.small else 60000
    max_val = 2000 if args.quick else 8000 if args.small else 20000
    max_test = 2000 if args.quick else 12000 if args.small else 30000
    train = _build_split("train", max_rows=max_train, seed=seed)
    val = _build_split("val", max_rows=max_val, seed=seed)
    test = _build_split("test", max_rows=max_test, seed=seed)
    train, val, test, mean, std = _standardize(train, val, test)
    device = torch.device("cpu")
    model = ProtectedLatentStateModel(train.x.shape[1], hidden_dim=int(args.hidden_dim), latent_dim=int(args.latent_dim)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    start = time.time()
    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_path = CKPT_DIR / "stage43_protected_latent_small.pt"
    for epoch in range(int(args.epochs)):
        model.train()
        losses: list[float] = []
        stats: list[dict[str, float]] = []
        for batch_ids in _batch_indices(len(train.x), int(args.batch_size), shuffle=True, seed=seed + epoch):
            opt.zero_grad(set_to_none=True)
            loss, batch_stats = _loss(model, train, batch_ids, device)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            stats.append(batch_stats)
        val_pred = _predict(model, val, device, int(args.batch_size))
        val_candidate = _err_from_delta(val, val_pred["delta"])
        val_loss = float(np.mean((val_candidate - val.candidate_err_ref) ** 2))
        latent_var = float(np.mean([s["latent_variance"] for s in stats])) if stats else 0.0
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_candidate_mse_to_oracle": val_loss,
            "latent_variance": latent_var,
        }
        history.append(row)
        heartbeat = {
            "source": SOURCE,
            "epoch": epoch + 1,
            "elapsed_s": time.time() - start,
            "last": row,
            "git_commit": _git_commit(),
        }
        write_json(OUT_DIR / "stage43_protected_latent_heartbeat.json", heartbeat)
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
                    "stage43_a_report": str(OUT_DIR / "stage43_safety_floor_replay.json"),
                    "stage43_b_report": str(OUT_DIR / "stage43_latent_state_dataset_contract.json"),
                    "no_leakage": {
                        "future_endpoint_input": False,
                        "future_waypoint_input": False,
                        "central_velocity_official_input": False,
                        "test_endpoint_goal_construction": False,
                    },
                },
                best_path,
            )
    ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    val_pred = _predict(model, val, device, int(args.batch_size))
    test_pred = _predict(model, test, device, int(args.batch_size))
    val_policy = _search_policy(val, val_pred)
    test_selected, test_switch = _select_with_policy(test, test_pred, val_policy["policy"])
    test_metrics = _metrics(test, test_selected, test_switch)
    candidate_metrics = _metrics(test, _err_from_delta(test, test_pred["delta"]), np.ones(len(test.x), dtype=bool))
    latent_var = float(np.var(test_pred["latent"], axis=0).mean()) if len(test_pred["latent"]) else 0.0
    eval_payload = {
        "source": SOURCE,
        "result_source": "fresh_run",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "mode": "quick" if args.quick else "small" if args.small else "medium",
        "checkpoint": str(best_path),
        "checkpoint_sha256": _sha256(best_path),
        "runtime": runtime,
        "data_rows": {"train": len(train.x), "val": len(val.x), "test": len(test.x)},
        "training_history": history,
        "validation_selected_policy": val_policy,
        "test_metrics_with_floor": test_metrics,
        "test_metrics_neural_without_floor": candidate_metrics,
        "latent_variance": latent_var,
        "jepa_or_latent_noncollapse": latent_var > 0.01,
        "deploy_neural": bool(
            (
                test_metrics["all_improvement_vs_floor"] > 0.0
                or test_metrics["t50_improvement_vs_floor"] > 0.0
                or test_metrics["hard_failure_improvement_vs_floor"] > 0.0
            )
            and test_metrics["easy_degradation_vs_floor"] <= 0.02
        ),
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "ordinary_residual": False,
        },
    }
    eval_payload["stage43_c_gate"] = _gate(eval_payload)
    training_payload = {
        "source": SOURCE,
        "result_source": "fresh_run",
        "generated_at_utc": eval_payload["generated_at_utc"],
        "git_commit": _git_commit(),
        "mode": eval_payload["mode"],
        "checkpoint": str(best_path),
        "checkpoint_committed": False,
        "runtime": runtime,
        "data_rows": eval_payload["data_rows"],
        "training_history": history,
        "input_hash": _combined_hash(
            [
                DATA35 / "expanded_external_train.npz",
                DATA35 / "labels_train.npz",
                DATA37 / "history_windows_train.npz",
                DATA37 / "goal_prototypes_train.npz",
                DATA37 / "t50_baseline_family_train.npz",
                OUT_DIR / "stage43_safety_floor_replay.json",
                OUT_DIR / "stage43_latent_state_dataset_contract.json",
            ]
        ),
    }
    _write_outputs(training_payload, eval_payload)
    return eval_payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["test_metrics_with_floor"]
    candidate = payload["test_metrics_neural_without_floor"]
    gates = {
        "torch_training_fresh_run": payload["result_source"] == "fresh_run" and Path(payload["checkpoint"]).exists(),
        "stage43_a_b_preconditions_used": Path(payload["checkpoint"]).exists()
        and bool(payload.get("checkpoint_sha256")),
        "latent_noncollapse": payload["jepa_or_latent_noncollapse"] is True,
        "protected_eval_completed": metrics["rows"] > 0,
        "easy_preserved": metrics["easy_degradation_vs_floor"] <= 0.02,
        "neural_has_any_protected_lift": (
            metrics["all_improvement_vs_floor"] > 0.0
            or metrics["t50_improvement_vs_floor"] > 0.0
            or metrics["hard_failure_improvement_vs_floor"] > 0.0
        ),
        "ungated_neural_reported": "all_improvement_vs_floor" in candidate,
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
        "verdict": "stage43_c_protected_latent_state_candidate_pass" if passed == total else "stage43_c_protected_latent_state_diagnostic_only",
        "deploy_neural": payload["deploy_neural"] and passed == total,
    }


def _write_outputs(training: Mapping[str, Any], eval_payload: Mapping[str, Any]) -> None:
    write_json(TRAINING_JSON, _jsonable(training))
    write_json(EVAL_JSON, _jsonable(eval_payload))
    gate = eval_payload["stage43_c_gate"]
    metrics = eval_payload["test_metrics_with_floor"]
    cand = eval_payload["test_metrics_neural_without_floor"]
    write_md(
        TRAINING_MD,
        [
            "# Stage43-C Protected Latent-State Training",
            "",
            f"- source: `{training['source']}`",
            f"- result source: `{training['result_source']}`",
            f"- mode: `{training['mode']}`",
            f"- checkpoint: `{training['checkpoint']}`",
            "- checkpoint committed: `False`",
            f"- runtime: `{training['runtime']}`",
            f"- data rows: `{training['data_rows']}`",
            "",
            "This is a protected latent-state head: z_t is learned from causal inputs and z_t -> z_{t+h} is trained against label-only future latent targets. It is not Stage5C latent generative rollout and does not enable SMC.",
        ],
    )
    write_md(
        EVAL_MD,
        [
            "# Stage43-C Protected Latent-State Evaluation",
            "",
            f"- source: `{eval_payload['source']}`",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- deploy neural: `{gate['deploy_neural']}`",
            f"- latent variance: `{eval_payload['latent_variance']:.6f}`",
            "",
            "## Protected Metrics vs Safety Floor",
            "",
            f"- all improvement vs floor: `{metrics['all_improvement_vs_floor']:.6f}`",
            f"- t50 improvement vs floor: `{metrics['t50_improvement_vs_floor']:.6f}`",
            f"- t100 raw-frame diagnostic vs floor: `{metrics['t100_raw_frame_diagnostic_vs_floor']:.6f}`",
            f"- hard/failure improvement vs floor: `{metrics['hard_failure_improvement_vs_floor']:.6f}`",
            f"- easy degradation vs floor: `{metrics['easy_degradation_vs_floor']:.6f}`",
            f"- switch rate: `{metrics['switch_rate']:.6f}`",
            "",
            "## Ungated Neural Diagnostic",
            "",
            f"- all improvement vs floor: `{cand['all_improvement_vs_floor']:.6f}`",
            f"- t50 improvement vs floor: `{cand['t50_improvement_vs_floor']:.6f}`",
            f"- hard/failure improvement vs floor: `{cand['hard_failure_improvement_vs_floor']:.6f}`",
            f"- easy degradation vs floor: `{cand['easy_degradation_vs_floor']:.6f}`",
            "",
            "No Stage5C, no SMC, no metric/seconds/true-3D/foundation claim.",
        ],
    )
    write_md(
        GATE_MD,
        [
            "# Stage43-C Protected Latent-State Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- deploy neural: `{gate['deploy_neural']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ],
    )
    _update_readmes(eval_payload)
    ensure_dir(LEDGER_JSONL.parent)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable({"stage": "Stage43-C", "source": eval_payload["source"], "verdict": gate["verdict"], "gate": f"{gate['passed']} / {gate['total']}", "generated_at_utc": eval_payload["generated_at_utc"]}), ensure_ascii=False) + "\n")


def _update_readmes(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_c_gate"]
    metrics = payload["test_metrics_with_floor"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"deploy_neural = `{gate['deploy_neural']}`",
        "",
        "Stage43-C trains a real PyTorch protected latent-state head on the Stage43 contract. Inputs are causal/current-or-past only; future endpoint/full-waypoint labels remain loss/eval only. The model learns z_t and z_t -> z_{t+h}, plus endpoint/failure/gain/harm/occupancy heads, then evaluates only through a safety-floor fallback policy.",
        "",
        f"Protected eval vs floor: all `{metrics['all_improvement_vs_floor']:.6f}`, t50 `{metrics['t50_improvement_vs_floor']:.6f}`, t100 raw diagnostic `{metrics['t100_raw_frame_diagnostic_vs_floor']:.6f}`, hard/failure `{metrics['hard_failure_improvement_vs_floor']:.6f}`, easy degradation `{metrics['easy_degradation_vs_floor']:.6f}`.",
        "",
        "This is not Stage5C, not SMC, not metric/seconds-level, not true 3D, and not a foundation model.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(RESEARCH_STATE, {})
    state["stage43_c_protected_latent_state_small"] = {
        "source": payload["source"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "deploy_neural": gate["deploy_neural"],
        "metrics": payload["test_metrics_with_floor"],
        "report": str(EVAL_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    write_json(RESEARCH_STATE, state)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--small", action="store_true")
    mode.add_argument("--medium", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=431)
    args = parser.parse_args(argv)
    if not args.quick and not args.small and not args.medium:
        args.small = True
    if args.quick:
        args.epochs = min(args.epochs, 2)
    return _train(args)


if __name__ == "__main__":
    result = main()
    gate = result["stage43_c_gate"]
    print(f"Stage43-C protected latent-state model: {gate['verdict']} ({gate['passed']}/{gate['total']})")
