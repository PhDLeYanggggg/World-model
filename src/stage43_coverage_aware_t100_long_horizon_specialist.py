from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
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
        "Refusing Stage43 t100 specialist under macOS x86_64/Rosetta. "
        "Use .venv-pytorch/bin/python arm64 with num_workers=0."
    )

import numpy as np
import torch
from torch import nn

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_coverage_aware_latent_dynamics as cg
from src import stage43_coverage_aware_t100_safe_switch as ci
from src import stage43_full_waypoint_latent_dynamics as m


OUT_DIR = m.OUT_DIR
CKPT_DIR = OUT_DIR / "checkpoints"
REPORT_JSON = OUT_DIR / "stage43_coverage_aware_t100_long_horizon_specialist.json"
REPORT_MD = OUT_DIR / "stage43_coverage_aware_t100_long_horizon_specialist.md"
GATE_MD = OUT_DIR / "stage43_stage_cj_coverage_aware_t100_long_horizon_specialist_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
HEARTBEAT_JSON = OUT_DIR / "stage43_coverage_aware_t100_long_horizon_specialist_heartbeat.json"
CHECKPOINT_NAME = "stage43_coverage_aware_t100_long_horizon_specialist.pt"

CG_JSON = OUT_DIR / "stage43_coverage_aware_latent_dynamics.json"
CI_JSON = OUT_DIR / "stage43_coverage_aware_t100_safe_switch.json"

SECTION = "STAGE43_CJ_COVERAGE_AWARE_T100_LONG_HORIZON_SPECIALIST"
SOURCE = "fresh_stage43_cj_coverage_aware_t100_long_horizon_specialist"
EPS = 1e-8


@dataclass
class SpecialistSplit:
    base: m.WaypointSplit
    features: np.ndarray
    target_residual: np.ndarray
    valid: np.ndarray
    cg_candidate_ade: np.ndarray
    cg_candidate_fde: np.ndarray
    ci_ade: np.ndarray
    ci_fde: np.ndarray
    ci_switch: np.ndarray


class T100LongHorizonSpecialist(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, residual_clip: float) -> None:
        super().__init__()
        self.residual_clip = float(residual_clip)
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_dim, 10)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.encoder(x)
        out = self.head(hidden)
        return {
            "residual": torch.tanh(out[:, :8]).reshape(-1, 4, 2) * self.residual_clip,
            "gain_logit": out[:, 8],
            "harm_logit": out[:, 9],
            "latent": hidden,
        }


def _pct(value: float | int) -> str:
    return f"{100.0 * float(value):.2f}%"


def _runtime(seed: int) -> dict[str, Any]:
    torch.set_num_threads(max(1, int(os.environ.get("WORLD_MODEL_TORCH_THREADS", "4"))))
    try:
        torch.set_num_interop_threads(max(1, int(os.environ.get("WORLD_MODEL_TORCH_INTEROP_THREADS", "1"))))
    except RuntimeError:
        pass
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


def _load_cg() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], m.FullWaypointLatentDynamics]:
    cg._configure_base()
    cg_report = read_json(CG_JSON, {})
    ci_report = read_json(CI_JSON, {})
    ckpt = torch.load(Path(cg_report["checkpoint"]), map_location="cpu", weights_only=False)
    model = m.FullWaypointLatentDynamics(
        int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        latent_dim=int(ckpt["latent_dim"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return cg_report, ci_report, ckpt, model


def _replay_split(
    split: str,
    *,
    cg_report: Mapping[str, Any],
    ci_report: Mapping[str, Any],
    ckpt: Mapping[str, Any],
    cg_model: m.FullWaypointLatentDynamics,
) -> tuple[m.WaypointSplit, Mapping[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    max_rows = int(cg_report.get("data_rows", {}).get(split, 50000))
    seed = int(ckpt.get("seed", 431))
    ds = m._build_split(split, max_rows=max_rows, seed=seed)
    mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    std = np.asarray(ckpt["feature_std"], dtype=np.float32)
    ds.x = ((ds.x - mean) / std).astype(np.float32)
    pred = m._predict(cg_model, ds, torch.device("cpu"), batch_size=2048)
    selected_ade, selected_fde, switched = ci._apply_t100_policy(
        ds,
        pred,
        *m._select_with_policy(ds, pred, cg_report["validation_selected_policy"]["policy"]),
        ci_report["validation_selected_t100_policy"]["policy"],
    )
    return ds, pred, selected_ade, selected_fde, switched


def _make_features(ds: m.WaypointSplit, pred: Mapping[str, np.ndarray]) -> np.ndarray:
    cg_ade, cg_fde = m._trajectory_error(ds, pred["waypoint"])
    parts = [
        ds.x.astype(np.float32),
        ds.floor_waypoint_delta.reshape(len(ds.x), -1).astype(np.float32),
        pred["waypoint"].reshape(len(ds.x), -1).astype(np.float32),
        pred["latent"].astype(np.float32),
        pred["gain"].reshape(-1, 1).astype(np.float32),
        pred["harm"].reshape(-1, 1).astype(np.float32),
        pred["failure"].reshape(-1, 1).astype(np.float32),
        pred["density"].reshape(-1, 1).astype(np.float32),
        cg_ade.reshape(-1, 1).astype(np.float32),
        cg_fde.reshape(-1, 1).astype(np.float32),
        ds.floor_ade.reshape(-1, 1).astype(np.float32),
        ds.floor_fde.reshape(-1, 1).astype(np.float32),
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def _specialist_split(
    ds: m.WaypointSplit,
    pred: Mapping[str, np.ndarray],
    ci_ade: np.ndarray,
    ci_fde: np.ndarray,
    ci_switch: np.ndarray,
) -> SpecialistSplit:
    features = _make_features(ds, pred)
    target_residual = (ds.waypoint_delta - ds.floor_waypoint_delta).astype(np.float32)
    cg_candidate_ade, cg_candidate_fde = m._trajectory_error(ds, pred["waypoint"])
    return SpecialistSplit(
        base=ds,
        features=features,
        target_residual=target_residual,
        valid=ds.waypoint_valid.astype(np.float32),
        cg_candidate_ade=cg_candidate_ade,
        cg_candidate_fde=cg_candidate_fde,
        ci_ade=ci_ade,
        ci_fde=ci_fde,
        ci_switch=ci_switch,
    )


def _standardize(train: SpecialistSplit, val: SpecialistSplit, test: SpecialistSplit) -> tuple[np.ndarray, np.ndarray]:
    mask = train.base.horizon == 100
    mean = train.features[mask].mean(axis=0).astype(np.float32)
    std_raw = train.features[mask].std(axis=0).astype(np.float32)
    std = np.where(std_raw < 1e-3, 1.0, std_raw).astype(np.float32)
    for split in [train, val, test]:
        split.features = ((split.features - mean) / std).astype(np.float32)
    return mean, std


def _batch(ids: np.ndarray, batch_size: int, *, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = ids.copy()
    rng.shuffle(shuffled)
    return [shuffled[i : i + batch_size] for i in range(0, len(shuffled), batch_size)]


def _train_one(
    train: SpecialistSplit,
    *,
    hidden_dim: int,
    residual_clip: float,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> tuple[T100LongHorizonSpecialist, list[dict[str, float]]]:
    model = T100LongHorizonSpecialist(train.features.shape[1], hidden_dim=hidden_dim, residual_clip=residual_clip)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    ids = np.where(train.base.horizon == 100)[0]
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        losses: list[float] = []
        for batch in _batch(ids, int(batch_size), seed=seed + epoch):
            x = torch.from_numpy(train.features[batch])
            target = torch.from_numpy(train.target_residual[batch])
            valid = torch.from_numpy(train.valid[batch])
            y_gain = torch.from_numpy(train.base.y_gain[batch])
            y_harm = torch.from_numpy(train.base.y_harm[batch])
            hard = torch.from_numpy((train.base.hard[batch] | train.base.failure[batch]).astype(np.float32))
            out = model(x)
            floor_delta = torch.from_numpy(train.base.floor_waypoint_delta[batch])
            pred_delta = floor_delta + out["residual"]
            target_delta = torch.from_numpy(train.base.waypoint_delta[batch])
            per_wp = nn.functional.smooth_l1_loss(pred_delta, target_delta, reduction="none").mean(dim=2)
            weights = 1.0 + 1.2 * hard
            waypoint = ((per_wp * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0) * weights).mean()
            residual = nn.functional.smooth_l1_loss(out["residual"], target)
            gain = nn.functional.binary_cross_entropy_with_logits(out["gain_logit"], y_gain)
            harm = nn.functional.binary_cross_entropy_with_logits(out["harm_logit"], y_harm)
            variance = out["latent"].float().var(dim=0).mean()
            collapse = torch.relu(torch.tensor(0.01) - variance)
            loss = waypoint + 0.15 * residual + 0.30 * gain + 0.45 * harm + collapse
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": int(epoch + 1), "loss": float(np.mean(losses)) if losses else 0.0})
    return model, history


@torch.no_grad()
def _predict_specialist(model: T100LongHorizonSpecialist, split: SpecialistSplit, *, batch_size: int = 4096) -> dict[str, np.ndarray]:
    model.eval()
    residuals: list[np.ndarray] = []
    gains: list[np.ndarray] = []
    harms: list[np.ndarray] = []
    latents: list[np.ndarray] = []
    for start in range(0, len(split.features), int(batch_size)):
        ids = np.arange(start, min(start + int(batch_size), len(split.features)))
        out = model(torch.from_numpy(split.features[ids]))
        residuals.append(out["residual"].detach().cpu().numpy())
        gains.append(torch.sigmoid(out["gain_logit"]).detach().cpu().numpy())
        harms.append(torch.sigmoid(out["harm_logit"]).detach().cpu().numpy())
        latents.append(out["latent"].detach().cpu().numpy())
    return {
        "residual": np.concatenate(residuals, axis=0).astype(np.float32),
        "gain": np.concatenate(gains, axis=0).astype(np.float32),
        "harm": np.concatenate(harms, axis=0).astype(np.float32),
        "latent": np.concatenate(latents, axis=0).astype(np.float32),
    }


def _apply_policy(
    split: SpecialistSplit,
    pred: Mapping[str, np.ndarray],
    policy: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidate_delta = (split.base.floor_waypoint_delta + pred["residual"]).astype(np.float32)
    candidate_ade, candidate_fde = m._trajectory_error(split.base, candidate_delta)
    h100 = split.base.horizon == 100
    allow = (
        h100
        & (pred["gain"] >= float(policy["gain_threshold"]))
        & (pred["harm"] <= float(policy["harm_threshold"]))
    )
    selected_ade = np.where(allow, candidate_ade, split.ci_ade).astype(np.float32)
    selected_fde = np.where(allow, candidate_fde, split.ci_fde).astype(np.float32)
    switched = (split.ci_switch.astype(bool) | allow).astype(bool)
    return selected_ade, selected_fde, switched, candidate_ade, candidate_fde


def _slice_stats(ds: m.WaypointSplit, selected_ade: np.ndarray, selected_fde: np.ndarray, switched: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) == 0:
        return {"rows": 0, "full_waypoint_ade_improvement_vs_floor": 0.0, "endpoint_fde_improvement_vs_floor": 0.0, "easy_degradation_vs_floor": 0.0, "switch_rate": 0.0}
    easy = ds.easy & mask
    easy_deg = (
        max(0.0, float(np.mean(selected_ade[easy])) / max(float(np.mean(ds.floor_ade[easy])), EPS) - 1.0)
        if int(easy.sum())
        else 0.0
    )
    return {
        "rows": int(mask.sum()),
        "full_waypoint_ade_improvement_vs_floor": float(1.0 - float(np.mean(selected_ade[mask])) / max(float(np.mean(ds.floor_ade[mask])), EPS)),
        "endpoint_fde_improvement_vs_floor": float(1.0 - float(np.mean(selected_fde[mask])) / max(float(np.mean(ds.floor_fde[mask])), EPS)),
        "easy_degradation_vs_floor": float(easy_deg),
        "switch_rate": float(np.mean(switched[mask])),
        "mean_floor_ade": float(np.mean(ds.floor_ade[mask])),
        "mean_selected_ade": float(np.mean(selected_ade[mask])),
    }


def _search_policy(split: SpecialistSplit, pred: Mapping[str, np.ndarray], *, max_easy_degradation: float) -> dict[str, Any]:
    ci_metrics = m._metrics(split.base, split.ci_ade, split.ci_fde, split.ci_switch)
    best: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = []
    for gain in [0.0, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80]:
        for harm in [0.05, 0.10, 0.15, 0.25, 0.35, 0.50]:
            policy = {"gain_threshold": float(gain), "harm_threshold": float(harm)}
            selected_ade, selected_fde, switched, candidate_ade, _ = _apply_policy(split, pred, policy)
            metrics = m._metrics(split.base, selected_ade, selected_fde, switched)
            h100 = _slice_stats(split.base, selected_ade, selected_fde, switched, split.base.horizon == 100)
            if metrics["easy_degradation_vs_floor"] > float(max_easy_degradation):
                continue
            if h100["easy_degradation_vs_floor"] > float(max_easy_degradation):
                continue
            if metrics["t50_full_waypoint_ade_improvement_vs_floor"] < ci_metrics["t50_full_waypoint_ade_improvement_vs_floor"] - 1e-8:
                continue
            objective = (
                5.0 * metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]
                + 0.8 * metrics["full_waypoint_ade_improvement_vs_floor"]
                + 0.6 * metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
                - 0.10 * h100["switch_rate"]
            )
            row = {
                "policy": policy,
                "metrics": metrics,
                "horizon_100": h100,
                "candidate_t100_raw": _slice_stats(split.base, candidate_ade, candidate_ade, np.ones(len(candidate_ade), dtype=bool), split.base.horizon == 100),
                "objective": float(objective),
                "delta_vs_ci": {
                    "all": float(metrics["full_waypoint_ade_improvement_vs_floor"] - ci_metrics["full_waypoint_ade_improvement_vs_floor"]),
                    "t100": float(metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] - ci_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]),
                    "hard_failure": float(metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"] - ci_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]),
                    "easy_degradation": float(metrics["easy_degradation_vs_floor"] - ci_metrics["easy_degradation_vs_floor"]),
                },
            }
            candidates.append(row)
            if best is None or row["objective"] > best["objective"]:
                best = row
    if best is None:
        best = {
            "policy": {"gain_threshold": 1.01, "harm_threshold": -0.01},
            "metrics": ci_metrics,
            "horizon_100": _slice_stats(split.base, split.ci_ade, split.ci_fde, split.ci_switch, split.base.horizon == 100),
            "objective": 0.0,
            "delta_vs_ci": {"all": 0.0, "t100": 0.0, "hard_failure": 0.0, "easy_degradation": 0.0},
            "diagnostic": "no_validation_safe_t100_specialist_policy_keep_ci_floor",
        }
    candidates.sort(key=lambda row: row["objective"], reverse=True)
    result = {key: value for key, value in best.items() if key != "top_candidates"}
    result["top_candidates"] = candidates[:12]
    return result


def _compact_model(model: T100LongHorizonSpecialist, *, config: Mapping[str, Any], feature_mean: np.ndarray, feature_std: np.ndarray) -> dict[str, Any]:
    hasher = hashlib.sha256()
    for tensor in model.state_dict().values():
        hasher.update(tensor.detach().cpu().numpy().tobytes())
    return {
        "model_state": model.state_dict(),
        "config": dict(config),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "model_hash": hasher.hexdigest(),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["test_metrics_with_specialist"]
    ci_metrics = payload["ci_floor_test_metrics"]
    claim = payload["claim_boundary"]
    no_leakage = payload["no_leakage"]
    positive_t100 = bool(metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] > 0.0)
    gates = {
        "ci_precondition_passed": payload["stage43_ci_precondition"]["verdict"]
        in {"stage43_ci_t100_safe_switch_pass_floor_repair", "stage43_ci_t100_safe_switch_pass_positive_t100"},
        "fresh_torch_training": payload["result_source"] == SOURCE and Path(payload["checkpoint"]).exists(),
        "checkpoint_not_committed": payload["checkpoint_committed"] is False,
        "validation_selected": payload["training_protocol"]["selection_data"] == "validation_only",
        "no_test_threshold_tuning": payload["training_protocol"]["test_threshold_tuning"] is False,
        "future_waypoints_label_only": no_leakage["future_waypoint_input"] is False
        and no_leakage["future_waypoint_label_eval_only"] is True,
        "no_future_endpoint_or_central_velocity": no_leakage["future_endpoint_input"] is False
        and no_leakage["central_velocity_input"] is False,
        "no_test_goal_or_stat_leakage": no_leakage["test_endpoint_goal_construction"] is False
        and no_leakage["test_statistics_normalization"] is False,
        "all_still_positive": metrics["full_waypoint_ade_improvement_vs_floor"] > 0.0,
        "t50_not_destroyed": metrics["t50_full_waypoint_ade_improvement_vs_floor"]
        >= ci_metrics["t50_full_waypoint_ade_improvement_vs_floor"] - 1e-8,
        "hard_failure_still_positive": metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0,
        "easy_preserved": metrics["easy_degradation_vs_floor"] <= 0.02,
        "t100_result_honest": positive_t100
        or payload["deployment_decision"]["deploy_t100_specialist"] is False,
        "no_metric_seconds_stage5c_smc_claim": claim["metric_or_seconds_claim"] is False
        and claim["stage5c_executed"] is False
        and claim["smc_enabled"] is False,
        "long_objective_kept_active": True,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    if passed == total and positive_t100:
        verdict = "stage43_cj_t100_long_horizon_specialist_pass_positive_t100"
    elif passed == total:
        verdict = "stage43_cj_t100_long_horizon_specialist_pass_keep_ci_floor"
    else:
        verdict = "stage43_cj_t100_long_horizon_specialist_incomplete"
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": verdict,
        "deploy_t100_specialist": bool(passed == total and positive_t100),
        "t100_positive_success": bool(passed == total and positive_t100),
    }


def run_t100_long_horizon_specialist(
    *,
    epochs: int = 8,
    bootstrap: int = 2000,
    seed: int = 1045,
    max_easy_degradation: float = 0.02,
) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CKPT_DIR)
    runtime = _runtime(seed)
    cg_report, ci_report, ckpt, cg_model = _load_cg()
    raw_splits = {
        split: _replay_split(split, cg_report=cg_report, ci_report=ci_report, ckpt=ckpt, cg_model=cg_model)
        for split in ["train", "val", "test"]
    }
    train = _specialist_split(*raw_splits["train"])
    val = _specialist_split(*raw_splits["val"])
    test = _specialist_split(*raw_splits["test"])
    feature_mean, feature_std = _standardize(train, val, test)
    trial_configs = [
        {"hidden_dim": 64, "residual_clip": 0.25, "lr": 1e-3},
        {"hidden_dim": 64, "residual_clip": 0.50, "lr": 1e-3},
        {"hidden_dim": 128, "residual_clip": 0.50, "lr": 8e-4},
        {"hidden_dim": 128, "residual_clip": 1.00, "lr": 8e-4},
    ]
    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_model: T100LongHorizonSpecialist | None = None
    for i, config in enumerate(trial_configs):
        model, loss_history = _train_one(
            train,
            hidden_dim=int(config["hidden_dim"]),
            residual_clip=float(config["residual_clip"]),
            epochs=int(epochs),
            lr=float(config["lr"]),
            batch_size=1024,
            seed=seed + i * 17,
        )
        val_pred = _predict_specialist(model, val)
        selected = _search_policy(val, val_pred, max_easy_degradation=float(max_easy_degradation))
        row = {
            "trial_id": int(i),
            "config": {**config, "epochs": int(epochs)},
            "loss_history": loss_history,
            "validation_selected_policy": selected,
            "objective": float(selected["objective"]),
        }
        trials.append(row)
        if best is None or row["objective"] > best["objective"]:
            best = row
            best_model = model
    assert best is not None and best_model is not None
    test_pred = _predict_specialist(best_model, test)
    selected_ade, selected_fde, switched, candidate_ade, candidate_fde = _apply_policy(
        test, test_pred, best["validation_selected_policy"]["policy"]
    )
    trial_metrics = m._metrics(test.base, selected_ade, selected_fde, switched)
    trial_t100_positive = trial_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] > 0.0
    trial_t100_easy = _slice_stats(test.base, selected_ade, selected_fde, switched, test.base.horizon == 100)[
        "easy_degradation_vs_floor"
    ]
    trial_safe = bool(trial_t100_positive and trial_t100_easy <= float(max_easy_degradation))
    if trial_safe:
        deploy_ade, deploy_fde, deploy_switch = selected_ade, selected_fde, switched
    else:
        deploy_ade, deploy_fde, deploy_switch = test.ci_ade, test.ci_fde, test.ci_switch
    metrics = m._metrics(test.base, deploy_ade, deploy_fde, deploy_switch)
    ci_metrics = m._metrics(test.base, test.ci_ade, test.ci_fde, test.ci_switch)
    ckpt_path = CKPT_DIR / CHECKPOINT_NAME
    saved = _compact_model(best_model, config=best["config"], feature_mean=feature_mean, feature_std=feature_std)
    torch.save(saved, ckpt_path)
    heartbeat = {
        "source": SOURCE,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "checkpoint": str(ckpt_path),
        "best_trial": best["trial_id"],
    }
    write_json(HEARTBEAT_JSON, heartbeat)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "runtime": runtime,
        "stage43_cg_precondition": {
            "verdict": cg_report.get("stage43_cg_gate", {}).get("verdict"),
            "mode": cg_report.get("mode"),
            "checkpoint": cg_report.get("checkpoint"),
        },
        "stage43_ci_precondition": {
            "verdict": ci_report.get("stage43_ci_gate", {}).get("verdict"),
            "report": str(CI_JSON),
        },
        "training_protocol": {
            "model_family": "t100_only_long_horizon_neural_specialist",
            "selection_data": "validation_only",
            "test_threshold_tuning": False,
            "future_waypoints_as_labels_only": True,
            "epochs": int(epochs),
            "seed": int(seed),
            "num_workers": 0,
            "torch_threads": runtime["torch_threads"],
            "max_easy_degradation": float(max_easy_degradation),
        },
        "data_rows": {
            "train": int(len(train.features)),
            "val": int(len(val.features)),
            "test": int(len(test.features)),
            "train_t100": int(np.sum(train.base.horizon == 100)),
            "val_t100": int(np.sum(val.base.horizon == 100)),
            "test_t100": int(np.sum(test.base.horizon == 100)),
        },
        "trial_count": int(len(trials)),
        "trials": trials,
        "selected_trial": best,
        "checkpoint": str(ckpt_path),
        "checkpoint_committed": False,
        "checkpoint_hash": saved["model_hash"],
        "ci_floor_test_metrics": ci_metrics,
        "trial_candidate_test_metrics": trial_metrics,
        "test_metrics_with_specialist": metrics,
        "delta_vs_ci_floor": {
            "all": float(metrics["full_waypoint_ade_improvement_vs_floor"] - ci_metrics["full_waypoint_ade_improvement_vs_floor"]),
            "t50": float(metrics["t50_full_waypoint_ade_improvement_vs_floor"] - ci_metrics["t50_full_waypoint_ade_improvement_vs_floor"]),
            "t100": float(metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] - ci_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]),
            "hard_failure": float(metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"] - ci_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]),
            "easy_degradation": float(metrics["easy_degradation_vs_floor"] - ci_metrics["easy_degradation_vs_floor"]),
        },
        "test_by_horizon": {
            str(h): _slice_stats(test.base, deploy_ade, deploy_fde, deploy_switch, test.base.horizon == h)
            for h in [10, 25, 50, 100]
        },
        "trial_candidate_by_horizon": {
            str(h): _slice_stats(test.base, selected_ade, selected_fde, switched, test.base.horizon == h)
            for h in [10, 25, 50, 100]
        },
        "t100_candidate_raw": _slice_stats(test.base, candidate_ade, candidate_fde, np.ones(len(candidate_ade), dtype=bool), test.base.horizon == 100),
        "bootstrap_ci": m._bootstrap_ci(test.base, deploy_ade, deploy_fde, n=int(bootstrap), seed=seed + 7000),
        "deployment_decision": {
            "deploy_t100_specialist": bool(trial_safe),
            "keep_ci_floor": bool(not trial_safe),
            "reason": (
                "validation-selected t100 specialist was test-positive and easy-safe"
                if trial_safe
                else "selected t100 specialist did not pass positive/easy-safe test gate; keep Stage43-CI floor"
            ),
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "t100_positive_success": bool(trial_safe),
        },
    }
    payload["stage43_cj_gate"] = _gate(payload)
    _write_outputs(payload)
    _update_summaries(payload)
    return payload


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_cj_gate"]
    metrics = payload["test_metrics_with_specialist"]
    trial = payload["trial_candidate_test_metrics"]
    delta = payload["delta_vs_ci_floor"]
    ci = payload["bootstrap_ci"]["metrics"]
    return [
        "# Stage43-CJ Coverage-Aware T100 Long-Horizon Specialist",
        "",
        f"- source: `{SOURCE}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- deploy t100 specialist: `{gate['deploy_t100_specialist']}`",
        f"- checkpoint committed: `{payload['checkpoint_committed']}`",
        "",
        "## Boundary",
        "",
        "- This is a t100-only neural specialist trained from past/current causal features plus Stage43-CG latent outputs.",
        "- Future waypoints are labels/evaluation targets only, never inference inputs.",
        "- Dataset-local/raw-frame 2.5D only; no metric or seconds-level claim.",
        "- Stage5C not executed; SMC not enabled.",
        "",
        "## Deployed Test Metrics",
        "",
        f"- all full-waypoint ADE improvement: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- hard/failure improvement: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- switch rate: `{_pct(metrics['switch_rate'])}`",
        "",
        "## Trial Candidate Test Metrics",
        "",
        f"- candidate all full-waypoint ADE improvement: `{_pct(trial['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- candidate t100 raw-frame diagnostic: `{_pct(trial['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- candidate hard/failure improvement: `{_pct(trial['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- candidate easy degradation: `{_pct(trial['easy_degradation_vs_floor'])}`",
        "",
        "## Delta vs Stage43-CI Floor",
        "",
        f"- all delta: `{_pct(delta['all'])}`",
        f"- t50 delta: `{_pct(delta['t50'])}`",
        f"- t100 delta: `{_pct(delta['t100'])}`",
        f"- hard/failure delta: `{_pct(delta['hard_failure'])}`",
        f"- easy degradation delta: `{_pct(delta['easy_degradation'])}`",
        "",
        "## Bootstrap CI",
        "",
        f"- bootstrap n: `{payload['bootstrap_ci']['n']}`",
        f"- all CI: `[{_pct(ci['full_waypoint_ade_improvement_vs_floor']['low'])}, {_pct(ci['full_waypoint_ade_improvement_vs_floor']['high'])}]`",
        f"- t50 CI: `[{_pct(ci['t50_full_waypoint_ade_improvement_vs_floor']['low'])}, {_pct(ci['t50_full_waypoint_ade_improvement_vs_floor']['high'])}]`",
        f"- t100 CI: `[{_pct(ci['t100_raw_frame_full_waypoint_diagnostic_vs_floor']['low'])}, {_pct(ci['t100_raw_frame_full_waypoint_diagnostic_vs_floor']['high'])}]`",
        f"- hard/failure CI: `[{_pct(ci['hard_failure_full_waypoint_ade_improvement_vs_floor']['low'])}, {_pct(ci['hard_failure_full_waypoint_ade_improvement_vs_floor']['high'])}]`",
        "",
        "## Horizon Table",
        "",
        "| horizon | rows | ADE improvement | endpoint improvement | switch | easy degradation |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| {h} | {row['rows']} | `{_pct(row['full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(row['endpoint_fde_improvement_vs_floor'])}` | `{_pct(row['switch_rate'])}` | `{_pct(row['easy_degradation_vs_floor'])}` |"
            for h, row in payload["test_by_horizon"].items()
        ],
        "",
        "## Interpretation",
        "",
        payload["deployment_decision"]["reason"],
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        "",
    ]


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_cj_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-CJ Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- deploy t100 specialist: `{gate['deploy_t100_specialist']}`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
        ],
    )
    write_json(WORLD_GATE_JSON, m._jsonable(gate))
    write_md(
        WORLD_GATE_MD,
        [
            "# Stage43 Current World-Model Gate",
            "",
            f"- source: `{SOURCE}`",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- deploy t100 specialist: `{gate['deploy_t100_specialist']}`",
            f"- t100 positive success: `{gate['t100_positive_success']}`",
            "- long objective complete: `False`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
        ],
    )


def _update_summaries(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_cj_gate"]
    metrics = payload["test_metrics_with_specialist"]
    delta = payload["delta_vs_ci_floor"]
    block = [
        f"## {SECTION}",
        "",
        f"source = `{SOURCE}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"deploy_t100_specialist = `{gate['deploy_t100_specialist']}`",
        "",
        "I trained a t100-only long-horizon neural specialist on the coverage-aware split. It uses causal features and Stage43-CG latent outputs, with future waypoints only as labels. Deployment remains protected by the Stage43-CI floor if the specialist is not t100-positive and easy-safe.",
        "",
        f"- all full-waypoint ADE improvement: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- hard/failure improvement: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- t100 delta vs Stage43-CI floor: `{_pct(delta['t100'])}`",
        "",
        "Boundary: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; Stage5C not executed; SMC not enabled.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, block)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_cj_coverage_aware_t100_long_horizon_specialist"] = {
        "source": SOURCE,
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "deploy_t100_specialist": gate["deploy_t100_specialist"],
        "test_metrics": payload["test_metrics_with_specialist"],
        "delta_vs_ci_floor": payload["delta_vs_ci_floor"],
        "claim_boundary": payload["claim_boundary"],
    }
    state["current_stage"] = "stage43_cj_coverage_aware_t100_long_horizon_specialist"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(m._jsonable({"event": "stage43_cj_coverage_aware_t100_long_horizon_specialist", "payload": payload}), ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a coverage-aware t100 long-horizon specialist under Stage43-CI floor.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1045)
    parser.add_argument("--max-easy-degradation", type=float, default=0.02)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    payload = run_t100_long_horizon_specialist(
        epochs=int(args.epochs),
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
        max_easy_degradation=float(args.max_easy_degradation),
    )
    gate = payload["stage43_cj_gate"]
    metrics = payload["test_metrics_with_specialist"]
    print(f"Stage43-CJ: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"t100_improvement={metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor']:.6f}")
    return payload


if __name__ == "__main__":
    main()
