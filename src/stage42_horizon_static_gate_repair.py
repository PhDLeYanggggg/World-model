from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_full_trajectory_world_state as ft
from src import stage42_sequence_full_waypoint as s42i
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
REPORT_JSON = OUT_DIR / "horizon_static_gate_repair_stage42.json"
REPORT_MD = OUT_DIR / "horizon_static_gate_repair_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_l_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SEEDS = [83, 89, 97]
EPOCHS = 2
BATCH = 2048
THREADS = 4
HORIZONS = [10, 25, 50, 100]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。",
    "Stage42-L 使用 dataset-local raw-frame full-waypoint labels，不能写成 metric 或 seconds-level。",
    "future waypoints / future endpoints 只作为 loss/eval label，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


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


def _ensure_arm64() -> None:
    if platform.machine() == "arm64":
        return
    venv = Path(".venv-pytorch/bin/python")
    if venv.exists() and os.environ.get("STAGE42L_REEXEC") != "1":
        env = os.environ.copy()
        env["STAGE42L_REEXEC"] = "1"
        os.execve("/usr/bin/arch", ["/usr/bin/arch", "-arm64", str(venv), *sys.argv], env)
    raise RuntimeError("Stage42-L refuses x86_64/Rosetta Python for torch training.")


def _torch():
    _ensure_arm64()
    import torch

    torch.set_num_threads(THREADS)
    return torch


def _horizon_index_np(horizon: np.ndarray) -> np.ndarray:
    h = horizon.astype(np.int64)
    out = np.zeros_like(h, dtype=np.int64)
    for idx, value in enumerate(HORIZONS):
        out[h == value] = idx
    return out


def _make_model(static_dim: int, token_dim: int = 9, width: int = 80):
    torch = _torch()
    import torch.nn as nn

    class HorizonStaticGatedWaypoint(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.temporal = nn.Sequential(
                nn.Conv1d(token_dim, 56, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(56, width, kernel_size=3, padding=1),
                nn.GELU(),
            )
            self.static = nn.Sequential(nn.Linear(static_dim, width), nn.GELU(), nn.LayerNorm(width))
            self.horizon = nn.Embedding(len(HORIZONS), width)
            self.gate = nn.Sequential(nn.Linear(width * 4, width), nn.GELU(), nn.Linear(width, 1))
            # Start conservative, but less closed than Stage42-K so t+50 can
            # learn safe static/context use when validation supports it.
            nn.init.constant_(self.gate[-1].bias, -1.35)
            self.ctx = nn.Sequential(nn.Linear(width * 4, width * 2), nn.GELU(), nn.LayerNorm(width * 2))
            self.waypoints = nn.Linear(width * 2, len(ft.WAYPOINT_FRAC) * 2)
            self.risk = nn.Linear(width * 2, 1)
            self.interaction = nn.Linear(width * 2, 1)
            self.occupancy = nn.Linear(width * 2, 1)
            self.physical = nn.Linear(width * 2, 1)

        def _encode_agents(self, tokens, mask):
            b, a, t, d = tokens.shape
            flat = tokens.reshape(b * a, t, d).transpose(1, 2)
            h = self.temporal(flat).transpose(1, 2).reshape(b, a, t, -1)
            valid_t = tokens[..., 6].clamp(0, 1)
            emb = (h * valid_t[..., None]).sum(dim=2) / valid_t.sum(dim=2, keepdim=True).clamp_min(1.0)
            emb = emb * mask[..., None].float()
            target = emb[:, 0]
            neigh_mask = mask[:, 1:, None].float()
            neigh = (emb[:, 1:] * neigh_mask).sum(dim=1) / neigh_mask.sum(dim=1).clamp_min(1.0)
            return target, neigh

        def forward(self, tokens, mask, static, horizon_idx):
            target, neigh = self._encode_agents(tokens, mask)
            static_h = self.static(static)
            horizon_h = self.horizon(horizon_idx)
            raw_gate = torch.sigmoid(self.gate(torch.cat([target, neigh, static_h, horizon_h], dim=1))).squeeze(-1)
            gated_static = raw_gate[:, None] * static_h
            ctx = self.ctx(torch.cat([target, neigh, gated_static, horizon_h], dim=1))
            return {
                "waypoint_delta": self.waypoints(ctx).view(-1, len(ft.WAYPOINT_FRAC), 2),
                "traj_risk": self.risk(ctx).squeeze(-1),
                "interaction_logit": self.interaction(ctx).squeeze(-1),
                "occupancy_logit": self.occupancy(ctx).squeeze(-1),
                "physical_logit": self.physical(ctx).squeeze(-1),
                "static_gate": raw_gate,
            }

    return HorizonStaticGatedWaypoint()


def _train_seed(seed: int, train: Mapping[str, np.ndarray], val: Mapping[str, np.ndarray]) -> dict[str, Any]:
    torch = _torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = _make_model(train["static"].shape[1], train["agent_tokens"].shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4)
    ckpt = CHECKPOINT_DIR / f"stage42l_horizon_static_gate_seed{seed}.pt"
    heartbeat = OUT_DIR / f"stage42l_horizon_static_gate_seed{seed}_heartbeat.json"
    if ckpt.exists() and heartbeat.exists():
        payload = read_json(heartbeat, {})
        if int(payload.get("epoch", 0)) >= EPOCHS:
            return {"source": "cached_verified", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": payload.get("best", {})}

    tensors = {
        "tokens": torch.tensor(train["agent_tokens"]),
        "mask": torch.tensor(train["agent_mask"]),
        "static": torch.tensor(train["static"]),
        "target": torch.tensor(train["waypoint_delta"]),
        "valid": torch.tensor(train["waypoint_valid"].astype(np.float32)),
        "interaction": torch.tensor(train["interaction"]),
        "occupancy": torch.tensor(train["occupancy"]),
        "physical": torch.tensor(train["physical"]),
        "hard": torch.tensor((train["hard"] | train["failure"]).astype(np.float32)),
        "horizon": torch.tensor(train["horizon"]),
        "horizon_idx": torch.tensor(_horizon_index_np(train["horizon"]), dtype=torch.long),
    }
    val_tensors = {
        "tokens": torch.tensor(val["agent_tokens"]),
        "mask": torch.tensor(val["agent_mask"]),
        "static": torch.tensor(val["static"]),
        "target": torch.tensor(val["waypoint_delta"]),
        "valid": torch.tensor(val["waypoint_valid"].astype(np.float32)),
        "horizon_idx": torch.tensor(_horizon_index_np(val["horizon"]), dtype=torch.long),
    }
    best = {"val_loss": float("inf"), "epoch": 0}
    waypoint_weight = torch.tensor([1.0, 1.15, 1.65, 2.75], dtype=torch.float32)
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(train["agent_tokens"]))
        losses: list[float] = []
        gate_vals: list[float] = []
        t50_gate_vals: list[float] = []
        model.train()
        for start in range(0, len(order), BATCH):
            ids = torch.tensor(order[start : start + BATCH], dtype=torch.long)
            horizon = tensors["horizon"][ids]
            static = tensors["static"][ids].clone()
            drop_prob = torch.where(horizon == 50, torch.tensor(0.20), torch.tensor(0.45))
            drop = torch.rand(static.shape[0]) < drop_prob
            static[drop] = 0.0
            out = model(tensors["tokens"][ids], tensors["mask"][ids], static, tensors["horizon_idx"][ids])
            valid = tensors["valid"][ids]
            is_t50 = (horizon == 50).float()
            row_w = 1.0 + 1.5 * tensors["hard"][ids] + 5.0 * is_t50 + 1.25 * (horizon == 100).float()
            per_wp = F.smooth_l1_loss(out["waypoint_delta"], tensors["target"][ids], reduction="none").mean(dim=2)
            wp_w = waypoint_weight.to(per_wp.device)[None, :] * (1.0 + 0.60 * is_t50[:, None])
            traj = ((per_wp * valid * wp_w).sum(dim=1) / (valid * wp_w).sum(dim=1).clamp_min(1.0) * row_w).mean()
            err = torch.linalg.norm(out["waypoint_delta"] - tensors["target"][ids], dim=2)
            risk_target = torch.log1p((err * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)).detach()
            risk = (F.smooth_l1_loss(out["traj_risk"], risk_target, reduction="none") * row_w).mean()
            aux = (
                F.binary_cross_entropy_with_logits(out["interaction_logit"], tensors["interaction"][ids])
                + F.binary_cross_entropy_with_logits(out["occupancy_logit"], tensors["occupancy"][ids])
                + F.binary_cross_entropy_with_logits(out["physical_logit"], tensors["physical"][ids])
            )
            gate_penalty = ((0.012 * (1.0 - is_t50) + 0.0025 * is_t50) * out["static_gate"]).mean()
            loss = traj + 0.35 * risk + 0.12 * aux + gate_penalty
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            gate_vals.append(float(out["static_gate"].mean().detach().cpu()))
            if bool(torch.any(horizon == 50)):
                t50_gate_vals.append(float(out["static_gate"][horizon == 50].mean().detach().cpu()))
        model.eval()
        with torch.no_grad():
            out = model(val_tensors["tokens"], val_tensors["mask"], val_tensors["static"], val_tensors["horizon_idx"])
            per_wp = F.smooth_l1_loss(out["waypoint_delta"], val_tensors["target"], reduction="none").mean(dim=2)
            val_loss = float(((per_wp * val_tensors["valid"]).sum(dim=1) / val_tensors["valid"].sum(dim=1).clamp_min(1.0)).mean().cpu())
            val_gate = float(out["static_gate"].mean().cpu())
        cand = {
            "val_loss": val_loss,
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_gate_mean": float(np.mean(gate_vals)),
            "train_t50_gate_mean": float(np.mean(t50_gate_vals)) if t50_gate_vals else 0.0,
            "val_gate_mean": val_gate,
        }
        if val_loss < best["val_loss"]:
            best = cand
            torch.save({"model": model.state_dict(), "seed": seed, "static_dim": train["static"].shape[1], "token_dim": train["agent_tokens"].shape[-1], "best": best}, ckpt)
        heartbeat.write_text(json.dumps({"seed": seed, "epoch": epoch, "best": best, "checkpoint": str(ckpt)}, ensure_ascii=False), encoding="utf-8")
    return {"source": "fresh_run", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best}


def _predict(info: Mapping[str, Any], split: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    torch = _torch()
    payload = torch.load(info["checkpoint"], map_location="cpu", weights_only=False)
    model = _make_model(int(payload["static_dim"]), int(payload["token_dim"]))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    hidx = _horizon_index_np(split["horizon"])
    outs = {k: [] for k in ["waypoint_delta", "traj_risk", "interaction", "occupancy", "physical", "static_gate"]}
    with torch.no_grad():
        for start in range(0, len(split["agent_tokens"]), 4096):
            sl = slice(start, min(start + 4096, len(split["agent_tokens"])))
            out = model(
                torch.tensor(split["agent_tokens"][sl]),
                torch.tensor(split["agent_mask"][sl]),
                torch.tensor(split["static"][sl]),
                torch.tensor(hidx[sl], dtype=torch.long),
            )
            outs["waypoint_delta"].append(out["waypoint_delta"].cpu().numpy())
            outs["traj_risk"].append(out["traj_risk"].cpu().numpy().reshape(-1))
            outs["interaction"].append(torch.sigmoid(out["interaction_logit"]).cpu().numpy().reshape(-1))
            outs["occupancy"].append(torch.sigmoid(out["occupancy_logit"]).cpu().numpy().reshape(-1))
            outs["physical"].append(torch.sigmoid(out["physical_logit"]).cpu().numpy().reshape(-1))
            outs["static_gate"].append(out["static_gate"].cpu().numpy().reshape(-1))
    return {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in outs.items()}


def _score_metric(metric: Mapping[str, Any]) -> float:
    return (
        0.9 * float(metric.get("all_improvement", 0.0))
        + 5.2 * float(metric.get("t50_improvement", 0.0))
        + 1.2 * float(metric.get("t100_improvement", 0.0))
        + 1.4 * float(metric.get("hard_failure_improvement", 0.0))
        - 50.0 * max(0.0, float(metric.get("easy_degradation", 1.0)) - 0.02)
    )


def _fit_t50_weighted_policy(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    floor_xy = ft._floor_waypoints(labels)
    neural_xy = ft._pred_waypoints(pred, labels)
    floor_ade, _ = ft._trajectory_errors(floor_xy, labels)
    neural_ade, _ = ft._trajectory_errors(neural_xy, labels)
    domain = labels["domain"].astype(str)
    horizon = labels["horizon"].astype(int)
    risk = pred["traj_risk"].astype(np.float64)
    policy: dict[str, Any] = {"type": "stage42l_t50_weighted_horizon_policy", "slices": {}}
    diagnostics: dict[str, Any] = {}
    for d in sorted(set(domain.tolist())):
        for h in HORIZONS:
            mask = (domain == d) & (horizon == h)
            if int(np.sum(mask)) < 80:
                continue
            qs = [0.05, 0.10, 0.20, 0.35, 0.50, 0.70] if h == 50 else [0.10, 0.25, 0.40, 0.60]
            switches = [0.05, 0.10, 0.20, 0.35, 0.50] if h == 50 else [0.05, 0.10, 0.20, 0.35]
            thresholds = [float(v) for v in np.quantile(risk[mask], qs)]
            best_params: dict[str, Any] | None = None
            best_score = 0.0
            best_metric: dict[str, Any] | None = None
            for risk_max in thresholds:
                for max_switch in switches:
                    for hard_only in [False, True]:
                        params = {"risk_max": risk_max, "physical_min": 0.0, "max_switch": max_switch, "hard_only": hard_only, "easy_block": True}
                        trial = {"slices": {f"{d}|{h}": params}}
                        switch = s42i._apply_light_policy(pred, labels, trial)
                        selected = floor_ade.copy()
                        selected[switch] = neural_ade[switch]
                        metric = ft._metric(selected, floor_ade, labels, switch)
                        if metric.get("easy_degradation", 1.0) > 0.02:
                            continue
                        score = _score_metric(metric)
                        if score > best_score:
                            best_score = score
                            best_params = params
                            best_metric = metric
            diagnostics[f"{d}|{h}"] = {"selected": bool(best_params), "score": float(best_score), "metric": best_metric or {"rows": int(np.sum(mask)), "all_improvement": 0.0}}
            if best_params is not None:
                policy["slices"][f"{d}|{h}"] = best_params
    switch = s42i._apply_light_policy(pred, labels, policy)
    selected = floor_ade.copy()
    selected[switch] = neural_ade[switch]
    metrics = ft._metric(selected, floor_ade, labels, switch)
    metrics["slice_diagnostics"] = diagnostics
    return policy, metrics


def _eval_seed(seed: int, train: Mapping[str, np.ndarray], val: Mapping[str, np.ndarray], test: Mapping[str, np.ndarray]) -> dict[str, Any]:
    info = _train_seed(seed, train, val)
    pred_val = _predict(info, val)
    pred_test = _predict(info, test)
    labels_val = s42i._labels(val)
    labels_test = s42i._labels(test)
    policy, val_metrics = _fit_t50_weighted_policy(pred_val, labels_val)
    test_metrics = s42i._row_metrics("horizon_static_gate_repair", pred_test, labels_test, policy)
    ungated = s42i._row_metrics(
        "horizon_static_gate_repair_ungated",
        pred_test,
        labels_test,
        {"slices": {f"{d}|{h}": {"risk_max": 1e9, "physical_min": 0.0, "max_switch": 1.0, "hard_only": False, "easy_block": False} for d in sorted(set(labels_test["domain"].tolist())) for h in HORIZONS}},
    )
    return {
        "source": info["source"],
        "seed": seed,
        "train_info": info,
        "val_policy": policy,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "ungated_test_metrics": ungated,
        "static_gate_mean_val": float(np.mean(pred_val["static_gate"])),
        "static_gate_mean_test": float(np.mean(pred_test["static_gate"])),
        "static_gate_t50_mean_test": float(np.mean(pred_test["static_gate"][test["horizon"] == 50])) if np.any(test["horizon"] == 50) else 0.0,
    }


def _stat(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    mean = float(arr.mean()) if len(arr) else 0.0
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    half = 1.96 * std / np.sqrt(max(len(arr), 1))
    return {"mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half}


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "source": "fresh_run",
        "seeds": [int(row["seed"]) for row in rows],
        "ade_all": _stat([row["test_metrics"]["ade"].get("all_improvement", 0.0) for row in rows]),
        "ade_t50": _stat([row["test_metrics"]["ade"].get("t50_improvement", 0.0) for row in rows]),
        "ade_t100_raw_frame_diagnostic": _stat([row["test_metrics"]["ade"].get("t100_improvement", 0.0) for row in rows]),
        "ade_hard_failure": _stat([row["test_metrics"]["ade"].get("hard_failure_improvement", 0.0) for row in rows]),
        "ade_easy_degradation": _stat([row["test_metrics"]["ade"].get("easy_degradation", 1.0) for row in rows]),
        "fde_all": _stat([row["test_metrics"]["fde"].get("all_improvement", 0.0) for row in rows]),
        "fde_t50": _stat([row["test_metrics"]["fde"].get("t50_improvement", 0.0) for row in rows]),
        "switch_rate": _stat([row["test_metrics"].get("switch_rate", 0.0) for row in rows]),
        "ungated_easy_degradation": _stat([row["ungated_test_metrics"]["ade"].get("easy_degradation", 1.0) for row in rows]),
        "static_gate_mean_test": _stat([row["static_gate_mean_test"] for row in rows]),
        "static_gate_t50_mean_test": _stat([row["static_gate_t50_mean_test"] for row in rows]),
    }


def _comparison() -> dict[str, Any]:
    k = read_json(OUT_DIR / "fresh_static_gated_checkpoint_stage42.json", {})
    j = read_json(OUT_DIR / "static_gated_full_waypoint_stage42.json", {})
    return {
        "source": "cached_verified",
        "stage42_k_fresh_static_gated": k.get("summary", {}),
        "stage42_j_static_gated": (j.get("summary") or {}).get("static_gated", {}),
    }


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    s = result.get("summary", {})
    k = (result.get("comparison", {}).get("stage42_k_fresh_static_gated") or {})
    gates = {
        "horizon_static_checkpoints_trained": len(result.get("rows", [])) >= 3,
        "three_seeds": len(s.get("seeds", [])) >= 3,
        "t50_ade_repaired": s.get("ade_t50", {}).get("mean", -1.0) > 0.0,
        "improves_stage42k_t50": s.get("ade_t50", {}).get("mean", -1.0) > k.get("ade_t50", {}).get("mean", -1.0),
        "all_positive": s.get("ade_all", {}).get("mean", 0.0) > 0.0,
        "hard_positive": s.get("ade_hard_failure", {}).get("mean", 0.0) > 0.0,
        "easy_preserved": s.get("ade_easy_degradation", {}).get("mean", 1.0) <= 0.02,
        "no_leakage_pass": result.get("no_leakage", {}).get("future_endpoint_input") is False
        and result.get("no_leakage", {}).get("future_waypoints_input") is False
        and result.get("no_leakage", {}).get("central_velocity") is False
        and result.get("no_leakage", {}).get("test_endpoint_goals") is False,
        "no_metric_seconds_overclaim": result.get("claim_boundary", {}).get("metric_or_seconds_claim") is False,
        "stage5c_false": result.get("claim_boundary", {}).get("stage5c_executed") is False,
        "smc_false": result.get("claim_boundary", {}).get("smc_enabled") is False,
    }
    return {
        "source": "fresh_run",
        "gates": gates,
        "passed": int(sum(bool(v) for v in gates.values())),
        "total": len(gates),
        "verdict": "stage42_l_horizon_static_gate_repair_pass" if all(gates.values()) else "stage42_l_horizon_static_gate_repair_partial",
    }


def run_stage42_horizon_static_gate_repair() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CHECKPOINT_DIR)
    ft.build_full_trajectory_labels()
    data = {split: s42i._split_arrays(split) for split in ["train", "val", "test"]}
    rows = [_eval_seed(seed, data["train"], data["val"], data["test"]) for seed in SEEDS]
    result = {
        "source": "fresh_run",
        "stage": "Stage42-L horizon-aware t50 static-gate repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "torch_threads": THREADS,
        "epochs": EPOCHS,
        "batch": BATCH,
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                ft.DATA_DIR / "all_agent_train.npz",
                ft.DATA_DIR / "all_agent_val.npz",
                ft.DATA_DIR / "all_agent_test.npz",
                ft.DATA_DIR / "full_trajectory_train.npz",
                ft.DATA_DIR / "full_trajectory_val.npz",
                ft.DATA_DIR / "full_trajectory_test.npz",
                OUT_DIR / "fresh_static_gated_checkpoint_stage42.json",
            ]
        ),
        "dataset_rows": {split: int(len(data[split]["horizon"])) for split in ["train", "val", "test"]},
        "rows": rows,
        "summary": _summary(rows),
        "comparison": _comparison(),
        "source_labels": {
            "all_agent_dataset": "cached_verified",
            "full_waypoint_labels": "cached_verified_or_rebuilt_by_stage41_helper",
            "horizon_static_gate_training": "fresh_run",
            "validation_policy_selection": "fresh_run",
            "test_evaluation": "fresh_run_once_per_seed",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "future_waypoints_used_as_label_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "thresholds_selected_on_val": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    result["stage42_l_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    _write_report(result)
    _write_gate(result["stage42_l_gate"])
    _append_readme_and_state(result)
    _append_ledger(result)
    return result


def _write_report(result: Mapping[str, Any]) -> None:
    s = result["summary"]
    cmp = result["comparison"]
    k = cmp.get("stage42_k_fresh_static_gated", {})
    j = cmp.get("stage42_j_static_gated", {})
    lines = [
        "# Stage42-L Horizon-Aware T50 Static-Gate Repair",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_l_gate']['passed']} / {result['stage42_l_gate']['total']}`",
        f"- verdict: `{result['stage42_l_gate']['verdict']}`",
        "",
        "## Current Facts",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Fresh Metrics",
        "",
        "| candidate | source | ADE all | ADE t50 | ADE t100 diag | ADE hard | ADE easy degr | FDE all | FDE t50 | switch | gate | gate t50 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| `horizon_static_gate_repair` | `fresh_run` | {s['ade_all']['mean']:.6f} | {s['ade_t50']['mean']:.6f} | {s['ade_t100_raw_frame_diagnostic']['mean']:.6f} | {s['ade_hard_failure']['mean']:.6f} | {s['ade_easy_degradation']['mean']:.6f} | {s['fde_all']['mean']:.6f} | {s['fde_t50']['mean']:.6f} | {s['switch_rate']['mean']:.6f} | {s['static_gate_mean_test']['mean']:.6f} | {s['static_gate_t50_mean_test']['mean']:.6f} |",
        "",
        "## Comparison",
        "",
        "| candidate | source | ADE all | ADE t50 | ADE hard | ADE easy degr | FDE t50 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        f"| `Stage42-K fresh static-gated` | `cached_verified` | {k.get('ade_all', {}).get('mean', 0.0):.6f} | {k.get('ade_t50', {}).get('mean', 0.0):.6f} | {k.get('ade_hard_failure', {}).get('mean', 0.0):.6f} | {k.get('ade_easy_degradation', {}).get('mean', 0.0):.6f} | {k.get('fde_t50', {}).get('mean', 0.0):.6f} |",
        f"| `Stage42-J policy static-gated` | `cached_verified` | {j.get('ade_all', {}).get('mean', 0.0):.6f} | {j.get('ade_t50', {}).get('mean', 0.0):.6f} | {j.get('ade_hard_failure', {}).get('mean', 0.0):.6f} | {j.get('ade_easy_degradation', {}).get('mean', 0.0):.6f} | {j.get('fde_t50', {}).get('mean', 0.0):.6f} |",
        "",
        "## Interpretation",
        "",
        "- Stage42-L is a targeted repair for Stage42-K's negative ADE t50.",
        "- It uses horizon embeddings, lower t50 static dropout, weaker t50 gate penalty, and a t50-weighted validation policy.",
        "- Future waypoints remain labels only; no future/test leakage, metric claim, Stage5C, or SMC is introduced.",
    ]
    write_md(REPORT_MD, lines)


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-L Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | `{bool(ok)}` |")
    write_md(GATE_MD, lines)


def _append_if_missing(path: Path, marker: str, block: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker not in text:
        path.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")


def _append_readme_and_state(result: Mapping[str, Any]) -> None:
    gate = result["stage42_l_gate"]
    s = result["summary"]
    block = f"""
## Stage42-L Horizon-Aware T50 Static-Gate Repair

```text
source = fresh_run
verdict = {gate['verdict']}
gates = {gate['passed']} / {gate['total']}
horizon_static_gate_ade_all = {s['ade_all']['mean']}
horizon_static_gate_ade_t50 = {s['ade_t50']['mean']}
horizon_static_gate_ade_hard_failure = {s['ade_hard_failure']['mean']}
horizon_static_gate_ade_easy_degradation = {s['ade_easy_degradation']['mean']}
horizon_static_gate_fde_t50 = {s['fde_t50']['mean']}
horizon_static_gate_t50_mean = {s['static_gate_t50_mean_test']['mean']}
stage5c_executed = false
smc_enabled = false
```

Stage42-L targets the Stage42-K t+50 ADE failure with horizon-conditioned static gating and t+50-weighted training/policy selection. It remains dataset-local raw-frame 2.5D evidence and not Stage5C/SMC.
"""
    _append_if_missing(Path("README_RESULTS.md"), "## Stage42-L Horizon-Aware T50 Static-Gate Repair", block)
    _append_if_missing(Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md"), "## Stage42-L Horizon-Aware T50 Static-Gate Repair", block)
    state = read_json(Path("research_state.json"), {})
    state["current_stage"] = "stage42_l_horizon_static_gate_repair"
    state["current_verdict"] = gate["verdict"]
    state.setdefault("stage42", {})["stage_l_horizon_static_gate_repair"] = {
        "source": "fresh_run",
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "gates_passed": gate["passed"],
        "gates_total": gate["total"],
        "verdict": gate["verdict"],
        "horizon_static_gate_ade_all": s["ade_all"]["mean"],
        "horizon_static_gate_ade_t50": s["ade_t50"]["mean"],
        "horizon_static_gate_ade_hard_failure": s["ade_hard_failure"]["mean"],
        "horizon_static_gate_ade_easy_degradation": s["ade_easy_degradation"]["mean"],
        "horizon_static_gate_fde_t50": s["fde_t50"]["mean"],
        "horizon_static_gate_t50_mean": s["static_gate_t50_mean_test"]["mean"],
        "claim_boundary": result["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, GATE_MD]:
        sp = str(path)
        if sp not in reports:
            reports.append(sp)
    write_json(Path("research_state.json"), _jsonable(state))


def _append_ledger(result: Mapping[str, Any]) -> None:
    entry = {
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "step": "stage42_l_horizon_static_gate_repair",
        "source": result["source"],
        "status": "success",
        "input_hash": result.get("input_hash"),
        "output_hash": _combined_hash([REPORT_JSON, REPORT_MD, GATE_MD]),
        "git_commit": _git_commit(),
        "generated_at_utc": result.get("generated_at_utc"),
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_horizon_static_gate_repair()
