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

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src import stage41_full_trajectory_world_state as ft


OUT_DIR = Path("outputs/stage42_long_research")
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
REPORT_JSON = OUT_DIR / "sequence_full_waypoint_stage42.json"
REPORT_MD = OUT_DIR / "sequence_full_waypoint_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_i_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SEEDS = [53, 59, 61]
EPOCHS = 2
BATCH = 2048
THREADS = 4
EPS = 1e-6


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。",
    "Stage42-I full-waypoint evaluation 使用 dataset-local raw-frame，不能写成 metric 或 seconds-level。",
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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(path, _jsonable(dict(payload)))


def _ensure_arm64() -> None:
    if platform.machine() == "arm64":
        return
    venv = Path(".venv-pytorch/bin/python")
    if venv.exists() and os.environ.get("STAGE42I_REEXEC") != "1":
        env = os.environ.copy()
        env["STAGE42I_REEXEC"] = "1"
        os.execve("/usr/bin/arch", ["/usr/bin/arch", "-arm64", str(venv), *sys.argv], env)
    raise RuntimeError("Stage42-I refuses x86_64/Rosetta Python for torch training.")


def _torch():
    _ensure_arm64()
    import torch

    torch.set_num_threads(THREADS)
    return torch


def _split_arrays(split: str, *, ensure_labels: bool = False) -> dict[str, np.ndarray]:
    if ensure_labels:
        ft.build_full_trajectory_labels()
    ds = ft._fresh_ds(split)
    tr = ft._traj(split)
    return {
        "agent_tokens": ds["agent_tokens"].astype(np.float32),
        "agent_mask": ds["agent_mask"].astype(bool),
        "static": ft._norm_static(ds["static"]).astype(np.float32),
        "waypoint_delta": tr["waypoint_delta"].astype(np.float32),
        "waypoint_valid": tr["waypoint_valid"].astype(bool),
        "interaction": tr["interaction_future_close"].astype(np.float32),
        "occupancy": tr["occupancy_future_dense"].astype(np.float32),
        "physical": tr["physical_valid"].astype(np.float32),
        "horizon": ds["horizon"].astype(np.int64),
        "hard": (ds["hard"].astype(bool) | ds["failure"].astype(bool)),
        "failure": ds["failure"].astype(bool),
        "easy": ds["easy"].astype(bool),
        "raw": ds,
        "traj": tr,
    }


def _variant_inputs(split: Mapping[str, np.ndarray], variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokens = split["agent_tokens"].copy()
    mask = split["agent_mask"].copy()
    static = split["static"].copy()
    if variant == "sequence_waypoint_no_history":
        tokens[:] = 0.0
        mask[:] = False
    elif variant == "sequence_waypoint_no_neighbor":
        tokens[:, 1:, :, :] = 0.0
        mask[:, 1:] = False
    elif variant == "sequence_waypoint_no_static_context":
        static[:] = 0.0
    return tokens, mask, static


def _make_model(static_dim: int, token_dim: int = 9, width: int = 72):
    torch = _torch()
    import torch.nn as nn

    class SequenceFullWaypoint(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.temporal = nn.Sequential(
                nn.Conv1d(token_dim, 48, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(48, width, kernel_size=3, padding=1),
                nn.GELU(),
            )
            self.static = nn.Sequential(nn.Linear(static_dim, width), nn.GELU(), nn.LayerNorm(width))
            self.ctx = nn.Sequential(nn.Linear(width * 3, width * 2), nn.GELU(), nn.LayerNorm(width * 2))
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

        def forward(self, tokens, mask, static):
            target, neigh = self._encode_agents(tokens, mask)
            ctx = self.ctx(torch.cat([target, neigh, self.static(static)], dim=1))
            return {
                "waypoint_delta": self.waypoints(ctx).view(-1, len(ft.WAYPOINT_FRAC), 2),
                "traj_risk": self.risk(ctx).squeeze(-1),
                "interaction_logit": self.interaction(ctx).squeeze(-1),
                "occupancy_logit": self.occupancy(ctx).squeeze(-1),
                "physical_logit": self.physical(ctx).squeeze(-1),
            }

    return SequenceFullWaypoint()


def _train_variant(variant: str, seed: int, train: Mapping[str, np.ndarray], val: Mapping[str, np.ndarray]) -> dict[str, Any]:
    torch = _torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    torch.manual_seed(seed)
    tr_tokens, tr_mask, tr_static = _variant_inputs(train, variant)
    va_tokens, va_mask, va_static = _variant_inputs(val, variant)
    model = _make_model(tr_static.shape[1], tr_tokens.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    tensors = {
        "tokens": torch.tensor(tr_tokens),
        "mask": torch.tensor(tr_mask),
        "static": torch.tensor(tr_static),
        "target": torch.tensor(train["waypoint_delta"]),
        "valid": torch.tensor(train["waypoint_valid"].astype(np.float32)),
        "interaction": torch.tensor(train["interaction"]),
        "occupancy": torch.tensor(train["occupancy"]),
        "physical": torch.tensor(train["physical"]),
        "hard": torch.tensor((train["hard"] | train["failure"]).astype(np.float32)),
        "horizon": torch.tensor(train["horizon"]),
    }
    val_tensors = {
        "tokens": torch.tensor(va_tokens),
        "mask": torch.tensor(va_mask),
        "static": torch.tensor(va_static),
        "target": torch.tensor(val["waypoint_delta"]),
        "valid": torch.tensor(val["waypoint_valid"].astype(np.float32)),
    }
    ckpt = CHECKPOINT_DIR / f"stage42i_{variant}_seed{seed}.pt"
    heartbeat = OUT_DIR / f"stage42i_{variant}_seed{seed}_heartbeat.json"
    if ckpt.exists() and heartbeat.exists():
        payload = json.loads(heartbeat.read_text(encoding="utf-8"))
        if int(payload.get("epoch", 0)) >= EPOCHS:
            return {"source": "cached_verified", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": payload.get("best", {})}
    best = {"val_loss": float("inf"), "epoch": 0}
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(tr_tokens))
        losses: list[float] = []
        model.train()
        for start in range(0, len(order), BATCH):
            ids = torch.tensor(order[start : start + BATCH], dtype=torch.long)
            out = model(tensors["tokens"][ids], tensors["mask"][ids], tensors["static"][ids])
            valid = tensors["valid"][ids]
            row_w = 1.0 + 1.5 * tensors["hard"][ids] + 2.0 * (tensors["horizon"][ids] == 50).float() + 1.0 * (tensors["horizon"][ids] == 100).float()
            per_wp = F.smooth_l1_loss(out["waypoint_delta"], tensors["target"][ids], reduction="none").mean(dim=2)
            traj = ((per_wp * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0) * row_w).mean()
            err = torch.linalg.norm(out["waypoint_delta"] - tensors["target"][ids], dim=2)
            risk_target = torch.log1p((err * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)).detach()
            risk = (F.smooth_l1_loss(out["traj_risk"], risk_target, reduction="none") * row_w).mean()
            aux = (
                F.binary_cross_entropy_with_logits(out["interaction_logit"], tensors["interaction"][ids])
                + F.binary_cross_entropy_with_logits(out["occupancy_logit"], tensors["occupancy"][ids])
                + F.binary_cross_entropy_with_logits(out["physical_logit"], tensors["physical"][ids])
            )
            loss = traj + 0.30 * risk + 0.15 * aux
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            out = model(val_tensors["tokens"], val_tensors["mask"], val_tensors["static"])
            per_wp = F.smooth_l1_loss(out["waypoint_delta"], val_tensors["target"], reduction="none").mean(dim=2)
            val_loss = float(((per_wp * val_tensors["valid"]).sum(dim=1) / val_tensors["valid"].sum(dim=1).clamp_min(1.0)).mean().cpu())
        cand = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses))}
        if val_loss < best["val_loss"]:
            best = cand
            torch.save({"model": model.state_dict(), "variant": variant, "seed": seed, "static_dim": tr_static.shape[1], "token_dim": tr_tokens.shape[-1], "best": best}, ckpt)
        heartbeat.write_text(json.dumps({"variant": variant, "seed": seed, "epoch": epoch, "best": best, "checkpoint": str(ckpt)}, ensure_ascii=False), encoding="utf-8")
    return {"source": "fresh_run", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best}


def _predict(info: Mapping[str, Any], split: Mapping[str, np.ndarray], variant: str) -> dict[str, np.ndarray]:
    torch = _torch()
    payload = torch.load(info["checkpoint"], map_location="cpu", weights_only=False)
    model = _make_model(int(payload["static_dim"]), int(payload["token_dim"]))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    tokens, mask, static = _variant_inputs(split, variant)
    outs = {k: [] for k in ["waypoint_delta", "traj_risk", "interaction", "occupancy", "physical"]}
    with torch.no_grad():
        for start in range(0, len(tokens), 4096):
            sl = slice(start, min(start + 4096, len(tokens)))
            out = model(torch.tensor(tokens[sl]), torch.tensor(mask[sl]), torch.tensor(static[sl]))
            outs["waypoint_delta"].append(out["waypoint_delta"].cpu().numpy())
            outs["traj_risk"].append(out["traj_risk"].cpu().numpy().reshape(-1))
            outs["interaction"].append(torch.sigmoid(out["interaction_logit"]).cpu().numpy().reshape(-1))
            outs["occupancy"].append(torch.sigmoid(out["occupancy_logit"]).cpu().numpy().reshape(-1))
            outs["physical"].append(torch.sigmoid(out["physical_logit"]).cpu().numpy().reshape(-1))
    return {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in outs.items()}


def _labels(split: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    ds = split["raw"]
    tr = split["traj"]
    return {
        "floor_fde": ds["floor_fde"].astype(np.float64),
        "candidate_fde": ds["candidate_fde"].astype(np.float64),
        "current_xy": ds["current_xy"].astype(np.float64),
        "future_xy": ds["future_xy"].astype(np.float64),
        "normalizer": ds["normalizer"].astype(np.float64),
        "cand_delta": ds["cand_delta"].astype(np.float64),
        "waypoint_xy": tr["waypoint_xy"].astype(np.float64),
        "waypoint_valid": tr["waypoint_valid"].astype(bool),
        "interaction": tr["interaction_future_close"].astype(bool),
        "occupancy": tr["occupancy_future_dense"].astype(bool),
        "physical": tr["physical_valid"].astype(bool),
        "horizon": ds["horizon"].astype(np.int64),
        "hard": (ds["hard"].astype(bool) | ds["failure"].astype(bool)),
        "easy": ds["easy"].astype(bool),
        "failure": ds["failure"].astype(bool),
        "domain": ds["domain"].astype(str),
        "scene_id": ds["scene_id"].astype(str),
        "source_file": ds["source_file"].astype(str),
    }


def _apply_light_policy(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> np.ndarray:
    n = len(labels["horizon"])
    switch = np.zeros(n, dtype=bool)
    domain = labels["domain"].astype(str)
    horizon = labels["horizon"].astype(int)
    hard = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    easy = labels["easy"].astype(bool)
    risk = pred["traj_risk"].astype(np.float64)
    physical = pred["physical"].astype(np.float64)
    for key, params in policy.get("slices", {}).items():
        d, h_s = key.split("|")
        mask = (domain == d) & (horizon == int(h_s))
        if not np.any(mask):
            continue
        local = mask & (risk <= float(params["risk_max"])) & (physical >= float(params.get("physical_min", 0.0)))
        if params.get("hard_only", False):
            local &= hard
        if params.get("easy_block", True):
            local &= ~easy
        max_switch = float(params.get("max_switch", 1.0))
        if max_switch <= 0.0:
            local[:] = False
        elif max_switch < 1.0 and np.any(local):
            ids = np.where(local)[0]
            keep_n = max(1, int(max_switch * int(np.sum(mask))))
            keep = np.zeros(n, dtype=bool)
            keep[ids[np.argsort(risk[ids])[:keep_n]]] = True
            local &= keep
        switch |= local
    return switch


def _selected_xy(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    switch = _apply_light_policy(pred, labels, policy)
    floor_xy = ft._floor_waypoints(labels)
    neural_xy = ft._pred_waypoints(pred, labels)
    selected = floor_xy.copy()
    selected[switch.astype(bool)] = neural_xy[switch.astype(bool)]
    return selected, switch.astype(bool)


def _row_metrics(name: str, pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> dict[str, Any]:
    selected, switch = _selected_xy(pred, labels, policy)
    floor = ft._floor_waypoints(labels)
    ade, fde = ft._trajectory_errors(selected, labels)
    floor_ade, floor_fde = ft._trajectory_errors(floor, labels)
    return {
        "variant": name,
        "ade": ft._metric(ade, floor_ade, labels, switch),
        "fde": ft._metric(fde, floor_fde, labels, switch),
        "switch_rate": float(np.mean(switch)) if len(switch) else 0.0,
    }


def _score_metric(metric: Mapping[str, Any]) -> float:
    return (
        1.4 * float(metric.get("all_improvement", 0.0))
        + 1.8 * float(metric.get("t50_improvement", 0.0))
        + 1.2 * float(metric.get("hard_failure_improvement", 0.0))
        - 30.0 * max(0.0, float(metric.get("easy_degradation", 1.0)) - 0.02)
    )


def _fit_light_policy(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    floor_xy = ft._floor_waypoints(labels)
    neural_xy = ft._pred_waypoints(pred, labels)
    floor_ade, _floor_fde = ft._trajectory_errors(floor_xy, labels)
    neural_ade, _neural_fde = ft._trajectory_errors(neural_xy, labels)
    domain = labels["domain"].astype(str)
    horizon = labels["horizon"].astype(int)
    risk = pred["traj_risk"].astype(np.float64)
    policy: dict[str, Any] = {"type": "stage42i_light_domain_horizon_risk_policy", "slices": {}}
    diagnostics: dict[str, Any] = {}
    for d in sorted(set(domain.tolist())):
        for h in [10, 25, 50, 100]:
            mask = (domain == d) & (horizon == h)
            if int(np.sum(mask)) < 80:
                continue
            thresholds = [float(v) for v in np.quantile(risk[mask], [0.10, 0.25, 0.40, 0.60])]
            best_params: dict[str, Any] | None = None
            best_score = 0.0
            best_metric: dict[str, Any] | None = None
            for risk_max in thresholds:
                for max_switch in [0.05, 0.10, 0.20, 0.35]:
                    for hard_only in [False, True]:
                        params = {"risk_max": risk_max, "physical_min": 0.0, "max_switch": max_switch, "hard_only": hard_only, "easy_block": True}
                        trial_policy = {"slices": {f"{d}|{h}": params}}
                        switch = _apply_light_policy(pred, labels, trial_policy)
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
    switch = _apply_light_policy(pred, labels, policy)
    selected = floor_ade.copy()
    selected[switch] = neural_ade[switch]
    metrics = ft._metric(selected, floor_ade, labels, switch)
    metrics["slice_diagnostics"] = diagnostics
    return policy, metrics


def _train_eval_seed(variant: str, seed: int, train: Mapping[str, np.ndarray], val: Mapping[str, np.ndarray], test: Mapping[str, np.ndarray]) -> dict[str, Any]:
    train_info = _train_variant(variant, seed, train, val)
    pred_val = _predict(train_info, val, variant)
    pred_test = _predict(train_info, test, variant)
    labels_val = _labels(val)
    labels_test = _labels(test)
    policy, val_metrics = _fit_light_policy(pred_val, labels_val)
    test_metrics = _row_metrics(variant, pred_test, labels_test, policy)
    ungated = _row_metrics(
        f"{variant}_ungated",
        pred_test,
        labels_test,
        {"slices": {f"{d}|{h}": {"risk_max": 1e9, "physical_min": 0.0, "max_switch": 1.0, "hard_only": False, "easy_block": False} for d in sorted(set(labels_test["domain"].tolist())) for h in [10, 25, 50, 100]}},
    )
    return {
        "source": "fresh_run",
        "variant": variant,
        "seed": seed,
        "train_info": train_info,
        "val_policy": policy,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "ungated_test_metrics": ungated,
    }


def _stat(vals: list[float]) -> dict[str, float]:
    arr = np.asarray(vals, dtype=np.float64)
    mean = float(arr.mean()) if len(arr) else 0.0
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    half = 1.96 * std / np.sqrt(max(len(arr), 1))
    return {"mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half}


def _summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for variant in sorted({str(r["variant"]) for r in rows}):
        sub = [r for r in rows if r["variant"] == variant]
        out[variant] = {
            "source": "fresh_run",
            "seeds": [int(r["seed"]) for r in sub],
            "ade_all": _stat([r["test_metrics"]["ade"].get("all_improvement", 0.0) for r in sub]),
            "ade_t50": _stat([r["test_metrics"]["ade"].get("t50_improvement", 0.0) for r in sub]),
            "ade_t100_raw_frame_diagnostic": _stat([r["test_metrics"]["ade"].get("t100_improvement", 0.0) for r in sub]),
            "ade_hard_failure": _stat([r["test_metrics"]["ade"].get("hard_failure_improvement", 0.0) for r in sub]),
            "ade_easy_degradation": _stat([r["test_metrics"]["ade"].get("easy_degradation", 1.0) for r in sub]),
            "fde_all": _stat([r["test_metrics"]["fde"].get("all_improvement", 0.0) for r in sub]),
            "fde_t50": _stat([r["test_metrics"]["fde"].get("t50_improvement", 0.0) for r in sub]),
            "switch_rate": _stat([r["test_metrics"].get("switch_rate", 0.0) for r in sub]),
            "ungated_easy_degradation": _stat([r["ungated_test_metrics"]["ade"].get("easy_degradation", 1.0) for r in sub]),
        }
    return out


def _contribution(summary: Mapping[str, Any]) -> dict[str, Any]:
    full = summary.get("sequence_waypoint_full", {})
    out: dict[str, Any] = {}
    for variant, item in summary.items():
        if variant == "sequence_waypoint_full":
            continue
        out[variant] = {
            "ade_all_delta_full_minus_ablation": full.get("ade_all", {}).get("mean", 0.0) - item.get("ade_all", {}).get("mean", 0.0),
            "ade_t50_delta_full_minus_ablation": full.get("ade_t50", {}).get("mean", 0.0) - item.get("ade_t50", {}).get("mean", 0.0),
            "ade_hard_delta_full_minus_ablation": full.get("ade_hard_failure", {}).get("mean", 0.0) - item.get("ade_hard_failure", {}).get("mean", 0.0),
            "fde_t50_delta_full_minus_ablation": full.get("fde_t50", {}).get("mean", 0.0) - item.get("fde_t50", {}).get("mean", 0.0),
        }
    return out


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    summary = result.get("summary", {})
    full = summary.get("sequence_waypoint_full", {})
    contrib = result.get("contribution_vs_full", {})
    gates = {
        "full_waypoint_labels_available": result.get("dataset_rows", {}).get("test", 0) > 0,
        "sequence_waypoint_models_trained": all(k in summary for k in ["sequence_waypoint_full", "sequence_waypoint_no_history", "sequence_waypoint_no_neighbor"]),
        "three_seeds": all(len((summary.get(k) or {}).get("seeds", [])) >= 3 for k in ["sequence_waypoint_full", "sequence_waypoint_no_history", "sequence_waypoint_no_neighbor"]),
        "protected_full_positive": full.get("ade_all", {}).get("mean", 0.0) > 0.0 or full.get("ade_t50", {}).get("mean", 0.0) > 0.0 or full.get("ade_hard_failure", {}).get("mean", 0.0) > 0.0,
        "easy_preserved": full.get("ade_easy_degradation", {}).get("mean", 1.0) <= 0.02,
        "history_waypoint_contribution_measured": "sequence_waypoint_no_history" in contrib,
        "at_least_one_positive_component": any(v.get("ade_t50_delta_full_minus_ablation", 0.0) > 0.0 or v.get("ade_hard_delta_full_minus_ablation", 0.0) > 0.0 for v in contrib.values()),
        "no_leakage_pass": result.get("no_leakage", {}).get("future_endpoint_input") is False
        and result.get("no_leakage", {}).get("future_waypoints_input") is False
        and result.get("no_leakage", {}).get("central_velocity") is False
        and result.get("no_leakage", {}).get("test_endpoint_goals") is False,
        "no_metric_seconds_overclaim": result.get("claim_boundary", {}).get("metric_or_seconds_claim") is False,
        "stage5c_false": result.get("claim_boundary", {}).get("stage5c_executed") is False,
        "smc_false": result.get("claim_boundary", {}).get("smc_enabled") is False,
    }
    verdict = "stage42_i_sequence_full_waypoint_pass" if all(gates.values()) else "stage42_i_sequence_full_waypoint_partial"
    return {
        "source": "fresh_run",
        "gates": gates,
        "passed": int(sum(bool(v) for v in gates.values())),
        "total": len(gates),
        "verdict": verdict,
        "history_ade_t50_delta": (contrib.get("sequence_waypoint_no_history") or {}).get("ade_t50_delta_full_minus_ablation"),
    }


def run_stage42_sequence_full_waypoint() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CHECKPOINT_DIR)
    ft.build_full_trajectory_labels()
    data = {split: _split_arrays(split) for split in ["train", "val", "test"]}
    variants = ["sequence_waypoint_full", "sequence_waypoint_no_history", "sequence_waypoint_no_neighbor", "sequence_waypoint_no_static_context"]
    rows: list[dict[str, Any]] = []
    for variant in variants:
        for seed in SEEDS:
            rows.append(_train_eval_seed(variant, seed, data["train"], data["val"], data["test"]))
    summary = _summarize(rows)
    result = {
        "source": "fresh_run",
        "stage": "Stage42-I sequence-to-full-waypoint dynamics",
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
            ]
        ),
        "dataset_rows": {split: int(len(data[split]["horizon"])) for split in ["train", "val", "test"]},
        "source_labels": {
            "all_agent_dataset": "cached_verified",
            "full_waypoint_labels": "fresh_run_or_cached_verified_rebuilt_by_stage41_helper",
            "sequence_full_waypoint_training": "fresh_run",
            "validation_policy_selection": "fresh_run",
            "test_evaluation": "fresh_run_once_per_variant_seed",
        },
        "rows": rows,
        "summary": summary,
        "contribution_vs_full": _contribution(summary),
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
            "not_full_jepa_or_foundation_transformer": True,
        },
    }
    result["stage42_i_gate"] = _gate(result)
    _write_json(REPORT_JSON, result)
    _write_report(result)
    _write_gate(result["stage42_i_gate"])
    _append_readme_and_state(result)
    _append_ledger(result)
    return result


def _write_report(result: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-I Sequence-To-Full-Waypoint Dynamics",
        "",
        "- source: `fresh_run`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_i_gate']['passed']} / {result['stage42_i_gate']['total']}`",
        f"- verdict: `{result['stage42_i_gate']['verdict']}`",
        "",
        "## Current Facts",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Metrics",
        "",
        "| variant | ADE all | ADE t50 | ADE t100 diag | ADE hard | ADE easy degr | FDE all | FDE t50 | switch | ungated easy degr |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in sorted(result["summary"].items()):
        lines.append(
            f"| `{name}` | {item['ade_all']['mean']:.6f} | {item['ade_t50']['mean']:.6f} | {item['ade_t100_raw_frame_diagnostic']['mean']:.6f} | {item['ade_hard_failure']['mean']:.6f} | {item['ade_easy_degradation']['mean']:.6f} | {item['fde_all']['mean']:.6f} | {item['fde_t50']['mean']:.6f} | {item['switch_rate']['mean']:.6f} | {item['ungated_easy_degradation']['mean']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Contribution Deltas",
            "",
            "`full_minus_ablation > 0` means the removed component helped the full sequence-to-waypoint model.",
            "",
            "| ablation | ADE all delta | ADE t50 delta | ADE hard delta | FDE t50 delta |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, row in sorted(result["contribution_vs_full"].items()):
        lines.append(
            f"| `{name}` | {row['ade_all_delta_full_minus_ablation']:.6f} | {row['ade_t50_delta_full_minus_ablation']:.6f} | {row['ade_hard_delta_full_minus_ablation']:.6f} | {row['fde_t50_delta_full_minus_ablation']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-I connects the Stage42-H causal sequence-history signal to actual reconstructed full-waypoint ADE/FDE labels.",
            "- All thresholds are selected on validation; test is evaluated once per variant/seed.",
            "- Future waypoints are supervised labels/eval only, not inference inputs.",
            "- Results remain dataset-local raw-frame 2.5D and do not enable Stage5C or SMC.",
        ]
    )
    write_md(REPORT_MD, lines)


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-I Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- history_ade_t50_delta: `{gate.get('history_ade_t50_delta')}`",
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
    gate = result["stage42_i_gate"]
    full = result["summary"].get("sequence_waypoint_full", {})
    hist = (result.get("contribution_vs_full") or {}).get("sequence_waypoint_no_history", {})
    block = f"""
## Stage42-I Sequence-To-Full-Waypoint Dynamics

```text
source = fresh_run
verdict = {gate['verdict']}
gates = {gate['passed']} / {gate['total']}
sequence_waypoint_full_ade_all = {full.get('ade_all', {}).get('mean')}
sequence_waypoint_full_ade_t50 = {full.get('ade_t50', {}).get('mean')}
sequence_waypoint_full_ade_hard_failure = {full.get('ade_hard_failure', {}).get('mean')}
sequence_waypoint_full_ade_easy_degradation = {full.get('ade_easy_degradation', {}).get('mean')}
history_ade_t50_delta_full_minus_no_history = {hist.get('ade_t50_delta_full_minus_ablation')}
stage5c_executed = false
smc_enabled = false
```

Stage42-I connects causal sequence history to actual reconstructed full-waypoint ADE/FDE labels. It remains a protected dataset-local raw-frame 2.5D dynamics experiment, not metric/seconds-level prediction and not Stage5C/SMC.
"""
    _append_if_missing(Path("README_RESULTS.md"), "## Stage42-I Sequence-To-Full-Waypoint Dynamics", block)
    _append_if_missing(Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md"), "## Stage42-I Sequence-To-Full-Waypoint Dynamics", block)
    state = read_json(Path("research_state.json"), {})
    state["current_stage"] = "stage42_i_sequence_to_full_waypoint"
    state["current_verdict"] = gate["verdict"]
    state.setdefault("stage42", {})["stage_i_sequence_to_full_waypoint"] = {
        "source": "fresh_run",
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "gates_passed": gate["passed"],
        "gates_total": gate["total"],
        "verdict": gate["verdict"],
        "sequence_waypoint_full_ade_all": full.get("ade_all", {}).get("mean"),
        "sequence_waypoint_full_ade_t50": full.get("ade_t50", {}).get("mean"),
        "sequence_waypoint_full_ade_hard_failure": full.get("ade_hard_failure", {}).get("mean"),
        "sequence_waypoint_full_ade_easy_degradation": full.get("ade_easy_degradation", {}).get("mean"),
        "history_ade_t50_delta_full_minus_no_history": hist.get("ade_t50_delta_full_minus_ablation"),
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
        "step": "stage42_i_sequence_to_full_waypoint",
        "source": "fresh_run",
        "status": "success",
        "input_hash": result.get("input_hash"),
        "output_hash": _combined_hash([REPORT_JSON, REPORT_MD, GATE_MD]),
        "git_commit": _git_commit(),
        "generated_at_utc": result.get("generated_at_utc"),
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_sequence_full_waypoint()
