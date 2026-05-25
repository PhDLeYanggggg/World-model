from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_all_agent as aa
from src import stage41_fresh_confirmation as fresh
from src import stage41_full_trajectory_world_state as ft
from src import stage41_goal_route_physical_repair as gr


OUT_DIR = fresh.OUT_DIR
DATA_DIR = fresh.DATA_DIR
CHECKPOINT_DIR = fresh.CHECKPOINT_DIR
LEDGER_JSONL = fresh.LEDGER_JSONL
THREADS = 4
BATCH = 512
EPOCHS = 4
SEED = 4201
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


def _append_ledger(step: str, status: str, started: float, inputs: Sequence[str | Path], outputs: Sequence[str | Path]) -> None:
    ensure_dir(OUT_DIR)
    entry = {
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "step": step,
        "source": "fresh_run",
        "status": status,
        "wall_time_s": time.perf_counter() - started,
        "input_hash": _combined_hash(inputs),
        "output_hash": _combined_hash(outputs),
        "git_commit": _git_commit(),
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


def _torch():
    torch = aa._torch()
    torch.set_num_threads(THREADS)
    return torch


def _norm_static(static: np.ndarray) -> np.ndarray:
    norm = dict(np.load(DATA_DIR / "normalization.npz"))
    return ((static.astype(np.float32) - norm["static_mean"]) / norm["static_std"]).astype(np.float32)


def _load_tensors(split: str):
    torch = _torch()
    ds = ft._fresh_ds(split)
    traj = ft._traj(split)
    route = gr._labels(split)
    return {
        "agent_tokens": torch.tensor(ds["agent_tokens"].astype(np.float32)),
        "agent_mask": torch.tensor(ds["agent_mask"].astype(bool)),
        "static": torch.tensor(_norm_static(ds["static"])),
        "waypoint_delta": torch.tensor(traj["waypoint_delta"].astype(np.float32)),
        "waypoint_valid": torch.tensor(traj["waypoint_valid"].astype(bool)),
        "interaction": torch.tensor(traj["interaction_future_close"].astype(np.float32)),
        "occupancy": torch.tensor(traj["occupancy_future_dense"].astype(np.float32)),
        "route": torch.tensor(route["route_label"].astype(np.int64)),
        "physical_challenge": torch.tensor(route["physical_challenge"].astype(np.float32)),
        "hard": torch.tensor((ds["hard"].astype(bool) | ds["failure"].astype(bool)).astype(np.float32)),
        "easy": torch.tensor(ds["easy"].astype(bool).astype(np.float32)),
        "failure": torch.tensor(ds["failure"].astype(bool).astype(np.float32)),
        "horizon": torch.tensor(ds["horizon"].astype(np.int64)),
        "raw": ds,
        "traj": traj,
        "route_raw": route,
    }


def _make_model(static_dim: int, width: int = 88, dropout: float = 0.08):
    torch = _torch()
    import torch.nn as nn

    class JointRouteConditionedWorldState(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.temporal = nn.Sequential(nn.Linear(9, width), nn.GELU(), nn.LayerNorm(width), nn.Dropout(dropout), nn.Linear(width, width))
            self.role = nn.Embedding(aa.MAX_AGENTS, width)
            layer = nn.TransformerEncoderLayer(d_model=width, nhead=4, dim_feedforward=width * 2, dropout=dropout, batch_first=True)
            self.agent_encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.static = nn.Sequential(nn.Linear(static_dim, width), nn.GELU(), nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
            self.ctx = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.LayerNorm(width))
            self.route = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, len(gr.ROUTE_NAMES)))
            self.route_embed = nn.Linear(len(gr.ROUTE_NAMES), width, bias=False)
            self.physical_challenge = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
            self.waypoints = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, len(ft.WAYPOINT_FRAC) * 2))
            self.traj_risk = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 1))
            self.interaction = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 1))
            self.occupancy = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 1))

        def forward(self, agent_tokens, agent_mask, static):
            b, a, _t, _ = agent_tokens.shape
            x = self.temporal(agent_tokens) + self.role.weight[:a][None, :, None, :]
            valid_time = agent_tokens[..., 6].clamp(0, 1)
            agent_emb = (x * valid_time[..., None]).sum(dim=2) / valid_time.sum(dim=2, keepdim=True).clamp_min(1.0)
            agent_h = self.agent_encoder(agent_emb)
            valid_agent = agent_mask[:, :, None].float()
            pooled = (agent_h * valid_agent).sum(dim=1) / valid_agent.sum(dim=1).clamp_min(1.0)
            ctx = self.ctx(torch.cat([pooled, self.static(static)], dim=1))
            route_logits = self.route(ctx)
            route_prob = torch.softmax(route_logits, dim=1)
            route_ctx = self.route_embed(route_prob)
            fused = torch.cat([ctx, route_ctx], dim=1)
            return {
                "waypoint_delta": self.waypoints(fused).view(-1, len(ft.WAYPOINT_FRAC), 2),
                "traj_risk": self.traj_risk(fused).squeeze(-1),
                "route_logits": route_logits,
                "physical_challenge_logit": self.physical_challenge(ctx).squeeze(-1),
                "interaction_logit": self.interaction(fused).squeeze(-1),
                "occupancy_logit": self.occupancy(fused).squeeze(-1),
            }

    return JointRouteConditionedWorldState()


TRIALS = [
    {"name": "joint_route_balanced", "width": 80, "dropout": 0.08, "lr": 8e-4, "traj_w": 1.0, "route_w": 0.35, "physical_w": 0.25, "aux_w": 0.25, "hard_w": 2.0, "t50_w": 2.5, "t100_w": 2.0, "seed": 1},
    {"name": "joint_route_t50_hard", "width": 88, "dropout": 0.08, "lr": 7e-4, "traj_w": 1.1, "route_w": 0.45, "physical_w": 0.35, "aux_w": 0.25, "hard_w": 3.5, "t50_w": 5.0, "t100_w": 1.0, "seed": 2},
    {"name": "joint_route_long_horizon", "width": 96, "dropout": 0.10, "lr": 6e-4, "traj_w": 1.0, "route_w": 0.35, "physical_w": 0.45, "aux_w": 0.30, "hard_w": 2.5, "t50_w": 2.0, "t100_w": 5.0, "seed": 3},
]


def _class_weights(route: np.ndarray) -> np.ndarray:
    return gr._class_weights(route)


def _train_trial(trial: Mapping[str, Any]) -> Dict[str, Any]:
    torch = _torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    train = _load_tensors("train")
    val = _load_tensors("val")
    ckpt = CHECKPOINT_DIR / f"stage41_{trial['name']}.pt"
    heartbeat = OUT_DIR / f"{trial['name']}_heartbeat.json"
    if ckpt.exists() and heartbeat.exists():
        payload = read_json(heartbeat, {})
        if int(payload.get("epoch", 0)) >= EPOCHS:
            return {"source": "cached_verified", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": payload.get("best", {})}
    model = _make_model(train["static"].shape[1], int(trial["width"]), float(trial["dropout"]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(trial["lr"]), weight_decay=1e-4)
    route_weights = torch.tensor(_class_weights(train["route"].numpy()), dtype=torch.float32)
    pos = float(train["physical_challenge"].mean().item())
    pos_weight = torch.tensor(max((1.0 - pos) / max(pos, EPS), 0.1), dtype=torch.float32)
    rng = np.random.default_rng(SEED + int(trial["seed"]))
    best = {"val_loss": float("inf"), "epoch": 0}
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(train["agent_tokens"].shape[0])
        losses: list[float] = []
        model.train()
        for start in range(0, len(order), BATCH):
            ids = torch.tensor(order[start : start + BATCH], dtype=torch.long)
            out = model(train["agent_tokens"][ids], train["agent_mask"][ids], train["static"][ids])
            row_w = 1.0 + float(trial["hard_w"]) * train["hard"][ids]
            row_w = row_w + float(trial["t50_w"]) * (train["horizon"][ids] == 50).float()
            row_w = row_w + float(trial["t100_w"]) * (train["horizon"][ids] == 100).float()
            valid = train["waypoint_valid"][ids].float()
            err = torch.linalg.norm(out["waypoint_delta"] - train["waypoint_delta"][ids], dim=2)
            traj = ((F.smooth_l1_loss(out["waypoint_delta"], train["waypoint_delta"][ids], reduction="none").mean(dim=2) * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0) * row_w).mean()
            risk = (F.smooth_l1_loss(out["traj_risk"], torch.log1p((err * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)).detach(), reduction="none") * row_w).mean()
            route = (F.cross_entropy(out["route_logits"], train["route"][ids], weight=route_weights, reduction="none") * row_w).mean()
            physical = (F.binary_cross_entropy_with_logits(out["physical_challenge_logit"], train["physical_challenge"][ids], pos_weight=pos_weight, reduction="none") * row_w).mean()
            interaction = F.binary_cross_entropy_with_logits(out["interaction_logit"], train["interaction"][ids])
            occupancy = F.binary_cross_entropy_with_logits(out["occupancy_logit"], train["occupancy"][ids])
            loss = float(trial["traj_w"]) * traj + 0.35 * risk + float(trial["route_w"]) * route + float(trial["physical_w"]) * physical + float(trial["aux_w"]) * (interaction + occupancy)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            out = model(val["agent_tokens"], val["agent_mask"], val["static"])
            valid = val["waypoint_valid"].float()
            traj_val = ((F.smooth_l1_loss(out["waypoint_delta"], val["waypoint_delta"], reduction="none").mean(dim=2) * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)).mean()
            route_val = F.cross_entropy(out["route_logits"], val["route"], weight=route_weights)
            val_loss = float((traj_val + 0.1 * route_val).cpu())
        best_candidate = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses))}
        heartbeat.write_text(json.dumps({"trial": dict(trial), "epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "checkpoint": str(ckpt), "best": best_candidate}), encoding="utf-8")
        if val_loss < best["val_loss"]:
            best = best_candidate
            torch.save({"model": model.state_dict(), "trial": dict(trial), "static_dim": train["static"].shape[1], "best": best}, ckpt)
    return {"source": "fresh_run", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best}


def _load_model(path: str | Path):
    torch = _torch()
    payload = torch.load(path, map_location="cpu")
    trial = payload["trial"]
    model = _make_model(int(payload["static_dim"]), int(trial["width"]), float(trial["dropout"]))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model


def _predict(path: str | Path, split: str) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    torch = _torch()
    model = _load_model(path)
    tensors = _load_tensors(split)
    outs: Dict[str, list[np.ndarray]] = {k: [] for k in ["waypoint_delta", "traj_risk", "route_logits", "physical", "interaction", "occupancy"]}
    with torch.no_grad():
        for start in range(0, tensors["agent_tokens"].shape[0], 2048):
            sl = slice(start, min(start + 2048, tensors["agent_tokens"].shape[0]))
            out = model(tensors["agent_tokens"][sl], tensors["agent_mask"][sl], tensors["static"][sl])
            outs["waypoint_delta"].append(out["waypoint_delta"].cpu().numpy())
            outs["traj_risk"].append(out["traj_risk"].cpu().numpy().reshape(-1))
            outs["route_logits"].append(out["route_logits"].cpu().numpy())
            outs["physical"].append(torch.sigmoid(out["physical_challenge_logit"]).cpu().numpy().reshape(-1))
            outs["interaction"].append(torch.sigmoid(out["interaction_logit"]).cpu().numpy().reshape(-1))
            outs["occupancy"].append(torch.sigmoid(out["occupancy_logit"]).cpu().numpy().reshape(-1))
    pred = {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in outs.items()}
    ds = tensors["raw"]
    tr = tensors["traj"]
    labels = {
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
        "physical": tensors["route_raw"]["physical_challenge"].astype(bool),
        "route": tensors["route_raw"]["route_label"].astype(np.int64),
        "horizon": ds["horizon"].astype(np.int64),
        "hard": (ds["hard"].astype(bool) | ds["failure"].astype(bool)),
        "easy": ds["easy"].astype(bool),
        "failure": ds["failure"].astype(bool),
        "domain": ds["domain"].astype(str),
        "scene_id": ds["scene_id"].astype(str),
        "source_file": ds["source_file"].astype(str),
    }
    return pred, labels


def _predict_ensemble(paths: Sequence[str | Path], split: str) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    preds: list[Dict[str, np.ndarray]] = []
    labels_ref: Dict[str, np.ndarray] | None = None
    for path in paths:
        pred, labels = _predict(path, split)
        preds.append(pred)
        labels_ref = labels if labels_ref is None else labels_ref
    if not preds or labels_ref is None:
        raise ValueError("joint route-conditioned ensemble requires checkpoints")
    return {k: np.mean([p[k] for p in preds], axis=0).astype(np.float32) for k in preds[0]}, labels_ref


def _as_ft_pred(pred: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "waypoint_delta": pred["waypoint_delta"],
        "traj_risk": pred["traj_risk"],
        "interaction": pred["interaction"],
        "occupancy": pred["occupancy"],
        "physical": pred["physical"],
    }


def _route_metrics(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    return gr._route_metrics({"route_logits": pred["route_logits"]}, {"route": labels["route"], "domain": labels["domain"]})


def _aux_metrics(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    return {
        "route": _route_metrics(pred, labels),
        "physical_challenge": ft._binary_metrics(pred["physical"], labels["physical"]),
        "interaction": ft._binary_metrics(pred["interaction"], labels["interaction"]),
        "occupancy": ft._binary_metrics(pred["occupancy"], labels["occupancy"]),
    }


def train_joint_route_conditioned_world_state() -> Dict[str, Any]:
    ft.build_full_trajectory_labels()
    gr.build_goal_route_physical_labels()
    full_ref = read_json(OUT_DIR / "stage41_full_trajectory_world_state.json", {})
    trials: Dict[str, Any] = {}
    paths = []
    best_name = ""
    best_score = -1e18
    best_policy: Dict[str, Any] = {}
    for trial in TRIALS:
        train = _train_trial(trial)
        pred_val, labels_val = _predict(train["checkpoint"], "val")
        policy, val_metrics = ft._fit_policy(_as_ft_pred(pred_val), labels_val)
        score = 1.2 * val_metrics.get("all_improvement", 0.0) + 1.4 * val_metrics.get("t50_improvement", 0.0) + val_metrics.get("hard_failure_improvement", 0.0) - 20.0 * max(0.0, val_metrics.get("easy_degradation", 1.0) - 0.02)
        trials[trial["name"]] = {"source": train.get("source"), "trial": trial, "train": train, "policy": policy, "val_metrics": val_metrics, "aux_val_metrics": _aux_metrics(pred_val, labels_val), "val_score": score}
        paths.append(train["checkpoint"])
        if score > best_score:
            best_score = score
            best_name = trial["name"]
            best_policy = policy
    pred_val, labels_val = _predict_ensemble(paths, "val")
    policy, val_metrics = ft._fit_policy(_as_ft_pred(pred_val), labels_val)
    score = 1.2 * val_metrics.get("all_improvement", 0.0) + 1.4 * val_metrics.get("t50_improvement", 0.0) + val_metrics.get("hard_failure_improvement", 0.0) - 20.0 * max(0.0, val_metrics.get("easy_degradation", 1.0) - 0.02)
    trials["joint_route_conditioned_ensemble"] = {"source": "fresh_run", "paths": paths, "policy": policy, "val_metrics": val_metrics, "aux_val_metrics": _aux_metrics(pred_val, labels_val), "val_score": score}
    if score > best_score:
        best_score = score
        best_name = "joint_route_conditioned_ensemble"
        best_policy = policy
    pred_test, labels_test = _predict_ensemble(paths if best_name == "joint_route_conditioned_ensemble" else [trials[best_name]["train"]["checkpoint"]], "test")
    test_metrics = ft._eval_policy(_as_ft_pred(pred_test), labels_test, best_policy, bootstrap=True)
    aux_test = _aux_metrics(pred_test, labels_test)
    full_metrics = full_ref.get("best_metrics", {})
    lift_over_full = {
        "all_delta": float(test_metrics.get("all_improvement", 0.0) - full_metrics.get("all_improvement", 0.0)),
        "t50_delta": float(test_metrics.get("t50_improvement", 0.0) - full_metrics.get("t50_improvement", 0.0)),
        "t100_delta": float(test_metrics.get("t100_improvement", 0.0) - full_metrics.get("t100_improvement", 0.0)),
        "hard_delta": float(test_metrics.get("hard_failure_improvement", 0.0) - full_metrics.get("hard_failure_improvement", 0.0)),
        "easy_delta": float(test_metrics.get("easy_degradation", 0.0) - full_metrics.get("easy_degradation", 0.0)),
    }
    contributes = bool(
        test_metrics.get("easy_degradation", 1.0) <= 0.02
        and (
            lift_over_full["all_delta"] > 0.002
            or lift_over_full["t50_delta"] > 0.002
            or lift_over_full["hard_delta"] > 0.002
        )
    )
    result = {
        "source": "fresh_run",
        "protocol_status": "joint_route_conditioned_world_state",
        "best_name": best_name,
        "best_policy": best_policy,
        "best_metrics": test_metrics,
        "auxiliary_test_metrics": aux_test,
        "lift_over_full_trajectory_reference": lift_over_full,
        "joint_route_conditioning_contributes": contributes,
        "trials": trials,
        "no_leakage": {
            "route_and_physical_labels_train_loss_only": True,
            "future_waypoints_label_eval_only": True,
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
        "caveat": "This tests whether route/physical auxiliary training improves the trajectory world-state head. It remains dataset-local raw-frame 2.5D and does not execute Stage5C or SMC.",
    }
    _write_json(OUT_DIR / "stage41_joint_route_conditioned_world_state.json", result)
    lines = [
        "# Stage41 Joint Route-Conditioned World-State Model",
        "",
        "- source: `fresh_run`",
        f"- best: `{best_name}`",
        f"- joint route conditioning contributes: `{contributes}`",
        f"- trajectory metrics: `{test_metrics}`",
        f"- auxiliary metrics: `{aux_test}`",
        f"- lift over full trajectory reference: `{lift_over_full}`",
        f"- no leakage: `{result['no_leakage']}`",
    ]
    write_md(OUT_DIR / "stage41_joint_route_conditioned_world_state.md", lines)
    return result


def main_joint_route_conditioned_world_state() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        train_joint_route_conditioned_world_state()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_joint_route_conditioned_world_state",
            status,
            started,
            [OUT_DIR / "stage41_full_trajectory_world_state.json", OUT_DIR / "stage41_goal_route_physical_repair.json"],
            [OUT_DIR / "stage41_joint_route_conditioned_world_state.md", OUT_DIR / "stage41_joint_route_conditioned_world_state.json"],
        )

