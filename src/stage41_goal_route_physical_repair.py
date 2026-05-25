from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_all_agent as aa
from src import stage41_fresh_confirmation as fresh
from src import stage41_full_trajectory_world_state as ft


OUT_DIR = fresh.OUT_DIR
DATA_DIR = fresh.DATA_DIR
CHECKPOINT_DIR = fresh.CHECKPOINT_DIR
LEDGER_JSONL = fresh.LEDGER_JSONL
ROUTE_NAMES = ["stop", "straight", "left_turn", "right_turn", "reverse_or_uturn", "interaction_detour"]
THREADS = 4
BATCH = 512
EPOCHS = 4
SEED = 4199
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


def _angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.sum(a * b, axis=1)
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return np.arctan2(cross, dot)


def _past_heading(ds: Mapping[str, np.ndarray]) -> np.ndarray:
    tokens = ds["agent_tokens"].astype(np.float32)
    primary = tokens[:, 0, :, :]
    valid = primary[:, :, 6] > 0.5
    dxdy = primary[:, :, 0:2]
    out = np.zeros((len(primary), 2), dtype=np.float32)
    for i in range(len(primary)):
        ids = np.where(valid[i])[0]
        if len(ids):
            tail = ids[-min(8, len(ids)) :]
            out[i] = dxdy[i, tail].sum(axis=0)
    norm = np.linalg.norm(out, axis=1)
    zero = norm < EPS
    out[zero] = np.asarray([1.0, 0.0], dtype=np.float32)
    norm = np.maximum(np.linalg.norm(out, axis=1, keepdims=True), EPS)
    return out / norm


def _route_and_physical_labels(split: str) -> Dict[str, np.ndarray]:
    ds = ft._fresh_ds(split)
    tr = ft._traj(split)
    current = ds["current_xy"].astype(np.float32)
    normalizer = np.maximum(ds["normalizer"].astype(np.float32), EPS)
    waypoints = tr["waypoint_xy"].astype(np.float32)
    valid = tr["waypoint_valid"].astype(bool)
    future = waypoints[:, -1, :] - current
    past = _past_heading(ds)
    disp_norm = np.linalg.norm(future, axis=1)
    future_dir = future / np.maximum(disp_norm[:, None], EPS)
    angle = _angle_between(past, future_dir)
    route = np.ones(len(ds["horizon"]), dtype=np.int64)
    route[disp_norm <= np.maximum(0.03 * normalizer, 1e-4)] = 0
    route[np.abs(angle) > 2.35] = 4
    route[(angle >= 0.45) & (route != 0) & (route != 4)] = 2
    route[(angle <= -0.45) & (route != 0) & (route != 4)] = 3

    seg_valid_count = valid.sum(axis=1)
    speed_change = np.zeros(len(route), dtype=np.float32)
    max_turn = np.zeros(len(route), dtype=np.float32)
    endpoint_only = seg_valid_count < len(ft.WAYPOINT_FRAC)
    for i in range(len(route)):
        pts = np.concatenate([current[i][None, :], waypoints[i, valid[i]]], axis=0)
        if len(pts) < 3:
            continue
        d = np.diff(pts, axis=0)
        seg = np.linalg.norm(d, axis=1)
        speed_change[i] = float(np.max(np.abs(np.diff(seg))) / max(float(np.median(seg) + EPS), EPS)) if len(seg) > 1 else 0.0
        if len(d) > 1:
            a = d[:-1] / np.maximum(np.linalg.norm(d[:-1], axis=1, keepdims=True), EPS)
            b = d[1:] / np.maximum(np.linalg.norm(d[1:], axis=1, keepdims=True), EPS)
            max_turn[i] = float(np.max(np.abs(_angle_between(a, b))))
    interaction = tr["interaction_future_close"].astype(bool)
    occupancy = tr["occupancy_future_dense"].astype(bool)
    route[(interaction | occupancy) & (np.abs(angle) > 0.25) & (route != 0)] = 5
    physical_challenge = (
        (speed_change > 1.20)
        | (max_turn > 0.85)
        | interaction
        | occupancy
        | endpoint_only
        | (ds["horizon"].astype(int) == 100)
    )
    np.savez_compressed(
        DATA_DIR / f"goal_route_physical_{split}.npz",
        route_label=route.astype(np.int16),
        route_angle=angle.astype(np.float32),
        route_displacement=disp_norm.astype(np.float32),
        physical_challenge=physical_challenge.astype(bool),
        speed_change=speed_change.astype(np.float32),
        max_turn=max_turn.astype(np.float32),
        endpoint_only=endpoint_only.astype(bool),
    )
    return {
        "route_label": route,
        "physical_challenge": physical_challenge,
        "speed_change": speed_change,
        "max_turn": max_turn,
        "endpoint_only": endpoint_only,
    }


def build_goal_route_physical_labels() -> Dict[str, Any]:
    ft.build_full_trajectory_labels()
    report: Dict[str, Any] = {
        "source": "fresh_run",
        "route_names": ROUTE_NAMES,
        "splits": {},
        "no_leakage": {
            "route_label_from_future_waypoints": "label_eval_only",
            "physical_challenge_from_future_waypoints": "label_eval_only",
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
    }
    for split in ["train", "val", "test"]:
        labels = _route_and_physical_labels(split)
        ds = ft._fresh_ds(split)
        route_counts = Counter(labels["route_label"].astype(int).tolist())
        report["splits"][split] = {
            "rows": int(len(labels["route_label"])),
            "route_distribution": {ROUTE_NAMES[k]: int(v) for k, v in sorted(route_counts.items())},
            "physical_challenge_positive": int(np.sum(labels["physical_challenge"])),
            "physical_challenge_rate": float(np.mean(labels["physical_challenge"])),
            "endpoint_only_rows": int(np.sum(labels["endpoint_only"])),
            "speed_change_p95": float(np.percentile(labels["speed_change"], 95)) if len(labels["speed_change"]) else 0.0,
            "max_turn_p95": float(np.percentile(labels["max_turn"], 95)) if len(labels["max_turn"]) else 0.0,
            "domains": dict(Counter(ds["domain"].astype(str).tolist())),
        }
    _write_json(OUT_DIR / "stage41_goal_route_physical_labels.json", report)
    write_md(
        OUT_DIR / "stage41_goal_route_physical_labels.md",
        [
            "# Stage41 Goal/Route and Physical-Consistency Labels",
            "",
            "- source: `fresh_run`",
            f"- route names: `{ROUTE_NAMES}`",
            f"- splits: `{report['splits']}`",
            f"- no leakage: `{report['no_leakage']}`",
            "",
            "Route and physical-challenge labels are supervised/evaluation targets derived from future waypoints; they are never input features.",
        ],
    )
    return report


def _labels(split: str) -> Dict[str, np.ndarray]:
    path = DATA_DIR / f"goal_route_physical_{split}.npz"
    if not path.exists():
        build_goal_route_physical_labels()
    return dict(np.load(path))


def _norm_static(static: np.ndarray) -> np.ndarray:
    norm = dict(np.load(DATA_DIR / "normalization.npz"))
    return ((static.astype(np.float32) - norm["static_mean"]) / norm["static_std"]).astype(np.float32)


def _load_tensors(split: str):
    torch = _torch()
    ds = ft._fresh_ds(split)
    labels = _labels(split)
    return {
        "agent_tokens": torch.tensor(ds["agent_tokens"].astype(np.float32)),
        "agent_mask": torch.tensor(ds["agent_mask"].astype(bool)),
        "static": torch.tensor(_norm_static(ds["static"])),
        "route": torch.tensor(labels["route_label"].astype(np.int64)),
        "physical": torch.tensor(labels["physical_challenge"].astype(np.float32)),
        "hard": torch.tensor((ds["hard"].astype(bool) | ds["failure"].astype(bool)).astype(np.float32)),
        "horizon": torch.tensor(ds["horizon"].astype(np.int64)),
        "raw": ds,
        "labels": labels,
    }


def _make_model(static_dim: int, width: int = 72, dropout: float = 0.08):
    torch = _torch()
    import torch.nn as nn

    class GoalRoutePhysicalHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.temporal = nn.Sequential(nn.Linear(9, width), nn.GELU(), nn.LayerNorm(width), nn.Dropout(dropout), nn.Linear(width, width))
            self.role = nn.Embedding(aa.MAX_AGENTS, width)
            layer = nn.TransformerEncoderLayer(d_model=width, nhead=4, dim_feedforward=width * 2, dropout=dropout, batch_first=True)
            self.agent_encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.static = nn.Sequential(nn.Linear(static_dim, width), nn.GELU(), nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
            self.ctx = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.LayerNorm(width))
            self.route = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, len(ROUTE_NAMES)))
            self.physical = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

        def forward(self, agent_tokens, agent_mask, static):
            b, a, _t, _ = agent_tokens.shape
            x = self.temporal(agent_tokens) + self.role.weight[:a][None, :, None, :]
            valid_time = agent_tokens[..., 6].clamp(0, 1)
            agent_emb = (x * valid_time[..., None]).sum(dim=2) / valid_time.sum(dim=2, keepdim=True).clamp_min(1.0)
            agent_h = self.agent_encoder(agent_emb)
            valid_agent = agent_mask[:, :, None].float()
            pooled = (agent_h * valid_agent).sum(dim=1) / valid_agent.sum(dim=1).clamp_min(1.0)
            ctx = self.ctx(torch.cat([pooled, self.static(static)], dim=1))
            return {"route_logits": self.route(ctx), "physical_logit": self.physical(ctx).squeeze(-1)}

    return GoalRoutePhysicalHead()


TRIALS = [
    {"name": "goal_route_balanced", "width": 72, "dropout": 0.08, "lr": 8e-4, "route_w": 1.0, "physical_w": 0.8, "hard_w": 1.5, "t100_w": 1.0, "seed": 1},
    {"name": "goal_route_physical_hard", "width": 80, "dropout": 0.10, "lr": 7e-4, "route_w": 0.8, "physical_w": 1.2, "hard_w": 3.0, "t100_w": 1.5, "seed": 2},
    {"name": "goal_route_long_horizon", "width": 80, "dropout": 0.08, "lr": 7e-4, "route_w": 1.0, "physical_w": 1.0, "hard_w": 2.0, "t100_w": 3.0, "seed": 3},
]


def _class_weights(route: np.ndarray) -> np.ndarray:
    counts = np.bincount(route.astype(int), minlength=len(ROUTE_NAMES)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / np.maximum(weights.mean(), EPS)
    return weights.astype(np.float32)


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
    pos = float(train["physical"].mean().item())
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
            row_w = 1.0 + float(trial["hard_w"]) * train["hard"][ids] + float(trial["t100_w"]) * (train["horizon"][ids] == 100).float()
            route_loss = F.cross_entropy(out["route_logits"], train["route"][ids], weight=route_weights, reduction="none")
            physical_loss = F.binary_cross_entropy_with_logits(out["physical_logit"], train["physical"][ids], pos_weight=pos_weight, reduction="none")
            loss = (float(trial["route_w"]) * route_loss * row_w).mean() + (float(trial["physical_w"]) * physical_loss * row_w).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            out = model(val["agent_tokens"], val["agent_mask"], val["static"])
            val_loss = float((F.cross_entropy(out["route_logits"], val["route"], weight=route_weights) + F.binary_cross_entropy_with_logits(out["physical_logit"], val["physical"], pos_weight=pos_weight)).cpu())
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
    route_logits = []
    physical = []
    with torch.no_grad():
        for start in range(0, tensors["agent_tokens"].shape[0], 2048):
            sl = slice(start, min(start + 2048, tensors["agent_tokens"].shape[0]))
            out = model(tensors["agent_tokens"][sl], tensors["agent_mask"][sl], tensors["static"][sl])
            route_logits.append(out["route_logits"].cpu().numpy())
            physical.append(torch.sigmoid(out["physical_logit"]).cpu().numpy().reshape(-1))
    pred = {"route_logits": np.concatenate(route_logits, axis=0).astype(np.float32), "physical": np.concatenate(physical).astype(np.float32)}
    labels = {
        "route": tensors["labels"]["route_label"].astype(np.int64),
        "physical": tensors["labels"]["physical_challenge"].astype(bool),
        "domain": tensors["raw"]["domain"].astype(str),
        "horizon": tensors["raw"]["horizon"].astype(np.int64),
        "hard": (tensors["raw"]["hard"].astype(bool) | tensors["raw"]["failure"].astype(bool)),
        "easy": tensors["raw"]["easy"].astype(bool),
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
        raise ValueError("goal/route/physical ensemble requires checkpoints")
    return {k: np.mean([p[k] for p in preds], axis=0).astype(np.float32) for k in preds[0]}, labels_ref


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), EPS)


def _binary_metrics(score: np.ndarray, label: np.ndarray) -> Dict[str, Any]:
    return ft._binary_metrics(score.astype(np.float64), label.astype(bool))


def _route_metrics(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    p = _softmax(pred["route_logits"].astype(np.float64))
    y = labels["route"].astype(int)
    top1 = np.argmax(p, axis=1)
    top3 = np.argsort(p, axis=1)[:, -3:]
    majority = Counter(y.tolist()).most_common(1)[0][0] if len(y) else 0
    nll = -np.log(np.maximum(p[np.arange(len(y)), y], EPS)).mean() if len(y) else 0.0
    return {
        "top1": float(np.mean(top1 == y)) if len(y) else 0.0,
        "top3": float(np.mean([yy in row for yy, row in zip(y, top3)])) if len(y) else 0.0,
        "nll": float(nll),
        "majority_top1": float(np.mean(y == majority)) if len(y) else 0.0,
        "majority_class": ROUTE_NAMES[int(majority)] if len(y) else None,
        "distribution": {ROUTE_NAMES[k]: int(v) for k, v in sorted(Counter(y.tolist()).items())},
    }


def _eval(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    route = _route_metrics(pred, labels)
    physical = _binary_metrics(pred["physical"], labels["physical"])
    by_domain = {}
    for d in sorted(set(labels["domain"].astype(str).tolist())):
        mask = labels["domain"].astype(str) == d
        by_domain[d] = {
            "route": _route_metrics({k: v[mask] for k, v in pred.items()}, {k: v[mask] for k, v in labels.items()}),
            "physical": _binary_metrics(pred["physical"][mask], labels["physical"][mask]),
            "rows": int(np.sum(mask)),
        }
    return {
        "rows": int(len(labels["route"])),
        "route": route,
        "physical_challenge": physical,
        "by_domain": by_domain,
        "route_lift_over_majority": float(route["top1"] - route["majority_top1"]),
        "non_degenerate_physical_label": bool(0.05 <= physical.get("positive_rate", 0.0) <= 0.95),
    }


def train_goal_route_physical_repair() -> Dict[str, Any]:
    label_report = build_goal_route_physical_labels()
    trials: Dict[str, Any] = {}
    paths = []
    best_name = ""
    best_score = -1e18
    for trial in TRIALS:
        train = _train_trial(trial)
        pred_val, labels_val = _predict(train["checkpoint"], "val")
        val = _eval(pred_val, labels_val)
        score = val["route_lift_over_majority"] + 0.5 * float((val["physical_challenge"].get("auroc") or 0.0))
        trials[trial["name"]] = {"source": train.get("source"), "trial": trial, "train": train, "val_metrics": val, "val_score": score}
        paths.append(train["checkpoint"])
        if score > best_score:
            best_score = score
            best_name = trial["name"]
    pred_test, labels_test = _predict_ensemble(paths, "test")
    test = _eval(pred_test, labels_test)
    pass_gate = bool(
        test["route"]["top1"] > test["route"]["majority_top1"]
        and (test["physical_challenge"].get("auroc") or 0.0) >= 0.70
        and test["non_degenerate_physical_label"]
    )
    result = {
        "source": "fresh_run",
        "protocol_status": "goal_route_physical_repair",
        "best_name": best_name,
        "ensemble_test_metrics": test,
        "pass_gate": pass_gate,
        "label_report": label_report,
        "trials": trials,
        "no_leakage": {
            "future_route_label_input": False,
            "future_physical_label_input": False,
            "future_waypoints_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
        "caveat": "Route and physical labels are supervised targets from future waypoints only. This repairs auxiliary heads; it is not metric, not true 3D, and does not execute Stage5C or SMC.",
    }
    _write_json(OUT_DIR / "stage41_goal_route_physical_repair.json", result)
    write_md(
        OUT_DIR / "stage41_goal_route_physical_repair.md",
        [
            "# Stage41 Goal/Route and Physical-Consistency Repair",
            "",
            "- source: `fresh_run`",
            f"- pass gate: `{pass_gate}`",
            f"- best trial: `{best_name}`",
            f"- route metrics: `{test['route']}`",
            f"- physical challenge metrics: `{test['physical_challenge']}`",
            f"- by domain: `{test['by_domain']}`",
            f"- no leakage: `{result['no_leakage']}`",
        ],
    )
    return result


def main_goal_route_physical_repair() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        train_goal_route_physical_repair()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_goal_route_physical_repair",
            status,
            started,
            [DATA_DIR / "full_trajectory_train.npz"],
            [OUT_DIR / "stage41_goal_route_physical_repair.md", OUT_DIR / "stage41_goal_route_physical_repair.json"],
        )

