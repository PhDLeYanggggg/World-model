from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_full_trajectory_world_state as ft
from src import stage41_joint_rollout_consistency as jrc
from src import stage41_joint_policy_distillation as jpd
from src import stage41_joint_latent_rollout as jlr


OUT_DIR = jpd.OUT_DIR
CHECKPOINT_DIR = jpd.CHECKPOINT_DIR
REPORT_JSON = OUT_DIR / "stage41_joint_residual_rollout.json"
REPORT_MD = OUT_DIR / "stage41_joint_residual_rollout.md"
THREADS = 4
GROUP_BATCH = 64
EPOCHS = 3
SEED = 4179
EPS = 1e-6

TRIALS = [
    {"name": "joint_residual_clip050_safe", "width": 80, "dropout": 0.10, "lr": 8e-4, "clip": 0.50, "hard_w": 1.5, "t50_w": 2.0, "t100_w": 1.0, "seed": 1},
    {"name": "joint_residual_clip100_balanced", "width": 96, "dropout": 0.08, "lr": 7e-4, "clip": 1.00, "hard_w": 2.0, "t50_w": 2.5, "t100_w": 1.5, "seed": 2},
    {"name": "joint_residual_clip150_long", "width": 96, "dropout": 0.08, "lr": 6e-4, "clip": 1.50, "hard_w": 2.5, "t50_w": 1.5, "t100_w": 3.0, "seed": 3},
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


def _torch():
    torch = jpd._torch()
    torch.set_num_threads(THREADS)
    return torch


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
    with (OUT_DIR / "run_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


def _labels_for_eval(data: Mapping[str, Any]) -> dict[str, np.ndarray]:
    labels = dict(data["labels"])
    labels["waypoint_valid"] = data["waypoint_valid"].astype(bool)
    return labels


def _residual_bundle(split: str, clip: float) -> dict[str, Any]:
    data = jlr._bundle(split)
    labels = _labels_for_eval(data)
    norm = np.maximum(labels["normalizer"].astype(np.float64), EPS)
    residual = ((labels["waypoint_xy"].astype(np.float64) - data["floor_xy"].astype(np.float64)) / norm[:, None, None]).astype(np.float32)
    clipped = np.clip(residual, -float(clip), float(clip)).astype(np.float32)
    clipped_xy = data["floor_xy"].astype(np.float64) + clipped.astype(np.float64) * norm[:, None, None]
    clipped_ade, _clipped_fde = ft._trajectory_errors(clipped_xy, labels)
    gain = (data["floor_ade"].astype(np.float64) - clipped_ade).astype(np.float32)
    harm = (clipped_ade > data["floor_ade"].astype(np.float64)).astype(np.float32)
    residual_norm = np.linalg.norm(clipped, axis=2).mean(axis=1).astype(np.float32)
    risk = np.log1p(clipped_ade / norm).astype(np.float32)
    out = dict(data)
    out["residual_target"] = clipped
    out["waypoint_delta"] = clipped
    out["gain_target"] = gain
    out["harm_target"] = harm
    out["risk_target"] = risk
    out["residual_norm_target"] = residual_norm
    out["clip"] = float(clip)
    out["target_clip_metrics"] = jlr._rollout_eval(
        {"waypoint_delta": clipped, "traj_risk": risk, "interaction": data["interaction"], "occupancy": data["occupancy"], "physical": data["physical"], "future_close": data["future_close"]},
        out,
        np.ones(len(gain), dtype=bool),
        f"{split}_oracle_clipped_residual",
    )["selected_metrics"]
    return out


def _pack_groups(data: Mapping[str, Any], group_ids: Sequence[int]):
    torch = _torch()
    groups = [data["groups"][int(i)] for i in group_ids]
    b = len(groups)
    max_g = max((len(g) for g in groups), default=1)
    feat = np.zeros((b, max_g, int(data["feature_dim"])), dtype=np.float32)
    residual = np.zeros((b, max_g, len(ft.WAYPOINT_FRAC), 2), dtype=np.float32)
    valid = np.zeros((b, max_g, len(ft.WAYPOINT_FRAC)), dtype=np.float32)
    row_ids = np.full((b, max_g), -1, dtype=np.int64)
    mask = np.zeros((b, max_g), dtype=bool)
    aux = {
        name: np.zeros((b, max_g), dtype=np.float32)
        for name in ["interaction", "occupancy", "physical", "future_close", "hard", "gain_target", "harm_target", "risk_target", "residual_norm_target"]
    }
    for bi, rows in enumerate(groups):
        n = len(rows)
        feat[bi, :n] = data["x"][rows]
        residual[bi, :n] = data["residual_target"][rows]
        valid[bi, :n] = data["waypoint_valid"][rows].astype(np.float32)
        row_ids[bi, :n] = rows
        mask[bi, :n] = True
        for name in aux:
            aux[name][bi, :n] = data[name][rows].astype(np.float32)
    return {
        "x": torch.tensor(feat),
        "residual": torch.tensor(residual),
        "valid": torch.tensor(valid),
        "mask": torch.tensor(mask),
        "row_ids": row_ids,
        **{name: torch.tensor(value) for name, value in aux.items()},
    }


def _make_model(feature_dim: int, width: int, dropout: float, clip: float):
    torch = _torch()
    import torch.nn as nn

    class JointResidualRollout(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.clip = float(clip)
            self.row = nn.Sequential(nn.Linear(feature_dim, width), nn.GELU(), nn.LayerNorm(width), nn.Dropout(dropout))
            layer = nn.TransformerEncoderLayer(d_model=width, nhead=4, dim_feedforward=width * 2, dropout=dropout, batch_first=True)
            self.agent_encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.group = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width))
            self.ctx = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.LayerNorm(width))
            self.residual = nn.Linear(width, len(ft.WAYPOINT_FRAC) * 2)
            self.risk = nn.Linear(width, 1)
            self.gain = nn.Linear(width, 1)
            self.harm = nn.Linear(width, 1)
            self.uncertainty = nn.Linear(width, 1)
            self.interaction = nn.Linear(width, 1)
            self.occupancy = nn.Linear(width, 1)
            self.physical = nn.Linear(width, 1)
            self.future_close = nn.Linear(width, 1)

        def forward(self, x, mask):
            h = self.row(x)
            h = self.agent_encoder(h, src_key_padding_mask=~mask.bool())
            valid = mask.float().unsqueeze(-1)
            pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            g = self.group(pooled).unsqueeze(1).expand_as(h)
            z = self.ctx(torch.cat([h, g], dim=-1))
            return {
                "residual_delta": self.clip * torch.tanh(self.residual(z).view(x.shape[0], x.shape[1], len(ft.WAYPOINT_FRAC), 2)),
                "traj_risk": self.risk(z).squeeze(-1),
                "gain": self.gain(z).squeeze(-1),
                "harm_logit": self.harm(z).squeeze(-1),
                "uncertainty_logit": self.uncertainty(z).squeeze(-1),
                "interaction_logit": self.interaction(z).squeeze(-1),
                "occupancy_logit": self.occupancy(z).squeeze(-1),
                "physical_logit": self.physical(z).squeeze(-1),
                "future_close_logit": self.future_close(z).squeeze(-1),
            }

    return JointResidualRollout()


def _masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def _train_trial(trial: Mapping[str, Any]) -> dict[str, Any]:
    torch = _torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    train = _residual_bundle("train", float(trial["clip"]))
    val = _residual_bundle("val", float(trial["clip"]))
    ckpt = CHECKPOINT_DIR / f"stage41_{trial['name']}.pt"
    heartbeat = OUT_DIR / f"{trial['name']}_heartbeat.json"
    if ckpt.exists() and heartbeat.exists():
        payload = read_json(heartbeat, {})
        if int(payload.get("epoch", 0)) >= EPOCHS:
            return {"source": "cached_verified", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": payload.get("best", {}), "trial": dict(trial)}
    model = _make_model(train["feature_dim"], int(trial["width"]), float(trial["dropout"]), float(trial["clip"]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(trial["lr"]), weight_decay=1e-4)
    rng = np.random.default_rng(SEED + int(trial["seed"]))
    best = {"val_loss": float("inf"), "epoch": 0}
    group_order = np.arange(len(train["groups"]), dtype=np.int64)
    for epoch in range(1, EPOCHS + 1):
        rng.shuffle(group_order)
        losses: list[float] = []
        model.train()
        for start in range(0, len(group_order), GROUP_BATCH):
            batch = _pack_groups(train, group_order[start : start + GROUP_BATCH])
            out = model(batch["x"], batch["mask"])
            row_mask = batch["mask"].float()
            waypoint_valid = batch["valid"] * row_mask.unsqueeze(-1)
            horizon_proxy = (batch["risk_target"] > 0.25).float()
            row_w = 1.0 + float(trial["hard_w"]) * batch["hard"] + float(trial["t50_w"]) * horizon_proxy + 1.2 * torch.clamp(torch.relu(batch["gain_target"]), max=2.0)
            residual_loss = (
                F.smooth_l1_loss(out["residual_delta"], batch["residual"], reduction="none").mean(dim=-1)
                * waypoint_valid
            ).sum(dim=-1) / waypoint_valid.sum(dim=-1).clamp_min(1.0)
            residual_loss = _masked_mean(residual_loss * row_w, row_mask)
            risk_loss = _masked_mean(F.smooth_l1_loss(out["traj_risk"], batch["risk_target"], reduction="none") * row_w, row_mask)
            gain_loss = _masked_mean(F.smooth_l1_loss(out["gain"], batch["gain_target"], reduction="none") * row_w, row_mask)
            harm_loss = _masked_mean(F.binary_cross_entropy_with_logits(out["harm_logit"], batch["harm_target"], reduction="none") * row_w, row_mask)
            uncertainty_loss = _masked_mean(F.binary_cross_entropy_with_logits(out["uncertainty_logit"], batch["harm_target"], reduction="none"), row_mask)
            aux = (
                _masked_mean(F.binary_cross_entropy_with_logits(out["interaction_logit"], batch["interaction"], reduction="none"), row_mask)
                + _masked_mean(F.binary_cross_entropy_with_logits(out["occupancy_logit"], batch["occupancy"], reduction="none"), row_mask)
                + _masked_mean(F.binary_cross_entropy_with_logits(out["physical_logit"], batch["physical"], reduction="none"), row_mask)
                + _masked_mean(F.binary_cross_entropy_with_logits(out["future_close_logit"], batch["future_close"], reduction="none"), row_mask)
            )
            loss = residual_loss + 0.20 * risk_loss + 0.30 * gain_loss + 0.65 * harm_loss + 0.35 * uncertainty_loss + 0.20 * aux
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_losses: list[float] = []
            for start in range(0, len(val["groups"]), GROUP_BATCH):
                batch = _pack_groups(val, np.arange(start, min(start + GROUP_BATCH, len(val["groups"])), dtype=np.int64))
                out = model(batch["x"], batch["mask"])
                row_mask = batch["mask"].float()
                waypoint_valid = batch["valid"] * row_mask.unsqueeze(-1)
                residual_loss = (
                    F.smooth_l1_loss(out["residual_delta"], batch["residual"], reduction="none").mean(dim=-1)
                    * waypoint_valid
                ).sum(dim=-1) / waypoint_valid.sum(dim=-1).clamp_min(1.0)
                val_losses.append(float(_masked_mean(residual_loss, row_mask).cpu()))
            val_loss = float(np.mean(val_losses))
        heartbeat.write_text(
            json.dumps(
                {
                    "trial": dict(trial),
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "val_loss": val_loss,
                    "checkpoint": str(ckpt),
                    "best": best,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses))}
            torch.save({"model": model.state_dict(), "feature_dim": int(train["feature_dim"]), "trial": dict(trial), "best": best}, ckpt)
    return {"source": "fresh_run", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best, "trial": dict(trial)}


def _predict(path: str | Path, data: Mapping[str, Any]) -> dict[str, np.ndarray]:
    torch = _torch()
    payload = torch.load(path, map_location="cpu")
    trial = payload["trial"]
    model = _make_model(int(payload["feature_dim"]), int(trial["width"]), float(trial["dropout"]), float(trial["clip"]))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    out_arrays = {
        "residual_delta": np.zeros((len(data["x"]), len(ft.WAYPOINT_FRAC), 2), dtype=np.float32),
        "traj_risk": np.zeros(len(data["x"]), dtype=np.float32),
        "gain": np.zeros(len(data["x"]), dtype=np.float32),
        "harm": np.zeros(len(data["x"]), dtype=np.float32),
        "uncertainty": np.zeros(len(data["x"]), dtype=np.float32),
        "interaction": np.zeros(len(data["x"]), dtype=np.float32),
        "occupancy": np.zeros(len(data["x"]), dtype=np.float32),
        "physical": np.zeros(len(data["x"]), dtype=np.float32),
        "future_close": np.zeros(len(data["x"]), dtype=np.float32),
    }
    with torch.no_grad():
        for start in range(0, len(data["groups"]), GROUP_BATCH):
            group_ids = np.arange(start, min(start + GROUP_BATCH, len(data["groups"])), dtype=np.int64)
            batch = _pack_groups(data, group_ids)
            out = model(batch["x"], batch["mask"])
            row_ids = batch["row_ids"]
            valid = row_ids >= 0
            idx = row_ids[valid]
            out_arrays["residual_delta"][idx] = out["residual_delta"].cpu().numpy()[valid]
            out_arrays["traj_risk"][idx] = out["traj_risk"].cpu().numpy()[valid]
            out_arrays["gain"][idx] = out["gain"].cpu().numpy()[valid]
            out_arrays["harm"][idx] = torch.sigmoid(out["harm_logit"]).cpu().numpy()[valid]
            out_arrays["uncertainty"][idx] = torch.sigmoid(out["uncertainty_logit"]).cpu().numpy()[valid]
            out_arrays["interaction"][idx] = torch.sigmoid(out["interaction_logit"]).cpu().numpy()[valid]
            out_arrays["occupancy"][idx] = torch.sigmoid(out["occupancy_logit"]).cpu().numpy()[valid]
            out_arrays["physical"][idx] = torch.sigmoid(out["physical_logit"]).cpu().numpy()[valid]
            out_arrays["future_close"][idx] = torch.sigmoid(out["future_close_logit"]).cpu().numpy()[valid]
    return out_arrays


def _pred_waypoints(pred: Mapping[str, np.ndarray], data: Mapping[str, Any]) -> np.ndarray:
    labels = _labels_for_eval(data)
    return data["floor_xy"].astype(np.float64) + pred["residual_delta"].astype(np.float64) * labels["normalizer"][:, None, None]


def _rollout_eval(pred: Mapping[str, np.ndarray], data: Mapping[str, Any], switch: np.ndarray, name: str) -> dict[str, Any]:
    labels = _labels_for_eval(data)
    neural_xy = _pred_waypoints(pred, data)
    bundle = {
        "labels": labels,
        "keys": data["keys"],
        "floor_xy": data["floor_xy"],
        "neural_xy": neural_xy,
        "floor_ade": data["floor_ade"],
    }
    return jrc._evaluate_split_rollout(bundle, switch.astype(bool), name)


def _policy_switch(pred: Mapping[str, np.ndarray], policy: Mapping[str, float]) -> np.ndarray:
    return (
        (pred["gain"] >= float(policy.get("gain_min", 0.0)))
        & (pred["harm"] <= float(policy.get("harm_max", 1.0)))
        & (pred["uncertainty"] <= float(policy.get("uncertainty_max", 1.0)))
        & (pred["traj_risk"] <= float(policy.get("traj_risk_max", 1e9)))
        & (pred["physical"] >= float(policy.get("physical_min", 0.0)))
        & (pred["future_close"] <= float(policy.get("future_close_max", 1.0)))
    )


def _policy_grid(pred: Mapping[str, np.ndarray]) -> list[dict[str, float]]:
    gain_grid = [0.0] + [float(v) for v in np.quantile(pred["gain"], [0.50, 0.70, 0.85])]
    harm_grid = [float(v) for v in np.quantile(pred["harm"], [0.20, 0.40, 0.60, 0.80])]
    risk_grid = [float(v) for v in np.quantile(pred["traj_risk"], [0.10, 0.25, 0.40, 0.60])]
    out: list[dict[str, float]] = [{"gain_min": 1e9, "harm_max": 0.0, "uncertainty_max": 0.0, "traj_risk_max": -1e9, "physical_min": 1.1, "future_close_max": 0.0}]
    for gain in gain_grid:
        for harm in harm_grid:
            for risk in risk_grid:
                out.append(
                    {
                        "gain_min": gain,
                        "harm_max": harm,
                        "uncertainty_max": harm,
                        "traj_risk_max": risk,
                        "physical_min": 0.35,
                        "future_close_max": 0.85,
                    }
                )
    return out


def _score(metrics: Mapping[str, Any], collision_delta: float) -> float:
    return (
        1.0 * float(metrics.get("all_improvement", 0.0))
        + 1.35 * float(metrics.get("t50_improvement", 0.0))
        + 1.0 * float(metrics.get("t100_improvement", 0.0))
        + 1.25 * float(metrics.get("hard_failure_improvement", 0.0))
        - 35.0 * max(0.0, float(metrics.get("easy_degradation", 1.0)) - 0.02)
        - 8.0 * max(0.0, collision_delta - 0.01)
    )


def _fit_policy(pred: Mapping[str, np.ndarray], val: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for policy in _policy_grid(pred):
        switch = _policy_switch(pred, policy)
        ev = _rollout_eval(pred, val, switch, "val_joint_residual_rollout")
        metrics = ev["selected_metrics"]
        eligible = bool(
            metrics.get("all_improvement", 0.0) > 0
            and metrics.get("t50_improvement", 0.0) > 0
            and metrics.get("hard_failure_improvement", 0.0) > 0
            and metrics.get("easy_degradation", 1.0) <= 0.02
            and ev["collision_delta_005"] <= 0.01
            and float(np.mean(switch)) > 0.0
        )
        candidates.append(
            {
                "policy": dict(policy),
                "metrics": metrics,
                "collision_delta_005": ev["collision_delta_005"],
                "switch_rate": float(np.mean(switch)),
                "eligible": eligible,
                "score": _score(metrics, ev["collision_delta_005"]),
            }
        )
    pool = [row for row in candidates if row["eligible"]] or candidates
    best = max(pool, key=lambda row: row["score"])
    return {"type": "joint_residual_rollout_val_selected", **best["policy"], "val_eligible": best["eligible"]}, candidates


def _aux_metrics(pred: Mapping[str, np.ndarray], data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "interaction": ft._binary_metrics(pred["interaction"], data["interaction"].astype(bool)),
        "occupancy": ft._binary_metrics(pred["occupancy"], data["occupancy"].astype(bool)),
        "physical": ft._binary_metrics(pred["physical"], data["physical"].astype(bool)),
        "future_group_close": ft._binary_metrics(1.0 - pred["future_close"], ~data["future_close"].astype(bool)),
        "harm": ft._binary_metrics(1.0 - pred["harm"], ~data["harm_target"].astype(bool)),
    }


def run_joint_residual_rollout() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    trial_results: dict[str, Any] = {}
    val_rank: list[dict[str, Any]] = []
    for trial in TRIALS:
        train_result = _train_trial(trial)
        val = _residual_bundle("val", float(trial["clip"]))
        pred_val = _predict(train_result["checkpoint"], val)
        policy, candidates = _fit_policy(pred_val, val)
        best_val = max(candidates, key=lambda row: row["score"]) if candidates else {}
        val_rank.append({"trial": trial["name"], "score": best_val.get("score", -1e9), "policy": policy, "val_best": best_val})
        trial_results[str(trial["name"])] = {"train": train_result, "policy": policy, "val_best": best_val, "target_clip_metrics": val.get("target_clip_metrics")}
    selected = max(val_rank, key=lambda row: row["score"])
    selected_trial = next(row for row in TRIALS if row["name"] == selected["trial"])
    selected_train = trial_results[selected["trial"]]["train"]
    test = _residual_bundle("test", float(selected_trial["clip"]))
    pred_test = _predict(selected_train["checkpoint"], test)
    switch = _policy_switch(pred_test, selected["policy"])
    selected_eval = _rollout_eval(pred_test, test, switch, "test_joint_residual_rollout")
    raw_eval = _rollout_eval(pred_test, test, np.ones(len(test["x"]), dtype=bool), "raw_joint_residual_rollout")
    floor_eval = _rollout_eval(pred_test, test, np.zeros(len(test["x"]), dtype=bool), "floor_reference")
    group_repair = read_json(OUT_DIR / "stage41_group_consistency_multiseed_repair.json", {})
    group_single = read_json(OUT_DIR / "stage41_group_consistency_distiller.json", {})
    group_summary = group_repair.get("metric_summary") or {}
    single_metrics = group_single.get("test_metrics") or {}
    group_basis = {
        "all_improvement": (group_summary.get("all_improvement") or {}).get("mean", single_metrics.get("all_improvement", 0.0)),
        "t50_improvement": (group_summary.get("t50_improvement") or {}).get("mean", single_metrics.get("t50_improvement", 0.0)),
        "t100_improvement": (group_summary.get("t100_improvement") or {}).get("mean", single_metrics.get("t100_improvement", 0.0)),
        "hard_failure_improvement": (group_summary.get("hard_failure_improvement") or {}).get("mean", single_metrics.get("hard_failure_improvement", 0.0)),
        "easy_degradation": (group_summary.get("easy_degradation") or {}).get("max", single_metrics.get("easy_degradation", 1.0)),
        "collision_delta_vs_floor_005": (group_summary.get("collision_delta_vs_floor_005") or {}).get("max", single_metrics.get("collision_delta_vs_floor_005", 1.0)),
    }
    metrics = selected_eval["selected_metrics"]
    lift = {
        "all_delta": float(metrics.get("all_improvement", 0.0) - float(group_basis.get("all_improvement") or 0.0)),
        "t50_delta": float(metrics.get("t50_improvement", 0.0) - float(group_basis.get("t50_improvement") or 0.0)),
        "t100_delta": float(metrics.get("t100_improvement", 0.0) - float(group_basis.get("t100_improvement") or 0.0)),
        "hard_delta": float(metrics.get("hard_failure_improvement", 0.0) - float(group_basis.get("hard_failure_improvement") or 0.0)),
        "easy_delta": float(metrics.get("easy_degradation", 0.0) - float(group_basis.get("easy_degradation") or 0.0)),
    }
    deployable = bool(
        metrics.get("all_improvement", 0.0) > 0
        and metrics.get("t50_improvement", 0.0) > 0
        and metrics.get("hard_failure_improvement", 0.0) > 0
        and metrics.get("easy_degradation", 1.0) <= 0.02
        and selected_eval["collision_delta_005"] <= 0.01
        and float(np.mean(switch)) > 0.0
    )
    improves_current = bool(deployable and (lift["all_delta"] > 0 or lift["t50_delta"] > 0 or lift["hard_delta"] > 0))
    result = {
        "source": "fresh_run",
        "protocol_status": "joint_baseline_relative_residual_rollout",
        "hypothesis": "Baseline-relative bounded residual targets reduce raw neural rollout harm versus direct future-delta prediction.",
        "trained_trials": trial_results,
        "selected_trial": selected["trial"],
        "selected_policy": selected["policy"],
        "test_metrics": metrics,
        "multi_agent_metrics": selected_eval["multi_agent_metrics"],
        "raw_neural_without_fallback_metrics": raw_eval["selected_metrics"],
        "floor_reference_metrics": floor_eval["selected_metrics"],
        "auxiliary_metrics": _aux_metrics(pred_test, test),
        "rollout_stats": {
            "selected": selected_eval["selected_stats"],
            "raw_neural_without_fallback": raw_eval["selected_stats"],
            "floor": selected_eval["floor_stats"],
        },
        "collision_delta_vs_floor_005": selected_eval["collision_delta_005"],
        "switch_rate": float(np.mean(switch)),
        "current_best_group_consistency_basis": group_basis,
        "lift_over_current_group_consistency_basis": lift,
        "joint_residual_rollout_deployable": deployable,
        "joint_residual_rollout_improves_current_deployable": improves_current,
        "no_leakage": {
            "future_waypoints_input": False,
            "future_waypoints_label_eval_only": True,
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "policy_selected_on_val": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "caveat": "This is bounded residual world-dynamics under the Stage37 safety floor. Coordinates remain dataset-local raw-frame 2.5D; no metric/seconds/true-3D/foundation claim.",
    }
    write_json(REPORT_JSON, _jsonable(result))
    write_md(
        REPORT_MD,
        [
            "# Stage41 Joint Residual Rollout",
            "",
            "- source: `fresh_run`",
            "- hypothesis: baseline-relative bounded residual targets reduce raw neural rollout harm.",
            f"- selected trial: `{selected['trial']}`",
            f"- selected policy: `{selected['policy']}`",
            f"- deployable: `{deployable}`",
            f"- improves current deployable: `{improves_current}`",
            f"- test metrics: `{metrics}`",
            f"- raw neural without fallback metrics: `{raw_eval['selected_metrics']}`",
            f"- lift over current group-consistency basis: `{lift}`",
            f"- collision delta vs floor @0.05 normalized: `{selected_eval['collision_delta_005']}`",
            f"- switch rate: `{float(np.mean(switch))}`",
            f"- auxiliary metrics: `{result['auxiliary_metrics']}`",
            f"- no leakage: `{result['no_leakage']}`",
            "",
            "Future waypoints supervise the residual target only; they are not input features. This does not execute Stage5C or SMC.",
        ],
    )
    return result


def main_joint_residual_rollout() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_joint_residual_rollout()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_joint_residual_rollout",
            status,
            started,
            [OUT_DIR / "stage41_joint_latent_rollout.json", OUT_DIR / "stage41_group_consistency_multiseed_repair.json"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_joint_residual_rollout()
