from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_full_trajectory_world_state as ft
from src import stage41_joint_multiagent_consistency as jmc
from src import stage41_joint_policy_distillation as jpd
from src import stage41_joint_rollout_consistency as jrc


OUT_DIR = jpd.OUT_DIR
CHECKPOINT_DIR = jpd.CHECKPOINT_DIR
REPORT_JSON = OUT_DIR / "stage41_joint_latent_rollout.json"
REPORT_MD = OUT_DIR / "stage41_joint_latent_rollout.md"
THREADS = 4
GROUP_BATCH = 64
EPOCHS = 3
SEED = 4169
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


def _group_indices(keys: np.ndarray) -> list[np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, key in enumerate(keys):
        groups[str(key)].append(i)
    return [np.asarray(v, dtype=np.int64) for _k, v in sorted(groups.items())]


def _group_count(keys: np.ndarray) -> np.ndarray:
    counts = Counter(map(str, keys.tolist()))
    return np.asarray([counts[str(k)] for k in keys], dtype=np.float32)


def _safe_float(x: np.ndarray, fill: float = 9.0) -> np.ndarray:
    return np.where(np.isfinite(x), x, fill).astype(np.float32)


def _joint_future_close_label(waypoint_xy: np.ndarray, keys: np.ndarray, normalizer: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    min_dist = jmc._min_group_distance(waypoint_xy, keys, normalizer.astype(np.float64))
    return (np.isfinite(min_dist) & (min_dist < threshold)).astype(np.float32)


def _bundle(split: str) -> dict[str, Any]:
    feature = jpd._feature_bundle(split)
    pred_ref, labels, _ref = jmc._split_predictions(split)
    meta = jmc._group_metadata(split)
    traj = ft._traj(split)
    ds = ft._fresh_ds(split)
    keys = meta["key"]
    floor_xy = ft._floor_waypoints(labels)
    floor_ade, floor_fde = ft._trajectory_errors(floor_xy, labels)
    normalizer = labels["normalizer"].astype(np.float64)
    current_xy = labels["current_xy"][:, None, :].repeat(len(ft.WAYPOINT_FRAC), axis=1)
    floor_min = jmc._min_group_distance(floor_xy, keys, normalizer)
    current_min = jmc._min_group_distance(current_xy, keys, normalizer)
    group_count = _group_count(keys)
    # Causal group context: current geometry, baseline rollout geometry, and
    # static/prediction diagnostics only. Future waypoints are labels/eval only.
    group_context = np.stack(
        [
            group_count / 10.0,
            _safe_float(current_min),
            _safe_float(floor_min),
            (ds["neighbor_counts"].astype(np.float32) / 10.0),
        ],
        axis=1,
    ).astype(np.float32)
    x = np.concatenate([feature["x"].astype(np.float32), group_context], axis=1).astype(np.float32)
    waypoint_valid = traj["waypoint_valid"].astype(bool)
    waypoint_delta = traj["waypoint_delta"].astype(np.float32)
    future_close = _joint_future_close_label(traj["waypoint_xy"].astype(np.float64), keys, normalizer)
    groups = _group_indices(keys)
    return {
        "x": x,
        "groups": groups,
        "keys": keys,
        "waypoint_delta": waypoint_delta,
        "waypoint_valid": waypoint_valid,
        "interaction": traj["interaction_future_close"].astype(np.float32),
        "occupancy": traj["occupancy_future_dense"].astype(np.float32),
        "physical": traj["physical_valid"].astype(np.float32),
        "future_close": future_close.astype(np.float32),
        "floor_xy": floor_xy,
        "floor_ade": floor_ade.astype(np.float64),
        "floor_fde": floor_fde.astype(np.float64),
        "labels": labels,
        "domain": labels["domain"].astype(str),
        "horizon": labels["horizon"].astype(np.int64),
        "hard": (labels["hard"].astype(bool) | labels["failure"].astype(bool)),
        "failure": labels["failure"].astype(bool),
        "easy": labels["easy"].astype(bool),
        "candidate_fde": labels["candidate_fde"],
        "feature_dim": int(x.shape[1]),
        "group_summary": {
            "groups": int(len(groups)),
            "rows": int(len(x)),
            "max_group_size": int(max((len(g) for g in groups), default=0)),
            "multi_agent_groups": int(sum(1 for g in groups if len(g) >= 2)),
            "multi_agent_rows": int(sum(len(g) for g in groups if len(g) >= 2)),
        },
    }


def _pack_groups(data: Mapping[str, Any], group_ids: Sequence[int]):
    torch = _torch()
    groups = [data["groups"][int(i)] for i in group_ids]
    b = len(groups)
    max_g = max((len(g) for g in groups), default=1)
    feat = np.zeros((b, max_g, int(data["feature_dim"])), dtype=np.float32)
    waypoint = np.zeros((b, max_g, len(ft.WAYPOINT_FRAC), 2), dtype=np.float32)
    valid = np.zeros((b, max_g, len(ft.WAYPOINT_FRAC)), dtype=np.float32)
    aux = {name: np.zeros((b, max_g), dtype=np.float32) for name in ["interaction", "occupancy", "physical", "future_close", "hard"]}
    row_ids = np.full((b, max_g), -1, dtype=np.int64)
    mask = np.zeros((b, max_g), dtype=bool)
    for bi, rows in enumerate(groups):
        n = len(rows)
        feat[bi, :n] = data["x"][rows]
        waypoint[bi, :n] = data["waypoint_delta"][rows]
        valid[bi, :n] = data["waypoint_valid"][rows].astype(np.float32)
        row_ids[bi, :n] = rows
        mask[bi, :n] = True
        for name in aux:
            aux[name][bi, :n] = data[name][rows].astype(np.float32)
    return {
        "x": torch.tensor(feat),
        "waypoint": torch.tensor(waypoint),
        "valid": torch.tensor(valid),
        "mask": torch.tensor(mask),
        "row_ids": row_ids,
        **{name: torch.tensor(value) for name, value in aux.items()},
    }


def _make_model(feature_dim: int, width: int = 96, dropout: float = 0.08):
    torch = _torch()
    import torch.nn as nn

    class JointLatentRollout(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.row = nn.Sequential(nn.Linear(feature_dim, width), nn.GELU(), nn.LayerNorm(width), nn.Dropout(dropout))
            layer = nn.TransformerEncoderLayer(
                d_model=width,
                nhead=4,
                dim_feedforward=width * 2,
                dropout=dropout,
                batch_first=True,
            )
            self.agent_encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.group = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.LayerNorm(width))
            self.ctx = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.LayerNorm(width))
            self.waypoint = nn.Linear(width, len(ft.WAYPOINT_FRAC) * 2)
            self.traj_risk = nn.Linear(width, 1)
            self.interaction = nn.Linear(width, 1)
            self.occupancy = nn.Linear(width, 1)
            self.physical = nn.Linear(width, 1)
            self.future_close = nn.Linear(width, 1)

        def forward(self, x, mask):
            h = self.row(x)
            h = self.agent_encoder(h, src_key_padding_mask=~mask.bool())
            valid = mask.float().unsqueeze(-1)
            pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            group_latent = self.group(pooled).unsqueeze(1).expand_as(h)
            z = self.ctx(torch.cat([h, group_latent], dim=-1))
            return {
                "waypoint_delta": self.waypoint(z).view(x.shape[0], x.shape[1], len(ft.WAYPOINT_FRAC), 2),
                "traj_risk": self.traj_risk(z).squeeze(-1),
                "interaction_logit": self.interaction(z).squeeze(-1),
                "occupancy_logit": self.occupancy(z).squeeze(-1),
                "physical_logit": self.physical(z).squeeze(-1),
                "future_close_logit": self.future_close(z).squeeze(-1),
            }

    return JointLatentRollout()


def _masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def _train_model(train: Mapping[str, Any], val: Mapping[str, Any]) -> dict[str, Any]:
    torch = _torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    ckpt = CHECKPOINT_DIR / "stage41_joint_latent_rollout.pt"
    heartbeat = OUT_DIR / "joint_latent_rollout_heartbeat.json"
    model = _make_model(int(train["feature_dim"]))
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
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
            hard_weight = 1.0 + 1.6 * batch["hard"] + 1.2 * (batch["future_close"] > 0.5).float()
            err = torch.linalg.norm(out["waypoint_delta"] - batch["waypoint"], dim=-1)
            traj_loss = (
                F.smooth_l1_loss(out["waypoint_delta"], batch["waypoint"], reduction="none").mean(dim=-1)
                * waypoint_valid
            ).sum(dim=-1) / waypoint_valid.sum(dim=-1).clamp_min(1.0)
            traj_loss = _masked_mean(traj_loss * hard_weight, row_mask)
            risk_target = torch.log1p((err * waypoint_valid).sum(dim=-1) / waypoint_valid.sum(dim=-1).clamp_min(1.0)).detach()
            risk_loss = _masked_mean(F.smooth_l1_loss(out["traj_risk"], risk_target, reduction="none"), row_mask)
            interaction = _masked_mean(F.binary_cross_entropy_with_logits(out["interaction_logit"], batch["interaction"], reduction="none"), row_mask)
            occupancy = _masked_mean(F.binary_cross_entropy_with_logits(out["occupancy_logit"], batch["occupancy"], reduction="none"), row_mask)
            physical = _masked_mean(F.binary_cross_entropy_with_logits(out["physical_logit"], batch["physical"], reduction="none"), row_mask)
            close = _masked_mean(F.binary_cross_entropy_with_logits(out["future_close_logit"], batch["future_close"], reduction="none"), row_mask)
            loss = traj_loss + 0.35 * risk_loss + 0.25 * (interaction + occupancy + physical) + 0.40 * close
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
                traj_loss = (
                    F.smooth_l1_loss(out["waypoint_delta"], batch["waypoint"], reduction="none").mean(dim=-1)
                    * waypoint_valid
                ).sum(dim=-1) / waypoint_valid.sum(dim=-1).clamp_min(1.0)
                val_losses.append(float(_masked_mean(traj_loss, row_mask).cpu()))
            val_loss = float(np.mean(val_losses))
        payload = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
            "checkpoint": str(ckpt),
            "best": best,
        }
        heartbeat.write_text(json.dumps(_jsonable(payload), ensure_ascii=False), encoding="utf-8")
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses))}
            torch.save({"model": model.state_dict(), "feature_dim": int(train["feature_dim"]), "best": best}, ckpt)
    return {"source": "fresh_run", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best}


def _predict(path: str | Path, data: Mapping[str, Any]) -> dict[str, np.ndarray]:
    torch = _torch()
    payload = torch.load(path, map_location="cpu")
    model = _make_model(int(payload["feature_dim"]))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    out_arrays = {
        "waypoint_delta": np.zeros((len(data["x"]), len(ft.WAYPOINT_FRAC), 2), dtype=np.float32),
        "traj_risk": np.zeros(len(data["x"]), dtype=np.float32),
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
            out_arrays["waypoint_delta"][idx] = out["waypoint_delta"].cpu().numpy()[valid]
            out_arrays["traj_risk"][idx] = out["traj_risk"].cpu().numpy()[valid]
            out_arrays["interaction"][idx] = torch.sigmoid(out["interaction_logit"]).cpu().numpy()[valid]
            out_arrays["occupancy"][idx] = torch.sigmoid(out["occupancy_logit"]).cpu().numpy()[valid]
            out_arrays["physical"][idx] = torch.sigmoid(out["physical_logit"]).cpu().numpy()[valid]
            out_arrays["future_close"][idx] = torch.sigmoid(out["future_close_logit"]).cpu().numpy()[valid]
    return out_arrays


def _labels_for_eval(data: Mapping[str, Any]) -> dict[str, np.ndarray]:
    labels = dict(data["labels"])
    labels["waypoint_valid"] = data["waypoint_valid"].astype(bool)
    return labels


def _rollout_eval(pred: Mapping[str, np.ndarray], data: Mapping[str, Any], switch: np.ndarray, name: str) -> dict[str, Any]:
    labels = _labels_for_eval(data)
    neural_xy = ft._pred_waypoints(pred, labels)
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
        (pred["traj_risk"] <= float(policy.get("traj_risk_max", 1e9)))
        & (pred["physical"] >= float(policy.get("physical_min", 0.0)))
        & (pred["future_close"] <= float(policy.get("future_close_max", 1.0)))
    )


def _policy_grid(pred: Mapping[str, np.ndarray]) -> list[dict[str, float]]:
    risk = pred["traj_risk"]
    close = pred["future_close"]
    risk_grid = [float(v) for v in np.quantile(risk, [0.02, 0.05, 0.10, 0.20, 0.35])]
    close_grid = [float(v) for v in np.quantile(close, [0.20, 0.40, 0.60])]
    out: list[dict[str, float]] = [{"traj_risk_max": -1e9, "physical_min": 1.1, "future_close_max": 0.0}]
    for r in risk_grid:
        for physical in [0.20, 0.40, 0.60]:
            for c in close_grid:
                out.append({"traj_risk_max": r, "physical_min": physical, "future_close_max": c})
    return out


def _score(metrics: Mapping[str, Any], collision_delta: float) -> float:
    return (
        float(metrics.get("all_improvement", 0.0))
        + 1.35 * float(metrics.get("t50_improvement", 0.0))
        + 0.8 * float(metrics.get("t100_improvement", 0.0))
        + 1.25 * float(metrics.get("hard_failure_improvement", 0.0))
        - 35.0 * max(0.0, float(metrics.get("easy_degradation", 1.0)) - 0.02)
        - 8.0 * max(0.0, collision_delta - 0.01)
    )


def _fit_policy(pred: Mapping[str, np.ndarray], val: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for policy in _policy_grid(pred):
        switch = _policy_switch(pred, policy)
        ev = _rollout_eval(pred, val, switch, "val_joint_latent_rollout")
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
    return {"type": "joint_latent_rollout_val_selected", **best["policy"], "val_eligible": best["eligible"]}, candidates


def _aux_metrics(pred: Mapping[str, np.ndarray], data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "interaction": ft._binary_metrics(pred["interaction"], data["interaction"].astype(bool)),
        "occupancy": ft._binary_metrics(pred["occupancy"], data["occupancy"].astype(bool)),
        "physical": ft._binary_metrics(pred["physical"], data["physical"].astype(bool)),
        "future_group_close": ft._binary_metrics(1.0 - pred["future_close"], ~data["future_close"].astype(bool)),
    }


def run_joint_latent_rollout() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    train = _bundle("train")
    val = _bundle("val")
    test = _bundle("test")
    train_result = _train_model(train, val)
    pred_val = _predict(train_result["checkpoint"], val)
    policy, candidates = _fit_policy(pred_val, val)
    pred_test = _predict(train_result["checkpoint"], test)
    switch = _policy_switch(pred_test, policy)
    test_eval = _rollout_eval(pred_test, test, switch, "test_joint_latent_rollout")
    selected_metrics = test_eval["selected_metrics"]
    multi_metrics = test_eval["multi_agent_metrics"]
    no_switch_eval = _rollout_eval(pred_test, test, np.zeros(len(test["x"]), dtype=bool), "floor_reference")
    raw_neural_eval = _rollout_eval(pred_test, test, np.ones(len(test["x"]), dtype=bool), "raw_joint_latent_rollout")
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
    lift_over_current = {
        "all_delta": float(selected_metrics.get("all_improvement", 0.0) - float(group_basis.get("all_improvement") or 0.0)),
        "t50_delta": float(selected_metrics.get("t50_improvement", 0.0) - float(group_basis.get("t50_improvement") or 0.0)),
        "t100_delta": float(selected_metrics.get("t100_improvement", 0.0) - float(group_basis.get("t100_improvement") or 0.0)),
        "hard_delta": float(selected_metrics.get("hard_failure_improvement", 0.0) - float(group_basis.get("hard_failure_improvement") or 0.0)),
        "easy_delta": float(selected_metrics.get("easy_degradation", 0.0) - float(group_basis.get("easy_degradation") or 0.0)),
    }
    deployable = bool(
        selected_metrics.get("all_improvement", 0.0) > 0
        and selected_metrics.get("t50_improvement", 0.0) > 0
        and selected_metrics.get("hard_failure_improvement", 0.0) > 0
        and selected_metrics.get("easy_degradation", 1.0) <= 0.02
        and test_eval["collision_delta_005"] <= 0.01
        and float(np.mean(switch)) > 0.0
    )
    improves_current = bool(
        deployable
        and (
            lift_over_current["all_delta"] > 0
            or lift_over_current["t50_delta"] > 0
            or lift_over_current["hard_delta"] > 0
        )
    )
    result = {
        "source": "fresh_run",
        "protocol_status": "joint_latent_rollout_prototype",
        "trained_group_token_transformer": True,
        "checkpoint": train_result["checkpoint"],
        "train": train_result,
        "data_summary": {
            "train": train["group_summary"],
            "val": val["group_summary"],
            "test": test["group_summary"],
        },
        "selected_policy": policy,
        "val_candidate_count": len(candidates),
        "val_best_candidate": max(candidates, key=lambda row: row["score"]) if candidates else None,
        "test_metrics": selected_metrics,
        "multi_agent_metrics": multi_metrics,
        "raw_neural_without_fallback_metrics": raw_neural_eval["selected_metrics"],
        "floor_reference_metrics": no_switch_eval["selected_metrics"],
        "auxiliary_metrics": _aux_metrics(pred_test, test),
        "rollout_stats": {
            "selected": test_eval["selected_stats"],
            "raw_neural_without_fallback": raw_neural_eval["selected_stats"],
            "floor": test_eval["floor_stats"],
        },
        "collision_delta_vs_floor_005": test_eval["collision_delta_005"],
        "switch_rate": float(np.mean(switch)),
        "current_best_group_consistency_basis": group_basis,
        "lift_over_current_group_consistency_basis": lift_over_current,
        "joint_latent_rollout_deployable": deployable,
        "joint_latent_rollout_improves_current_deployable": improves_current,
        "completion_contribution_status": "partial_joint_latent_world_state_evidence",
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
        "caveat": "This is a bounded group-token neural rollout prototype under the Stage37 safety floor. It predicts all rows in each current-frame group together, but it is not Stage5C latent generative execution, not SMC, not metric, and not seconds-level.",
    }
    write_json(REPORT_JSON, _jsonable(result))
    write_md(
        REPORT_MD,
        [
            "# Stage41 Joint Latent Rollout Prototype",
            "",
            "- source: `fresh_run`",
            "- trained group-token Transformer: `true`",
            f"- selected policy: `{policy}`",
            f"- deployable by standalone gate: `{deployable}`",
            f"- improves current group-consistency deployable basis: `{improves_current}`",
            f"- test metrics: `{selected_metrics}`",
            f"- multi-agent metrics: `{multi_metrics}`",
            f"- raw neural without fallback metrics: `{raw_neural_eval['selected_metrics']}`",
            f"- collision delta vs floor @0.05 normalized: `{test_eval['collision_delta_005']}`",
            f"- switch rate: `{float(np.mean(switch))}`",
            f"- lift over current group-consistency basis: `{lift_over_current}`",
            f"- auxiliary metrics: `{result['auxiliary_metrics']}`",
            f"- data summary: `{result['data_summary']}`",
            f"- no leakage: `{result['no_leakage']}`",
            "",
            "This run moves from row-level policy distillation toward a joint latent all-agent rollout: agents sharing a current-frame/horizon group are encoded together and their future waypoint deltas are predicted together. Future waypoints remain labels/evaluation only. The current deployable model is not replaced unless this prototype beats the existing group-consistency safety-buffer candidate.",
        ],
    )
    return result


def main_joint_latent_rollout() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_joint_latent_rollout()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_joint_latent_rollout",
            status,
            started,
            [
                OUT_DIR / "stage41_full_trajectory_world_state.json",
                OUT_DIR / "stage41_group_consistency_multiseed_repair.json",
            ],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_joint_latent_rollout()
