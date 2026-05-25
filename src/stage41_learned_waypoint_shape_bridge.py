from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_breakthrough as s41
from src import stage41_domain_local_all_agent_world_state as dlaa
from src import stage41_domain_local_neural_retrain as dl
from src import stage41_endpoint_to_full_trajectory_repair as bridge
from src import stage41_full_trajectory_world_state as ft
from src import stage41_joint_multiagent_consistency as jmc
from src import stage41_joint_rollout_consistency as jrc


OUT_DIR = dl.OUT_DIR
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
REPORT_JSON = OUT_DIR / "stage41_learned_waypoint_shape_bridge.json"
REPORT_MD = OUT_DIR / "stage41_learned_waypoint_shape_bridge.md"
THREADS = 4
BATCH = 1536
EPOCHS = 4
SEED = 41671
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


def _torch():
    torch = dl._torch()
    torch.set_num_threads(THREADS)
    return torch


def _fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0).astype(np.float32)
    std = np.maximum(x.std(axis=0), 1e-3).astype(np.float32)
    return mean, std


def _standardize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x.astype(np.float32) - mean) / std).astype(np.float32)


def _feature_matrix(data: Mapping[str, np.ndarray], endpoint_pred: Mapping[str, np.ndarray]) -> np.ndarray:
    floor_delta = data["cand_delta"][:, 0, :].astype(np.float32)
    endpoint_delta = endpoint_pred["delta"].astype(np.float32)
    gap = endpoint_delta - floor_delta
    uncertainty = endpoint_pred["uncertainty"][:, None].astype(np.float32)
    endpoint_norm = np.linalg.norm(endpoint_delta, axis=1, keepdims=True).astype(np.float32)
    gap_norm = np.linalg.norm(gap, axis=1, keepdims=True).astype(np.float32)
    horizon = data["horizon"].astype(np.float32)
    horizon_onehot = np.stack([(horizon == h).astype(np.float32) for h in [10, 25, 50, 100]], axis=1)
    return np.concatenate(
        [
            dl._feature_matrix(data).astype(np.float32),
            endpoint_delta,
            floor_delta,
            gap,
            uncertainty,
            endpoint_norm,
            gap_norm,
            horizon_onehot,
        ],
        axis=1,
    ).astype(np.float32)


def _linear_xy(data: Mapping[str, np.ndarray], endpoint_pred: Mapping[str, np.ndarray]) -> np.ndarray:
    return bridge._linear_waypoints_from_delta(data, endpoint_pred["delta"])


def _target_residual(data: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], endpoint_pred: Mapping[str, np.ndarray]) -> np.ndarray:
    linear = _linear_xy(data, endpoint_pred)
    norm = np.maximum(labels["normalizer"].astype(np.float64), EPS)
    return ((labels["waypoint_xy"].astype(np.float64) - linear) / norm[:, None, None]).astype(np.float32)


def _make_shape_model(dim: int, width: int = 96, dropout: float = 0.06):
    torch = _torch()
    import torch.nn as nn

    class WaypointShapeBridge(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, width),
                nn.GELU(),
                nn.LayerNorm(width),
                nn.Dropout(dropout),
                nn.Linear(width, width),
                nn.GELU(),
                nn.LayerNorm(width),
                nn.Dropout(dropout),
            )
            self.residual = nn.Linear(width, len(ft.WAYPOINT_FRAC) * 2)
            self.risk = nn.Linear(width, 1)

        def forward(self, x):
            h = self.net(x)
            return self.residual(h).view(-1, len(ft.WAYPOINT_FRAC), 2), self.risk(h).squeeze(-1)

    return WaypointShapeBridge()


def _train_shape_head(domain: str, train: Mapping[str, np.ndarray], val: Mapping[str, np.ndarray], pred_train: Mapping[str, np.ndarray], pred_val: Mapping[str, np.ndarray], labels_train: Mapping[str, np.ndarray], labels_val: Mapping[str, np.ndarray]) -> dict[str, Any]:
    torch = _torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    ckpt = CHECKPOINT_DIR / f"stage41_learned_waypoint_shape_{domain}.pt"
    heartbeat = OUT_DIR / f"stage41_learned_waypoint_shape_{domain}_heartbeat.json"
    if ckpt.exists() and heartbeat.exists():
        try:
            hb = json.loads(heartbeat.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            hb = {}
        if int(hb.get("epoch", 0)) >= EPOCHS:
            return {"source": "cached_verified", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": hb.get("best", {})}

    x_raw = _feature_matrix(train, pred_train)
    mean, std = _fit_standardizer(x_raw)
    x = torch.tensor(_standardize(x_raw, mean, std))
    y = torch.tensor(_target_residual(train, labels_train, pred_train))
    valid = torch.tensor(labels_train["waypoint_valid"].astype(np.float32))
    hard = torch.tensor((train["hard"].astype(bool) | train["failure"].astype(bool)).astype(np.float32))
    horizon = torch.tensor(train["horizon"].astype(np.int64))

    vx = torch.tensor(_standardize(_feature_matrix(val, pred_val), mean, std))
    vy = torch.tensor(_target_residual(val, labels_val, pred_val))
    vvalid = torch.tensor(labels_val["waypoint_valid"].astype(np.float32))

    domain_seed = sum((i + 1) * ord(ch) for i, ch in enumerate(domain)) % 1000
    torch.manual_seed(SEED + domain_seed)
    model = _make_shape_model(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    rng = np.random.default_rng(SEED + domain_seed)
    waypoint_w = torch.tensor([1.2, 1.1, 1.0, 0.45], dtype=torch.float32)
    best = {"val_loss": float("inf"), "epoch": 0}
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(x.shape[0])
        losses: list[float] = []
        model.train()
        for start in range(0, len(order), BATCH):
            ids = torch.tensor(order[start : start + BATCH], dtype=torch.long)
            pred, risk = model(x[ids])
            row_w = 1.0 + 1.5 * hard[ids]
            row_w = row_w + 2.2 * (horizon[ids] == 50).float() + 1.3 * (horizon[ids] == 100).float()
            loss_per_wp = F.smooth_l1_loss(pred, y[ids], reduction="none").mean(dim=2)
            masked = loss_per_wp * valid[ids] * waypoint_w[None, :]
            shape_loss = (masked.sum(dim=1) / (valid[ids] * waypoint_w[None, :]).sum(dim=1).clamp_min(1.0) * row_w).mean()
            shape_err = torch.linalg.norm((pred.detach() - y[ids]) * valid[ids, :, None], dim=2).sum(dim=1) / valid[ids].sum(dim=1).clamp_min(1.0)
            risk_loss = F.smooth_l1_loss(torch.exp(risk), torch.clamp(shape_err, min=1e-4))
            loss = shape_loss + 0.12 * risk_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            pred, risk = model(vx)
            loss_per_wp = F.smooth_l1_loss(pred, vy, reduction="none").mean(dim=2)
            val_loss = float(((loss_per_wp * vvalid * waypoint_w[None, :]).sum(dim=1) / (vvalid * waypoint_w[None, :]).sum(dim=1).clamp_min(1.0)).mean().cpu())
        best_candidate = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses))}
        heartbeat.write_text(json.dumps({"domain": domain, "epoch": epoch, "best": best_candidate, "checkpoint": str(ckpt)}, ensure_ascii=False), encoding="utf-8")
        if val_loss < best["val_loss"]:
            best = best_candidate
            torch.save({"model": model.state_dict(), "dim": int(x.shape[1]), "mean": mean, "std": std, "domain": domain, "best": best}, ckpt)
    return {"source": "fresh_run", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best}


def _predict_shape(path: str | Path, data: Mapping[str, np.ndarray], endpoint_pred: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    torch = _torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = _make_shape_model(int(payload["dim"]))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    x = torch.tensor(_standardize(_feature_matrix(data, endpoint_pred), payload["mean"], payload["std"]))
    residuals: list[np.ndarray] = []
    risks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], 4096):
            residual, risk = model(x[start : start + 4096])
            residuals.append(residual.cpu().numpy())
            risks.append(np.exp(risk.cpu().numpy()))
    return {"residual": np.concatenate(residuals).astype(np.float32), "risk": np.concatenate(risks).astype(np.float32)}


def _shape_xy(data: Mapping[str, np.ndarray], endpoint_pred: Mapping[str, np.ndarray], shape_pred: Mapping[str, np.ndarray], mode: str, residual_scale: float, residual_clip: float) -> np.ndarray:
    linear = _linear_xy(data, endpoint_pred)
    residual = np.clip(shape_pred["residual"].astype(np.float64), -residual_clip, residual_clip)
    if mode == "intermediate_only":
        residual = residual.copy()
        residual[:, -1, :] = 0.0
    elif mode == "endpoint_half":
        residual = residual.copy()
        residual[:, -1, :] *= 0.5
    elif mode != "all_waypoints":
        raise ValueError(f"unknown shape bridge mode: {mode}")
    return linear + residual_scale * residual * data["normalizer"].astype(np.float64)[:, None, None]


def _metric_ds(labels: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "horizon": labels["horizon"],
        "hard": labels["hard"],
        "failure": labels["failure"],
        "easy": labels["easy"],
        "domain": labels["domain"],
        "candidate_fde": labels["candidate_fde"],
    }


def _safe_metrics(selected: np.ndarray, floor: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    if not np.any(mask):
        return {"rows": 0, "all_improvement": 0.0, "t50_improvement": 0.0, "t100_improvement": 0.0, "hard_failure_improvement": 0.0, "easy_degradation": 0.0, "switch_rate": 0.0}
    return s41._metrics(selected[mask], floor[mask], {k: v[mask] for k, v in _metric_ds(labels).items()}, switch[mask])


def _gain(selected: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return 1.0 - float(np.mean(selected[mask])) / max(float(np.mean(reference[mask])), EPS)


def _group_keys(data: Mapping[str, np.ndarray]) -> np.ndarray:
    return dlaa._group_keys(data)


def _guard(floor_xy: np.ndarray, selected_xy: np.ndarray, bridge_xy: np.ndarray, labels: Mapping[str, np.ndarray], data: Mapping[str, np.ndarray], switch: np.ndarray, min_sep: float) -> tuple[np.ndarray, np.ndarray, int]:
    if min_sep <= 0:
        return selected_xy.copy(), switch.copy(), 0
    keys = _group_keys(data)
    norm = labels["normalizer"].astype(np.float64)
    floor_min = jmc._min_group_distance(floor_xy, keys, norm)
    selected_min = jmc._min_group_distance(selected_xy, keys, norm)
    bridge_min = jmc._min_group_distance(bridge_xy, keys, norm)
    guard = switch & np.isfinite(selected_min) & (selected_min < min_sep) & (selected_min < np.minimum(floor_min, bridge_min))
    out = selected_xy.copy()
    out_switch = switch.copy()
    out[guard] = bridge_xy[guard]
    out_switch[guard] = False
    return out, out_switch, int(np.sum(guard))


def _eval_xy(floor_xy: np.ndarray, bridge_selected_xy: np.ndarray, shape_selected_xy: np.ndarray, neural_shape_xy: np.ndarray, labels: Mapping[str, np.ndarray], data: Mapping[str, np.ndarray], switch: np.ndarray, shape_switch: np.ndarray) -> dict[str, Any]:
    floor_ade, floor_fde = ft._trajectory_errors(floor_xy, labels)
    bridge_ade, bridge_fde = ft._trajectory_errors(bridge_selected_xy, labels)
    selected_ade, selected_fde = ft._trajectory_errors(shape_selected_xy, labels)
    neural_ade, neural_fde = ft._trajectory_errors(neural_shape_xy, labels)
    keys = _group_keys(data)
    counts = Counter(map(str, keys.tolist()))
    multi = np.asarray([counts[str(k)] >= 2 for k in keys], dtype=bool)
    floor_stats = jrc._joint_stats("floor", floor_xy, labels, keys, np.zeros(len(switch), dtype=bool))
    selected_stats = jrc._joint_stats("learned_shape_selected", shape_selected_xy, labels, keys, switch)
    neural_stats = jrc._joint_stats("learned_shape_without_fallback", neural_shape_xy, labels, keys, np.ones(len(switch), dtype=bool))
    horizon = labels["horizon"].astype(int)
    hard = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    return {
        "floor_ade": floor_ade,
        "bridge_ade": bridge_ade,
        "selected_ade": selected_ade,
        "neural_ade": neural_ade,
        "ade_metrics_vs_floor": s41._metrics(selected_ade, floor_ade, _metric_ds(labels), switch),
        "fde_metrics_vs_floor": s41._metrics(selected_fde, floor_fde, _metric_ds(labels), switch),
        "bridge_ade_metrics_vs_floor": s41._metrics(bridge_ade, floor_ade, _metric_ds(labels), switch),
        "learned_shape_gain_vs_bridge": {
            "all": _gain(selected_ade, bridge_ade, np.ones(len(switch), dtype=bool)),
            "t50": _gain(selected_ade, bridge_ade, horizon == 50),
            "t100": _gain(selected_ade, bridge_ade, horizon == 100),
            "hard_failure": _gain(selected_ade, bridge_ade, hard),
            "shape_switch_rate": float(np.mean(shape_switch)) if len(shape_switch) else 0.0,
        },
        "neural_shape_without_fallback_ade": s41._metrics(neural_ade, floor_ade, _metric_ds(labels), np.ones(len(switch), dtype=bool)),
        "multi_agent_ade_metrics": _safe_metrics(selected_ade, floor_ade, labels, switch, multi),
        "rollout_stats": {"floor": floor_stats, "selected": selected_stats, "neural_shape_without_fallback": neural_stats},
        "collision_delta_vs_floor_005": float(selected_stats["near_collision_rate_005"] - floor_stats["near_collision_rate_005"]),
        "smoothness_jagged_delta": float(selected_stats["smoothness"]["jagged_rate"] - floor_stats["smoothness"]["jagged_rate"]),
    }


def _build_bridge_bundle(data: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], endpoint_pred: Mapping[str, np.ndarray], gate_pred: Mapping[str, np.ndarray], policy_row: Mapping[str, Any]) -> dict[str, np.ndarray]:
    selected_delta, switch = bridge._apply_endpoint_policy(
        data,
        endpoint_pred,
        gate_pred,
        policy_row["policy"],
        {"all_horizons": None, "t50_only": {50}, "long_horizon": {50, 100}}[policy_row["variant"]],
    )
    selected_delta, switch, _guarded = bridge._guard(data, labels, selected_delta, switch, float(policy_row.get("min_sep", 0.0)))
    floor_delta = data["cand_delta"][:, 0, :].astype(np.float64)
    floor_xy = bridge._linear_waypoints_from_delta(data, floor_delta)
    bridge_xy = bridge._linear_waypoints_from_delta(data, selected_delta)
    return {"floor_xy": floor_xy, "bridge_xy": bridge_xy, "switch": switch.astype(bool)}


def _select_shape_policy_on_val(data: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], endpoint_pred: Mapping[str, np.ndarray], gate_pred: Mapping[str, np.ndarray], shape_pred: Mapping[str, np.ndarray], bridge_selection: Mapping[str, Any]) -> dict[str, Any]:
    bridge_bundle = _build_bridge_bundle(data, labels, endpoint_pred, gate_pred, bridge_selection["selected"])
    floor_ade, _floor_fde = ft._trajectory_errors(bridge_bundle["floor_xy"], labels)
    bridge_ade, _bridge_fde = ft._trajectory_errors(bridge_bundle["bridge_xy"], labels)
    residual_norm = np.linalg.norm(shape_pred["residual"].astype(np.float64), axis=2).mean(axis=1)
    risk = shape_pred["risk"].astype(np.float64)
    uncertainty = endpoint_pred["uncertainty"].astype(np.float64)
    base_switch = bridge_bundle["switch"]
    fast_rows = []
    horizon_variants = {"all_horizons": None, "t50_only": {50}, "t100_only": {100}, "long_horizon": {50, 100}}
    for mode in ["intermediate_only", "endpoint_half", "all_waypoints"]:
        for residual_scale in [0.25, 0.50, 0.75, 1.0]:
            for residual_clip in [0.04, 0.08, 0.16, 0.32]:
                shape_xy = _shape_xy(data, endpoint_pred, shape_pred, mode, residual_scale, residual_clip)
                for horizons_name, horizons in horizon_variants.items():
                    allowed = np.ones(len(base_switch), dtype=bool) if horizons is None else np.isin(data["horizon"].astype(int), sorted(horizons))
                    rn_grid = [float(v) for v in np.quantile(residual_norm[base_switch & allowed], [0.35, 0.55, 0.75, 0.90])] if np.any(base_switch & allowed) else [0.0]
                    risk_grid = [float(v) for v in np.quantile(risk[base_switch & allowed], [0.50, 0.70, 0.90])] if np.any(base_switch & allowed) else [0.0]
                    unc_grid = [float(v) for v in np.quantile(uncertainty[base_switch & allowed], [0.50, 0.70, 0.90])] if np.any(base_switch & allowed) else [0.0]
                    for residual_norm_max in rn_grid:
                        for risk_max in risk_grid:
                            for uncertainty_max in unc_grid:
                                shape_switch = base_switch & allowed & (residual_norm <= residual_norm_max) & (risk <= risk_max) & (uncertainty <= uncertainty_max)
                                if not np.any(shape_switch):
                                    continue
                                selected_xy = bridge_bundle["bridge_xy"].copy()
                                selected_xy[shape_switch] = shape_xy[shape_switch]
                                selected_ade, _selected_fde = ft._trajectory_errors(selected_xy, labels)
                                m = s41._metrics(selected_ade, floor_ade, _metric_ds(labels), base_switch)
                                horizon = labels["horizon"].astype(int)
                                hard = labels["hard"].astype(bool) | labels["failure"].astype(bool)
                                g = {
                                    "all": _gain(selected_ade, bridge_ade, np.ones(len(base_switch), dtype=bool)),
                                    "t50": _gain(selected_ade, bridge_ade, horizon == 50),
                                    "t100": _gain(selected_ade, bridge_ade, horizon == 100),
                                    "hard_failure": _gain(selected_ade, bridge_ade, hard),
                                    "shape_switch_rate": float(np.mean(shape_switch)) if len(shape_switch) else 0.0,
                                }
                                score = (
                                    m.get("all_improvement", 0.0)
                                    + 1.6 * m.get("t50_improvement", 0.0)
                                    + 1.0 * m.get("t100_improvement", 0.0)
                                    + 1.2 * m.get("hard_failure_improvement", 0.0)
                                    + 3.0 * max(g["all"], g["t50"], g["t100"], g["hard_failure"])
                                    - 40.0 * max(0.0, m.get("easy_degradation", 1.0) - 0.02)
                                )
                                fast_rows.append(
                                    {
                                        "mode": mode,
                                        "horizons": horizons_name,
                                        "residual_scale": residual_scale,
                                        "residual_clip": residual_clip,
                                        "residual_norm_max": residual_norm_max,
                                        "risk_max": risk_max,
                                        "endpoint_uncertainty_max": uncertainty_max,
                                        "score": float(score),
                                        "fast_metrics": {"ade": m, "shape_gain_vs_bridge": g, "shape_rows": int(np.sum(shape_switch))},
                                    }
                                )
    rows = []
    for candidate in sorted(fast_rows, key=lambda r: r["score"], reverse=True)[:32]:
        shape_xy = _shape_xy(data, endpoint_pred, shape_pred, str(candidate["mode"]), float(candidate["residual_scale"]), float(candidate["residual_clip"]))
        horizons = horizon_variants[str(candidate["horizons"])]
        allowed = np.ones(len(base_switch), dtype=bool) if horizons is None else np.isin(data["horizon"].astype(int), sorted(horizons))
        shape_switch = (
            base_switch
            & allowed
            & (residual_norm <= float(candidate["residual_norm_max"]))
            & (risk <= float(candidate["risk_max"]))
            & (uncertainty <= float(candidate["endpoint_uncertainty_max"]))
        )
        selected_xy = bridge_bundle["bridge_xy"].copy()
        selected_xy[shape_switch] = shape_xy[shape_switch]
        for min_sep in [0.0, 0.05]:
            guarded_xy, guarded_shape_switch, guarded = _guard(
                bridge_bundle["floor_xy"],
                selected_xy,
                bridge_bundle["bridge_xy"],
                labels,
                data,
                shape_switch,
                min_sep,
            )
            switch = base_switch.copy()
            shape_rows = int(np.sum(guarded_shape_switch))
            ev = _eval_xy(bridge_bundle["floor_xy"], bridge_bundle["bridge_xy"], guarded_xy, shape_xy, labels, data, switch, guarded_shape_switch)
            m = ev["ade_metrics_vs_floor"]
            g = ev["learned_shape_gain_vs_bridge"]
            score = (
                m.get("all_improvement", 0.0)
                + 1.6 * m.get("t50_improvement", 0.0)
                + 1.0 * m.get("t100_improvement", 0.0)
                + 1.2 * m.get("hard_failure_improvement", 0.0)
                + 3.0 * max(g["all"], g["t50"], g["t100"], g["hard_failure"])
                - 40.0 * max(0.0, m.get("easy_degradation", 1.0) - 0.02)
                - 8.0 * max(0.0, ev["collision_delta_vs_floor_005"] - 0.01)
            )
            eligible = (
                shape_rows > 0
                and m.get("all_improvement", 0.0) > 0.0
                and m.get("t50_improvement", 0.0) > 0.0
                and m.get("hard_failure_improvement", 0.0) > 0.0
                and m.get("easy_degradation", 1.0) <= 0.02
                and ev["collision_delta_vs_floor_005"] <= 0.01
                and max(g["all"], g["t50"], g["t100"], g["hard_failure"]) > 0.0
            )
            rows.append(
                {
                    **candidate,
                    "min_sep": min_sep,
                    "guarded_shape_rows": guarded,
                    "shape_rows": shape_rows,
                    "eligible": bool(eligible),
                    "score": float(score),
                    "val_metrics": {
                        "ade": m,
                        "shape_gain_vs_bridge": g,
                        "collision_delta_005": ev["collision_delta_vs_floor_005"],
                        "smoothness_jagged_delta": ev["smoothness_jagged_delta"],
                    },
                }
            )
    pool = [r for r in rows if r["eligible"]] or rows
    if not pool:
        return {"selected": {"eligible": False, "score": 0.0, "reason": "no_shape_policy_candidates"}, "candidate_count": len(fast_rows), "eligible_count": 0, "top_candidates": [], "fast_candidate_count": len(fast_rows)}
    selected = max(pool, key=lambda r: (bool(r["eligible"]), r["score"]))
    return {
        "selected": selected,
        "candidate_count": len(rows),
        "fast_candidate_count": len(fast_rows),
        "eligible_count": int(sum(r["eligible"] for r in rows)),
        "top_candidates": sorted(rows, key=lambda r: r["score"], reverse=True)[:10],
    }


def _apply_shape_policy(data: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], endpoint_pred: Mapping[str, np.ndarray], gate_pred: Mapping[str, np.ndarray], shape_pred: Mapping[str, np.ndarray], bridge_selection: Mapping[str, Any], shape_policy: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bridge_bundle = _build_bridge_bundle(data, labels, endpoint_pred, gate_pred, bridge_selection["selected"])
    if not shape_policy or shape_policy.get("reason"):
        return bridge_bundle["floor_xy"], bridge_bundle["bridge_xy"], bridge_bundle["bridge_xy"], bridge_bundle["switch"], np.zeros(len(bridge_bundle["switch"]), dtype=bool)
    shape_xy = _shape_xy(data, endpoint_pred, shape_pred, str(shape_policy["mode"]), float(shape_policy["residual_scale"]), float(shape_policy["residual_clip"]))
    horizons = {"all_horizons": None, "t50_only": {50}, "t100_only": {100}, "long_horizon": {50, 100}}[str(shape_policy["horizons"])]
    allowed = np.ones(len(bridge_bundle["switch"]), dtype=bool) if horizons is None else np.isin(data["horizon"].astype(int), sorted(horizons))
    residual_norm = np.linalg.norm(shape_pred["residual"].astype(np.float64), axis=2).mean(axis=1)
    shape_switch = (
        bridge_bundle["switch"]
        & allowed
        & (residual_norm <= float(shape_policy["residual_norm_max"]))
        & (shape_pred["risk"].astype(np.float64) <= float(shape_policy["risk_max"]))
        & (endpoint_pred["uncertainty"].astype(np.float64) <= float(shape_policy["endpoint_uncertainty_max"]))
    )
    selected_xy = bridge_bundle["bridge_xy"].copy()
    selected_xy[shape_switch] = shape_xy[shape_switch]
    selected_xy, guarded_shape_switch, _guarded = _guard(
        bridge_bundle["floor_xy"],
        selected_xy,
        bridge_bundle["bridge_xy"],
        labels,
        data,
        shape_switch,
        float(shape_policy.get("min_sep", 0.0)),
    )
    return bridge_bundle["floor_xy"], bridge_bundle["bridge_xy"], selected_xy, bridge_bundle["switch"], guarded_shape_switch


def _domain_data(split: str, domain: str) -> dict[str, np.ndarray]:
    data = dl._load_split(split)
    return bridge._subset(data, dl._domain_mask(data, domain))


def _evaluate_domain(domain: str) -> dict[str, Any]:
    train = _domain_data("train", domain)
    val = _domain_data("val", domain)
    test = _domain_data("test", domain)
    if min(len(train["horizon"]), len(val["horizon"]), len(test["horizon"])) < 500:
        return {"domain": domain, "status": "not_run", "reason": "not enough domain rows"}

    endpoint_training = dl._train_endpoint(domain, train, val)
    pred_train = dl._predict_endpoint(endpoint_training["checkpoint"], train)
    pred_val = dl._predict_endpoint(endpoint_training["checkpoint"], val)
    pred_test = dl._predict_endpoint(endpoint_training["checkpoint"], test)
    fde_train = dl._endpoint_fde(pred_train["delta"], train)
    fde_val = dl._endpoint_fde(pred_val["delta"], val)
    fde_test = dl._endpoint_fde(pred_test["delta"], test)
    gate = dl._train_gate(train, pred_train, fde_train)
    gate_val = dl._predict_gate(gate, val, pred_val, fde_val)
    gate_test = dl._predict_gate(gate, test, pred_test, fde_test)

    labels_train = bridge._align_full_labels("train", train)
    labels_val = bridge._align_full_labels("val", val)
    labels_test = bridge._align_full_labels("test", test)
    bridge_selection = bridge._select_policy_on_val(val, labels_val, pred_val, gate_val)
    shape_training = _train_shape_head(domain, train, val, pred_train, pred_val, labels_train, labels_val)
    shape_val = _predict_shape(shape_training["checkpoint"], val, pred_val)
    shape_test = _predict_shape(shape_training["checkpoint"], test, pred_test)
    shape_selection = _select_shape_policy_on_val(val, labels_val, pred_val, gate_val, shape_val, bridge_selection)
    floor_xy, bridge_xy, selected_xy, bridge_switch, shape_switch = _apply_shape_policy(test, labels_test, pred_test, gate_test, shape_test, bridge_selection, shape_selection["selected"])
    neural_shape_xy = _shape_xy(
        test,
        pred_test,
        shape_test,
        str(shape_selection["selected"].get("mode", "intermediate_only")),
        float(shape_selection["selected"].get("residual_scale", 1.0)),
        float(shape_selection["selected"].get("residual_clip", 0.16)),
    )
    ev = _eval_xy(floor_xy, bridge_xy, selected_xy, neural_shape_xy, labels_test, test, bridge_switch, shape_switch)
    m = ev["ade_metrics_vs_floor"]
    g = ev["learned_shape_gain_vs_bridge"]
    mm = ev["multi_agent_ade_metrics"]
    pass_gate = bool(
        m.get("all_improvement", 0.0) > 0.0
        and m.get("t50_improvement", 0.0) > 0.0
        and m.get("hard_failure_improvement", 0.0) > 0.0
        and m.get("easy_degradation", 1.0) <= 0.02
        and mm.get("all_improvement", 0.0) > 0.0
        and ev["collision_delta_vs_floor_005"] <= 0.01
        and ev["smoothness_jagged_delta"] <= 0.01
        and g["shape_switch_rate"] > 0.0
        and max(g["all"], g["t50"], g["t100"], g["hard_failure"]) > 0.0
    )
    return {
        "domain": domain,
        "source": "fresh_run",
        "status": "ok",
        "rows": {"train": int(len(train["horizon"])), "val": int(len(val["horizon"])), "test": int(len(test["horizon"]))},
        "endpoint_training": endpoint_training,
        "shape_training": shape_training,
        "bridge_selection": bridge_selection,
        "shape_selection": shape_selection,
        "bridge_direct_endpoint_without_fallback_test": dl._metrics(fde_test, test["floor_fde"], test, np.ones(len(test["horizon"]), dtype=bool)),
        **ev,
        "learned_waypoint_shape_gate": pass_gate,
        "caveat": "Learned waypoint-shape residuals are trained from past-only features and future waypoint labels. Future endpoints/waypoints are labels/eval only. The final selected policy is still protected by the endpoint bridge/floor fallback and is dataset-local raw-frame 2.5D evidence, not Stage5C/SMC/metric/seconds/true-3D/foundation.",
    }


def run_learned_waypoint_shape_bridge() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ft.build_full_trajectory_labels()
    domains = ["ETH_UCY", "TrajNet"]
    results = {domain: _evaluate_domain(domain) for domain in domains}
    positive = [d for d, r in results.items() if r.get("learned_waypoint_shape_gate")]
    result = {
        "source": "fresh_run",
        "protocol": "learned_waypoint_shape_bridge",
        "stage5c_executed": False,
        "smc_enabled": False,
        "metric_or_seconds_claim": False,
        "positive_domains": positive,
        "positive_domain_count": len(positive),
        "two_domain_learned_shape_gate": len(positive) >= 2,
        "domain_results": results,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_endpoint_label_eval_only": True,
            "future_waypoints_input": False,
            "future_waypoints_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "val_selected_policy": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "claim_boundary": {
            "endpoint_neural_dynamics": True,
            "learned_waypoint_shape_residual": True,
            "stage37_or_floor_protected": True,
            "latent_generative_rollout": False,
            "metric_or_seconds_claim": False,
            "true_3d": False,
            "foundation_world_model": False,
        },
    }
    write_json(REPORT_JSON, _jsonable(result))
    lines = [
        "# Stage41 Learned Waypoint-Shape Bridge",
        "",
        "- source: `fresh_run`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "- metric/seconds claim: `False`",
        f"- positive domains: `{positive}`",
        f"- two-domain learned-shape gate: `{result['two_domain_learned_shape_gate']}`",
        "",
        "| domain | shape mode | all ADE | t50 ADE | t100 ADE | hard ADE | easy | shape gain all/t50/t100/hard | shape switch | collision d005 | pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for domain, row in results.items():
        if row.get("status") != "ok":
            lines.append(f"| `{domain}` | `{row.get('reason')}` | 0 | 0 | 0 | 0 | 0 | 0/0/0 | 0 | 0 | `False` |")
            continue
        m = row["ade_metrics_vs_floor"]
        g = row["learned_shape_gain_vs_bridge"]
        selected = row["shape_selection"]["selected"]
        lines.append(
            f"| `{domain}` | `{selected.get('mode', 'none')}` | "
            f"{m.get('all_improvement', 0.0):.4f} | {m.get('t50_improvement', 0.0):.4f} | {m.get('t100_improvement', 0.0):.4f} | "
            f"{m.get('hard_failure_improvement', 0.0):.4f} | {m.get('easy_degradation', 0.0):.4f} | "
            f"{g.get('all', 0.0):.6f}/{g.get('t50', 0.0):.6f}/{g.get('t100', 0.0):.6f}/{g.get('hard_failure', 0.0):.6f} | {g.get('shape_switch_rate', 0.0):.6f} | "
            f"{row.get('collision_delta_vs_floor_005', 0.0):.4f} | `{row.get('learned_waypoint_shape_gate')}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This experiment is the next step after the endpoint-to-full linear bridge: it learns a waypoint-shape residual around the endpoint neural bridge from past-only features.",
            "- The model is still protected by the endpoint bridge/floor fallback; future waypoints are labels/eval only.",
            "- A positive learned-shape gain versus the linear bridge is required before claiming learned waypoint-shape contribution.",
            f"- no leakage: `{result['no_leakage']}`",
            f"- claim boundary: `{result['claim_boundary']}`",
        ]
    )
    write_md(REPORT_MD, lines)
    return result


def main_learned_waypoint_shape_bridge() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_learned_waypoint_shape_bridge()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_learned_waypoint_shape_bridge",
            status,
            started,
            [dl.DATA_DIR / "seq2seq_train.npz", dl.DATA_DIR / "seq2seq_val.npz", dl.DATA_DIR / "seq2seq_test.npz", ft.DATA_DIR / "full_trajectory_test.npz"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_learned_waypoint_shape_bridge()
