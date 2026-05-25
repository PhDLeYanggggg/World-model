from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_breakthrough as s41
from src import stage41_domain_local_neural_retrain as dl
from src import stage41_full_trajectory_world_state as ft
from src import stage41_joint_multiagent_consistency as jmc
from src import stage41_joint_rollout_consistency as jrc


OUT_DIR = dl.OUT_DIR
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
REPORT_JSON = OUT_DIR / "stage41_domain_local_full_trajectory_world_state.json"
REPORT_MD = OUT_DIR / "stage41_domain_local_full_trajectory_world_state.md"
BOOTSTRAP_N = 1000
THREADS = 4
BATCH = 768
EPOCHS = 3
SEED = 41559
EPS = 1e-6


TRIALS = [
    {"name": "domain_full_balanced", "width": 72, "dropout": 0.08, "lr": 8e-4, "hard_w": 2.0, "t50_w": 2.5, "t100_w": 2.0, "aux_w": 0.30, "seed": 1},
    {"name": "domain_full_long_horizon", "width": 88, "dropout": 0.10, "lr": 6e-4, "hard_w": 2.5, "t50_w": 2.0, "t100_w": 4.0, "aux_w": 0.35, "seed": 2},
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
    torch = ft._torch()
    torch.set_num_threads(THREADS)
    return torch


def _domain_mask(tensors: Mapping[str, Any], domain: str) -> np.ndarray:
    return tensors["raw"]["domain"].astype(str) == domain


def _subset_tensors(tensors: Mapping[str, Any], mask: np.ndarray) -> dict[str, Any]:
    torch = _torch()
    ids_np = np.where(mask)[0].astype(np.int64)
    ids = torch.tensor(ids_np, dtype=torch.long)
    out: dict[str, Any] = {}
    for key, value in tensors.items():
        if key in {"raw", "traj"}:
            out[key] = {k: (v[ids_np] if isinstance(v, np.ndarray) and v.shape[:1] == mask.shape else v) for k, v in value.items()}
        elif hasattr(value, "shape") and tuple(value.shape[:1]) == mask.shape:
            out[key] = value[ids]
        else:
            out[key] = value
    return out


def _domain_summary(tensors: Mapping[str, Any]) -> dict[str, Any]:
    raw = tensors["raw"]
    traj = tensors["traj"]
    full = np.all(traj["waypoint_valid"].astype(bool), axis=1)
    horizon = raw["horizon"].astype(int)
    hard = raw["hard"].astype(bool) | raw["failure"].astype(bool)
    easy = raw["easy"].astype(bool)
    return {
        "rows": int(len(horizon)),
        "full_waypoint_rows": int(np.sum(full)),
        "t10_rows": int(np.sum(horizon == 10)),
        "t25_rows": int(np.sum(horizon == 25)),
        "t50_rows": int(np.sum(horizon == 50)),
        "t100_rows": int(np.sum(horizon == 100)),
        "hard_failure_rate": float(np.mean(hard)) if len(hard) else 0.0,
        "easy_rate": float(np.mean(easy)) if len(easy) else 0.0,
        "sources": int(len(set(raw["source_file"].astype(str).tolist()))) if len(horizon) else 0,
        "scenes": int(len(set(raw["scene_id"].astype(str).tolist()))) if len(horizon) else 0,
    }


def _labels(tensors: Mapping[str, Any]) -> dict[str, np.ndarray]:
    raw = tensors["raw"]
    traj = tensors["traj"]
    return {
        "floor_fde": raw["floor_fde"].astype(np.float64),
        "candidate_fde": raw["candidate_fde"].astype(np.float64),
        "current_xy": raw["current_xy"].astype(np.float64),
        "future_xy": raw["future_xy"].astype(np.float64),
        "normalizer": raw["normalizer"].astype(np.float64),
        "cand_delta": raw["cand_delta"].astype(np.float64),
        "waypoint_xy": traj["waypoint_xy"].astype(np.float64),
        "waypoint_valid": traj["waypoint_valid"].astype(bool),
        "interaction": traj["interaction_future_close"].astype(bool),
        "occupancy": traj["occupancy_future_dense"].astype(bool),
        "physical": traj["physical_valid"].astype(bool),
        "horizon": raw["horizon"].astype(np.int64),
        "hard": (raw["hard"].astype(bool) | raw["failure"].astype(bool)),
        "easy": raw["easy"].astype(bool),
        "failure": raw["failure"].astype(bool),
        "domain": raw["domain"].astype(str),
        "scene_id": raw["scene_id"].astype(str),
        "source_file": raw["source_file"].astype(str),
        "ids": raw["ids"].astype(np.int64),
    }


def _keys(labels: Mapping[str, np.ndarray]) -> np.ndarray:
    data = s41._combined()
    ids = labels["ids"].astype(np.int64)
    frame = np.rint(data["frame_id"].astype(float)[ids]).astype(int)
    source = data["source_file"].astype(str)[ids]
    horizon = labels["horizon"].astype(int)
    return np.asarray([f"{source[i]}|{frame[i]}|{horizon[i]}" for i in range(len(ids))], dtype=object)


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


def _bootstrap(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray, seed: int) -> dict[str, Any]:
    ids = np.where(mask)[0]
    if len(ids) < 20:
        return {"low": 0.0, "mid": 0.0, "high": 0.0, "n": int(len(ids)), "bootstrap_n": 0}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(BOOTSTRAP_N):
        sample = rng.choice(ids, size=len(ids), replace=True)
        vals.append(1.0 - float(selected[sample].mean()) / max(float(floor[sample].mean()), EPS))
    return {"low": float(np.percentile(vals, 2.5)), "mid": float(np.percentile(vals, 50)), "high": float(np.percentile(vals, 97.5)), "n": int(len(ids)), "bootstrap_n": BOOTSTRAP_N}


def _train_trial(domain: str, trial: Mapping[str, Any], train: Mapping[str, Any], val: Mapping[str, Any]) -> dict[str, Any]:
    torch = _torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    ckpt = CHECKPOINT_DIR / f"stage41_domain_local_full_traj_{domain}_{trial['name']}.pt"
    heartbeat = OUT_DIR / f"stage41_domain_local_full_traj_{domain}_{trial['name']}_heartbeat.json"
    if ckpt.exists() and heartbeat.exists():
        try:
            payload = json.loads(heartbeat.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if int(payload.get("epoch", 0)) >= EPOCHS:
            return {"source": "cached_verified", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": payload.get("best", {})}
    model = ft._make_model(train["static"].shape[1], int(trial["width"]), float(trial["dropout"]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(trial["lr"]), weight_decay=1e-4)
    rng = np.random.default_rng(SEED + int(trial["seed"]) + sum(ord(c) for c in domain))
    best = {"val_loss": float("inf"), "epoch": 0}
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(train["agent_tokens"].shape[0])
        losses: list[float] = []
        model.train()
        for start in range(0, len(order), BATCH):
            ids = torch.tensor(order[start : start + BATCH], dtype=torch.long)
            out = model(train["agent_tokens"][ids], train["agent_mask"][ids], train["static"][ids])
            valid = train["waypoint_valid"][ids].float()
            row_w = 1.0 + float(trial["hard_w"]) * train["hard"][ids]
            row_w = row_w + float(trial["t50_w"]) * (train["horizon"][ids] == 50).float()
            row_w = row_w + float(trial["t100_w"]) * (train["horizon"][ids] == 100).float()
            err = torch.linalg.norm(out["waypoint_delta"] - train["waypoint_delta"][ids], dim=2)
            traj = ((F.smooth_l1_loss(out["waypoint_delta"], train["waypoint_delta"][ids], reduction="none").mean(dim=2) * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0) * row_w).mean()
            risk = (F.smooth_l1_loss(out["traj_risk"], torch.log1p((err * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)).detach(), reduction="none") * row_w).mean()
            interaction = F.binary_cross_entropy_with_logits(out["interaction_logit"], train["interaction"][ids])
            occupancy = F.binary_cross_entropy_with_logits(out["occupancy_logit"], train["occupancy"][ids])
            physical = F.binary_cross_entropy_with_logits(out["physical_logit"], train["physical"][ids])
            loss = traj + 0.35 * risk + float(trial["aux_w"]) * (interaction + occupancy + physical)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            out = model(val["agent_tokens"], val["agent_mask"], val["static"])
            valid = val["waypoint_valid"].float()
            val_loss = float(((F.smooth_l1_loss(out["waypoint_delta"], val["waypoint_delta"], reduction="none").mean(dim=2) * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)).mean().cpu())
        best_candidate = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses))}
        heartbeat.write_text(json.dumps({"domain": domain, "trial": dict(trial), "epoch": epoch, "best": best_candidate, "checkpoint": str(ckpt)}, ensure_ascii=False), encoding="utf-8")
        if val_loss < best["val_loss"]:
            best = best_candidate
            torch.save({"model": model.state_dict(), "trial": dict(trial), "static_dim": train["static"].shape[1], "best": best, "domain": domain}, ckpt)
    return {"source": "fresh_run", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best}


def _predict(path: str | Path, tensors: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    torch = _torch()
    model, _trial = ft._load_model(path)
    outs: dict[str, list[np.ndarray]] = {k: [] for k in ["waypoint_delta", "traj_risk", "interaction", "occupancy", "physical"]}
    with torch.no_grad():
        for start in range(0, tensors["agent_tokens"].shape[0], 2048):
            sl = slice(start, min(start + 2048, tensors["agent_tokens"].shape[0]))
            out = model(tensors["agent_tokens"][sl], tensors["agent_mask"][sl], tensors["static"][sl])
            outs["waypoint_delta"].append(out["waypoint_delta"].cpu().numpy())
            outs["traj_risk"].append(out["traj_risk"].cpu().numpy().reshape(-1))
            outs["interaction"].append(torch.sigmoid(out["interaction_logit"]).cpu().numpy().reshape(-1))
            outs["occupancy"].append(torch.sigmoid(out["occupancy_logit"]).cpu().numpy().reshape(-1))
            outs["physical"].append(torch.sigmoid(out["physical_logit"]).cpu().numpy().reshape(-1))
    pred = {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in outs.items()}
    return pred, _labels(tensors)


def _apply_policy_xy(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> dict[str, Any]:
    selected_ade, selected_fde, switch, floor_ade = ft._apply_policy(pred, labels, policy)
    floor_xy = ft._floor_waypoints(labels)
    neural_xy = ft._pred_waypoints(pred, labels)
    selected_xy = floor_xy.copy()
    selected_xy[switch] = neural_xy[switch]
    floor_fde = ft._trajectory_errors(floor_xy, labels)[1]
    neural_ade, neural_fde = ft._trajectory_errors(neural_xy, labels)
    return {"selected_ade": selected_ade, "selected_fde": selected_fde, "switch": switch, "floor_ade": floor_ade, "floor_fde": floor_fde, "floor_xy": floor_xy, "neural_xy": neural_xy, "selected_xy": selected_xy, "neural_ade": neural_ade, "neural_fde": neural_fde}


def _fast_apply_params(
    pred: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    floor_ade: np.ndarray,
    neural_ade: np.ndarray,
    mask: np.ndarray,
    params: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    hard = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    easy = labels["easy"].astype(bool)
    local = (
        mask
        & (pred["traj_risk"] <= float(params.get("traj_risk_max", 1e9)))
        & (pred["physical"] >= float(params.get("physical_prob_min", 0.0)))
    )
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
        keep = np.zeros(len(local), dtype=bool)
        keep[ids[np.argsort(pred["traj_risk"][ids])[:keep_n]]] = True
        local &= keep
    selected = floor_ade.copy()
    selected[local] = neural_ade[local]
    return selected, local


def _fast_fit_policy(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    floor_xy = ft._floor_waypoints(labels)
    neural_xy = ft._pred_waypoints(pred, labels)
    floor_ade, _floor_fde = ft._trajectory_errors(floor_xy, labels)
    neural_ade, _neural_fde = ft._trajectory_errors(neural_xy, labels)
    domain = labels["domain"].astype(str)
    horizon = labels["horizon"].astype(int)
    policy = {"type": "domain_local_fast_full_trajectory_policy", "slices": {}}
    selected_all = floor_ade.copy()
    switch_all = np.zeros(len(floor_ade), dtype=bool)
    diagnostics: dict[str, Any] = {}
    for d in sorted(set(domain.tolist())):
        for h in [10, 25, 50, 100]:
            mask = (domain == d) & (horizon == h)
            if int(np.sum(mask)) < 80:
                continue
            risk = pred["traj_risk"][mask]
            risk_grid = [float(v) for v in np.quantile(risk, [0.05, 0.15, 0.30, 0.50])] if len(risk) else [0.0]
            best_params: dict[str, Any] | None = None
            best_score = -1e18
            best_metrics: dict[str, Any] | None = None
            for traj_risk_max in risk_grid:
                for physical_prob_min in [0.0, 0.35, 0.55]:
                    for max_switch in [0.0, 0.05, 0.10, 0.20, 0.40]:
                        for hard_only in [False, True]:
                            params = {
                                "traj_risk_max": traj_risk_max,
                                "physical_prob_min": physical_prob_min,
                                "max_switch": max_switch,
                                "easy_block": True,
                                "hard_only": hard_only,
                            }
                            selected, switch = _fast_apply_params(pred, labels, floor_ade, neural_ade, mask, params)
                            metrics = s41._metrics(selected[mask], floor_ade[mask], {k: v[mask] for k, v in _metric_ds(labels).items()}, switch[mask])
                            if metrics.get("all_improvement", 0.0) <= 0.0 or metrics.get("easy_degradation", 0.0) > 0.02:
                                continue
                            score = (
                                metrics.get("all_improvement", 0.0)
                                + 1.4 * metrics.get("t50_improvement", 0.0)
                                + metrics.get("t100_improvement", 0.0)
                                + 1.2 * metrics.get("hard_failure_improvement", 0.0)
                            )
                            if score > best_score:
                                best_score = score
                                best_params = params
                                best_metrics = metrics
            if best_params:
                selected, switch = _fast_apply_params(pred, labels, floor_ade, neural_ade, mask, best_params)
                selected_all[mask] = selected[mask]
                switch_all[mask] = switch[mask]
                policy["slices"][f"{d}|{h}"] = best_params
            diagnostics[f"{d}|{h}"] = {"selected": bool(best_params), "val_score": float(best_score if best_params else 0.0), "val_metrics": best_metrics or {"rows": int(np.sum(mask)), "all_improvement": 0.0}}
    metrics = s41._metrics(selected_all, floor_ade, _metric_ds(labels), switch_all)
    metrics["slice_diagnostics"] = diagnostics
    return policy, metrics


def _apply_proximity_guard(bundle: Mapping[str, Any], min_sep: float) -> tuple[np.ndarray, np.ndarray, int]:
    switch = bundle["switch"].copy()
    if min_sep <= 0:
        return bundle["selected_xy"], switch, 0
    labels = bundle["labels"]
    keys = bundle["keys"]
    floor_min = jmc._min_group_distance(bundle["floor_xy"], keys, labels["normalizer"].astype(np.float64))
    selected_min = jmc._min_group_distance(bundle["selected_xy"], keys, labels["normalizer"].astype(np.float64))
    guard = switch & np.isfinite(selected_min) & (selected_min < min_sep) & (selected_min < floor_min)
    out_switch = switch.copy()
    out_switch[guard] = False
    selected_xy = bundle["floor_xy"].copy()
    selected_xy[out_switch] = bundle["neural_xy"][out_switch]
    return selected_xy, out_switch, int(np.sum(guard))


def _evaluate_xy(selected_xy: np.ndarray, switch: np.ndarray, floor_xy: np.ndarray, neural_xy: np.ndarray, labels: Mapping[str, np.ndarray], keys: np.ndarray, *, bootstrap: bool = True) -> dict[str, Any]:
    selected_ade, selected_fde = ft._trajectory_errors(selected_xy, labels)
    floor_ade, floor_fde = ft._trajectory_errors(floor_xy, labels)
    neural_ade, neural_fde = ft._trajectory_errors(neural_xy, labels)
    group_counts = np.asarray([Counter(map(str, keys.tolist()))[str(k)] for k in keys], dtype=np.int32)
    multi = group_counts >= 2
    ade_metrics = s41._metrics(selected_ade, floor_ade, _metric_ds(labels), switch)
    fde_metrics = s41._metrics(selected_fde, floor_fde, _metric_ds(labels), switch)
    multi_ade = _safe_metrics(selected_ade, floor_ade, labels, switch, multi)
    floor_stats = jrc._joint_stats("floor", floor_xy, labels, keys, np.zeros(len(switch), dtype=bool))
    selected_stats = jrc._joint_stats("domain_local_full_trajectory_selected", selected_xy, labels, keys, switch)
    neural_stats = jrc._joint_stats("domain_local_full_trajectory_neural_without_fallback", neural_xy, labels, keys, np.ones(len(switch), dtype=bool))
    collision_delta = float(selected_stats["near_collision_rate_005"] - floor_stats["near_collision_rate_005"])
    smoothness_delta = float(selected_stats["smoothness"]["jagged_rate"] - floor_stats["smoothness"]["jagged_rate"])
    horizon = labels["horizon"].astype(int)
    hard = labels["hard"].astype(bool) | labels["failure"].astype(bool)
    bootstrap_report = {}
    if bootstrap:
        bootstrap_report = {
            "all": _bootstrap(selected_ade, floor_ade, np.ones(len(switch), dtype=bool), SEED),
            "t50": _bootstrap(selected_ade, floor_ade, horizon == 50, SEED + 1),
            "t100": _bootstrap(selected_ade, floor_ade, horizon == 100, SEED + 2),
            "hard_failure": _bootstrap(selected_ade, floor_ade, hard, SEED + 3),
            "multi_agent": _bootstrap(selected_ade, floor_ade, multi, SEED + 4),
        }
    return {
        "ade_metrics_vs_floor": ade_metrics,
        "fde_metrics_vs_floor": fde_metrics,
        "multi_agent_ade_metrics": multi_ade,
        "neural_without_fallback_ade": s41._metrics(neural_ade, floor_ade, _metric_ds(labels), np.ones(len(switch), dtype=bool)),
        "rollout_stats": {"floor": floor_stats, "selected": selected_stats, "neural_without_fallback": neural_stats},
        "collision_delta_vs_floor_005": collision_delta,
        "smoothness_jagged_delta": smoothness_delta,
        "bootstrap_ade": bootstrap_report,
    }


def _select_guard_on_val(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> dict[str, Any]:
    base = _apply_policy_xy(pred, labels, policy)
    m = s41._metrics(base["selected_ade"], base["floor_ade"], _metric_ds(labels), base["switch"])
    score = (
        m.get("all_improvement", 0.0)
        + 1.4 * m.get("t50_improvement", 0.0)
        + m.get("t100_improvement", 0.0)
        + 1.2 * m.get("hard_failure_improvement", 0.0)
        - 30.0 * max(0.0, m.get("easy_degradation", 1.0) - 0.02)
    )
    row = {
        "min_sep": 0.0,
        "guarded_off": 0,
        "eligible": bool(m.get("all_improvement", 0.0) > 0 and m.get("t50_improvement", 0.0) > 0 and m.get("hard_failure_improvement", 0.0) > 0 and m.get("easy_degradation", 1.0) <= 0.02),
        "score": score,
        "val_metrics": m,
        "val_collision_delta_005": "not_run_fast_validation_guard_disabled",
    }
    return {"selected": row, "candidates": [row], "note": "Fast evidence run: validation proximity search is disabled to avoid repeated group-distance scans; test still performs full same-frame proximity/smoothness audit."}


def _evaluate_domain(domain: str, train_all: Mapping[str, Any], val_all: Mapping[str, Any], test_all: Mapping[str, Any]) -> dict[str, Any]:
    train = _subset_tensors(train_all, _domain_mask(train_all, domain))
    val = _subset_tensors(val_all, _domain_mask(val_all, domain))
    test = _subset_tensors(test_all, _domain_mask(test_all, domain))
    if min(train["agent_tokens"].shape[0], val["agent_tokens"].shape[0], test["agent_tokens"].shape[0]) < 500:
        return {"domain": domain, "status": "not_run", "reason": "not enough domain train/val/test rows for learned full-waypoint dynamics", "rows": {"train": int(train["agent_tokens"].shape[0]), "val": int(val["agent_tokens"].shape[0]), "test": int(test["agent_tokens"].shape[0])}}
    trials = {}
    best_score = -1e18
    best: dict[str, Any] | None = None
    for trial in TRIALS:
        training = _train_trial(domain, trial, train, val)
        pred_val, labels_val = _predict(training["checkpoint"], val)
        policy, val_metrics = _fast_fit_policy(pred_val, labels_val)
        guard = _select_guard_on_val(pred_val, labels_val, policy)
        score = guard["selected"]["score"]
        trials[trial["name"]] = {"training": training, "policy": policy, "val_metrics": val_metrics, "proximity_guard_selection": guard, "val_score": score}
        if score > best_score:
            best_score = score
            best = trials[trial["name"]]
    assert best is not None
    pred_test, labels_test = _predict(best["training"]["checkpoint"], test)
    base = _apply_policy_xy(pred_test, labels_test, best["policy"])
    bundle = {**base, "labels": labels_test, "keys": _keys(labels_test)}
    selected_xy, switch, guarded_off = _apply_proximity_guard(bundle, float(best["proximity_guard_selection"]["selected"]["min_sep"]))
    ev = _evaluate_xy(selected_xy, switch, bundle["floor_xy"], bundle["neural_xy"], labels_test, bundle["keys"])
    m = ev["ade_metrics_vs_floor"]
    mm = ev["multi_agent_ade_metrics"]
    pass_gate = bool(
        m.get("all_improvement", 0.0) > 0
        and m.get("t50_improvement", 0.0) > 0
        and m.get("hard_failure_improvement", 0.0) > 0
        and m.get("easy_degradation", 1.0) <= 0.02
        and mm.get("all_improvement", 0.0) > 0
        and ev["collision_delta_vs_floor_005"] <= 0.01
        and ev["smoothness_jagged_delta"] <= 0.01
        and (ev["bootstrap_ade"]["all"]["low"] > 0 or ev["bootstrap_ade"]["t50"]["low"] > 0 or ev["bootstrap_ade"]["hard_failure"]["low"] > 0)
    )
    return {
        "domain": domain,
        "source": "fresh_run",
        "status": "ok",
        "summary": {"train": _domain_summary(train), "val": _domain_summary(val), "test": _domain_summary(test)},
        "best_trial": best["training"]["checkpoint"],
        "best_val_score": best_score,
        "policy": best["policy"],
        "proximity_guard_selection": best["proximity_guard_selection"],
        "test_guarded_off": guarded_off,
        **ev,
        "domain_local_full_trajectory_world_state_gate": pass_gate,
        "caveat": "Learned domain-local full-waypoint neural dynamics are trained from past-only all-agent tokens and future waypoint labels. Evaluation remains dataset-local raw-frame 2.5D; no Stage5C/SMC/metric/seconds claim.",
        "trials": trials,
    }


def run_domain_local_full_trajectory_world_state() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ft.build_full_trajectory_labels()
    train_all = ft._load_tensors("train")
    val_all = ft._load_tensors("val")
    test_all = ft._load_tensors("test")
    train_domains = set(train_all["raw"]["domain"].astype(str).tolist())
    val_domains = set(val_all["raw"]["domain"].astype(str).tolist())
    test_domains = set(test_all["raw"]["domain"].astype(str).tolist())
    domains = sorted(train_domains & val_domains & test_domains)
    blockers = {
        d: {
            "status": "not_run",
            "reason": "domain lacks strict train/val/test coverage in current all-agent split",
            "has_train": d in train_domains,
            "has_val": d in val_domains,
            "has_test": d in test_domains,
        }
        for d in sorted((train_domains | val_domains | test_domains) - set(domains))
    }
    results = {domain: _evaluate_domain(domain, train_all, val_all, test_all) for domain in domains}
    positive_domains = [domain for domain, row in results.items() if row.get("domain_local_full_trajectory_world_state_gate")]
    failure_taxonomy: dict[str, Any] = {}
    for domain, row in results.items():
        if row.get("status") != "ok" or row.get("domain_local_full_trajectory_world_state_gate"):
            continue
        m = row.get("ade_metrics_vs_floor", {})
        fm = row.get("fde_metrics_vs_floor", {})
        reasons = []
        if m.get("all_improvement", 0.0) <= 0:
            reasons.append("all_ade_not_positive")
        if m.get("t50_improvement", 0.0) <= 0:
            reasons.append("t50_ade_not_positive")
        if m.get("t100_improvement", 0.0) <= 0:
            reasons.append("t100_ade_not_positive")
        if m.get("hard_failure_improvement", 0.0) <= 0:
            reasons.append("hard_failure_ade_not_positive")
        if m.get("easy_degradation", 1.0) > 0.02:
            reasons.append("easy_degradation_over_2pct")
        if row.get("collision_delta_vs_floor_005", 1.0) > 0.01:
            reasons.append("same_frame_proximity_delta_unsafe")
        if (row.get("neural_without_fallback_ade") or {}).get("easy_degradation", 0.0) > 0.02:
            reasons.append("ungated_neural_catastrophic_easy_degradation")
        if fm.get("all_improvement", 0.0) > 0 and m.get("all_improvement", 0.0) <= 0:
            reasons.append("endpoint_fde_positive_but_waypoint_ade_negative")
        failure_taxonomy[domain] = {
            "reasons": reasons,
            "ade_all": m.get("all_improvement"),
            "ade_t50": m.get("t50_improvement"),
            "ade_t100": m.get("t100_improvement"),
            "fde_all": fm.get("all_improvement"),
            "fde_t50": fm.get("t50_improvement"),
            "collision_delta_vs_floor_005": row.get("collision_delta_vs_floor_005"),
            "next_fix": "train a validation-selected efficient proximity guard and horizon-specific waypoint policy; do not deploy learned full-waypoint domain-local model yet.",
        }
    result = {
        "source": "fresh_run",
        "protocol": "domain_local_learned_full_waypoint_world_state",
        "domains_run": domains,
        "domain_blockers": blockers,
        "positive_domains": positive_domains,
        "positive_domain_count": int(len(positive_domains)),
        "two_domain_full_trajectory_gate": bool(len(positive_domains) >= 2),
        "domain_results": results,
        "failure_taxonomy": failure_taxonomy,
        "no_leakage": {
            "future_waypoints_input": False,
            "future_waypoints_label_eval_only": True,
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "val_selected_policy": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "claim_boundary": {
            "learned_full_waypoint_neural_dynamics": True,
            "all_active_agent_world_state_audit": True,
            "latent_generative_rollout": False,
            "metric_or_seconds_claim": False,
            "true_3d": False,
            "foundation_world_model": False,
        },
    }
    write_json(REPORT_JSON, _jsonable(result))
    lines = [
        "# Stage41 Domain-Local Learned Full-Trajectory World-State Audit",
        "",
        "- source: `fresh_run`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "- metric/seconds claim: `False`",
        f"- domains run: `{domains}`",
        f"- blockers: `{blockers}`",
        f"- positive domains: `{positive_domains}`",
        f"- two-domain full-trajectory gate: `{result['two_domain_full_trajectory_gate']}`",
        f"- failure taxonomy: `{failure_taxonomy}`",
        "",
        "| domain | train/val/test rows | all | t50 | t100 | hard | easy | multi all | collision d005 | pass |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for domain, row in results.items():
        if row.get("status") != "ok":
            lines.append(f"| `{domain}` | `{row.get('reason')}` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | `False` |")
            continue
        m = row["ade_metrics_vs_floor"]
        mm = row["multi_agent_ade_metrics"]
        s = row["summary"]
        lines.append(
            f"| `{domain}` | `{s['train']['rows']}/{s['val']['rows']}/{s['test']['rows']}` | "
            f"{float(m.get('all_improvement', 0.0)):.4f} | {float(m.get('t50_improvement', 0.0)):.4f} | "
            f"{float(m.get('t100_improvement', 0.0)):.4f} | {float(m.get('hard_failure_improvement', 0.0)):.4f} | "
            f"{float(m.get('easy_degradation', 0.0)):.4f} | {float(mm.get('all_improvement', 0.0)):.4f} | "
            f"{float(row.get('collision_delta_vs_floor_005', 0.0)):.4f} | `{row['domain_local_full_trajectory_world_state_gate']}` |"
        )
    lines.extend(["", f"- no leakage: `{result['no_leakage']}`", f"- claim boundary: `{result['claim_boundary']}`"])
    write_md(REPORT_MD, lines)
    return result


def main_domain_local_full_trajectory_world_state() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_domain_local_full_trajectory_world_state()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_domain_local_full_trajectory_world_state",
            status,
            started,
            [ft.DATA_DIR / "all_agent_train.npz", ft.DATA_DIR / "all_agent_val.npz", ft.DATA_DIR / "all_agent_test.npz", ft.DATA_DIR / "full_trajectory_train.npz"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_domain_local_full_trajectory_world_state()
