from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_all_agent as aa
from src import stage41_breakthrough as s41
from src import stage41_fresh_confirmation as fresh
from src import stage41_full_trajectory_world_state as ft
from src import stage41_joint_multiagent_consistency as jmc


OUT_DIR = fresh.OUT_DIR
DATA_DIR = fresh.DATA_DIR
CHECKPOINT_DIR = fresh.CHECKPOINT_DIR
LEDGER_JSONL = fresh.LEDGER_JSONL
THREADS = 4
BATCH = 1024
EPOCHS = 4
SEED = 4213
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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _domain_one_hot(domain: np.ndarray) -> np.ndarray:
    names = ["ETH_UCY", "TrajNet", "UCY"]
    out = np.zeros((len(domain), len(names)), dtype=np.float32)
    mapping = {name: i for i, name in enumerate(names)}
    for i, value in enumerate(domain.astype(str)):
        if value in mapping:
            out[i, mapping[value]] = 1.0
    return out


def _horizon_features(horizon: np.ndarray) -> np.ndarray:
    vals = [10, 25, 50, 100]
    out = np.zeros((len(horizon), len(vals) + 1), dtype=np.float32)
    for i, h in enumerate(vals):
        out[:, i] = (horizon.astype(int) == h).astype(np.float32)
    out[:, -1] = horizon.astype(np.float32) / 100.0
    return out


def _group_scalar_features(xy: np.ndarray, labels: Mapping[str, np.ndarray], meta: Mapping[str, np.ndarray]) -> np.ndarray:
    key = meta["key"]
    norm = labels["normalizer"].astype(np.float64)
    min_dist = jmc._min_group_distance(xy, key, norm)
    group_count = np.zeros(len(key), dtype=np.float32)
    counts = Counter(map(str, key.tolist()))
    for i, value in enumerate(key):
        group_count[i] = float(counts[str(value)])
    finite_dist = np.where(np.isfinite(min_dist), min_dist, 9.0).astype(np.float32)
    return np.stack([group_count / 10.0, finite_dist], axis=1).astype(np.float32)


def _feature_bundle(split: str) -> dict[str, np.ndarray]:
    pred, labels, ref = jmc._split_predictions(split)
    meta = jmc._group_metadata(split)
    ds = ft._fresh_ds(split)
    base = jmc._base_selection(pred, labels, ref["policy"])
    floor_xy = ft._floor_waypoints(labels)
    neural_xy = ft._pred_waypoints(pred, labels)
    current_group = _group_scalar_features(labels["current_xy"][:, None, :].repeat(len(ft.WAYPOINT_FRAC), axis=1), labels, meta)
    floor_group = _group_scalar_features(floor_xy, labels, meta)
    neural_group = _group_scalar_features(neural_xy, labels, meta)
    gain = (base["floor_ade"] - base["neural_ade"]).astype(np.float32)
    harm = (base["neural_ade"] > base["floor_ade"]).astype(np.float32)
    switch_label = (gain > 0).astype(np.float32)
    pred_signals = np.stack(
        [
            pred["traj_risk"].astype(np.float32),
            pred["interaction"].astype(np.float32),
            pred["occupancy"].astype(np.float32),
            pred["physical"].astype(np.float32),
            np.linalg.norm(neural_xy[:, -1] - labels["current_xy"], axis=1).astype(np.float32) / np.maximum(labels["normalizer"].astype(np.float32), EPS),
        ],
        axis=1,
    )
    # Keep the inherited reference policy out of the model inputs. Earlier
    # Stage41 reference policies use evaluation labels for easy/hard reporting;
    # this distiller must be deployable from past/static/prediction signals only.
    neighbor_flags = (ds["neighbor_counts"].astype(np.float32) / 10.0)[:, None]
    features = np.concatenate(
        [
            _norm_static(ds["static"]),
            pred_signals,
            current_group,
            floor_group,
            neural_group,
            neighbor_flags,
            _domain_one_hot(labels["domain"]),
            _horizon_features(labels["horizon"]),
        ],
        axis=1,
    ).astype(np.float32)
    return {
        "x": features,
        "gain": gain,
        "harm": harm,
        "switch_label": switch_label,
        "base_switch": base["switch"].astype(bool),
        "floor_ade": base["floor_ade"].astype(np.float64),
        "floor_fde": base["floor_fde"].astype(np.float64),
        "neural_ade": base["neural_ade"].astype(np.float64),
        "neural_fde": base["neural_fde"].astype(np.float64),
        "domain": labels["domain"],
        "horizon": labels["horizon"],
        "hard": labels["hard"],
        "failure": labels["failure"],
        "easy": labels["easy"],
        "candidate_fde": labels["candidate_fde"],
        "feature_dim": np.asarray([features.shape[1]], dtype=np.int64),
    }


def _make_model(dim: int, width: int, dropout: float):
    torch = _torch()
    import torch.nn as nn

    class JointPolicyDistiller(nn.Module):
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
            )
            self.switch = nn.Linear(width, 1)
            self.gain = nn.Linear(width, 1)
            self.harm = nn.Linear(width, 1)

        def forward(self, x):
            h = self.net(x)
            return {
                "switch_logit": self.switch(h).squeeze(-1),
                "gain": self.gain(h).squeeze(-1),
                "harm_logit": self.harm(h).squeeze(-1),
            }

    return JointPolicyDistiller()


TRIALS = [
    {"name": "joint_distill_nobase_balanced", "width": 96, "dropout": 0.08, "lr": 8e-4, "gain_w": 1.0, "switch_w": 1.0, "harm_w": 1.2, "hard_w": 1.5, "seed": 1},
    {"name": "joint_distill_nobase_t50_hard", "width": 112, "dropout": 0.08, "lr": 7e-4, "gain_w": 1.2, "switch_w": 1.2, "harm_w": 1.4, "hard_w": 2.5, "seed": 2},
    {"name": "joint_distill_nobase_conservative", "width": 80, "dropout": 0.12, "lr": 7e-4, "gain_w": 0.8, "switch_w": 0.8, "harm_w": 2.0, "hard_w": 1.5, "seed": 3},
]


def _train_trial(trial: Mapping[str, Any]) -> dict[str, Any]:
    torch = _torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    train = _feature_bundle("train")
    val = _feature_bundle("val")
    ckpt = CHECKPOINT_DIR / f"stage41_{trial['name']}.pt"
    heartbeat = OUT_DIR / f"{trial['name']}_heartbeat.json"
    if ckpt.exists() and heartbeat.exists():
        payload = read_json(heartbeat, {})
        if int(payload.get("epoch", 0)) >= EPOCHS:
            return {"source": "cached_verified", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": payload.get("best", {})}
    x = torch.tensor(train["x"])
    y_gain = torch.tensor(train["gain"])
    y_switch = torch.tensor(train["switch_label"])
    y_harm = torch.tensor(train["harm"])
    hard = torch.tensor((train["hard"].astype(bool) | train["failure"].astype(bool)).astype(np.float32))
    vx = torch.tensor(val["x"])
    vg = torch.tensor(val["gain"])
    vs = torch.tensor(val["switch_label"])
    vh = torch.tensor(val["harm"])
    model = _make_model(x.shape[1], int(trial["width"]), float(trial["dropout"]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(trial["lr"]), weight_decay=1e-4)
    rng = np.random.default_rng(SEED + int(trial["seed"]))
    best = {"val_loss": float("inf"), "epoch": 0}
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(x.shape[0])
        losses: list[float] = []
        model.train()
        for start in range(0, len(order), BATCH):
            ids = torch.tensor(order[start : start + BATCH], dtype=torch.long)
            out = model(x[ids])
            row_w = 1.0 + float(trial["hard_w"]) * hard[ids] + 2.0 * torch.clamp(torch.abs(y_gain[ids]), max=2.0)
            gain_loss = (F.smooth_l1_loss(out["gain"], y_gain[ids], reduction="none") * row_w).mean()
            switch_loss = (F.binary_cross_entropy_with_logits(out["switch_logit"], y_switch[ids], reduction="none") * row_w).mean()
            harm_loss = (F.binary_cross_entropy_with_logits(out["harm_logit"], y_harm[ids], reduction="none") * row_w).mean()
            loss = float(trial["gain_w"]) * gain_loss + float(trial["switch_w"]) * switch_loss + float(trial["harm_w"]) * harm_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            out = model(vx)
            val_loss = float(
                (
                    F.smooth_l1_loss(out["gain"], vg)
                    + 0.5 * F.binary_cross_entropy_with_logits(out["switch_logit"], vs)
                    + 0.8 * F.binary_cross_entropy_with_logits(out["harm_logit"], vh)
                ).cpu()
            )
        best_candidate = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses))}
        heartbeat.write_text(json.dumps({"trial": dict(trial), "epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "checkpoint": str(ckpt), "best": best_candidate}), encoding="utf-8")
        if val_loss < best["val_loss"]:
            best = best_candidate
            torch.save({"model": model.state_dict(), "trial": dict(trial), "dim": int(x.shape[1]), "best": best}, ckpt)
    return {"source": "fresh_run", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best}


def _predict_checkpoint(path: str | Path, split: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    torch = _torch()
    payload = torch.load(path, map_location="cpu")
    trial = payload["trial"]
    data = _feature_bundle(split)
    model = _make_model(int(payload["dim"]), int(trial["width"]), float(trial["dropout"]))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    outs = {"switch_prob": [], "gain_pred": [], "harm_prob": []}
    x = torch.tensor(data["x"])
    with torch.no_grad():
        for start in range(0, x.shape[0], 4096):
            out = model(x[start : start + 4096])
            outs["switch_prob"].append(torch.sigmoid(out["switch_logit"]).cpu().numpy())
            outs["gain_pred"].append(out["gain"].cpu().numpy())
            outs["harm_prob"].append(torch.sigmoid(out["harm_logit"]).cpu().numpy())
    pred = {k: np.concatenate(v).astype(np.float32) for k, v in outs.items()}
    return pred, data


def _metric(selected_ade: np.ndarray, floor_ade: np.ndarray, data: Mapping[str, np.ndarray], switch: np.ndarray) -> dict[str, Any]:
    base = {
        "horizon": data["horizon"],
        "hard": data["hard"],
        "failure": data["failure"],
        "easy": data["easy"],
        "domain": data["domain"],
        "candidate_fde": data["candidate_fde"],
    }
    return s41._metrics(selected_ade, floor_ade, base, switch)


def _apply_policy(scores: Mapping[str, np.ndarray], data: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mode = str(policy.get("mode", "distiller_only"))
    switch = np.zeros(len(data["floor_ade"]), dtype=bool)
    if mode == "base_plus_distiller":
        switch |= data["base_switch"].astype(bool)
    domain = data["domain"].astype(str)
    horizon = data["horizon"].astype(int)
    for key, params in (policy.get("slices") or {}).items():
        d, h_s = key.split("|")
        mask = (domain == d) & (horizon == int(h_s))
        if not np.any(mask):
            continue
        expand = (
            mask
            & (scores["switch_prob"] >= float(params.get("switch_min", 0.5)))
            & (scores["gain_pred"] >= float(params.get("gain_min", 0.0)))
            & (scores["harm_prob"] <= float(params.get("harm_max", 1.0)))
        )
        max_switch = float(params.get("max_switch", 1.0))
        if max_switch < 1.0 and np.any(expand):
            ids = np.where(expand)[0]
            utility = scores["gain_pred"][ids] * scores["switch_prob"][ids] * (1.0 - scores["harm_prob"][ids])
            keep_n = max(1, int(max_switch * int(np.sum(mask))))
            keep = np.zeros(len(expand), dtype=bool)
            keep[ids[np.argsort(utility)[::-1][:keep_n]]] = True
            expand &= keep
        if mode == "base_plus_distiller":
            guard = (
                mask
                & switch
                & (scores["harm_prob"] >= float(params.get("guard_harm_min", 1.1)))
                & (scores["gain_pred"] <= float(params.get("guard_gain_max", -1e9)))
            )
            switch[guard] = False
        switch |= expand
    selected_ade = data["floor_ade"].copy()
    selected_fde = data["floor_fde"].copy()
    selected_ade[switch] = data["neural_ade"][switch]
    selected_fde[switch] = data["neural_fde"][switch]
    return selected_ade, selected_fde, switch


def _score(metrics: Mapping[str, Any]) -> float:
    max_domain_easy = max([float(row.get("easy_degradation", 0.0)) for row in (metrics.get("by_domain") or {}).values()] or [0.0])
    return (
        1.2 * float(metrics.get("all_improvement", 0.0))
        + 1.6 * float(metrics.get("t50_improvement", 0.0))
        + 1.0 * float(metrics.get("t100_improvement", 0.0))
        + 1.3 * float(metrics.get("hard_failure_improvement", 0.0))
        - 35.0 * max(0.0, float(metrics.get("easy_degradation", 1.0)) - 0.02)
        - 40.0 * max(0.0, max_domain_easy - 0.02)
    )


def _param_grid(scores: Mapping[str, np.ndarray], mask: np.ndarray) -> list[dict[str, float]]:
    gain_values = scores["gain_pred"][mask]
    gain_grid = [0.0]
    if len(gain_values):
        gain_grid += [float(v) for v in np.quantile(gain_values, [0.50, 0.60, 0.70, 0.80])]
    out: list[dict[str, float]] = []
    for switch_min in [0.45, 0.55, 0.65, 0.75]:
        for gain_min in gain_grid:
            for harm_max in [0.35, 0.45, 0.55, 0.70]:
                for max_switch in [0.05, 0.10, 0.20, 0.40, 0.70]:
                    out.append(
                        {
                            "switch_min": switch_min,
                            "gain_min": gain_min,
                            "harm_max": harm_max,
                            "max_switch": max_switch,
                            "guard_harm_min": 0.80,
                            "guard_gain_max": 0.0,
                        }
                    )
    return out


def _fit_policy(scores: Mapping[str, np.ndarray], data: Mapping[str, np.ndarray], mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    policy: dict[str, Any] = {"type": "joint_policy_distillation", "mode": mode, "slices": {}}
    selected_ade = data["floor_ade"].copy()
    switch = np.zeros(len(selected_ade), dtype=bool)
    if mode == "base_plus_distiller":
        selected_ade[data["base_switch"]] = data["neural_ade"][data["base_switch"]]
        switch |= data["base_switch"]
    diagnostics: dict[str, Any] = {}
    domain = data["domain"].astype(str)
    horizon = data["horizon"].astype(int)
    for d in sorted(set(domain.tolist())):
        for h in [10, 25, 50, 100]:
            mask = (domain == d) & (horizon == h)
            if int(np.sum(mask)) < 80:
                continue
            best_params: dict[str, float] | None = None
            best_score = -1e18
            best_metrics: dict[str, Any] | None = None
            local_data = {k: v[mask] for k, v in data.items() if isinstance(v, np.ndarray) and len(v) == len(mask)}
            local_scores = {k: v[mask] for k, v in scores.items()}
            for params in _param_grid(scores, mask):
                sel, _fde, sw = _apply_policy(local_scores, local_data, {"mode": mode, "slices": {f"{d}|{h}": params}})
                metrics = _metric(sel, local_data["floor_ade"], local_data, sw)
                if metrics.get("easy_degradation", 1.0) > 0.02:
                    continue
                score = _score(metrics)
                if score > best_score:
                    best_score = score
                    best_params = dict(params)
                    best_metrics = metrics
            if best_params:
                policy["slices"][f"{d}|{h}"] = best_params
                sel, _fde, sw = _apply_policy(local_scores, local_data, {"mode": mode, "slices": {f"{d}|{h}": best_params}})
                selected_ade[mask] = sel
                switch[mask] = sw
            diagnostics[f"{d}|{h}"] = {"selected": bool(best_params), "score": best_score, "metrics": best_metrics}
    metrics = _metric(selected_ade, data["floor_ade"], data, switch)
    metrics["slice_diagnostics"] = diagnostics
    return policy, metrics


def _evaluate(scores: Mapping[str, np.ndarray], data: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> dict[str, Any]:
    selected_ade, selected_fde, switch = _apply_policy(scores, data, policy)
    metrics = _metric(selected_ade, data["floor_ade"], data, switch)
    metrics["endpoint_fde_metrics"] = _metric(selected_fde, data["floor_fde"], data, switch)
    metrics["score_summary"] = {
        "switch_prob_mean": float(np.mean(scores["switch_prob"])),
        "gain_pred_mean": float(np.mean(scores["gain_pred"])),
        "harm_prob_mean": float(np.mean(scores["harm_prob"])),
        "true_gain_mean": float(np.mean(data["gain"])),
        "true_switch_rate": float(np.mean(data["switch_label"])),
        "base_switch_rate": float(np.mean(data["base_switch"])),
    }
    return metrics


def run_joint_policy_distillation() -> dict[str, Any]:
    trials: dict[str, Any] = {}
    best_name = ""
    best_policy: dict[str, Any] = {}
    best_ckpt = ""
    best_score = -1e18
    for trial in TRIALS:
        train = _train_trial(trial)
        scores_val, data_val = _predict_checkpoint(train["checkpoint"], "val")
        trial_info = {"source": train.get("source"), "trial": trial, "train": train, "policies": {}}
        # Only distiller_only is eligible for deployment because it does not
        # inherit the prior reference switch decisions.
        for mode in ["distiller_only"]:
            policy, val_metrics = _fit_policy(scores_val, data_val, mode)
            score = _score(val_metrics)
            trial_info["policies"][mode] = {"policy": policy, "val_metrics": val_metrics, "val_score": score}
            if score > best_score:
                best_name = f"{trial['name']}::{mode}"
                best_ckpt = train["checkpoint"]
                best_policy = policy
                best_score = score
        trials[trial["name"]] = trial_info
    scores_test, data_test = _predict_checkpoint(best_ckpt, "test")
    test_metrics = _evaluate(scores_test, data_test, best_policy)
    ref = read_json(OUT_DIR / "stage41_joint_multiagent_consistency.json", {})
    ref_metrics = ref.get("test_metrics") or {}
    full_ref = read_json(OUT_DIR / "stage41_full_trajectory_world_state.json", {}).get("best_metrics") or {}
    lift_joint = {
        "all_delta": float(test_metrics.get("all_improvement", 0.0) - ref_metrics.get("all_improvement", 0.0)),
        "t50_delta": float(test_metrics.get("t50_improvement", 0.0) - ref_metrics.get("t50_improvement", 0.0)),
        "t100_delta": float(test_metrics.get("t100_improvement", 0.0) - ref_metrics.get("t100_improvement", 0.0)),
        "hard_delta": float(test_metrics.get("hard_failure_improvement", 0.0) - ref_metrics.get("hard_failure_improvement", 0.0)),
        "easy_delta": float(test_metrics.get("easy_degradation", 0.0) - ref_metrics.get("easy_degradation", 0.0)),
    }
    lift_full = {
        "all_delta": float(test_metrics.get("all_improvement", 0.0) - full_ref.get("all_improvement", 0.0)),
        "t50_delta": float(test_metrics.get("t50_improvement", 0.0) - full_ref.get("t50_improvement", 0.0)),
        "t100_delta": float(test_metrics.get("t100_improvement", 0.0) - full_ref.get("t100_improvement", 0.0)),
        "hard_delta": float(test_metrics.get("hard_failure_improvement", 0.0) - full_ref.get("hard_failure_improvement", 0.0)),
        "easy_delta": float(test_metrics.get("easy_degradation", 0.0) - full_ref.get("easy_degradation", 0.0)),
    }
    contributes = bool(
        (lift_joint["all_delta"] > 0 or lift_joint["t50_delta"] > 0 or lift_joint["hard_delta"] > 0)
        and test_metrics.get("easy_degradation", 1.0) <= 0.02
        and test_metrics.get("all_improvement", 0.0) > 0
    )
    result = {
        "source": "fresh_run",
        "protocol_status": "joint_policy_distillation",
        "best_name": best_name,
        "best_checkpoint": best_ckpt,
        "best_policy": best_policy,
        "best_val_score": best_score,
        "test_metrics": test_metrics,
        "lift_over_joint_consistency_reference": lift_joint,
        "lift_over_full_trajectory_reference": lift_full,
        "joint_policy_distillation_contributes": contributes,
        "trials": trials,
        "no_leakage": {
            "future_waypoints_input": False,
            "future_waypoints_label_eval_only": True,
            "future_gain_harm_labels_train_only": True,
            "current_frame_grouping_only": True,
            "base_switch_input": False,
            "base_plus_distiller_deployable": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "caveat": "The distiller uses train labels to learn gain/harm/switch but uses only past/static/full-trajectory prediction signals at inference. Coordinates remain dataset-local raw-frame 2.5D.",
    }
    _write_json(OUT_DIR / "stage41_joint_policy_distillation.json", result)
    write_md(
        OUT_DIR / "stage41_joint_policy_distillation.md",
        [
            "# Stage41 Joint Policy Distillation",
            "",
            "- source: `fresh_run`",
            f"- best: `{best_name}`",
            f"- contributes over joint-consistency reference: `{contributes}`",
            f"- test metrics: `{test_metrics}`",
            f"- lift over joint consistency: `{lift_joint}`",
            f"- lift over full trajectory: `{lift_full}`",
            f"- no leakage: `{result['no_leakage']}`",
        ],
    )
    return result


def main_joint_policy_distillation() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_joint_policy_distillation()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_joint_policy_distillation",
            status,
            started,
            [OUT_DIR / "stage41_full_trajectory_world_state.json", OUT_DIR / "stage41_joint_multiagent_consistency.json"],
            [OUT_DIR / "stage41_joint_policy_distillation.md", OUT_DIR / "stage41_joint_policy_distillation.json"],
        )


if __name__ == "__main__":
    main_joint_policy_distillation()
