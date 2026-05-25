from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_breakthrough as s41
from src import stage41_full_trajectory_world_state as ft
from src import stage41_joint_multiagent_consistency as jmc
from src import stage41_joint_policy_distillation as jpd
from src import stage41_joint_rollout_consistency as jrc


OUT_DIR = jpd.OUT_DIR
CHECKPOINT_DIR = jpd.CHECKPOINT_DIR
REPORT_JSON = OUT_DIR / "stage41_group_consistency_distiller.json"
REPORT_MD = OUT_DIR / "stage41_group_consistency_distiller.md"
THREADS = 4
BATCH = 2048
EPOCHS = 4
SEED = 4147
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


def _group_count(keys: np.ndarray) -> np.ndarray:
    counts = Counter(map(str, keys.tolist()))
    return np.asarray([counts[str(k)] for k in keys], dtype=np.float32)


def _safe_float(x: np.ndarray, fill: float = 9.0) -> np.ndarray:
    return np.where(np.isfinite(x), x, fill).astype(np.float32)


def _load_policy_and_checkpoint() -> tuple[str, dict[str, Any], str]:
    distiller = read_json(OUT_DIR / "stage41_joint_policy_distillation.json", {})
    repair = read_json(OUT_DIR / "stage41_ucy_fallback_repair.json", {})
    if not distiller:
        raise FileNotFoundError("Run stage41_joint_policy_distillation first.")
    policy = repair.get("repaired_policy") or distiller.get("best_policy") or {}
    policy_source = "ucy_repaired_policy" if repair.get("repaired_policy") else "joint_distiller_policy"
    return str(distiller["best_checkpoint"]), policy, policy_source


def _bundle(split: str, checkpoint: str | Path, policy: Mapping[str, Any]) -> dict[str, Any]:
    pred, labels, _ref = jmc._split_predictions(split)
    scores, data = jpd._predict_checkpoint(checkpoint, split)
    _selected_ade, _selected_fde, proposal_switch = jpd._apply_policy(scores, data, policy)
    meta = jmc._group_metadata(split)
    keys = meta["key"]
    floor_xy = ft._floor_waypoints(labels)
    neural_xy = ft._pred_waypoints(pred, labels)
    raw_selected_xy = floor_xy.copy()
    raw_selected_xy[proposal_switch] = neural_xy[proposal_switch]
    floor_ade, floor_fde = ft._trajectory_errors(floor_xy, labels)
    neural_ade, neural_fde = ft._trajectory_errors(neural_xy, labels)
    normalizer = labels["normalizer"].astype(np.float64)
    current_xy = labels["current_xy"][:, None, :].repeat(floor_xy.shape[1], axis=1)
    current_min = jmc._min_group_distance(current_xy, keys, normalizer)
    floor_min = jmc._min_group_distance(floor_xy, keys, normalizer)
    neural_min = jmc._min_group_distance(neural_xy, keys, normalizer)
    raw_selected_min = jmc._min_group_distance(raw_selected_xy, keys, normalizer)
    group_count = _group_count(keys)
    unsafe = proposal_switch & np.isfinite(raw_selected_min) & (raw_selected_min < 0.05) & (raw_selected_min + EPS < floor_min)
    gain = (floor_ade - neural_ade).astype(np.float32)
    safe_switch = proposal_switch & (gain > 0.0) & (~unsafe)
    min_delta = np.zeros_like(raw_selected_min, dtype=np.float64)
    finite_delta = np.isfinite(raw_selected_min) & np.isfinite(floor_min)
    min_delta[finite_delta] = raw_selected_min[finite_delta] - floor_min[finite_delta]
    group_features = np.stack(
        [
            group_count / 10.0,
            _safe_float(current_min),
            _safe_float(floor_min),
            _safe_float(neural_min),
            _safe_float(raw_selected_min),
            _safe_float(min_delta, fill=0.0),
            proposal_switch.astype(np.float32),
            scores["switch_prob"].astype(np.float32),
            scores["gain_pred"].astype(np.float32),
            scores["harm_prob"].astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    features = np.concatenate([data["x"].astype(np.float32), group_features], axis=1).astype(np.float32)
    return {
        "x": features,
        "gain": gain,
        "unsafe": unsafe.astype(np.float32),
        "safe_switch": safe_switch.astype(np.float32),
        "proposal_switch": proposal_switch.astype(bool),
        "floor_ade": floor_ade.astype(np.float64),
        "floor_fde": floor_fde.astype(np.float64),
        "neural_ade": neural_ade.astype(np.float64),
        "neural_fde": neural_fde.astype(np.float64),
        "floor_xy": floor_xy,
        "neural_xy": neural_xy,
        "labels": labels,
        "keys": keys,
        "domain": labels["domain"],
        "horizon": labels["horizon"],
        "hard": labels["hard"],
        "failure": labels["failure"],
        "easy": labels["easy"],
        "candidate_fde": labels["candidate_fde"],
        "feature_dim": np.asarray([features.shape[1]], dtype=np.int64),
    }


def _make_model(dim: int, width: int = 96, dropout: float = 0.06):
    torch = _torch()
    import torch.nn as nn

    class GroupConsistencyDistiller(nn.Module):
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
            self.safe = nn.Linear(width, 1)
            self.gain = nn.Linear(width, 1)
            self.unsafe = nn.Linear(width, 1)

        def forward(self, x):
            h = self.net(x)
            return {
                "safe_logit": self.safe(h).squeeze(-1),
                "gain": self.gain(h).squeeze(-1),
                "unsafe_logit": self.unsafe(h).squeeze(-1),
            }

    return GroupConsistencyDistiller()


def _train_model(train: Mapping[str, np.ndarray], val: Mapping[str, np.ndarray]) -> dict[str, Any]:
    torch = _torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    ckpt = CHECKPOINT_DIR / "stage41_group_consistency_distiller.pt"
    heartbeat = OUT_DIR / "group_consistency_distiller_heartbeat.json"
    x = torch.tensor(train["x"])
    y_gain = torch.tensor(train["gain"])
    y_safe = torch.tensor(train["safe_switch"])
    y_unsafe = torch.tensor(train["unsafe"])
    hard = torch.tensor((train["hard"].astype(bool) | train["failure"].astype(bool)).astype(np.float32))
    vx = torch.tensor(val["x"])
    vg = torch.tensor(val["gain"])
    vs = torch.tensor(val["safe_switch"])
    vu = torch.tensor(val["unsafe"])
    model = _make_model(x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    rng = np.random.default_rng(SEED)
    best = {"val_loss": float("inf"), "epoch": 0}
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(x.shape[0])
        losses: list[float] = []
        model.train()
        for start in range(0, len(order), BATCH):
            ids = torch.tensor(order[start : start + BATCH], dtype=torch.long)
            out = model(x[ids])
            row_w = 1.0 + 1.5 * hard[ids] + 2.0 * torch.clamp(torch.abs(y_gain[ids]), max=2.0)
            gain_loss = (F.smooth_l1_loss(out["gain"], y_gain[ids], reduction="none") * row_w).mean()
            safe_loss = (F.binary_cross_entropy_with_logits(out["safe_logit"], y_safe[ids], reduction="none") * row_w).mean()
            unsafe_loss = (F.binary_cross_entropy_with_logits(out["unsafe_logit"], y_unsafe[ids], reduction="none") * row_w).mean()
            loss = gain_loss + 1.2 * safe_loss + 1.0 * unsafe_loss
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
                    + 1.2 * F.binary_cross_entropy_with_logits(out["safe_logit"], vs)
                    + F.binary_cross_entropy_with_logits(out["unsafe_logit"], vu)
                ).cpu()
            )
        payload = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss, "checkpoint": str(ckpt), "best": best}
        heartbeat.write_text(json.dumps(_jsonable(payload), ensure_ascii=False), encoding="utf-8")
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses))}
            torch.save({"model": model.state_dict(), "dim": int(x.shape[1]), "best": best, "width": 96, "dropout": 0.06}, ckpt)
    return {"source": "fresh_run", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best}


def _predict(path: str | Path, data: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    torch = _torch()
    payload = torch.load(path, map_location="cpu")
    model = _make_model(int(payload["dim"]), int(payload.get("width", 96)), float(payload.get("dropout", 0.06)))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    x = torch.tensor(data["x"])
    outs = {"safe_prob": [], "gain_pred": [], "unsafe_prob": []}
    with torch.no_grad():
        for start in range(0, x.shape[0], 4096):
            out = model(x[start : start + 4096])
            outs["safe_prob"].append(torch.sigmoid(out["safe_logit"]).cpu().numpy())
            outs["gain_pred"].append(out["gain"].cpu().numpy())
            outs["unsafe_prob"].append(torch.sigmoid(out["unsafe_logit"]).cpu().numpy())
    return {k: np.concatenate(v).astype(np.float32) for k, v in outs.items()}


def _policy_metrics(scores: Mapping[str, np.ndarray], data: Mapping[str, np.ndarray], policy: Mapping[str, float]) -> tuple[dict[str, Any], np.ndarray]:
    switch = (
        data["proposal_switch"].astype(bool)
        & (scores["safe_prob"] >= float(policy.get("safe_min", 0.5)))
        & (scores["gain_pred"] >= float(policy.get("gain_min", 0.0)))
        & (scores["unsafe_prob"] <= float(policy.get("unsafe_max", 1.0)))
    )
    selected_ade = data["floor_ade"].copy()
    selected_ade[switch] = data["neural_ade"][switch]
    selected_xy = data["floor_xy"].copy()
    selected_xy[switch] = data["neural_xy"][switch]
    ev = jrc._evaluate_split_rollout(
        {
            "labels": data["labels"],
            "keys": data["keys"],
            "floor_xy": data["floor_xy"],
            "neural_xy": data["neural_xy"],
            "floor_ade": data["floor_ade"],
        },
        switch,
        "group_consistency_distiller",
    )
    metrics = ev["selected_metrics"]
    metrics["collision_delta_vs_floor_005"] = ev["collision_delta_005"]
    metrics["multi_agent_metrics"] = ev["multi_agent_metrics"]
    metrics["rollout_stats"] = {"selected": ev["selected_stats"], "floor": ev["floor_stats"]}
    return metrics, switch


def _score(metrics: Mapping[str, Any]) -> float:
    return (
        1.0 * float(metrics.get("all_improvement", 0.0))
        + 1.4 * float(metrics.get("t50_improvement", 0.0))
        + 1.0 * float(metrics.get("t100_improvement", 0.0))
        + 1.2 * float(metrics.get("hard_failure_improvement", 0.0))
        - 35.0 * max(0.0, float(metrics.get("easy_degradation", 1.0)) - 0.02)
        - 8.0 * max(0.0, float(metrics.get("collision_delta_vs_floor_005", 1.0)) - 0.01)
    )


def _fit_policy(scores: Mapping[str, np.ndarray], val: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gain_values = scores["gain_pred"][val["proposal_switch"].astype(bool)]
    gain_grid = [0.0]
    if len(gain_values):
        gain_grid.extend(float(v) for v in np.quantile(gain_values, [0.50, 0.70]))
    candidates: list[dict[str, Any]] = []
    for safe_min in [0.40, 0.50, 0.60, 0.70]:
        for gain_min in gain_grid:
            for unsafe_max in [0.35, 0.50, 0.65, 0.80]:
                policy = {"safe_min": safe_min, "gain_min": gain_min, "unsafe_max": unsafe_max}
                metrics, switch = _policy_metrics(scores, val, policy)
                eligible = bool(
                    metrics.get("all_improvement", 0.0) > 0
                    and metrics.get("t50_improvement", 0.0) > 0
                    and metrics.get("hard_failure_improvement", 0.0) > 0
                    and metrics.get("easy_degradation", 1.0) <= 0.02
                    and metrics.get("collision_delta_vs_floor_005", 1.0) <= 0.01
                    and float(np.mean(switch)) > 0.0
                )
                candidates.append({"policy": policy, "metrics": metrics, "switch_rate": float(np.mean(switch)), "eligible": eligible, "score": _score(metrics)})
    pool = [c for c in candidates if c["eligible"]] or candidates
    best = max(pool, key=lambda row: row["score"])
    return {"type": "group_consistency_distiller", **best["policy"], "val_eligible": best["eligible"]}, candidates


def _label_summary(data: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "rows": int(len(data["gain"])),
        "proposal_switch_rate": float(np.mean(data["proposal_switch"])),
        "safe_switch_rate": float(np.mean(data["safe_switch"])),
        "unsafe_rate": float(np.mean(data["unsafe"])),
        "mean_gain": float(np.mean(data["gain"])),
        "hard_rows": int(np.sum(data["hard"].astype(bool) | data["failure"].astype(bool))),
    }


def run_group_consistency_distiller() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    checkpoint, repaired_policy, policy_source = _load_policy_and_checkpoint()
    train = _bundle("train", checkpoint, repaired_policy)
    val = _bundle("val", checkpoint, repaired_policy)
    test = _bundle("test", checkpoint, repaired_policy)
    train_result = _train_model(train, val)
    scores_val = _predict(train_result["checkpoint"], val)
    policy, candidates = _fit_policy(scores_val, val)
    scores_test = _predict(train_result["checkpoint"], test)
    test_metrics, test_switch = _policy_metrics(scores_test, test, policy)
    raw_report = read_json(OUT_DIR / "stage41_ucy_fallback_repair.json", {})
    guard_report = read_json(OUT_DIR / "stage41_joint_rollout_consistency.json", {})
    raw_metrics = raw_report.get("repaired_metrics") or {}
    guard_metrics = guard_report.get("selected_metrics") or {}
    lift_over_fixed_guard = {
        "all_delta": float(test_metrics.get("all_improvement", 0.0) - guard_metrics.get("all_improvement", 0.0)),
        "t50_delta": float(test_metrics.get("t50_improvement", 0.0) - guard_metrics.get("t50_improvement", 0.0)),
        "t100_delta": float(test_metrics.get("t100_improvement", 0.0) - guard_metrics.get("t100_improvement", 0.0)),
        "hard_delta": float(test_metrics.get("hard_failure_improvement", 0.0) - guard_metrics.get("hard_failure_improvement", 0.0)),
        "easy_delta": float(test_metrics.get("easy_degradation", 0.0) - guard_metrics.get("easy_degradation", 0.0)),
    }
    deployable = bool(
        test_metrics.get("all_improvement", 0.0) > 0
        and test_metrics.get("t50_improvement", 0.0) > 0
        and test_metrics.get("hard_failure_improvement", 0.0) > 0
        and test_metrics.get("easy_degradation", 1.0) <= 0.02
        and test_metrics.get("collision_delta_vs_floor_005", 1.0) <= 0.01
        and float(np.mean(test_switch)) > 0.0
    )
    improves_guard = bool(
        deployable
        and (
            lift_over_fixed_guard["all_delta"] > 0
            or lift_over_fixed_guard["t50_delta"] > 0
            or lift_over_fixed_guard["hard_delta"] > 0
        )
    )
    result = {
        "source": "fresh_run",
        "policy_source": policy_source,
        "trained_neural_group_consistency_head": True,
        "checkpoint": train_result["checkpoint"],
        "train": train_result,
        "label_summary": {"train": _label_summary(train), "val": _label_summary(val), "test": _label_summary(test)},
        "selected_policy": policy,
        "val_candidates": candidates,
        "test_metrics": test_metrics,
        "raw_repaired_policy_metrics": raw_metrics,
        "fixed_proximity_guard_metrics": guard_metrics,
        "lift_over_fixed_proximity_guard": lift_over_fixed_guard,
        "group_consistency_distiller_deployable": deployable,
        "group_consistency_distiller_improves_fixed_guard": improves_guard,
        "no_leakage": {
            "future_waypoints_input": False,
            "future_waypoints_label_eval_only": True,
            "train_gain_safe_unsafe_labels_only": True,
            "test_threshold_tuning": False,
            "policy_selected_on_val": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "caveat": "This is a neural group-consistency safety/gain head over a proposed switch policy. It remains dataset-local raw-frame 2.5D and is not a latent generative rollout.",
    }
    write_json(REPORT_JSON, _jsonable(result))
    write_md(
        REPORT_MD,
        [
            "# Stage41 Group Consistency Distiller",
            "",
            "- source: `fresh_run`",
            "- trained neural group-consistency head: `true`",
            f"- selected policy: `{policy}`",
            f"- deployable: `{deployable}`",
            f"- improves fixed proximity guard: `{improves_guard}`",
            f"- test metrics: `{test_metrics}`",
            f"- lift over fixed proximity guard: `{lift_over_fixed_guard}`",
            f"- label summary: `{result['label_summary']}`",
            f"- no leakage: `{result['no_leakage']}`",
            "",
            "The model learns train-only gain/safe-switch/unsafe labels and gates proposed switches at inference with only past/static/predicted-rollout group features. It does not execute Stage5C or SMC.",
        ],
    )
    return result


def main_group_consistency_distiller() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_group_consistency_distiller()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_group_consistency_distiller",
            status,
            started,
            [
                OUT_DIR / "stage41_joint_policy_distillation.json",
                OUT_DIR / "stage41_ucy_fallback_repair.json",
                OUT_DIR / "stage41_joint_rollout_consistency.json",
            ],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_group_consistency_distiller()
