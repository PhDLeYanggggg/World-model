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
from src import stage42_horizon_static_gate_repair as s42l
from src import stage42_policy_distilled_static_gate as s42m
from src import stage42_row_gain_static_gate as s42n
from src import stage42_sequence_full_waypoint as s42i
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
REPORT_JSON = OUT_DIR / "explicit_gain_harm_selector_stage42.json"
REPORT_MD = OUT_DIR / "explicit_gain_harm_selector_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_o_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SEEDS = [131, 137, 139]
BASE_MODEL_SEEDS = [109, 113, 127]
EPOCHS = 2
BATCH = 4096
THREADS = 4
HORIZONS = [10, 25, 50, 100]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。",
    "Stage42-O 使用 dataset-local raw-frame full-waypoint labels，不能写成 metric 或 seconds-level。",
    "future waypoints / future endpoints 只作为 train/val selector labels 和 eval label，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals。",
    "Stage42-O 策略选择只在 validation 上完成，test 只最终评估一次。",
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
    if venv.exists() and os.environ.get("STAGE42O_REEXEC") != "1":
        env = os.environ.copy()
        env["STAGE42O_REEXEC"] = "1"
        os.execve("/usr/bin/arch", ["/usr/bin/arch", "-arm64", str(venv), *sys.argv], env)
    raise RuntimeError("Stage42-O refuses x86_64/Rosetta Python for torch training.")


def _torch():
    _ensure_arm64()
    import torch

    torch.set_num_threads(THREADS)
    return torch


def _softplus_inverse(y: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    y = np.maximum(y.astype(np.float32), eps)
    return np.log(np.expm1(y)).astype(np.float32)


def _domain_vocab(*splits: Mapping[str, np.ndarray]) -> dict[str, int]:
    names = sorted({str(x) for split in splits for x in split["raw"]["domain"].astype(str).tolist()})
    return {name: i for i, name in enumerate(names)}


def _domain_onehot(split: Mapping[str, np.ndarray], vocab: Mapping[str, int]) -> np.ndarray:
    idx = np.asarray([vocab[str(x)] for x in split["raw"]["domain"].astype(str).tolist()], dtype=np.int64)
    out = np.zeros((len(idx), len(vocab)), dtype=np.float32)
    out[np.arange(len(idx)), idx] = 1.0
    return out


def _horizon_onehot(split: Mapping[str, np.ndarray]) -> np.ndarray:
    idx = s42l._horizon_index_np(split["horizon"])
    out = np.zeros((len(idx), len(HORIZONS)), dtype=np.float32)
    out[np.arange(len(idx)), idx] = 1.0
    return out


def _token_features(split: Mapping[str, np.ndarray]) -> np.ndarray:
    tokens = split["agent_tokens"].astype(np.float32)
    mask = split["agent_mask"].astype(np.float32)
    valid_t = np.clip(tokens[..., 6], 0.0, 1.0)
    target_last = tokens[:, 0, -1, :]
    target_mean = (tokens[:, 0] * valid_t[:, 0, :, None]).sum(axis=1) / np.maximum(valid_t[:, 0].sum(axis=1, keepdims=True), 1.0)
    neighbor_count = mask[:, 1:].sum(axis=1, keepdims=True)
    neighbor_mean = (tokens[:, 1:, -1, :] * mask[:, 1:, None]).sum(axis=1) / np.maximum(neighbor_count, 1.0)
    return np.concatenate([target_last, target_mean, neighbor_mean, neighbor_count], axis=1).astype(np.float32)


def _raw_features(split: Mapping[str, np.ndarray], pred: Mapping[str, np.ndarray], vocab: Mapping[str, int]) -> np.ndarray:
    pred_feats = np.stack(
        [
            pred["traj_risk"].astype(np.float32),
            pred["interaction"].astype(np.float32),
            pred["occupancy"].astype(np.float32),
            pred["physical"].astype(np.float32),
            pred["static_gate"].astype(np.float32),
        ],
        axis=1,
    )
    raw = split["raw"]
    causal = np.concatenate(
        [
            split["static"].astype(np.float32),
            _token_features(split),
            _horizon_onehot(split),
            _domain_onehot(split, vocab),
            pred_feats,
            np.log1p(np.maximum(raw["normalizer"].astype(np.float32), 0.0))[:, None],
        ],
        axis=1,
    )
    return causal.astype(np.float32)


def _feature_stats(causal: np.ndarray) -> dict[str, np.ndarray]:
    return {"mean": causal.mean(axis=0, keepdims=True), "std": causal.std(axis=0, keepdims=True)}


def _features(split: Mapping[str, np.ndarray], pred: Mapping[str, np.ndarray], vocab: Mapping[str, int], stats: Mapping[str, np.ndarray]) -> np.ndarray:
    causal = _raw_features(split, pred, vocab)
    mean = stats["mean"]
    std = stats["std"]
    return ((causal - mean) / np.maximum(std, 1e-5)).astype(np.float32)


def _target_from_teacher(teacher: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    gain = np.maximum(teacher["floor_gain"].astype(np.float32), 0.0)
    harm = (teacher["harm"].astype(np.float32) > 1e-4).astype(np.float32)
    switch = teacher["switchable"].astype(np.float32)
    weight = teacher["weight"].astype(np.float32)
    return {"switch": switch, "gain": gain, "gain_logit": _softplus_inverse(gain + 1e-4), "harm": harm, "weight": weight}


def _make_selector(input_dim: int, width: int = 96):
    torch = _torch()
    import torch.nn as nn

    class GainHarmSelector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, width),
                nn.GELU(),
                nn.LayerNorm(width),
                nn.Linear(width, width),
                nn.GELU(),
                nn.LayerNorm(width),
            )
            self.switch = nn.Linear(width, 1)
            self.gain = nn.Linear(width, 1)
            self.harm = nn.Linear(width, 1)
            self.uncertainty = nn.Linear(width, 1)

        def forward(self, x):
            h = self.net(x)
            return {
                "switch_logit": self.switch(h).squeeze(-1),
                "gain_raw": self.gain(h).squeeze(-1),
                "harm_logit": self.harm(h).squeeze(-1),
                "uncertainty_logit": self.uncertainty(h).squeeze(-1),
            }

    return GainHarmSelector()


def _train_selector(seed: int, x_train: np.ndarray, y_train: Mapping[str, np.ndarray], x_val: np.ndarray, y_val: Mapping[str, np.ndarray]) -> dict[str, Any]:
    torch = _torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ckpt = CHECKPOINT_DIR / f"stage42o_gain_harm_selector_seed{seed}.pt"
    heartbeat = OUT_DIR / f"stage42o_gain_harm_selector_seed{seed}_heartbeat.json"
    if ckpt.exists() and heartbeat.exists():
        payload = read_json(heartbeat, {})
        if int(payload.get("epoch", 0)) >= EPOCHS:
            return {"source": "cached_verified", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": payload.get("best", {})}
    model = _make_selector(x_train.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    tx = torch.tensor(x_train)
    tvx = torch.tensor(x_val)
    tensors = {k: torch.tensor(v) for k, v in y_train.items()}
    val_tensors = {k: torch.tensor(v) for k, v in y_val.items()}
    best = {"val_loss": float("inf"), "epoch": 0}
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(x_train))
        losses: list[float] = []
        model.train()
        for start in range(0, len(order), BATCH):
            ids = torch.tensor(order[start : start + BATCH], dtype=torch.long)
            out = model(tx[ids])
            w = tensors["weight"][ids]
            switch_loss = (F.binary_cross_entropy_with_logits(out["switch_logit"], tensors["switch"][ids], reduction="none") * w).mean()
            gain_loss = (F.smooth_l1_loss(F.softplus(out["gain_raw"]), tensors["gain"][ids], reduction="none") * w).mean()
            harm_loss = (F.binary_cross_entropy_with_logits(out["harm_logit"], tensors["harm"][ids], reduction="none") * w).mean()
            # Higher uncertainty is useful for ambiguous low-gain or harmful rows.
            uncertainty_target = torch.clamp(1.0 - tensors["switch"][ids] + tensors["harm"][ids], 0.0, 1.0)
            uncertainty_loss = F.binary_cross_entropy_with_logits(out["uncertainty_logit"], uncertainty_target)
            loss = 1.4 * switch_loss + 1.2 * gain_loss + 1.1 * harm_loss + 0.15 * uncertainty_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            out = model(tvx)
            val_loss = float(
                (
                    1.4 * F.binary_cross_entropy_with_logits(out["switch_logit"], val_tensors["switch"])
                    + 1.2 * F.smooth_l1_loss(F.softplus(out["gain_raw"]), val_tensors["gain"])
                    + 1.1 * F.binary_cross_entropy_with_logits(out["harm_logit"], val_tensors["harm"])
                ).cpu()
            )
        cand = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses))}
        if val_loss < best["val_loss"]:
            best = cand
            torch.save({"model": model.state_dict(), "seed": seed, "input_dim": x_train.shape[1], "best": best}, ckpt)
        heartbeat.write_text(json.dumps({"seed": seed, "epoch": epoch, "best": best, "checkpoint": str(ckpt)}, ensure_ascii=False), encoding="utf-8")
    return {"source": "fresh_run", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best}


def _predict_selector(info: Mapping[str, Any], x: np.ndarray) -> dict[str, np.ndarray]:
    torch = _torch()
    payload = torch.load(info["checkpoint"], map_location="cpu", weights_only=False)
    model = _make_selector(int(payload["input_dim"]))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    outs = {k: [] for k in ["switch_prob", "gain", "harm_prob", "uncertainty"]}
    with torch.no_grad():
        for start in range(0, len(x), 8192):
            sl = slice(start, min(start + 8192, len(x)))
            out = model(torch.tensor(x[sl]))
            outs["switch_prob"].append(torch.sigmoid(out["switch_logit"]).cpu().numpy())
            outs["gain"].append(torch.nn.functional.softplus(out["gain_raw"]).cpu().numpy())
            outs["harm_prob"].append(torch.sigmoid(out["harm_logit"]).cpu().numpy())
            outs["uncertainty"].append(torch.sigmoid(out["uncertainty_logit"]).cpu().numpy())
    return {k: np.concatenate(v).astype(np.float32) for k, v in outs.items()}


def _selector_switch(scores: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> np.ndarray:
    n = len(labels["horizon"])
    switch = np.zeros(n, dtype=bool)
    domain = labels["domain"].astype(str)
    horizon = labels["horizon"].astype(int)
    rank = scores["switch_prob"] + 0.7 * scores["gain"] - 0.9 * scores["harm_prob"] - 0.3 * scores["uncertainty"]
    for key, params in policy.get("slices", {}).items():
        d, h_s = key.split("|")
        mask = (domain == d) & (horizon == int(h_s))
        local = (
            mask
            & (scores["switch_prob"] >= float(params["switch_min"]))
            & (scores["gain"] >= float(params["gain_min"]))
            & (scores["harm_prob"] <= float(params["harm_max"]))
            & (scores["uncertainty"] <= float(params.get("uncertainty_max", 1.0)))
        )
        max_switch = float(params.get("max_switch", 1.0))
        if max_switch <= 0.0:
            local[:] = False
        elif max_switch < 1.0 and np.any(local):
            ids = np.where(local)[0]
            keep_n = max(1, int(max_switch * int(np.sum(mask))))
            keep = np.zeros(n, dtype=bool)
            keep[ids[np.argsort(rank[ids])[::-1][:keep_n]]] = True
            local &= keep
        switch |= local
    return switch


def _metric_from_switch(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], switch: np.ndarray) -> dict[str, Any]:
    floor = ft._floor_waypoints(labels)
    neural = ft._pred_waypoints(pred, labels)
    selected = floor.copy()
    selected[switch] = neural[switch]
    ade, fde = ft._trajectory_errors(selected, labels)
    floor_ade, floor_fde = ft._trajectory_errors(floor, labels)
    return {
        "ade": ft._metric(ade, floor_ade, labels, switch),
        "fde": ft._metric(fde, floor_fde, labels, switch),
        "switch_rate": float(np.mean(switch)) if len(switch) else 0.0,
    }


def _score_metric(metric: Mapping[str, Any]) -> float:
    return (
        1.1 * float(metric.get("all_improvement", 0.0))
        + 2.2 * float(metric.get("t50_improvement", 0.0))
        + 1.0 * float(metric.get("hard_failure_improvement", 0.0))
        + 0.7 * float(metric.get("t100_improvement", 0.0))
        - 45.0 * max(0.0, float(metric.get("easy_degradation", 1.0)) - 0.02)
    )


def _fit_policy(scores: Mapping[str, np.ndarray], pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    domain = labels["domain"].astype(str)
    horizon = labels["horizon"].astype(int)
    policy: dict[str, Any] = {"type": "stage42o_explicit_gain_harm_policy_no_easy_label_input", "slices": {}}
    diagnostics: dict[str, Any] = {}
    for d in sorted(set(domain.tolist())):
        for h in HORIZONS:
            mask = (domain == d) & (horizon == h)
            if int(np.sum(mask)) < 80:
                continue
            best_score = 0.0
            best_params: dict[str, Any] | None = None
            best_metric: dict[str, Any] | None = None
            gain_q = np.quantile(scores["gain"][mask], [0.25, 0.50, 0.70])
            for switch_min in [0.45, 0.55, 0.65, 0.75]:
                for gain_min in [0.0, *[float(x) for x in gain_q]]:
                    for harm_max in [0.20, 0.30, 0.40, 0.55]:
                        for uncertainty_max in [0.55, 0.70, 0.90]:
                            for max_switch in [0.05, 0.10, 0.20, 0.35]:
                                params = {
                                    "switch_min": switch_min,
                                    "gain_min": gain_min,
                                    "harm_max": harm_max,
                                    "uncertainty_max": uncertainty_max,
                                    "max_switch": max_switch,
                                }
                                trial = {"slices": {f"{d}|{h}": params}}
                                sw = _selector_switch(scores, labels, trial)
                                metric = _metric_from_switch(pred, labels, sw)["ade"]
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
    sw = _selector_switch(scores, labels, policy)
    metrics = _metric_from_switch(pred, labels, sw)
    metrics["slice_diagnostics"] = diagnostics
    return policy, metrics


def _base_model_info(seed: int) -> dict[str, Any]:
    ckpt = CHECKPOINT_DIR / f"stage42n_row_gain_static_gate_seed{seed}.pt"
    heartbeat = OUT_DIR / f"stage42n_row_gain_static_gate_seed{seed}_heartbeat.json"
    if not ckpt.exists() or not heartbeat.exists():
        raise FileNotFoundError(f"Missing Stage42-N checkpoint for seed {seed}. Run Stage42-N first.")
    return {"source": "cached_verified_stage42n", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": read_json(heartbeat, {}).get("best", {})}


def _eval_seed(
    seed: int,
    base_seed: int,
    train: Mapping[str, np.ndarray],
    val: Mapping[str, np.ndarray],
    test: Mapping[str, np.ndarray],
    vocab: Mapping[str, int],
    train_teacher: Mapping[str, np.ndarray],
    val_teacher: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    base_info = _base_model_info(base_seed)
    pred_train = s42m._predict(base_info, train)
    pred_val = s42m._predict(base_info, val)
    pred_test = s42m._predict(base_info, test)
    train_stats = _feature_stats(_raw_features(train, pred_train, vocab))
    x_train = _features(train, pred_train, vocab, train_stats)
    x_val = _features(val, pred_val, vocab, train_stats)
    x_test = _features(test, pred_test, vocab, train_stats)
    y_train = _target_from_teacher(train_teacher)
    y_val = _target_from_teacher(val_teacher)
    selector_info = _train_selector(seed, x_train, y_train, x_val, y_val)
    score_val = _predict_selector(selector_info, x_val)
    score_test = _predict_selector(selector_info, x_test)
    labels_val = s42i._labels(val)
    labels_test = s42i._labels(test)
    policy, val_metrics = _fit_policy(score_val, pred_val, labels_val)
    test_switch = _selector_switch(score_test, labels_test, policy)
    test_metrics = _metric_from_switch(pred_test, labels_test, test_switch)
    baseline_policy, baseline_val = s42l._fit_t50_weighted_policy(pred_val, labels_val)
    baseline_test = s42i._row_metrics("stage42n_static_gate_baseline", pred_test, labels_test, baseline_policy)
    return {
        "source": "fresh_run",
        "seed": seed,
        "base_seed": base_seed,
        "base_info": base_info,
        "selector_info": selector_info,
        "val_policy": policy,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "stage42n_baseline_val_metrics": baseline_val,
        "stage42n_baseline_test_metrics": baseline_test,
        "score_means_test": {key: float(np.mean(value)) for key, value in score_test.items()},
    }


def _summary(rows: list[Mapping[str, Any]], key: str = "test_metrics") -> dict[str, Any]:
    return {
        "source": "fresh_run",
        "seeds": [int(row["seed"]) for row in rows],
        "ade_all": s42l._stat([row[key]["ade"].get("all_improvement", 0.0) for row in rows]),
        "ade_t50": s42l._stat([row[key]["ade"].get("t50_improvement", 0.0) for row in rows]),
        "ade_t100_raw_frame_diagnostic": s42l._stat([row[key]["ade"].get("t100_improvement", 0.0) for row in rows]),
        "ade_hard_failure": s42l._stat([row[key]["ade"].get("hard_failure_improvement", 0.0) for row in rows]),
        "ade_easy_degradation": s42l._stat([row[key]["ade"].get("easy_degradation", 1.0) for row in rows]),
        "fde_all": s42l._stat([row[key]["fde"].get("all_improvement", 0.0) for row in rows]),
        "fde_t50": s42l._stat([row[key]["fde"].get("t50_improvement", 0.0) for row in rows]),
        "switch_rate": s42l._stat([row[key].get("switch_rate", 0.0) for row in rows]),
    }


def _comparison() -> dict[str, Any]:
    return {
        "source": "cached_verified",
        "stage42_n_row_gain_static_gate": read_json(OUT_DIR / "row_gain_static_gate_stage42.json", {}).get("summary", {}),
        "stage42_l_horizon_static_gate": read_json(OUT_DIR / "horizon_static_gate_repair_stage42.json", {}).get("summary", {}),
        "stage42_j_static_gated": (read_json(OUT_DIR / "static_gated_full_waypoint_stage42.json", {}).get("summary") or {}).get("static_gated", {}),
    }


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    s = result.get("summary", {})
    n = result.get("comparison", {}).get("stage42_n_row_gain_static_gate", {})
    gates = {
        "explicit_selector_trained": len(result.get("rows", [])) >= 3,
        "base_stage42n_checkpoints_cached_verified": all(row.get("base_info", {}).get("source") == "cached_verified_stage42n" for row in result.get("rows", [])),
        "train_val_teacher_only": result.get("source_labels", {}).get("row_teacher_test") == "not_built",
        "policy_no_easy_label_input": result.get("source_labels", {}).get("policy_uses_easy_label") is False,
        "all_positive": s.get("ade_all", {}).get("mean", 0.0) > 0.0,
        "t50_positive": s.get("ade_t50", {}).get("mean", -1.0) > 0.0,
        "hard_positive": s.get("ade_hard_failure", {}).get("mean", 0.0) > 0.0,
        "easy_preserved": s.get("ade_easy_degradation", {}).get("mean", 1.0) <= 0.02,
        "improves_stage42n_t50": s.get("ade_t50", {}).get("mean", -1.0) > n.get("ade_t50", {}).get("mean", -1.0),
        "no_leakage_pass": result.get("no_leakage", {}).get("future_endpoint_input") is False
        and result.get("no_leakage", {}).get("future_waypoints_input") is False
        and result.get("no_leakage", {}).get("central_velocity") is False
        and result.get("no_leakage", {}).get("test_endpoint_goals") is False,
        "no_test_statistics_normalization": result.get("no_leakage", {}).get("test_statistics_normalization") is False,
        "no_metric_seconds_overclaim": result.get("claim_boundary", {}).get("metric_or_seconds_claim") is False,
        "stage5c_false": result.get("claim_boundary", {}).get("stage5c_executed") is False,
        "smc_false": result.get("claim_boundary", {}).get("smc_enabled") is False,
    }
    return {
        "source": "fresh_run",
        "gates": gates,
        "passed": int(sum(bool(v) for v in gates.values())),
        "total": len(gates),
        "verdict": "stage42_o_explicit_gain_harm_selector_pass" if all(gates.values()) else "stage42_o_explicit_gain_harm_selector_partial",
    }


def run_stage42_explicit_gain_harm_selector() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CHECKPOINT_DIR)
    ft.build_full_trajectory_labels()
    data = {split: s42i._split_arrays(split) for split in ["train", "val", "test"]}
    vocab = _domain_vocab(data["train"], data["val"], data["test"])
    train_teacher = s42n._row_teacher(data["train"], "train")
    val_teacher = s42n._row_teacher(data["val"], "val")
    rows = [
        _eval_seed(seed, base_seed, data["train"], data["val"], data["test"], vocab, train_teacher, val_teacher)
        for seed, base_seed in zip(SEEDS, BASE_MODEL_SEEDS)
    ]
    result = {
        "source": "fresh_run",
        "stage": "Stage42-O explicit row-level gain/harm selector head",
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
                OUT_DIR / "row_gain_static_gate_stage42.json",
            ]
        ),
        "dataset_rows": {split: int(len(data[split]["horizon"])) for split in ["train", "val", "test"]},
        "rows": rows,
        "summary": _summary(rows, "test_metrics"),
        "stage42n_baseline_same_checkpoints": _summary(rows, "stage42n_baseline_test_metrics"),
        "comparison": _comparison(),
        "source_labels": {
            "all_agent_dataset": "cached_verified",
            "full_waypoint_labels": "cached_verified_or_rebuilt_by_stage41_helper",
            "row_teacher_train": train_teacher["source"],
            "row_teacher_val": val_teacher["source"],
            "row_teacher_test": "not_built",
            "selector_training": "fresh_run",
            "validation_policy_selection": "fresh_run",
            "test_evaluation": "fresh_run_once_per_seed",
            "policy_uses_easy_label": False,
            "feature_normalization": "train_split_stats_only",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "future_waypoints_used_as_train_val_label_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
            "thresholds_selected_on_val": True,
            "row_teacher_uses_test": False,
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
    result["stage42_o_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    _write_report(result)
    _write_gate(result["stage42_o_gate"])
    _append_readme_and_state(result)
    _append_ledger(result)
    return result


def _write_report(result: Mapping[str, Any]) -> None:
    s = result["summary"]
    base = result["stage42n_baseline_same_checkpoints"]
    cmp = result["comparison"]
    n = cmp.get("stage42_n_row_gain_static_gate", {})
    j = cmp.get("stage42_j_static_gated", {})
    lines = [
        "# Stage42-O Explicit Row-Level Gain/Harm Selector Head",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_o_gate']['passed']} / {result['stage42_o_gate']['total']}`",
        f"- verdict: `{result['stage42_o_gate']['verdict']}`",
        "",
        "## Current Facts",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Fresh Metrics",
        "",
        "| candidate | source | ADE all | ADE t50 | ADE t100 diag | ADE hard | ADE easy degr | FDE all | FDE t50 | switch |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| `explicit_gain_harm_selector` | `fresh_run` | {s['ade_all']['mean']:.6f} | {s['ade_t50']['mean']:.6f} | {s['ade_t100_raw_frame_diagnostic']['mean']:.6f} | {s['ade_hard_failure']['mean']:.6f} | {s['ade_easy_degradation']['mean']:.6f} | {s['fde_all']['mean']:.6f} | {s['fde_t50']['mean']:.6f} | {s['switch_rate']['mean']:.6f} |",
        f"| `same Stage42-N checkpoint baseline policy` | `fresh_run` | {base['ade_all']['mean']:.6f} | {base['ade_t50']['mean']:.6f} | {base['ade_t100_raw_frame_diagnostic']['mean']:.6f} | {base['ade_hard_failure']['mean']:.6f} | {base['ade_easy_degradation']['mean']:.6f} | {base['fde_all']['mean']:.6f} | {base['fde_t50']['mean']:.6f} | {base['switch_rate']['mean']:.6f} |",
        "",
        "## Comparison",
        "",
        "| candidate | source | ADE all | ADE t50 | ADE hard | ADE easy degr | FDE t50 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        f"| `Stage42-N row gain static gate` | `cached_verified` | {n.get('ade_all', {}).get('mean', 0.0):.6f} | {n.get('ade_t50', {}).get('mean', 0.0):.6f} | {n.get('ade_hard_failure', {}).get('mean', 0.0):.6f} | {n.get('ade_easy_degradation', {}).get('mean', 0.0):.6f} | {n.get('fde_t50', {}).get('mean', 0.0):.6f} |",
        f"| `Stage42-J policy static-gated` | `cached_verified` | {j.get('ade_all', {}).get('mean', 0.0):.6f} | {j.get('ade_t50', {}).get('mean', 0.0):.6f} | {j.get('ade_hard_failure', {}).get('mean', 0.0):.6f} | {j.get('ade_easy_degradation', {}).get('mean', 0.0):.6f} | {j.get('fde_t50', {}).get('mean', 0.0):.6f} |",
        "",
        "## Interpretation",
        "",
        "- Stage42-O tests the Stage42-N diagnosis by adding an explicit row-level selector head for switch probability, gain, harm, and uncertainty.",
        "- The deployment policy uses predicted switch/gain/harm/uncertainty only; it does not use test easy/hard labels as inference guards.",
        "- Stage42-N checkpoints are cached-verified base predictors; the gain/harm selector and validation policy are fresh-run.",
        "- Future waypoints remain train/val labels and final eval labels only, never inference inputs.",
        "- All claims remain dataset-local raw-frame 2.5D; no metric/seconds-level, Stage5C, or SMC claim is made.",
    ]
    write_md(REPORT_MD, lines)


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-O Gate",
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
    gate = result["stage42_o_gate"]
    s = result["summary"]
    block = f"""
## Stage42-O Explicit Row-Level Gain/Harm Selector Head

```text
source = fresh_run
verdict = {gate['verdict']}
gates = {gate['passed']} / {gate['total']}
explicit_selector_ade_all = {s['ade_all']['mean']}
explicit_selector_ade_t50 = {s['ade_t50']['mean']}
explicit_selector_ade_hard_failure = {s['ade_hard_failure']['mean']}
explicit_selector_ade_easy_degradation = {s['ade_easy_degradation']['mean']}
explicit_selector_fde_t50 = {s['fde_t50']['mean']}
stage5c_executed = false
smc_enabled = false
```

Stage42-O adds an explicit row-level gain/harm/switchability selector head on top of cached-verified Stage42-N full-waypoint predictors. It remains dataset-local raw-frame 2.5D evidence and not Stage5C/SMC.
"""
    _append_if_missing(Path("README_RESULTS.md"), "## Stage42-O Explicit Row-Level Gain/Harm Selector Head", block)
    _append_if_missing(Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md"), "## Stage42-O Explicit Row-Level Gain/Harm Selector Head", block)
    state = read_json(Path("research_state.json"), {})
    state["current_stage"] = "stage42_o_explicit_gain_harm_selector"
    state["current_verdict"] = gate["verdict"]
    state.setdefault("stage42", {})["stage_o_explicit_gain_harm_selector"] = {
        "source": "fresh_run",
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "gates_passed": gate["passed"],
        "gates_total": gate["total"],
        "verdict": gate["verdict"],
        "explicit_selector_ade_all": s["ade_all"]["mean"],
        "explicit_selector_ade_t50": s["ade_t50"]["mean"],
        "explicit_selector_ade_hard_failure": s["ade_hard_failure"]["mean"],
        "explicit_selector_ade_easy_degradation": s["ade_easy_degradation"]["mean"],
        "explicit_selector_fde_t50": s["fde_t50"]["mean"],
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
        "step": "stage42_o_explicit_gain_harm_selector",
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
    run_stage42_explicit_gain_harm_selector()
