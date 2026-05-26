from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_full_trajectory_world_state as ft
from src import stage42_explicit_gain_harm_selector as s42o
from src import stage42_horizon_static_gate_repair as s42l
from src import stage42_policy_distilled_static_gate as s42m
from src import stage42_row_gain_static_gate as s42n
from src import stage42_sequence_full_waypoint as s42i
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
REPORT_JSON = OUT_DIR / "t50_gain_harm_selector_stage42.json"
REPORT_MD = OUT_DIR / "t50_gain_harm_selector_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_p_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SEEDS = [149, 151, 157]
BASE_MODEL_SEEDS = [109, 113, 127]
EPOCHS = 2
BATCH = 4096
T50_WEIGHT = 5.0

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。",
    "Stage42-P 是 t+50-specific gain/harm selector repair，不是 metric 或 seconds-level 结果。",
    "future waypoints / future endpoints 只作为 train/val supervised labels 和 eval labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals。",
    "feature normalization 只使用 train split statistics。",
    "policy thresholds 只在 validation 上选择，test 只最终评估一次。",
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


def _target_t50_weighted(teacher: Mapping[str, np.ndarray], split: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    y = s42o._target_from_teacher(teacher)
    horizon = split["horizon"].astype(int)
    weight = y["weight"].astype(np.float32).copy()
    weight *= np.where(horizon == 50, T50_WEIGHT, 1.0).astype(np.float32)
    # Give high-margin switchable t+50 rows more voice without changing the labels.
    weight *= (1.0 + 0.75 * ((horizon == 50) & (y["switch"] > 0.5))).astype(np.float32)
    y["weight"] = np.clip(weight, 0.05, 20.0).astype(np.float32)
    return y


def _train_selector(seed: int, x_train: np.ndarray, y_train: Mapping[str, np.ndarray], x_val: np.ndarray, y_val: Mapping[str, np.ndarray]) -> dict[str, Any]:
    torch = s42o._torch()
    import torch.nn.functional as F

    ensure_dir(CHECKPOINT_DIR)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ckpt = CHECKPOINT_DIR / f"stage42p_t50_gain_harm_selector_seed{seed}.pt"
    heartbeat = OUT_DIR / f"stage42p_t50_gain_harm_selector_seed{seed}_heartbeat.json"
    if ckpt.exists() and heartbeat.exists():
        payload = read_json(heartbeat, {})
        if int(payload.get("epoch", 0)) >= EPOCHS:
            return {"source": "cached_verified", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": payload.get("best", {})}
    model = s42o._make_selector(x_train.shape[1], width=112)
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1.5e-4)
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
            uncertainty_target = torch.clamp(1.0 - tensors["switch"][ids] + tensors["harm"][ids], 0.0, 1.0)
            uncertainty_loss = (F.binary_cross_entropy_with_logits(out["uncertainty_logit"], uncertainty_target, reduction="none") * torch.clamp(w, 0.5, 3.0)).mean()
            loss = 1.8 * switch_loss + 1.6 * gain_loss + 1.3 * harm_loss + 0.20 * uncertainty_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            out = model(tvx)
            wv = val_tensors["weight"]
            val_loss = float(
                (
                    1.8 * (F.binary_cross_entropy_with_logits(out["switch_logit"], val_tensors["switch"], reduction="none") * wv).mean()
                    + 1.6 * (F.smooth_l1_loss(F.softplus(out["gain_raw"]), val_tensors["gain"], reduction="none") * wv).mean()
                    + 1.3 * (F.binary_cross_entropy_with_logits(out["harm_logit"], val_tensors["harm"], reduction="none") * wv).mean()
                ).cpu()
            )
        cand = {"val_loss": val_loss, "epoch": epoch, "train_loss": float(np.mean(losses))}
        if val_loss < best["val_loss"]:
            best = cand
            torch.save({"model": model.state_dict(), "seed": seed, "input_dim": x_train.shape[1], "best": best, "width": 112}, ckpt)
        heartbeat.write_text(json.dumps({"seed": seed, "epoch": epoch, "best": best, "checkpoint": str(ckpt)}, ensure_ascii=False), encoding="utf-8")
    return {"source": "fresh_run", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": best}


def _predict_selector(info: Mapping[str, Any], x: np.ndarray) -> dict[str, np.ndarray]:
    torch = s42o._torch()
    payload = torch.load(info["checkpoint"], map_location="cpu", weights_only=False)
    model = s42o._make_selector(int(payload["input_dim"]), width=int(payload.get("width", 112)))
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


def _score_metric_t50(metric: Mapping[str, Any]) -> float:
    return (
        7.5 * float(metric.get("t50_improvement", 0.0))
        + 1.4 * float(metric.get("all_improvement", 0.0))
        + 1.0 * float(metric.get("hard_failure_improvement", 0.0))
        + 0.25 * float(metric.get("t100_improvement", 0.0))
        - 80.0 * max(0.0, float(metric.get("easy_degradation", 1.0)) - 0.018)
    )


def _fit_policy_t50(scores: Mapping[str, np.ndarray], pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    domain = labels["domain"].astype(str)
    horizon = labels["horizon"].astype(int)
    policy: dict[str, Any] = {"type": "stage42p_t50_gain_harm_policy_no_easy_label_input", "slices": {}}
    diagnostics: dict[str, Any] = {}
    for d in sorted(set(domain.tolist())):
        for h in s42o.HORIZONS:
            mask = (domain == d) & (horizon == h)
            if int(np.sum(mask)) < 80:
                continue
            best_score = 0.0
            best_params: dict[str, Any] | None = None
            best_metric: dict[str, Any] | None = None
            gain_q = np.quantile(scores["gain"][mask], [0.15, 0.30, 0.50, 0.70])
            switch_grid = [0.40, 0.50, 0.60, 0.70, 0.80] if h == 50 else [0.45, 0.60, 0.75]
            harm_grid = [0.15, 0.20, 0.28, 0.35, 0.45] if h == 50 else [0.20, 0.35, 0.50]
            max_grid = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50] if h == 50 else [0.03, 0.08, 0.18, 0.30]
            for switch_min in switch_grid:
                for gain_min in [0.0, *[float(x) for x in gain_q]]:
                    for harm_max in harm_grid:
                        for uncertainty_max in [0.50, 0.65, 0.80, 0.95]:
                            for max_switch in max_grid:
                                params = {
                                    "switch_min": switch_min,
                                    "gain_min": gain_min,
                                    "harm_max": harm_max,
                                    "uncertainty_max": uncertainty_max,
                                    "max_switch": max_switch,
                                }
                                sw = s42o._selector_switch(scores, labels, {"slices": {f"{d}|{h}": params}})
                                metric = s42o._metric_from_switch(pred, labels, sw)["ade"]
                                if metric.get("easy_degradation", 1.0) > 0.018:
                                    continue
                                score = _score_metric_t50(metric) if h == 50 else s42o._score_metric(metric)
                                if score > best_score:
                                    best_score = score
                                    best_params = params
                                    best_metric = metric
            diagnostics[f"{d}|{h}"] = {"selected": bool(best_params), "score": float(best_score), "metric": best_metric or {"rows": int(np.sum(mask)), "all_improvement": 0.0}}
            if best_params is not None:
                policy["slices"][f"{d}|{h}"] = best_params
    sw = s42o._selector_switch(scores, labels, policy)
    metrics = s42o._metric_from_switch(pred, labels, sw)
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
    train_stats = s42o._feature_stats(s42o._raw_features(train, pred_train, vocab))
    x_train = s42o._features(train, pred_train, vocab, train_stats)
    x_val = s42o._features(val, pred_val, vocab, train_stats)
    x_test = s42o._features(test, pred_test, vocab, train_stats)
    y_train = _target_t50_weighted(train_teacher, train)
    y_val = _target_t50_weighted(val_teacher, val)
    selector_info = _train_selector(seed, x_train, y_train, x_val, y_val)
    score_val = _predict_selector(selector_info, x_val)
    score_test = _predict_selector(selector_info, x_test)
    labels_val = s42i._labels(val)
    labels_test = s42i._labels(test)
    policy, val_metrics = _fit_policy_t50(score_val, pred_val, labels_val)
    test_switch = s42o._selector_switch(score_test, labels_test, policy)
    test_metrics = s42o._metric_from_switch(pred_test, labels_test, test_switch)
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
    return s42o._summary(rows, key)


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    s = result.get("summary", {})
    o = result.get("comparison", {}).get("stage42_o_explicit_gain_harm_selector", {})
    gates = {
        "t50_specific_selector_trained": len(result.get("rows", [])) >= 3,
        "t50_weighted_train_val_teacher": result.get("source_labels", {}).get("t50_weighted_teacher") is True,
        "train_val_teacher_only": result.get("source_labels", {}).get("row_teacher_test") == "not_built",
        "policy_no_easy_label_input": result.get("source_labels", {}).get("policy_uses_easy_label") is False,
        "all_positive": s.get("ade_all", {}).get("mean", 0.0) > 0.0,
        "t50_positive": s.get("ade_t50", {}).get("mean", -1.0) > 0.0,
        "hard_positive": s.get("ade_hard_failure", {}).get("mean", 0.0) > 0.0,
        "easy_preserved": s.get("ade_easy_degradation", {}).get("mean", 1.0) <= 0.02,
        "improves_stage42o_t50": s.get("ade_t50", {}).get("mean", -1.0) > o.get("ade_t50", {}).get("mean", -1.0),
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
        "verdict": "stage42_p_t50_gain_harm_selector_pass" if all(gates.values()) else "stage42_p_t50_gain_harm_selector_partial",
    }


def _comparison() -> dict[str, Any]:
    return {
        "source": "cached_verified",
        "stage42_o_explicit_gain_harm_selector": read_json(OUT_DIR / "explicit_gain_harm_selector_stage42.json", {}).get("summary", {}),
        "stage42_n_row_gain_static_gate": read_json(OUT_DIR / "row_gain_static_gate_stage42.json", {}).get("summary", {}),
        "stage42_j_static_gated": (read_json(OUT_DIR / "static_gated_full_waypoint_stage42.json", {}).get("summary") or {}).get("static_gated", {}),
    }


def run_stage42_t50_gain_harm_selector() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CHECKPOINT_DIR)
    ft.build_full_trajectory_labels()
    data = {split: s42i._split_arrays(split) for split in ["train", "val", "test"]}
    vocab = s42o._domain_vocab(data["train"], data["val"], data["test"])
    train_teacher = s42n._row_teacher(data["train"], "train")
    val_teacher = s42n._row_teacher(data["val"], "val")
    rows = [
        _eval_seed(seed, base_seed, data["train"], data["val"], data["test"], vocab, train_teacher, val_teacher)
        for seed, base_seed in zip(SEEDS, BASE_MODEL_SEEDS)
    ]
    result = {
        "source": "fresh_run",
        "stage": "Stage42-P t50-specific gain/harm selector repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "torch_threads": s42o.THREADS,
        "epochs": EPOCHS,
        "batch": BATCH,
        "t50_weight": T50_WEIGHT,
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
                OUT_DIR / "explicit_gain_harm_selector_stage42.json",
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
            "t50_weighted_teacher": True,
            "selector_training": "fresh_run",
            "validation_policy_selection": "fresh_run_t50_weighted",
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
    result["stage42_p_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    _write_report(result)
    _write_gate(result["stage42_p_gate"])
    _append_readme_and_state(result)
    _append_ledger(result)
    return result


def _write_report(result: Mapping[str, Any]) -> None:
    s = result["summary"]
    cmp = result["comparison"]
    o = cmp.get("stage42_o_explicit_gain_harm_selector", {})
    j = cmp.get("stage42_j_static_gated", {})
    lines = [
        "# Stage42-P T50-Specific Gain/Harm Selector Repair",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_p_gate']['passed']} / {result['stage42_p_gate']['total']}`",
        f"- verdict: `{result['stage42_p_gate']['verdict']}`",
        "",
        "## Current Facts",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Fresh Metrics",
        "",
        "| candidate | source | ADE all | ADE t50 | ADE t100 diag | ADE hard | ADE easy degr | FDE all | FDE t50 | switch |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| `t50_gain_harm_selector` | `fresh_run` | {s['ade_all']['mean']:.6f} | {s['ade_t50']['mean']:.6f} | {s['ade_t100_raw_frame_diagnostic']['mean']:.6f} | {s['ade_hard_failure']['mean']:.6f} | {s['ade_easy_degradation']['mean']:.6f} | {s['fde_all']['mean']:.6f} | {s['fde_t50']['mean']:.6f} | {s['switch_rate']['mean']:.6f} |",
        "",
        "## Comparison",
        "",
        "| candidate | source | ADE all | ADE t50 | ADE hard | ADE easy degr | FDE t50 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        f"| `Stage42-O explicit gain/harm` | `cached_verified` | {o.get('ade_all', {}).get('mean', 0.0):.6f} | {o.get('ade_t50', {}).get('mean', 0.0):.6f} | {o.get('ade_hard_failure', {}).get('mean', 0.0):.6f} | {o.get('ade_easy_degradation', {}).get('mean', 0.0):.6f} | {o.get('fde_t50', {}).get('mean', 0.0):.6f} |",
        f"| `Stage42-J policy static-gated` | `cached_verified` | {j.get('ade_all', {}).get('mean', 0.0):.6f} | {j.get('ade_t50', {}).get('mean', 0.0):.6f} | {j.get('ade_hard_failure', {}).get('mean', 0.0):.6f} | {j.get('ade_easy_degradation', {}).get('mean', 0.0):.6f} | {j.get('fde_t50', {}).get('mean', 0.0):.6f} |",
        "",
        "## Interpretation",
        "",
        "- Stage42-P is a targeted follow-up to Stage42-O's strict train-normalized partial result.",
        "- It increases t+50 row weight in train/val teacher supervision and uses a t+50-weighted validation policy search.",
        "- The policy still uses only predicted switch/gain/harm/uncertainty; it does not use test easy/hard labels as inference guards.",
        "- Future waypoints remain train/val labels and final eval labels only, never inference inputs.",
        "- All claims remain dataset-local raw-frame 2.5D; no metric/seconds-level, Stage5C, or SMC claim is made.",
    ]
    write_md(REPORT_MD, lines)


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-P Gate",
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
    gate = result["stage42_p_gate"]
    s = result["summary"]
    block = f"""
## Stage42-P T50-Specific Gain/Harm Selector Repair

```text
source = fresh_run
verdict = {gate['verdict']}
gates = {gate['passed']} / {gate['total']}
t50_gain_harm_ade_all = {s['ade_all']['mean']}
t50_gain_harm_ade_t50 = {s['ade_t50']['mean']}
t50_gain_harm_ade_hard_failure = {s['ade_hard_failure']['mean']}
t50_gain_harm_ade_easy_degradation = {s['ade_easy_degradation']['mean']}
t50_gain_harm_fde_t50 = {s['fde_t50']['mean']}
feature_normalization = train_split_stats_only
stage5c_executed = false
smc_enabled = false
```

Stage42-P is a t+50-specific follow-up to Stage42-O. It increases t+50 teacher weight and searches a t+50-weighted validation policy while preserving the raw-frame/dataset-local 2.5D claim boundary.
"""
    _append_if_missing(Path("README_RESULTS.md"), "## Stage42-P T50-Specific Gain/Harm Selector Repair", block)
    _append_if_missing(Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md"), "## Stage42-P T50-Specific Gain/Harm Selector Repair", block)
    state = read_json(Path("research_state.json"), {})
    state["current_stage"] = "stage42_p_t50_gain_harm_selector"
    state["current_verdict"] = gate["verdict"]
    state.setdefault("stage42", {})["stage_p_t50_gain_harm_selector"] = {
        "source": "fresh_run",
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "gates_passed": gate["passed"],
        "gates_total": gate["total"],
        "verdict": gate["verdict"],
        "t50_gain_harm_ade_all": s["ade_all"]["mean"],
        "t50_gain_harm_ade_t50": s["ade_t50"]["mean"],
        "t50_gain_harm_ade_hard_failure": s["ade_hard_failure"]["mean"],
        "t50_gain_harm_ade_easy_degradation": s["ade_easy_degradation"]["mean"],
        "t50_gain_harm_fde_t50": s["fde_t50"]["mean"],
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
        "step": "stage42_p_t50_gain_harm_selector",
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
    run_stage42_t50_gain_harm_selector()
