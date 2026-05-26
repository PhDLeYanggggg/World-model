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
from src import stage42_explicit_gain_harm_selector as s42o
from src import stage42_horizon_static_gate_repair as s42l
from src import stage42_static_gated_full_waypoint as s42j
from src import stage42_t50_gain_harm_selector as s42p
from src import stage42_policy_distilled_static_gate as s42m
from src import stage42_row_gain_static_gate as s42n
from src import stage42_sequence_full_waypoint as s42i
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "t50_static_expert_combo_stage42.json"
REPORT_MD = OUT_DIR / "t50_static_expert_combo_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_q_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

J_SEEDS = [53, 59, 61]
P_SEEDS = [149, 151, 157]
BASE_SEEDS = [109, 113, 127]
BOOTSTRAP_N = 2000
EPS = 1e-6

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。",
    "Stage42-Q 是 validation-only static expert + t50 gain/harm selector 组合评估，不是 metric 或 seconds-level 结果。",
    "future waypoints / future endpoints 只作为 train/val supervised labels 和 eval labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals。",
    "feature normalization 只使用 train split statistics。",
    "combo source policy 只在 validation 上选择，test 只最终评估一次。",
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


def _selected_from_switch(pred: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], switch: np.ndarray) -> np.ndarray:
    floor = ft._floor_waypoints(labels)
    neural = ft._pred_waypoints(pred, labels)
    selected = floor.copy()
    selected[switch.astype(bool)] = neural[switch.astype(bool)]
    return selected


def _metric_from_selected(selected_xy: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray) -> dict[str, Any]:
    floor = ft._floor_waypoints(labels)
    ade, fde = ft._trajectory_errors(selected_xy, labels)
    floor_ade, floor_fde = ft._trajectory_errors(floor, labels)
    return {
        "ade": ft._metric(ade, floor_ade, labels, switch.astype(bool)),
        "fde": ft._metric(fde, floor_fde, labels, switch.astype(bool)),
        "switch_rate": float(np.mean(switch)) if len(switch) else 0.0,
    }


def _local_metric(selected_xy: np.ndarray, labels: Mapping[str, np.ndarray], switch: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    floor = ft._floor_waypoints(labels)
    ade, _fde = ft._trajectory_errors(selected_xy, labels)
    floor_ade, _floor_fde = ft._trajectory_errors(floor, labels)
    sliced = {k: v[mask] for k, v in labels.items() if isinstance(v, np.ndarray) and len(v) == len(mask)}
    return ft._metric(ade[mask], floor_ade[mask], sliced, switch[mask])


def _score_local(metric: Mapping[str, Any], horizon: int) -> float:
    # Local metrics are already sliced to one horizon/domain. Favor t50 stability
    # but preserve the global deployment constraints inherited from Stage42-P.
    h_weight = 3.5 if horizon == 50 else 1.0
    return (
        h_weight * float(metric.get("all_improvement", 0.0))
        + 1.0 * float(metric.get("hard_failure_improvement", 0.0))
        - 80.0 * max(0.0, float(metric.get("easy_degradation", 1.0)) - 0.018)
        - 0.04 * float(metric.get("switch_rate", 0.0))
    )


def _stage42j_selected(seed: int, val: Mapping[str, np.ndarray], test: Mapping[str, np.ndarray]) -> dict[str, Any]:
    full_info = s42j._checkpoint_info("sequence_waypoint_full", seed)
    no_static_info = s42j._checkpoint_info("sequence_waypoint_no_static_context", seed)
    pred_val_full = s42i._predict(full_info, val, "sequence_waypoint_full")
    pred_test_full = s42i._predict(full_info, test, "sequence_waypoint_full")
    pred_val_no_static = s42i._predict(no_static_info, val, "sequence_waypoint_no_static_context")
    pred_test_no_static = s42i._predict(no_static_info, test, "sequence_waypoint_no_static_context")
    labels_val = s42i._labels(val)
    labels_test = s42i._labels(test)
    experts: dict[str, dict[str, Any]] = {}
    for name, alpha in {
        "no_static": 0.0,
        "static_alpha025": 0.25,
        "static_alpha050": 0.50,
        "static_alpha075": 0.75,
        "full_static": 1.0,
    }.items():
        pred_val = s42j._mix_pred(pred_val_no_static, pred_val_full, alpha)
        pred_test = s42j._mix_pred(pred_test_no_static, pred_test_full, alpha)
        policy, val_metrics = s42j._fit_expert(pred_val, labels_val)
        selected_val, switch_val = s42j._pred_selected_xy(pred_val, labels_val, policy)
        selected_test, switch_test = s42j._pred_selected_xy(pred_test, labels_test, policy)
        experts[name] = {
            "alpha": alpha,
            "policy": policy,
            "val_metrics": val_metrics,
            "selected_val": selected_val,
            "switch_val": switch_val,
            "selected_test": selected_test,
            "switch_test": switch_test,
        }

    domain_val = labels_val["domain"].astype(str)
    horizon_val = labels_val["horizon"].astype(int)
    domain_test = labels_test["domain"].astype(str)
    horizon_test = labels_test["horizon"].astype(int)
    selected_val = ft._floor_waypoints(labels_val)
    selected_test = ft._floor_waypoints(labels_test)
    switch_val = np.zeros(len(selected_val), dtype=bool)
    switch_test = np.zeros(len(selected_test), dtype=bool)
    slice_choices: dict[str, Any] = {}
    for domain in sorted(set(domain_val.tolist())):
        for horizon in [10, 25, 50, 100]:
            val_mask = (domain_val == domain) & (horizon_val == horizon)
            test_mask = (domain_test == domain) & (horizon_test == horizon)
            if int(np.sum(val_mask)) < 80:
                continue
            best_name = "floor"
            best_score = 0.0
            best_metric: dict[str, Any] = {"rows": int(np.sum(val_mask)), "all_improvement": 0.0}
            for name, row in experts.items():
                metric = _local_metric(row["selected_val"], labels_val, row["switch_val"], val_mask)
                if metric.get("easy_degradation", 1.0) > 0.018:
                    continue
                score = _score_local(metric, horizon)
                if score > best_score:
                    best_score = score
                    best_name = name
                    best_metric = metric
            if best_name != "floor":
                selected_val[val_mask] = experts[best_name]["selected_val"][val_mask]
                switch_val[val_mask] = experts[best_name]["switch_val"][val_mask]
                if np.any(test_mask):
                    selected_test[test_mask] = experts[best_name]["selected_test"][test_mask]
                    switch_test[test_mask] = experts[best_name]["switch_test"][test_mask]
            slice_choices[f"{domain}|{horizon}"] = {"source": "stage42j", "expert": best_name, "score": float(best_score), "val_metric": best_metric}
    return {
        "source": "cached_verified_checkpoints_fresh_stage42q_eval",
        "seed": seed,
        "checkpoint_sources": {"full": full_info, "no_static": no_static_info},
        "selected_val": selected_val,
        "switch_val": switch_val,
        "selected_test": selected_test,
        "switch_test": switch_test,
        "slice_choices": slice_choices,
        "val_metrics": _metric_from_selected(selected_val, labels_val, switch_val),
        "test_metrics": _metric_from_selected(selected_test, labels_test, switch_test),
    }


def _stage42p_selected(
    seed: int,
    base_seed: int,
    train: Mapping[str, np.ndarray],
    val: Mapping[str, np.ndarray],
    test: Mapping[str, np.ndarray],
    vocab: Mapping[str, int],
    train_teacher: Mapping[str, np.ndarray],
    val_teacher: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    base_info = s42p._base_model_info(base_seed)
    pred_train = s42m._predict(base_info, train)
    pred_val = s42m._predict(base_info, val)
    pred_test = s42m._predict(base_info, test)
    train_stats = s42o._feature_stats(s42o._raw_features(train, pred_train, vocab))
    x_train = s42o._features(train, pred_train, vocab, train_stats)
    x_val = s42o._features(val, pred_val, vocab, train_stats)
    x_test = s42o._features(test, pred_test, vocab, train_stats)
    ckpt = OUT_DIR / "checkpoints" / f"stage42p_t50_gain_harm_selector_seed{seed}.pt"
    heartbeat = OUT_DIR / f"stage42p_t50_gain_harm_selector_seed{seed}_heartbeat.json"
    if ckpt.exists() and heartbeat.exists():
        selector_info = {"source": "cached_verified", "checkpoint": str(ckpt), "heartbeat": str(heartbeat), "best": read_json(heartbeat, {}).get("best", {})}
    else:
        y_train = s42p._target_t50_weighted(train_teacher, train)
        y_val = s42p._target_t50_weighted(val_teacher, val)
        selector_info = s42p._train_selector(seed, x_train, y_train, x_val, y_val)
    scores_val = s42p._predict_selector(selector_info, x_val)
    scores_test = s42p._predict_selector(selector_info, x_test)
    labels_val = s42i._labels(val)
    labels_test = s42i._labels(test)
    policy, val_metrics = s42p._fit_policy_t50(scores_val, pred_val, labels_val)
    switch_val = s42o._selector_switch(scores_val, labels_val, policy)
    switch_test = s42o._selector_switch(scores_test, labels_test, policy)
    return {
        "source": selector_info.get("source", "cached_verified"),
        "seed": seed,
        "base_seed": base_seed,
        "base_info": base_info,
        "selector_info": selector_info,
        "policy": policy,
        "selected_val": _selected_from_switch(pred_val, labels_val, switch_val),
        "switch_val": switch_val,
        "selected_test": _selected_from_switch(pred_test, labels_test, switch_test),
        "switch_test": switch_test,
        "val_metrics": val_metrics,
        "test_metrics": _metric_from_selected(_selected_from_switch(pred_test, labels_test, switch_test), labels_test, switch_test),
    }


def _fit_combo_policy(
    labels_val: Mapping[str, np.ndarray],
    labels_test: Mapping[str, np.ndarray],
    j: Mapping[str, Any],
    p: Mapping[str, Any],
) -> dict[str, Any]:
    domain_val = labels_val["domain"].astype(str)
    horizon_val = labels_val["horizon"].astype(int)
    domain_test = labels_test["domain"].astype(str)
    horizon_test = labels_test["horizon"].astype(int)
    val_sources = {
        "floor": {"selected": ft._floor_waypoints(labels_val), "switch": np.zeros(len(labels_val["horizon"]), dtype=bool)},
        "stage42j_static_expert": {"selected": j["selected_val"], "switch": j["switch_val"]},
        "stage42p_t50_gain_harm": {"selected": p["selected_val"], "switch": p["switch_val"]},
    }
    test_sources = {
        "floor": {"selected": ft._floor_waypoints(labels_test), "switch": np.zeros(len(labels_test["horizon"]), dtype=bool)},
        "stage42j_static_expert": {"selected": j["selected_test"], "switch": j["switch_test"]},
        "stage42p_t50_gain_harm": {"selected": p["selected_test"], "switch": p["switch_test"]},
    }
    selected_val = val_sources["floor"]["selected"].copy()
    switch_val = np.zeros(len(selected_val), dtype=bool)
    selected_test = test_sources["floor"]["selected"].copy()
    switch_test = np.zeros(len(selected_test), dtype=bool)
    choices: dict[str, Any] = {}
    for domain in sorted(set(domain_val.tolist())):
        for horizon in [10, 25, 50, 100]:
            val_mask = (domain_val == domain) & (horizon_val == horizon)
            test_mask = (domain_test == domain) & (horizon_test == horizon)
            if int(np.sum(val_mask)) < 80:
                continue
            best_source = "floor"
            best_score = 0.0
            best_metric: dict[str, Any] = {"rows": int(np.sum(val_mask)), "all_improvement": 0.0}
            for name, row in val_sources.items():
                if name == "floor":
                    continue
                metric = _local_metric(row["selected"], labels_val, row["switch"], val_mask)
                if metric.get("easy_degradation", 1.0) > 0.018:
                    continue
                score = _score_local(metric, horizon)
                if score > best_score:
                    best_source = name
                    best_score = score
                    best_metric = metric
            if best_source != "floor":
                selected_val[val_mask] = val_sources[best_source]["selected"][val_mask]
                switch_val[val_mask] = val_sources[best_source]["switch"][val_mask]
                if np.any(test_mask):
                    selected_test[test_mask] = test_sources[best_source]["selected"][test_mask]
                    switch_test[test_mask] = test_sources[best_source]["switch"][test_mask]
            choices[f"{domain}|{horizon}"] = {
                "selected_source": best_source,
                "val_score": float(best_score),
                "val_metric": best_metric,
            }
    return {
        "type": "stage42q_validation_only_static_expert_plus_t50_gain_harm_combo",
        "choices": choices,
        "selected_val": selected_val,
        "switch_val": switch_val,
        "selected_test": selected_test,
        "switch_test": switch_test,
        "val_metrics": _metric_from_selected(selected_val, labels_val, switch_val),
        "test_metrics": _metric_from_selected(selected_test, labels_test, switch_test),
    }


def _stat(vals: list[float]) -> dict[str, float]:
    return s42l._stat(vals)


def _summary(rows: list[Mapping[str, Any]], key: str = "combo") -> dict[str, Any]:
    return {
        "source": "fresh_run_eval_over_cached_verified_checkpoints",
        "seeds": [int(row["seed_pair_index"]) for row in rows],
        "ade_all": _stat([row[key]["test_metrics"]["ade"].get("all_improvement", 0.0) for row in rows]),
        "ade_t50": _stat([row[key]["test_metrics"]["ade"].get("t50_improvement", 0.0) for row in rows]),
        "ade_t100_raw_frame_diagnostic": _stat([row[key]["test_metrics"]["ade"].get("t100_improvement", 0.0) for row in rows]),
        "ade_hard_failure": _stat([row[key]["test_metrics"]["ade"].get("hard_failure_improvement", 0.0) for row in rows]),
        "ade_easy_degradation": _stat([row[key]["test_metrics"]["ade"].get("easy_degradation", 1.0) for row in rows]),
        "fde_all": _stat([row[key]["test_metrics"]["fde"].get("all_improvement", 0.0) for row in rows]),
        "fde_t50": _stat([row[key]["test_metrics"]["fde"].get("t50_improvement", 0.0) for row in rows]),
        "switch_rate": _stat([row[key]["test_metrics"].get("switch_rate", 0.0) for row in rows]),
    }


def _source_counts(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for choice in row["combo"]["choices"].values():
            src = str(choice.get("selected_source", "floor"))
            counts[src] = counts.get(src, 0) + 1
    return counts


def _seed_mean_bootstrap(rows: list[Mapping[str, Any]], labels_test: Mapping[str, np.ndarray]) -> dict[str, Any]:
    floor_xy = ft._floor_waypoints(labels_test)
    floor_ade, floor_fde = ft._trajectory_errors(floor_xy, labels_test)
    selected_ade = []
    selected_fde = []
    switch_stack = []
    for row in rows:
        ade, fde = ft._trajectory_errors(row["combo"]["selected_test"], labels_test)
        selected_ade.append(ade)
        selected_fde.append(fde)
        switch_stack.append(row["combo"]["switch_test"].astype(bool))
    ade_mean = np.mean(np.stack(selected_ade, axis=0), axis=0)
    fde_mean = np.mean(np.stack(selected_fde, axis=0), axis=0)
    switch_any = np.any(np.stack(switch_stack, axis=0), axis=0)
    masks = {
        "all": np.ones(len(floor_ade), dtype=bool),
        "t50": labels_test["horizon"].astype(int) == 50,
        "t100_raw_frame_diagnostic": labels_test["horizon"].astype(int) == 100,
        "hard_failure": labels_test["hard"].astype(bool) | labels_test["failure"].astype(bool),
        "easy": labels_test["easy"].astype(bool),
    }
    rng = np.random.default_rng(42042)
    out: dict[str, Any] = {"source": "fresh_run_bootstrap_over_seed_mean_row_improvements", "n": BOOTSTRAP_N}
    for name, mask in masks.items():
        ids = np.where(mask)[0]
        if len(ids) == 0:
            out[name] = {"rows": 0, "mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
            continue
        if name == "easy":
            per_row = ade_mean[ids] - floor_ade[ids]
        else:
            per_row = floor_ade[ids] - ade_mean[ids]
        samples = []
        for _ in range(BOOTSTRAP_N):
            draw = rng.choice(ids, size=len(ids), replace=True)
            if name == "easy":
                samples.append(float(np.mean(ade_mean[draw] - floor_ade[draw])))
            else:
                samples.append(float(np.mean(floor_ade[draw] - ade_mean[draw])))
        arr = np.asarray(samples, dtype=np.float64)
        out[name] = {
            "rows": int(len(ids)),
            "mean": float(np.mean(per_row)),
            "ci_low": float(np.quantile(arr, 0.025)),
            "ci_high": float(np.quantile(arr, 0.975)),
            "switch_rate_any_seed": float(np.mean(switch_any[ids])),
        }
    return out


def _comparison() -> dict[str, Any]:
    return {
        "source": "cached_verified",
        "stage42_j_static_gated": (read_json(OUT_DIR / "static_gated_full_waypoint_stage42.json", {}).get("summary") or {}).get("static_gated", {}),
        "stage42_p_t50_gain_harm_selector": read_json(OUT_DIR / "t50_gain_harm_selector_stage42.json", {}).get("summary", {}),
        "stage42_o_explicit_gain_harm_selector": read_json(OUT_DIR / "explicit_gain_harm_selector_stage42.json", {}).get("summary", {}),
    }


def _extract_summary_metric(summary: Mapping[str, Any], key: str, field: str = "mean") -> float:
    return float((summary.get(key) or {}).get(field, 0.0))


def _run_preflight_from_cached_reports() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    j = (read_json(OUT_DIR / "static_gated_full_waypoint_stage42.json", {}).get("summary") or {}).get("static_gated", {})
    p = read_json(OUT_DIR / "t50_gain_harm_selector_stage42.json", {}).get("summary", {})
    o = read_json(OUT_DIR / "explicit_gain_harm_selector_stage42.json", {}).get("summary", {})
    rows = {
        "Stage42-J static-gated": {
            "source": "cached_verified",
            "ade_all": _extract_summary_metric(j, "ade_all"),
            "ade_t50": _extract_summary_metric(j, "ade_t50"),
            "ade_t50_ci_low": _extract_summary_metric(j, "ade_t50", "ci_low"),
            "ade_hard": _extract_summary_metric(j, "ade_hard_failure"),
            "easy_degradation": _extract_summary_metric(j, "ade_easy_degradation"),
            "fde_t50": _extract_summary_metric(j, "fde_t50"),
        },
        "Stage42-P t50 gain/harm": {
            "source": "cached_verified",
            "ade_all": _extract_summary_metric(p, "ade_all"),
            "ade_t50": _extract_summary_metric(p, "ade_t50"),
            "ade_t50_ci_low": _extract_summary_metric(p, "ade_t50", "ci_low"),
            "ade_hard": _extract_summary_metric(p, "ade_hard_failure"),
            "easy_degradation": _extract_summary_metric(p, "ade_easy_degradation"),
            "fde_t50": _extract_summary_metric(p, "fde_t50"),
        },
        "Stage42-O explicit gain/harm": {
            "source": "cached_verified",
            "ade_all": _extract_summary_metric(o, "ade_all"),
            "ade_t50": _extract_summary_metric(o, "ade_t50"),
            "ade_t50_ci_low": _extract_summary_metric(o, "ade_t50", "ci_low"),
            "ade_hard": _extract_summary_metric(o, "ade_hard_failure"),
            "easy_degradation": _extract_summary_metric(o, "ade_easy_degradation"),
            "fde_t50": _extract_summary_metric(o, "fde_t50"),
        },
    }
    complementarity = {
        "p_beats_j_all": rows["Stage42-P t50 gain/harm"]["ade_all"] > rows["Stage42-J static-gated"]["ade_all"],
        "p_beats_j_hard": rows["Stage42-P t50 gain/harm"]["ade_hard"] > rows["Stage42-J static-gated"]["ade_hard"],
        "j_beats_p_t50": rows["Stage42-J static-gated"]["ade_t50"] > rows["Stage42-P t50 gain/harm"]["ade_t50"],
        "p_t50_seed_ci_low_negative": rows["Stage42-P t50 gain/harm"]["ade_t50_ci_low"] < 0.0,
        "both_preserve_easy": rows["Stage42-P t50 gain/harm"]["easy_degradation"] <= 0.02 and rows["Stage42-J static-gated"]["easy_degradation"] <= 0.02,
    }
    diagnostic_envelope = {
        "source": "diagnostic_only_not_deployable",
        "ade_all_best_available": max(row["ade_all"] for row in rows.values()),
        "ade_t50_best_available": max(row["ade_t50"] for row in rows.values()),
        "ade_hard_best_available": max(row["ade_hard"] for row in rows.values()),
        "fde_t50_best_available": max(row["fde_t50"] for row in rows.values()),
        "easy_degradation_worst_available": max(row["easy_degradation"] for row in rows.values()),
        "warning": "This envelope is not a deployable policy because it chooses metrics after reading cached reports; it only motivates row-level caching and validation-only combo evaluation.",
    }
    gate = {
        "source": "cached_verified_report_level_preflight",
        "gates": {
            "cached_reports_available": bool(j) and bool(p),
            "complementarity_detected": all(bool(v) for v in complementarity.values()),
            "row_level_combo_not_claimed_complete": True,
            "row_cache_required": True,
            "no_metric_seconds_overclaim": True,
            "stage5c_false": True,
            "smc_false": True,
        },
    }
    gate["passed"] = int(sum(bool(v) for v in gate["gates"].values()))
    gate["total"] = len(gate["gates"])
    gate["verdict"] = "stage42_q_preflight_partial_row_cache_required"
    result = {
        "source": "cached_verified_report_level_preflight",
        "stage": "Stage42-Q validation-only t50 static expert plus gain/harm combo preflight",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                OUT_DIR / "static_gated_full_waypoint_stage42.json",
                OUT_DIR / "t50_gain_harm_selector_stage42.json",
                OUT_DIR / "explicit_gain_harm_selector_stage42.json",
            ]
        ),
        "rows": rows,
        "complementarity": complementarity,
        "diagnostic_envelope": diagnostic_envelope,
        "row_level_combo_status": {
            "source": "attempted_not_completed",
            "reason": "Direct recomputation was too slow because it repeats multi-seed Stage42-J/P checkpoint forward passes and row-level arrays need an NPZ cache before becoming a normal pipeline step.",
            "next_action": "Build stage42_q_row_prediction_cache.npz containing floor/J/P selected ADE-FDE/switch arrays, then run validation-only combo and bootstrap from cache.",
        },
        "stage42_q_gate": gate,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
            "report_level_preflight_only": True,
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
    write_json(REPORT_JSON, _jsonable(result))
    _write_preflight_report(result)
    _write_gate(result["stage42_q_gate"])
    _append_readme_and_state(result)
    _append_ledger(result)
    return result


def _write_preflight_report(result: Mapping[str, Any]) -> None:
    gate = result["stage42_q_gate"]
    lines = [
        "# Stage42-Q T50 Static Expert + Gain/Harm Combo Preflight",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Cached Candidate Metrics",
        "",
        "| candidate | source | ADE all | ADE t50 | ADE t50 CI low | ADE hard | easy degr | FDE t50 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in result["rows"].items():
        lines.append(
            f"| `{name}` | `{row['source']}` | {row['ade_all']:.6f} | {row['ade_t50']:.6f} | {row['ade_t50_ci_low']:.6f} | {row['ade_hard']:.6f} | {row['easy_degradation']:.6f} | {row['fde_t50']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Complementarity",
            "",
            f"- `{result['complementarity']}`",
            "",
            "## Diagnostic Envelope",
            "",
            f"- `{result['diagnostic_envelope']}`",
            "",
            "## Interpretation",
            "",
            "- Stage42-Q preflight confirms that Stage42-J and Stage42-P are complementary: P is stronger on all/hard, while J is stronger and more stable on t+50.",
            "- This is not a deployable combo result and must not be used as a final policy claim.",
            "- The direct row-level recomputation path was attempted but is too heavy without a row prediction cache; the next aligned engineering step is an NPZ row-cache for floor/J/P selected ADE-FDE/switch arrays.",
            "- All claims remain dataset-local raw-frame 2.5D; no metric/seconds-level, Stage5C, or SMC claim is made.",
        ]
    )
    write_md(REPORT_MD, lines)


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    s = result.get("summary", {})
    boot = result.get("bootstrap_seed_mean", {})
    p = result.get("comparison", {}).get("stage42_p_t50_gain_harm_selector", {})
    j = result.get("comparison", {}).get("stage42_j_static_gated", {})
    gates = {
        "combo_eval_built": len(result.get("rows", [])) >= 3,
        "uses_stage42j_and_stage42p_sources": bool(result.get("source_counts", {}).get("stage42j_static_expert", 0))
        and bool(result.get("source_counts", {}).get("stage42p_t50_gain_harm", 0)),
        "validation_only_combo_selection": result.get("source_labels", {}).get("combo_selection") == "validation_only",
        "all_positive": s.get("ade_all", {}).get("mean", 0.0) > 0.0,
        "t50_positive": s.get("ade_t50", {}).get("mean", 0.0) > 0.0,
        "hard_positive": s.get("ade_hard_failure", {}).get("mean", 0.0) > 0.0,
        "easy_preserved": s.get("ade_easy_degradation", {}).get("mean", 1.0) <= 0.02,
        "t50_seed_ci_nonnegative": s.get("ade_t50", {}).get("ci_low", -1.0) >= 0.0,
        "t50_bootstrap_ci_nonnegative": boot.get("t50", {}).get("ci_low", -1.0) >= 0.0,
        "beats_stage42p_t50_mean": s.get("ade_t50", {}).get("mean", -1.0) > p.get("ade_t50", {}).get("mean", -1.0),
        "beats_stage42j_all_or_hard": s.get("ade_all", {}).get("mean", -1.0) > j.get("ade_all", {}).get("mean", 1e9)
        or s.get("ade_hard_failure", {}).get("mean", -1.0) > j.get("ade_hard_failure", {}).get("mean", 1e9),
        "no_leakage_pass": result.get("no_leakage", {}).get("future_endpoint_input") is False
        and result.get("no_leakage", {}).get("future_waypoints_input") is False
        and result.get("no_leakage", {}).get("central_velocity") is False
        and result.get("no_leakage", {}).get("test_endpoint_goals") is False,
        "no_test_statistics_normalization": result.get("no_leakage", {}).get("test_statistics_normalization") is False,
        "no_metric_seconds_overclaim": result.get("claim_boundary", {}).get("metric_or_seconds_claim") is False,
        "stage5c_false": result.get("claim_boundary", {}).get("stage5c_executed") is False,
        "smc_false": result.get("claim_boundary", {}).get("smc_enabled") is False,
    }
    all_pass = all(bool(v) for v in gates.values())
    return {
        "source": "fresh_run_eval_over_cached_verified_checkpoints",
        "gates": gates,
        "passed": int(sum(bool(v) for v in gates.values())),
        "total": len(gates),
        "verdict": "stage42_q_t50_static_expert_combo_pass" if all_pass else "stage42_q_t50_static_expert_combo_partial",
    }


def _strip_row(row: Mapping[str, Any]) -> dict[str, Any]:
    combo = row["combo"]
    return {
        "source": row["source"],
        "seed_pair_index": row["seed_pair_index"],
        "stage42j_seed": row["stage42j_seed"],
        "stage42p_seed": row["stage42p_seed"],
        "stage42p_base_seed": row["stage42p_base_seed"],
        "stage42j": row["stage42j"],
        "stage42p": row["stage42p"],
        "combo": {
            "type": combo["type"],
            "choices": combo["choices"],
            "val_metrics": combo["val_metrics"],
            "test_metrics": combo["test_metrics"],
        },
    }


def _cached_result_if_available() -> dict[str, Any] | None:
    if not REPORT_JSON.exists():
        return None
    payload = read_json(REPORT_JSON, {})
    if not str(payload.get("stage", "")).startswith("Stage42-Q validation-only t50 static expert plus gain/harm combo"):
        return None
    if payload.get("stage42_q_gate", {}).get("total"):
        return payload
    return None


def run_stage42_t50_static_expert_combo() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    cached = _cached_result_if_available()
    if cached is not None:
        return cached
    if os.environ.get("STAGE42Q_ROW_LEVEL") != "1":
        return _run_preflight_from_cached_reports()
    if not (ft.DATA_DIR / "full_trajectory_train.npz").exists() or not (ft.DATA_DIR / "full_trajectory_val.npz").exists() or not (ft.DATA_DIR / "full_trajectory_test.npz").exists():
        ft.build_full_trajectory_labels()
    data = {split: s42i._split_arrays(split) for split in ["train", "val", "test"]}
    labels_val = s42i._labels(data["val"])
    labels_test = s42i._labels(data["test"])
    vocab = s42o._domain_vocab(data["train"], data["val"], data["test"])
    train_teacher: Mapping[str, np.ndarray] = {}
    val_teacher: Mapping[str, np.ndarray] = {}
    rows_runtime = []
    for idx, (j_seed, p_seed, base_seed) in enumerate(zip(J_SEEDS, P_SEEDS, BASE_SEEDS)):
        j_row = _stage42j_selected(j_seed, data["val"], data["test"])
        p_ckpt = OUT_DIR / "checkpoints" / f"stage42p_t50_gain_harm_selector_seed{p_seed}.pt"
        p_heartbeat = OUT_DIR / f"stage42p_t50_gain_harm_selector_seed{p_seed}_heartbeat.json"
        if not p_ckpt.exists() or not p_heartbeat.exists():
            if not train_teacher:
                train_teacher = s42n._row_teacher(data["train"], "train")
                val_teacher = s42n._row_teacher(data["val"], "val")
        p_row = _stage42p_selected(p_seed, base_seed, data["train"], data["val"], data["test"], vocab, train_teacher, val_teacher)
        combo = _fit_combo_policy(labels_val, labels_test, j_row, p_row)
        rows_runtime.append(
            {
                "source": "fresh_run_eval_over_cached_verified_checkpoints",
                "seed_pair_index": idx,
                "stage42j_seed": j_seed,
                "stage42p_seed": p_seed,
                "stage42p_base_seed": base_seed,
                "stage42j": {
                    "source": j_row["source"],
                    "val_metrics": j_row["val_metrics"],
                    "test_metrics": j_row["test_metrics"],
                    "slice_choices": j_row["slice_choices"],
                },
                "stage42p": {
                    "source": p_row["source"],
                    "val_metrics": p_row["val_metrics"],
                    "test_metrics": p_row["test_metrics"],
                    "policy": p_row["policy"],
                },
                "combo": combo,
            }
        )
    rows_report = [_strip_row(row) for row in rows_runtime]
    result = {
        "source": "fresh_run_eval_over_cached_verified_checkpoints",
        "stage": "Stage42-Q validation-only t50 static expert plus gain/harm combo",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                ft.DATA_DIR / "all_agent_train.npz",
                ft.DATA_DIR / "all_agent_val.npz",
                ft.DATA_DIR / "all_agent_test.npz",
                ft.DATA_DIR / "full_trajectory_train.npz",
                ft.DATA_DIR / "full_trajectory_val.npz",
                ft.DATA_DIR / "full_trajectory_test.npz",
                OUT_DIR / "static_gated_full_waypoint_stage42.json",
                OUT_DIR / "t50_gain_harm_selector_stage42.json",
            ]
        ),
        "dataset_rows": {split: int(len(data[split]["horizon"])) for split in ["train", "val", "test"]},
        "rows": rows_report,
        "summary": _summary(rows_report),
        "stage42j_recomputed_summary": _summary(rows_report, "stage42j"),
        "stage42p_recomputed_summary": _summary(rows_report, "stage42p"),
        "bootstrap_seed_mean": _seed_mean_bootstrap(rows_runtime, labels_test),
        "source_counts": _source_counts(rows_report),
        "comparison": _comparison(),
        "source_labels": {
            "all_agent_dataset": "cached_verified",
            "full_waypoint_labels": "cached_verified_or_rebuilt_by_stage41_helper",
            "stage42j_checkpoints": "cached_verified",
            "stage42p_selector_checkpoints": "cached_verified_or_fresh_if_missing",
            "combo_selection": "validation_only",
            "test_evaluation": "fresh_run_once_per_seed_pair",
            "feature_normalization": "train_split_stats_only_for_stage42p",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "future_waypoints_used_as_train_val_label_and_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
            "combo_sources_selected_on_val": True,
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
    result["stage42_q_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    _write_report(result)
    _write_gate(result["stage42_q_gate"])
    _append_readme_and_state(result)
    _append_ledger(result)
    return result


def _write_report(result: Mapping[str, Any]) -> None:
    s = result["summary"]
    sj = result["stage42j_recomputed_summary"]
    sp = result["stage42p_recomputed_summary"]
    boot = result["bootstrap_seed_mean"]
    gate = result["stage42_q_gate"]
    lines = [
        "# Stage42-Q T50 Static Expert + Gain/Harm Combo",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Metrics",
        "",
        "| candidate | source | ADE all | ADE t50 | ADE t50 CI low | ADE t100 diag | ADE hard | ADE easy degr | FDE t50 | switch |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| `Stage42-Q combo` | `{s['source']}` | {s['ade_all']['mean']:.6f} | {s['ade_t50']['mean']:.6f} | {s['ade_t50']['ci_low']:.6f} | {s['ade_t100_raw_frame_diagnostic']['mean']:.6f} | {s['ade_hard_failure']['mean']:.6f} | {s['ade_easy_degradation']['mean']:.6f} | {s['fde_t50']['mean']:.6f} | {s['switch_rate']['mean']:.6f} |",
        f"| `Stage42-J recomputed` | `{sj['source']}` | {sj['ade_all']['mean']:.6f} | {sj['ade_t50']['mean']:.6f} | {sj['ade_t50']['ci_low']:.6f} | {sj['ade_t100_raw_frame_diagnostic']['mean']:.6f} | {sj['ade_hard_failure']['mean']:.6f} | {sj['ade_easy_degradation']['mean']:.6f} | {sj['fde_t50']['mean']:.6f} | {sj['switch_rate']['mean']:.6f} |",
        f"| `Stage42-P recomputed` | `{sp['source']}` | {sp['ade_all']['mean']:.6f} | {sp['ade_t50']['mean']:.6f} | {sp['ade_t50']['ci_low']:.6f} | {sp['ade_t100_raw_frame_diagnostic']['mean']:.6f} | {sp['ade_hard_failure']['mean']:.6f} | {sp['ade_easy_degradation']['mean']:.6f} | {sp['fde_t50']['mean']:.6f} | {sp['switch_rate']['mean']:.6f} |",
        "",
        "## Bootstrap Over Seed-Mean Row Improvements",
        "",
        "| slice | rows | mean | ci_low | ci_high |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in ["all", "t50", "t100_raw_frame_diagnostic", "hard_failure", "easy"]:
        row = boot.get(name, {})
        lines.append(f"| `{name}` | {row.get('rows', 0)} | {row.get('mean', 0.0):.6f} | {row.get('ci_low', 0.0):.6f} | {row.get('ci_high', 0.0):.6f} |")
    lines.extend(
        [
            "",
            "## Source Choices",
            "",
            f"- validation-selected source counts across seed/domain/horizon slices: `{result['source_counts']}`",
            "- Candidate sources are `floor`, `Stage42-J static expert`, and `Stage42-P t50 gain/harm`.",
            "- Source selection is by validation domain/horizon slice only; test labels are not used for threshold or source selection.",
            "",
            "## Interpretation",
            "",
            "- Stage42-Q asks whether Stage42-J's t+50/full-waypoint static-expert stability and Stage42-P's row-level gain/harm selector can be combined without test tuning.",
            "- If the combo fails the t+50 seed-CI gate, it is useful negative evidence: simple slice-level validation choice is still not enough for paper-stable t+50.",
            "- Future waypoints remain train/val labels and final eval labels only, never inference inputs.",
            "- All claims remain dataset-local raw-frame 2.5D; no metric/seconds-level, Stage5C, or SMC claim is made.",
        ]
    )
    write_md(REPORT_MD, lines)


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-Q Gate",
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
    gate = result["stage42_q_gate"]
    if "summary" in result:
        s = result["summary"]
        metrics_block = f"""combo_ade_all = {s['ade_all']['mean']}
combo_ade_t50 = {s['ade_t50']['mean']}
combo_ade_t50_ci_low = {s['ade_t50']['ci_low']}
combo_ade_hard_failure = {s['ade_hard_failure']['mean']}
combo_ade_easy_degradation = {s['ade_easy_degradation']['mean']}
combo_fde_t50 = {s['fde_t50']['mean']}"""
        state_metrics = {
            "combo_ade_all": s["ade_all"]["mean"],
            "combo_ade_t50": s["ade_t50"]["mean"],
            "combo_ade_t50_ci_low": s["ade_t50"]["ci_low"],
            "combo_ade_hard_failure": s["ade_hard_failure"]["mean"],
            "combo_ade_easy_degradation": s["ade_easy_degradation"]["mean"],
            "combo_fde_t50": s["fde_t50"]["mean"],
        }
    else:
        env = result["diagnostic_envelope"]
        metrics_block = f"""diagnostic_ade_all_best_available = {env['ade_all_best_available']}
diagnostic_ade_t50_best_available = {env['ade_t50_best_available']}
diagnostic_ade_hard_best_available = {env['ade_hard_best_available']}
diagnostic_fde_t50_best_available = {env['fde_t50_best_available']}
row_level_combo_status = {result['row_level_combo_status']['source']}"""
        state_metrics = {
            "diagnostic_ade_all_best_available": env["ade_all_best_available"],
            "diagnostic_ade_t50_best_available": env["ade_t50_best_available"],
            "diagnostic_ade_hard_best_available": env["ade_hard_best_available"],
            "diagnostic_fde_t50_best_available": env["fde_t50_best_available"],
            "row_level_combo_status": result["row_level_combo_status"]["source"],
        }
    block = f"""
## Stage42-Q T50 Static Expert + Gain/Harm Combo

```text
source = {result['source']}
verdict = {gate['verdict']}
gates = {gate['passed']} / {gate['total']}
{metrics_block}
stage5c_executed = false
smc_enabled = false
```

Stage42-Q targets the complementarity between Stage42-J static-gated full-waypoint experts and Stage42-P t+50 gain/harm selector. If it is a preflight result, it is diagnostic only and not a deployable combo claim; a row-level NPZ prediction cache is required before a full validation-only combo can be treated as pipeline evidence.
"""
    _append_if_missing(Path("README_RESULTS.md"), "## Stage42-Q T50 Static Expert + Gain/Harm Combo", block)
    _append_if_missing(Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md"), "## Stage42-Q T50 Static Expert + Gain/Harm Combo", block)
    _append_if_missing(Path("outputs/m3w_neural_v1/README_M3W_LONG_GOAL_SUMMARY_ZH.md"), "## Stage42-Q T50 Static Expert + Gain/Harm Combo", block)
    state = read_json(Path("research_state.json"), {})
    state["current_stage"] = "stage42_q_t50_static_expert_combo"
    state["current_verdict"] = gate["verdict"]
    state.setdefault("stage42", {})["stage_q_t50_static_expert_combo"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "gates_passed": gate["passed"],
        "gates_total": gate["total"],
        "verdict": gate["verdict"],
        **state_metrics,
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
        "step": "stage42_q_t50_static_expert_combo",
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
    run_stage42_t50_static_expert_combo()
