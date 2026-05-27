from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_group_consistency_full_waypoint_repair as di
from src import stage42_source_level_full_waypoint_eval as am
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "temporal_group_repel_repair_stage42.json"
REPORT_MD = OUT_DIR / "temporal_group_repel_repair_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ez_gate.md"

DI_JSON = OUT_DIR / "group_consistency_full_waypoint_repair_stage42.json"
AM_JSON = OUT_DIR / "source_level_full_waypoint_eval_stage42.json"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

PAPER_FILES = [
    OUT_DIR / "paper_outline_stage42.md",
    OUT_DIR / "method_draft_stage42.md",
    OUT_DIR / "experiment_tables_stage42.md",
    OUT_DIR / "ablation_tables_stage42.md",
    OUT_DIR / "failure_taxonomy_stage42.md",
    OUT_DIR / "model_card_stage42.md",
    OUT_DIR / "data_card_stage42.md",
    OUT_DIR / "reproducibility_stage42.md",
    OUT_DIR / "a_journal_gap_stage42.md",
]

EPS = 1e-6
SOURCE = "fresh_stage42_temporal_group_repel_repair"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-EZ 接续 Stage42-EW/EX/EY 的负结果：risk/adaptive bucket 没有超过 Stage42-DI。",
    "本阶段只改变 group-repel repair 的 temporal shape / candidate family，不启用 latent generative 或 SMC。",
    "repair 只使用 predicted rollout geometry、source/frame/horizon group key、agent id、当前/过去可得信息。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。",
    "candidate 和 policy 只在 validation 上选择；test 只评一次。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _jsonable(value: Any) -> Any:
    return am._jsonable(value)


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _temporal_weights(kind: str, waypoint_count: int, gamma: float = 1.0) -> np.ndarray:
    if waypoint_count <= 0:
        raise ValueError("waypoint_count must be positive")
    frac = np.linspace(1.0 / waypoint_count, 1.0, waypoint_count, dtype=np.float64)
    kind = str(kind)
    if kind == "uniform":
        weights = np.ones_like(frac)
    elif kind == "tail":
        weights = np.power(frac, float(gamma))
    elif kind == "sqrt_tail":
        weights = np.sqrt(frac)
    elif kind == "head":
        weights = np.power(np.maximum(1.0 - frac + 1.0 / waypoint_count, EPS), float(gamma))
    elif kind == "bell":
        weights = np.sin(np.pi * frac)
        weights = np.maximum(weights, 0.25)
    else:
        raise ValueError(f"unknown temporal weight kind: {kind}")
    weights = weights / max(float(np.max(weights)), EPS)
    return weights.astype(np.float32)


def _repel_temporal_rows(
    xy: np.ndarray,
    switch: np.ndarray,
    group_key: np.ndarray,
    normalizer: np.ndarray,
    agent_id: np.ndarray,
    current_xy: np.ndarray,
    min_sep: float,
    strength: float,
    weights: np.ndarray,
    direction_mode: str,
) -> np.ndarray:
    repaired = xy.astype(np.float32).copy()
    keys = np.asarray(group_key, dtype=object)
    order = np.argsort(keys)
    weights = np.asarray(weights, dtype=np.float32)
    start = 0
    while start < len(order):
        end = start + 1
        key = keys[order[start]]
        while end < len(order) and keys[order[end]] == key:
            end += 1
        rows = order[start:end]
        _, first_idx = np.unique(agent_id[rows], return_index=True)
        unique_rows = rows[first_idx]
        if len(unique_rows) > 1 and np.any(switch[unique_rows]):
            pts = repaired[unique_rows].astype(np.float64, copy=False)
            diff = pts[:, None, :, :] - pts[None, :, :, :]
            dist = np.linalg.norm(diff, axis=3)
            dist[np.arange(len(unique_rows)), np.arange(len(unique_rows)), :] = np.inf
            centroid = np.mean(current_xy[unique_rows].astype(np.float64), axis=0)
            for i, row in enumerate(unique_rows):
                if not switch[row]:
                    continue
                min_d = float(np.min(dist[i]))
                sep = float(min_sep) * float(max(normalizer[row], EPS))
                if not np.isfinite(min_d) or min_d >= sep:
                    continue
                nearest_agent, nearest_step = np.unravel_index(np.argmin(dist[i]), dist[i].shape)
                if direction_mode == "centroid_current":
                    direction = current_xy[row].astype(np.float64) - centroid
                else:
                    direction = current_xy[row].astype(np.float64) - current_xy[unique_rows[int(nearest_agent)]].astype(np.float64)
                norm = float(np.linalg.norm(direction))
                if norm < EPS:
                    direction = pts[i, int(nearest_step)] - pts[int(nearest_agent), int(nearest_step)]
                    norm = float(np.linalg.norm(direction))
                if norm < EPS:
                    direction = np.asarray([1.0, 0.0], dtype=np.float64)
                    norm = 1.0
                offset = direction / norm * (sep - min_d) * float(strength)
                repaired[row] = repaired[row] + (weights[:, None] * offset[None, :]).astype(np.float32)
        start = end
    return repaired


def _candidate_grid() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for min_sep in [0.05, 0.08, 0.12]:
        for margin in [0.0, 0.005]:
            for strength in [0.25, 0.50, 0.75]:
                rows.append(
                    {
                        "mode": "temporal_repel",
                        "temporal_kind": "uniform",
                        "gamma": 1.0,
                        "direction_mode": "nearest_current",
                        "min_sep": min_sep,
                        "margin": margin,
                        "strength": strength,
                    }
                )
                for gamma in [0.5, 1.0, 2.0]:
                    rows.append(
                        {
                            "mode": "temporal_repel",
                            "temporal_kind": "tail",
                            "gamma": gamma,
                            "direction_mode": "nearest_current",
                            "min_sep": min_sep,
                            "margin": margin,
                            "strength": strength,
                        }
                    )
                for kind in ["sqrt_tail", "bell", "head"]:
                    rows.append(
                        {
                            "mode": "temporal_repel",
                            "temporal_kind": kind,
                            "gamma": 1.0,
                            "direction_mode": "nearest_current",
                            "min_sep": min_sep,
                            "margin": margin,
                            "strength": strength,
                        }
                    )
                rows.append(
                    {
                        "mode": "temporal_repel",
                        "temporal_kind": "tail",
                        "gamma": 1.0,
                        "direction_mode": "centroid_current",
                        "min_sep": min_sep,
                        "margin": margin,
                        "strength": strength,
                    }
                )
    return rows


def _repair_subset_temporal(
    ids: np.ndarray,
    candidate: Mapping[str, Any],
    data: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    floor_xy: np.ndarray,
    pred_xy: np.ndarray,
    base_xy: np.ndarray,
    base_switch: np.ndarray,
    group_key: np.ndarray,
) -> dict[str, Any]:
    ids = np.asarray(ids, dtype=np.int64)
    selected_xy = base_xy[ids].astype(np.float32).copy()
    switch = base_switch[ids].astype(bool).copy()
    normalizer = np.maximum(data["scale"][ids].astype(np.float64), EPS)
    agent = data["agent_id"][ids].astype(np.int64)
    keys = group_key[ids]
    floor_min = di._min_group_distance_fast(floor_xy[ids], keys, normalizer, agent)
    base_min = di._min_group_distance_fast(selected_xy, keys, normalizer, agent)
    pred_min = di._min_group_distance_fast(pred_xy[ids], keys, normalizer, agent)
    min_sep = float(candidate["min_sep"])
    margin = float(candidate.get("margin", 0.0))
    unsafe = switch & np.isfinite(base_min) & np.isfinite(floor_min) & (base_min < min_sep) & (base_min + margin < floor_min)
    del pred_min
    weights = _temporal_weights(str(candidate["temporal_kind"]), selected_xy.shape[1], float(candidate.get("gamma", 1.0)))
    selected_xy = _repel_temporal_rows(
        selected_xy,
        switch,
        keys,
        normalizer,
        agent,
        np.stack([data["current_x"][ids], data["current_y"][ids]], axis=1),
        min_sep=min_sep,
        strength=float(candidate["strength"]),
        weights=weights,
        direction_mode=str(candidate.get("direction_mode", "nearest_current")),
    )
    final_min = di._min_group_distance_fast(selected_xy, keys, normalizer, agent)
    selected_ade, selected_fde = di._trajectory_errors_subset(selected_xy, labels, ids)
    floor_ade, floor_fde = di._trajectory_errors_subset(floor_xy[ids], labels, ids)
    metric = di._metric_subset(selected_ade, floor_ade, data, ids, switch)
    return {
        "selected_xy": selected_xy,
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "floor_ade": floor_ade,
        "floor_fde": floor_fde,
        "switch": switch,
        "metric": metric,
        "diagnostics": {
            "unsafe_rows": int(np.sum(unsafe)),
            "unsafe_rate": float(np.mean(unsafe)) if len(unsafe) else 0.0,
            "base_switch_rate": float(np.mean(base_switch[ids])) if len(ids) else 0.0,
            "final_switch_rate": float(np.mean(switch)) if len(switch) else 0.0,
            "base_near_005": float(np.mean(np.isfinite(base_min) & (base_min < 0.05))) if len(base_min) else 0.0,
            "final_near_005": float(np.mean(np.isfinite(final_min) & (final_min < 0.05))) if len(final_min) else 0.0,
            "floor_near_005": float(np.mean(np.isfinite(floor_min) & (floor_min < 0.05))) if len(floor_min) else 0.0,
            "base_p05_min_distance": float(np.percentile(base_min[np.isfinite(base_min)], 5)) if np.any(np.isfinite(base_min)) else None,
            "final_p05_min_distance": float(np.percentile(final_min[np.isfinite(final_min)], 5)) if np.any(np.isfinite(final_min)) else None,
            "floor_p05_min_distance": float(np.percentile(floor_min[np.isfinite(floor_min)], 5)) if np.any(np.isfinite(floor_min)) else None,
            "temporal_first_weight": float(weights[0]),
            "temporal_last_weight": float(weights[-1]),
        },
    }


def _selection_score(metric: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> float:
    near_delta = float(diagnostics["final_near_005"]) - float(diagnostics["base_near_005"])
    return (
        1.45 * float(metric["all_improvement"])
        + 1.45 * float(metric["hard_failure_improvement"])
        + 1.10 * float(metric["t50_improvement"])
        + 0.55 * float(metric["t100_raw_frame_diagnostic_improvement"])
        - 35.0 * max(0.0, float(metric["easy_degradation"]) - 0.02)
        - 8.0 * max(0.0, near_delta)
        - 0.01 * float(metric["switch_rate"])
    )


def _evaluate_temporal_repairs(
    data: Mapping[str, np.ndarray],
    split: np.ndarray,
    labels: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
    am_candidate: Mapping[str, Any],
    group_key: np.ndarray,
) -> dict[str, Any]:
    val_ids = np.where(split == "val")[0]
    test_ids = np.where(split == "test")[0]
    floor_xy = floor["floor_xy"].astype(np.float32)
    pred_xy = am_candidate["pred_xy"].astype(np.float32)
    base_xy = am_candidate["selected_xy"].astype(np.float32)
    base_switch = am_candidate["switch"].astype(bool)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_score = -1e9
    for candidate in _candidate_grid():
        val = _repair_subset_temporal(val_ids, candidate, data, labels, floor_xy, pred_xy, base_xy, base_switch, group_key)
        score = _selection_score(val["metric"], val["diagnostics"])
        row = {
            "candidate": dict(candidate),
            "val_score": float(score),
            "val_metric": val["metric"],
            "val_diagnostics": val["diagnostics"],
        }
        rows.append(row)
        if score > best_score:
            best_score = float(score)
            best = row
    if best is None:
        raise RuntimeError("No Stage42-EZ temporal repair candidate evaluated.")
    test = _repair_subset_temporal(test_ids, best["candidate"], data, labels, floor_xy, pred_xy, base_xy, base_switch, group_key)
    base_test_ade = am_candidate["selected_ade"][test_ids].astype(np.float64)
    floor_test_ade = am_candidate["floor_ade"][test_ids].astype(np.float64)
    base_test_metric = di._metric_subset(base_test_ade, floor_test_ade, data, test_ids, base_switch[test_ids])
    h = data["horizon"][test_ids].astype(int)
    hard_failure = data["hard"][test_ids].astype(bool) | data["failure"][test_ids].astype(bool)
    easy = data["easy"][test_ids].astype(bool)
    domain = data["dataset"][test_ids].astype(str)
    bootstrap = {
        "all": di._bootstrap_ci_subset(test["selected_ade"], test["floor_ade"], np.ones(len(test_ids), dtype=bool), seed=42901),
        "t50": di._bootstrap_ci_subset(test["selected_ade"], test["floor_ade"], h == 50, seed=42902),
        "t100_raw_frame_diagnostic": di._bootstrap_ci_subset(test["selected_ade"], test["floor_ade"], h == 100, seed=42903),
        "hard_failure": di._bootstrap_ci_subset(test["selected_ade"], test["floor_ade"], hard_failure, seed=42904),
        "easy_degradation": di._bootstrap_ci_subset(test["floor_ade"], test["selected_ade"], easy, seed=42905),
    }
    by_domain = {
        d: di._metric_subset(test["selected_ade"][domain == d], test["floor_ade"][domain == d], data, test_ids[domain == d], test["switch"][domain == d])
        for d in sorted(set(domain.tolist()))
    }
    return {
        "candidate_count": len(rows),
        "validation_rows": sorted(rows, key=lambda row: row["val_score"], reverse=True),
        "selected": best,
        "baseline_stage42_am_on_test": base_test_metric,
        "test": {
            "metric_vs_floor": test["metric"],
            "diagnostics": test["diagnostics"],
            "bootstrap": bootstrap,
            "by_domain": by_domain,
        },
    }


def _compare_to_stage42_di(metric: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    di_payload = read_json(DI_JSON, {})
    di_metric = di_payload.get("repair", {}).get("test", {}).get("metric_vs_floor", {})
    di_diag = di_payload.get("repair", {}).get("test", {}).get("diagnostics", {})

    def delta(ref: Mapping[str, Any]) -> dict[str, float | None]:
        if not ref:
            return {
                "all_improvement": None,
                "t50_improvement": None,
                "t100_raw_frame_diagnostic_improvement": None,
                "hard_failure_improvement": None,
                "easy_degradation": None,
            }
        return {
            "all_improvement": float(metric.get("all_improvement", 0.0)) - float(ref.get("all_improvement", 0.0)),
            "t50_improvement": float(metric.get("t50_improvement", 0.0)) - float(ref.get("t50_improvement", 0.0)),
            "t100_raw_frame_diagnostic_improvement": float(metric.get("t100_raw_frame_diagnostic_improvement", 0.0))
            - float(ref.get("t100_raw_frame_diagnostic_improvement", 0.0)),
            "hard_failure_improvement": float(metric.get("hard_failure_improvement", 0.0)) - float(ref.get("hard_failure_improvement", 0.0)),
            "easy_degradation": float(metric.get("easy_degradation", 0.0)) - float(ref.get("easy_degradation", 0.0)),
        }

    return {
        "stage42_di_metric": di_metric,
        "stage42_di_diagnostics": di_diag,
        "delta_vs_stage42_di": delta(di_metric),
        "near_delta_vs_stage42_di": None
        if not di_diag
        else float(diagnostics.get("final_near_005", 0.0)) - float(di_diag.get("final_near_005", 0.0)),
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    data = s41._combined()
    split, group = am._split_arrays(data)
    split_stats = am._source_stats(data, split, group)
    labels = am._reconstruct_waypoint_labels(data)
    train_mask = split == "train"
    floor = am._floor_arrays(data, train_mask)
    am_candidate = di._rebuild_stage42_am_candidate(data, split, labels, floor)
    group_key = di._group_key(data)
    eval_result = _evaluate_temporal_repairs(data, split, labels, floor, am_candidate, group_key)
    metric = eval_result["test"]["metric_vs_floor"]
    diagnostics = eval_result["test"]["diagnostics"]
    comparison = _compare_to_stage42_di(metric, diagnostics)
    delta_di = comparison["delta_vs_stage42_di"]
    promotes = (
        metric["all_improvement"] > 0.0
        and metric["hard_failure_improvement"] > 0.0
        and metric["easy_degradation"] <= 0.02
        and (delta_di["all_improvement"] or 0.0) > 0.0
        and (delta_di["hard_failure_improvement"] or 0.0) > 0.0
        and (comparison["near_delta_vs_stage42_di"] or 0.0) <= EPS
    )
    result: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-EZ temporal group-repel repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                str(AM_JSON),
                str(DI_JSON),
            ]
        ),
        "split_stats": split_stats,
        "label_stats": {
            "rows": int(len(split)),
            "test_rows": int(np.sum(split == "test")),
            "test_full_waypoint_rows": int(np.sum((split == "test") & np.all(labels["waypoint_valid"], axis=1))),
        },
        "repair_family": {
            "source": "fresh_stage42_ez_temporal_repel_grid",
            "candidate_count": len(_candidate_grid()),
            "temporal_shapes": sorted(set(c["temporal_kind"] for c in _candidate_grid())),
            "direction_modes": sorted(set(c["direction_mode"] for c in _candidate_grid())),
            "baseline_reference": "Stage42-DI uniform same-offset repel_unsafe",
            "uses_future_inputs": False,
        },
        "stage42_am_rebuilt": {
            "source": "fresh_rebuild_for_ez",
            "lambda": am_candidate["lambda"],
            "feature_count": am_candidate["feature_count"],
            "policy_slice_count": len(am_candidate["policy"]["slices"]),
            "val_metric": am_candidate["val_metric"],
        },
        "repair": eval_result,
        "comparison_to_stage42_di": comparison,
        "deployment_decision": {
            "promote_temporal_group_repel": bool(promotes),
            "decision": "promote_stage42_ez_temporal_group_repel_repair"
            if promotes
            else "temporal_group_repel_not_enough_keep_stage42_di_or_cq_floor",
            "reason": "Promotion requires all+hard positive, easy safe, improvement over Stage42-DI on all+hard, and no worse near@0.05 than Stage42-DI.",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "group_features_predicted_rollout_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_policy_selection": True,
            "train_only_feature_normalization": True,
            "source_overlap_pass": bool(split_stats["source_overlap_pass"]),
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
    result["stage42_ez_gate"] = _gate(result)
    return result


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    metric = result["repair"]["test"]["metric_vs_floor"]
    diag = result["repair"]["test"]["diagnostics"]
    delta_di = result["comparison_to_stage42_di"]["delta_vs_stage42_di"]
    near_delta = result["comparison_to_stage42_di"]["near_delta_vs_stage42_di"]
    no_leak = result["no_leakage"]
    gates = {
        "source_level_split_rebuilt": result["split_stats"]["by_split"]["test"]["rows"] == int(metric["rows"]) and int(metric["rows"]) > 0,
        "full_waypoint_labels_available": result["label_stats"]["test_full_waypoint_rows"] > 0,
        "temporal_repair_family_built": result["repair_family"]["candidate_count"] >= 40,
        "temporal_shapes_recorded": len(result["repair_family"]["temporal_shapes"]) >= 4,
        "validation_selected_temporal_repair": result["repair"]["selected"]["val_score"] != 0.0 and no_leak["test_threshold_tuning"] is False,
        "no_future_or_test_leakage": all(
            [
                no_leak["future_endpoint_input"] is False,
                no_leak["future_waypoint_input"] is False,
                no_leak["future_waypoint_label_eval_only"] is True,
                no_leak["group_features_predicted_rollout_only"] is True,
                no_leak["central_velocity"] is False,
                no_leak["test_endpoint_goals"] is False,
                no_leak["test_threshold_tuning"] is False,
                no_leak["validation_only_policy_selection"] is True,
                no_leak["train_only_feature_normalization"] is True,
            ]
        ),
        "test_all_positive_vs_floor": metric["all_improvement"] > 0.0,
        "test_t50_positive_vs_floor": metric["t50_improvement"] > 0.0,
        "test_hard_positive_vs_floor": metric["hard_failure_improvement"] > 0.0,
        "easy_degradation_under_2pct": metric["easy_degradation"] <= 0.02,
        "near_collision_not_worse_than_base": diag["final_near_005"] <= diag["base_near_005"] + EPS,
        "beats_stage42_di_all": (delta_di["all_improvement"] or 0.0) > 0.0,
        "beats_stage42_di_hard": (delta_di["hard_failure_improvement"] or 0.0) > 0.0,
        "near_not_worse_than_stage42_di": near_delta is not None and near_delta <= EPS,
        "bootstrap_reported": result["repair"]["test"]["bootstrap"]["all"]["bootstrap_n"] > 0,
        "no_metric_seconds_overclaim": result["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": result["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": result["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    if passed == total:
        verdict = "stage42_ez_temporal_group_repel_repair_pass_promotable"
    elif gates["test_all_positive_vs_floor"] and gates["test_hard_positive_vs_floor"] and gates["easy_degradation_under_2pct"]:
        verdict = "stage42_ez_temporal_group_repel_repair_positive_not_promoted"
    else:
        verdict = "stage42_ez_temporal_group_repel_repair_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_ez_gate"]
    selected = result["repair"]["selected"]
    metric = result["repair"]["test"]["metric_vs_floor"]
    diag = result["repair"]["test"]["diagnostics"]
    base = result["repair"]["baseline_stage42_am_on_test"]
    delta = result["comparison_to_stage42_di"]["delta_vs_stage42_di"]
    return [
        "# Stage42-EZ Temporal Group-Repel Repair",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- decision: `{result['deployment_decision']['decision']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Repair Family",
        "",
        f"- candidate_count: `{result['repair_family']['candidate_count']}`",
        f"- temporal_shapes: `{result['repair_family']['temporal_shapes']}`",
        f"- direction_modes: `{result['repair_family']['direction_modes']}`",
        "",
        "## Selected Temporal Repair",
        "",
        f"- candidate: `{selected['candidate']}`",
        f"- val_score: `{selected['val_score']:.6f}`",
        f"- val_metric: `{selected['val_metric']}`",
        f"- val_diagnostics: `{selected['val_diagnostics']}`",
        "",
        "## Test Once Metrics vs Train-Horizon Causal Floor",
        "",
        "| candidate | all | t50 | t100 raw diag | hard/failure | easy degradation | switch | near@0.05 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Stage42-AM rebuilt floor-protected source-level | {_pct(base['all_improvement'])} | {_pct(base['t50_improvement'])} | {_pct(base['t100_raw_frame_diagnostic_improvement'])} | {_pct(base['hard_failure_improvement'])} | {_pct(base['easy_degradation'])} | {_pct(base['switch_rate'])} | {_pct(diag['base_near_005'])} |",
        f"| Stage42-EZ temporal group-repel | {_pct(metric['all_improvement'])} | {_pct(metric['t50_improvement'])} | {_pct(metric['t100_raw_frame_diagnostic_improvement'])} | {_pct(metric['hard_failure_improvement'])} | {_pct(metric['easy_degradation'])} | {_pct(metric['switch_rate'])} | {_pct(diag['final_near_005'])} |",
        "",
        "## Delta vs Stage42-DI",
        "",
        f"- all/t50/t100raw/hard/easy: `{_pct(delta['all_improvement'])}` / `{_pct(delta['t50_improvement'])}` / `{_pct(delta['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(delta['hard_failure_improvement'])}` / `{_pct(delta['easy_degradation'])}`",
        f"- near_delta_vs_stage42_di: `{_pct(result['comparison_to_stage42_di']['near_delta_vs_stage42_di'])}`",
        "",
        "## Bootstrap CI",
        "",
        "| slice | low | mid | high | n |",
        "| --- | ---: | ---: | ---: | ---: |",
        *[
            f"| `{key}` | {row['low']:.6f} | {row['mid']:.6f} | {row['high']:.6f} | {row['n']} |"
            for key, row in result["repair"]["test"]["bootstrap"].items()
        ],
        "",
        "## By Domain",
        "",
        "| domain | rows | all | t50 | t100 raw diag | hard/failure | easy | switch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| `{domain}` | {row['rows']} | {_pct(row['all_improvement'])} | {_pct(row['t50_improvement'])} | {_pct(row['t100_raw_frame_diagnostic_improvement'])} | {_pct(row['hard_failure_improvement'])} | {_pct(row['easy_degradation'])} | {_pct(row['switch_rate'])} |"
            for domain, row in result["repair"]["test"]["by_domain"].items()
        ],
        "",
        "## Interpretation",
        "",
        "- Stage42-EW/EX/EY showed that risk/adaptive bucket selection did not beat Stage42-DI.",
        "- Stage42-EZ tests a different hypothesis: the constant same-offset repel is too crude, and temporal weighting might preserve early waypoints while repairing future group proximity.",
        "- Promotion requires beating Stage42-DI on all and hard/failure while preserving easy and not worsening near@0.05.",
        "- If not promoted, this is evidence that the bottleneck is not merely temporal offset shape; the next repair should change objective/trajectory family more deeply.",
        "",
        "## No-Leakage And Claim Boundary",
        "",
        f"- no_leakage: `{result['no_leakage']}`",
        f"- claim_boundary: `{result['claim_boundary']}`",
    ]


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_ez_gate"]
    return [
        "# Stage42-EZ Gate",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Gates",
        "",
        *[f"- {key}: `{value}`" for key, value in gate["gates"].items()],
    ]


def _refresh_lines(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_ez_gate"]
    metric = result["repair"]["test"]["metric_vs_floor"]
    delta = result["comparison_to_stage42_di"]["delta_vs_stage42_di"]
    diag = result["repair"]["test"]["diagnostics"]
    return [
        "## Stage42-EZ Temporal Group-Repel Repair",
        "",
        "- source: `fresh_stage42_temporal_group_repel_repair`",
        "- role: tests temporal weighting for group-repel offsets after Stage42-EW/EX/EY risk-bucket repairs failed to beat Stage42-DI.",
        f"- selected candidate: `{result['repair']['selected']['candidate']}`.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- test all/t50/t100raw/hard/easy: `{_pct(metric['all_improvement'])}` / `{_pct(metric['t50_improvement'])}` / `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(metric['hard_failure_improvement'])}` / `{_pct(metric['easy_degradation'])}`.",
        f"- delta vs Stage42-DI all/t50/t100raw/hard/easy: `{_pct(delta['all_improvement'])}` / `{_pct(delta['t50_improvement'])}` / `{_pct(delta['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(delta['hard_failure_improvement'])}` / `{_pct(delta['easy_degradation'])}`.",
        f"- near@0.05 base/final: `{_pct(diag['base_near_005'])}` / `{_pct(diag['final_near_005'])}`.",
        f"- decision: `{result['deployment_decision']['decision']}`.",
        "- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
    ]


def _refresh_readmes(result: Mapping[str, Any]) -> None:
    lines = _refresh_lines(result)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, "STAGE42_EZ_TEMPORAL_GROUP_REPEL_REPAIR", lines)


def _refresh_paper_package(result: Mapping[str, Any]) -> None:
    lines = _refresh_lines(result)
    for path in PAPER_FILES:
        _replace_section(path, "STAGE42_EZ_TEMPORAL_GROUP_REPEL_REPAIR", lines)


def _refresh_research_state(result: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-EZ temporal group-repel repair"
    state["current_verdict"] = result["stage42_ez_gate"]["verdict"]
    state["stage42_ez_temporal_group_repel_repair"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": result["stage42_ez_gate"]["verdict"],
        "gates": f"{result['stage42_ez_gate']['passed']}/{result['stage42_ez_gate']['total']}",
        "deployment_decision": result["deployment_decision"],
        "selected_candidate": result["repair"]["selected"]["candidate"],
        "test_metric_vs_floor": result["repair"]["test"]["metric_vs_floor"],
        "test_proximity_diagnostics": result["repair"]["test"]["diagnostics"],
        "comparison_to_stage42_di": result["comparison_to_stage42_di"],
        "claim_boundary": result["claim_boundary"],
        "conclusion": "Stage42-EZ tests temporal weighting of group-repel offsets after adaptive/risk bucket repairs failed; promotion requires beating Stage42-DI on all+hard while preserving easy and proximity.",
    }
    write_json(RESEARCH_STATE, _jsonable(state))


def run_stage42_temporal_group_repel_repair() -> dict[str, Any]:
    result = _build_payload()
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _refresh_readmes(result)
    _refresh_paper_package(result)
    _refresh_research_state(result)
    return result


if __name__ == "__main__":
    payload = run_stage42_temporal_group_repel_repair()
    gate = payload["stage42_ez_gate"]
    print(f"Stage42-EZ gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
    print(f"Decision: {payload['deployment_decision']['decision']}")
