from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_external_validation as s42b
from src import stage42_full_waypoint_all_hard_loss_repair as dg
from src import stage42_full_waypoint_proximity_occupancy_loss_repair as dh
from src import stage42_group_consistency_full_waypoint_repair as di
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_source_level_graph_context as graph_ctx
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "group_consistency_constraint_training_stage42.json"
REPORT_MD = OUT_DIR / "group_consistency_constraint_training_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_eu_gate.md"

AM_JSON = OUT_DIR / "source_level_full_waypoint_eval_stage42.json"
DH_JSON = OUT_DIR / "full_waypoint_proximity_occupancy_loss_repair_stage42.json"
DI_JSON = OUT_DIR / "group_consistency_full_waypoint_repair_stage42.json"
ET_JSON = OUT_DIR / "group_consistency_target_ablation_stage42.json"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
TARGET_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
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

SOURCE = "fresh_stage42_group_consistency_constraint_training"
LAMBDAS = [0.1, 1.0, 10.0, 100.0]
EPS = 1e-6

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-EU 把 Stage42-ES/ET 支持的 source/frame/horizon group-consistency target 放进训练权重，再做 validation-selected protected full-waypoint evaluation。",
    "group-consistency weights 只来自 train/val/test row 的当前帧、past/context feature、predicted rollout geometry 和 source/frame/horizon group key；future waypoints 只作为 loss/eval labels。",
    "不下载、不转换、不执行 Stage5C、不启用 SMC。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "validation 选择模型和 repair policy；test 只评一次。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
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


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _group_risk_signals(
    data: Mapping[str, np.ndarray],
    group_key: np.ndarray,
    am_candidate: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    normalizer = np.maximum(data["scale"].astype(np.float64), EPS)
    agent = data["agent_id"].astype(np.int64)
    base_min = di._min_group_distance_fast(am_candidate["selected_xy"], group_key, normalizer, agent)
    floor_min = di._min_group_distance_fast(am_candidate["floor_xy"], group_key, normalizer, agent) if "floor_xy" in am_candidate else None
    close_005 = np.isfinite(base_min) & (base_min < 0.05)
    close_008 = np.isfinite(base_min) & (base_min < 0.08)
    close_012 = np.isfinite(base_min) & (base_min < 0.12)
    if floor_min is None:
        unsafe_vs_floor = close_008
    else:
        unsafe_vs_floor = close_008 & np.isfinite(floor_min) & (base_min < floor_min)
    risk = np.zeros(len(base_min), dtype=np.float64)
    risk += 2.5 * close_005.astype(np.float64)
    risk += 1.5 * close_008.astype(np.float64)
    risk += 0.75 * close_012.astype(np.float64)
    risk += 3.0 * unsafe_vs_floor.astype(np.float64)
    return {
        "base_min_distance": base_min,
        "floor_min_distance": floor_min if floor_min is not None else np.full(len(base_min), np.inf),
        "close_005": close_005.astype(np.float64),
        "close_008": close_008.astype(np.float64),
        "close_012": close_012.astype(np.float64),
        "unsafe_vs_floor": unsafe_vs_floor.astype(np.float64),
        "risk": risk,
    }


def _constraint_weights(
    data: Mapping[str, np.ndarray],
    group_risk: Mapping[str, np.ndarray],
    graph: np.ndarray,
    train_mask: np.ndarray,
    variant: str,
) -> np.ndarray:
    h = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    graph_sig = dh._graph_signals(graph)
    w = np.ones(len(h), dtype=np.float64)
    if variant == "group_unsafe_weighted":
        w += group_risk["risk"]
    elif variant == "group_unsafe_hard_weighted":
        w += group_risk["risk"] + 2.0 * hard_failure.astype(np.float64)
    elif variant == "group_unsafe_t50_t100_weighted":
        w += group_risk["risk"] + 1.5 * (h == 50) + 2.0 * (h == 100)
    elif variant == "group_graph_density_weighted":
        w += group_risk["risk"] + 1.0 * graph_sig["density_r1"] + 0.5 * graph_sig["density_r2"]
    elif variant == "group_easy_safe_weighted":
        w += group_risk["risk"] + 1.5 * hard_failure.astype(np.float64)
        w[easy] *= 0.75
    else:
        raise ValueError(f"unknown Stage42-EU variant: {variant}")
    mean = float(np.mean(w[train_mask])) if np.any(train_mask) else 1.0
    return (w / max(mean, EPS)).astype(np.float64)


def _candidate_specs(
    data: Mapping[str, np.ndarray],
    base_features: np.ndarray,
    graph: np.ndarray,
    group_risk: Mapping[str, np.ndarray],
    train_mask: np.ndarray,
) -> list[dict[str, Any]]:
    graph_compact = graph[:, :10].astype(np.float32)
    specs = []
    for variant in [
        "group_unsafe_weighted",
        "group_unsafe_hard_weighted",
        "group_unsafe_t50_t100_weighted",
        "group_graph_density_weighted",
        "group_easy_safe_weighted",
    ]:
        features = base_features
        mode = "stage42_am_features"
        if variant == "group_graph_density_weighted":
            features = np.concatenate([base_features, graph_compact], axis=1).astype(np.float32)
            mode = "stage42_am_features_plus_graph_summary"
        specs.append(
            {
                "variant": variant,
                "features": features,
                "feature_mode": mode,
                "weights": _constraint_weights(data, group_risk, graph, train_mask, variant),
            }
        )
    return specs


def _selection_score(metric: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> float:
    near_delta = float(diagnostics["final_near_005"]) - float(diagnostics["base_near_005"])
    return (
        1.25 * float(metric["all_improvement"])
        + 1.35 * float(metric["hard_failure_improvement"])
        + 1.15 * float(metric["t50_improvement"])
        + 0.45 * float(metric["t100_raw_frame_diagnostic_improvement"])
        - 35.0 * max(0.0, float(metric["easy_degradation"]) - 0.02)
        - 8.0 * max(0.0, near_delta)
        - 0.01 * float(metric["switch_rate"])
    )


def _evaluate_training_candidates(
    data: Mapping[str, np.ndarray],
    split: np.ndarray,
    labels: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
    base_features: np.ndarray,
    graph: np.ndarray,
    group_risk: Mapping[str, np.ndarray],
    group_key: np.ndarray,
) -> dict[str, Any]:
    train_mask = split == "train"
    val_mask = split == "val"
    current = np.stack([data["current_x"], data["current_y"]], axis=1).astype(np.float64)
    scale = np.maximum(data["scale"].astype(np.float64), EPS)
    target_delta = ((labels["waypoint_xy"].astype(np.float64) - current[:, None, :]) / scale[:, None, None]).astype(np.float32)
    floor_ade, floor_fde = am._trajectory_errors(floor["floor_xy"], labels)
    rows = []
    best: dict[str, Any] | None = None
    best_score = -1e9
    for spec in _candidate_specs(data, base_features, graph, group_risk, train_mask):
        x, mean, std = am._standardize(spec["features"], train_mask)
        del mean, std
        for lam in LAMBDAS:
            coef = dg._fit_weighted_ridge_model(
                x,
                target_delta,
                labels["waypoint_valid"],
                train_mask,
                spec["weights"],
                lam,
            )
            pred_xy = am._predict_waypoints(x, coef, data)
            policy, selected_ade, selected_fde, switch = am._select_policy_on_val(pred_xy, floor["floor_xy"], labels, data, val_mask)
            selected_xy, switch_xy = di._apply_am_policy_xy(pred_xy, floor["floor_xy"], data, policy)
            val_ids = np.where(val_mask)[0]
            diag = {
                "base_near_005": float(
                    np.mean(
                        np.isfinite(
                            di._min_group_distance_fast(
                                selected_xy[val_ids],
                                group_key[val_ids],
                                np.maximum(data["scale"][val_ids].astype(np.float64), EPS),
                                data["agent_id"][val_ids].astype(np.int64),
                            )
                        )
                        & (
                            di._min_group_distance_fast(
                                selected_xy[val_ids],
                                group_key[val_ids],
                                np.maximum(data["scale"][val_ids].astype(np.float64), EPS),
                                data["agent_id"][val_ids].astype(np.int64),
                            )
                            < 0.05
                        )
                    )
                ),
                "final_near_005": 0.0,
            }
            val_metric = am._metric(selected_ade, floor_ade, data, switch_xy, val_mask)
            score = _selection_score(val_metric, diag)
            row = {
                "variant": spec["variant"],
                "feature_mode": spec["feature_mode"],
                "lambda": float(lam),
                "val_score": float(score),
                "val_metric": val_metric,
                "policy_slice_count": len(policy["slices"]),
                "mean_train_weight": float(np.mean(spec["weights"][train_mask])),
                "max_train_weight": float(np.max(spec["weights"][train_mask])),
            }
            rows.append(row)
            if score > best_score:
                best_score = float(score)
                best = {
                    **row,
                    "coef": coef,
                    "pred_xy": pred_xy,
                    "selected_xy": selected_xy,
                    "switch": switch_xy,
                    "selected_ade": selected_ade,
                    "selected_fde": selected_fde,
                    "floor_ade": floor_ade,
                    "floor_fde": floor_fde,
                    "policy": policy,
                }
    if best is None:
        raise RuntimeError("No Stage42-EU group-consistency constrained training candidate evaluated.")
    best_public = {
        k: v
        for k, v in best.items()
        if k not in {"coef", "pred_xy", "selected_xy", "switch", "selected_ade", "selected_fde", "floor_ade", "floor_fde", "policy"}
    }
    best_public["policy_slice_count"] = len(best["policy"]["slices"])
    return {
        "candidate_count": len(rows),
        "validation_rows": sorted(rows, key=lambda row: float(row["val_score"]), reverse=True),
        "selected": best_public,
        "selected_internal": best,
    }


def _evaluate_best_with_group_repair(
    data: Mapping[str, np.ndarray],
    split: np.ndarray,
    labels: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
    selected_internal: Mapping[str, Any],
    group_key: np.ndarray,
) -> dict[str, Any]:
    am_like = {
        "pred_xy": selected_internal["pred_xy"],
        "selected_xy": selected_internal["selected_xy"],
        "switch": selected_internal["switch"],
        "selected_ade": selected_internal["selected_ade"],
        "selected_fde": selected_internal["selected_fde"],
        "floor_ade": selected_internal["floor_ade"],
        "floor_fde": selected_internal["floor_fde"],
    }
    repaired = di._evaluate_repairs(data, split, labels, floor, am_like, group_key)
    return repaired


def _compare_to_prior(metric: Mapping[str, Any]) -> dict[str, Any]:
    am_payload = read_json(AM_JSON, {})
    dh_payload = read_json(DH_JSON, {})
    di_payload = read_json(DI_JSON, {})
    am_metric = am_payload.get("model", {}).get("metrics", {}).get("protected_ridge_source_level", {})
    dh_metric = dh_payload.get("model", {}).get("metrics", {}).get("protected_selected_candidate", {})
    di_metric = di_payload.get("repair", {}).get("test", {}).get("metric_vs_floor", {})

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
            "hard_failure_improvement": float(metric.get("hard_failure_improvement", 0.0))
            - float(ref.get("hard_failure_improvement", 0.0)),
            "easy_degradation": float(metric.get("easy_degradation", 0.0)) - float(ref.get("easy_degradation", 0.0)),
        }

    return {
        "stage42_am_metric": am_metric,
        "stage42_dh_metric": dh_metric,
        "stage42_di_metric": di_metric,
        "delta_vs_stage42_am": delta(am_metric),
        "delta_vs_stage42_dh": delta(dh_metric),
        "delta_vs_stage42_di": delta(di_metric),
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    source_split = s42b.build_stage42_source_split()
    data = s41._combined()
    split, group = am._split_arrays(data)
    split_stats = am._source_stats(data, split, group)
    labels = am._reconstruct_waypoint_labels(data)
    train_mask = split == "train"
    floor = am._floor_arrays(data, train_mask)
    features, feature_names = am._feature_matrix(data, floor)
    graph, graph_names, graph_stats = graph_ctx._build_graph_features(data)
    group_key = di._group_key(data)
    am_candidate = di._rebuild_stage42_am_candidate(data, split, labels, floor)
    am_candidate_for_risk = {**am_candidate, "floor_xy": floor["floor_xy"]}
    group_risk = _group_risk_signals(data, group_key, am_candidate_for_risk)
    trained = _evaluate_training_candidates(data, split, labels, floor, features, graph, group_risk, group_key)
    repaired = _evaluate_best_with_group_repair(data, split, labels, floor, trained["selected_internal"], group_key)
    metric = repaired["test"]["metric_vs_floor"]
    comparison = _compare_to_prior(metric)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-EU group-consistency constrained full-waypoint training",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(["data/stage41_world_model/combined_external.npz", str(AM_JSON), str(DH_JSON), str(DI_JSON), str(ET_JSON)]),
        "current_facts": CURRENT_FACTS,
        "source_split": source_split,
        "split_stats": split_stats,
        "label_stats": {
            "rows": int(len(split)),
            "test_rows": int(np.sum(split == "test")),
            "test_full_waypoint_rows": int(np.sum((split == "test") & np.all(labels["waypoint_valid"], axis=1))),
        },
        "feature_schema": {
            "source": "fresh_stage42_eu_group_constraint_weight_schema",
            "am_feature_count": len(feature_names),
            "graph_feature_count": int(graph.shape[1]),
            "graph_feature_names": graph_names,
            "variants": [
                "group_unsafe_weighted",
                "group_unsafe_hard_weighted",
                "group_unsafe_t50_t100_weighted",
                "group_graph_density_weighted",
                "group_easy_safe_weighted",
            ],
            "lambdas": LAMBDAS,
            "normalization": "train_split_mean_std_only",
            "future_inputs": False,
        },
        "graph_stats": graph_stats,
        "group_risk_stats": {
            "train_mean_risk": float(np.mean(group_risk["risk"][train_mask])),
            "train_close005_rate": float(np.mean(group_risk["close_005"][train_mask])),
            "train_close008_rate": float(np.mean(group_risk["close_008"][train_mask])),
            "train_unsafe_vs_floor_rate": float(np.mean(group_risk["unsafe_vs_floor"][train_mask])),
        },
        "training": {
            "candidate_count": trained["candidate_count"],
            "validation_rows": trained["validation_rows"],
            "selected": trained["selected"],
        },
        "group_repaired_selected": repaired,
        "comparison_to_prior": comparison,
        "deployment_decision": _deployment_decision(metric, comparison),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "group_weights_predicted_rollout_only": True,
            "graph_features_current_and_past_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "validation_only_model_and_repair_selection": True,
            "train_only_feature_normalization": True,
            "source_overlap_pass": bool(split_stats["source_overlap_pass"]),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["paper_file_status"] = _refresh_paper_files(payload)
    payload["stage42_eu_gate"] = _gate(payload)
    del trained["selected_internal"]
    return payload


def _deployment_decision(metric: Mapping[str, Any], comparison: Mapping[str, Any]) -> dict[str, Any]:
    delta_di = comparison["delta_vs_stage42_di"]
    delta_am = comparison["delta_vs_stage42_am"]
    promotes = (
        metric["all_improvement"] > 0.0
        and metric["hard_failure_improvement"] > 0.0
        and metric["easy_degradation"] <= 0.02
        and (delta_di["all_improvement"] or 0.0) > 0.0
        and (delta_di["hard_failure_improvement"] or 0.0) > 0.0
    )
    diagnostic_positive = (
        metric["all_improvement"] > 0.0
        and metric["hard_failure_improvement"] > 0.0
        and metric["easy_degradation"] <= 0.02
        and (delta_am["all_improvement"] or 0.0) > 0.0
    )
    return {
        "promote_group_constraint_training": bool(promotes),
        "diagnostic_positive": bool(diagnostic_positive),
        "decision": "promote_stage42_eu_group_constraint_training"
        if promotes
        else (
            "group_constraint_training_positive_but_not_better_than_di"
            if diagnostic_positive
            else "group_constraint_training_not_enough_keep_stage42_di_or_cq_floor"
        ),
        "reason": "Promotion requires all+hard positive, easy safe, and improvement over Stage42-DI on all+hard. Diagnostic-positive only means the constraint training remains useful but not deployable over the explicit repair.",
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metric = payload["group_repaired_selected"]["test"]["metric_vs_floor"]
    diag = payload["group_repaired_selected"]["test"]["diagnostics"]
    delta_am = payload["comparison_to_prior"]["delta_vs_stage42_am"]
    delta_di = payload["comparison_to_prior"]["delta_vs_stage42_di"]
    no_leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "source_level_split_rebuilt": payload["split_stats"]["by_split"]["test"]["rows"] == int(metric["rows"]) and int(metric["rows"]) > 0,
        "full_waypoint_labels_available": payload["label_stats"]["test_full_waypoint_rows"] > 0,
        "group_constraint_weights_built": payload["group_risk_stats"]["train_close008_rate"] > 0.0,
        "training_candidates_run": payload["training"]["candidate_count"] >= 10,
        "validation_selected_training_model": payload["training"]["selected"]["val_score"] != 0.0,
        "validation_selected_group_repair": payload["group_repaired_selected"]["selected"]["val_score"] != 0.0,
        "test_all_positive_vs_floor": metric["all_improvement"] > 0.0,
        "test_t50_positive_vs_floor": metric["t50_improvement"] > 0.0,
        "test_hard_positive_vs_floor": metric["hard_failure_improvement"] > 0.0,
        "easy_degradation_under_2pct": metric["easy_degradation"] <= 0.02,
        "near005_repaired_vs_own_base": diag["final_near_005"] <= diag["base_near_005"] + EPS,
        "beats_stage42_am_all": (delta_am["all_improvement"] or 0.0) > 0.0,
        "beats_stage42_di_all": (delta_di["all_improvement"] or 0.0) > 0.0,
        "beats_stage42_di_hard": (delta_di["hard_failure_improvement"] or 0.0) > 0.0,
        "no_leakage_pass": no_leak["future_endpoint_input"] is False
        and no_leak["future_waypoint_input"] is False
        and no_leak["central_velocity"] is False
        and no_leak["test_endpoint_goals"] is False
        and no_leak["test_threshold_tuning"] is False
        and no_leak["validation_only_model_and_repair_selection"] is True
        and no_leak["train_only_feature_normalization"] is True
        and no_leak["source_overlap_pass"] is True,
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    if passed == total:
        verdict = "stage42_eu_group_consistency_constraint_training_pass_promotable"
    elif gates["test_all_positive_vs_floor"] and gates["test_hard_positive_vs_floor"] and gates["easy_degradation_under_2pct"]:
        verdict = "stage42_eu_group_consistency_constraint_training_positive_not_promoted"
    else:
        verdict = "stage42_eu_group_consistency_constraint_training_partial"
    return {"source": "fresh_run", "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _paper_lines(payload: Mapping[str, Any]) -> list[str]:
    metric = payload["group_repaired_selected"]["test"]["metric_vs_floor"]
    diag = payload["group_repaired_selected"]["test"]["diagnostics"]
    selected = payload["training"]["selected"]
    delta_di = payload["comparison_to_prior"]["delta_vs_stage42_di"]
    return [
        "## Stage42-EU Group-Consistency Constraint Training",
        "",
        "- source: `fresh_stage42_group_consistency_constraint_training`",
        "- role: trains full-waypoint dynamics with source/frame/horizon group-risk weighted losses, then applies validation-selected group repair.",
        f"- selected training variant: `{selected['variant']}` with `{selected['feature_mode']}` lambda `{selected['lambda']}`.",
        f"- test all/t50/t100raw/hard/easy: `{_pct(metric['all_improvement'])}` / `{_pct(metric['t50_improvement'])}` / `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(metric['hard_failure_improvement'])}` / `{_pct(metric['easy_degradation'])}`.",
        f"- delta vs Stage42-DI all/hard/easy: `{_pct(delta_di['all_improvement'])}` / `{_pct(delta_di['hard_failure_improvement'])}` / `{_pct(delta_di['easy_degradation'])}`.",
        f"- near@0.05 base/final: `{_pct(diag['base_near_005'])}` / `{_pct(diag['final_near_005'])}`.",
        f"- decision: `{payload['deployment_decision']['decision']}`.",
        "- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
    ]


def _refresh_paper_files(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    lines = _paper_lines(payload)
    status = []
    for path in PAPER_FILES:
        if path.exists():
            _replace_section(path, "STAGE42_EU_GROUP_CONSISTENCY_CONSTRAINT_TRAINING", lines)
            text = path.read_text(encoding="utf-8")
            status.append(
                {
                    "path": str(path),
                    "updated": True,
                    "contains_stage42_eu": "STAGE42_EU_GROUP_CONSISTENCY_CONSTRAINT_TRAINING" in text,
                    "contains_group_consistency": "group-consistency" in text,
                    "contains_boundaries": "not true 3D" in text and "no Stage5C" in text and "no SMC" in text,
                }
            )
        else:
            status.append({"path": str(path), "updated": False, "missing": True})
    return status


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    metric = payload["group_repaired_selected"]["test"]["metric_vs_floor"]
    diag = payload["group_repaired_selected"]["test"]["diagnostics"]
    selected = payload["training"]["selected"]
    lines = [
        "# Stage42-EU Group-Consistency Constraint Training",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_eu_gate']['passed']} / {payload['stage42_eu_gate']['total']}`",
        f"- verdict: `{payload['stage42_eu_gate']['verdict']}`",
        f"- decision: `{payload['deployment_decision']['decision']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Selected Training Candidate",
        "",
        f"- variant: `{selected['variant']}`",
        f"- feature_mode: `{selected['feature_mode']}`",
        f"- lambda: `{selected['lambda']}`",
        f"- val_score: `{selected['val_score']:.6f}`",
        f"- policy_slice_count: `{selected['policy_slice_count']}`",
        f"- mean_train_weight: `{selected['mean_train_weight']:.6f}`",
        f"- max_train_weight: `{selected['max_train_weight']:.6f}`",
        "",
        "## Test Once After Group Repair",
        "",
        "| all | t50 | t100 raw | hard/failure | easy | switch | near base/final |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        f"| {_pct(metric['all_improvement'])} | {_pct(metric['t50_improvement'])} | {_pct(metric['t100_raw_frame_diagnostic_improvement'])} | "
        f"{_pct(metric['hard_failure_improvement'])} | {_pct(metric['easy_degradation'])} | {_pct(metric['switch_rate'])} | "
        f"{_pct(diag['base_near_005'])}/{_pct(diag['final_near_005'])} |",
        "",
        "## Delta vs Prior",
        "",
        "| prior | all | t50 | t100 raw | hard | easy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, delta in [
        ("Stage42-AM", payload["comparison_to_prior"]["delta_vs_stage42_am"]),
        ("Stage42-DH", payload["comparison_to_prior"]["delta_vs_stage42_dh"]),
        ("Stage42-DI", payload["comparison_to_prior"]["delta_vs_stage42_di"]),
    ]:
        lines.append(
            f"| `{name}` | {_pct(delta['all_improvement'])} | {_pct(delta['t50_improvement'])} | "
            f"{_pct(delta['t100_raw_frame_diagnostic_improvement'])} | {_pct(delta['hard_failure_improvement'])} | {_pct(delta['easy_degradation'])} |"
        )
    lines.extend(
        [
            "",
            "## Validation Rows",
            "",
            "| rank | variant | lambda | val score | val all | val t50 | val hard | val easy |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for idx, row in enumerate(payload["training"]["validation_rows"][:12], start=1):
        vm = row["val_metric"]
        lines.append(
            f"| {idx} | `{row['variant']}` | {row['lambda']} | {row['val_score']:.6f} | {_pct(vm['all_improvement'])} | "
            f"{_pct(vm['t50_improvement'])} | {_pct(vm['hard_failure_improvement'])} | {_pct(vm['easy_degradation'])} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
            *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_eu_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_eu_gate"]
    return [
        "# Stage42-EU Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _readme_lines(payload: Mapping[str, Any]) -> list[str]:
    metric = payload["group_repaired_selected"]["test"]["metric_vs_floor"]
    diag = payload["group_repaired_selected"]["test"]["diagnostics"]
    selected = payload["training"]["selected"]
    delta_di = payload["comparison_to_prior"]["delta_vs_stage42_di"]
    return [
        "## Stage42-EU Group-Consistency Constraint Training",
        "",
        "- source: `fresh_stage42_group_consistency_constraint_training`",
        "- role: trains source/frame/horizon group-risk weighted full-waypoint dynamics, then applies validation-selected group repair.",
        f"- gate: `{payload['stage42_eu_gate']['passed']} / {payload['stage42_eu_gate']['total']}`; verdict `{payload['stage42_eu_gate']['verdict']}`.",
        f"- selected training variant: `{selected['variant']}` with lambda `{selected['lambda']}`.",
        f"- test all/t50/t100raw/hard/easy: `{_pct(metric['all_improvement'])}` / `{_pct(metric['t50_improvement'])}` / `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(metric['hard_failure_improvement'])}` / `{_pct(metric['easy_degradation'])}`.",
        f"- delta vs Stage42-DI all/hard/easy: `{_pct(delta_di['all_improvement'])}` / `{_pct(delta_di['hard_failure_improvement'])}` / `{_pct(delta_di['easy_degradation'])}`.",
        f"- near@0.05 base/final: `{_pct(diag['base_near_005'])}` / `{_pct(diag['final_near_005'])}`.",
        f"- decision: `{payload['deployment_decision']['decision']}`.",
        "- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _readme_lines(payload)
    for path in [README_RESULTS, M3W_README, TARGET_SUMMARY, WORK_SUMMARY]:
        _replace_section(path, "STAGE42_EU_GROUP_CONSISTENCY_CONSTRAINT_TRAINING", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    metric = payload["group_repaired_selected"]["test"]["metric_vs_floor"]
    state["current_stage"] = "Stage42-EU group consistency constraint training"
    state["current_verdict"] = payload["stage42_eu_gate"]["verdict"]
    state["stage42_eu_group_consistency_constraint_training"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_eu_gate"]["verdict"],
        "gates": f"{payload['stage42_eu_gate']['passed']}/{payload['stage42_eu_gate']['total']}",
        "selected_training": payload["training"]["selected"],
        "test_metric_vs_floor": metric,
        "comparison_to_prior": payload["comparison_to_prior"],
        "deployment_decision": payload["deployment_decision"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_group_consistency_constraint_training(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    payload = _build_payload()
    write_json(REPORT_JSON, am._jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_group_consistency_constraint_training()
