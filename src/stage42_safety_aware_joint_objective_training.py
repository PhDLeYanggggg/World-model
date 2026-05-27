from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_external_validation as s42b
from src import stage42_full_waypoint_all_hard_loss_repair as dg
from src import stage42_group_consistency_full_waypoint_repair as di
from src import stage42_objective_level_proximity_training as fc
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_source_level_graph_context as graph_ctx
from src import stage42_waypointwise_group_repel_repair as fa
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "safety_aware_joint_objective_training_stage42.json"
REPORT_MD = OUT_DIR / "safety_aware_joint_objective_training_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_fd_gate.md"

AM_JSON = OUT_DIR / "source_level_full_waypoint_eval_stage42.json"
DI_JSON = OUT_DIR / "group_consistency_full_waypoint_repair_stage42.json"
FA_JSON = OUT_DIR / "waypointwise_group_repel_repair_stage42.json"
FB_JSON = OUT_DIR / "proximity_pareto_composer_stage42.json"
FC_JSON = OUT_DIR / "objective_level_proximity_training_stage42.json"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
PAPER_FILES = fc.PAPER_FILES

SOURCE = "fresh_stage42_safety_aware_joint_objective_training"
LAMBDAS = [0.1, 1.0, 10.0, 100.0]
EPS = 1e-6

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-FC 证明 objective-level proximity training 能提高 all/t50/hard，但 near@0.05 比 Stage42-DI 差。",
    "Stage42-FD 把 safety-aware teacher regularization 加进训练目标，测试是否能同时保留 FC 的 all/hard 增益和 DI/FA 的 proximity safety。",
    "FA waypoint-wise safety teacher 只作为 train loss regularizer；future labels 只作为 supervised loss/eval labels，不是 inference input。",
    "features 仍只来自当前/过去 history、baseline rollout context、graph/current neighbor summary、domain/horizon metadata。",
    "validation 选择 objective variant、teacher blend、lambda 和 safe policy；test 只评一次。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

CLAIM_BOUNDARY = dict(fc.CLAIM_BOUNDARY)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _jsonable(value: Any) -> Any:
    return am._jsonable(value)


def _load_prior() -> dict[str, Any]:
    return {
        "am": read_json(AM_JSON, {}),
        "di": read_json(DI_JSON, {}),
        "fa": read_json(FA_JSON, {}),
        "fb": read_json(FB_JSON, {}),
        "fc": read_json(FC_JSON, {}),
    }


def _safe_teacher_xy(
    data: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
    am_candidate: Mapping[str, Any],
    group_key: np.ndarray,
    fa_candidate: Mapping[str, Any],
) -> np.ndarray:
    ids = np.arange(len(data["row_id"]), dtype=np.int64)
    repaired = fa._repair_subset_waypointwise(
        ids,
        fa_candidate,
        data,
        labels,
        floor["floor_xy"].astype(np.float32),
        am_candidate["pred_xy"].astype(np.float32),
        am_candidate["selected_xy"].astype(np.float32),
        am_candidate["switch"].astype(bool),
        group_key,
    )
    out = labels["waypoint_xy"].astype(np.float32).copy()
    out[ids] = repaired["selected_xy"].astype(np.float32)
    return out


def _safety_mask(signals: Mapping[str, np.ndarray], train_mask: np.ndarray, mode: str) -> np.ndarray:
    if mode == "future_or_base_unsafe":
        mask = (signals["future_close_008"] > 0.0) | (signals["base_unsafe"] > 0.0) | (signals["base_close_008"] > 0.0)
    elif mode == "strict_near005":
        mask = (signals["future_close_005"] > 0.0) | (signals["base_unsafe"] > 0.0)
    elif mode == "risk_top_quartile":
        threshold = float(np.percentile(signals["risk"][train_mask], 75)) if np.any(train_mask) else 0.0
        mask = signals["risk"] >= threshold
    else:
        raise ValueError(f"unknown safety mask mode: {mode}")
    return mask.astype(bool)


def _blend_teacher_targets(
    labels_xy: np.ndarray,
    teacher_xy: np.ndarray,
    safety_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    target = labels_xy.astype(np.float32).copy()
    rows = np.asarray(safety_mask, dtype=bool)
    target[rows] = ((1.0 - float(alpha)) * target[rows].astype(np.float64) + float(alpha) * teacher_xy[rows].astype(np.float64)).astype(np.float32)
    return target


def _candidate_specs(
    data: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    base_features: np.ndarray,
    graph: np.ndarray,
    signals: Mapping[str, np.ndarray],
    teacher_xy: np.ndarray,
    train_mask: np.ndarray,
) -> list[dict[str, Any]]:
    labels_xy = labels["waypoint_xy"].astype(np.float32)
    graph_compact = graph[:, :10].astype(np.float32)
    specs: list[dict[str, Any]] = [
        {
            "variant": "fc_label_proximity_control",
            "feature_mode": "stage42_am_features",
            "features": base_features,
            "weights": fc._objective_weights(data, signals, graph, train_mask, "label_proximity_objective"),
            "target_xy": labels_xy,
            "teacher_alpha": 0.0,
            "safety_mask_mode": "none",
        },
        {
            "variant": "fc_graph_label_proximity_control",
            "feature_mode": "stage42_am_features_plus_graph_summary",
            "features": np.concatenate([base_features, graph_compact], axis=1).astype(np.float32),
            "weights": fc._objective_weights(data, signals, graph, train_mask, "graph_label_proximity_objective"),
            "target_xy": labels_xy,
            "teacher_alpha": 0.0,
            "safety_mask_mode": "none",
        },
    ]
    for mask_mode in ["strict_near005", "future_or_base_unsafe", "risk_top_quartile"]:
        mask = _safety_mask(signals, train_mask, mask_mode)
        for alpha in [0.25, 0.50, 0.75]:
            target_xy = _blend_teacher_targets(labels_xy, teacher_xy, mask, alpha)
            weights = fc._objective_weights(data, signals, graph, train_mask, "label_proximity_hard_long_objective")
            weights = weights.copy()
            weights[mask] *= 1.0 + alpha
            specs.append(
                {
                    "variant": f"safety_teacher_blend_{mask_mode}_{int(alpha * 100):02d}",
                    "feature_mode": "stage42_am_features",
                    "features": base_features,
                    "weights": weights,
                    "target_xy": target_xy,
                    "teacher_alpha": float(alpha),
                    "safety_mask_mode": mask_mode,
                    "train_safety_mask_rate": float(np.mean(mask[train_mask])) if np.any(train_mask) else 0.0,
                }
            )
            specs.append(
                {
                    "variant": f"graph_safety_teacher_blend_{mask_mode}_{int(alpha * 100):02d}",
                    "feature_mode": "stage42_am_features_plus_graph_summary",
                    "features": np.concatenate([base_features, graph_compact], axis=1).astype(np.float32),
                    "weights": weights,
                    "target_xy": target_xy,
                    "teacher_alpha": float(alpha),
                    "safety_mask_mode": mask_mode,
                    "train_safety_mask_rate": float(np.mean(mask[train_mask])) if np.any(train_mask) else 0.0,
                }
            )
    return specs


def _selection_score(metric: Mapping[str, Any], diagnostics: Mapping[str, Any], di_val_near: float) -> float:
    near_penalty = max(0.0, float(diagnostics["near_005"]) - float(di_val_near))
    return (
        1.45 * float(metric["all_improvement"])
        + 1.45 * float(metric["hard_failure_improvement"])
        + 1.10 * float(metric["t50_improvement"])
        + 0.45 * float(metric["t100_raw_frame_diagnostic_improvement"])
        - 45.0 * max(0.0, float(metric["easy_degradation"]) - 0.02)
        - 25.0 * near_penalty
        - 0.01 * float(metric["switch_rate"])
    )


def _delta(metric: Mapping[str, Any], ref: Mapping[str, Any]) -> dict[str, float | None]:
    return fc._delta(metric, ref)


def _evaluate_di_fa_reference(
    data: Mapping[str, np.ndarray],
    split: np.ndarray,
    labels: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
    am_candidate: Mapping[str, Any],
    group_key: np.ndarray,
    prior: Mapping[str, Any],
) -> dict[str, Any]:
    di_candidate = prior["di"].get("repair", {}).get("selected", {}).get("candidate")
    fa_candidate = prior["fa"].get("repair", {}).get("selected", {}).get("candidate")
    if not di_candidate or not fa_candidate:
        raise RuntimeError("Stage42-FD requires Stage42-DI and Stage42-FA selected candidate artifacts.")
    refs: dict[str, Any] = {"di_candidate": di_candidate, "fa_candidate": fa_candidate}
    for split_name in ["val", "test"]:
        ids = np.where(split == split_name)[0]
        refs[f"di_{split_name}"] = di._repair_subset(
            ids,
            di_candidate,
            data,
            labels,
            floor["floor_xy"].astype(np.float32),
            am_candidate["pred_xy"].astype(np.float32),
            am_candidate["selected_xy"].astype(np.float32),
            am_candidate["switch"].astype(bool),
            group_key,
        )
        refs[f"fa_{split_name}"] = fa._repair_subset_waypointwise(
            ids,
            fa_candidate,
            data,
            labels,
            floor["floor_xy"].astype(np.float32),
            am_candidate["pred_xy"].astype(np.float32),
            am_candidate["selected_xy"].astype(np.float32),
            am_candidate["switch"].astype(bool),
            group_key,
        )
    return refs


def _evaluate_candidates(
    data: Mapping[str, np.ndarray],
    split: np.ndarray,
    labels: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
    base_features: np.ndarray,
    graph: np.ndarray,
    signals: Mapping[str, np.ndarray],
    teacher_xy: np.ndarray,
    group_key: np.ndarray,
    references: Mapping[str, Any],
) -> dict[str, Any]:
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    val_ids = np.where(val_mask)[0]
    test_ids = np.where(test_mask)[0]
    current = np.stack([data["current_x"], data["current_y"]], axis=1).astype(np.float64)
    scale = np.maximum(data["scale"].astype(np.float64), EPS)
    floor_ade, floor_fde = am._trajectory_errors(floor["floor_xy"], labels)
    h = data["horizon"].astype(int)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_score = -1e9
    di_val_near = float(references["di_val"]["diagnostics"]["final_near_005"])
    for spec in _candidate_specs(data, labels, base_features, graph, signals, teacher_xy, train_mask):
        x, mean, std = am._standardize(spec["features"], train_mask)
        del mean, std
        target_delta = ((spec["target_xy"].astype(np.float64) - current[:, None, :]) / scale[:, None, None]).astype(np.float32)
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
            policy, _selected_ade, _selected_fde, _switch = am._select_policy_on_val(
                pred_xy,
                floor["floor_xy"],
                labels,
                data,
                val_mask,
            )
            selected_xy, switch_xy = di._apply_am_policy_xy(pred_xy, floor["floor_xy"], data, policy)
            selected_ade, selected_fde = am._trajectory_errors(selected_xy, labels)
            pred_ade, pred_fde = am._trajectory_errors(pred_xy, labels)
            val_metric = am._metric(selected_ade, floor_ade, data, switch_xy, val_mask)
            val_diag = fc._near_diagnostics(selected_xy, data, val_ids, group_key)
            score = _selection_score(val_metric, val_diag, di_val_near)
            row = {
                "variant": spec["variant"],
                "feature_mode": spec["feature_mode"],
                "lambda": float(lam),
                "teacher_alpha": float(spec.get("teacher_alpha", 0.0)),
                "safety_mask_mode": spec.get("safety_mask_mode", "none"),
                "train_safety_mask_rate": float(spec.get("train_safety_mask_rate", 0.0)),
                "val_score": float(score),
                "val_metric": val_metric,
                "val_near_diagnostics": val_diag,
                "val_near_delta_vs_di": float(val_diag["near_005"]) - di_val_near,
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
                    "pred_ade": pred_ade,
                    "pred_fde": pred_fde,
                    "floor_ade": floor_ade,
                    "floor_fde": floor_fde,
                    "policy": policy,
                }
    if best is None:
        raise RuntimeError("No Stage42-FD safety-aware objective candidate evaluated.")
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    domain = data["dataset"].astype(str)
    selected_xy = best["selected_xy"]
    pred_xy = best["pred_xy"]
    metric = am._metric(best["selected_ade"], best["floor_ade"], data, best["switch"], test_mask)
    ungated_metric = am._metric(best["pred_ade"], best["floor_ade"], data, np.ones(len(h), dtype=bool), test_mask)
    return {
        "candidate_count": len(rows),
        "validation_rows": sorted(rows, key=lambda row: float(row["val_score"]), reverse=True),
        "selected": {
            k: v
            for k, v in best.items()
            if k
            not in {
                "coef",
                "pred_xy",
                "selected_xy",
                "switch",
                "selected_ade",
                "selected_fde",
                "pred_ade",
                "pred_fde",
                "floor_ade",
                "floor_fde",
                "policy",
            }
        },
        "policy": best["policy"],
        "metrics": {
            "protected_selected_candidate": metric,
            "ungated_selected_candidate": ungated_metric,
            "protected_selected_candidate_fde": am._metric(best["selected_fde"], best["floor_fde"], data, best["switch"], test_mask),
        },
        "diagnostics": {
            "protected_near": fc._near_diagnostics(selected_xy, data, test_ids, group_key),
            "ungated_near": fc._near_diagnostics(pred_xy, data, test_ids, group_key),
            "floor_near": fc._near_diagnostics(floor["floor_xy"], data, test_ids, group_key),
        },
        "bootstrap": {
            "all": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask, seed=43301),
            "t50": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & (h == 50), seed=43302),
            "t100_raw_frame_diagnostic": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & (h == 100), seed=43303),
            "hard_failure": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & hard_failure, seed=43304),
            "easy_degradation": am._bootstrap_ci(best["floor_ade"], best["selected_ade"], test_mask & easy, seed=43305),
        },
        "by_domain": {
            d: am._metric(best["selected_ade"], best["floor_ade"], data, best["switch"], test_mask & (domain == d))
            for d in sorted(set(domain[test_mask].tolist()))
        },
        "by_horizon": {
            str(hh): am._metric(best["selected_ade"], best["floor_ade"], data, best["switch"], test_mask & (h == hh))
            for hh in [10, 25, 50, 100]
        },
    }


def _comparison_to_prior(metric: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    prior = _load_prior()
    am_metric = prior["am"].get("model", {}).get("metrics", {}).get("protected_ridge_source_level", {})
    di_metric = prior["di"].get("repair", {}).get("test", {}).get("metric_vs_floor", {})
    fa_metric = prior["fa"].get("repair", {}).get("test", {}).get("metric_vs_floor", {})
    fb_metric = prior["fb"].get("repair", {}).get("test", {}).get("metric_vs_floor", {})
    fc_metric = prior["fc"].get("model", {}).get("metrics", {}).get("protected_selected_candidate", {})
    di_diag = prior["di"].get("repair", {}).get("test", {}).get("diagnostics", {})
    fb_diag = prior["fb"].get("repair", {}).get("test", {}).get("diagnostics", {})
    fc_diag = prior["fc"].get("model", {}).get("diagnostics", {}).get("protected_near", {})
    protected_near = diagnostics.get("protected_near", {})
    return {
        "stage42_am_metric": am_metric,
        "stage42_di_metric": di_metric,
        "stage42_fa_metric": fa_metric,
        "stage42_fb_metric": fb_metric,
        "stage42_fc_metric": fc_metric,
        "delta_vs_stage42_am": _delta(metric, am_metric),
        "delta_vs_stage42_di": _delta(metric, di_metric),
        "delta_vs_stage42_fa": _delta(metric, fa_metric),
        "delta_vs_stage42_fb": _delta(metric, fb_metric),
        "delta_vs_stage42_fc": _delta(metric, fc_metric),
        "near_delta_vs_stage42_di": float(protected_near.get("near_005", 0.0)) - float(di_diag.get("final_near_005", 0.0)),
        "near_delta_vs_stage42_fb": float(protected_near.get("near_005", 0.0)) - float(fb_diag.get("final_near_005", 0.0)),
        "near_delta_vs_stage42_fc": float(protected_near.get("near_005", 0.0)) - float(fc_diag.get("near_005", 0.0)),
        "stage42_di_near_005": di_diag.get("final_near_005"),
        "stage42_fb_near_005": fb_diag.get("final_near_005"),
        "stage42_fc_near_005": fc_diag.get("near_005"),
    }


def _deployment_decision(metric: Mapping[str, Any], comparison: Mapping[str, Any]) -> dict[str, Any]:
    delta_di = comparison.get("delta_vs_stage42_di", {})
    delta_fb = comparison.get("delta_vs_stage42_fb", {})
    delta_fc = comparison.get("delta_vs_stage42_fc", {})
    near_di = comparison.get("near_delta_vs_stage42_di")
    positive = (
        float(metric.get("all_improvement", 0.0)) > 0.0
        and float(metric.get("hard_failure_improvement", 0.0)) > 0.0
        and float(metric.get("easy_degradation", 1.0)) <= 0.02
    )
    promotable = (
        positive
        and float(delta_di.get("all_improvement") or 0.0) >= 0.0
        and float(delta_di.get("hard_failure_improvement") or 0.0) >= 0.0
        and float(delta_fb.get("all_improvement") or 0.0) >= 0.0
        and float(delta_fc.get("all_improvement") or 0.0) >= 0.0
        and float(delta_fc.get("hard_failure_improvement") or 0.0) >= 0.0
        and (near_di is not None and float(near_di) <= 0.0)
    )
    return {
        "promote_safety_aware_objective": bool(promotable),
        "diagnostic_positive": bool(positive),
        "decision": "promote_stage42_fd_safety_aware_joint_objective"
        if promotable
        else "safety_aware_objective_not_enough_keep_stage42_di_or_cq_floor",
        "reason": "Promotion requires positive all+hard, easy safe, no all/hard loss vs Stage42-DI/FB/FC, and no worse near@0.05 than Stage42-DI.",
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metric = payload.get("model", {}).get("metrics", {}).get("protected_selected_candidate", {})
    comparison = payload.get("comparison_to_prior", {})
    delta_di = comparison.get("delta_vs_stage42_di", {})
    delta_fc = comparison.get("delta_vs_stage42_fc", {})
    no_leak = payload.get("no_leakage", {})
    boundary = payload.get("claim_boundary", {})
    gates = {
        "safety_objective_source_fresh": payload.get("source") == SOURCE,
        "test_rows_present": payload.get("split_stats", {}).get("by_split", {}).get("test", {}).get("rows", 0) > 0,
        "full_waypoint_labels_present": payload.get("label_stats", {}).get("test_full_waypoint_rows", 0) > 0,
        "fa_teacher_regularizer_built": payload.get("teacher_regularizer", {}).get("teacher_source") == "Stage42-FA waypointwise group repel",
        "candidate_grid_nonempty": payload.get("model", {}).get("candidate_count", 0) >= 10,
        "validation_selected": payload.get("model", {}).get("selected", {}).get("val_score") is not None,
        "positive_all": float(metric.get("all_improvement", 0.0)) > 0.0,
        "positive_hard": float(metric.get("hard_failure_improvement", 0.0)) > 0.0,
        "easy_safe": float(metric.get("easy_degradation", 1.0)) <= 0.02,
        "beats_stage42_di_all": float(delta_di.get("all_improvement") or 0.0) >= 0.0,
        "beats_stage42_di_hard": float(delta_di.get("hard_failure_improvement") or 0.0) >= 0.0,
        "beats_stage42_fc_all": float(delta_fc.get("all_improvement") or 0.0) >= 0.0,
        "beats_stage42_fc_hard": float(delta_fc.get("hard_failure_improvement") or 0.0) >= 0.0,
        "near_not_worse_than_stage42_di": comparison.get("near_delta_vs_stage42_di") is not None
        and float(comparison["near_delta_vs_stage42_di"]) <= 0.0,
        "near_better_than_stage42_fc": comparison.get("near_delta_vs_stage42_fc") is not None
        and float(comparison["near_delta_vs_stage42_fc"]) <= 0.0,
        "future_labels_loss_only": no_leak.get("future_waypoint_labels_loss_only") is True,
        "teacher_regularizer_train_loss_only": no_leak.get("teacher_regularizer_train_loss_only") is True,
        "no_future_inference_input": no_leak.get("future_endpoint_input") is False and no_leak.get("future_waypoint_input") is False,
        "no_central_velocity": no_leak.get("central_velocity") is False,
        "no_test_endpoint_goals": no_leak.get("test_endpoint_goals") is False,
        "no_test_threshold_tuning": no_leak.get("test_threshold_tuning") is False,
        "train_only_feature_normalization": no_leak.get("train_only_feature_normalization") is True,
        "validation_only_model_selection": no_leak.get("validation_only_model_selection") is True,
        "no_metric_seconds_claim": boundary.get("global_metric_claim_allowed") is False
        and boundary.get("global_seconds_claim_allowed") is False,
        "stage5c_false": boundary.get("stage5c_executed") is False,
        "smc_false": boundary.get("smc_enabled") is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    promotable = (
        gates["positive_all"]
        and gates["positive_hard"]
        and gates["easy_safe"]
        and gates["beats_stage42_di_all"]
        and gates["beats_stage42_di_hard"]
        and gates["beats_stage42_fc_all"]
        and gates["beats_stage42_fc_hard"]
        and gates["near_not_worse_than_stage42_di"]
    )
    verdict = (
        "stage42_fd_safety_aware_joint_objective_pass_promotable"
        if passed == total and promotable
        else "stage42_fd_safety_aware_joint_objective_positive_not_promoted"
        if gates["positive_all"] and gates["positive_hard"] and gates["easy_safe"]
        else "stage42_fd_safety_aware_joint_objective_fail"
    )
    return {"passed": passed, "total": total, "gates": gates, "verdict": verdict}


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
    am_candidate = di._rebuild_stage42_am_candidate(data, split, labels, floor)
    am_candidate["floor_xy"] = floor["floor_xy"]
    group_key = di._group_key(data)
    prior = _load_prior()
    references = _evaluate_di_fa_reference(data, split, labels, floor, am_candidate, group_key, prior)
    teacher_xy = _safe_teacher_xy(data, labels, floor, am_candidate, group_key, references["fa_candidate"])
    signals = fc._objective_signals(data, labels, graph, group_key, am_candidate)
    model = _evaluate_candidates(data, split, labels, floor, features, graph, signals, teacher_xy, group_key, references)
    metric = model["metrics"]["protected_selected_candidate"]
    comparison = _comparison_to_prior(metric, model["diagnostics"])
    decision = _deployment_decision(metric, comparison)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-FD safety-aware joint objective training",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                str(AM_JSON),
                str(DI_JSON),
                str(FA_JSON),
                str(FB_JSON),
                str(FC_JSON),
            ]
        ),
        "source_split_summary": source_split.get("summary", {}),
        "split_stats": split_stats,
        "label_stats": {
            "missing_track": int(np.sum(labels["missing_track"])),
            "test_full_waypoint_rows": int(np.sum((split == "test") & np.all(labels["waypoint_valid"], axis=1))),
        },
        "feature_schema": {
            "base_feature_count": len(feature_names),
            "graph_feature_count": len(graph_names),
            "future_label_weighting_train_loss_only": True,
            "teacher_regularizer_train_loss_only": True,
            "objective_variants": sorted(
                {
                    row["variant"]
                    for row in model["validation_rows"]
                }
            ),
        },
        "graph_stats": graph_stats,
        "teacher_regularizer": {
            "teacher_source": "Stage42-FA waypointwise group repel",
            "teacher_candidate": references["fa_candidate"],
            "train_loss_only": True,
            "future_label_input": False,
        },
        "reference_validation": {
            "di_metric": references["di_val"]["metric"],
            "di_diagnostics": references["di_val"]["diagnostics"],
            "fa_metric": references["fa_val"]["metric"],
            "fa_diagnostics": references["fa_val"]["diagnostics"],
        },
        "objective_signal_stats": fc._signal_stats(signals, split),
        "model": model,
        "comparison_to_prior": comparison,
        "deployment_decision": decision,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_labels_loss_only": True,
            "teacher_regularizer_train_loss_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_feature_normalization": True,
            "validation_only_model_selection": True,
            "source_overlap_pass": bool(split_stats.get("source_overlap_pass", False)),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["stage42_fd_gate"] = _gate(payload)
    return _jsonable(payload)


def _write_reports(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, payload)
    metric = payload["model"]["metrics"]["protected_selected_candidate"]
    ungated = payload["model"]["metrics"]["ungated_selected_candidate"]
    selected = payload["model"]["selected"]
    comparison = payload["comparison_to_prior"]
    diagnostics = payload["model"]["diagnostics"]
    gate = payload["stage42_fd_gate"]
    lines = [
        "# Stage42-FD Safety-Aware Joint Objective Training",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- selected objective: `{selected['variant']}`",
        f"- feature mode: `{selected['feature_mode']}`",
        f"- lambda: `{selected['lambda']}`",
        f"- teacher alpha: `{selected['teacher_alpha']}`",
        f"- safety mask: `{selected['safety_mask_mode']}`",
        "",
        "## Protected Test Metrics vs Floor",
        "",
        f"- all improvement: `{_pct(metric['all_improvement'])}`",
        f"- t50 improvement: `{_pct(metric['t50_improvement'])}`",
        f"- t100 raw-frame diagnostic improvement: `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}`",
        f"- hard/failure improvement: `{_pct(metric['hard_failure_improvement'])}`",
        f"- easy degradation: `{_pct(metric['easy_degradation'])}`",
        f"- switch rate: `{_pct(metric['switch_rate'])}`",
        "",
        "## Ungated Candidate",
        "",
        f"- all improvement: `{_pct(ungated['all_improvement'])}`",
        f"- t50 improvement: `{_pct(ungated['t50_improvement'])}`",
        f"- hard/failure improvement: `{_pct(ungated['hard_failure_improvement'])}`",
        f"- easy degradation: `{_pct(ungated['easy_degradation'])}`",
        "",
        "## Proximity Diagnostics",
        "",
        f"- protected near@0.05: `{_pct(diagnostics['protected_near']['near_005'])}`",
        f"- floor near@0.05: `{_pct(diagnostics['floor_near']['near_005'])}`",
        f"- ungated near@0.05: `{_pct(diagnostics['ungated_near']['near_005'])}`",
        f"- delta near@0.05 vs Stage42-DI: `{_pct(comparison['near_delta_vs_stage42_di'])}`",
        f"- delta near@0.05 vs Stage42-FB: `{_pct(comparison['near_delta_vs_stage42_fb'])}`",
        f"- delta near@0.05 vs Stage42-FC: `{_pct(comparison['near_delta_vs_stage42_fc'])}`",
        "",
        "## Comparison To Prior",
        "",
        "| reference | all | t50 | t100 raw | hard/failure | easy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ["delta_vs_stage42_am", "delta_vs_stage42_di", "delta_vs_stage42_fa", "delta_vs_stage42_fb", "delta_vs_stage42_fc"]:
        delta = comparison[name]
        lines.append(
            f"| `{name}` | `{_pct(delta['all_improvement'])}` | `{_pct(delta['t50_improvement'])}` | "
            f"`{_pct(delta['t100_raw_frame_diagnostic_improvement'])}` | `{_pct(delta['hard_failure_improvement'])}` | `{_pct(delta['easy_degradation'])}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- promote safety-aware objective: `{payload['deployment_decision']['promote_safety_aware_objective']}`",
            f"- diagnostic positive: `{payload['deployment_decision']['diagnostic_positive']}`",
            f"- decision: `{payload['deployment_decision']['decision']}`",
            f"- reason: {payload['deployment_decision']['reason']}",
            "",
            "## No Leakage / Claim Boundary",
            "",
            "- FA teacher is used only as train loss regularizer; it is not an inference feature.",
            "- future labels are used only for supervised train loss and evaluation labels.",
            "- no central velocity, no test endpoint goals, no test threshold tuning.",
            "- dataset-local/raw-frame 2.5D only; no metric/seconds claim.",
            "- Stage5C not executed; SMC not enabled.",
        ]
    )
    write_md(REPORT_MD, lines)
    gate_lines = ["# Stage42-FD Gates", "", f"Verdict: `{gate['verdict']}`", f"Passed: `{gate['passed']} / {gate['total']}`", ""]
    for key, value in gate["gates"].items():
        gate_lines.append(f"- `{key}`: `{value}`")
    write_md(GATE_MD, gate_lines)


def _summary_section(payload: Mapping[str, Any]) -> str:
    metric = payload["model"]["metrics"]["protected_selected_candidate"]
    selected = payload["model"]["selected"]
    comparison = payload["comparison_to_prior"]
    return "\n".join(
        [
            "<!-- STAGE42_FD_SAFETY_AWARE_JOINT_OBJECTIVE:START -->",
            "## Stage42-FD Safety-Aware Joint Objective Training",
            "",
            f"- source: `{payload['source']}`",
            "- role: tests whether FA safety-teacher regularization inside the training objective can break the FC accuracy/proximity tradeoff.",
            f"- selected objective: `{selected['variant']}`; feature mode `{selected['feature_mode']}`; lambda `{selected['lambda']}`; teacher alpha `{selected['teacher_alpha']}`.",
            f"- gate: `{payload['stage42_fd_gate']['passed']} / {payload['stage42_fd_gate']['total']}`; verdict `{payload['stage42_fd_gate']['verdict']}`.",
            f"- test all/t50/t100raw/hard/easy: `{_pct(metric['all_improvement'])}` / `{_pct(metric['t50_improvement'])}` / `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(metric['hard_failure_improvement'])}` / `{_pct(metric['easy_degradation'])}`.",
            f"- delta vs Stage42-FC all/hard/near005: `{_pct(comparison['delta_vs_stage42_fc']['all_improvement'])}` / `{_pct(comparison['delta_vs_stage42_fc']['hard_failure_improvement'])}` / `{_pct(comparison['near_delta_vs_stage42_fc'])}`.",
            f"- delta vs Stage42-DI all/hard/near005: `{_pct(comparison['delta_vs_stage42_di']['all_improvement'])}` / `{_pct(comparison['delta_vs_stage42_di']['hard_failure_improvement'])}` / `{_pct(comparison['near_delta_vs_stage42_di'])}`.",
            f"- decision: `{payload['deployment_decision']['decision']}`.",
            "- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
            "<!-- STAGE42_FD_SAFETY_AWARE_JOINT_OBJECTIVE:END -->",
            "",
        ]
    )


def _replace_text_section(old: str, tag: str, block: str) -> str:
    start = f"<!-- {tag}:START -->"
    end = f"<!-- {tag}:END -->"
    if start in old and end in old:
        before, rest = old.split(start, 1)
        _, after = rest.split(end, 1)
        return before.rstrip() + "\n\n" + block.strip() + after
    return old.rstrip() + "\n\n" + block.strip() + "\n"


def _append_to_docs(payload: Mapping[str, Any]) -> None:
    section = _summary_section(payload)
    targets = [README_RESULTS, M3W_README, WORK_SUMMARY, *PAPER_FILES]
    for path in targets:
        old = path.read_text() if path.exists() else ""
        new = _replace_text_section(old, "STAGE42_FD_SAFETY_AWARE_JOINT_OBJECTIVE", section)
        path.write_text(new)


def _update_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-FD safety-aware joint objective training"
    state["current_verdict"] = payload["stage42_fd_gate"]["verdict"]
    state["stage42_fd_safety_aware_joint_objective_training"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_fd_gate"]["verdict"],
        "gates": f"{payload['stage42_fd_gate']['passed']}/{payload['stage42_fd_gate']['total']}",
        "selected_objective": payload["model"]["selected"],
        "test_metric_vs_floor": payload["model"]["metrics"]["protected_selected_candidate"],
        "comparison_to_prior": payload["comparison_to_prior"],
        "deployment_decision": payload["deployment_decision"],
        "claim_boundary": payload["claim_boundary"],
        "conclusion": "Stage42-FD tests whether safety-teacher regularization inside the supervised objective can break the Stage42-FC all/hard vs proximity tradeoff.",
    }
    block = state.get("m3w_work_attempts_failures_successes_readme")
    if isinstance(block, dict):
        evidence = list(block.get("latest_included_evidence", []))
        if "Stage42-FD safety-aware joint objective training" not in evidence:
            evidence.append("Stage42-FD safety-aware joint objective training")
        block["latest_included_evidence"] = evidence
        block["source"] = "cached_verified_summary_from_stage18_to_stage42_reports_plus_stage42_es_to_fd_fresh_audits"
        block[
            "latest_conclusion"
        ] = "Stage42-FD follows FC by adding FA safety-teacher regularization inside the supervised objective, testing whether all/hard gains and proximity safety can be achieved together without changing raw-frame/dataset-local boundaries."
        state["m3w_work_attempts_failures_successes_readme"] = block
    write_json(RESEARCH_STATE, state)


def run_stage42_safety_aware_joint_objective_training() -> dict[str, Any]:
    payload = _build_payload()
    _write_reports(payload)
    _append_to_docs(payload)
    _update_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_safety_aware_joint_objective_training()
    gate = result["stage42_fd_gate"]
    print(f"Stage42-FD gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
