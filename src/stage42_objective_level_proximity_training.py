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


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "objective_level_proximity_training_stage42.json"
REPORT_MD = OUT_DIR / "objective_level_proximity_training_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_fc_gate.md"

AM_JSON = OUT_DIR / "source_level_full_waypoint_eval_stage42.json"
DI_JSON = OUT_DIR / "group_consistency_full_waypoint_repair_stage42.json"
FB_JSON = OUT_DIR / "proximity_pareto_composer_stage42.json"

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

LAMBDAS = [0.05, 0.1, 1.0, 10.0, 100.0]
EPS = 1e-6
SOURCE = "fresh_stage42_objective_level_proximity_training"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-FC 接续 Stage42-FB：post-hoc DI/FA Pareto composer 能改善 proximity，但 all/hard 有小损失。",
    "本阶段把 proximity / future group interaction target 放入 supervised training objective，而不是只做 post-hoc repair。",
    "future waypoint labels 只用于 train loss weighting 和 eval label；它们不是 inference input。",
    "features 仍只来自当前/过去 history、baseline rollout context、graph/current neighbor summary、domain/horizon metadata。",
    "validation 选择 objective variant、lambda 和 safe deployment policy；test 只评一次。",
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


def _future_group_min_distance(
    labels: Mapping[str, np.ndarray],
    group_key: np.ndarray,
    normalizer: np.ndarray,
    agent_id: np.ndarray,
) -> np.ndarray:
    """Compute label-only group proximity for supervised train weighting.

    This uses future waypoints as the *loss target* only. It is never emitted as
    an inference feature and never used to choose test thresholds.
    """

    return di._min_group_distance_fast(
        labels["waypoint_xy"].astype(np.float32),
        group_key,
        np.maximum(normalizer.astype(np.float64), EPS),
        agent_id.astype(np.int64),
    )


def _objective_signals(
    data: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    graph: np.ndarray,
    group_key: np.ndarray,
    am_candidate: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    normalizer = np.maximum(data["scale"].astype(np.float64), EPS)
    agent = data["agent_id"].astype(np.int64)
    base_min = di._min_group_distance_fast(am_candidate["selected_xy"], group_key, normalizer, agent)
    floor_min = di._min_group_distance_fast(am_candidate["floor_xy"], group_key, normalizer, agent)
    future_min = _future_group_min_distance(labels, group_key, normalizer, agent)
    graph_sig = dh._graph_signals(graph)
    close_label = np.isfinite(future_min) & (future_min < 0.08)
    very_close_label = np.isfinite(future_min) & (future_min < 0.05)
    base_unsafe = np.isfinite(base_min) & np.isfinite(floor_min) & (base_min < floor_min) & (base_min < 0.08)
    base_close = np.isfinite(base_min) & (base_min < 0.08)
    risk = (
        3.0 * very_close_label.astype(np.float64)
        + 1.75 * close_label.astype(np.float64)
        + 2.0 * base_unsafe.astype(np.float64)
        + 0.75 * base_close.astype(np.float64)
        + 0.5 * graph_sig["close"].astype(np.float64)
    )
    return {
        "future_min_distance": future_min,
        "base_min_distance": base_min,
        "floor_min_distance": floor_min,
        "future_close_005": very_close_label.astype(np.float64),
        "future_close_008": close_label.astype(np.float64),
        "base_unsafe": base_unsafe.astype(np.float64),
        "base_close_008": base_close.astype(np.float64),
        "graph_close": graph_sig["close"].astype(np.float64),
        "risk": risk.astype(np.float64),
    }


def _objective_weights(
    data: Mapping[str, np.ndarray],
    signals: Mapping[str, np.ndarray],
    graph: np.ndarray,
    train_mask: np.ndarray,
    variant: str,
) -> np.ndarray:
    h = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    graph_sig = dh._graph_signals(graph)
    w = np.ones(len(h), dtype=np.float64)
    if variant == "label_proximity_objective":
        w += signals["risk"]
    elif variant == "label_proximity_hard_long_objective":
        w += signals["risk"] + 2.0 * hard_failure.astype(np.float64) + 1.5 * (h == 50) + 2.0 * (h == 100)
    elif variant == "label_proximity_easy_safe_objective":
        w += signals["risk"] + 1.5 * hard_failure.astype(np.float64)
        w[easy] *= 0.7
    elif variant == "graph_label_proximity_objective":
        w += signals["risk"] + graph_sig["density_r1"] + 0.5 * graph_sig["mean_closing"]
    elif variant == "balanced_accuracy_control":
        w += 0.5 * hard_failure.astype(np.float64) + 0.5 * (h == 50)
    else:
        raise ValueError(f"unknown Stage42-FC objective variant: {variant}")
    mean = float(np.mean(w[train_mask])) if np.any(train_mask) else 1.0
    return (w / max(mean, EPS)).astype(np.float64)


def _candidate_specs(
    data: Mapping[str, np.ndarray],
    base_features: np.ndarray,
    graph: np.ndarray,
    signals: Mapping[str, np.ndarray],
    train_mask: np.ndarray,
) -> list[dict[str, Any]]:
    graph_compact = graph[:, :10].astype(np.float32)
    specs: list[dict[str, Any]] = []
    for variant in [
        "balanced_accuracy_control",
        "label_proximity_objective",
        "label_proximity_hard_long_objective",
        "label_proximity_easy_safe_objective",
        "graph_label_proximity_objective",
    ]:
        features = base_features
        mode = "stage42_am_features"
        if variant == "graph_label_proximity_objective":
            features = np.concatenate([base_features, graph_compact], axis=1).astype(np.float32)
            mode = "stage42_am_features_plus_graph_summary"
        specs.append(
            {
                "variant": variant,
                "feature_mode": mode,
                "features": features,
                "weights": _objective_weights(data, signals, graph, train_mask, variant),
            }
        )
    return specs


def _near_diagnostics(xy: np.ndarray, data: Mapping[str, np.ndarray], ids: np.ndarray, group_key: np.ndarray) -> dict[str, Any]:
    ids = np.asarray(ids, dtype=np.int64)
    mind = di._min_group_distance_fast(
        xy[ids],
        group_key[ids],
        np.maximum(data["scale"][ids].astype(np.float64), EPS),
        data["agent_id"][ids].astype(np.int64),
    )
    finite = mind[np.isfinite(mind)]
    return {
        "near_005": float(np.mean(np.isfinite(mind) & (mind < 0.05))) if len(mind) else 0.0,
        "near_008": float(np.mean(np.isfinite(mind) & (mind < 0.08))) if len(mind) else 0.0,
        "p05_min_distance": float(np.percentile(finite, 5)) if len(finite) else None,
    }


def _selection_score(metric: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> float:
    return (
        1.30 * float(metric["all_improvement"])
        + 1.35 * float(metric["hard_failure_improvement"])
        + 1.10 * float(metric["t50_improvement"])
        + 0.45 * float(metric["t100_raw_frame_diagnostic_improvement"])
        - 35.0 * max(0.0, float(metric["easy_degradation"]) - 0.02)
        - 2.0 * float(diagnostics["near_005"])
        - 0.01 * float(metric["switch_rate"])
    )


def _delta(metric: Mapping[str, Any], ref: Mapping[str, Any]) -> dict[str, float | None]:
    if not ref:
        return {k: None for k in ["all_improvement", "t50_improvement", "t100_raw_frame_diagnostic_improvement", "hard_failure_improvement", "easy_degradation"]}
    return {
        "all_improvement": float(metric.get("all_improvement", 0.0)) - float(ref.get("all_improvement", 0.0)),
        "t50_improvement": float(metric.get("t50_improvement", 0.0)) - float(ref.get("t50_improvement", 0.0)),
        "t100_raw_frame_diagnostic_improvement": float(metric.get("t100_raw_frame_diagnostic_improvement", 0.0))
        - float(ref.get("t100_raw_frame_diagnostic_improvement", 0.0)),
        "hard_failure_improvement": float(metric.get("hard_failure_improvement", 0.0)) - float(ref.get("hard_failure_improvement", 0.0)),
        "easy_degradation": float(metric.get("easy_degradation", 0.0)) - float(ref.get("easy_degradation", 0.0)),
    }


def _evaluate_objective_candidates(
    data: Mapping[str, np.ndarray],
    split: np.ndarray,
    labels: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
    base_features: np.ndarray,
    graph: np.ndarray,
    signals: Mapping[str, np.ndarray],
    group_key: np.ndarray,
) -> dict[str, Any]:
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    val_ids = np.where(val_mask)[0]
    test_ids = np.where(test_mask)[0]
    current = np.stack([data["current_x"], data["current_y"]], axis=1).astype(np.float64)
    scale = np.maximum(data["scale"].astype(np.float64), EPS)
    target_delta = ((labels["waypoint_xy"].astype(np.float64) - current[:, None, :]) / scale[:, None, None]).astype(np.float32)
    floor_ade, floor_fde = am._trajectory_errors(floor["floor_xy"], labels)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_score = -1e9
    for spec in _candidate_specs(data, base_features, graph, signals, train_mask):
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
            selected_ade, selected_fde = am._trajectory_errors(selected_xy, labels)
            val_metric = am._metric(selected_ade, floor_ade, data, switch_xy, val_mask)
            val_diag = _near_diagnostics(selected_xy, data, val_ids, group_key)
            score = _selection_score(val_metric, val_diag)
            pred_ade, pred_fde = am._trajectory_errors(pred_xy, labels)
            row = {
                "variant": spec["variant"],
                "feature_mode": spec["feature_mode"],
                "lambda": float(lam),
                "val_score": float(score),
                "val_metric": val_metric,
                "val_near_diagnostics": val_diag,
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
        raise RuntimeError("No Stage42-FC objective-level candidate evaluated.")
    h = data["horizon"].astype(int)
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
        "selected": {k: v for k, v in best.items() if k not in {"coef", "pred_xy", "selected_xy", "switch", "selected_ade", "selected_fde", "pred_ade", "pred_fde", "floor_ade", "floor_fde", "policy"}},
        "policy": best["policy"],
        "metrics": {
            "protected_selected_candidate": metric,
            "ungated_selected_candidate": ungated_metric,
            "protected_selected_candidate_fde": am._metric(best["selected_fde"], best["floor_fde"], data, best["switch"], test_mask),
        },
        "diagnostics": {
            "protected_near": _near_diagnostics(selected_xy, data, test_ids, group_key),
            "ungated_near": _near_diagnostics(pred_xy, data, test_ids, group_key),
            "floor_near": _near_diagnostics(floor["floor_xy"], data, test_ids, group_key),
        },
        "bootstrap": {
            "all": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask, seed=43201),
            "t50": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & (h == 50), seed=43202),
            "t100_raw_frame_diagnostic": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & (h == 100), seed=43203),
            "hard_failure": am._bootstrap_ci(best["selected_ade"], best["floor_ade"], test_mask & hard_failure, seed=43204),
            "easy_degradation": am._bootstrap_ci(best["floor_ade"], best["selected_ade"], test_mask & easy, seed=43205),
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
    am_payload = read_json(AM_JSON, {})
    di_payload = read_json(DI_JSON, {})
    fb_payload = read_json(FB_JSON, {})
    am_metric = am_payload.get("model", {}).get("metrics", {}).get("protected_ridge_source_level", {})
    di_metric = di_payload.get("repair", {}).get("test", {}).get("metric_vs_floor", {})
    fb_metric = fb_payload.get("repair", {}).get("test", {}).get("metric_vs_floor", {})
    di_diag = di_payload.get("repair", {}).get("test", {}).get("diagnostics", {})
    fb_diag = fb_payload.get("repair", {}).get("test", {}).get("diagnostics", {})
    protected_near = diagnostics.get("protected_near", {})
    return {
        "stage42_am_metric": am_metric,
        "stage42_di_metric": di_metric,
        "stage42_fb_metric": fb_metric,
        "delta_vs_stage42_am": _delta(metric, am_metric),
        "delta_vs_stage42_di": _delta(metric, di_metric),
        "delta_vs_stage42_fb": _delta(metric, fb_metric),
        "near_delta_vs_stage42_di": float(protected_near.get("near_005", 0.0)) - float(di_diag.get("final_near_005", 0.0)),
        "near_delta_vs_stage42_fb": float(protected_near.get("near_005", 0.0)) - float(fb_diag.get("final_near_005", 0.0)),
        "stage42_di_near_005": di_diag.get("final_near_005"),
        "stage42_fb_near_005": fb_diag.get("final_near_005"),
    }


def _deployment_decision(metric: Mapping[str, Any], comparison: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    delta_di = comparison.get("delta_vs_stage42_di", {})
    delta_fb = comparison.get("delta_vs_stage42_fb", {})
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
        and (near_di is not None and float(near_di) <= 0.0)
    )
    return {
        "promote_objective_level_training": bool(promotable),
        "diagnostic_positive": bool(positive),
        "decision": "promote_stage42_fc_objective_level_training"
        if promotable
        else "objective_level_training_not_enough_keep_stage42_di_or_cq_floor",
        "reason": "Promotion requires positive all+hard, easy safe, no all/hard loss vs Stage42-DI/FB, and no worse near@0.05 than Stage42-DI.",
        "protected_near_005": diagnostics.get("protected_near", {}).get("near_005"),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metric = payload.get("model", {}).get("metrics", {}).get("protected_selected_candidate", {})
    comparison = payload.get("comparison_to_prior", {})
    delta_di = comparison.get("delta_vs_stage42_di", {})
    delta_fb = comparison.get("delta_vs_stage42_fb", {})
    no_leak = payload.get("no_leakage", {})
    boundary = payload.get("claim_boundary", {})
    gates = {
        "objective_training_source_fresh": payload.get("source") == SOURCE,
        "test_rows_present": payload.get("split_stats", {}).get("by_split", {}).get("test", {}).get("rows", 0) > 0,
        "full_waypoint_labels_present": payload.get("label_stats", {}).get("test_full_waypoint_rows", 0) > 0,
        "objective_label_weights_built": payload.get("objective_signal_stats", {}).get("train_future_close008_rate") is not None,
        "candidate_grid_nonempty": payload.get("model", {}).get("candidate_count", 0) >= 10,
        "validation_selected": payload.get("model", {}).get("selected", {}).get("val_score") is not None,
        "positive_all": float(metric.get("all_improvement", 0.0)) > 0.0,
        "positive_hard": float(metric.get("hard_failure_improvement", 0.0)) > 0.0,
        "easy_safe": float(metric.get("easy_degradation", 1.0)) <= 0.02,
        "beats_stage42_di_all": float(delta_di.get("all_improvement") or 0.0) >= 0.0,
        "beats_stage42_di_hard": float(delta_di.get("hard_failure_improvement") or 0.0) >= 0.0,
        "beats_stage42_fb_all": float(delta_fb.get("all_improvement") or 0.0) >= 0.0,
        "near_not_worse_than_stage42_di": (comparison.get("near_delta_vs_stage42_di") is not None and float(comparison["near_delta_vs_stage42_di"]) <= 0.0),
        "future_labels_loss_only": no_leak.get("future_waypoint_labels_loss_only") is True,
        "no_future_inference_input": no_leak.get("future_endpoint_input") is False and no_leak.get("future_waypoint_input") is False,
        "no_central_velocity": no_leak.get("central_velocity") is False,
        "no_test_endpoint_goals": no_leak.get("test_endpoint_goals") is False,
        "no_test_threshold_tuning": no_leak.get("test_threshold_tuning") is False,
        "train_only_feature_normalization": no_leak.get("train_only_feature_normalization") is True,
        "validation_only_model_selection": no_leak.get("validation_only_model_selection") is True,
        "no_metric_seconds_claim": boundary.get("global_metric_claim_allowed") is False and boundary.get("global_seconds_claim_allowed") is False,
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
        and gates["beats_stage42_fb_all"]
        and gates["near_not_worse_than_stage42_di"]
    )
    verdict = (
        "stage42_fc_objective_level_proximity_training_pass_promotable"
        if passed == total and promotable
        else "stage42_fc_objective_level_proximity_training_positive_not_promoted"
        if gates["positive_all"] and gates["positive_hard"] and gates["easy_safe"]
        else "stage42_fc_objective_level_proximity_training_fail"
    )
    return {"passed": passed, "total": total, "gates": gates, "verdict": verdict}


def _signal_stats(signals: Mapping[str, np.ndarray], split: np.ndarray) -> dict[str, Any]:
    train = split == "train"
    val = split == "val"
    test = split == "test"

    def rate(name: str, mask: np.ndarray) -> float:
        return float(np.mean(signals[name][mask])) if np.any(mask) else 0.0

    return {
        "train_future_close005_rate": rate("future_close_005", train),
        "train_future_close008_rate": rate("future_close_008", train),
        "train_base_unsafe_rate": rate("base_unsafe", train),
        "val_future_close008_rate": rate("future_close_008", val),
        "test_future_close008_rate": rate("future_close_008", test),
        "mean_train_objective_risk": float(np.mean(signals["risk"][train])) if np.any(train) else 0.0,
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
    am_candidate = di._rebuild_stage42_am_candidate(data, split, labels, floor)
    am_candidate["floor_xy"] = floor["floor_xy"]
    group_key = di._group_key(data)
    signals = _objective_signals(data, labels, graph, group_key, am_candidate)
    model = _evaluate_objective_candidates(data, split, labels, floor, features, graph, signals, group_key)
    metric = model["metrics"]["protected_selected_candidate"]
    comparison = _comparison_to_prior(metric, model["diagnostics"])
    decision = _deployment_decision(metric, comparison, model["diagnostics"])
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-FC objective-level proximity / group-interaction training",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                str(AM_JSON),
                str(DI_JSON),
                str(FB_JSON),
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
            "objective_variants": [
                "balanced_accuracy_control",
                "label_proximity_objective",
                "label_proximity_hard_long_objective",
                "label_proximity_easy_safe_objective",
                "graph_label_proximity_objective",
            ],
            "future_label_weighting_train_loss_only": True,
        },
        "graph_stats": graph_stats,
        "objective_signal_stats": _signal_stats(signals, split),
        "model": model,
        "comparison_to_prior": comparison,
        "deployment_decision": decision,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_labels_loss_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_feature_normalization": True,
            "validation_only_model_selection": True,
            "source_overlap_pass": bool(split_stats.get("source_overlap_pass", False)),
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["stage42_fc_gate"] = _gate(payload)
    return _jsonable(payload)


def _write_reports(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, payload)
    metric = payload["model"]["metrics"]["protected_selected_candidate"]
    ungated = payload["model"]["metrics"]["ungated_selected_candidate"]
    selected = payload["model"]["selected"]
    comparison = payload["comparison_to_prior"]
    diagnostics = payload["model"]["diagnostics"]
    gate = payload["stage42_fc_gate"]
    lines = [
        "# Stage42-FC Objective-Level Proximity Training",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- selected objective: `{selected['variant']}`",
        f"- feature mode: `{selected['feature_mode']}`",
        f"- lambda: `{selected['lambda']}`",
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
        "",
        "## Comparison To Prior",
        "",
        "| reference | all | t50 | t100 raw | hard/failure | easy |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ["delta_vs_stage42_am", "delta_vs_stage42_di", "delta_vs_stage42_fb"]:
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
            f"- promote objective-level training: `{payload['deployment_decision']['promote_objective_level_training']}`",
            f"- diagnostic positive: `{payload['deployment_decision']['diagnostic_positive']}`",
            f"- decision: `{payload['deployment_decision']['decision']}`",
            f"- reason: {payload['deployment_decision']['reason']}",
            "",
            "## No Leakage / Claim Boundary",
            "",
            "- future labels are used only for supervised train loss weighting and evaluation labels, not inference features.",
            "- no central velocity, no test endpoint goals, no test threshold tuning.",
            "- dataset-local/raw-frame 2.5D only; no metric/seconds claim.",
            "- Stage5C not executed; SMC not enabled.",
        ]
    )
    write_md(REPORT_MD, lines)
    gate_lines = ["# Stage42-FC Gates", "", f"Verdict: `{gate['verdict']}`", f"Passed: `{gate['passed']} / {gate['total']}`", ""]
    for key, value in gate["gates"].items():
        gate_lines.append(f"- `{key}`: `{value}`")
    write_md(GATE_MD, gate_lines)


def _summary_section(payload: Mapping[str, Any]) -> str:
    metric = payload["model"]["metrics"]["protected_selected_candidate"]
    selected = payload["model"]["selected"]
    comparison = payload["comparison_to_prior"]
    return "\n".join(
        [
            "<!-- STAGE42_FC_OBJECTIVE_LEVEL_PROXIMITY_TRAINING:START -->",
            "## Stage42-FC Objective-Level Proximity Training",
            "",
            f"- source: `{payload['source']}`",
            "- role: moves proximity/group-interaction signal from post-hoc repair into supervised full-waypoint training objective.",
            f"- selected objective: `{selected['variant']}`; feature mode `{selected['feature_mode']}`; lambda `{selected['lambda']}`.",
            f"- gate: `{payload['stage42_fc_gate']['passed']} / {payload['stage42_fc_gate']['total']}`; verdict `{payload['stage42_fc_gate']['verdict']}`.",
            f"- test all/t50/t100raw/hard/easy: `{_pct(metric['all_improvement'])}` / `{_pct(metric['t50_improvement'])}` / `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(metric['hard_failure_improvement'])}` / `{_pct(metric['easy_degradation'])}`.",
            f"- delta vs Stage42-DI all/hard/near005: `{_pct(comparison['delta_vs_stage42_di']['all_improvement'])}` / `{_pct(comparison['delta_vs_stage42_di']['hard_failure_improvement'])}` / `{_pct(comparison['near_delta_vs_stage42_di'])}`.",
            f"- decision: `{payload['deployment_decision']['decision']}`.",
            "- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
            "<!-- STAGE42_FC_OBJECTIVE_LEVEL_PROXIMITY_TRAINING:END -->",
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
        new = _replace_text_section(old, "STAGE42_FC_OBJECTIVE_LEVEL_PROXIMITY_TRAINING", section)
        path.write_text(new)


def _update_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-FC objective-level proximity training"
    state["current_verdict"] = payload["stage42_fc_gate"]["verdict"]
    state["stage42_fc_objective_level_proximity_training"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_fc_gate"]["verdict"],
        "gates": f"{payload['stage42_fc_gate']['passed']}/{payload['stage42_fc_gate']['total']}",
        "selected_objective": payload["model"]["selected"],
        "test_metric_vs_floor": payload["model"]["metrics"]["protected_selected_candidate"],
        "comparison_to_prior": payload["comparison_to_prior"],
        "deployment_decision": payload["deployment_decision"],
        "claim_boundary": payload["claim_boundary"],
        "conclusion": "Stage42-FC tests whether proximity/group-interaction should move into the training objective after Stage42-FB exposed a post-hoc Pareto boundary.",
    }
    block = state.get("m3w_work_attempts_failures_successes_readme")
    if isinstance(block, dict):
        evidence = list(block.get("latest_included_evidence", []))
        if "Stage42-FC objective-level proximity training" not in evidence:
            evidence.append("Stage42-FC objective-level proximity training")
        block["latest_included_evidence"] = evidence
        block["source"] = "cached_verified_summary_from_stage18_to_stage42_reports_plus_stage42_es_to_fc_fresh_audits"
        block[
            "latest_conclusion"
        ] = "Stage42-FB showed DI/FA post-hoc composition is Pareto-bounded. Stage42-FC moves proximity/group-interaction into supervised training objective and reports whether this breaks the all/hard/proximity tradeoff without changing metric/seconds or Stage5C/SMC boundaries."
        state["m3w_work_attempts_failures_successes_readme"] = block
    write_json(RESEARCH_STATE, state)


def run_stage42_objective_level_proximity_training() -> dict[str, Any]:
    payload = _build_payload()
    _write_reports(payload)
    _append_to_docs(payload)
    _update_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_objective_level_proximity_training()
    gate = result["stage42_fc_gate"]
    print(f"Stage42-FC gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
