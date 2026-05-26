from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_source_level_full_waypoint_eval as am
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage42_calibrated_subset_eval as bo


OUT_DIR = Path("outputs/stage42_long_research")
BO_JSON = OUT_DIR / "calibrated_subset_eval_stage42.json"
REPORT_JSON = OUT_DIR / "calibrated_subset_safety_repair_stage42.json"
REPORT_MD = OUT_DIR / "calibrated_subset_safety_repair_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_bp_gate.md"

EPS = 1e-6
LAMBDAS = am.LAMBDAS

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-BP 修复 Stage42-BO calibrated-subset source-CV 的 easy/t50 failure，不使用 test 调阈值。",
    "本步骤只允许 train+val support source/source-family guard，holdout source 只最终评估一次。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _source_family(rel_source: np.ndarray) -> np.ndarray:
    """Map relative source paths into coarse causal metadata families.

    The family is source metadata available at inference time. It is not a
    future endpoint, future waypoint, central velocity, or test endpoint goal.
    Its purpose is to avoid applying high-switch validation policies to a held
    out source family that was never represented in train/val support.
    """

    families: list[str] = []
    for raw in rel_source.astype(str):
        parts = raw.split("/")
        if len(parts) >= 2 and parts[0] in {"ETH", "UCY"}:
            scene = parts[1]
            if scene.startswith("zara"):
                families.append("UCY_zara")
            elif scene.startswith("students"):
                families.append("UCY_students")
            elif scene.startswith("seq_"):
                families.append("ETH_seq")
            else:
                families.append(f"{parts[0]}_{scene}")
        else:
            families.append(parts[0] if parts else raw)
    return np.asarray(families, dtype="U64")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _source_guard_ok(
    selected: np.ndarray,
    floor: np.ndarray,
    data: Mapping[str, np.ndarray],
    switch: np.ndarray,
    support_mask: np.ndarray,
    rel_source: np.ndarray,
) -> bool:
    if not np.any(support_mask):
        return False
    global_metric = am._metric(selected, floor, data, switch, support_mask)
    if global_metric["all_improvement"] < 0.0 or global_metric["easy_degradation"] > 0.015 or global_metric["harm_over_fallback"] > 0.0:
        return False
    h = data["horizon"].astype(int)
    for src in sorted(set(rel_source[support_mask].tolist())):
        sm = support_mask & (rel_source == src)
        if int(np.sum(sm)) < 80:
            continue
        metric = am._metric(selected, floor, data, switch, sm)
        if metric["all_improvement"] < 0.0 or metric["easy_degradation"] > 0.02 or metric["harm_over_fallback"] > 0.0:
            return False
        if np.sum(sm & (h == 50)) >= 80 and metric["t50_improvement"] < -0.005:
            return False
    return True


def _select_policy_source_robust(
    pred_xy: np.ndarray,
    floor_xy: np.ndarray,
    labels: Mapping[str, np.ndarray],
    data: Mapping[str, np.ndarray],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    rel_source: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    pred_ade, pred_fde = am._trajectory_errors(pred_xy, labels)
    floor_ade, floor_fde = am._trajectory_errors(floor_xy, labels)
    residual_norm = np.linalg.norm(pred_xy[:, -1] - floor_xy[:, -1], axis=1) / np.maximum(data["scale"].astype(np.float64), EPS)
    domain = data["dataset"].astype(str)
    horizon = data["horizon"].astype(int)
    source_family = _source_family(rel_source)
    selected_ade = floor_ade.copy()
    selected_fde = floor_fde.copy()
    switch = np.zeros(len(floor_ade), dtype=bool)
    blended_cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    policy: dict[str, Any] = {
        "type": "stage42bp_source_robust_calibrated_subset_policy",
        "selection_source": "train_val_support_guard_only",
        "test_threshold_tuning": False,
        "slices": {},
    }
    for d in sorted(set(domain[val_mask | train_mask].tolist())):
        for h_val in [10, 25, 50, 100]:
            vm = val_mask & (domain == d) & (horizon == h_val)
            support = (train_mask | val_mask) & (domain == d) & (horizon == h_val)
            if int(np.sum(vm)) < 80 or int(np.sum(support)) < 300:
                continue
            thresholds = [float(np.quantile(residual_norm[support], q)) for q in [0.03, 0.05, 0.10, 0.20, 0.35, 0.50]]
            best: dict[str, Any] | None = None
            best_score = 0.0
            for direction in ["low", "high"]:
                for threshold in thresholds:
                    gate = (residual_norm <= threshold) if direction == "low" else (residual_norm >= threshold)
                    for alpha in [0.10, 0.20, 0.35, 0.50, 0.75, 1.0]:
                        if alpha not in blended_cache:
                            blended = floor_xy + float(alpha) * (pred_xy - floor_xy)
                            b_ade, b_fde = am._trajectory_errors(blended, labels)
                            blended_cache[float(alpha)] = (blended, b_ade, b_fde)
                        _, b_ade, b_fde = blended_cache[float(alpha)]
                        trial_ade = floor_ade.copy()
                        trial_fde = floor_fde.copy()
                        local = (domain == d) & (horizon == h_val) & gate
                        if not np.any(local & vm):
                            continue
                        trial_ade[local] = b_ade[local]
                        trial_fde[local] = b_fde[local]
                        trial_switch = local.astype(bool)
                        if not _source_guard_ok(trial_ade, floor_ade, data, trial_switch, support, rel_source):
                            continue
                        val_metric = am._metric(trial_ade, floor_ade, data, trial_switch, vm)
                        if val_metric["all_improvement"] < 0.0 or val_metric["easy_degradation"] > 0.02 or val_metric["harm_over_fallback"] > 0.0:
                            continue
                        score = (
                            1.4 * val_metric["all_improvement"]
                            + 1.8 * val_metric["t50_improvement"]
                            + 1.0 * val_metric["hard_failure_improvement"]
                            - 0.05 * val_metric["switch_rate"]
                        )
                        if score > best_score:
                            best_score = float(score)
                            best = {
                                "direction": direction,
                                "residual_norm_threshold": float(threshold),
                                "alpha": float(alpha),
                                "val_score": float(score),
                                "val_metric": val_metric,
                                "support_sources": sorted(set(rel_source[support].tolist())),
                                "support_source_families": sorted(set(source_family[support].tolist())),
                            }
            if best is not None and best_score > 0.0:
                policy["slices"][f"{d}|{h_val}"] = best
    for key, params in policy["slices"].items():
        d, h_s = key.split("|")
        h_val = int(h_s)
        gate = (
            residual_norm <= float(params["residual_norm_threshold"])
            if params["direction"] == "low"
            else residual_norm >= float(params["residual_norm_threshold"])
        )
        local = (domain == d) & (horizon == h_val) & gate
        allowed_families = set(params.get("support_source_families", []))
        if allowed_families:
            local = local & np.isin(source_family, list(allowed_families))
        alpha = float(params["alpha"])
        if alpha not in blended_cache:
            blended = floor_xy + alpha * (pred_xy - floor_xy)
            b_ade, b_fde = am._trajectory_errors(blended, labels)
            blended_cache[alpha] = (blended, b_ade, b_fde)
        _, b_ade, b_fde = blended_cache[alpha]
        selected_ade[local] = b_ade[local]
        selected_fde[local] = b_fde[local]
        switch[local] = True
    return policy, selected_ade, selected_fde, switch


def _evaluate_fold_repair(data: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray], rel_source: np.ndarray, fold: Mapping[str, Any]) -> dict[str, Any]:
    split = bo._split_for_fold(rel_source, fold)
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    floor = am._floor_arrays(data, train_mask)
    features, feature_names = am._feature_matrix(data, floor)
    x, _, _ = am._standardize(features, train_mask)
    target_delta = (
        (labels["waypoint_xy"].astype(np.float64) - np.stack([data["current_x"], data["current_y"]], axis=1)[:, None, :])
        / np.maximum(data["scale"].astype(np.float64)[:, None, None], EPS)
    ).astype(np.float32)
    best: dict[str, Any] | None = None
    best_score = -1e9
    floor_ade, floor_fde = am._trajectory_errors(floor["floor_xy"], labels)
    for lam in LAMBDAS:
        coef = am._fit_ridge_model(x, target_delta, labels["waypoint_valid"], train_mask, lam)
        pred_xy = am._predict_waypoints(x, coef, data)
        policy, selected_ade, selected_fde, switch = _select_policy_source_robust(
            pred_xy, floor["floor_xy"], labels, data, train_mask, val_mask, rel_source
        )
        val_metric = am._metric(selected_ade, floor_ade, data, switch, val_mask)
        score = (
            1.5 * val_metric["all_improvement"]
            + 1.8 * val_metric["t50_improvement"]
            + 1.0 * val_metric["hard_failure_improvement"]
            - 50.0 * max(0.0, val_metric["easy_degradation"] - 0.02)
            - 0.05 * val_metric["switch_rate"]
        )
        if score > best_score:
            best_score = float(score)
            best = {
                "lambda": float(lam),
                "policy": policy,
                "selected_ade": selected_ade,
                "selected_fde": selected_fde,
                "switch": switch,
                "val_metric": val_metric,
            }
    if best is None:
        raise RuntimeError("No calibrated subset repair model evaluated.")
    h = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    return {
        "source": "fresh_calibrated_subset_safety_repair_fold",
        "fold": fold,
        "fold_stats": bo._fold_stats(data, rel_source, split),
        "feature_count": len(feature_names),
        "selected_lambda": best["lambda"],
        "policy_slice_count": len(best["policy"]["slices"]),
        "validation_metric": best["val_metric"],
        "protected_ade": am._metric(best["selected_ade"], floor_ade, data, best["switch"], test_mask),
        "protected_fde": am._metric(best["selected_fde"], floor_fde, data, best["switch"], test_mask),
        "bootstrap": {
            "all": am._bootstrap_ci(best["selected_ade"], floor_ade, test_mask, seed=42101),
            "t50": am._bootstrap_ci(best["selected_ade"], floor_ade, test_mask & (h == 50), seed=42102),
            "hard_failure": am._bootstrap_ci(best["selected_ade"], floor_ade, test_mask & hard_failure, seed=42103),
            "easy_degradation": am._bootstrap_ci(floor_ade, best["selected_ade"], test_mask & easy, seed=42104),
        },
        "policy": best["policy"],
    }


def _aggregate(folds: list[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = [fold["protected_ade"] for fold in folds]
    def vals(key: str) -> list[float]:
        return [float(m[key]) for m in metrics]

    all_imp = vals("all_improvement")
    t50 = vals("t50_improvement")
    hard = vals("hard_failure_improvement")
    easy = vals("easy_degradation")
    t100 = vals("t100_raw_frame_diagnostic_improvement")
    return {
        "folds": len(folds),
        "rows_total": int(sum(int(m["rows"]) for m in metrics)),
        "all_improvement_macro_mean": float(np.mean(all_imp)),
        "all_improvement_min": float(np.min(all_imp)),
        "t50_improvement_macro_mean": float(np.mean(t50)),
        "t50_improvement_min": float(np.min(t50)),
        "t100_raw_frame_diagnostic_macro_mean": float(np.mean(t100)),
        "hard_failure_improvement_macro_mean": float(np.mean(hard)),
        "hard_failure_improvement_min": float(np.min(hard)),
        "easy_degradation_max": float(np.max(easy)),
        "easy_degradation_macro_mean": float(np.mean(easy)),
        "positive_all_folds": bool(all(v > 0.0 for v in all_imp)),
        "nonnegative_all_folds": bool(all(v >= 0.0 for v in all_imp)),
        "positive_t50_folds": bool(all(v > 0.0 for v in t50)),
        "easy_safe_all_folds": bool(all(v <= 0.02 for v in easy)),
        "positive_fold_count": int(sum(v > 0.0 for v in all_imp)),
        "positive_t50_fold_count": int(sum(v > 0.0 for v in t50)),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    claim = payload["claim_boundary"]
    no_leak = payload["no_leakage"]
    gates = {
        "bo_input_verified": payload["bo_verdict"] in {
            "stage42_bo_calibrated_subset_eval_partial",
            "stage42_bo_calibrated_subset_eval_pass_diagnostic_claim_limited",
            "stage42_bo_calibrated_subset_eval_pass_positive_claim_limited",
        },
        "repair_rerun_completed": summary["source_cv_folds"] == summary["calibrated_sources_evaluated"],
        "all_nonnegative": summary["nonnegative_all_folds"] is True,
        "easy_safe": summary["easy_safe_all_folds"] is True,
        "positive_support_exists": summary["positive_fold_count"] >= 2 or summary["positive_t50_fold_count"] >= 2,
        "no_future_inputs": no_leak["future_endpoint_input"] is False and no_leak["future_waypoint_input"] is False,
        "validation_only_selection": no_leak["test_threshold_tuning"] is False,
        "global_metric_blocked": claim["global_metric_claim_allowed"] is False,
        "global_seconds_blocked": claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    deployable = gates["all_nonnegative"] and gates["easy_safe"] and gates["positive_support_exists"]
    verdict = (
        "stage42_bp_calibrated_subset_safety_repair_pass_limited_positive"
        if passed == total and deployable
        else "stage42_bp_calibrated_subset_safety_repair_pass_safe_diagnostic"
        if passed == total
        else "stage42_bp_calibrated_subset_safety_repair_partial"
    )
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "limited_positive_transfer": bool(deployable), "verdict": verdict}


def run_stage42_calibrated_subset_safety_repair() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    bo_payload = _load_json(BO_JSON)
    data = s41._combined()
    labels = am._reconstruct_waypoint_labels(data)
    rel_source = bo._source_rel_array(data)
    source_ids = bo_payload["summary"]["calibrated_source_ids"]
    folds = bo._build_source_cv_folds(source_ids)
    fold_results = [_evaluate_fold_repair(data, labels, rel_source, fold) for fold in folds]
    aggregate = _aggregate(fold_results)
    no_leakage = {
        "future_endpoint_input": False,
        "future_waypoint_input": False,
        "future_waypoint_label_eval_only": True,
        "central_velocity": False,
        "test_endpoint_goals": False,
        "test_threshold_tuning": False,
        "train_only_feature_normalization": True,
        "source_overlap_pass": all(fold["fold_stats"]["source_overlap_pass"] for fold in fold_results),
    }
    claim_boundary = {
        "true_3d": False,
        "foundation_world_model": False,
        "source_specific_annotation_step_subset_claim_allowed": True,
        "global_metric_claim_allowed": False,
        "global_seconds_claim_allowed": False,
        "m3w_official_metric_seconds_claim_allowed": False,
        "raw_frame_dataset_local_global_claim_required": True,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    summary = {
        "source": "fresh_calibrated_subset_safety_repair",
        "calibrated_sources_evaluated": len(source_ids),
        "calibrated_source_ids": source_ids,
        "bo_previous_all_min": bo_payload["summary"]["all_improvement_min"],
        "bo_previous_t50_min": bo_payload["summary"]["t50_improvement_min"],
        "bo_previous_easy_max": bo_payload["summary"]["easy_degradation_max"],
        "source_cv_folds": len(fold_results),
        **aggregate,
        "global_metric_claim_allowed": False,
        "global_seconds_claim_allowed": False,
        "training_run": True,
        "auto_download_executed": False,
    }
    payload: dict[str, Any] = {
        "source": "fresh_calibrated_subset_safety_repair",
        "stage": "Stage42-BP calibrated subset safety repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(BO_JSON), "data/stage41_world_model/combined_external.npz"]),
        "current_facts": CURRENT_FACTS,
        "bo_verdict": bo_payload.get("stage42_bo_gate", {}).get("verdict"),
        "summary": summary,
        "fold_results": fold_results,
        "no_leakage": no_leakage,
        "claim_boundary": claim_boundary,
    }
    payload["stage42_bp_gate"] = _gate(payload)
    write_json(REPORT_JSON, am._jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    return payload


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-BP Calibrated-Subset Safety Repair",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_bp_gate']['passed']} / {payload['stage42_bp_gate']['total']}`",
        f"- verdict: `{payload['stage42_bp_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## What Failed In BO",
        "",
        f"- BO all_improvement_min: `{s['bo_previous_all_min']}`",
        f"- BO t50_improvement_min: `{s['bo_previous_t50_min']}`",
        f"- BO easy_degradation_max: `{s['bo_previous_easy_max']}`",
        "",
        "## Repair Summary",
        "",
        f"- source_cv_folds: `{s['source_cv_folds']}`",
        f"- all_improvement_macro_mean: `{s['all_improvement_macro_mean']}`",
        f"- all_improvement_min: `{s['all_improvement_min']}`",
        f"- t50_improvement_macro_mean: `{s['t50_improvement_macro_mean']}`",
        f"- t50_improvement_min: `{s['t50_improvement_min']}`",
        f"- hard_failure_improvement_macro_mean: `{s['hard_failure_improvement_macro_mean']}`",
        f"- easy_degradation_max: `{s['easy_degradation_max']}`",
        f"- positive_fold_count: `{s['positive_fold_count']}`",
        f"- positive_t50_fold_count: `{s['positive_t50_fold_count']}`",
        "",
        "## Fold Results",
        "",
        "| holdout | rows | all | t50 | t100 raw diag | hard/failure | easy degradation | switch | policy slices |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for fold in payload["fold_results"]:
        m = fold["protected_ade"]
        lines.append(
            f"| `{fold['fold']['holdout_source']}` | {m['rows']} | {m['all_improvement']:.6f} | {m['t50_improvement']:.6f} | {m['t100_raw_frame_diagnostic_improvement']:.6f} | {m['hard_failure_improvement']:.6f} | {m['easy_degradation']:.6f} | {m['switch_rate']:.6f} | {fold['policy_slice_count']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- BP adds train+val source-robust easy/harm guards to repair BO's calibrated-subset over-switching.",
            "- This remains a limited source-specific annotation-step calibrated subset result, not global metric/seconds-level M3W.",
            "- If a source still falls back to zero, it is treated as unsupported rather than overclaimed.",
            "",
            "## Claim Boundary",
            "",
            f"- no_leakage: `{payload['no_leakage']}`",
            f"- claim_boundary: `{payload['claim_boundary']}`",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_bp_gate"]
    lines = [
        "# Stage42-BP Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- limited_positive_transfer: `{gate['limited_positive_transfer']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | {bool(ok)} |")
    return lines


if __name__ == "__main__":
    run_stage42_calibrated_subset_safety_repair()
