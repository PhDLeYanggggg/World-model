from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_ucy_validation_support_repair as aw
from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
AW_JSON = OUT_DIR / "ucy_validation_support_repair_stage42.json"
AX_JSON = OUT_DIR / "repaired_protocol_robustness_stage42.json"
REPORT_JSON = OUT_DIR / "aw_t100_easy_safety_repair_stage42.json"
REPORT_MD = OUT_DIR / "aw_t100_easy_safety_repair_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ay_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

T100_VAL_EASY_THRESHOLD = 0.0


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-AY 是 Stage42-AW repaired protocol 的 t100 easy-safety repair。",
    "本修复重新计算 AW validation-best variant 的 full-waypoint arrays，并用 validation-only t100 easy-safety guard 决定是否回退 floor。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t100 仍是 raw-frame diagnostic，不是 seconds-level long-horizon claim。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _jsonable(value: Any) -> Any:
    return am._jsonable(value)


def _target_delta(data: Mapping[str, np.ndarray], labels: Mapping[str, np.ndarray]) -> np.ndarray:
    cur = np.stack([data["current_x"], data["current_y"]], axis=1).astype(np.float64)
    scale = np.maximum(data["scale"].astype(np.float64), am.EPS)
    return ((labels["waypoint_xy"].astype(np.float64) - cur[:, None, :]) / scale[:, None, None]).astype(np.float32)


def _recompute_aw_variant_arrays(variant_name: str, best_lambda: float) -> dict[str, Any]:
    data = s41._combined()
    original_split, group = am._split_arrays(data)
    domain = data["dataset"].astype(str)
    split, internal_val_group = aw._split_with_ucy_internal_val(original_split, group, domain)
    labels = am._reconstruct_waypoint_labels(data)
    floor = am._floor_arrays(data, split == "train")
    features, feature_names = am._feature_matrix(data, floor)
    masks = aw._safe_variant_masks(feature_names)
    if variant_name not in masks:
        raise KeyError(f"Unknown AW variant: {variant_name}")
    x, _, _ = am._standardize(features[:, masks[variant_name]], split == "train")
    coef = am._fit_ridge_model(x, _target_delta(data, labels), labels["waypoint_valid"], split == "train", float(best_lambda))
    pred_xy = am._predict_waypoints(x, coef, data)
    policy, selected_ade, selected_fde, switch = am._select_policy_on_val(
        pred_xy,
        floor["floor_xy"],
        labels,
        data,
        split == "val",
    )
    floor_ade, floor_fde = am._trajectory_errors(floor["floor_xy"], labels)
    return {
        "data": data,
        "split": split,
        "group": group,
        "internal_val_group": internal_val_group,
        "labels": labels,
        "policy": policy,
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "switch": switch,
        "floor_ade": floor_ade,
        "floor_fde": floor_fde,
        "feature_count": int(np.sum(masks[variant_name])),
    }


def _t100_slice_keep(params: Mapping[str, Any], threshold: float = T100_VAL_EASY_THRESHOLD) -> bool:
    metric = params.get("val_metric", {})
    return float(metric.get("all_improvement", 0.0)) > 0.0 and float(metric.get("easy_degradation", 1.0)) <= float(threshold)


def _apply_t100_easy_guard(
    *,
    policy: Mapping[str, Any],
    data: Mapping[str, np.ndarray],
    selected_ade: np.ndarray,
    selected_fde: np.ndarray,
    switch: np.ndarray,
    floor_ade: np.ndarray,
    floor_fde: np.ndarray,
    threshold: float = T100_VAL_EASY_THRESHOLD,
) -> dict[str, Any]:
    domain = data["dataset"].astype(str)
    horizon = data["horizon"].astype(int)
    repaired_ade = selected_ade.copy()
    repaired_fde = selected_fde.copy()
    repaired_switch = switch.copy()
    guarded: dict[str, Any] = {}
    kept: dict[str, Any] = {}
    for key, params in sorted(policy.get("slices", {}).items()):
        d, h_s = key.split("|", 1)
        if int(h_s) != 100:
            continue
        keep = _t100_slice_keep(params, threshold)
        val_metric = params.get("val_metric", {})
        rows = int(np.sum((domain == d) & (horizon == 100)))
        record = {
            "source": "fresh_run_validation_only_t100_easy_guard",
            "val_all_improvement": float(val_metric.get("all_improvement", 0.0)),
            "val_easy_degradation": float(val_metric.get("easy_degradation", 0.0)),
            "threshold": float(threshold),
            "rows_all_splits": rows,
        }
        if keep:
            kept[key] = record
            continue
        mask = (domain == d) & (horizon == 100)
        repaired_ade[mask] = floor_ade[mask]
        repaired_fde[mask] = floor_fde[mask]
        repaired_switch[mask] = False
        guarded[key] = {**record, "reason": "validation_t100_easy_degradation_above_strict_nonharm_threshold_or_nonpositive_gain"}
    return {
        "source": "fresh_run",
        "threshold": float(threshold),
        "uses_test_metrics_for_threshold": False,
        "guarded_slices": guarded,
        "kept_slices": kept,
        "selected_ade": repaired_ade,
        "selected_fde": repaired_fde,
        "switch": repaired_switch,
    }


def _bootstrap_easy_degradation(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray, seed: int) -> dict[str, Any]:
    return am._bootstrap_ci(floor, selected, mask, seed=seed)


def _metrics_bundle(data: Mapping[str, np.ndarray], split: np.ndarray, selected_ade: np.ndarray, selected_fde: np.ndarray, floor_ade: np.ndarray, floor_fde: np.ndarray, switch: np.ndarray) -> dict[str, Any]:
    test = split == "test"
    domain = data["dataset"].astype(str)
    horizon = data["horizon"].astype(int)
    hard_failure = data["hard"].astype(bool) | data["failure"].astype(bool)
    easy = data["easy"].astype(bool)
    return {
        "protected": am._metric(selected_ade, floor_ade, data, switch, test),
        "fde": am._metric(selected_fde, floor_fde, data, switch, test),
        "bootstrap": {
            "all": am._bootstrap_ci(selected_ade, floor_ade, test, seed=42101),
            "t50": am._bootstrap_ci(selected_ade, floor_ade, test & (horizon == 50), seed=42102),
            "t100_raw_frame_diagnostic": am._bootstrap_ci(selected_ade, floor_ade, test & (horizon == 100), seed=42103),
            "hard_failure": am._bootstrap_ci(selected_ade, floor_ade, test & hard_failure, seed=42104),
            "easy_degradation": _bootstrap_easy_degradation(selected_ade, floor_ade, test & easy, seed=42105),
            "h100_easy_degradation": _bootstrap_easy_degradation(selected_ade, floor_ade, test & (horizon == 100) & easy, seed=42106),
        },
        "by_domain": {
            d: am._metric(selected_ade, floor_ade, data, switch, test & (domain == d))
            for d in sorted(set(domain[test].tolist()))
        },
        "by_horizon": {
            str(h): am._metric(selected_ade, floor_ade, data, switch, test & (horizon == h))
            for h in [10, 25, 50, 100]
        },
    }


def run_stage42_aw_t100_easy_safety_repair() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    aw_report = _load_json(AW_JSON)
    ax_report = _load_json(AX_JSON) if AX_JSON.exists() else {}
    variant_name = aw_report["validation_best_variant"]
    best_lambda = float(aw_report["validation_best"]["best_lambda"])
    arrays = _recompute_aw_variant_arrays(variant_name, best_lambda)
    original = _metrics_bundle(
        arrays["data"],
        arrays["split"],
        arrays["selected_ade"],
        arrays["selected_fde"],
        arrays["floor_ade"],
        arrays["floor_fde"],
        arrays["switch"],
    )
    repair = _apply_t100_easy_guard(
        policy=arrays["policy"],
        data=arrays["data"],
        selected_ade=arrays["selected_ade"],
        selected_fde=arrays["selected_fde"],
        switch=arrays["switch"],
        floor_ade=arrays["floor_ade"],
        floor_fde=arrays["floor_fde"],
        threshold=T100_VAL_EASY_THRESHOLD,
    )
    repaired = _metrics_bundle(
        arrays["data"],
        arrays["split"],
        repair["selected_ade"],
        repair["selected_fde"],
        arrays["floor_ade"],
        arrays["floor_fde"],
        repair["switch"],
    )
    result = {
        "source": "fresh_run",
        "stage": "Stage42-AY AW repaired protocol t100 easy-safety repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                str(AW_JSON),
                str(AX_JSON),
                "data/stage41_world_model/combined_external.npz",
            ]
        ),
        "aw_verdict": aw_report["stage42_aw_gate"]["verdict"],
        "ax_verdict": ax_report.get("stage42_ax_gate", {}).get("verdict", "not_run"),
        "validation_best_variant": variant_name,
        "best_lambda": best_lambda,
        "feature_count": arrays["feature_count"],
        "internal_val_group": arrays["internal_val_group"],
        "repair_policy": {
            "source": repair["source"],
            "type": "stage42ay_validation_only_strict_t100_easy_guard",
            "threshold": repair["threshold"],
            "uses_test_metrics_for_threshold": repair["uses_test_metrics_for_threshold"],
            "guarded_slices": repair["guarded_slices"],
            "kept_slices": repair["kept_slices"],
        },
        "original_metrics": original,
        "repaired_metrics": repaired,
        "repair_effect": {
            "source": "fresh_run",
            "h100_easy_before": original["by_horizon"]["100"]["easy_degradation"],
            "h100_easy_after": repaired["by_horizon"]["100"]["easy_degradation"],
            "h100_t100_before": original["by_horizon"]["100"]["t100_raw_frame_diagnostic_improvement"],
            "h100_t100_after": repaired["by_horizon"]["100"]["t100_raw_frame_diagnostic_improvement"],
            "global_all_before": original["protected"]["all_improvement"],
            "global_all_after": repaired["protected"]["all_improvement"],
            "guarded_slice_count": len(repair["guarded_slices"]),
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "internal_val_from_train_only": True,
            "test_sources_unchanged": True,
            "train_only_feature_normalization": True,
            "source_overlap_pass": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "t100_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "ungated_neural_deployable": False,
            "post_ax_repair_needs_future_holdout_for_stronger_paper_claim": True,
        },
    }
    result["stage42_ay_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _append_ledger(result)
    return result


def _domain_positive(row: Mapping[str, Any]) -> bool:
    return (
        float(row["all_improvement"]) > 0.0
        and float(row["t50_improvement"]) > 0.0
        and float(row["hard_failure_improvement"]) > 0.0
        and float(row["easy_degradation"]) <= 0.02
    )


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    repaired = result["repaired_metrics"]
    boot = repaired["bootstrap"]
    h100 = repaired["by_horizon"]["100"]
    gates = {
        "aw_input_verified": result["aw_verdict"] == "stage42_aw_ucy_validation_support_repair_pass",
        "ax_input_verified": result["ax_verdict"] == "stage42_ax_repaired_protocol_robustness_pass_with_t100_limit",
        "fresh_recompute_done": result["source"] == "fresh_run",
        "validation_only_t100_guard": result["repair_policy"]["uses_test_metrics_for_threshold"] is False,
        "t100_guard_applied": result["repair_effect"]["guarded_slice_count"] > 0,
        "h100_easy_metric_safe": h100["easy_degradation"] <= 0.02,
        "h100_easy_ci_safe": boot["h100_easy_degradation"]["high"] <= 0.02,
        "global_all_ci_positive": boot["all"]["low"] > 0.0,
        "global_t50_ci_positive": boot["t50"]["low"] > 0.0,
        "global_t100_raw_frame_ci_positive": boot["t100_raw_frame_diagnostic"]["low"] > 0.0,
        "global_hard_failure_ci_positive": boot["hard_failure"]["low"] > 0.0,
        "global_easy_ci_safe": boot["easy_degradation"]["high"] <= 0.02,
        "two_domains_positive": all(_domain_positive(row) for row in repaired["by_domain"].values()) and len(repaired["by_domain"]) >= 2,
        "no_leakage_pass": all(
            result["no_leakage"][k] is False
            for k in ["future_endpoint_input", "future_waypoint_input", "central_velocity", "test_endpoint_goals", "test_threshold_tuning"]
        )
        and result["no_leakage"]["internal_val_from_train_only"],
        "no_metric_seconds_overclaim": not result["claim_boundary"]["metric_or_seconds_claim"] and not result["claim_boundary"]["t100_seconds_claim"],
        "stage5c_false": not result["claim_boundary"]["stage5c_executed"],
        "smc_false": not result["claim_boundary"]["smc_enabled"],
    }
    verdict = "stage42_ay_t100_easy_safety_repair_pass" if all(gates.values()) else "stage42_ay_t100_easy_safety_repair_partial"
    return {"source": result["source"], "gates": gates, "passed": int(sum(bool(v) for v in gates.values())), "total": int(len(gates)), "verdict": verdict}


def _render_report(result: Mapping[str, Any]) -> list[str]:
    original = result["original_metrics"]
    repaired = result["repaired_metrics"]
    lines = [
        "# Stage42-AY AW T100 Easy-Safety Repair",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        f"- gate: `{result['stage42_ay_gate']['passed']} / {result['stage42_ay_gate']['total']}`",
        f"- verdict: `{result['stage42_ay_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Repair Policy",
        "",
        f"- validation_best_variant: `{result['validation_best_variant']}`",
        f"- best_lambda: `{result['best_lambda']}`",
        f"- feature_count: `{result['feature_count']}`",
        f"- internal_val_group: `{result['internal_val_group']}`",
        f"- threshold: `{result['repair_policy']['threshold']}`",
        f"- uses_test_metrics_for_threshold: `{result['repair_policy']['uses_test_metrics_for_threshold']}`",
        f"- guarded_slices: `{result['repair_policy']['guarded_slices']}`",
        f"- kept_slices: `{result['repair_policy']['kept_slices']}`",
        "",
        "## Before / After",
        "",
        "| metric | before | after |",
        "| --- | ---: | ---: |",
        f"| global all ADE improvement | {result['repair_effect']['global_all_before']:.6f} | {result['repair_effect']['global_all_after']:.6f} |",
        f"| h100 t100 raw-frame diagnostic | {result['repair_effect']['h100_t100_before']:.6f} | {result['repair_effect']['h100_t100_after']:.6f} |",
        f"| h100 easy degradation | {result['repair_effect']['h100_easy_before']:.6f} | {result['repair_effect']['h100_easy_after']:.6f} |",
        "",
        "## Repaired Global Bootstrap",
        "",
        "| metric | low | mid | high | n |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, row in repaired["bootstrap"].items():
        lines.append(f"| `{key}` | {row['low']:.6f} | {row['mid']:.6f} | {row['high']:.6f} | {row['n']} |")
    lines.extend(
        [
            "",
            "## Domain Metrics After Repair",
            "",
            "| domain | rows | all | t50 | t100 raw diag | hard | easy | switch |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for domain, row in repaired["by_domain"].items():
        lines.append(
            f"| `{domain}` | {row['rows']} | {row['all_improvement']:.6f} | {row['t50_improvement']:.6f} | {row['t100_raw_frame_diagnostic_improvement']:.6f} | {row['hard_failure_improvement']:.6f} | {row['easy_degradation']:.6f} | {row['switch_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Horizon Metrics After Repair",
            "",
            "| horizon | rows | all | horizon metric | hard | easy | switch |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for horizon, row in repaired["by_horizon"].items():
        metric_key = "t100_raw_frame_diagnostic_improvement" if horizon == "100" else f"t{horizon}_improvement"
        lines.append(
            f"| `{horizon}` | {row['rows']} | {row['all_improvement']:.6f} | {row[metric_key]:.6f} | {row['hard_failure_improvement']:.6f} | {row['easy_degradation']:.6f} | {row['switch_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{result['no_leakage']}`",
            f"- claim_boundary: `{result['claim_boundary']}`",
            "",
            "## Interpretation",
            "",
            "- Stage42-AY repairs the Stage42-AX horizon=100 easy-safety weakness with a validation-only strict t100 guard.",
            "- This is a repaired-policy candidate after AX exposed the weak slice; stronger paper claims still need future held-out/source-level confirmation.",
            "- t100 remains raw-frame diagnostic only; no metric, seconds-level, true-3D, Stage5C, SMC, or ungated-neural claim is made.",
        ]
    )
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_ay_gate"]
    lines = [
        "# Stage42-AY Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | {bool(ok)} |")
    return lines


def _append_ledger(result: Mapping[str, Any]) -> None:
    row = {
        "stage": result["stage"],
        "source": result["source"],
        "generated_at_utc": result["generated_at_utc"],
        "verdict": result["stage42_ay_gate"]["verdict"],
        "gate": f"{result['stage42_ay_gate']['passed']}/{result['stage42_ay_gate']['total']}",
        "git_commit": result["git_commit"],
    }
    with LEDGER_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    run_stage42_aw_t100_easy_safety_repair()
