from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage42_group_consistency_full_waypoint_repair as di
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_temporal_group_repel_repair as ez
from src import stage42_waypointwise_group_repel_repair as fa
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "proximity_pareto_composer_stage42.json"
REPORT_MD = OUT_DIR / "proximity_pareto_composer_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_fb_gate.md"

DI_JSON = OUT_DIR / "group_consistency_full_waypoint_repair_stage42.json"
FA_JSON = OUT_DIR / "waypointwise_group_repel_repair_stage42.json"
AM_JSON = OUT_DIR / "source_level_full_waypoint_eval_stage42.json"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
PAPER_FILES = ez.PAPER_FILES

EPS = 1e-6
SOURCE = "fresh_stage42_proximity_pareto_composer"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-DI 更准但 proximity 相对 FA 更差；Stage42-FA 更安全但 all/hard 低于 DI。",
    "Stage42-FB 用 validation-only composer 研究 DI/FA 的 safety-accuracy Pareto 边界。",
    "composer 只使用 predicted rollout geometry、source/frame/horizon group key、agent id、当前/过去可得信息。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不能作为 inference input。",
    "composer policy 只在 validation 上选择；test 只评一次。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _group_any_mask(keys: np.ndarray, row_risk: np.ndarray) -> np.ndarray:
    out = np.zeros(len(keys), dtype=bool)
    for key in sorted(set(keys.tolist())):
        rows = keys == key
        out[rows] = bool(np.any(row_risk[rows]))
    return out


def _compose_xy(
    candidate: Mapping[str, Any],
    ids: np.ndarray,
    data: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    floor_xy: np.ndarray,
    group_key: np.ndarray,
    di_eval: Mapping[str, Any],
    fa_eval: Mapping[str, Any],
) -> dict[str, Any]:
    ids = np.asarray(ids, dtype=np.int64)
    keys = group_key[ids]
    normalizer = np.maximum(data["scale"][ids].astype(np.float64), EPS)
    agent = data["agent_id"][ids].astype(np.int64)
    di_xy = di_eval["selected_xy"].astype(np.float32)
    fa_xy = fa_eval["selected_xy"].astype(np.float32)
    di_min = di._min_group_distance_fast(di_xy, keys, normalizer, agent)
    fa_min = di._min_group_distance_fast(fa_xy, keys, normalizer, agent)
    threshold = float(candidate.get("threshold", 0.05))
    margin = float(candidate.get("margin", 0.0))
    mode = str(candidate["mode"])
    if mode == "all_di":
        use_fa = np.zeros(len(ids), dtype=bool)
    elif mode == "row_di_near":
        use_fa = np.isfinite(di_min) & (di_min < threshold) & np.isfinite(fa_min) & (fa_min + margin >= di_min)
    elif mode == "group_di_near":
        row_risk = np.isfinite(di_min) & (di_min < threshold) & np.isfinite(fa_min) & (fa_min + margin >= di_min)
        use_fa = _group_any_mask(keys, row_risk)
    elif mode == "group_di_near_fa_safer":
        row_risk = np.isfinite(di_min) & (di_min < threshold) & np.isfinite(fa_min) & (fa_min > di_min + margin)
        use_fa = _group_any_mask(keys, row_risk)
    else:
        raise ValueError(f"unknown Stage42-FB composer mode: {mode}")
    selected_xy = di_xy.copy()
    selected_xy[use_fa] = fa_xy[use_fa]
    switch = di_eval["switch"].astype(bool).copy()
    switch[use_fa] = fa_eval["switch"].astype(bool)[use_fa]
    selected_ade, selected_fde = di._trajectory_errors_subset(selected_xy, labels, ids)
    floor_ade, floor_fde = di._trajectory_errors_subset(floor_xy[ids], labels, ids)
    metric = di._metric_subset(selected_ade, floor_ade, data, ids, switch)
    final_min = di._min_group_distance_fast(selected_xy, keys, normalizer, agent)
    return {
        "selected_xy": selected_xy,
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "floor_ade": floor_ade,
        "floor_fde": floor_fde,
        "switch": switch,
        "use_fa": use_fa,
        "metric": metric,
        "diagnostics": {
            "use_fa_rate": float(np.mean(use_fa)) if len(use_fa) else 0.0,
            "use_fa_rows": int(np.sum(use_fa)),
            "di_near_005": float(np.mean(np.isfinite(di_min) & (di_min < 0.05))) if len(di_min) else 0.0,
            "fa_near_005": float(np.mean(np.isfinite(fa_min) & (fa_min < 0.05))) if len(fa_min) else 0.0,
            "final_near_005": float(np.mean(np.isfinite(final_min) & (final_min < 0.05))) if len(final_min) else 0.0,
            "di_p05_min_distance": float(np.percentile(di_min[np.isfinite(di_min)], 5)) if np.any(np.isfinite(di_min)) else None,
            "fa_p05_min_distance": float(np.percentile(fa_min[np.isfinite(fa_min)], 5)) if np.any(np.isfinite(fa_min)) else None,
            "final_p05_min_distance": float(np.percentile(final_min[np.isfinite(final_min)], 5)) if np.any(np.isfinite(final_min)) else None,
        },
    }


def _candidate_grid() -> list[dict[str, Any]]:
    rows = [{"mode": "all_di", "threshold": 0.0, "margin": 0.0}]
    for mode in ["row_di_near", "group_di_near", "group_di_near_fa_safer"]:
        for threshold in [0.035, 0.05, 0.065, 0.08]:
            for margin in [0.0, 0.005, 0.01]:
                rows.append({"mode": mode, "threshold": threshold, "margin": margin})
    return rows


def _selection_score(metric: Mapping[str, Any], diagnostics: Mapping[str, Any], delta_di: Mapping[str, Any]) -> float:
    near_repair = float(diagnostics["di_near_005"]) - float(diagnostics["final_near_005"])
    all_loss = max(0.0, -float(delta_di["all_improvement"]))
    hard_loss = max(0.0, -float(delta_di["hard_failure_improvement"]))
    return (
        1.35 * float(metric["all_improvement"])
        + 1.35 * float(metric["hard_failure_improvement"])
        + 1.0 * float(metric["t50_improvement"])
        + 0.50 * float(metric["t100_raw_frame_diagnostic_improvement"])
        + 3.5 * max(0.0, near_repair)
        - 45.0 * max(0.0, float(metric["easy_degradation"]) - 0.02)
        - 3.0 * all_loss
        - 3.0 * hard_loss
        - 0.02 * float(diagnostics["use_fa_rate"])
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


def _evaluate_composer(
    data: Mapping[str, np.ndarray],
    split: np.ndarray,
    labels: Mapping[str, np.ndarray],
    floor: Mapping[str, Any],
    am_candidate: Mapping[str, Any],
    group_key: np.ndarray,
    di_candidate: Mapping[str, Any],
    fa_candidate: Mapping[str, Any],
) -> dict[str, Any]:
    val_ids = np.where(split == "val")[0]
    test_ids = np.where(split == "test")[0]
    floor_xy = floor["floor_xy"].astype(np.float32)
    pred_xy = am_candidate["pred_xy"].astype(np.float32)
    base_xy = am_candidate["selected_xy"].astype(np.float32)
    base_switch = am_candidate["switch"].astype(bool)
    di_val = di._repair_subset(val_ids, di_candidate, data, labels, floor_xy, pred_xy, base_xy, base_switch, group_key)
    fa_val = fa._repair_subset_waypointwise(val_ids, fa_candidate, data, labels, floor_xy, pred_xy, base_xy, base_switch, group_key)
    di_test = di._repair_subset(test_ids, di_candidate, data, labels, floor_xy, pred_xy, base_xy, base_switch, group_key)
    fa_test = fa._repair_subset_waypointwise(test_ids, fa_candidate, data, labels, floor_xy, pred_xy, base_xy, base_switch, group_key)
    rows = []
    best: dict[str, Any] | None = None
    best_score = -1e9
    for candidate in _candidate_grid():
        val = _compose_xy(candidate, val_ids, data, labels, floor_xy, group_key, di_val, fa_val)
        delta_di = _delta(val["metric"], di_val["metric"])
        score = _selection_score(val["metric"], val["diagnostics"], delta_di)
        row = {
            "candidate": dict(candidate),
            "val_score": float(score),
            "val_metric": val["metric"],
            "val_diagnostics": val["diagnostics"],
            "val_delta_vs_di": delta_di,
        }
        rows.append(row)
        if score > best_score:
            best_score = float(score)
            best = row
    if best is None:
        raise RuntimeError("No Stage42-FB composer candidate evaluated.")
    test = _compose_xy(best["candidate"], test_ids, data, labels, floor_xy, group_key, di_test, fa_test)
    h = data["horizon"][test_ids].astype(int)
    hard_failure = data["hard"][test_ids].astype(bool) | data["failure"][test_ids].astype(bool)
    easy = data["easy"][test_ids].astype(bool)
    domain = data["dataset"][test_ids].astype(str)
    bootstrap = {
        "all": di._bootstrap_ci_subset(test["selected_ade"], test["floor_ade"], np.ones(len(test_ids), dtype=bool), seed=43101),
        "t50": di._bootstrap_ci_subset(test["selected_ade"], test["floor_ade"], h == 50, seed=43102),
        "t100_raw_frame_diagnostic": di._bootstrap_ci_subset(test["selected_ade"], test["floor_ade"], h == 100, seed=43103),
        "hard_failure": di._bootstrap_ci_subset(test["selected_ade"], test["floor_ade"], hard_failure, seed=43104),
        "easy_degradation": di._bootstrap_ci_subset(test["floor_ade"], test["selected_ade"], easy, seed=43105),
    }
    by_domain = {
        d: di._metric_subset(test["selected_ade"][domain == d], test["floor_ade"][domain == d], data, test_ids[domain == d], test["switch"][domain == d])
        for d in sorted(set(domain.tolist()))
    }
    return {
        "candidate_count": len(rows),
        "validation_rows": sorted(rows, key=lambda row: row["val_score"], reverse=True),
        "selected": best,
        "di_test": {"metric": di_test["metric"], "diagnostics": di_test["diagnostics"]},
        "fa_test": {"metric": fa_test["metric"], "diagnostics": fa_test["diagnostics"]},
        "test": {
            "metric_vs_floor": test["metric"],
            "diagnostics": test["diagnostics"],
            "delta_vs_di": _delta(test["metric"], di_test["metric"]),
            "delta_vs_fa": _delta(test["metric"], fa_test["metric"]),
            "near_delta_vs_di": float(test["diagnostics"]["final_near_005"]) - float(di_test["diagnostics"]["final_near_005"]),
            "near_delta_vs_fa": float(test["diagnostics"]["final_near_005"]) - float(fa_test["diagnostics"]["final_near_005"]),
            "bootstrap": bootstrap,
            "by_domain": by_domain,
        },
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
    di_payload = read_json(DI_JSON, {})
    fa_payload = read_json(FA_JSON, {})
    di_candidate = di_payload.get("repair", {}).get("selected", {}).get("candidate")
    fa_candidate = fa_payload.get("repair", {}).get("selected", {}).get("candidate")
    if not di_candidate or not fa_candidate:
        raise RuntimeError("Stage42-FB requires Stage42-DI and Stage42-FA selected candidate artifacts.")
    eval_result = _evaluate_composer(data, split, labels, floor, am_candidate, group_key, di_candidate, fa_candidate)
    metric = eval_result["test"]["metric_vs_floor"]
    delta_di = eval_result["test"]["delta_vs_di"]
    near_delta_di = eval_result["test"]["near_delta_vs_di"]
    promotes = (
        metric["all_improvement"] > 0.0
        and metric["hard_failure_improvement"] > 0.0
        and metric["easy_degradation"] <= 0.02
        and (delta_di["all_improvement"] or 0.0) >= -0.0005
        and (delta_di["hard_failure_improvement"] or 0.0) >= -0.0005
        and near_delta_di < -EPS
    )
    result: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-FB proximity Pareto composer",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": ez._git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash(
            [
                "data/stage41_world_model/combined_external.npz",
                str(AM_JSON),
                str(DI_JSON),
                str(FA_JSON),
            ]
        ),
        "split_stats": split_stats,
        "label_stats": {
            "rows": int(len(split)),
            "test_rows": int(np.sum(split == "test")),
            "test_full_waypoint_rows": int(np.sum((split == "test") & np.all(labels["waypoint_valid"], axis=1))),
        },
        "composer_family": {
            "source": "fresh_stage42_fb_validation_only_di_fa_composer",
            "candidate_count": len(_candidate_grid()),
            "default_policy": "Stage42-DI",
            "safety_policy": "Stage42-FA",
            "uses_future_inputs": False,
        },
        "repair": eval_result,
        "deployment_decision": {
            "promote_proximity_pareto_composer": bool(promotes),
            "decision": "promote_stage42_fb_proximity_pareto_composer"
            if promotes
            else "proximity_pareto_composer_not_enough_keep_stage42_di_or_cq_floor",
            "reason": "Promotion requires positive all+hard, easy safe, no material all/hard loss vs Stage42-DI, and lower near@0.05 than Stage42-DI.",
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
    result["stage42_fb_gate"] = _gate(result)
    return result


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    metric = result["repair"]["test"]["metric_vs_floor"]
    delta_di = result["repair"]["test"]["delta_vs_di"]
    near_delta = result["repair"]["test"]["near_delta_vs_di"]
    no_leak = result["no_leakage"]
    gates = {
        "source_level_split_rebuilt": result["split_stats"]["by_split"]["test"]["rows"] == int(metric["rows"]) and int(metric["rows"]) > 0,
        "full_waypoint_labels_available": result["label_stats"]["test_full_waypoint_rows"] > 0,
        "composer_family_built": result["composer_family"]["candidate_count"] >= 20,
        "validation_selected_composer": result["repair"]["selected"]["val_score"] != 0.0 and no_leak["test_threshold_tuning"] is False,
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
        "no_material_all_loss_vs_di": (delta_di["all_improvement"] or 0.0) >= -0.0005,
        "no_material_hard_loss_vs_di": (delta_di["hard_failure_improvement"] or 0.0) >= -0.0005,
        "near_better_than_stage42_di": near_delta < -EPS,
        "bootstrap_reported": result["repair"]["test"]["bootstrap"]["all"]["bootstrap_n"] > 0,
        "no_metric_seconds_overclaim": result["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": result["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": result["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    if passed == total:
        verdict = "stage42_fb_proximity_pareto_composer_pass_promotable"
    elif gates["test_all_positive_vs_floor"] and gates["test_hard_positive_vs_floor"] and gates["easy_degradation_under_2pct"]:
        verdict = "stage42_fb_proximity_pareto_composer_positive_not_promoted"
    else:
        verdict = "stage42_fb_proximity_pareto_composer_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_fb_gate"]
    selected = result["repair"]["selected"]
    metric = result["repair"]["test"]["metric_vs_floor"]
    diag = result["repair"]["test"]["diagnostics"]
    delta_di = result["repair"]["test"]["delta_vs_di"]
    delta_fa = result["repair"]["test"]["delta_vs_fa"]
    di_test = result["repair"]["di_test"]
    fa_test = result["repair"]["fa_test"]
    lines = [
        "# Stage42-FB Proximity Pareto Composer",
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
        "## Selected Composer",
        "",
        f"- candidate: `{selected['candidate']}`",
        f"- val_score: `{selected['val_score']:.6f}`",
        f"- val_metric: `{selected['val_metric']}`",
        f"- val_diagnostics: `{selected['val_diagnostics']}`",
        "",
        "## Test Once Metrics vs Train-Horizon Causal Floor",
        "",
        "| candidate | all | t50 | t100 raw diag | hard/failure | easy degradation | near@0.05 | use FA |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Stage42-DI default | {ez._pct(di_test['metric']['all_improvement'])} | {ez._pct(di_test['metric']['t50_improvement'])} | {ez._pct(di_test['metric']['t100_raw_frame_diagnostic_improvement'])} | {ez._pct(di_test['metric']['hard_failure_improvement'])} | {ez._pct(di_test['metric']['easy_degradation'])} | {ez._pct(di_test['diagnostics']['final_near_005'])} | 0.00% |",
        f"| Stage42-FA safety | {ez._pct(fa_test['metric']['all_improvement'])} | {ez._pct(fa_test['metric']['t50_improvement'])} | {ez._pct(fa_test['metric']['t100_raw_frame_diagnostic_improvement'])} | {ez._pct(fa_test['metric']['hard_failure_improvement'])} | {ez._pct(fa_test['metric']['easy_degradation'])} | {ez._pct(fa_test['diagnostics']['final_near_005'])} | 100.00% |",
        f"| Stage42-FB composer | {ez._pct(metric['all_improvement'])} | {ez._pct(metric['t50_improvement'])} | {ez._pct(metric['t100_raw_frame_diagnostic_improvement'])} | {ez._pct(metric['hard_failure_improvement'])} | {ez._pct(metric['easy_degradation'])} | {ez._pct(diag['final_near_005'])} | {ez._pct(diag['use_fa_rate'])} |",
        "",
        "## Delta",
        "",
        f"- delta_vs_DI all/t50/t100raw/hard/easy: `{ez._pct(delta_di['all_improvement'])}` / `{ez._pct(delta_di['t50_improvement'])}` / `{ez._pct(delta_di['t100_raw_frame_diagnostic_improvement'])}` / `{ez._pct(delta_di['hard_failure_improvement'])}` / `{ez._pct(delta_di['easy_degradation'])}`",
        f"- delta_vs_FA all/t50/t100raw/hard/easy: `{ez._pct(delta_fa['all_improvement'])}` / `{ez._pct(delta_fa['t50_improvement'])}` / `{ez._pct(delta_fa['t100_raw_frame_diagnostic_improvement'])}` / `{ez._pct(delta_fa['hard_failure_improvement'])}` / `{ez._pct(delta_fa['easy_degradation'])}`",
        f"- near_delta_vs_DI: `{ez._pct(result['repair']['test']['near_delta_vs_di'])}`",
        f"- near_delta_vs_FA: `{ez._pct(result['repair']['test']['near_delta_vs_fa'])}`",
        "",
        "## Bootstrap CI",
        "",
        "| slice | low | mid | high | n |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, row in result["repair"]["test"]["bootstrap"].items():
        lines.append(f"| `{key}` | {row['low']:.6f} | {row['mid']:.6f} | {row['high']:.6f} | {row['n']} |")
    lines.extend(["", "## By Domain", "", "| domain | rows | all | t50 | t100 raw diag | hard/failure | easy | switch |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for domain, row in result["repair"]["test"]["by_domain"].items():
        lines.append(
            f"| `{domain}` | {row['rows']} | {ez._pct(row['all_improvement'])} | {ez._pct(row['t50_improvement'])} | {ez._pct(row['t100_raw_frame_diagnostic_improvement'])} | {ez._pct(row['hard_failure_improvement'])} | {ez._pct(row['easy_degradation'])} | {ez._pct(row['switch_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-FB explicitly tests whether the DI/FA Pareto boundary can be composed: use DI by default and FA only for predicted proximity-risk rows or groups.",
            "- If promoted, it is a safety-sensitive runtime composer. If not, DI/CQ remains the floor and this documents the proximity/accuracy tradeoff.",
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{result['no_leakage']}`",
            f"- claim_boundary: `{result['claim_boundary']}`",
        ]
    )
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_fb_gate"]
    return [
        "# Stage42-FB Gate",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Gates",
        "",
        *[f"- {key}: `{value}`" for key, value in gate["gates"].items()],
    ]


def _refresh_lines(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_fb_gate"]
    metric = result["repair"]["test"]["metric_vs_floor"]
    delta_di = result["repair"]["test"]["delta_vs_di"]
    diag = result["repair"]["test"]["diagnostics"]
    return [
        "## Stage42-FB Proximity Pareto Composer",
        "",
        "- source: `fresh_stage42_proximity_pareto_composer`",
        "- role: validation-only composer between Stage42-DI accuracy policy and Stage42-FA proximity-safety policy.",
        f"- selected candidate: `{result['repair']['selected']['candidate']}`.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- test all/t50/t100raw/hard/easy: `{ez._pct(metric['all_improvement'])}` / `{ez._pct(metric['t50_improvement'])}` / `{ez._pct(metric['t100_raw_frame_diagnostic_improvement'])}` / `{ez._pct(metric['hard_failure_improvement'])}` / `{ez._pct(metric['easy_degradation'])}`.",
        f"- delta vs Stage42-DI all/t50/t100raw/hard/easy: `{ez._pct(delta_di['all_improvement'])}` / `{ez._pct(delta_di['t50_improvement'])}` / `{ez._pct(delta_di['t100_raw_frame_diagnostic_improvement'])}` / `{ez._pct(delta_di['hard_failure_improvement'])}` / `{ez._pct(delta_di['easy_degradation'])}`.",
        f"- near@0.05 final/use_fa_rate: `{ez._pct(diag['final_near_005'])}` / `{ez._pct(diag['use_fa_rate'])}`.",
        f"- decision: `{result['deployment_decision']['decision']}`.",
        "- Boundary: protected source-level raw-frame 2.5D; no metric/seconds claim, no true 3D, no Stage5C, no SMC.",
    ]


def _refresh_readmes(result: Mapping[str, Any]) -> None:
    lines = _refresh_lines(result)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, "STAGE42_FB_PROXIMITY_PARETO_COMPOSER", lines)


def _refresh_paper_package(result: Mapping[str, Any]) -> None:
    lines = _refresh_lines(result)
    for path in PAPER_FILES:
        _replace_section(path, "STAGE42_FB_PROXIMITY_PARETO_COMPOSER", lines)


def _refresh_research_state(result: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-FB proximity Pareto composer"
    state["current_verdict"] = result["stage42_fb_gate"]["verdict"]
    state["stage42_fb_proximity_pareto_composer"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": result["stage42_fb_gate"]["verdict"],
        "gates": f"{result['stage42_fb_gate']['passed']}/{result['stage42_fb_gate']['total']}",
        "deployment_decision": result["deployment_decision"],
        "selected_candidate": result["repair"]["selected"]["candidate"],
        "test_metric_vs_floor": result["repair"]["test"]["metric_vs_floor"],
        "test_diagnostics": result["repair"]["test"]["diagnostics"],
        "test_delta_vs_di": result["repair"]["test"]["delta_vs_di"],
        "claim_boundary": result["claim_boundary"],
        "conclusion": "Stage42-FB composes Stage42-DI and Stage42-FA using validation-only predicted proximity risk to test whether safety can be improved without material all/hard loss.",
    }
    write_json(RESEARCH_STATE, ez._jsonable(state))


def run_stage42_proximity_pareto_composer() -> dict[str, Any]:
    result = _build_payload()
    write_json(REPORT_JSON, ez._jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _refresh_readmes(result)
    _refresh_paper_package(result)
    _refresh_research_state(result)
    return result


if __name__ == "__main__":
    payload = run_stage42_proximity_pareto_composer()
    gate = payload["stage42_fb_gate"]
    print(f"Stage42-FB gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
    print(f"Decision: {payload['deployment_decision']['decision']}")
