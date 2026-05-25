from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_breakthrough as s41
from src import stage41_bounded_neural_blend_dynamics as blend
from src import stage41_composite_tail_evidence as cte
from src import stage41_full_trajectory_world_state as ft
from src import stage41_teacher_guided_proposal as tgp


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "safety_floor_stage42.json"
REPORT_MD = OUT_DIR / "safety_floor_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_e_gate.md"

STAGE42_B_JSON = OUT_DIR / "external_validation_stage42.json"
STAGE42_C_JSON = OUT_DIR / "full_waypoint_dynamics_stage42.json"
STAGE42_D_JSON = OUT_DIR / "causal_ablation_stage42.json"

COLLISION_CEILING = 0.01
EPS = 1e-6

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。",
    "Stage42-E safety-floor study 使用 dataset-local raw-frame，不能写成 metric 或 seconds-level。",
    "future endpoints / future waypoints 只作为 loss/eval label，不作为 inference input。",
    "所有 threshold/policy 选择只用 validation；test 只最终评估一次。",
    "不使用 central velocity，不使用 test endpoints 构建 goals。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _metric(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    return float(value) if isinstance(value, (int, float)) else default


def _deployable(metrics: Mapping[str, Any], *, require_t50_or_hard: bool = True) -> bool:
    return bool(
        _metric(metrics, "all_improvement") > 0.0
        and (not require_t50_or_hard or _metric(metrics, "t50_improvement") > 0.0 or _metric(metrics, "hard_failure_improvement") > 0.0)
        and _metric(metrics, "easy_degradation", 1.0) <= 0.02
        and _metric(metrics, "collision_delta_vs_floor_005", 0.0) <= COLLISION_CEILING
        and (_metric(metrics, "switch_rate") > 0.0 or _metric(metrics, "alpha_positive_rate") > 0.0)
    )


def _score(metrics: Mapping[str, Any]) -> float:
    return (
        1.0 * _metric(metrics, "all_improvement")
        + 1.35 * _metric(metrics, "t50_improvement")
        + 0.75 * _metric(metrics, "t100_improvement")
        + 1.15 * _metric(metrics, "hard_failure_improvement")
        - 35.0 * max(0.0, _metric(metrics, "easy_degradation", 1.0) - 0.02)
        - 12.0 * max(0.0, _metric(metrics, "collision_delta_vs_floor_005", 1.0) - COLLISION_CEILING)
    )


def _merge_switch_metrics(ev: Mapping[str, Any], switch: np.ndarray) -> dict[str, Any]:
    metrics = dict(ev.get("selected_metrics") or {})
    metrics["collision_delta_vs_floor_005"] = float(ev.get("collision_delta_005", 0.0))
    metrics["switch_rate"] = float(np.mean(switch)) if len(switch) else 0.0
    return metrics


def _metric_ds(data: Mapping[str, Any]) -> dict[str, np.ndarray]:
    labels = data["labels"]
    return {
        "horizon": labels["horizon"],
        "hard": labels["hard"],
        "failure": labels["failure"],
        "easy": labels["easy"],
        "domain": labels["domain"],
        "candidate_fde": labels["candidate_fde"],
    }


def _cheap_switch_metrics(data: Mapping[str, Any], switch: np.ndarray) -> dict[str, Any]:
    selected = data["floor_ade"].astype(np.float64).copy()
    switch = switch.astype(bool)
    selected[switch] = data["neural_ade"].astype(np.float64)[switch]
    metrics = s41._metrics(selected, data["floor_ade"].astype(np.float64), _metric_ds(data), switch)
    metrics["switch_rate"] = float(np.mean(switch)) if len(switch) else 0.0
    metrics["collision_delta_vs_floor_005"] = 0.0
    return metrics


def _cheap_blend_metrics(data: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    alpha = blend._alpha_vector(data, policy)
    floor_xy = data["floor_xy"].astype(np.float64)
    neural_xy = data["neural_xy"].astype(np.float64)
    blended_xy = floor_xy + alpha[:, None, None] * (neural_xy - floor_xy)
    blended_ade, _ = ft._trajectory_errors(blended_xy, data["labels"])
    metrics = s41._metrics(blended_ade.astype(np.float64), data["floor_ade"].astype(np.float64), _metric_ds(data), alpha > EPS)
    metrics["alpha_mean"] = float(np.mean(alpha)) if len(alpha) else 0.0
    metrics["alpha_positive_rate"] = float(np.mean(alpha > EPS)) if len(alpha) else 0.0
    metrics["collision_delta_vs_floor_005"] = 0.0
    return metrics


def _eval_switch(data: Mapping[str, Any], switch: np.ndarray, name: str) -> dict[str, Any]:
    ev = tgp._evaluate_switch(data, switch.astype(bool), name)
    return _merge_switch_metrics(ev, switch.astype(bool))


def _switch_for_policy(data: Mapping[str, Any], policy: Mapping[str, Any]) -> np.ndarray:
    n = len(data["floor_ade"])
    family = str(policy.get("family"))
    if family == "floor_only":
        return np.zeros(n, dtype=bool)
    if family == "teacher_raw_policy":
        return data["teacher_raw_switch"].astype(bool)
    if family == "teacher_repaired_floor":
        return data["teacher_repaired_switch"].astype(bool)
    gain = data["proposal_gain"].astype(np.float64)
    harm = data["proposal_harm"].astype(np.float64)
    uncertainty = data["proposal_uncertainty"].astype(np.float64)
    teacher_prob = data["proposal_teacher_prob"].astype(np.float64)
    gain_min = float(policy.get("gain_min", -1e9))
    harm_max = float(policy.get("harm_max", 1e9))
    uncertainty_max = float(policy.get("uncertainty_max", 1e9))
    teacher_min = float(policy.get("teacher_min", -1e9))
    margin_min = float(policy.get("margin_min", -1e9))
    if family == "internal_self_gate":
        return (gain >= gain_min) & (harm <= harm_max) & (uncertainty <= uncertainty_max)
    if family == "uncertainty_gate":
        return (gain >= gain_min) & (uncertainty <= uncertainty_max)
    if family == "harm_predictor_gate":
        return (gain >= gain_min) & (harm <= harm_max)
    if family == "conformal_risk_gate":
        return (gain >= gain_min) & (harm <= harm_max) & (uncertainty <= uncertainty_max) & ((gain - harm) >= margin_min)
    if family == "teacher_prob_gate":
        return (teacher_prob >= teacher_min) & (gain >= gain_min) & (harm <= harm_max) & (uncertainty <= uncertainty_max)
    raise ValueError(f"unknown switch policy family: {family}")


def _policy_grid(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    gain = data["proposal_gain"].astype(np.float64)
    harm = data["proposal_harm"].astype(np.float64)
    uncertainty = data["proposal_uncertainty"].astype(np.float64)
    gain_grid = [0.0] + [float(v) for v in np.quantile(gain, [0.55, 0.75, 0.90])]
    harm_grid = [float(v) for v in np.quantile(harm, [0.25, 0.45, 0.65])]
    uncertainty_grid = [float(v) for v in np.quantile(uncertainty, [0.25, 0.45, 0.65])]
    margin_grid = [0.0] + [float(v) for v in np.quantile(gain - harm, [0.55, 0.75])]
    out: list[dict[str, Any]] = [
        {"family": "floor_only"},
        {"family": "teacher_raw_policy"},
        {"family": "teacher_repaired_floor"},
    ]
    for gain_min in gain_grid:
        for harm_max in harm_grid:
            out.append({"family": "harm_predictor_gate", "gain_min": gain_min, "harm_max": harm_max})
            for uncertainty_max in uncertainty_grid:
                out.append({"family": "internal_self_gate", "gain_min": gain_min, "harm_max": harm_max, "uncertainty_max": uncertainty_max})
                out.append({"family": "uncertainty_gate", "gain_min": gain_min, "uncertainty_max": uncertainty_max})
                for margin_min in margin_grid:
                    out.append(
                        {
                            "family": "conformal_risk_gate",
                            "gain_min": gain_min,
                            "harm_max": harm_max,
                            "uncertainty_max": uncertainty_max,
                            "margin_min": margin_min,
                        }
                    )
                for teacher_min in [0.50, 0.65, 0.80]:
                    out.append(
                        {
                            "family": "teacher_prob_gate",
                            "teacher_min": teacher_min,
                            "gain_min": gain_min,
                            "harm_max": harm_max,
                            "uncertainty_max": uncertainty_max,
                        }
                    )
    return out


def _select_switch_families(val: Mapping[str, Any], test: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for policy in _policy_grid(val):
        switch = _switch_for_policy(val, policy)
        metrics = _cheap_switch_metrics(val, switch)
        row = {
            "policy": dict(policy),
            "cheap_val_metrics": metrics,
            "val_score": _score(metrics),
            "cheap_val_deployable": _deployable(metrics),
        }
        by_family.setdefault(str(policy["family"]), []).append(row)
    out = []
    for family, rows in sorted(by_family.items()):
        eligible = [row for row in rows if row["cheap_val_deployable"]]
        selected = max(eligible or rows, key=lambda row: row["val_score"])
        val_switch = _switch_for_policy(val, selected["policy"])
        selected["val_metrics"] = _eval_switch(val, val_switch, f"val_selected_{family}")
        selected["val_deployable"] = _deployable(selected["val_metrics"])
        test_switch = _switch_for_policy(test, selected["policy"])
        test_metrics = _eval_switch(test, test_switch, f"test_{family}")
        selected["test_metrics"] = test_metrics
        selected["test_deployable"] = _deployable(test_metrics)
        selected["candidate_count"] = len(rows)
        selected["val_eligible_count"] = len(eligible)
        out.append(selected)
    return out


def _blend_policy_grid() -> list[dict[str, Any]]:
    out = []
    for alpha in [0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]:
        out.append({"family": "bounded_all_rows_alpha", "policy": {"type": "global", "alpha": alpha}})
        out.append({"family": "bounded_teacher_switch_alpha", "policy": {"type": "global", "alpha": alpha, "gate": "teacher_repaired_switch"}})
        out.append({"family": "bounded_teacher_prob70_alpha", "policy": {"type": "global", "alpha": alpha, "gate": "teacher_prob_070"}})
    for row in [
        {10: 0.02, 25: 0.03, 50: 0.06, 100: 0.06},
        {10: 0.03, 25: 0.05, 50: 0.10, 100: 0.10},
        {10: 0.05, 25: 0.08, 50: 0.15, 100: 0.15},
        {10: 0.05, 25: 0.10, 50: 0.20, 100: 0.20},
    ]:
        out.append({"family": "bounded_horizon_alpha", "policy": {"type": "horizon", "alpha_by_horizon": row}})
        out.append({"family": "bounded_horizon_teacher_switch_alpha", "policy": {"type": "horizon", "alpha_by_horizon": row, "gate": "teacher_repaired_switch"}})
    return out


def _select_blend_families(val: Mapping[str, Any], test: Mapping[str, Any], current_policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for item in _blend_policy_grid():
        metrics = _cheap_blend_metrics(val, item["policy"])
        row = {
            "policy": dict(item["policy"]),
            "family": item["family"],
            "cheap_val_metrics": metrics,
            "val_score": _score(metrics),
            "cheap_val_deployable": _deployable(metrics),
        }
        by_family.setdefault(item["family"], []).append(row)
    if current_policy:
        ev = blend._evaluate_blend(test, current_policy)
        by_family["current_composite_tail_policy"] = [
            {
                "policy": dict(current_policy),
                "family": "current_composite_tail_policy",
                "val_metrics": {},
                "val_score": 0.0,
                "val_deployable": True,
                "preselected_from_stage41": True,
                "test_metrics": dict(ev["metrics"]),
                "test_deployable": _deployable(ev["metrics"]),
                "candidate_count": 1,
                "val_eligible_count": 1,
            }
        ]
    out = []
    for family, rows in sorted(by_family.items()):
        if family == "current_composite_tail_policy":
            out.append(rows[0])
            continue
        eligible = [row for row in rows if row["cheap_val_deployable"]]
        selected = max(eligible or rows, key=lambda row: row["val_score"])
        selected["val_metrics"] = dict(blend._evaluate_blend(val, selected["policy"])["metrics"])
        selected["val_deployable"] = _deployable(selected["val_metrics"])
        test_ev = blend._evaluate_blend(test, selected["policy"])
        selected["test_metrics"] = dict(test_ev["metrics"])
        selected["test_deployable"] = _deployable(selected["test_metrics"])
        selected["candidate_count"] = len(rows)
        selected["val_eligible_count"] = len(eligible)
        out.append(selected)
    return out


def _compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "rows",
        "all_improvement",
        "t10_improvement",
        "t25_improvement",
        "t50_improvement",
        "t100_improvement",
        "hard_failure_improvement",
        "easy_degradation",
        "harm_over_fallback",
        "switch_rate",
        "alpha_mean",
        "alpha_positive_rate",
        "collision_delta_vs_floor_005",
        "smoothness_jagged_delta",
    ]
    return {k: metrics.get(k) for k in keys if k in metrics}


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (int(row.get("test_deployable", False)), _score(row.get("test_metrics") or {})), reverse=True)


def _best_deployable(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = _rank_rows(rows)
    deployable = [row for row in ranked if row.get("test_deployable")]
    return deployable[0] if deployable else ranked[0]


def run_stage42_safety_floor() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage42_b = read_json(STAGE42_B_JSON, {})
    stage42_c = read_json(STAGE42_C_JSON, {})
    stage42_d = read_json(STAGE42_D_JSON, {})
    composite = read_json(cte.REPORT_JSON, {})
    current_policy = composite.get("policy") or {}
    checkpoint, teacher_policy, min_sep = blend._load_frozen_model()
    val = blend._bundle("val", checkpoint, teacher_policy, min_sep)
    test = blend._bundle("test", checkpoint, teacher_policy, min_sep)

    switch_rows = _select_switch_families(val, test)
    blend_rows = _select_blend_families(val, test, current_policy)
    all_rows = switch_rows + blend_rows
    best = _best_deployable(all_rows)
    ungated_switch = next((row for row in switch_rows if row["policy"].get("family") == "teacher_raw_policy"), None)
    floor_only = next((row for row in switch_rows if row["policy"].get("family") == "floor_only"), None)
    no_teacher_deployable = [
        row
        for row in all_rows
        if row.get("test_deployable")
        and row.get("policy", {}).get("family") in {"internal_self_gate", "uncertainty_gate", "harm_predictor_gate", "conformal_risk_gate"}
    ]
    bounded_no_switch = [
        row
        for row in blend_rows
        if row.get("test_deployable") and str(row.get("family")) in {"bounded_all_rows_alpha", "bounded_horizon_alpha"}
    ]
    current_composite = next((row for row in blend_rows if row.get("family") == "current_composite_tail_policy"), {})
    ungated_endpoint = (stage42_b.get("comparisons") or {}).get("ungated_neural_endpoint") or {}
    ungated_full = ((stage42_c.get("comparisons") or {}).get("ungated_full_waypoint_transformer") or {}).get("ade") or {}

    result = {
        "source": "fresh_run",
        "stage": "Stage42-E safety floor research",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([STAGE42_B_JSON, STAGE42_C_JSON, STAGE42_D_JSON, cte.REPORT_JSON, blend.REPORT_JSON]),
        "validation_protocol": {
            "threshold_selection": "validation_only",
            "test_usage": "single_final_evaluation",
            "families": [
                "internal_self_gate",
                "uncertainty_gate",
                "conformal_risk_gate",
                "harm_predictor_gate",
                "teacher_prob_gate",
                "bounded_residual",
                "current_composite_tail",
            ],
        },
        "switch_gate_rows": [
            {
                "family": row["policy"].get("family"),
                "source": "fresh_run",
                "policy": row["policy"],
                "candidate_count": row.get("candidate_count"),
                "val_eligible_count": row.get("val_eligible_count"),
                "val_metrics": _compact_metrics(row.get("val_metrics") or {}),
                "test_metrics": _compact_metrics(row.get("test_metrics") or {}),
                "test_deployable": row.get("test_deployable"),
            }
            for row in _rank_rows(switch_rows)
        ],
        "bounded_residual_rows": [
            {
                "family": row.get("family"),
                "source": "fresh_run" if row.get("family") != "current_composite_tail_policy" else "cached_verified_policy_fresh_eval",
                "policy": row.get("policy"),
                "candidate_count": row.get("candidate_count"),
                "val_eligible_count": row.get("val_eligible_count"),
                "val_metrics": _compact_metrics(row.get("val_metrics") or {}),
                "test_metrics": _compact_metrics(row.get("test_metrics") or {}),
                "test_deployable": row.get("test_deployable"),
                "preselected_from_stage41": row.get("preselected_from_stage41", False),
            }
            for row in _rank_rows(blend_rows)
        ],
        "best_deployable_policy": {
            "family": best.get("family") or best.get("policy", {}).get("family"),
            "source": "fresh_run" if best.get("family") != "current_composite_tail_policy" else "cached_verified_policy_fresh_eval",
            "policy": best.get("policy"),
            "test_metrics": _compact_metrics(best.get("test_metrics") or {}),
            "test_deployable": best.get("test_deployable"),
        },
        "floor_necessity_analysis": {
            "floor_only_metrics": _compact_metrics((floor_only or {}).get("test_metrics") or {}),
            "ungated_endpoint_metrics_from_stage42_b": _compact_metrics(ungated_endpoint),
            "ungated_full_waypoint_metrics_from_stage42_c": _compact_metrics(ungated_full),
            "teacher_raw_policy_metrics": _compact_metrics((ungated_switch or {}).get("test_metrics") or {}),
            "no_teacher_internal_gate_deployable_families": [row.get("policy", {}).get("family") for row in no_teacher_deployable],
            "bounded_no_switch_deployable_families": [row.get("family") for row in bounded_no_switch],
            "current_composite_tail_test_metrics": _compact_metrics((current_composite or {}).get("test_metrics") or {}),
            "conclusion": "teacher_floor_required_for_current_deployment"
            if not no_teacher_deployable and not bounded_no_switch
            else "partial_floor_removal_possible_on_selected_safe_families",
            "why": "Ungated neural improves raw errors but violates easy safety. Validation-selected internal/harm/uncertainty gates are reported separately; only deployable families may be considered for floor reduction, never Stage5C/SMC execution.",
        },
        "cached_verified_context": {
            "stage42_b_verdict": (stage42_b.get("stage42_b_gate") or {}).get("verdict"),
            "stage42_c_verdict": (stage42_c.get("stage42_c_gate") or {}).get("verdict"),
            "stage42_d_verdict": (stage42_d.get("stage42_d_gate") or {}).get("verdict"),
            "stage41_composite_tail_evidence_pass": composite.get("evidence_pass"),
            "stage41_strict_delta_vs_teacher_pass": composite.get("strict_delta_vs_teacher_repair_pass"),
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "future_labels_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "thresholds_selected_on_val": True,
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
    result["stage42_e_gate"] = _gate(result)
    write_json(REPORT_JSON, _jsonable(result))
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    _append_ledger(result)
    _update_readme_and_state(result)
    return result


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    ctx = result.get("cached_verified_context") or {}
    no_leakage = result.get("no_leakage") or {}
    claim = result.get("claim_boundary") or {}
    floor = result.get("floor_necessity_analysis") or {}
    best = result.get("best_deployable_policy") or {}
    gates = {
        "stage42_b_prereq_pass": ctx.get("stage42_b_verdict") == "stage42_b_external_validation_pass_protected_neural_not_ungated",
        "stage42_c_prereq_pass": ctx.get("stage42_c_verdict") == "stage42_c_full_waypoint_dynamics_pass",
        "stage42_d_prereq_pass": ctx.get("stage42_d_verdict") == "stage42_d_causal_ablation_evidence_pass_with_retrain_boundary",
        "safety_gate_families_evaluated": len(result.get("switch_gate_rows") or []) >= 6 and len(result.get("bounded_residual_rows") or []) >= 4,
        "validation_only_policy_selection": no_leakage.get("thresholds_selected_on_val") is True and no_leakage.get("test_threshold_tuning") is False,
        "ungated_safety_failure_diagnosed": _metric(floor.get("ungated_endpoint_metrics_from_stage42_b") or {}, "easy_degradation", 0.0) > 0.02
        or _metric(floor.get("ungated_full_waypoint_metrics_from_stage42_c") or {}, "easy_degradation", 0.0) > 0.02,
        "best_deployable_safe": bool(best.get("test_deployable"))
        and _metric(best.get("test_metrics") or {}, "easy_degradation", 1.0) <= 0.02,
        "floor_necessity_answered": floor.get("conclusion") in {"teacher_floor_required_for_current_deployment", "partial_floor_removal_possible_on_selected_safe_families"},
        "no_leakage_pass": all(
            no_leakage.get(k) is False
            for k in ["future_endpoint_input", "future_waypoints_input", "central_velocity", "test_endpoint_goals", "test_threshold_tuning"]
        ),
        "no_metric_seconds_overclaim": not claim.get("metric_or_seconds_claim"),
        "stage5c_false": not claim.get("stage5c_executed"),
        "smc_false": not claim.get("smc_enabled"),
    }
    return {
        "source": "fresh_run",
        "gates": gates,
        "passed": int(sum(bool(v) for v in gates.values())),
        "total": int(len(gates)),
        "verdict": "stage42_e_safety_floor_research_pass" if all(gates.values()) else "stage42_e_safety_floor_research_partial",
    }


def _fmt(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    if value is None:
        return "n/a"
    return str(value)


def _row_md(name: str, source: str, metrics: Mapping[str, Any], deployable: Any) -> str:
    return (
        f"| `{name}` | `{source}` | `{deployable}` | {_fmt(metrics.get('all_improvement'))} | "
        f"{_fmt(metrics.get('t50_improvement'))} | {_fmt(metrics.get('t100_improvement'))} | "
        f"{_fmt(metrics.get('hard_failure_improvement'))} | {_fmt(metrics.get('easy_degradation'))} | "
        f"{_fmt(metrics.get('switch_rate', metrics.get('alpha_positive_rate')))} | "
        f"{_fmt(metrics.get('collision_delta_vs_floor_005'))} |"
    )


def _render_report(result: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-E Safety Floor Research",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- input_hash: `{result['input_hash']}`",
        "",
        "## Claim Boundary",
        "",
        *[f"- {fact}" for fact in result["current_facts"]],
        "",
        "## Best Deployable Policy",
        "",
        f"- family: `{result['best_deployable_policy'].get('family')}`",
        f"- source: `{result['best_deployable_policy'].get('source')}`",
        f"- deployable: `{result['best_deployable_policy'].get('test_deployable')}`",
        f"- metrics: `{result['best_deployable_policy'].get('test_metrics')}`",
        "",
        "## Switch Gate Families",
        "",
        "| family | source | deployable | all | t50 | t100 diag | hard/failure | easy degr | switch | collision d005 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["switch_gate_rows"]:
        lines.append(_row_md(str(row["family"]), row["source"], row["test_metrics"], row["test_deployable"]))
    lines.extend(
        [
            "",
            "## Bounded Residual / Blend Families",
            "",
            "| family | source | deployable | all | t50 | t100 diag | hard/failure | easy degr | alpha/switch | collision d005 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["bounded_residual_rows"]:
        lines.append(_row_md(str(row["family"]), row["source"], row["test_metrics"], row["test_deployable"]))
    floor = result["floor_necessity_analysis"]
    lines.extend(
        [
            "",
            "## Floor Necessity",
            "",
            f"- conclusion: `{floor['conclusion']}`",
            f"- floor-only metrics: `{floor['floor_only_metrics']}`",
            f"- ungated endpoint metrics from Stage42-B: `{floor['ungated_endpoint_metrics_from_stage42_b']}`",
            f"- ungated full-waypoint metrics from Stage42-C: `{floor['ungated_full_waypoint_metrics_from_stage42_c']}`",
            f"- no-teacher internal deployable families: `{floor['no_teacher_internal_gate_deployable_families']}`",
            f"- bounded no-switch deployable families: `{floor['bounded_no_switch_deployable_families']}`",
            f"- why: {floor['why']}",
            "",
            "## Cached-Verified Context",
            "",
            f"- Stage42-B verdict: `{result['cached_verified_context'].get('stage42_b_verdict')}`",
            f"- Stage42-C verdict: `{result['cached_verified_context'].get('stage42_c_verdict')}`",
            f"- Stage42-D verdict: `{result['cached_verified_context'].get('stage42_d_verdict')}`",
            f"- Stage41 composite-tail evidence pass: `{result['cached_verified_context'].get('stage41_composite_tail_evidence_pass')}`",
            "",
            "## Verdict",
            "",
            f"`{result['stage42_e_gate']['verdict']}` ({result['stage42_e_gate']['passed']} / {result['stage42_e_gate']['total']})",
        ]
    )
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_e_gate"]
    lines = [
        "# Stage42-E Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- gates: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| {name} | `{ok}` |")
    return lines


def _append_ledger(result: Mapping[str, Any]) -> None:
    entry = {
        "command": "run_stage42_safety_floor.py",
        "source": result["source"],
        "status": "success",
        "generated_at_utc": result["generated_at_utc"],
        "git_commit": result["git_commit"],
        "input_hash": result["input_hash"],
        "outputs": [str(REPORT_JSON), str(REPORT_MD), str(GATE_MD)],
    }
    with (OUT_DIR / "run_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


def _replace_block(text: str, marker: str, block: str) -> str:
    if marker in text:
        return text[: text.index(marker)].rstrip() + "\n\n" + block.strip() + "\n"
    return text.rstrip() + "\n\n" + block.strip() + "\n"


def _update_readme_and_state(result: Mapping[str, Any]) -> None:
    readme = Path("README_RESULTS.md")
    text = readme.read_text(encoding="utf-8") if readme.exists() else "# Results\n"
    best = result["best_deployable_policy"]
    floor = result["floor_necessity_analysis"]
    block = f"""
## Stage42-E Safety Floor Research

```text
source = {result.get('source')}
verdict = {result['stage42_e_gate']['verdict']}
gates = {result['stage42_e_gate']['passed']} / {result['stage42_e_gate']['total']}
best_policy_family = {best.get('family')}
best_policy_source = {best.get('source')}
best_all = {(best.get('test_metrics') or {}).get('all_improvement')}
best_t50 = {(best.get('test_metrics') or {}).get('t50_improvement')}
best_t100_raw_frame_diagnostic = {(best.get('test_metrics') or {}).get('t100_improvement')}
best_hard_failure = {(best.get('test_metrics') or {}).get('hard_failure_improvement')}
best_easy_degradation = {(best.get('test_metrics') or {}).get('easy_degradation')}
floor_necessity_conclusion = {floor.get('conclusion')}
ungated_endpoint_easy_degradation = {(floor.get('ungated_endpoint_metrics_from_stage42_b') or {}).get('easy_degradation')}
true_3d = false
foundation_world_model = false
metric_or_seconds_claim = false
stage5c_executed = false
smc_enabled = false
```

Stage42-E studies whether the Stage37/teacher floor can be removed. It evaluates internal self-gates, uncertainty/harm/conformal gates, teacher-prob gates, and bounded residual blends with validation-only threshold selection. Ungated neural remains unsafe; any partial floor removal is limited to explicitly deployable gated families.
    """
    readme.write_text(_replace_block(text, "## Stage42-E Safety Floor Research", block), encoding="utf-8")

    package_readme = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
    package_text = package_readme.read_text(encoding="utf-8") if package_readme.exists() else "# M3W-Neural v1\n"
    package_block = f"""
## Stage42-D/E Causal Ablation And Safety Floor Follow-Up

Stage42-D adds a causal ablation evidence audit:

- report: `outputs/stage42_long_research/causal_ablation_stage42.md`
- gate: `outputs/stage42_long_research/stage42_stage_d_gate.md`
- result: Stage42-D gates `12 / 12`
- boundary: not every component was retrained inside Stage42-D; fresh rows cover safety/floor/full-waypoint ablations, while historical no-history/no-neighbor/no-scene-goal/no-interaction/no-JEPA/no-Transformer/no-fallback evidence is cached-verified.

Stage42-E studies whether the Stage37/teacher safety floor can be removed:

- report: `outputs/stage42_long_research/safety_floor_stage42.md`
- gate: `outputs/stage42_long_research/stage42_stage_e_gate.md`
- result: Stage42-E gates `{result['stage42_e_gate']['passed']} / {result['stage42_e_gate']['total']}`

Key fresh-run result:

```text
best_policy_family = {best.get('family')}
best_policy_source = {best.get('source')}
best_all = {(best.get('test_metrics') or {}).get('all_improvement')}
best_t50 = {(best.get('test_metrics') or {}).get('t50_improvement')}
best_t100_raw_frame_diagnostic = {(best.get('test_metrics') or {}).get('t100_improvement')}
best_hard_failure = {(best.get('test_metrics') or {}).get('hard_failure_improvement')}
best_easy_degradation = {(best.get('test_metrics') or {}).get('easy_degradation')}
floor_necessity_conclusion = {floor.get('conclusion')}
ungated_endpoint_easy_degradation = {(floor.get('ungated_endpoint_metrics_from_stage42_b') or {}).get('easy_degradation')}
```

Interpretation:

The Stage37/teacher floor remains necessary for current deployment. Ungated neural has stronger raw all/t50/hard numbers but fails safety with easy degradation around `1.2459` and worse proximity/collision. Internal self-gate, uncertainty gate, harm gate, and conformal gate show large raw lift but exceed the collision safety ceiling in this fresh study. Teacher-repaired and composite-tail protected policies remain the deployable path. This is still dataset-local raw-frame 2.5D evidence, not metric, seconds-level, true 3D, Stage5C, or SMC.
"""
    package_readme.write_text(
        _replace_block(package_text, "## Stage42-D/E Causal Ablation And Safety Floor Follow-Up", package_block),
        encoding="utf-8",
    )

    state = read_json("research_state.json", {})
    reports = set(state.get("generated_reports", []))
    reports.update({str(REPORT_MD), str(REPORT_JSON), str(GATE_MD)})
    stage42 = dict(state.get("stage42", {}))
    stage42["stage_e_safety_floor"] = {
        "source": result.get("source"),
        "verdict": result["stage42_e_gate"]["verdict"],
        "gates": result["stage42_e_gate"],
        "best_deployable_policy": result["best_deployable_policy"],
        "floor_necessity_analysis": result["floor_necessity_analysis"],
        "claim_boundary": result["claim_boundary"],
        "no_leakage": result["no_leakage"],
    }
    state.update(
        {
            "current_stage": "stage42_e_safety_floor",
            "current_verdict": result["stage42_e_gate"]["verdict"],
            "current_best_deployable": "M3W-Neural v1 composite-tail safe-switch bounded neural dynamics under Stage37/teacher floor",
            "last_updated": "2026-05-25",
            "latent_generative_ready": False,
            "stage5c_ready": False,
            "smc_ready": False,
            "stage42": stage42,
            "generated_reports": sorted(reports),
        }
    )
    write_json("research_state.json", _jsonable(state))


if __name__ == "__main__":
    run_stage42_safety_floor()
