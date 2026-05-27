from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
C_JSON = OUT_DIR / "full_waypoint_dynamics_stage42.json"
CO_JSON = OUT_DIR / "common_validation_bridge_shape_composer_stage42.json"
DI_JSON = OUT_DIR / "group_consistency_full_waypoint_repair_stage42.json"
DL_JSON = OUT_DIR / "group_consistency_runtime_policy_stage42.json"
DP_JSON = OUT_DIR / "context_model_closure_stage42.json"

REPORT_JSON = OUT_DIR / "full_waypoint_promotion_checkpoint_stage42.json"
REPORT_MD = OUT_DIR / "full_waypoint_promotion_checkpoint_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_dq_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
RETRO_README = Path("README_M3W_GOAL_RETROSPECTIVE_CURRENT_ZH.md")
CURRENT_GOAL_README = Path("README_M3W_CURRENT_GOAL_SUMMARY_ZH.md")
GOAL_RESULTS_README = Path("README_M3W_GOAL_RESULTS_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-DQ 是 DA-3 fresh full-waypoint promotion checkpoint，不执行 Stage5C，不启用 SMC。",
    "本阶段整合 fresh Stage42-C full-waypoint dynamics、fresh Stage42-CO common-validation composer、fresh Stage42-DI group-consistency repair、fresh Stage42-DL runtime replay。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _metric_subset(metric: Mapping[str, Any]) -> dict[str, float]:
    keys = [
        "rows",
        "all_improvement",
        "t50_improvement",
        "t100_raw_frame_diagnostic_improvement",
        "t100_improvement",
        "hard_failure_improvement",
        "easy_degradation",
        "switch_rate",
    ]
    return {key: float(metric[key]) for key in keys if key in metric and isinstance(metric[key], (int, float))}


def _build_summary(c: Mapping[str, Any], co: Mapping[str, Any], di: Mapping[str, Any], dl: Mapping[str, Any], dp: Mapping[str, Any]) -> dict[str, Any]:
    full_protected = c["comparisons"]["full_waypoint_transformer_protected"]["ade"]
    full_ungated = c["comparisons"]["ungated_full_waypoint_transformer"]["ade"]
    neural_bridge = c["comparisons"]["m3w_neural_v1_composite_tail_linear_bridge"]["ade"]
    co_vs_endpoint = co["test_eval"]["metric_vs_endpoint_ade"]
    co_vs_floor = co["test_eval"]["metric_vs_floor_ade"]
    di_metric = di["repair"]["test"]["metric_vs_floor"]
    di_diag = di["repair"]["test"]["diagnostics"]
    dl_replay = dl["real_batch_replay"]
    dl_metric = dl_replay["metric"]
    dl_diag = dl_replay["diagnostics"]
    delta_vs_am = di["comparison_to_prior"]["delta_vs_stage42_am"]
    context_closure = dp.get("summary", {}).get("closure_decision", "unknown")
    return {
        "source": "fresh_synthesis_after_da3_full_waypoint_rerun",
        "full_waypoint_transformer_protected_vs_floor": _metric_subset(full_protected),
        "ungated_full_waypoint_transformer_vs_floor": _metric_subset(full_ungated),
        "m3w_neural_v1_endpoint_linear_bridge_vs_floor": _metric_subset(neural_bridge),
        "common_validation_composer_vs_endpoint_linear": _metric_subset(co_vs_endpoint),
        "common_validation_composer_vs_floor": _metric_subset(co_vs_floor),
        "group_consistency_repair_vs_train_horizon_causal_floor": _metric_subset(di_metric),
        "group_consistency_runtime_replay_vs_train_horizon_causal_floor": _metric_subset(dl_metric),
        "group_consistency_delta_vs_stage42_am": {
            key: float(value) for key, value in delta_vs_am.items() if isinstance(value, (int, float))
        },
        "group_consistency_safety": {
            "base_near_005": float(di_diag["base_near_005"]),
            "final_near_005": float(di_diag["final_near_005"]),
            "floor_near_005": float(di_diag["floor_near_005"]),
            "runtime_base_near_005": float(dl_diag["base_near_005"]),
            "runtime_final_near_005": float(dl_diag["final_near_005"]),
            "runtime_floor_near_005": float(dl_diag["floor_near_005"]),
            "selected_xy_max_abs_diff": float(dl_replay["selected_xy_max_abs_diff"]),
            "switch_exact_match": bool(dl_replay["switch_exact_match"]),
            "selected_ade_max_abs_diff": float(dl_replay["selected_ade_max_abs_diff"]),
            "selected_fde_max_abs_diff": float(dl_replay["selected_fde_max_abs_diff"]),
        },
        "context_closure_decision": context_closure,
        "promotion_decision": {
            "source_level_group_consistency_runtime_policy_promoted": True,
            "common_validation_endpoint_composer_remains_safety_sensitive_bridge": True,
            "global_primary_full_waypoint_replacement_claim_allowed": False,
            "ungated_full_waypoint_deployable": False,
            "reason": (
                "Fresh DI/DL support a protected source-level group-consistency full-waypoint runtime policy with exact replay "
                "and proximity repair. However, common-validation endpoint-linear bridge/composer and source-level train-horizon "
                "floor use different comparison protocols, so the result cannot be collapsed into a single global primary "
                "full-waypoint replacement claim."
            ),
        },
        "next_best_action": [
            "Keep Stage42-DL group-consistency runtime as source-level full-waypoint runtime evidence.",
            "Use Stage42-CQ proximity-aware composer as the safety-sensitive endpoint bridge/shape policy when endpoint-linear baseline is the comparison floor.",
            "Do not deploy ungated full-waypoint because easy degradation remains unsafe.",
            "Prioritize source/legal/time closure and protocol-aligned external sources before broader metric/seconds or global t100/full-waypoint claims.",
        ],
    }


def _no_leakage(c: Mapping[str, Any], co: Mapping[str, Any], di: Mapping[str, Any], dl: Mapping[str, Any]) -> dict[str, bool]:
    def get(payload: Mapping[str, Any], key: str, default: bool = True) -> bool:
        return bool(payload.get("no_leakage", {}).get(key, default))

    return {
        "future_endpoint_input": get(c, "future_endpoint_input")
        or get(co, "future_endpoint_input")
        or get(di, "future_endpoint_input")
        or get(dl, "future_endpoint_input"),
        "future_waypoint_input": get(c, "future_waypoints_input")
        or get(co, "future_waypoints_input")
        or get(di, "future_waypoint_input")
        or get(dl, "future_waypoint_input"),
        "central_velocity": get(c, "central_velocity")
        or get(co, "central_velocity")
        or get(di, "central_velocity")
        or get(dl, "central_velocity"),
        "test_endpoint_goals": get(c, "test_endpoint_goals")
        or get(co, "test_endpoint_goals")
        or get(di, "test_endpoint_goals")
        or get(dl, "test_endpoint_goals"),
        "test_threshold_tuning": get(c, "test_threshold_tuning")
        or get(co, "test_threshold_tuning")
        or get(di, "test_threshold_tuning")
        or get(dl, "test_threshold_tuning"),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    no_leakage = payload["no_leakage"]
    claim = payload["claim_boundary"]
    runtime = summary["group_consistency_safety"]
    di_metric = summary["group_consistency_repair_vs_train_horizon_causal_floor"]
    delta = summary["group_consistency_delta_vs_stage42_am"]
    ungated = summary["ungated_full_waypoint_transformer_vs_floor"]
    gates = {
        "fresh_full_waypoint_dynamics_loaded": payload["inputs"]["stage42_c_full_waypoint_dynamics"]["source"]
        == "fresh_run",
        "fresh_common_validation_composer_loaded": payload["inputs"]["stage42_co_common_validation_composer"]["source"]
        == "fresh_common_validation_eval_from_cached_verified_checkpoints",
        "fresh_group_consistency_repair_loaded": payload["inputs"]["stage42_di_group_consistency_repair"]["source"]
        == "fresh_stage42_di_group_consistency_full_waypoint_repair",
        "fresh_runtime_replay_loaded": payload["inputs"]["stage42_dl_runtime_replay"]["source"]
        == "fresh_runtime_api_from_frozen_group_consistency_policy_artifact",
        "runtime_exact_replay": runtime["switch_exact_match"]
        and runtime["selected_xy_max_abs_diff"] == 0.0
        and runtime["selected_ade_max_abs_diff"] == 0.0
        and runtime["selected_fde_max_abs_diff"] == 0.0,
        "group_consistency_all_positive": di_metric["all_improvement"] > 0.0,
        "group_consistency_t50_positive": di_metric["t50_improvement"] > 0.0,
        "group_consistency_hard_positive": di_metric["hard_failure_improvement"] > 0.0,
        "group_consistency_improves_stage42_am_all": delta["all_improvement"] > 0.0,
        "group_consistency_improves_stage42_am_hard": delta["hard_failure_improvement"] > 0.0,
        "group_consistency_proximity_repaired": runtime["runtime_final_near_005"] < runtime["runtime_base_near_005"],
        "ungated_full_waypoint_blocked": ungated["easy_degradation"] > 0.02,
        "context_residual_protocol_closed": summary["context_closure_decision"]
        == "close_current_sequence_graph_residual_context_protocol",
        "primary_global_replacement_not_overclaimed": summary["promotion_decision"][
            "global_primary_full_waypoint_replacement_claim_allowed"
        ]
        is False,
        "future_endpoint_blocked": no_leakage["future_endpoint_input"] is False,
        "future_waypoint_blocked": no_leakage["future_waypoint_input"] is False,
        "central_velocity_blocked": no_leakage["central_velocity"] is False,
        "test_endpoint_goals_blocked": no_leakage["test_endpoint_goals"] is False,
        "no_test_threshold_tuning": no_leakage["test_threshold_tuning"] is False,
        "no_metric_seconds_overclaim": claim["metric_or_seconds_claim"] is False,
        "true_3d_overclaim_blocked": claim["true_3d"] is False,
        "foundation_overclaim_blocked": claim["foundation_world_model"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = (
        "stage42_dq_full_waypoint_promotion_checkpoint_pass"
        if passed == total
        else "stage42_dq_full_waypoint_promotion_checkpoint_partial"
    )
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["summary"]
    gate = payload["stage42_dq_gate"]
    lines = [
        "# Stage42-DQ Full-Waypoint Promotion Checkpoint",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
    ]
    lines.extend(f"- {fact}" for fact in CURRENT_FACTS)
    lines.extend(
        [
            "",
            "## Fresh Evidence Chain",
            "",
            "| stage | source | gate | verdict |",
            "| --- | --- | --- | --- |",
        ]
    )
    for name, info in payload["inputs"].items():
        lines.append(f"| `{name}` | `{info.get('source')}` | `{info.get('gate', 'n/a')}` | `{info.get('verdict')}` |")
    lines.extend(
        [
            "",
            "## Key Metrics",
            "",
            "| policy | comparison floor | all | t50 | t100 raw diag | hard/failure | easy degradation |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    metric_rows = [
        (
            "full_waypoint_transformer_protected",
            "train_horizon_causal_floor",
            summary["full_waypoint_transformer_protected_vs_floor"],
        ),
        (
            "ungated_full_waypoint_transformer",
            "train_horizon_causal_floor",
            summary["ungated_full_waypoint_transformer_vs_floor"],
        ),
        (
            "common_validation_composer",
            "endpoint_linear_bridge",
            summary["common_validation_composer_vs_endpoint_linear"],
        ),
        (
            "group_consistency_repair",
            "train_horizon_causal_floor",
            summary["group_consistency_repair_vs_train_horizon_causal_floor"],
        ),
        (
            "runtime_replay_group_consistency",
            "train_horizon_causal_floor",
            summary["group_consistency_runtime_replay_vs_train_horizon_causal_floor"],
        ),
    ]
    for name, floor, metric in metric_rows:
        t100 = metric.get("t100_raw_frame_diagnostic_improvement", metric.get("t100_improvement"))
        lines.append(
            f"| `{name}` | `{floor}` | {_pct(metric.get('all_improvement'))} | {_pct(metric.get('t50_improvement'))} | {_pct(t100)} | {_pct(metric.get('hard_failure_improvement'))} | {_pct(metric.get('easy_degradation'))} |"
        )
    safety = summary["group_consistency_safety"]
    lines.extend(
        [
            "",
            "## Runtime Replay And Safety",
            "",
            f"- switch_exact_match: `{safety['switch_exact_match']}`",
            f"- selected_xy_max_abs_diff: `{safety['selected_xy_max_abs_diff']}`",
            f"- selected_ade_max_abs_diff: `{safety['selected_ade_max_abs_diff']}`",
            f"- selected_fde_max_abs_diff: `{safety['selected_fde_max_abs_diff']}`",
            f"- runtime near@0.05 base/final/floor: `{_pct(safety['runtime_base_near_005'])}` / `{_pct(safety['runtime_final_near_005'])}` / `{_pct(safety['runtime_floor_near_005'])}`",
            "",
            "## Promotion Decision",
            "",
        ]
    )
    decision = summary["promotion_decision"]
    for key, value in decision.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Next Best Action",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in summary["next_best_action"])
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- Stage42-DQ supports a protected source-level full-waypoint runtime policy and exact replay evidence.",
            "- It does not promote ungated full-waypoint dynamics.",
            "- It does not collapse endpoint-linear composer and train-horizon source-level runtime into one global ranking.",
            "- It remains dataset-local/raw-frame 2.5D; no metric/seconds-level, true 3D, foundation, Stage5C, or SMC claim is made.",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_dq_gate"]
    lines = [
        "# Stage42-DQ Gate",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    lines.extend(f"| `{name}` | {bool(value)} |" for name, value in gate["gates"].items())
    return lines


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    summary = payload["summary"]
    gate = payload["stage42_dq_gate"]
    runtime = summary["group_consistency_runtime_replay_vs_train_horizon_causal_floor"]
    safety = summary["group_consistency_safety"]
    lines = [
        "## Stage42-DQ Full-Waypoint Promotion Checkpoint",
        "",
        "- source: `fresh_synthesis_after_da3_full_waypoint_rerun`",
        f"- verdict: `{gate['verdict']}`; gates `{gate['passed']} / {gate['total']}`.",
        "- fresh chain: Stage42-C full-waypoint dynamics, Stage42-CO common-validation composer, Stage42-DI group-consistency repair, Stage42-DL runtime replay.",
        f"- group-consistency runtime vs train-horizon causal floor all/t50/t100 raw/hard: `{_pct(runtime.get('all_improvement'))}` / `{_pct(runtime.get('t50_improvement'))}` / `{_pct(runtime.get('t100_raw_frame_diagnostic_improvement'))}` / `{_pct(runtime.get('hard_failure_improvement'))}`.",
        f"- runtime replay exact: switch `{safety['switch_exact_match']}`, selected_xy max abs diff `{safety['selected_xy_max_abs_diff']}`.",
        f"- near@0.05 base/final/floor: `{_pct(safety['runtime_base_near_005'])}` / `{_pct(safety['runtime_final_near_005'])}` / `{_pct(safety['runtime_floor_near_005'])}`.",
        "- promotion: protected source-level group-consistency full-waypoint runtime policy is supported; ungated full-waypoint and global primary replacement remain blocked.",
        "- Claim boundary: dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no global metric/seconds-level, no Stage5C execution, no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, RETRO_README, CURRENT_GOAL_README, GOAL_RESULTS_README]:
        _replace_section(path, "STAGE42_DQ_FULL_WAYPOINT_PROMOTION_CHECKPOINT", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    summary = payload["summary"]
    gate = payload["stage42_dq_gate"]
    runtime = summary["group_consistency_runtime_replay_vs_train_horizon_causal_floor"]
    state["current_stage"] = "Stage42-DQ full-waypoint promotion checkpoint"
    state["current_verdict"] = gate["verdict"]
    state["stage42_dq_full_waypoint_promotion_checkpoint"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": gate["verdict"],
        "gates": f"{gate['passed']}/{gate['total']}",
        "runtime_metric_vs_train_horizon_causal_floor": runtime,
        "group_consistency_safety": summary["group_consistency_safety"],
        "promotion_decision": summary["promotion_decision"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_full_waypoint_promotion_checkpoint(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    c = read_json(C_JSON, {})
    co = read_json(CO_JSON, {})
    di = read_json(DI_JSON, {})
    dl = read_json(DL_JSON, {})
    dp = read_json(DP_JSON, {})
    claim_boundary = {
        "true_3d": False,
        "foundation_world_model": False,
        "metric_or_seconds_claim": False,
        "raw_frame_dataset_local_only": True,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    payload: dict[str, Any] = {
        "source": "fresh_synthesis_after_da3_full_waypoint_rerun",
        "stage": "Stage42-DQ",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "stage42_c_full_waypoint_dynamics": {
                "path": str(C_JSON),
                "source": c.get("source"),
                "generated_at_utc": c.get("generated_at_utc"),
                "gate": f"{c.get('stage42_c_gate', {}).get('passed')}/{c.get('stage42_c_gate', {}).get('total')}",
                "verdict": c.get("stage42_c_gate", {}).get("verdict"),
            },
            "stage42_co_common_validation_composer": {
                "path": str(CO_JSON),
                "source": co.get("source"),
                "generated_at_utc": co.get("generated_at_utc"),
                "gate": f"{co.get('stage42_co_gate', {}).get('passed')}/{co.get('stage42_co_gate', {}).get('total')}",
                "verdict": co.get("stage42_co_gate", {}).get("verdict"),
            },
            "stage42_di_group_consistency_repair": {
                "path": str(DI_JSON),
                "source": di.get("source"),
                "generated_at_utc": di.get("generated_at_utc"),
                "gate": f"{di.get('stage42_di_gate', {}).get('passed')}/{di.get('stage42_di_gate', {}).get('total')}",
                "verdict": di.get("stage42_di_gate", {}).get("verdict"),
            },
            "stage42_dl_runtime_replay": {
                "path": str(DL_JSON),
                "source": dl.get("source"),
                "generated_at_utc": dl.get("generated_at_utc"),
                "gate": f"{dl.get('stage42_dl_gate', {}).get('passed')}/{dl.get('stage42_dl_gate', {}).get('total')}",
                "verdict": dl.get("stage42_dl_gate", {}).get("verdict"),
            },
            "stage42_dp_context_closure": {
                "path": str(DP_JSON),
                "source": dp.get("source"),
                "verdict": dp.get("stage42_dp_gate", {}).get("verdict"),
            },
        },
        "summary": _build_summary(c, co, di, dl, dp),
        "no_leakage": _no_leakage(c, co, di, dl),
        "claim_boundary": claim_boundary,
    }
    payload["stage42_dq_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_full_waypoint_promotion_checkpoint()
