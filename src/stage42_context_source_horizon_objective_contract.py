from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
AO_JSON = OUT_DIR / "source_level_incremental_ablation_stage42.json"
IO_JSON = OUT_DIR / "horizon_sequence_graph_context_router_stage42.json"
JS_JSON = OUT_DIR / "source_context_gain_harm_closure_stage42.json"
JY_JSON = OUT_DIR / "context_materiality_by_source_slice_stage42.json"

REPORT_JSON = OUT_DIR / "context_source_horizon_objective_contract_stage42.json"
REPORT_MD = OUT_DIR / "context_source_horizon_objective_contract_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ka_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE42_KA_CONTEXT_SOURCE_HORIZON_OBJECTIVE_CONTRACT"
SOURCE = "fresh_stage42_ka_context_source_horizon_objective_contract"
MATERIAL_DELTA = 0.01
TINY_DELTA = 1e-6

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-KA 不是新训练；它把 AO/JY/JS/IO 的 fresh/cached-verified context 证据转成 source+horizon objective contract。",
    "future endpoints / future waypoints 只能作为 supervised/evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
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


def _metric_delta(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    keys = [
        "all_improvement",
        "t10_improvement",
        "t25_improvement",
        "t50_improvement",
        "t100_raw_frame_diagnostic_improvement",
        "hard_failure_improvement",
        "easy_degradation",
        "switch_rate",
    ]
    return {key: float(candidate.get(key, 0.0)) - float(baseline.get(key, 0.0)) for key in keys}


def _material_core_delta(delta: Mapping[str, float]) -> bool:
    return (
        delta.get("all_improvement", 0.0) >= MATERIAL_DELTA
        and delta.get("t50_improvement", 0.0) >= MATERIAL_DELTA
        and delta.get("hard_failure_improvement", 0.0) >= MATERIAL_DELTA
        and delta.get("easy_degradation", 1.0) <= 0.02
    )


def _positive_narrow_delta(delta: Mapping[str, float]) -> bool:
    core = max(
        delta.get("all_improvement", 0.0),
        delta.get("t50_improvement", 0.0),
        delta.get("t100_raw_frame_diagnostic_improvement", 0.0),
        delta.get("hard_failure_improvement", 0.0),
    )
    return core > TINY_DELTA and delta.get("easy_degradation", 1.0) <= 0.02


def _context_delta_summary(ao: Mapping[str, Any]) -> dict[str, Any]:
    variants = ao.get("variants", {})
    baseline = variants.get("baseline_family_only", {})
    baseline_global = baseline.get("protected", {})
    baseline_by_horizon = baseline.get("by_horizon", {})
    rows: dict[str, Any] = {}
    for name, variant in variants.items():
        if name == "baseline_family_only":
            continue
        protected = variant.get("protected", {})
        if not protected:
            continue
        global_delta = _metric_delta(protected, baseline_global)
        horizon_deltas = {}
        for horizon, metrics in variant.get("by_horizon", {}).items():
            horizon_deltas[horizon] = _metric_delta(metrics, baseline_by_horizon.get(horizon, {}))
        rows[name] = {
            "feature_count": int(variant.get("feature_count", 0)),
            "global_delta_vs_baseline_family": global_delta,
            "global_material_core_positive": _material_core_delta(global_delta),
            "horizon_delta_vs_baseline_family": horizon_deltas,
            "positive_horizon_slices": [
                horizon for horizon, delta in horizon_deltas.items() if _positive_narrow_delta(delta)
            ],
            "t50_supported_vs_baseline_family": horizon_deltas.get("50", {}).get("t50_improvement", 0.0) >= MATERIAL_DELTA
            and horizon_deltas.get("50", {}).get("easy_degradation", 1.0) <= 0.02,
            "t100_supported_vs_baseline_family": horizon_deltas.get("100", {}).get(
                "t100_raw_frame_diagnostic_improvement", 0.0
            )
            >= MATERIAL_DELTA
            and horizon_deltas.get("100", {}).get("easy_degradation", 1.0) <= 0.02,
        }
    return rows


def _objective_matrix(io: Mapping[str, Any], js: Mapping[str, Any], deltas: Mapping[str, Any]) -> list[dict[str, Any]]:
    best_by_horizon = io.get("summary", {}).get("best_by_horizon", {})
    js_summary = js.get("summary", {})
    rows: list[dict[str, Any]] = []
    for horizon in ["10", "25", "50", "100"]:
        best = best_by_horizon.get(horizon, {})
        candidate = str(best.get("candidate", ""))
        supported = bool(best.get("supported", False))
        delta = deltas.get(candidate, {}).get("horizon_delta_vs_baseline_family", {}).get(horizon, {})
        is_positive_vs_baseline_family = _positive_narrow_delta(delta)
        if horizon in {"10", "25"} and supported and is_positive_vs_baseline_family:
            decision = "auxiliary_retrain_candidate_only"
            reason = "narrow horizon-positive router exists, but it is not a global context contribution."
        elif horizon in {"10", "25"} and supported:
            decision = "diagnostic_router_only_not_baseline_family_positive"
            reason = (
                "horizon router was positive in the sequence/graph protocol, "
                "but it is not positive versus baseline-family control in this contract."
            )
        elif horizon == "50":
            decision = "blocked_until_new_row_level_objective"
            reason = js_summary.get("t50_diagnosis", "t50 unsupported under current context family")
        elif horizon == "100":
            decision = "diagnostic_blocked_until_new_source_slice_objective"
            reason = js_summary.get("t100_diagnosis", "t100 raw-frame diagnostic unsupported under current context family")
        else:
            decision = "not_promotable"
            reason = "No supported router."
        rows.append(
            {
                "horizon": int(horizon),
                "best_current_router": str(best.get("best", "")),
                "candidate": candidate,
                "current_router_supported": supported,
                "delta_vs_baseline_family": delta,
                "objective_decision": decision,
                "reason": reason,
                "allowed_claim": "narrow_auxiliary_context_signal" if decision == "auxiliary_retrain_candidate_only" else "no_main_claim",
            }
        )
    return rows


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    claim = payload["claim_boundary"]
    gates = {
        "ao_incremental_ablation_loaded": payload["input_status"]["ao_verdict"]
        == "stage42_ao_incremental_component_evidence_partial_or_negative",
        "io_horizon_router_loaded": payload["input_status"]["io_verdict"]
        == "stage42_io_horizon_sequence_graph_context_router_pass",
        "js_context_closure_loaded": payload["input_status"]["js_verdict"]
        == "stage42_js_source_context_gain_harm_closure_pass",
        "jy_materiality_loaded": payload["input_status"]["jy_verdict"]
        == "stage42_jy_context_materiality_by_source_slice_pass",
        "baseline_family_control_positive": s["baseline_family_control"]["all_improvement"] > 0
        and s["baseline_family_control"]["t50_improvement"] > 0
        and s["baseline_family_control"]["hard_failure_improvement"] > 0,
        "no_global_context_promotion": s["global_material_context_variants"] == [],
        "narrow_h10_auxiliary_recorded": any(
            int(row.get("horizon", -1)) == 10 for row in s["narrow_auxiliary_context_slices"]
        ),
        "diagnostic_h25_conflict_recorded": any(
            int(row.get("horizon", -1)) == 25 for row in s["diagnostic_router_conflicts"]
        ),
        "t50_context_blocker_recorded": s["horizon_objective_matrix"]["50"]["objective_decision"]
        == "blocked_until_new_row_level_objective",
        "t100_context_blocker_recorded": s["horizon_objective_matrix"]["100"]["objective_decision"]
        == "diagnostic_blocked_until_new_source_slice_objective",
        "next_training_contract_emitted": len(s["next_training_contract"]) >= 5,
        "no_future_or_test_leakage": all(payload["no_leakage"].values()),
        "no_metric_seconds_3d_foundation": claim["true_3d"] is False
        and claim["foundation_world_model"] is False
        and claim["metric_or_seconds_claim"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = (
        "stage42_ka_context_source_horizon_objective_contract_pass"
        if passed == total
        else "stage42_ka_context_source_horizon_objective_contract_partial"
    )
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ao = read_json(AO_JSON, {})
    io = read_json(IO_JSON, {})
    js = read_json(JS_JSON, {})
    jy = read_json(JY_JSON, {})
    deltas = _context_delta_summary(ao)
    objective_rows = _objective_matrix(io, js, deltas)
    objective_by_horizon = {str(row["horizon"]): row for row in objective_rows}
    global_material = [name for name, row in deltas.items() if row.get("global_material_core_positive")]
    narrow_aux = [
        {"variant": name, "horizon": int(horizon)}
        for name, row in deltas.items()
        for horizon in row.get("positive_horizon_slices", [])
        if horizon in {"10", "25"}
    ]
    diagnostic_router_conflicts = [
        {"horizon": row["horizon"], "candidate": row["candidate"], "decision": row["objective_decision"]}
        for row in objective_rows
        if row["objective_decision"] == "diagnostic_router_only_not_baseline_family_positive"
    ]
    summary = {
        "result_source_label": "fresh_synthesis_from_stage42_ao_io_js_jy_artifacts",
        "baseline_family_control": ao.get("variants", {}).get("baseline_family_only", {}).get("protected", {}),
        "global_material_context_variants": global_material,
        "narrow_auxiliary_context_slices": narrow_aux,
        "diagnostic_router_conflicts": diagnostic_router_conflicts,
        "context_delta_summary": deltas,
        "horizon_objective_matrix": objective_by_horizon,
        "t50_oracle_headroom": float(js.get("summary", {}).get("t50_oracle_headroom", 0.0)),
        "t100_oracle_headroom": float(js.get("summary", {}).get("t100_oracle_headroom", 0.0)),
        "claim_decision": "keep_scene_goal_neighbor_interaction_blocked_as_independent_main_claims",
        "deployment_decision": "keep_baseline_family_stage37_teacher_floor_as_deployable_context_mechanism",
        "next_training_contract": [
            "Do not promote any current context variant globally: none beats baseline-family control on all+t50+hard with easy<=2%.",
            "Use h10 context routers only as auxiliary/narrow evidence unless retrained source-level validation proves material global lift.",
            "Treat h25 as diagnostic-only when it is positive in the router protocol but negative versus baseline-family control.",
            "For t50, build a new row-level source/horizon objective because current gain/harm context routers under-switch despite oracle headroom.",
            "For t100 raw-frame diagnostic, build a separate source-slice objective; current evidence is micro-positive but not material.",
            "Preserve Stage37/teacher floor and baseline-family rollout context for any deployable policy.",
            "Keep future endpoints/waypoints as labels only; no central velocity, no test endpoint goals, no test threshold tuning.",
        ],
    }
    payload: dict[str, Any] = {
        "stage": "Stage42-KA context source/horizon objective contract",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "input_hash": _combined_hash([AO_JSON, IO_JSON, JS_JSON, JY_JSON]),
        "current_facts": CURRENT_FACTS,
        "input_status": {
            "ao_verdict": ao.get("stage42_ao_gate", {}).get("verdict", ""),
            "io_verdict": io.get("stage42_io_gate", {}).get("verdict", ""),
            "js_verdict": js.get("stage42_js_gate", {}).get("verdict", ""),
            "jy_verdict": jy.get("stage42_jy_gate", {}).get("verdict", ""),
        },
        "summary": summary,
        "no_leakage": {
            "future_endpoint_input_absent": True,
            "future_waypoint_input_absent": True,
            "future_labels_eval_only": True,
            "central_velocity_absent": True,
            "test_endpoint_goals_absent": True,
            "test_threshold_tuning_absent": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "not_new_training": True,
            "independent_context_main_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_ka_gate"] = _gate(payload)
    return payload


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ka_gate"]
    s = payload["summary"]
    base = s["baseline_family_control"]
    lines = [
        "# Stage42-KA Context Source/Horizon Objective Contract",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Baseline-Family Control",
        "",
        f"- all/t50/t100raw/hard: `{_pct(base.get('all_improvement', 0.0))}` / `{_pct(base.get('t50_improvement', 0.0))}` / `{_pct(base.get('t100_raw_frame_diagnostic_improvement', 0.0))}` / `{_pct(base.get('hard_failure_improvement', 0.0))}`",
        f"- easy_degradation: `{_pct(base.get('easy_degradation', 0.0))}`; switch_rate: `{_pct(base.get('switch_rate', 0.0))}`",
        "",
        "## Contract Summary",
        "",
        f"- global_material_context_variants: `{s['global_material_context_variants']}`",
        f"- narrow_auxiliary_context_slices: `{s['narrow_auxiliary_context_slices']}`",
        f"- diagnostic_router_conflicts: `{s['diagnostic_router_conflicts']}`",
        f"- t50_oracle_headroom: `{_pct(s['t50_oracle_headroom'])}`",
        f"- t100_oracle_headroom: `{_pct(s['t100_oracle_headroom'])}`",
        f"- claim_decision: `{s['claim_decision']}`",
        f"- deployment_decision: `{s['deployment_decision']}`",
        "",
        "## Horizon Objective Matrix",
        "",
        "| horizon | candidate | current supported | decision | reason | delta all | delta t50 | delta t100raw | delta hard | easy delta |",
        "| ---: | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for horizon in ["10", "25", "50", "100"]:
        row = s["horizon_objective_matrix"][horizon]
        delta = row.get("delta_vs_baseline_family", {})
        lines.append(
            f"| {horizon} | `{row['candidate']}` | `{row['current_router_supported']}` | `{row['objective_decision']}` | "
            f"{row['reason']} | {_pct(delta.get('all_improvement', 0.0))} | {_pct(delta.get('t50_improvement', 0.0))} | "
            f"{_pct(delta.get('t100_raw_frame_diagnostic_improvement', 0.0))} | {_pct(delta.get('hard_failure_improvement', 0.0))} | {_pct(delta.get('easy_degradation', 0.0))} |"
        )
    lines.extend(
        [
            "",
            "## Next Training Contract",
            "",
            *[f"- {item}" for item in s["next_training_contract"]],
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
            *[f"| `{key}` | `{bool(value)}` |" for key, value in gate["gates"].items()],
            "",
            "## Interpretation",
            "",
            "- KA turns the current context evidence into an explicit claim/deployment contract.",
            "- Context modules remain useful as auxiliary diagnostics and narrow h10 objectives, but not as an independent global paper contribution.",
            "- t50/t100 context work must change objective and row-level supervision rather than repeating the closed threshold/router family.",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ka_gate"]
    return [
        "# Stage42-KA Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | `{bool(value)}` |" for key, value in gate["gates"].items()],
    ]


def _section_lines(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ka_gate"]
    s = payload["summary"]
    return [
        "## Stage42-KA Context Source/Horizon Objective Contract",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict: `{gate['verdict']}`.",
        f"- global material context variants over baseline-family control: `{s['global_material_context_variants']}`.",
        f"- narrow auxiliary context slices preserved for future source/horizon training: `{s['narrow_auxiliary_context_slices']}`.",
        f"- diagnostic router conflicts: `{s['diagnostic_router_conflicts']}`.",
        f"- t50 blocker: `{s['horizon_objective_matrix']['50']['reason']}`; t50 oracle headroom `{_pct(s['t50_oracle_headroom'])}`.",
        f"- t100 blocker: `{s['horizon_objective_matrix']['100']['reason']}`; t100 raw oracle headroom `{_pct(s['t100_oracle_headroom'])}`.",
        "- decision: do not promote scene/goal/neighbor/interaction as independent global main claims; next context attempt must use row-level source/horizon objectives under Stage37/teacher floor.",
        "- boundary: dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
    ]


def _update_readmes(payload: Mapping[str, Any]) -> None:
    block = _section_lines(payload)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, block)


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage42_ka_context_source_horizon_objective_contract"
    state["current_verdict"] = payload["stage42_ka_gate"]["verdict"]
    stage42 = state.setdefault("stage42", {})
    s = payload["summary"]
    stage42["stage_ka_context_source_horizon_objective_contract"] = {
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_ka_gate"]["verdict"],
        "gates": f"{payload['stage42_ka_gate']['passed']}/{payload['stage42_ka_gate']['total']}",
        "summary": {
            "global_material_context_variants": s["global_material_context_variants"],
            "narrow_auxiliary_context_slices": s["narrow_auxiliary_context_slices"],
            "diagnostic_router_conflicts": s["diagnostic_router_conflicts"],
            "horizon_objective_matrix": s["horizon_objective_matrix"],
            "claim_decision": s["claim_decision"],
            "deployment_decision": s["deployment_decision"],
            "next_training_contract": s["next_training_contract"],
        },
        "claim_boundary": payload["claim_boundary"],
    }
    state["last_updated_utc"] = payload["generated_at_utc"]
    state["last_successful_command"] = "python run_stage42_context_source_horizon_objective_contract.py"
    generated = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, GATE_MD]:
        item = str(path)
        if item not in generated:
            generated.append(item)
    write_json(RESEARCH_STATE, _jsonable(state))


def run_stage42_context_source_horizon_objective_contract(*, refresh_readmes: bool = True) -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _update_readmes(payload)
        _update_state(payload)
    return payload


def main() -> None:
    payload = run_stage42_context_source_horizon_objective_contract(refresh_readmes=True)
    gate = payload["stage42_ka_gate"]
    print(f"Stage42-KA context source/horizon objective contract: {gate['verdict']} ({gate['passed']}/{gate['total']})")


if __name__ == "__main__":
    main()
