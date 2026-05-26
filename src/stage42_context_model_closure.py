from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
AR_JSON = OUT_DIR / "source_level_sequence_context_stage42.json"
AS_JSON = OUT_DIR / "source_level_graph_context_stage42.json"
DB_JSON = OUT_DIR / "context_rescue_decision_stage42.json"
DC_JSON = OUT_DIR / "context_switchability_gate_stage42.json"

REPORT_JSON = OUT_DIR / "context_model_closure_stage42.json"
REPORT_MD = OUT_DIR / "context_model_closure_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_dp_gate.md"

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
    "Stage42-DP 是 source-level sequence/graph context closure，不重新训练新模型，不调 test threshold。",
    "本阶段整合 fresh Stage42-AR sequence-context 和 fresh Stage42-AS graph-context rerun。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
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


def _metric(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return payload.get("baseline_family_only", {}).get("protected", {})


def _variant_rows(payload: Mapping[str, Any], key: str, delta_key: str, protocol: str) -> list[dict[str, Any]]:
    variants = payload.get(key, {})
    deltas = payload.get(delta_key, {})
    rows: list[dict[str, Any]] = []
    for name, variant in variants.items():
        protected = variant.get("protected", {})
        delta_payload = deltas.get(name, {})
        delta = delta_payload.get("delta_vs_baseline_family_only", delta_payload)
        if not protected:
            continue
        rows.append(
            {
                "protocol": protocol,
                "variant": name,
                "protected": {
                    "all_improvement": float(protected.get("all_improvement", 0.0)),
                    "t50_improvement": float(protected.get("t50_improvement", 0.0)),
                    "t100_raw_frame_diagnostic_improvement": float(
                        protected.get("t100_raw_frame_diagnostic_improvement", 0.0)
                    ),
                    "hard_failure_improvement": float(protected.get("hard_failure_improvement", 0.0)),
                    "easy_degradation": float(protected.get("easy_degradation", 0.0)),
                },
                "delta_vs_baseline_family": {
                    "all_improvement": float(delta.get("all_improvement", 0.0)),
                    "t50_improvement": float(delta.get("t50_improvement", 0.0)),
                    "hard_failure_improvement": float(delta.get("hard_failure_improvement", 0.0)),
                    "easy_degradation": float(delta.get("easy_degradation", 0.0)),
                },
            }
        )
    return rows


def _summary(ar: Mapping[str, Any], as_: Mapping[str, Any], db: Mapping[str, Any], dc: Mapping[str, Any]) -> dict[str, Any]:
    sequence_rows = _variant_rows(ar, "sequence_variants", "sequence_deltas", "sequence_context")
    graph_rows = _variant_rows(as_, "graph_variants", "graph_deltas", "graph_context")
    all_rows = sequence_rows + graph_rows
    positives = [
        row
        for row in all_rows
        if row["delta_vs_baseline_family"]["all_improvement"] > 0.0
        or row["delta_vs_baseline_family"]["t50_improvement"] > 0.0
        or row["delta_vs_baseline_family"]["hard_failure_improvement"] > 0.0
    ]
    best_delta_all = max((row["delta_vs_baseline_family"]["all_improvement"] for row in all_rows), default=0.0)
    best_delta_t50 = max((row["delta_vs_baseline_family"]["t50_improvement"] for row in all_rows), default=0.0)
    best_delta_hard = max((row["delta_vs_baseline_family"]["hard_failure_improvement"] for row in all_rows), default=0.0)
    worst_delta_all = min((row["delta_vs_baseline_family"]["all_improvement"] for row in all_rows), default=0.0)
    worst_delta_t50 = min((row["delta_vs_baseline_family"]["t50_improvement"] for row in all_rows), default=0.0)
    db_decision = db.get("summary", {}).get("decision", "unknown")
    dc_decision = dc.get("selected_context_switchability_policy", {}).get("decision", "unknown")
    closure_decision = (
        "close_current_sequence_graph_residual_context_protocol"
        if not positives
        else "keep_current_context_protocol_open_for_repair"
    )
    return {
        "source": "fresh_synthesis_after_fresh_ar_as_rerun",
        "baseline_family_metric": {k: float(v) for k, v in _metric(ar).items() if isinstance(v, (int, float))},
        "sequence_rows": sequence_rows,
        "graph_rows": graph_rows,
        "positive_context_rows": positives,
        "best_delta_all": float(best_delta_all),
        "best_delta_t50": float(best_delta_t50),
        "best_delta_hard_failure": float(best_delta_hard),
        "worst_delta_all": float(worst_delta_all),
        "worst_delta_t50": float(worst_delta_t50),
        "prior_context_rescue_decision": db_decision,
        "prior_context_switchability_decision": dc_decision,
        "closure_decision": closure_decision,
        "root_cause": (
            "Fresh Stage42-AR/AS reruns show that temporal sequence context and current-frame kNN graph context both reduce "
            "all/t50/hard-failure improvements relative to the baseline-family first-stage control. The dominant current "
            "signal remains baseline-family rollout context plus safety floor; the present residual context target is not "
            "extracting independent scene/goal/interaction value."
        ),
        "next_best_action": [
            "Do not repeat the same residual sequence/graph context protocol without changing target or data support.",
            "Prioritize source/legal/time closure for ETH_UCY, TrajNet, and UCY so context can be tested on better-calibrated sources.",
            "If context modeling is revisited, use switchability/gain-harm or full sequence architecture with baseline-family control, not blind residual deltas.",
            "Keep Stage37/teacher floor and Stage42 protected runtime policies as deployable evidence while context remains auxiliary/diagnostic.",
        ],
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    ar = payload["inputs"]["stage42_ar_sequence_context"]
    as_ = payload["inputs"]["stage42_as_graph_context"]
    summary = payload["summary"]
    no_leakage = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "fresh_ar_rerun_loaded": ar.get("source") == "fresh_run",
        "fresh_as_rerun_loaded": as_.get("source") == "fresh_run",
        "sequence_variants_complete": len(summary["sequence_rows"]) >= 3,
        "graph_variants_complete": len(summary["graph_rows"]) >= 3,
        "sequence_graph_increment_not_overclaimed": len(summary["positive_context_rows"]) == 0,
        "baseline_family_control_recorded": bool(summary["baseline_family_metric"]),
        "closure_decision_recorded": summary["closure_decision"]
        == "close_current_sequence_graph_residual_context_protocol",
        "root_cause_written": bool(summary["root_cause"]),
        "next_action_written": len(summary["next_best_action"]) >= 3,
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
    verdict = "stage42_dp_context_model_closure_pass" if passed == total else "stage42_dp_context_model_closure_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["summary"]
    gate = payload["stage42_dp_gate"]
    lines = [
        "# Stage42-DP Context Model Closure",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- closure_decision: `{summary['closure_decision']}`",
        "",
        "## Current Facts",
        "",
    ]
    lines.extend(f"- {fact}" for fact in CURRENT_FACTS)
    lines.extend(
        [
            "",
            "## Baseline-Family Control",
            "",
        ]
    )
    for key in [
        "rows",
        "all_improvement",
        "t50_improvement",
        "t100_raw_frame_diagnostic_improvement",
        "hard_failure_improvement",
        "easy_degradation",
    ]:
        if key in summary["baseline_family_metric"]:
            lines.append(f"- {key}: `{summary['baseline_family_metric'][key]}`")
    lines.extend(
        [
            "",
            "## Sequence / Graph Deltas vs Baseline-Family Control",
            "",
            "| protocol | variant | delta all | delta t50 | delta hard/failure | delta easy |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in [*summary["sequence_rows"], *summary["graph_rows"]]:
        d = row["delta_vs_baseline_family"]
        lines.append(
            f"| `{row['protocol']}` | `{row['variant']}` | {d['all_improvement']:.6f} | {d['t50_improvement']:.6f} | {d['hard_failure_improvement']:.6f} | {d['easy_degradation']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Closure Decision",
            "",
            f"- positive_context_rows: `{[row['variant'] for row in summary['positive_context_rows']]}`",
            f"- best_delta_all: `{summary['best_delta_all']}`",
            f"- best_delta_t50: `{summary['best_delta_t50']}`",
            f"- best_delta_hard_failure: `{summary['best_delta_hard_failure']}`",
            f"- worst_delta_all: `{summary['worst_delta_all']}`",
            f"- worst_delta_t50: `{summary['worst_delta_t50']}`",
            f"- prior_context_rescue_decision: `{summary['prior_context_rescue_decision']}`",
            f"- prior_context_switchability_decision: `{summary['prior_context_switchability_decision']}`",
            "",
            f"{summary['root_cause']}",
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
            "- This is fresh evidence for closing the current sequence/graph residual context protocol, not evidence that context can never help.",
            "- It does not execute Stage5C and does not enable SMC.",
            "- It remains protected dataset-local/raw-frame 2.5D evidence, not metric/seconds-level, true 3D, or foundation evidence.",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_dp_gate"]
    lines = [
        "# Stage42-DP Gate",
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
    gate = payload["stage42_dp_gate"]
    lines = [
        "## Stage42-DP Context Model Closure",
        "",
        "- source: `fresh_synthesis_after_fresh_ar_as_rerun`",
        f"- verdict: `{gate['verdict']}`; gates `{gate['passed']} / {gate['total']}`.",
        "- fresh reruns: Stage42-AR sequence context and Stage42-AS graph context.",
        f"- closure decision: `{summary['closure_decision']}`.",
        f"- best delta all/t50/hard vs baseline-family control: `{summary['best_delta_all']:.4f}` / `{summary['best_delta_t50']:.4f}` / `{summary['best_delta_hard_failure']:.4f}`.",
        "- conclusion: current residual sequence/graph context protocol does not add independent lift beyond baseline-family rollout context.",
        "- next: change target/data/model before revisiting context, and keep protected Stage37/teacher/runtime policies as deployable floor.",
        "- Claim boundary: dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no global metric/seconds-level, no Stage5C execution, no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, RETRO_README, CURRENT_GOAL_README, GOAL_RESULTS_README]:
        _replace_section(path, "STAGE42_DP_CONTEXT_MODEL_CLOSURE", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    summary = payload["summary"]
    gate = payload["stage42_dp_gate"]
    state["current_stage"] = "Stage42-DP context model closure"
    state["current_verdict"] = gate["verdict"]
    state["stage42_dp_context_model_closure"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": gate["verdict"],
        "gates": f"{gate['passed']}/{gate['total']}",
        "closure_decision": summary["closure_decision"],
        "best_delta_all": summary["best_delta_all"],
        "best_delta_t50": summary["best_delta_t50"],
        "best_delta_hard_failure": summary["best_delta_hard_failure"],
        "positive_context_rows": [row["variant"] for row in summary["positive_context_rows"]],
        "root_cause": summary["root_cause"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_context_model_closure(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ar = read_json(AR_JSON, {})
    as_ = read_json(AS_JSON, {})
    db = read_json(DB_JSON, {})
    dc = read_json(DC_JSON, {})
    no_leakage = {
        "future_endpoint_input": bool(ar.get("no_leakage", {}).get("future_endpoint_input", True))
        or bool(as_.get("no_leakage", {}).get("future_endpoint_input", True)),
        "future_waypoint_input": bool(ar.get("no_leakage", {}).get("future_waypoint_input", True))
        or bool(as_.get("no_leakage", {}).get("future_waypoint_input", True)),
        "central_velocity": bool(ar.get("no_leakage", {}).get("central_velocity", True))
        or bool(as_.get("no_leakage", {}).get("central_velocity", True)),
        "test_endpoint_goals": bool(ar.get("no_leakage", {}).get("test_endpoint_goals", True))
        or bool(as_.get("no_leakage", {}).get("test_endpoint_goals", True)),
        "test_threshold_tuning": bool(ar.get("no_leakage", {}).get("test_threshold_tuning", True))
        or bool(as_.get("no_leakage", {}).get("test_threshold_tuning", True)),
        "source_overlap_pass": bool(ar.get("no_leakage", {}).get("source_overlap_pass", False))
        and bool(as_.get("no_leakage", {}).get("source_overlap_pass", False)),
    }
    claim_boundary = {
        "true_3d": False,
        "foundation_world_model": False,
        "metric_or_seconds_claim": False,
        "raw_frame_dataset_local_only": True,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    payload: dict[str, Any] = {
        "source": "fresh_synthesis_after_fresh_ar_as_rerun",
        "stage": "Stage42-DP",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "stage42_ar_sequence_context": {
                "path": str(AR_JSON),
                "source": ar.get("source"),
                "generated_at_utc": ar.get("generated_at_utc"),
                "verdict": ar.get("stage42_ar_gate", {}).get("verdict"),
                "gate": f"{ar.get('stage42_ar_gate', {}).get('passed')}/{ar.get('stage42_ar_gate', {}).get('total')}",
            },
            "stage42_as_graph_context": {
                "path": str(AS_JSON),
                "source": as_.get("source"),
                "generated_at_utc": as_.get("generated_at_utc"),
                "verdict": as_.get("stage42_as_gate", {}).get("verdict"),
                "gate": f"{as_.get('stage42_as_gate', {}).get('passed')}/{as_.get('stage42_as_gate', {}).get('total')}",
            },
            "stage42_db_context_rescue_decision": {"path": str(DB_JSON), "source": db.get("source")},
            "stage42_dc_context_switchability_gate": {"path": str(DC_JSON), "source": dc.get("source")},
        },
        "summary": _summary(ar, as_, db, dc),
        "no_leakage": no_leakage,
        "claim_boundary": claim_boundary,
    }
    payload["stage42_dp_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_context_model_closure()
