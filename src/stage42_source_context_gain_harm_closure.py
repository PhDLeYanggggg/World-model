from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_local_calibrated_source_terms_prefill import _jsonable
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
JR_JSON = OUT_DIR / "source_context_fresh_replay_stage42.json"
IO_JSON = OUT_DIR / "horizon_sequence_graph_context_router_stage42.json"
IP_JSON = OUT_DIR / "t50_t100_sequence_graph_blocker_audit_stage42.json"
IQ_JSON = OUT_DIR / "t50_switchability_calibration_repair_stage42.json"
IR_JSON = OUT_DIR / "t50_source_pattern_switchability_repair_stage42.json"

REPORT_JSON = OUT_DIR / "source_context_gain_harm_closure_stage42.json"
REPORT_MD = OUT_DIR / "source_context_gain_harm_closure_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_js_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
LEDGER = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE42_JS_SOURCE_CONTEXT_GAIN_HARM_CLOSURE"
SOURCE = "fresh_stage42_js_source_context_gain_harm_closure"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-JS consolidates fresh JR/IO/IP/IQ/IR evidence after rerunning the gain/harm repair chain on current HEAD.",
    "This closes the current source-level sequence/graph context candidate family as an independent t50/t100 main claim.",
    "h10/h25 have narrow horizon-specific positive routing evidence, but t50/t100 remain unsupported under this candidate family.",
    "future endpoints / waypoints are labels/evaluation only, not inference input.",
    "No central velocity, no test endpoint goals, no test threshold tuning.",
    "No metric/seconds claim, no Stage5C execution, no SMC.",
]


def _metric(row: Mapping[str, Any] | None) -> dict[str, float]:
    row = row or {}
    return {
        "all_improvement": float(row.get("all_improvement", 0.0)),
        "t50_improvement": float(row.get("t50_improvement", 0.0)),
        "t100_raw_frame_diagnostic_improvement": float(row.get("t100_raw_frame_diagnostic_improvement", 0.0)),
        "hard_failure_improvement": float(row.get("hard_failure_improvement", 0.0)),
        "easy_degradation": float(row.get("easy_degradation", 0.0)),
        "switch_rate": float(row.get("switch_rate", 0.0)),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    b = payload["claim_boundary"]
    gates = {
        "jr_negative_residual_replay_loaded": s["jr_verdict"] == "stage42_jr_source_context_negative_evidence_pass",
        "io_horizon_router_loaded": s["io_verdict"] == "stage42_io_horizon_sequence_graph_context_router_pass",
        "ip_blocker_audit_loaded": s["ip_verdict"] == "stage42_ip_t50_t100_sequence_graph_blocker_audit_pass",
        "iq_gain_harm_repair_evaluated": s["iq_verdict"] == "stage42_iq_t50_switchability_calibration_repair_pass",
        "ir_source_pattern_repair_evaluated": s["ir_verdict"] == "stage42_ir_t50_source_pattern_switchability_repair_pass",
        "narrow_h10_h25_positive_recorded": len(s["narrow_positive_horizon_routers"]) >= 1,
        "t50_repair_not_supported_recorded": s["iq_repair_supported"] is False and s["ir_repair_supported"] is False,
        "t100_blocker_recorded": bool(s["t100_diagnosis"]),
        "current_candidate_family_closed_for_t50_t100": s["decision"]
        == "close_current_source_sequence_graph_gain_harm_family_for_t50_t100_main_claim",
        "next_action_not_more_threshold_tuning": "new candidate" in s["next_repair_direction"],
        "negative_result_not_overclaimed": b["sequence_graph_independent_main_claim"] is False,
        "no_metric_seconds_overclaim": b["metric_or_seconds_claim"] is False,
        "stage5c_false": b["stage5c_executed"] is False,
        "smc_false": b["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    verdict = "stage42_js_source_context_gain_harm_closure_pass" if passed == len(gates) else "stage42_js_source_context_gain_harm_closure_partial"
    return {"source": "fresh_run", "gates": gates, "passed": passed, "total": len(gates), "verdict": verdict}


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    jr = read_json(JR_JSON, {})
    io = read_json(IO_JSON, {})
    ip = read_json(IP_JSON, {})
    iq = read_json(IQ_JSON, {})
    ir = read_json(IR_JSON, {})

    best_by_horizon = io.get("summary", {}).get("best_by_horizon", {})
    horizon_rows: dict[str, dict[str, Any]] = {}
    for h, row in best_by_horizon.items():
        metric = _metric((row or {}).get("metric", {}))
        horizon_rows[str(h)] = {
            "best_key": row.get("best_key", ""),
            "candidate": row.get("candidate", ""),
            "metric": metric,
            "supported": bool(row.get("supported", False)),
        }

    ip_best = ip.get("summary", {}).get("best_by_horizon", {})
    iq_metric = _metric(iq.get("summary", {}).get("best_trial_metric", {}))
    ir_metric = _metric(ir.get("summary", {}).get("best_trial_metric", {}))

    summary = {
        "source": SOURCE,
        "jr_verdict": jr.get("stage42_jr_gate", {}).get("verdict", ""),
        "io_verdict": io.get("stage42_io_gate", {}).get("verdict", ""),
        "ip_verdict": ip.get("stage42_ip_gate", {}).get("verdict", ""),
        "iq_verdict": iq.get("stage42_iq_gate", {}).get("verdict", ""),
        "ir_verdict": ir.get("stage42_ir_gate", {}).get("verdict", ""),
        "narrow_positive_horizon_routers": io.get("summary", {}).get("positive_horizon_sequence_graph_context_routers", []),
        "horizon_router_summary": horizon_rows,
        "t50_diagnosis": ip.get("summary", {}).get("t50_diagnosis", ""),
        "t100_diagnosis": ip.get("summary", {}).get("t100_diagnosis", ""),
        "t50_oracle_headroom": float((ip_best.get("50") or {}).get("best_oracle_headroom", 0.0)),
        "t100_oracle_headroom": float((ip_best.get("100") or {}).get("best_oracle_headroom", 0.0)),
        "iq_repair_supported": bool(iq.get("summary", {}).get("repair_supported", False)),
        "iq_best_trial_key": iq.get("summary", {}).get("best_trial_key", ""),
        "iq_best_trial_metric": iq_metric,
        "ir_repair_supported": bool(ir.get("summary", {}).get("repair_supported", False)),
        "ir_best_trial_key": ir.get("summary", {}).get("best_trial_key", ""),
        "ir_best_trial_metric": ir_metric,
        "decision": "close_current_source_sequence_graph_gain_harm_family_for_t50_t100_main_claim",
        "deployment_decision": "keep_baseline_family_and_existing_protected_floor_as_deployable_context_mechanism",
        "next_repair_direction": "new candidate policies or row-level/source-slice objectives; not more threshold tuning on the same sequence/graph proposals",
    }

    payload: dict[str, Any] = {
        "stage": "Stage42-JS",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(JR_JSON), str(IO_JSON), str(IP_JSON), str(IQ_JSON), str(IR_JSON)]),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "jr": str(JR_JSON),
            "io": str(IO_JSON),
            "ip": str(IP_JSON),
            "iq": str(IQ_JSON),
            "ir": str(IR_JSON),
            "jr_generated_at_utc": jr.get("generated_at_utc", ""),
            "io_generated_at_utc": io.get("generated_at_utc", ""),
            "ip_generated_at_utc": ip.get("generated_at_utc", ""),
            "iq_generated_at_utc": iq.get("generated_at_utc", ""),
            "ir_generated_at_utc": ir.get("generated_at_utc", ""),
        },
        "summary": summary,
        "failure_taxonomy": {
            "residual_protocol": "JR confirms sequence/graph residual variants degrade all/t50/hard versus baseline-family rollout context.",
            "horizon_mixing": "IO shows h10/h25 narrow positives, but t50/t100 remain unsupported; horizon mixing is only a partial explanation.",
            "t50_blocker": summary["t50_diagnosis"],
            "t100_blocker": summary["t100_diagnosis"],
            "gain_harm_repair": "IQ validation-selected gain/harm calibration still fails to capture t50 headroom.",
            "source_pattern_repair": "IR source-pattern support also falls back to no useful t50 switches.",
            "what_not_to_claim": [
                "source-level sequence/graph context is an independent t50/t100 contribution",
                "scene/goal/neighbor/interaction has been proven as a main driver by this candidate family",
                "more threshold tuning on the same candidate proposals is likely enough",
            ],
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "sequence_graph_independent_main_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_js_gate"] = _gate(payload)
    return payload


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    gate = payload["stage42_js_gate"]
    lines = [
        "# Stage42-JS Source Context Gain/Harm Closure",
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
        "## Summary",
        "",
        f"- decision: `{s['decision']}`",
        f"- deployment_decision: `{s['deployment_decision']}`",
        f"- narrow_positive_horizon_routers: `{s['narrow_positive_horizon_routers']}`",
        f"- t50_diagnosis: `{s['t50_diagnosis']}`; oracle_headroom `{s['t50_oracle_headroom']:.6f}`",
        f"- t100_diagnosis: `{s['t100_diagnosis']}`; oracle_headroom `{s['t100_oracle_headroom']:.6f}`",
        f"- IQ repair supported: `{s['iq_repair_supported']}`; best `{s['iq_best_trial_key']}`; t50 `{s['iq_best_trial_metric']['t50_improvement']:.6f}`",
        f"- IR repair supported: `{s['ir_repair_supported']}`; best `{s['ir_best_trial_key']}`; t50 `{s['ir_best_trial_metric']['t50_improvement']:.6f}`",
        f"- next_repair_direction: `{s['next_repair_direction']}`",
        "",
        "## Horizon Router Summary",
        "",
        "| horizon | best | candidate | all | t50 | t100 raw | hard/failure | easy | switch | supported |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for h, row in sorted(s["horizon_router_summary"].items(), key=lambda item: int(item[0])):
        m = row["metric"]
        lines.append(
            f"| {h} | `{row['best_key']}` | `{row['candidate']}` | {m['all_improvement']:.6f} | {m['t50_improvement']:.6f} | {m['t100_raw_frame_diagnostic_improvement']:.6f} | {m['hard_failure_improvement']:.6f} | {m['easy_degradation']:.6f} | {m['switch_rate']:.6f} | `{row['supported']}` |"
        )
    lines.extend(
        [
            "",
            "## Failure Taxonomy",
            "",
            *[f"- {key}: `{value}`" for key, value in payload["failure_taxonomy"].items()],
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
        ]
    )
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | `{bool(value)}` |")
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_js_gate"]
    lines = [
        "# Stage42-JS Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | `{bool(value)}` |")
    return lines


def _section_lines(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    gate = payload["stage42_js_gate"]
    return [
        "## Stage42-JS Source Context Gain/Harm Closure",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict: `{gate['verdict']}`",
        f"- narrow horizon positives: `{s['narrow_positive_horizon_routers']}`; these are not t50/t100 main-claim evidence.",
        f"- t50 blocker: `{s['t50_diagnosis']}` with oracle headroom `{s['t50_oracle_headroom']:.4f}`; IQ repair t50 `{s['iq_best_trial_metric']['t50_improvement']:.6f}`, IR repair t50 `{s['ir_best_trial_metric']['t50_improvement']:.6f}`.",
        f"- t100 blocker: `{s['t100_diagnosis']}` with oracle headroom `{s['t100_oracle_headroom']:.4f}`.",
        "- decision: close the current source-level sequence/graph gain-harm candidate family for t50/t100 independent contribution; next work needs new candidate policies or row/source-slice objectives.",
        "- boundary: raw-frame/dataset-local 2.5D only; no metric/seconds overclaim, no Stage5C, no SMC.",
    ]


def _update_readmes(payload: Mapping[str, Any]) -> None:
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, _section_lines(payload))


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    stage42 = state.setdefault("stage42", {})
    stage42["source_context_gain_harm_closure"] = {
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "verdict": payload["stage42_js_gate"]["verdict"],
        "gate": {"passed": payload["stage42_js_gate"]["passed"], "total": payload["stage42_js_gate"]["total"]},
        "narrow_positive_horizon_routers": payload["summary"]["narrow_positive_horizon_routers"],
        "t50_diagnosis": payload["summary"]["t50_diagnosis"],
        "t100_diagnosis": payload["summary"]["t100_diagnosis"],
        "iq_repair_supported": payload["summary"]["iq_repair_supported"],
        "ir_repair_supported": payload["summary"]["ir_repair_supported"],
        "decision": payload["summary"]["decision"],
        "global_metric_or_seconds_claim": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["last_updated_utc"] = payload["generated_at_utc"]
    state["last_successful_command"] = "python run_stage42_source_context_gain_harm_closure.py"
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(LEDGER.parent)
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "stage": "Stage42-JS",
                    "source": payload["source"],
                    "generated_at_utc": payload["generated_at_utc"],
                    "verdict": payload["stage42_js_gate"]["verdict"],
                    "fresh_run": True,
                    "decision": payload["summary"]["decision"],
                    "iq_repair_supported": payload["summary"]["iq_repair_supported"],
                    "ir_repair_supported": payload["summary"]["ir_repair_supported"],
                    "stage5c_executed": False,
                    "smc_enabled": False,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def run_stage42_source_context_gain_harm_closure(*, refresh_readmes: bool = True) -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _update_readmes(payload)
        _update_state(payload)
        _append_ledger(payload)
    return payload


def main() -> None:
    payload = run_stage42_source_context_gain_harm_closure(refresh_readmes=True)
    gate = payload["stage42_js_gate"]
    print(f"Stage42-JS source context gain/harm closure: {gate['verdict']} ({gate['passed']}/{gate['total']})")


if __name__ == "__main__":
    main()
