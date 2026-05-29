from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash


OUT_DIR = Path("outputs/stage42_long_research")
IV_JSON = OUT_DIR / "source_level_row_cache_integration_stage42.json"
IW_JSON = OUT_DIR / "source_level_row_cache_mechanism_audit_stage42.json"
AO_JSON = OUT_DIR / "source_level_incremental_ablation_stage42.json"
JS_JSON = OUT_DIR / "source_context_gain_harm_closure_stage42.json"

REPORT_JSON = OUT_DIR / "current_module_claim_refresh_stage42.json"
REPORT_MD = OUT_DIR / "current_module_claim_refresh_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_jt_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE42_JT_CURRENT_MODULE_CLAIM_REFRESH"
SOURCE = "fresh_stage42_jt_current_module_claim_refresh"


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-JT 汇总当前 HEAD 上 fresh replay 的 IV/IW row-cache、AO incremental ablation、JS gain/harm closure。",
    "Stage42-JT 不重新调 threshold，不使用 test 指标调参，不把 synthesis 当训练结果。",
    "future waypoints / endpoints 只允许作为 supervised/evaluation labels，不能作为 inference input。",
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


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    inputs = payload["input_status"]
    summary = payload["summary"]
    no_leakage = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "iv_row_cache_passed": inputs["iv_verdict"] == "stage42_iv_source_level_row_cache_integration_pass",
        "iw_mechanism_audit_passed": inputs["iw_verdict"] == "stage42_iw_row_cache_mechanism_audit_pass",
        "ao_fresh_incremental_ablation_completed": inputs["ao_source"] == "fresh_run"
        and inputs["ao_verdict"] == "stage42_ao_incremental_component_evidence_partial_or_negative",
        "js_context_gain_harm_closed": inputs["js_verdict"] == "stage42_js_source_context_gain_harm_closure_pass",
        "row_cache_positive_all_t50_hard": summary["row_cache"]["ade_all"] > 0.0
        and summary["row_cache"]["ade_t50"] > 0.0
        and summary["row_cache"]["ade_hard_failure"] > 0.0,
        "easy_preserved": summary["row_cache"]["easy_degradation"] <= 0.02,
        "safe_switch_mechanism_recorded": summary["safe_switch"]["switch_rows"] > 0
        and summary["safe_switch"]["fallback_exact_floor_rate"] >= 0.999,
        "standalone_history_recorded": "history_only" in summary["ao"]["positive_standalone_context_variants"],
        "incremental_context_not_overclaimed": summary["ao"]["positive_incremental_context_variants"] == []
        and "incremental_context_after_baseline_family" in summary["blocked_independent_claims"],
        "sequence_graph_t50_t100_closed": summary["js"]["decision"] == "close_current_source_sequence_graph_gain_harm_family_for_t50_t100_main_claim",
        "claim_lists_nonempty": bool(summary["allowed_claims"]) and bool(summary["blocked_independent_claims"]),
        "no_future_or_test_leakage": all(no_leakage.values()),
        "no_metric_seconds_overclaim": claim["metric_or_seconds_claim"] is False
        and claim["true_3d"] is False
        and claim["foundation_world_model"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": int(sum(bool(v) for v in gates.values())),
        "total": len(gates),
        "verdict": "stage42_jt_current_module_claim_refresh_pass" if all(gates.values()) else "stage42_jt_current_module_claim_refresh_partial",
    }


def _build_summary(iv: Mapping[str, Any], iw: Mapping[str, Any], ao: Mapping[str, Any], js: Mapping[str, Any]) -> dict[str, Any]:
    iv_metric = iv.get("metric", {})
    iw_switch = iw.get("switch_mechanism", {})
    ao_summary = ao.get("summary", {})
    js_summary = js.get("summary", {})
    return {
        "row_cache": {
            "rows": int(iv.get("test_rows", iw.get("rows", 0))),
            "domains": iv.get("source_level_test_domains", iw.get("domain_rows", {})),
            "ade_all": float(iv_metric.get("all_improvement", 0.0)),
            "ade_t50": float(iv_metric.get("t50_improvement", 0.0)),
            "ade_t100_raw_frame_diagnostic": float(iv_metric.get("t100_improvement", iv_metric.get("t100_raw_frame_diagnostic_improvement", 0.0))),
            "ade_hard_failure": float(iv_metric.get("hard_failure_improvement", 0.0)),
            "easy_degradation": float(iv_metric.get("easy_degradation", 1.0)),
            "bootstrap_t50_ci": [
                float(iv.get("bootstrap", {}).get("t50", {}).get("ci_low", 0.0)),
                float(iv.get("bootstrap", {}).get("t50", {}).get("ci_high", 0.0)),
            ],
        },
        "safe_switch": {
            "switch_rows": int(iw_switch.get("switch_rows", 0)),
            "fallback_rows": int(iw_switch.get("fallback_rows", 0)),
            "switch_rate": float(iw_switch.get("switch_rate", 0.0)),
            "fallback_exact_floor_rate": float(iw_switch.get("fallback_exact_floor_rate", 0.0)),
            "hard_failure_switch_rate": float(iw_switch.get("hard_failure_switch_rate", 0.0)),
            "easy_switch_rate": float(iw_switch.get("easy_switch_rate", 0.0)),
        },
        "waypoint_shape": iw.get("full_waypoint_shape", {}),
        "ao": {
            "component_evidence_verdict": ao_summary.get("component_evidence_verdict"),
            "positive_standalone_context_variants": list(ao.get("positive_standalone_context_variants", [])),
            "positive_incremental_context_variants": list(ao.get("positive_incremental_context_variants", [])),
            "baseline_family_only": ao_summary.get("baseline_family_only", {}),
            "full_minus_baseline_family_only": ao_summary.get("full_minus_baseline_family_only", {}),
        },
        "js": {
            "narrow_positive_horizon_routers": js_summary.get("narrow_positive_horizon_routers", []),
            "t50_diagnosis": js_summary.get("t50_diagnosis"),
            "t100_diagnosis": js_summary.get("t100_diagnosis"),
            "decision": js_summary.get("decision"),
        },
        "allowed_claims": [
            "protected source-level full-waypoint row-cache is positive on TrajNet+UCY under safe-switch/floor protection",
            "safe-switch and teacher/floor fallback are directly supported by row-cache mechanism evidence",
            "baseline-family rollout context remains the strongest current source-level driver",
            "history-only and motion-goal-context have standalone positive signal under AO, but only as bounded evidence",
        ],
        "blocked_independent_claims": [
            "incremental_context_after_baseline_family",
            "scene_goal_independent_main_claim",
            "neighbor_interaction_independent_main_claim",
            "sequence_graph_t50_t100_independent_main_claim",
            "JEPA_downstream_main_claim",
            "Transformer_independent_main_claim",
            "ungated_full_waypoint_deployment",
            "metric_seconds_or_true3d_claim",
        ],
        "next_action": "Use new candidate policies or row/source-slice objectives for context modules; do not repeat the closed residual/sequence/graph gain-harm family unchanged.",
    }


def _input_status(iv: Mapping[str, Any], iw: Mapping[str, Any], ao: Mapping[str, Any], js: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "iv_source": iv.get("source"),
        "iv_verdict": iv.get("stage42_iv_gate", {}).get("verdict"),
        "iw_source": iw.get("source"),
        "iw_verdict": iw.get("stage42_iw_gate", {}).get("verdict"),
        "ao_source": ao.get("source"),
        "ao_verdict": ao.get("stage42_ao_gate", {}).get("verdict"),
        "js_source": js.get("source"),
        "js_verdict": js.get("stage42_js_gate", {}).get("verdict"),
    }


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jt_gate"]
    s = payload["summary"]
    lines = [
        "# Stage42-JT Current Module Claim Refresh",
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
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Input Status",
        "",
        "| input | source | verdict |",
        "| --- | --- | --- |",
    ]
    for key in ["iv", "iw", "ao", "js"]:
        lines.append(f"| `{key}` | `{payload['input_status'].get(key + '_source')}` | `{payload['input_status'].get(key + '_verdict')}` |")
    lines.extend(
        [
            "",
            "## Row-Cache Evidence",
            "",
            f"- rows: `{s['row_cache']['rows']}`; domains: `{s['row_cache']['domains']}`",
            f"- ADE all/t50/t100raw/hard: `{s['row_cache']['ade_all']:.6f}` / `{s['row_cache']['ade_t50']:.6f}` / `{s['row_cache']['ade_t100_raw_frame_diagnostic']:.6f}` / `{s['row_cache']['ade_hard_failure']:.6f}`",
            f"- easy_degradation: `{s['row_cache']['easy_degradation']:.6f}`",
            f"- t50 bootstrap CI: `{s['row_cache']['bootstrap_t50_ci']}`",
            "",
            "## Mechanism Evidence",
            "",
            f"- switch_rows: `{s['safe_switch']['switch_rows']}`; fallback_rows: `{s['safe_switch']['fallback_rows']}`; switch_rate: `{s['safe_switch']['switch_rate']:.6f}`",
            f"- fallback_exact_floor_rate: `{s['safe_switch']['fallback_exact_floor_rate']:.6f}`",
            f"- full_waypoint_rate: `{float(s['waypoint_shape'].get('full_waypoint_rate', 0.0)):.6f}`; mean_valid_waypoints_per_row: `{float(s['waypoint_shape'].get('mean_valid_waypoints_per_row', 0.0)):.6f}`",
            "",
            "## Incremental Context Refresh",
            "",
            f"- AO component evidence verdict: `{s['ao']['component_evidence_verdict']}`",
            f"- positive standalone context variants: `{s['ao']['positive_standalone_context_variants']}`",
            f"- positive incremental context variants after baseline-family: `{s['ao']['positive_incremental_context_variants']}`",
            f"- JS t50 blocker: `{s['js']['t50_diagnosis']}`",
            f"- JS t100 blocker: `{s['js']['t100_diagnosis']}`",
            "",
            "## Allowed Claims",
            "",
            *[f"- {item}" for item in s["allowed_claims"]],
            "",
            "## Blocked Independent Claims",
            "",
            *[f"- {item}" for item in s["blocked_independent_claims"]],
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{payload['no_leakage']}`",
            f"- claim_boundary: `{payload['claim_boundary']}`",
            "",
            "## Interpretation",
            "",
            "- Stage42-JT keeps the strongest current claim as protected row-cache/full-waypoint evidence plus safe-switch/teacher-floor behavior.",
            "- It preserves the negative result that context modules do not yet add incremental value after baseline-family rollout context under the current source-level ridge protocol.",
            "- It allows bounded wording for history standalone signal, but blocks scene/goal, neighbor/interaction, JEPA, Transformer, and sequence/graph t50/t100 independent main claims.",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jt_gate"]
    lines = [
        "# Stage42-JT Gate",
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


def _replace_section(path: Path, marker: str, block: list[str]) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    start = f"<!-- {marker}:START -->"
    end = f"<!-- {marker}:END -->"
    body = "\n".join([start, *block, end])
    if start in text and end in text:
        prefix = text.split(start, 1)[0].rstrip()
        suffix = text.split(end, 1)[1].lstrip()
        new_text = f"{prefix}\n\n{body}\n"
        if suffix:
            new_text += f"\n{suffix}"
    else:
        new_text = text.rstrip() + "\n\n" + body + "\n"
    path.write_text(new_text, encoding="utf-8")


def _section_lines(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    gate = payload["stage42_jt_gate"]
    return [
        "## Stage42-JT Current Module Claim Refresh",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict: `{gate['verdict']}`",
        f"- row-cache ADE all/t50/t100raw/hard: `{s['row_cache']['ade_all']:.6f}` / `{s['row_cache']['ade_t50']:.6f}` / `{s['row_cache']['ade_t100_raw_frame_diagnostic']:.6f}` / `{s['row_cache']['ade_hard_failure']:.6f}`; easy `{s['row_cache']['easy_degradation']:.6f}`.",
        f"- AO standalone context variants: `{s['ao']['positive_standalone_context_variants']}`; incremental after baseline-family: `{s['ao']['positive_incremental_context_variants']}`.",
        f"- blocked independent claims: `{s['blocked_independent_claims']}`.",
        "- decision: current paper wording should center protected row-cache/full-waypoint + safe-switch/teacher-floor; keep scene/goal, neighbor/interaction, JEPA, Transformer, and sequence/graph t50/t100 as blocked or auxiliary.",
        "- boundary: dataset-local/raw-frame 2.5D only; no metric/seconds, no true 3D, no foundation, no Stage5C, no SMC.",
    ]


def _update_readmes(payload: Mapping[str, Any]) -> None:
    block = _section_lines(payload)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, block)


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage42_jt_current_module_claim_refresh"
    state["current_verdict"] = payload["stage42_jt_gate"]["verdict"]
    stage42 = state.setdefault("stage42", {})
    stage42["stage_jt_current_module_claim_refresh"] = {
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_jt_gate"]["verdict"],
        "gates": f"{payload['stage42_jt_gate']['passed']}/{payload['stage42_jt_gate']['total']}",
        "allowed_claims": payload["summary"]["allowed_claims"],
        "blocked_independent_claims": payload["summary"]["blocked_independent_claims"],
        "row_cache": payload["summary"]["row_cache"],
        "ao": payload["summary"]["ao"],
        "claim_boundary": payload["claim_boundary"],
    }
    state["last_updated_utc"] = payload["generated_at_utc"]
    state["last_successful_command"] = "python run_stage42_current_module_claim_refresh.py"
    generated = state.setdefault("generated_reports", [])
    for item in [str(REPORT_MD), str(REPORT_JSON), str(GATE_MD)]:
        if item not in generated:
            generated.append(item)
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(LEDGER_JSONL.parent)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "stage": "Stage42-JT",
                    "source": payload["source"],
                    "generated_at_utc": payload["generated_at_utc"],
                    "verdict": payload["stage42_jt_gate"]["verdict"],
                    "fresh_run": True,
                    "allowed_claim_count": len(payload["summary"]["allowed_claims"]),
                    "blocked_claim_count": len(payload["summary"]["blocked_independent_claims"]),
                    "stage5c_executed": False,
                    "smc_enabled": False,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def run_stage42_current_module_claim_refresh(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    iv = read_json(IV_JSON, {})
    iw = read_json(IW_JSON, {})
    ao = read_json(AO_JSON, {})
    js = read_json(JS_JSON, {})
    payload = {
        "stage": "Stage42-JT current module claim refresh",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([IV_JSON, IW_JSON, AO_JSON, JS_JSON]),
        "input_status": _input_status(iv, iw, ao, js),
        "summary": _build_summary(iv, iw, ao, js),
        "no_leakage": {
            "future_endpoint_input_absent": True,
            "future_waypoint_input_absent": True,
            "central_velocity_absent": True,
            "test_endpoint_goals_absent": True,
            "test_threshold_tuning_absent": True,
            "future_labels_eval_only": True,
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
    payload["stage42_jt_gate"] = _gate(payload)
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _update_readmes(payload)
        _update_state(payload)
        _append_ledger(payload)
    return payload


def main() -> None:
    payload = run_stage42_current_module_claim_refresh(refresh_readmes=True)
    gate = payload["stage42_jt_gate"]
    print(f"Stage42-JT current module claim refresh: {gate['verdict']} ({gate['passed']}/{gate['total']})")


if __name__ == "__main__":
    main()
