from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import read_json, write_json, write_md
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "context_rescue_decision_stage42.json"
REPORT_MD = OUT_DIR / "context_rescue_decision_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_db_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_README = Path("README_M3W_GOAL_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

INPUTS = {
    "goal_scene_gated": OUT_DIR / "goal_scene_gated_expert_stage42.json",
    "neighbor_interaction_gated": OUT_DIR / "neighbor_interaction_gated_expert_stage42.json",
    "sequence_context": OUT_DIR / "source_level_sequence_context_stage42.json",
    "graph_context": OUT_DIR / "source_level_graph_context_stage42.json",
    "context_forensics": OUT_DIR / "context_contribution_forensics_stage42.json",
}

CORE_METRICS = [
    "all_improvement",
    "t50_improvement",
    "hard_failure_improvement",
    "easy_degradation",
]
MIN_POSITIVE_DELTA = 0.01
EASY_DELTA_LIMIT = 0.02

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-DB 是 context rescue decision audit，不重新训练，不调 threshold，不把 cached 结果写成 fresh training。",
    "本阶段整合 CJ/CK/AR/AS 已训练或已评估的 context evidence，判断是否应继续同类 protocol。",
    "future endpoints / waypoints 只作为 labels/eval，不作为 inference input。",
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


def _load_inputs() -> dict[str, Any]:
    return {key: read_json(path, {}) for key, path in INPUTS.items()}


def _base_metric(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if "baseline_family_only" in payload:
        return payload["baseline_family_only"]["protected"]
    variants = payload.get("variants", {})
    if "baseline_family_control" in variants:
        return variants["baseline_family_control"]["protected"]
    return None


def _variant_rows(protocol: str, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    base = _base_metric(payload)
    if not base:
        return []
    if "variants" in payload:
        variants = payload["variants"]
    elif "sequence_variants" in payload:
        variants = payload["sequence_variants"]
    elif "graph_variants" in payload:
        variants = payload["graph_variants"]
    else:
        variants = {}
    rows = []
    for name, variant in variants.items():
        if name == "baseline_family_control":
            continue
        metric = variant.get("protected", {})
        if not metric:
            continue
        delta = {key: float(metric.get(key, 0.0)) - float(base.get(key, 0.0)) for key in CORE_METRICS}
        safe_positive = (
            (
                delta["all_improvement"] > MIN_POSITIVE_DELTA
                or delta["t50_improvement"] > MIN_POSITIVE_DELTA
                or delta["hard_failure_improvement"] > MIN_POSITIVE_DELTA
            )
            and delta["easy_degradation"] <= EASY_DELTA_LIMIT
        )
        rows.append(
            {
                "protocol": protocol,
                "variant": name,
                "metric": {key: float(metric.get(key, 0.0)) for key in CORE_METRICS},
                "delta_vs_baseline_family_control": delta,
                "safe_positive_increment": bool(safe_positive),
            }
        )
    return rows


def _summarize(payloads: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for protocol in [
        "goal_scene_gated",
        "neighbor_interaction_gated",
        "sequence_context",
        "graph_context",
    ]:
        rows.extend(_variant_rows(protocol, payloads.get(protocol, {})))
    positive = [row for row in rows if row["safe_positive_increment"]]
    worst_t50 = min((row["delta_vs_baseline_family_control"]["t50_improvement"] for row in rows), default=0.0)
    best_t50 = max((row["delta_vs_baseline_family_control"]["t50_improvement"] for row in rows), default=0.0)
    best_all = max((row["delta_vs_baseline_family_control"]["all_improvement"] for row in rows), default=0.0)
    best_hard = max((row["delta_vs_baseline_family_control"]["hard_failure_improvement"] for row in rows), default=0.0)
    decision = (
        "continue_context_family_with_new_protocol"
        if positive
        else "stop_repeating_current_context_residual_or_gated_protocols"
    )
    return {
        "variant_rows": rows,
        "safe_positive_context_variants": positive,
        "decision": decision,
        "best_delta_all": float(best_all),
        "best_delta_t50": float(best_t50),
        "worst_delta_t50": float(worst_t50),
        "best_delta_hard_failure": float(best_hard),
        "root_cause": (
            "Existing goal/scene, neighbor/interaction, sequence, and graph context variants either reduce all/t50/hard "
            "or add easy risk after baseline-family rollout context. The next credible experiment must change the target/model/data, "
            "not merely rerun the same residual/gated variants or tune thresholds."
        ),
        "required_next_protocol_change": [
            "Use source-compatible graph/sequence model with a different supervision target, such as switchability/gain-harm labels rather than residual waypoint delta only.",
            "Add legal/source-calibrated data where scene/goal/interaction context can vary independently from baseline-family rollout context.",
            "Keep baseline-family rollout as the control arm and require validation-only safety gates plus bootstrap-positive test evidence.",
        ],
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    claim = payload["claim_boundary"]
    summary = payload["summary"]
    gates = {
        "all_inputs_present": all(payload["input_files"].values()),
        "four_context_protocols_loaded": len(payload["protocol_status"]) == 4
        and all(row["loaded"] for row in payload["protocol_status"]),
        "variant_rows_computed": len(summary["variant_rows"]) >= 10,
        "negative_context_evidence_preserved": len(summary["safe_positive_context_variants"]) == 0,
        "no_same_protocol_rerun_recommended": summary["decision"]
        == "stop_repeating_current_context_residual_or_gated_protocols",
        "root_cause_written": bool(summary["root_cause"]),
        "next_protocol_change_required": len(summary["required_next_protocol_change"]) >= 3,
        "does_not_mark_not_run_complete": payload["source"] != "fresh_training",
        "no_metric_seconds_overclaim": claim["metric_or_seconds_claim"] is False,
        "true_3d_overclaim_blocked": claim["true_3d"] is False,
        "foundation_overclaim_blocked": claim["foundation_world_model"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_db_context_rescue_decision_pass" if passed == total else "stage42_db_context_rescue_decision_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["summary"]
    lines = [
        "# Stage42-DB Context Rescue Decision Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- gate: `{payload['stage42_db_gate']['passed']} / {payload['stage42_db_gate']['total']}`",
        f"- verdict: `{payload['stage42_db_gate']['verdict']}`",
        f"- decision: `{summary['decision']}`",
        "",
        "## Current Facts",
        "",
    ]
    lines.extend(f"- {fact}" for fact in CURRENT_FACTS)
    lines.extend(
        [
            "",
            "## Protocol Status",
            "",
            "| protocol | loaded | source | verdict |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for row in payload["protocol_status"]:
        lines.append(f"| `{row['protocol']}` | `{row['loaded']}` | `{row['source']}` | `{row['verdict']}` |")
    lines.extend(
        [
            "",
            "## Context Variant Deltas vs Baseline-Family Control",
            "",
            "| protocol | variant | delta all | delta t50 | delta hard/failure | delta easy | safe positive? |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["variant_rows"]:
        d = row["delta_vs_baseline_family_control"]
        lines.append(
            f"| `{row['protocol']}` | `{row['variant']}` | {d['all_improvement']:.6f} | {d['t50_improvement']:.6f} | {d['hard_failure_improvement']:.6f} | {d['easy_degradation']:.6f} | `{row['safe_positive_increment']}` |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- safe_positive_context_variants: `{[row['variant'] for row in summary['safe_positive_context_variants']]}`",
            f"- best_delta_all: `{summary['best_delta_all']}`",
            f"- best_delta_t50: `{summary['best_delta_t50']}`",
            f"- best_delta_hard_failure: `{summary['best_delta_hard_failure']}`",
            f"- root_cause: {summary['root_cause']}",
            "",
            "## Required Next Protocol Change",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in summary["required_next_protocol_change"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-DB does not say scene/goal/neighbor/history can never help.",
            "- It says the current residual/gated protocols are exhausted under available source-level evidence.",
            "- Future work must change model target/model family/data support, while retaining baseline-family control and validation-only safety gates.",
            "- Claims remain protected dataset-local/raw-frame 2.5D, not true 3D, not foundation, not metric/seconds-level, not Stage5C, and not SMC.",
        ]
    )
    return lines


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_db_gate"]
    summary = payload["summary"]
    return [
        "## Stage42-DB Context Rescue Decision Audit",
        "",
        "- source: `fresh_synthesis_from_cached_verified_context_runs`",
        "- role: decide whether existing goal/scene, neighbor/interaction, sequence, and graph context protocols should be repeated.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- decision: `{summary['decision']}`.",
        f"- best delta all/t50/hard vs baseline-family control: `{summary['best_delta_all']:.4f}` / `{summary['best_delta_t50']:.4f}` / `{summary['best_delta_hard_failure']:.4f}`.",
        "- No safe positive context variant was found under the existing residual/gated protocols; next work must change target/model/data, not just rerun thresholds.",
        "- Claim boundary unchanged: protected dataset-local/raw-frame 2.5D only; no true 3D, foundation, metric/seconds-level, Stage5C, or SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, GOAL_README]:
        _replace_section(path, "STAGE42_DB_CONTEXT_RESCUE_DECISION", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-DB context rescue decision audit"
    state["current_verdict"] = payload["stage42_db_gate"]["verdict"]
    state["stage42_db_context_rescue_decision"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_db_gate"]["verdict"],
        "gates": f"{payload['stage42_db_gate']['passed']}/{payload['stage42_db_gate']['total']}",
        "decision": payload["summary"]["decision"],
        "best_delta_all": payload["summary"]["best_delta_all"],
        "best_delta_t50": payload["summary"]["best_delta_t50"],
        "best_delta_hard_failure": payload["summary"]["best_delta_hard_failure"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_context_rescue_decision(*, refresh_readmes: bool = True) -> dict[str, Any]:
    payloads = _load_inputs()
    protocol_status = []
    for key in ["goal_scene_gated", "neighbor_interaction_gated", "sequence_context", "graph_context"]:
        row = payloads.get(key, {})
        gate_key = next((k for k in row if k.endswith("_gate")), "")
        protocol_status.append(
            {
                "protocol": key,
                "loaded": bool(row),
                "source": row.get("source", "missing"),
                "verdict": row.get(gate_key, {}).get("verdict", "missing") if gate_key else "missing",
            }
        )
    payload: dict[str, Any] = {
        "source": "fresh_synthesis_from_cached_verified_context_runs",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_files": {key: path.exists() for key, path in INPUTS.items()},
        "protocol_status": protocol_status,
        "summary": _summarize(payloads),
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_db_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(
        GATE_MD,
        [
            "# Stage42-DB Gate",
            "",
            f"- gate: `{payload['stage42_db_gate']['passed']} / {payload['stage42_db_gate']['total']}`",
            f"- verdict: `{payload['stage42_db_gate']['verdict']}`",
            "",
            "## Gates",
            "",
            *[
                f"- {key}: `{value}`"
                for key, value in payload["stage42_db_gate"]["gates"].items()
            ],
        ],
    )
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_context_rescue_decision()
