from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

FL_JSON = OUT_DIR / "fh_horizon_weak_slice_forensics_stage42.json"
FM_JSON = OUT_DIR / "fh_horizon_row_switch_specialist_stage42.json"
FN_JSON = OUT_DIR / "fh_horizon_conservative_easy_guard_stage42.json"
FO_JSON = OUT_DIR / "fh_horizon_gain_harm_specialist_stage42.json"
FP_JSON = OUT_DIR / "h100_weak_horizon_source_support_audit_stage42.json"
FQ_JSON = OUT_DIR / "h100_source_support_repair_queue_stage42.json"
FW_JSON = OUT_DIR / "source_action_consolidator_stage42.json"

REPORT_JSON = OUT_DIR / "horizon_retry_decision_map_stage42.json"
REPORT_MD = OUT_DIR / "horizon_retry_decision_map_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_fy_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_horizon_retry_decision_stage42.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
ONE_FILE_SUMMARY = Path("README_M3W_ONE_FILE_DETAILED_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_horizon_retry_decision_map_from_fl_fq"

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "metric_seconds_claim": False,
    "download_executed": False,
    "conversion_executed": False,
    "training_executed": False,
    "evaluation_executed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}


def _load_inputs() -> dict[str, Any]:
    return {
        "fl_forensics": read_json(FL_JSON, {}),
        "fm_row_switch": read_json(FM_JSON, {}),
        "fn_easy_guard": read_json(FN_JSON, {}),
        "fo_gain_harm": read_json(FO_JSON, {}),
        "fp_source_support": read_json(FP_JSON, {}),
        "fq_repair_queue": read_json(FQ_JSON, {}),
        "fw_source_action": read_json(FW_JSON, {}),
    }


def _passed_gate(payload: Mapping[str, Any], key: str) -> bool:
    gate = payload.get(key, {})
    return bool(gate) and gate.get("passed") == gate.get("total")


def _verdict(payload: Mapping[str, Any]) -> str:
    if payload.get("verdict"):
        return str(payload["verdict"])
    for key, value in payload.items():
        if str(key).endswith("_gate") and isinstance(value, Mapping) and value.get("verdict"):
            return str(value["verdict"])
    return ""


def _weak_keys(inputs: Mapping[str, Any]) -> list[str]:
    candidates: list[str] = []
    for name, key in [
        ("fp_source_support", "h100_weak_horizons"),
        ("fq_repair_queue", "weak_keys"),
    ]:
        values = inputs.get(name, {}).get(key, [])
        candidates.extend(str(v) for v in values)
    fo_after = inputs.get("fo_gain_harm", {}).get("summary", {}).get("weak_domain_horizons_after", [])
    candidates.extend(str(v) for v in fo_after)
    out: list[str] = []
    for item in candidates:
        if item and item not in out:
            out.append(item)
    return out


def _attempt_rows(inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "attempt": "FL weak-horizon forensics",
            "source": "fresh_run",
            "verdict": _verdict(inputs.get("fl_forensics", {})),
            "outcome": "diagnosed low-margin ambiguous oracle labels",
            "policy_promoted": False,
        },
        {
            "attempt": "FM row-level switch specialist",
            "source": "fresh_run",
            "verdict": _verdict(inputs.get("fm_row_switch", {})),
            "outcome": "repaired one weak horizon but left TrajNet|100 and UCY|100 weak",
            "policy_promoted": False,
        },
        {
            "attempt": "FN conservative easy guard",
            "source": "fresh_run",
            "verdict": _verdict(inputs.get("fn_easy_guard", {})),
            "outcome": "did not repair remaining h100 weak horizons",
            "policy_promoted": False,
        },
        {
            "attempt": "FO gain/harm specialist",
            "source": "fresh_run",
            "verdict": _verdict(inputs.get("fo_gain_harm", {})),
            "outcome": "partial global metrics but uniform horizon claim still blocked",
            "policy_promoted": False,
        },
        {
            "attempt": "FP/FQ source-support audit and repair queue",
            "source": "fresh_run",
            "verdict": _verdict(inputs.get("fq_repair_queue", {})),
            "outcome": "TrajNet h100 needs longer legal source; UCY h100 needs terms and guarded conversion",
            "policy_promoted": False,
        },
    ]


def _decision_rows(inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    fq = inputs.get("fq_repair_queue", {})
    key_rows = fq.get("key_rows", {})
    fw_actions = {row.get("target"): row for row in inputs.get("fw_source_action", {}).get("consolidated_actions", [])}
    rows: list[dict[str, Any]] = []
    for weak_key in _weak_keys(inputs):
        queue_row = key_rows.get(weak_key, {})
        status = queue_row.get("repair_status", {})
        domain = weak_key.split("|", 1)[0]
        if "TrajNet" in weak_key:
            decision = "stop_model_retry_until_longer_legal_source"
            required = [
                "official longer TrajNet-compatible raw source",
                "terms confirmation",
                "guarded conversion",
                "no-leakage audit",
                "train-only source-CV",
            ]
            next_action_ids = ["FW-H100-TrajNet|100", "FW-DOMAIN-TrajNet", "FW-TERMS-trajnetplusplus_official"]
        elif "UCY" in weak_key:
            decision = "stop_model_retry_until_terms_and_guarded_conversion"
            required = [
                "UCY original terms/user confirmation",
                "guarded conversion of local h100 candidates",
                "no-leakage audit",
                "train-only source-CV",
            ]
            next_action_ids = ["FW-TERMS-ucy_crowd_original", "FW-H100-UCY|100", "FW-DOMAIN-UCY"]
        else:
            decision = "stop_model_retry_until_source_support_closure"
            required = ["source support closure", "guarded conversion", "no-leakage audit"]
            next_action_ids = []
        rows.append(
            {
                "weak_key": weak_key,
                "domain": domain,
                "decision": decision,
                "result_source": "fresh_decision_from_cached_verified_repair_attempts",
                "repair_status": status.get("status") or queue_row.get("repair_status"),
                "candidate_count": int(queue_row.get("candidate_count", 0) or 0),
                "required_before_retry": required,
                "blocked_retries": [
                    "repeat row_switch on same features",
                    "repeat conservative easy guard on same features",
                    "repeat gain/harm specialist on same source support",
                    "claim uniform horizon robustness",
                ],
                "allowed_retry_after": [
                    "new legal source support",
                    "new train-only validation source family",
                    "new guarded converted h100 rows",
                    "fresh no-leakage and source-CV pass",
                ],
                "next_action_ids": next_action_ids,
                "next_actions": [fw_actions.get(action_id, {}).get("next_user_action", action_id) for action_id in next_action_ids],
            }
        )
    return rows


def _summary(decisions: list[Mapping[str, Any]], attempts: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "weak_horizon_count": len(decisions),
        "weak_horizons": [row["weak_key"] for row in decisions],
        "model_retry_attempts_considered": len(attempts),
        "promoted_policy_count": sum(1 for row in attempts if row.get("policy_promoted")),
        "stop_repeat_modeling_now": True,
        "uniform_horizon_claim_allowed": False,
        "highest_priority_data_action": "FW-TERMS-ucy_crowd_original",
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    decisions = list(payload.get("horizon_retry_decisions", []))
    summary = payload.get("summary", {})
    claim = payload.get("claim_boundary", {})
    gates = {
        "source_fresh": payload.get("source") == SOURCE,
        "weak_horizons_identified": set(summary.get("weak_horizons", [])) >= {"TrajNet|100", "UCY|100"},
        "multiple_model_retries_considered": int(summary.get("model_retry_attempts_considered", 0)) >= 4,
        "no_policy_promoted_from_failed_retries": int(summary.get("promoted_policy_count", -1)) == 0,
        "stop_repeat_modeling_now": summary.get("stop_repeat_modeling_now") is True,
        "uniform_horizon_not_claimed": summary.get("uniform_horizon_claim_allowed") is False,
        "every_weak_key_has_decision": all(row.get("decision") for row in decisions) and len(decisions) >= 2,
        "each_decision_has_allowed_retry_conditions": all(row.get("allowed_retry_after") for row in decisions),
        "each_decision_has_user_actions": all(row.get("next_action_ids") for row in decisions),
        "no_download_conversion_training_eval": claim.get("download_executed") is False
        and claim.get("conversion_executed") is False
        and claim.get("training_executed") is False
        and claim.get("evaluation_executed") is False,
        "no_metric_seconds_overclaim": claim.get("metric_seconds_claim") is False,
        "no_true3d_foundation_overclaim": claim.get("true_3d") is False and claim.get("foundation_world_model") is False,
        "stage5c_false": claim.get("stage5c_executed") is False,
        "smc_false": claim.get("smc_enabled") is False,
    }
    passed = sum(bool(v) for v in gates.values())
    total = len(gates)
    return {
        "passed": passed,
        "total": total,
        "gates": gates,
        "verdict": "stage42_fy_horizon_retry_decision_pass" if passed == total else "stage42_fy_horizon_retry_decision_partial",
    }


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["summary"]
    lines = [
        "# Stage42-FY Horizon Retry Decision Map",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_fy_gate']['passed']} / {payload['stage42_fy_gate']['total']}`",
        f"- verdict: `{payload['stage42_fy_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        "- 当前不是 true 3D world model。",
        "- 当前不是 foundation world model。",
        "- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
        "- Stage42-FY does not train, evaluate, download, convert, or tune thresholds.",
        "- It decides when weak-horizon modeling retries are scientifically justified.",
        "- Stage5C is not executed; SMC is not enabled.",
        "",
        "## Summary",
        "",
        f"- weak_horizons: `{summary['weak_horizons']}`",
        f"- model_retry_attempts_considered: `{summary['model_retry_attempts_considered']}`",
        f"- promoted_policy_count: `{summary['promoted_policy_count']}`",
        f"- stop_repeat_modeling_now: `{summary['stop_repeat_modeling_now']}`",
        f"- uniform_horizon_claim_allowed: `{summary['uniform_horizon_claim_allowed']}`",
        f"- highest_priority_data_action: `{summary['highest_priority_data_action']}`",
        "",
        "## Prior Attempts Considered",
        "",
        "| attempt | verdict | outcome | policy promoted |",
        "| --- | --- | --- | ---: |",
    ]
    for row in payload["prior_attempts"]:
        lines.append(f"| `{row['attempt']}` | `{row['verdict']}` | {row['outcome']} | {row['policy_promoted']} |")
    lines.extend(
        [
            "",
            "## Retry Decisions",
            "",
            "| weak key | decision | required before retry | blocked retries | next actions |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in payload["horizon_retry_decisions"]:
        lines.append(
            "| `{}` | `{}` | {} | {} | {} |".format(
                row["weak_key"],
                row["decision"],
                "<br>".join(row["required_before_retry"]),
                "<br>".join(row["blocked_retries"]),
                "<br>".join(row["next_action_ids"]),
            )
        )
    lines.extend(["", "## Gate", "", "| gate | pass |", "| --- | ---: |"])
    for gate, ok in payload["stage42_fy_gate"]["gates"].items():
        lines.append(f"| `{gate}` | {bool(ok)} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Current weak h100 horizons should not receive more same-feature model retries until new legal source support or train-only validation support exists.",
            "- This protects the research loop from overfitting or repeatedly tuning on weak, low-margin slices.",
            "- The correct next move is source/legal support closure, not claiming uniform horizon robustness.",
        ]
    )
    return lines


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-FY Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | {bool(ok)} |")
    write_md(GATE_MD, lines)


def _write_user_action(payload: Mapping[str, Any]) -> None:
    lines = [
        "# User Action Required: Stage42-FY Horizon Retry Decision",
        "",
        "Do not run more same-feature h100 weak-horizon model retries until these blockers are closed.",
        "",
        "| weak key | required before retry | next action ids |",
        "| --- | --- | --- |",
    ]
    for row in payload["horizon_retry_decisions"]:
        lines.append(
            "| `{}` | {} | {} |".format(
                row["weak_key"],
                "<br>".join(row["required_before_retry"]),
                "<br>".join(row["next_action_ids"]),
            )
        )
    write_md(USER_ACTION_MD, lines)


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "## Stage42-FY Horizon Retry Decision Map",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{payload['stage42_fy_gate']['passed']} / {payload['stage42_fy_gate']['total']}`; verdict `{payload['stage42_fy_gate']['verdict']}`.",
        f"- weak horizons: `{summary['weak_horizons']}`.",
        f"- model retry attempts considered: `{summary['model_retry_attempts_considered']}`; promoted policy count `{summary['promoted_policy_count']}`.",
        f"- decision: stop repeating same-feature weak-horizon model retries now = `{summary['stop_repeat_modeling_now']}`.",
        f"- highest-priority unblocker: `{summary['highest_priority_data_action']}`.",
        "- role: retry decision map for h100 weak slices; no training, no download, no conversion, no threshold tuning.",
        "- boundary: uniform horizon robustness remains blocked; protected dataset-local/raw-frame 2.5D only; no metric/seconds, true 3D, foundation, Stage5C, or SMC claim.",
    ]
    for path in [README_RESULTS, M3W_README, ONE_FILE_SUMMARY]:
        _replace_section(path, "STAGE42_FY_HORIZON_RETRY_DECISION_MAP", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-FY horizon retry decision map"
    state["current_verdict"] = payload["stage42_fy_gate"]["verdict"]
    state["stage42_fy_horizon_retry_decision_map"] = {
        "source": payload["source"],
        "path": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "updated_at": payload["generated_at_utc"],
        "gate": payload["stage42_fy_gate"],
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_horizon_retry_decision_map() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    inputs = _load_inputs()
    attempts = _attempt_rows(inputs)
    decisions = _decision_rows(inputs)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([FL_JSON, FM_JSON, FN_JSON, FO_JSON, FP_JSON, FQ_JSON, FW_JSON]),
        "input_status": {
            name: {
                "exists": path.exists(),
                "verdict": _verdict(inputs.get(name, {})),
            }
            for name, path in [
                ("fl_forensics", FL_JSON),
                ("fm_row_switch", FM_JSON),
                ("fn_easy_guard", FN_JSON),
                ("fo_gain_harm", FO_JSON),
                ("fp_source_support", FP_JSON),
                ("fq_repair_queue", FQ_JSON),
                ("fw_source_action", FW_JSON),
            ]
        },
        "prior_attempts": attempts,
        "horizon_retry_decisions": decisions,
        "summary": _summary(decisions, attempts),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["stage42_fy_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    _write_gate(payload["stage42_fy_gate"])
    _write_user_action(payload)
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


__all__ = ["run_stage42_horizon_retry_decision_map", "_gate", "_decision_rows", "_summary"]
