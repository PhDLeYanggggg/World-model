from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

FP_JSON = OUT_DIR / "h100_weak_horizon_source_support_audit_stage42.json"
FQ_JSON = OUT_DIR / "h100_source_support_repair_queue_stage42.json"
BV_JSON = OUT_DIR / "source_acquisition_status_stage42.json"
CG_JSON = OUT_DIR / "source_terms_validation_stage42.json"
LEGAL_TIME_ACTION_MD = OUT_DIR / "user_action_required_source_legal_time_stage42.md"

REPORT_JSON = OUT_DIR / "h100_blocker_closure_decision_stage42.json"
REPORT_MD = OUT_DIR / "h100_blocker_closure_decision_stage42.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_h100_blocker_closure_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_gw_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
CONSOLIDATED_SUMMARY = Path("README_M3W_CURRENT_GOAL_CONSOLIDATED_SUMMARY_ZH.md")
PAPER_EVIDENCE = OUT_DIR / "paper_ready_evidence_matrix_stage42.md"
A_JOURNAL_GAP = OUT_DIR / "a_journal_gap_stage42.md"
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_gw_h100_blocker_closure_decision"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-GW 是 h100/source/legal blocker closure decision，不下载、不转换、不训练、不评估。",
    "technical local candidate 不等于 legal conversion readiness。",
    "terms accepted、local path、source identity、guarded conversion、no-leakage/source-CV 都通过前，不能把 repair 写成完成。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "uniform_h100_or_t100_claim_allowed": False,
    "download_executed": False,
    "conversion_executed": False,
    "training_executed": False,
    "evaluation_executed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}

DATASET_BY_DOMAIN = {
    "UCY": "ucy_crowd_original",
    "TrajNet": "trajnetplusplus_official",
    "ETH_UCY": "eth_biwi_original",
    "OpenTraj": "opentraj_toolkit",
}


def _load_inputs() -> dict[str, Any]:
    return {
        "fp": read_json(FP_JSON, {}),
        "fq": read_json(FQ_JSON, {}),
        "bv": read_json(BV_JSON, {}),
        "cg": read_json(CG_JSON, {}),
    }


def _find_verdict(payload: Mapping[str, Any]) -> str:
    for key, value in payload.items():
        if key.endswith("_gate") and isinstance(value, Mapping):
            return str(value.get("verdict", ""))
    return str(payload.get("verdict", ""))


def _input_status(inputs: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    paths = {"fp": FP_JSON, "fq": FQ_JSON, "bv": BV_JSON, "cg": CG_JSON}
    return {
        key: {
            "path": str(path),
            "exists": path.exists(),
            "source": inputs.get(key, {}).get("source", ""),
            "verdict": _find_verdict(inputs.get(key, {})),
            "generated_at_utc": inputs.get(key, {}).get("generated_at_utc", ""),
        }
        for key, path in paths.items()
    }


def _validation_by_dataset(cg_payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("dataset_id", "")): row for row in cg_payload.get("validations", [])}


def _active_blocker_by_id(bv_payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("blocker_id", "")): row for row in bv_payload.get("blocker_matrix", [])}


def _domain_from_key(key: str) -> str:
    return key.split("|", 1)[0]


def _technical_support_exists(key: str, fq_row: Mapping[str, Any]) -> bool:
    status = str(fq_row.get("repair_status", {}).get("status", ""))
    if status == "hard_blocker_no_local_trajnet_h100_long_source":
        return False
    return int(fq_row.get("candidate_count", 0) or 0) > 0


def _hard_blocker(key: str, fq_row: Mapping[str, Any], bv_blockers: Mapping[str, Mapping[str, Any]]) -> str | None:
    status = str(fq_row.get("repair_status", {}).get("status", ""))
    if status == "hard_blocker_no_local_trajnet_h100_long_source":
        return "missing_official_long_raw_trajnet_source"
    if key.startswith("TrajNet") and bv_blockers.get("TrajNet_raw_long_t100_source_support", {}).get("status") == "blocked":
        return "trajnet_raw_long_t100_source_support_blocked"
    return None


def _legal_validation_for_key(key: str, validations: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    dataset_id = DATASET_BY_DOMAIN.get(_domain_from_key(key), "")
    return validations.get(dataset_id, {"dataset_id": dataset_id, "conversion_ready": False, "confirmation_blockers": ["validation_missing"]})


def _decision_for_key(
    key: str,
    fp_payload: Mapping[str, Any],
    fq_payload: Mapping[str, Any],
    bv_payload: Mapping[str, Any],
    validations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    fq_row = fq_payload.get("key_rows", {}).get(key, {})
    fp_row = fp_payload.get("audits", {}).get(key, {})
    bv_blockers = _active_blocker_by_id(bv_payload)
    legal_row = _legal_validation_for_key(key, validations)
    technical = _technical_support_exists(key, fq_row)
    legal_ready = bool(legal_row.get("conversion_ready") is True)
    hard = _hard_blocker(key, fq_row, bv_blockers)
    can_run = bool(technical and legal_ready and hard is None)
    if can_run:
        closure_status = "ready_for_future_guarded_h100_repair"
        next_action = "run guarded conversion, no-leakage, source-CV, then validation-selected h100 repair; test once only"
    elif hard:
        closure_status = "hard_blocked_missing_source_support"
        next_action = "provide a legal official long raw source before any h100/t100 repair can run"
    elif technical and not legal_ready:
        closure_status = "blocked_by_terms_and_conversion_readiness"
        next_action = "confirm official terms, allowed use, local path, and source identity; then run guarded conversion/no-leakage/source-CV"
    else:
        closure_status = "blocked_no_technical_support_candidate"
        next_action = "find a legal t100-capable source candidate before conversion or evaluation"

    return {
        "key": key,
        "domain": _domain_from_key(key),
        "result_source": "fresh_run_decision_from_cached_verified_inputs",
        "technical_support_exists": technical,
        "candidate_count": int(fq_row.get("candidate_count", 0) or 0),
        "top_candidates": fq_row.get("top_candidates", [])[:10],
        "legal_dataset_id": legal_row.get("dataset_id", DATASET_BY_DOMAIN.get(_domain_from_key(key), "")),
        "legal_conversion_ready": legal_ready,
        "terms_accepted_by_user": bool(legal_row.get("terms_accepted_by_user") is True),
        "confirmation_blockers": list(legal_row.get("confirmation_blockers", [])),
        "cf_blockers": list(legal_row.get("cf_blockers", [])),
        "fp_blockers": list(fp_row.get("blockers", [])),
        "bv_related_blocker": _related_bv_blocker(key, bv_blockers),
        "hard_blocker": hard,
        "can_run_repair_now": can_run,
        "closure_status": closure_status,
        "not_run_reason": "legal_or_source_support_not_closed" if not can_run else "",
        "next_action": next_action,
        "claim_guard": "Do not write h100/t100/uniform-horizon repair as complete until guarded conversion, no-leakage, source-CV, and final eval pass.",
    }


def _related_bv_blocker(key: str, bv_blockers: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if key.startswith("TrajNet"):
        row = bv_blockers.get("TrajNet_raw_long_t100_source_support", {})
    elif key.startswith("UCY"):
        row = bv_blockers.get("ETH_UCY_global_t100_source_support", {})
    else:
        row = {}
    return {
        "blocker_id": row.get("blocker_id", ""),
        "status": row.get("status", ""),
        "root_cause": row.get("root_cause", ""),
        "claim_allowed": row.get("claim_allowed", ""),
    }


def _summary(decisions: list[Mapping[str, Any]], inputs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    technical = [row for row in decisions if row["technical_support_exists"]]
    legal_ready = [row for row in decisions if row["legal_conversion_ready"]]
    hard = [row for row in decisions if row["hard_blocker"]]
    can_run = [row for row in decisions if row["can_run_repair_now"]]
    return {
        "source": SOURCE,
        "weak_keys": [row["key"] for row in decisions],
        "weak_key_count": len(decisions),
        "technical_support_exists_count": len(technical),
        "legal_conversion_ready_count": len(legal_ready),
        "hard_blocker_count": len(hard),
        "can_run_repair_now_count": len(can_run),
        "requires_user_action_count": len([row for row in decisions if not row["can_run_repair_now"]]),
        "uniform_h100_or_t100_claim_allowed": False,
        "download_conversion_training_eval_executed": False,
        "fp_verdict": _find_verdict(inputs["fp"]),
        "fq_verdict": _find_verdict(inputs["fq"]),
        "bv_verdict": _find_verdict(inputs["bv"]),
        "cg_verdict": _find_verdict(inputs["cg"]),
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    by_key = {row["key"]: row for row in payload["closure_decisions"]}
    gates = {
        "fp_input_loaded": payload["input_status"]["fp"]["exists"] and str(s["fp_verdict"]).startswith("stage42_fp_h100_source_support_audit_pass"),
        "fq_input_loaded": payload["input_status"]["fq"]["exists"] and str(s["fq_verdict"]).startswith("stage42_fq_h100_source_support_repair_queue_pass"),
        "bv_input_loaded": payload["input_status"]["bv"]["exists"] and str(s["bv_verdict"]).startswith("stage42_bv_source_acquisition_status_pass"),
        "cg_input_loaded": payload["input_status"]["cg"]["exists"] and str(s["cg_verdict"]).startswith("stage42_cg_source_terms_confirmation_validator_pass"),
        "weak_h100_keys_mapped": set(s["weak_keys"]) >= {"TrajNet|100", "UCY|100"},
        "per_key_decision_built": all("closure_status" in row for row in payload["closure_decisions"]),
        "trajnet_hard_blocker_preserved": by_key.get("TrajNet|100", {}).get("hard_blocker") == "missing_official_long_raw_trajnet_source",
        "ucy_technical_support_preserved": by_key.get("UCY|100", {}).get("technical_support_exists") is True,
        "ucy_legal_blocker_preserved": by_key.get("UCY|100", {}).get("legal_conversion_ready") is False,
        "no_repair_claimed_ready_now": s["can_run_repair_now_count"] == 0,
        "user_action_written": payload["user_action_required_written"] is True and len(payload["user_action_required"]) == s["requires_user_action_count"],
        "uniform_horizon_claim_blocked": payload["claim_boundary"]["uniform_h100_or_t100_claim_allowed"] is False,
        "no_download_conversion_training_eval": (
            payload["claim_boundary"]["download_executed"] is False
            and payload["claim_boundary"]["conversion_executed"] is False
            and payload["claim_boundary"]["training_executed"] is False
            and payload["claim_boundary"]["evaluation_executed"] is False
        ),
        "no_future_test_or_central_velocity_leakage": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["central_velocity"] is False
        and payload["no_leakage"]["test_endpoint_goals"] is False
        and payload["no_leakage"]["test_threshold_tuning"] is False,
        "no_metric_seconds_overclaim": payload["claim_boundary"]["global_metric_claim_allowed"] is False
        and payload["claim_boundary"]["global_seconds_claim_allowed"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_gw_h100_blocker_closure_decision_pass" if passed == total else "stage42_gw_h100_blocker_closure_decision_partial"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    inputs = _load_inputs()
    validations = _validation_by_dataset(inputs["cg"])
    weak_keys = list(inputs["fp"].get("summary", {}).get("h100_weak_horizons", []))
    decisions = [_decision_for_key(key, inputs["fp"], inputs["fq"], inputs["bv"], validations) for key in weak_keys]
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-GW H100 blocker closure decision",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([FP_JSON, FQ_JSON, BV_JSON, CG_JSON, LEGAL_TIME_ACTION_MD]),
        "current_facts": CURRENT_FACTS,
        "input_status": _input_status(inputs),
        "summary": _summary(decisions, inputs),
        "closure_decisions": decisions,
        "user_action_required": [_user_action(row) for row in decisions if not row["can_run_repair_now"]],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "local_inventory_and_decision_only": True,
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "user_action_required_written": True,
    }
    payload["stage42_gw_gate"] = _gate(payload)
    return payload


def _user_action(row: Mapping[str, Any]) -> dict[str, Any]:
    if row["hard_blocker"]:
        action_type = "provide_legal_official_long_raw_source"
        required = [
            "official source URL",
            "terms accepted / allowed use",
            "local path",
            "source identity",
            "track length sufficient for raw-frame h100/t100",
        ]
    elif row["technical_support_exists"]:
        action_type = "confirm_terms_and_run_guarded_conversion"
        required = [
            "terms accepted / allowed use",
            "acceptance date",
            "local path confirmation",
            "source identity",
            "guarded conversion and no-leakage/source-CV after confirmation",
        ]
    else:
        action_type = "find_legal_t100_capable_source_candidate"
        required = ["legal source candidate", "terms confirmation", "local path", "source identity"]
    return {
        "key": row["key"],
        "domain": row["domain"],
        "action_type": action_type,
        "closure_status": row["closure_status"],
        "hard_blocker": row["hard_blocker"],
        "required_fields": required,
        "next_action": row["next_action"],
        "claim_guard": row["claim_guard"],
    }


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_gw_gate"]
    s = payload["summary"]
    lines = [
        "# Stage42-GW H100 Blocker Closure Decision",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- result source: `fresh_run decision from cached_verified inputs`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Summary",
        "",
        f"- weak keys: `{s['weak_keys']}`",
        f"- technical support exists count: `{s['technical_support_exists_count']}`",
        f"- legal conversion ready count: `{s['legal_conversion_ready_count']}`",
        f"- hard blocker count: `{s['hard_blocker_count']}`",
        f"- can run repair now count: `{s['can_run_repair_now_count']}`",
        f"- requires user action count: `{s['requires_user_action_count']}`",
        f"- uniform h100/t100 claim allowed: `{s['uniform_h100_or_t100_claim_allowed']}`",
        "",
        "## Input Status",
        "",
        "| input | exists | verdict | generated |",
        "| --- | ---: | --- | --- |",
    ]
    for key, row in payload["input_status"].items():
        lines.append(f"| `{key}` | {row['exists']} | `{row['verdict']}` | `{row['generated_at_utc']}` |")
    lines += [
        "",
        "## Closure Decisions",
        "",
        "| key | technical support | legal ready | can run now | hard blocker | closure status | next action |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in payload["closure_decisions"]:
        lines.append(
            f"| `{row['key']}` | {row['technical_support_exists']} | {row['legal_conversion_ready']} | "
            f"{row['can_run_repair_now']} | `{row['hard_blocker']}` | `{row['closure_status']}` | {row['next_action']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- `UCY|100`: technical local candidates exist, but legal/conversion readiness is false, so h100 repair cannot run now.",
        "- `TrajNet|100`: current local TrajNet snippets do not provide long raw h100/t100 support, so this remains a hard source-support blocker.",
        "- No download, conversion, training, or evaluation is executed in this stage.",
        "- Uniform h100/t100 robustness remains blocked; reports must keep raw-frame/dataset-local wording.",
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | {bool(ok)} |")
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage42-GW H100 Blocker Closure",
        "",
        "No h100/t100 repair is conversion-ready now. Do not count the rows below as converted, evaluated, repaired, metric, or seconds-level evidence.",
        "",
    ]
    for row in payload["user_action_required"]:
        lines += [
            f"## `{row['key']}`",
            "",
            f"- domain: `{row['domain']}`",
            f"- action_type: `{row['action_type']}`",
            f"- closure_status: `{row['closure_status']}`",
            f"- hard_blocker: `{row['hard_blocker']}`",
            f"- required_fields: `{row['required_fields']}`",
            f"- next_action: {row['next_action']}",
            f"- claim_guard: {row['claim_guard']}",
            "",
        ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_gw_gate"]
    lines = [
        "# Stage42-GW Gate",
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


def _refresh_docs(payload: Mapping[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "## Stage42-GW H100 Blocker Closure Decision",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{payload['stage42_gw_gate']['passed']} / {payload['stage42_gw_gate']['total']}`; verdict `{payload['stage42_gw_gate']['verdict']}`",
        f"- weak keys: `{s['weak_keys']}`",
        f"- technical support exists count: `{s['technical_support_exists_count']}`; legal conversion ready count: `{s['legal_conversion_ready_count']}`; can run repair now count: `{s['can_run_repair_now_count']}`",
        "- `UCY|100`: local technical candidates exist, but terms/source identity/guarded conversion are not ready; user action required before repair.",
        "- `TrajNet|100`: hard blocker remains because current local TrajNet snippets are too short for raw-frame h100/t100 repair.",
        "- Boundary: no download, no conversion, no training, no evaluation; uniform h100/t100 claim remains blocked; no metric/seconds, no Stage5C, no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, CONSOLIDATED_SUMMARY, PAPER_EVIDENCE, A_JOURNAL_GAP]:
        _replace_section(path, "STAGE42_GW_H100_BLOCKER_CLOSURE_DECISION", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    s = payload["summary"]
    state["current_stage"] = "Stage42-GW h100 blocker closure decision"
    state["current_verdict"] = payload["stage42_gw_gate"]["verdict"]
    state["stage42_gw_h100_blocker_closure_decision"] = {
        "source": payload["source"],
        "result_source": "fresh_run_decision_from_cached_verified_inputs",
        "report": str(REPORT_MD),
        "report_json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "verdict": payload["stage42_gw_gate"]["verdict"],
        "gates": f"{payload['stage42_gw_gate']['passed']}/{payload['stage42_gw_gate']['total']}",
        "weak_keys": s["weak_keys"],
        "technical_support_exists_count": s["technical_support_exists_count"],
        "legal_conversion_ready_count": s["legal_conversion_ready_count"],
        "can_run_repair_now_count": s["can_run_repair_now_count"],
        "uniform_h100_or_t100_claim_allowed": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "conclusion": "UCY|100 has technical local candidates but remains legal/conversion blocked; TrajNet|100 remains hard-blocked by missing long raw source support.",
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_h100_blocker_closure_decision() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    _refresh_docs(payload)
    _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_h100_blocker_closure_decision()
    gate = result["stage42_gw_gate"]
    print(f"Stage42-GW gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
