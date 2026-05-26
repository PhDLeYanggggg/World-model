from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "source_support_closure_audit_stage42.json"
REPORT_MD = OUT_DIR / "source_support_closure_audit_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_dd_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_source_support_closure_stage42.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_README = Path("README_M3W_GOAL_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

INPUTS = {
    "source_terms": OUT_DIR / "source_terms_validation_stage42.json",
    "time_geometry": OUT_DIR / "source_time_geometry_calibration_stage42.json",
    "t100_gap": OUT_DIR / "t100_data_gap_audit_stage42.json",
    "conversion_manifest": OUT_DIR / "source_conversion_readiness_manifest_stage42.json",
    "source_diversity_preflight": OUT_DIR / "source_diversity_conversion_preflight_stage42.json",
    "local_t100_schema": OUT_DIR / "local_t100_schema_conversion_stage42.json",
    "t100_source_cv": OUT_DIR / "t100_source_cv_repair_stage42.json",
}

DOMAINS = ["ETH_UCY", "TrajNet", "UCY"]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-DD 是 DA-1 legal/source/time-calibration closure audit，不训练模型、不下载数据、不把计划当完成。",
    "local path、parseability、source-specific calibration hints 不等于 legal conversion / deployable claim。",
    "future endpoints / future waypoints 只能作为 supervised/evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level，除非下游结果显式限制到 verified source-specific calibrated subset。",
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


def _input_status(payloads: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: {
            "path": str(INPUTS[key]),
            "exists": INPUTS[key].exists(),
            "source": payload.get("source", "missing_or_unparseable"),
            "stage": payload.get("stage", ""),
            "generated_at_utc": payload.get("generated_at_utc", ""),
        }
        for key, payload in payloads.items()
    }


def _source_specific_metric_sources(time_geometry: Mapping[str, Any], domain: str) -> list[str]:
    records = time_geometry.get("source_records", [])
    return [
        str(row.get("source_id", ""))
        for row in records
        if row.get("domain") == domain
        and (
            row.get("source_specific_metric_time_evidence") is True
            or "source_specific" in str(row.get("allowed_local_claim", ""))
            or "source_specific" in str(row.get("local_claim", ""))
        )
        and row.get("global_metric_claim_allowed") is False
        and row.get("global_seconds_claim_allowed") is False
    ]


def _preflight_targets(preflight: Mapping[str, Any], domain: str) -> list[Mapping[str, Any]]:
    rows = preflight.get("target_summaries", [])
    domain_map = {
        "ETH_UCY": ["eth_biwi_original", "opentraj_toolkit"],
        "TrajNet": ["trajnetplusplus_official", "opentraj_toolkit"],
        "UCY": ["ucy_crowd_original", "opentraj_toolkit"],
    }
    allowed = set(domain_map.get(domain, []))
    return [row for row in rows if row.get("target") in allowed or row.get("id") in allowed]


def _domain_status(domain: str, payloads: Mapping[str, Any]) -> dict[str, Any]:
    terms = payloads["source_terms"]
    time_geometry = payloads["time_geometry"]
    gap = payloads["t100_gap"]
    manifest = payloads["conversion_manifest"]
    preflight = payloads["source_diversity_preflight"]
    local = payloads["local_t100_schema"]
    source_cv = payloads["t100_source_cv"]

    terms_ready = int(terms.get("summary", {}).get("conversion_ready_targets", 0))
    conversion_ready_targets = manifest.get("conversion_ready_targets", [])
    unsupported_t100 = set(gap.get("summary", {}).get("unsupported_t100_domains", []))
    additional_needed = gap.get("summary", {}).get("additional_t100_sources_needed_by_domain", {})
    local_cv_domains = set(local.get("summary", {}).get("source_cv_domains_evaluated", []))
    local_positive = set(local.get("summary", {}).get("source_cv_domains_positive_vs_constant_velocity", []))
    t100_supported = set(source_cv.get("summary", {}).get("supported_t100_domains", []))
    source_specific_metric = _source_specific_metric_sources(time_geometry, domain)
    targets = _preflight_targets(preflight, domain)
    target_has_t50 = sum(int((row.get("t50_files", row.get("t50_capable_files", 0)) or 0) > 0) for row in targets)
    target_has_t100 = sum(int((row.get("t100_files", row.get("t100_capable_files", 0)) or 0) > 0) for row in targets)
    legal_blocked_targets = [row.get("target", row.get("id")) for row in targets if row.get("legal_blocked") or row.get("legal_terms_blocked")]

    blockers: list[str] = []
    if terms_ready == 0 or not conversion_ready_targets:
        blockers.append("source_terms_confirmation_or_conversion_readiness_missing")
    if domain in unsupported_t100 or domain not in t100_supported:
        blockers.append("train_only_t100_source_cv_support_missing")
    if int(additional_needed.get(domain, 0)) > 0:
        blockers.append(f"additional_t100_sources_needed={additional_needed.get(domain)}")
    if domain == "TrajNet" and not source_specific_metric:
        blockers.append("source_specific_metric_time_calibration_missing")
    if legal_blocked_targets:
        blockers.append(f"legal_terms_blocked_targets={','.join(map(str, legal_blocked_targets))}")
    if target_has_t50 == 0:
        blockers.append("no_ready_t50_source_diversity_target")
    if target_has_t100 == 0:
        blockers.append("no_ready_t100_source_diversity_target")

    partial_support = {
        "source_specific_metric_time_sources": source_specific_metric,
        "local_t100_schema_source_cv_evaluated": domain in local_cv_domains,
        "local_t100_schema_positive_vs_constant_velocity": domain in local_positive,
        "preflight_targets_with_t50_files": target_has_t50,
        "preflight_targets_with_t100_files": target_has_t100,
    }
    claim_status = "closed" if not blockers else "not_closed"
    return {
        "domain": domain,
        "source": "fresh_stage42_dd_synthesis_from_cached_verified_inputs",
        "claim_status": claim_status,
        "blockers": blockers,
        "partial_support": partial_support,
        "global_or_restricted_metric_seconds_claim_allowed": False,
        "global_t100_deployable_claim_allowed": False,
        "next_action": _next_action(domain, blockers, partial_support),
    }


def _next_action(domain: str, blockers: list[str], partial_support: Mapping[str, Any]) -> str:
    if not blockers:
        return "ready for a future restricted source-specific calibration/evaluation stage; Stage42-DD itself does not train or evaluate"
    if domain == "TrajNet":
        return "provide/confirm legal TrajNet++ or TrajNet-compatible long-track source with timing/geometry evidence, then rerun conversion, no-leakage, and train-only source-CV"
    if domain == "ETH_UCY":
        return "confirm ETH/BIWI or ETH-Person source terms and add enough independent t100-capable ETH_UCY train sources, then rerun source-CV without test tuning"
    if domain == "UCY":
        if partial_support.get("source_specific_metric_time_sources"):
            return "confirm UCY original terms/source identity and add one independent t100-capable UCY source or source split before claiming stable t100"
        return "provide UCY source terms and source-specific H/FPS evidence, then rerun guarded source-CV"
    return "resolve listed blockers before conversion/evaluation claims"


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    payloads = _load_inputs()
    statuses = [_domain_status(domain, payloads) for domain in DOMAINS]
    closure = {
        "domains_closed": [row["domain"] for row in statuses if row["claim_status"] == "closed"],
        "domains_not_closed": [row["domain"] for row in statuses if row["claim_status"] != "closed"],
        "global_metric_seconds_claim_allowed": False,
        "global_t100_deployable_claim_allowed": False,
        "restricted_source_specific_metric_time_candidate_exists": any(
            row["partial_support"]["source_specific_metric_time_sources"] for row in statuses
        ),
        "requires_user_or_external_state": True,
        "paper_claim": "DA-1 remains open: source-specific calibration candidates exist for ETH/UCY, but legal conversion readiness and train-only t100 source-CV support are not closed for ETH_UCY, TrajNet, or UCY.",
    }
    result: dict[str, Any] = {
        "source": "fresh_stage42_dd_source_support_closure_audit",
        "stage": "Stage42-DD Source Support Closure Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_status": _input_status(payloads),
        "domain_status": statuses,
        "closure_summary": closure,
        "user_action_required": [
            {
                "domain": row["domain"],
                "blockers": row["blockers"],
                "partial_support": row["partial_support"],
                "action": row["next_action"],
            }
            for row in statuses
            if row["claim_status"] != "closed"
        ],
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "m3w_official_metric_seconds_claim_allowed": False,
            "global_t100_deployable_claim_allowed": False,
            "raw_frame_dataset_local_global_claim_required": True,
            "converted_dataset_claim_from_stage42_dd": False,
            "evaluation_claim_from_stage42_dd": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    result["stage42_dd_gate"] = _gate(result)
    return result


def _gate(result: Mapping[str, Any]) -> dict[str, Any]:
    claim = result["claim_boundary"]
    inputs = result["input_status"]
    closure = result["closure_summary"]
    gates = {
        "all_required_inputs_loaded": all(row["exists"] for row in inputs.values()) and len(inputs) >= 7,
        "terms_validator_loaded": result["input_status"]["source_terms"]["source"].startswith("fresh_stage42_cg"),
        "time_geometry_loaded": result["input_status"]["time_geometry"]["source"].startswith("fresh_source_time"),
        "t100_gap_loaded": bool(result["input_status"]["t100_gap"]["source"]),
        "three_domains_audited": len(result["domain_status"]) == 3,
        "blockers_explicit_for_each_open_domain": all(
            row["blockers"] for row in result["domain_status"] if row["claim_status"] != "closed"
        ),
        "user_action_required_written": bool(result["user_action_required"]),
        "restricted_calibration_candidate_recorded": closure["restricted_source_specific_metric_time_candidate_exists"],
        "global_metric_claim_blocked": claim["global_metric_claim_allowed"] is False,
        "global_seconds_claim_blocked": claim["global_seconds_claim_allowed"] is False,
        "global_t100_claim_blocked": claim["global_t100_deployable_claim_allowed"] is False,
        "dd_not_counted_as_conversion": claim["converted_dataset_claim_from_stage42_dd"] is False,
        "dd_not_counted_as_evaluation": claim["evaluation_claim_from_stage42_dd"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_dd_source_support_closure_audit_pass_open_blockers" if passed == total else "stage42_dd_source_support_closure_audit_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_dd_gate"]
    lines = [
        "# Stage42-DD Source Support Closure Audit",
        "",
        f"- source: `{result['source']}`",
        f"- generated_at_utc: `{result['generated_at_utc']}`",
        f"- git_commit: `{result['git_commit']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Closure Summary",
        "",
        f"- domains_closed: `{result['closure_summary']['domains_closed']}`",
        f"- domains_not_closed: `{result['closure_summary']['domains_not_closed']}`",
        f"- restricted_source_specific_metric_time_candidate_exists: `{result['closure_summary']['restricted_source_specific_metric_time_candidate_exists']}`",
        f"- global_metric_seconds_claim_allowed: `{result['closure_summary']['global_metric_seconds_claim_allowed']}`",
        f"- global_t100_deployable_claim_allowed: `{result['closure_summary']['global_t100_deployable_claim_allowed']}`",
        f"- paper_claim: {result['closure_summary']['paper_claim']}",
        "",
        "## Domain Status",
        "",
        "| domain | status | partial support | blockers | next action |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in result["domain_status"]:
        lines.append(
            f"| `{row['domain']}` | `{row['claim_status']}` | `{row['partial_support']}` | `{row['blockers']}` | {row['next_action']} |"
        )
    lines.extend(
        [
            "",
            "## Input Status",
            "",
            "| input | exists | source | generated_at_utc |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for key, row in result["input_status"].items():
        lines.append(f"| `{key}` | `{row['exists']}` | `{row['source']}` | `{row['generated_at_utc']}` |")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            f"- claim_boundary: `{result['claim_boundary']}`",
            "",
            "## Interpretation",
            "",
            "- Stage42-DD closes the DA-1 question negatively for now: the current repository has useful calibration candidates, but not enough legal/source-CV closure for global or restricted deployable metric/seconds/t100 claims.",
            "- This is not a model-training stage and does not count any local path as legal conversion.",
        ]
    )
    return lines


def _render_user_action(result: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-DD User Action Required For Source Support Closure",
        "",
        "- source: `fresh_stage42_dd_source_support_closure_audit`",
        "- purpose: close DA-1 blockers for legal/source/time-calibrated ETH_UCY, TrajNet, and UCY evidence.",
        "",
    ]
    for row in result["user_action_required"]:
        lines.extend(
            [
                f"## {row['domain']}",
                "",
                f"- blockers: `{row['blockers']}`",
                f"- partial_support: `{row['partial_support']}`",
                f"- action: {row['action']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Non-Claims",
            "",
            "- Do not claim global metric or seconds-level M3W results from these blockers.",
            "- Do not claim global t100 deployable success while train-only source-CV support is missing.",
            "- Do not treat local path existence, parseability, or OpenTraj toolkit license as underlying dataset permission.",
        ]
    )
    return lines


def _render_gate(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_dd_gate"]
    lines = [
        "# Stage42-DD Gate",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- {key}: `{value}`" for key, value in gate["gates"].items())
    return lines


def _refresh_lines(result: Mapping[str, Any]) -> list[str]:
    gate = result["stage42_dd_gate"]
    summary = result["closure_summary"]
    return [
        "## Stage42-DD Source Support Closure Audit",
        "",
        "- source: `fresh_stage42_dd_source_support_closure_audit`",
        "- role: close or explicitly block DA-1 legal/source/time-calibration support for ETH_UCY, TrajNet, and UCY.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- domains_not_closed: `{summary['domains_not_closed']}`.",
        "- restricted ETH/UCY source-specific metric/time candidates exist, but global metric/seconds and global t100 deployable claims remain blocked.",
        "- User/external action remains required before official converted/evaluated metric-time or t100 source-CV claims.",
        "- Stage5C remains false; SMC remains false.",
    ]


def _refresh_readmes(result: Mapping[str, Any]) -> None:
    lines = _refresh_lines(result)
    for path in [README_RESULTS, M3W_README, GOAL_README]:
        _replace_section(path, "STAGE42_DD_SOURCE_SUPPORT_CLOSURE_AUDIT", lines)


def _refresh_research_state(result: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-DD source support closure audit"
    state["current_verdict"] = result["stage42_dd_gate"]["verdict"]
    state["stage42_dd_source_support_closure_audit"] = {
        "source": result["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "verdict": result["stage42_dd_gate"]["verdict"],
        "gates": f"{result['stage42_dd_gate']['passed']}/{result['stage42_dd_gate']['total']}",
        "domains_not_closed": result["closure_summary"]["domains_not_closed"],
        "global_metric_seconds_claim_allowed": result["closure_summary"]["global_metric_seconds_claim_allowed"],
        "global_t100_deployable_claim_allowed": result["closure_summary"]["global_t100_deployable_claim_allowed"],
        "claim_boundary": result["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_source_support_closure_audit(*, refresh_readmes: bool = True) -> dict[str, Any]:
    result = _build_payload()
    write_json(REPORT_JSON, result)
    write_md(REPORT_MD, _render_report(result))
    write_md(GATE_MD, _render_gate(result))
    write_md(USER_ACTION_MD, _render_user_action(result))
    if refresh_readmes:
        _refresh_readmes(result)
        _refresh_research_state(result)
    return result


if __name__ == "__main__":
    run_stage42_source_support_closure_audit()
