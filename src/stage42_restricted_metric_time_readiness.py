from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_full_waypoint_bridge_shape_audit import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

BN_JSON = OUT_DIR / "source_time_geometry_calibration_stage42.json"
CH_JSON = OUT_DIR / "metric_time_claim_guard_stage42.json"
CG_JSON = OUT_DIR / "source_terms_validation_stage42.json"
DATA_JSON = OUT_DIR / "data_calibration_stage42.json"
TEMPLATE_JSON = OUT_DIR / "source_terms_confirmation_template_stage42.json"
INTAKE_TEMPLATE_JSON = OUT_DIR / "source_terms_confirmation_intake_template_stage42.json"

REPORT_JSON = OUT_DIR / "restricted_metric_time_readiness_stage42.json"
REPORT_MD = OUT_DIR / "restricted_metric_time_readiness_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_hi_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_restricted_metric_time_readiness_stage42.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
USER_SUMMARY = Path("README_M3W_USER_DETAILED_SUMMARY_ZH.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
GOAL_SUMMARY = Path("README_M3W_GOAL_FULL_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_hi_restricted_metric_time_readiness"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-HI 是 restricted metric/time readiness recheck，不训练、不转换、不下载、不调 threshold。",
    "ETH/UCY 源级 H/FPS/stride 线索存在，但 legal/source terms readiness 仍是独立前置条件。",
    "restricted metric/time subset 只有在 user terms/path/source identity 确认、conversion、no-leakage、source-CV、final test 后才能写。",
    "SDD 仍是 pixel raw-frame；TrajNet snippets 仍无可用 homography/FPS/scale；TGSIM 只可 traffic diagnostic。",
    "future endpoint / future waypoint 只可作为 supervised/evaluation labels，不可作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 建 goals，不使用 test metrics 调 threshold。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

SOURCE_TO_TERMS_TARGET = {
    "ETH": "eth_biwi_original",
    "UCY": "ucy_crowd_original",
}


def _gate_passed(payload: Mapping[str, Any], key: str) -> bool:
    gate = payload.get(key, {})
    return bool(gate and gate.get("passed") == gate.get("total") and int(gate.get("total", 0)) > 0)


def _validation_by_dataset(cg: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("dataset_id")): row for row in cg.get("validations", [])}


def _confirmation_template_source() -> str:
    if INTAKE_TEMPLATE_JSON.exists():
        return str(INTAKE_TEMPLATE_JSON)
    return str(TEMPLATE_JSON)


def _terms_status(record: Mapping[str, Any], validations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    target = SOURCE_TO_TERMS_TARGET.get(str(record.get("dataset")))
    validation = validations.get(target or "", {})
    return {
        "terms_target_id": target,
        "official_url": validation.get("official_url", ""),
        "terms_accepted_by_user": bool(validation.get("terms_accepted_by_user")),
        "conversion_ready": bool(validation.get("conversion_ready")),
        "confirmation_blockers": list(validation.get("confirmation_blockers", [])),
        "cf_blockers": list(validation.get("cf_blockers", [])),
        "next_action": validation.get("next_action", "fill source terms confirmation"),
    }


def _readiness_row(record: Mapping[str, Any], validations: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    terms = _terms_status(record, validations)
    path = Path(str(record.get("trajectory_file", "")))
    timing = record.get("timing", {})
    homography = record.get("homography", {})
    calibration_ready = bool(record.get("source_specific_metric_time_evidence"))
    local_path_found = path.exists()
    ready_after_terms = bool(calibration_ready and local_path_found and terms["terms_target_id"])
    ready_now = bool(ready_after_terms and terms["conversion_ready"])
    blockers: list[str] = []
    if not calibration_ready:
        blockers.append("source_specific_metric_time_evidence_missing")
    if not local_path_found:
        blockers.append("trajectory_file_missing")
    if not terms["terms_target_id"]:
        blockers.append("terms_target_mapping_missing")
    if not terms["conversion_ready"]:
        blockers.append("source_terms_or_source_cv_not_conversion_ready")
    return {
        "source": "fresh_run",
        "source_id": record.get("source_id"),
        "dataset": record.get("dataset"),
        "domain": record.get("domain"),
        "terms_target_id": terms["terms_target_id"],
        "official_url": terms["official_url"],
        "trajectory_file": record.get("trajectory_file"),
        "trajectory_file_found": local_path_found,
        "homography_parseable": bool(homography.get("parseable")),
        "annotation_fps": timing.get("annotation_fps"),
        "annotation_timestep_seconds": timing.get("annotation_timestep_seconds"),
        "h50_seconds_if_restricted": timing.get("h50_annotation_seconds"),
        "h100_seconds_if_restricted": timing.get("h100_annotation_seconds"),
        "source_specific_metric_time_evidence": calibration_ready,
        "terms_accepted_by_user": terms["terms_accepted_by_user"],
        "conversion_ready_now": terms["conversion_ready"],
        "technical_conversion_ready_after_terms": ready_after_terms,
        "restricted_metric_time_ready_now": ready_now,
        "paper_claim_allowed_now": False,
        "blockers": blockers,
        "confirmation_blockers": terms["confirmation_blockers"],
        "cf_blockers": terms["cf_blockers"],
        "next_action": terms["next_action"],
    }


def _build_rows(bn: Mapping[str, Any], cg: Mapping[str, Any]) -> list[dict[str, Any]]:
    validations = _validation_by_dataset(cg)
    rows: list[dict[str, Any]] = []
    for record in bn.get("source_records", []):
        if record.get("source_specific_metric_time_evidence"):
            rows.append(_readiness_row(record, validations))
    return rows


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    claim = payload["claim_boundary"]
    gates = {
        "source_fresh": payload.get("source") == SOURCE,
        "bn_input_passed": payload["inputs"]["bn_gate_passed"] is True,
        "ch_input_passed": payload["inputs"]["ch_gate_passed"] is True,
        "cg_input_passed": payload["inputs"]["cg_gate_passed"] is True,
        "restricted_candidates_identified": summary["restricted_metric_time_candidate_count"] >= 1,
        "eth_and_ucy_candidates_present": "ETH_UCY" in summary["candidate_domains"]
        and "UCY" in summary["candidate_domains"],
        "technical_after_terms_candidates_present": summary["technical_ready_after_terms_count"] >= 1,
        "ready_now_zero": summary["restricted_metric_time_ready_now_count"] == 0,
        "paper_claim_blocked_now": claim["restricted_metric_time_claim_allowed_now"] is False,
        "global_metric_seconds_blocked": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "user_action_written": bool(payload["user_action_required"]),
        "no_training_or_conversion": summary["training_run"] is False
        and summary["conversion_executed"] is False
        and summary["evaluation_executed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = (
        "stage42_hi_restricted_metric_time_readiness_pass_blocked_by_terms"
        if passed == total
        else "stage42_hi_restricted_metric_time_readiness_partial"
    )
    return {"source": payload.get("source", SOURCE), "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    bn = read_json(BN_JSON, {})
    ch = read_json(CH_JSON, {})
    cg = read_json(CG_JSON, {})
    data = read_json(DATA_JSON, {})
    rows = _build_rows(bn, cg)
    candidate_domains = sorted({str(row["domain"]) for row in rows})
    ready_after_terms = [row for row in rows if row["technical_conversion_ready_after_terms"]]
    ready_now = [row for row in rows if row["restricted_metric_time_ready_now"]]
    user_actions = [
        {
            "terms_target_id": row["terms_target_id"],
            "source_id": row["source_id"],
            "official_url": row["official_url"],
            "action": "confirm official terms/local path/source identity, then rerun source terms validator",
            "template": _confirmation_template_source(),
            "next_command": ".venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py",
        }
        for row in rows
        if not row["restricted_metric_time_ready_now"]
    ]
    summary = {
        "source": SOURCE,
        "data_calibration_source": data.get("source"),
        "bn_verdict": bn.get("stage42_bn_gate", {}).get("verdict"),
        "ch_verdict": ch.get("stage42_ch_gate", {}).get("verdict"),
        "cg_verdict": cg.get("stage42_cg_gate", {}).get("verdict"),
        "restricted_metric_time_candidate_count": len(rows),
        "candidate_domains": candidate_domains,
        "technical_ready_after_terms_count": len(ready_after_terms),
        "restricted_metric_time_ready_now_count": len(ready_now),
        "paper_claim_allowed_now": False,
        "global_metric_claim_allowed": False,
        "global_seconds_claim_allowed": False,
        "training_run": False,
        "conversion_executed": False,
        "evaluation_executed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-HI Restricted Metric/Time Readiness",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([BN_JSON, CH_JSON, CG_JSON, DATA_JSON, TEMPLATE_JSON, INTAKE_TEMPLATE_JSON]),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "bn_gate_passed": _gate_passed(bn, "stage42_bn_gate"),
            "ch_gate_passed": _gate_passed(ch, "stage42_ch_gate"),
            "cg_gate_passed": _gate_passed(cg, "stage42_cg_gate"),
        },
        "summary": summary,
        "readiness_rows": rows,
        "user_action_required": user_actions,
        "claim_boundary": {
            "restricted_metric_time_claim_allowed_now": False,
            "restricted_claim_requires_future_conversion_eval": True,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "sdd_metric_claim_allowed": False,
            "trajnet_metric_time_claim_allowed": False,
            "tgsim_pedestrian_world_model_metric_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_hi_gate"] = _gate(payload)
    return payload


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hi_gate"]
    s = payload["summary"]
    lines = [
        "# Stage42-HI Restricted Metric/Time Readiness",
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
        "## Summary",
        "",
        f"- restricted_metric_time_candidate_count: `{s['restricted_metric_time_candidate_count']}`",
        f"- candidate_domains: `{s['candidate_domains']}`",
        f"- technical_ready_after_terms_count: `{s['technical_ready_after_terms_count']}`",
        f"- restricted_metric_time_ready_now_count: `{s['restricted_metric_time_ready_now_count']}`",
        f"- paper_claim_allowed_now: `{s['paper_claim_allowed_now']}`",
        f"- global_metric_claim_allowed: `{s['global_metric_claim_allowed']}`",
        f"- global_seconds_claim_allowed: `{s['global_seconds_claim_allowed']}`",
        "",
        "## Candidate Rows",
        "",
        "| source | domain | terms target | H | fps | h50 seconds if restricted | h100 seconds if restricted | after terms | ready now | blockers |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["readiness_rows"]:
        lines.append(
            f"| `{row['source_id']}` | `{row['domain']}` | `{row['terms_target_id']}` | "
            f"{row['homography_parseable']} | {row['annotation_fps']} | {row['h50_seconds_if_restricted']} | "
            f"{row['h100_seconds_if_restricted']} | {row['technical_conversion_ready_after_terms']} | "
            f"{row['restricted_metric_time_ready_now']} | {', '.join(row['blockers'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- ETH/UCY has restricted source-level metric/time candidates: H is parseable and annotation timing is available for selected sources.",
        "- The current claim is still blocked because source terms/path/source identity confirmation has zero conversion-ready targets.",
        "- No conversion, training, evaluation, Stage5C, or SMC execution occurred in this stage.",
        "- Current paper wording remains dataset-local/raw-frame 2.5D. A future restricted metric/time claim must run conversion, no-leakage, source-CV, and final test after user-confirmed source terms.",
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hi_gate"]
    return [
        "# Stage42-HI Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage42-HI Restricted Metric/Time Readiness",
        "",
        "- No restricted metric/time paper claim is allowed now.",
        "- Fill the source terms confirmation template only after checking the official terms, local path, and source identity.",
        "",
        "| terms target | source | official URL | action | next command |",
        "| --- | --- | --- | --- | --- |",
    ]
    seen: set[tuple[str, str]] = set()
    for row in payload["user_action_required"]:
        key = (str(row["terms_target_id"]), str(row["source_id"]))
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"| `{row['terms_target_id']}` | `{row['source_id']}` | {row['official_url']} | {row['action']} | `{row['next_command']}` |"
        )
    return lines


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hi_gate"]
    s = payload["summary"]
    return [
        "## Stage42-HI Restricted Metric/Time Readiness",
        "",
        "- source: `fresh_stage42_hi_restricted_metric_time_readiness`",
        f"- verdict: `{gate['verdict']}`",
        f"- gates: `{gate['passed']} / {gate['total']}`",
        f"- restricted metric/time candidates: `{s['restricted_metric_time_candidate_count']}` across `{s['candidate_domains']}`.",
        f"- technical ready after terms: `{s['technical_ready_after_terms_count']}`; ready now: `{s['restricted_metric_time_ready_now_count']}`.",
        "- conclusion: ETH/UCY source-level H/FPS/stride evidence exists, but no metric/seconds claim is allowed until user-confirmed source terms plus conversion/no-leakage/source-CV/final-test.",
        "- no training, conversion, download, Stage5C, or SMC occurred.",
    ]


def _refresh_a_journal_gap(payload: Mapping[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "## Stage42-HI Restricted Metric/Time Readiness Refresh",
        "",
        "- source: `fresh_stage42_hi_restricted_metric_time_readiness`",
        f"- verdict: `{payload['stage42_hi_gate']['verdict']}`",
        f"- restricted metric/time candidates: `{s['restricted_metric_time_candidate_count']}`.",
        f"- technical ready after terms: `{s['technical_ready_after_terms_count']}`.",
        f"- restricted metric/time ready now: `{s['restricted_metric_time_ready_now_count']}`.",
        "- Paper implication: source-specific ETH/UCY calibration can become a restricted subset claim only after user-confirmed terms, guarded conversion, no-leakage, source-CV, and final test.",
        "- Current global M3W claim remains raw-frame / dataset-local 2.5D.",
    ]
    _replace_section(OUT_DIR / "a_journal_gap_stage42.md", "STAGE42_HI_RESTRICTED_METRIC_TIME_READINESS", lines)


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, USER_SUMMARY, WORK_SUMMARY, GOAL_SUMMARY]:
        _replace_section(path, "STAGE42_HI_RESTRICTED_METRIC_TIME_READINESS", lines)
    _refresh_a_journal_gap(payload)


def _refresh_research_state(payload: Mapping[str, Any], *, verification: Mapping[str, Any] | None = None) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-HI restricted metric/time readiness"
    state["current_verdict"] = payload["stage42_hi_gate"]["verdict"]
    state["stage42_hi_restricted_metric_time_readiness"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "verdict": payload["stage42_hi_gate"]["verdict"],
        "gates": f"{payload['stage42_hi_gate']['passed']}/{payload['stage42_hi_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
        "verification": dict(verification or {"status": "pending"}),
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_restricted_metric_time_readiness(
    *,
    refresh_readmes: bool = True,
    verification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload, verification=verification)
    return payload


if __name__ == "__main__":
    run_stage42_restricted_metric_time_readiness()
