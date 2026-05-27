from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

GL_JSON = OUT_DIR / "source_conversion_contract_stage42.json"
GM_JSON = OUT_DIR / "guarded_conversion_harness_stage42.json"
GN_JSON = OUT_DIR / "source_confirmation_priority_board_stage42.json"
GO_JSON = OUT_DIR / "official_source_terms_live_verifier_stage42.json"
GP_JSON = OUT_DIR / "source_terms_paper_claim_guard_stage42.json"
GQ_JSON = OUT_DIR / "source_terms_package_claim_linter_stage42.json"
EK_JSON = OUT_DIR / "paper_ready_evidence_matrix_stage42.json"

REPORT_JSON = OUT_DIR / "long_objective_state_reconciler_stage42.json"
REPORT_MD = OUT_DIR / "long_objective_state_reconciler_stage42.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_long_objective_state_reconciler_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_gr_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_gr_long_objective_state_reconciler"

CURRENT_FACTS = [
    "Current model is not a true 3D world model.",
    "Current model is not a large-scale foundation world model.",
    "Current model remains a protected dataset-local/raw-frame 2.5D multi-agent world-state candidate.",
    "Stage42-GR reconciles the long A-F objective against latest source/legal/package guards.",
    "Stage42-GR does not download, convert, train, or evaluate data/models.",
    "OpenTraj toolkit MIT is not treated as ETH/UCY/TrajNet/AerialMPT underlying-data permission.",
    "Future endpoints/waypoints remain labels or evaluation targets only, not inference inputs.",
    "No central velocity, no test endpoints for goals, and no test-threshold tuning are allowed.",
    "t+50/t+100 remain raw-frame horizons unless a future source-specific guard proves otherwise.",
    "Dataset-local/raw-frame evidence is not global metric or seconds-level evidence.",
    "Stage5C latent generative execution remains disabled.",
    "SMC remains disabled.",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "restricted_metric_time_claim_allowed_now": False,
    "converted_dataset_claim_allowed": False,
    "auto_download_allowed_now": False,
    "download_executed": False,
    "conversion_executed": False,
    "training_executed": False,
    "evaluation_executed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}


def _gate_passed(payload: Mapping[str, Any], preferred_key: str | None = None) -> bool:
    if preferred_key:
        gate = payload.get(preferred_key, {})
        return bool(gate) and gate.get("passed") == gate.get("total")
    for key, value in payload.items():
        if key.endswith("_gate") and isinstance(value, Mapping):
            return value.get("passed") == value.get("total")
    return False


def _gate_summary(payload: Mapping[str, Any], preferred_key: str | None = None) -> dict[str, Any]:
    if preferred_key and isinstance(payload.get(preferred_key), Mapping):
        gate = payload[preferred_key]
    else:
        gate = next((value for key, value in payload.items() if key.endswith("_gate") and isinstance(value, Mapping)), {})
    return {
        "passed": gate.get("passed", 0),
        "total": gate.get("total", 0),
        "verdict": gate.get("verdict", ""),
    }


def _input_payloads() -> dict[str, Any]:
    return {
        "gl_contract": read_json(GL_JSON, {}),
        "gm_harness": read_json(GM_JSON, {}),
        "gn_priority": read_json(GN_JSON, {}),
        "go_live_terms": read_json(GO_JSON, {}),
        "gp_claim_guard": read_json(GP_JSON, {}),
        "gq_package_linter": read_json(GQ_JSON, {}),
        "ek_evidence_matrix": read_json(EK_JSON, {}),
    }


def _input_status(inputs: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    path_map = {
        "gl_contract": (GL_JSON, "stage42_gl_gate"),
        "gm_harness": (GM_JSON, "stage42_gm_gate"),
        "gn_priority": (GN_JSON, "stage42_gn_gate"),
        "go_live_terms": (GO_JSON, "stage42_go_gate"),
        "gp_claim_guard": (GP_JSON, "stage42_gp_gate"),
        "gq_package_linter": (GQ_JSON, "stage42_gq_gate"),
    }
    out: dict[str, dict[str, Any]] = {}
    for name, (path, gate_key) in path_map.items():
        payload = inputs.get(name, {})
        out[name] = {
            "path": str(path),
            "exists": path.exists(),
            "source": payload.get("source"),
            "gate": _gate_summary(payload, gate_key),
            "gate_passed": _gate_passed(payload, gate_key),
        }
    return out


def _source_summary(inputs: Mapping[str, Any]) -> dict[str, Any]:
    contract_summary = inputs.get("gl_contract", {}).get("summary", {})
    harness_summary = inputs.get("gm_harness", {}).get("summary", {})
    priority_summary = inputs.get("gn_priority", {}).get("summary", {})
    live_summary = inputs.get("go_live_terms", {}).get("summary", {})
    linter_summary = inputs.get("gq_package_linter", {}).get("summary", {})
    return {
        "contract_ready_now": int(contract_summary.get("contract_ready_now", contract_summary.get("ready_now", 0)) or 0),
        "guarded_execution_plan_count": int(harness_summary.get("execution_plan_count", 0) or 0),
        "guarded_conversion_refused": bool(harness_summary.get("conversion_refused", True)),
        "priority_blocked_now": int(priority_summary.get("blocked_now", 0) or 0),
        "priority_ready_now": int(priority_summary.get("ready_now", 0) or 0),
        "after_terms_t50_opportunity": int(
            priority_summary.get("total_t50_after_terms", contract_summary.get("calibrated_t50_windows_after_terms", 0)) or 0
        ),
        "after_terms_t100_opportunity": int(
            priority_summary.get("total_t100_after_terms", contract_summary.get("calibrated_t100_windows_after_terms", 0)) or 0
        ),
        "official_sources_reachable": int(live_summary.get("official_sources_reachable", 0) or 0),
        "underlying_data_license_confirmed": int(live_summary.get("underlying_data_license_confirmed", 0) or 0),
        "auto_download_allowed_now": int(live_summary.get("auto_download_allowed_now", 0) or 0),
        "package_source_claim_violations": int(linter_summary.get("violation_count", 0) or 0),
        "package_files_scanned": int(linter_summary.get("files_scanned", 0) or 0),
    }


def _objective_rows(source_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_blocked = int(source_summary["contract_ready_now"]) == 0
    package_clean = int(source_summary["package_source_claim_violations"]) == 0
    return [
        {
            "objective": "A_data_and_calibration",
            "status": "blocked_user_action_required" if source_blocked else "ready_for_guarded_conversion",
            "result_source": "fresh_run",
            "proved_now": [
                "latest source/terms guard was reconciled",
                "no blocked source is counted as converted",
                "metric/seconds claims remain blocked unless source-specific conversion and calibration pass",
            ],
            "missing_now": [
                "user-confirmed official terms and local source identity for UCY/ETH/TrajNet/AerialMPT",
                "guarded source-specific conversion",
                "source-specific no-leakage and time/geometry audit",
            ],
            "next_action": "User/source confirmation first; then run guarded conversion harness.",
        },
        {
            "objective": "B_external_validation",
            "status": "cached_verified_with_source_diversity_blocker",
            "result_source": "cached_verified",
            "proved_now": [
                "protected dataset-local/raw-frame external evidence exists in earlier Stage42 package",
                "latest package linter prevents unsupported source/legal overclaims",
            ],
            "missing_now": [
                "new legally confirmed external top-down source evaluation",
                "broader source-level generalization beyond current converted sources",
            ],
            "next_action": "After legal conversion, rebuild source/scene split and evaluate once on test.",
        },
        {
            "objective": "C_full_waypoint_dynamics",
            "status": "cached_verified_protected_not_floor_free",
            "result_source": "cached_verified",
            "proved_now": [
                "protected full-waypoint/group-consistency policy family is current paper-usable evidence",
                "ungated neural dynamics remain non-deployable unless future gates pass",
            ],
            "missing_now": [
                "floor-free neural deployment safety",
                "new full-waypoint evidence on newly confirmed external sources",
            ],
            "next_action": "Keep Stage37/teacher floor for deployment; only relax on validation-proven safe slices.",
        },
        {
            "objective": "D_causal_ablation",
            "status": "mixed_claims_locked",
            "result_source": "cached_verified",
            "proved_now": [
                "core deployable claims remain protected history/domain/safe-switch/teacher-floor/full-waypoint family",
                "JEPA/Transformer/scene-goal/neighbor-interaction are not independent main claims",
            ],
            "missing_now": [
                "stable independent context-module lift",
                "new retrained ablation that changes target/architecture rather than repeating weak context protocol",
            ],
            "next_action": "Use module claim lock and linter before any paper claim update.",
        },
        {
            "objective": "E_safety_floor_research",
            "status": "pass_with_floor_required",
            "result_source": "cached_verified",
            "proved_now": [
                "Stage37/teacher floor remains necessary for current deployable claim",
                "latest source/legal guard does not change deployment floor status",
            ],
            "missing_now": [
                "safe internal/floor-free self-gate that preserves easy and proximity safety",
            ],
            "next_action": "Do not deploy ungated neural; test any floor relaxation only behind validation-selected guard.",
        },
        {
            "objective": "F_paper_package",
            "status": "claim_safe_with_open_data_blocker" if package_clean else "claim_unsafe_fix_required",
            "result_source": "fresh_run",
            "proved_now": [
                "paper package source/legal claims scanned by Stage42-GQ",
                "latest reconciler maps A-F objective state to current blockers and allowed claims",
            ],
            "missing_now": [
                "external legal conversion evidence for broad/generalized source claims",
                "metric/time-calibrated subset evidence",
            ],
            "next_action": "Keep A-journal package as protected 2.5D evidence with explicit open blockers.",
        },
    ]


def _summary(inputs: Mapping[str, Any], source_summary: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    blocked = [row["objective"] for row in rows if "blocked" in str(row["status"]) or "open" in str(row["status"])]
    return {
        "source": SOURCE,
        "objectives_reconciled": len(rows),
        "fresh_objective_rows": sum(1 for row in rows if row["result_source"] == "fresh_run"),
        "cached_verified_rows": sum(1 for row in rows if row["result_source"] == "cached_verified"),
        "not_run_rows": sum(1 for row in rows if row["result_source"] == "not_run"),
        "blocked_or_open_objectives": blocked,
        "contract_ready_now": source_summary["contract_ready_now"],
        "auto_download_allowed_now": source_summary["auto_download_allowed_now"],
        "source_claim_violations": source_summary["package_source_claim_violations"],
        "after_terms_t50_opportunity": source_summary["after_terms_t50_opportunity"],
        "after_terms_t100_opportunity": source_summary["after_terms_t100_opportunity"],
        "download_executed": False,
        "conversion_executed": False,
        "training_executed": False,
        "evaluation_executed": False,
        "current_deployable_status": "protected_dataset_local_raw_frame_2_5d_candidate",
        "next_required_action": "close user-confirmed official source terms/local path, then guarded conversion; otherwise keep paper claims bounded",
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    cb = payload["claim_boundary"]
    input_status = payload["input_status"]
    gates = {
        "source_fresh": payload["source"] == SOURCE,
        "all_source_guard_inputs_loaded": all(row["exists"] for row in input_status.values()),
        "all_source_guard_gates_passed": all(row["gate_passed"] for row in input_status.values()),
        "a_to_f_objectives_reconciled": s["objectives_reconciled"] == 6,
        "result_sources_labeled": s["fresh_objective_rows"] + s["cached_verified_rows"] + s["not_run_rows"] == 6,
        "source_claim_linter_clean": s["source_claim_violations"] == 0,
        "contract_ready_zero_preserved": s["contract_ready_now"] == 0,
        "auto_download_zero_preserved": s["auto_download_allowed_now"] == 0,
        "no_download_conversion_training_eval": not (
            s["download_executed"] or s["conversion_executed"] or s["training_executed"] or s["evaluation_executed"]
        ),
        "claim_boundary_no_metric_seconds": not cb["global_metric_claim_allowed"] and not cb["global_seconds_claim_allowed"],
        "claim_boundary_no_true3d_foundation": not cb["true_3d"] and not cb["foundation_world_model"],
        "stage5c_not_executed": not cb["stage5c_executed"],
        "smc_not_enabled": not cb["smc_enabled"],
        "user_action_required_preserved": bool(payload["user_action_required"]),
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_gr_long_objective_state_reconciler_pass" if passed == total else "stage42_gr_long_objective_state_reconciler_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_md(payload: Mapping[str, Any]) -> str:
    s = payload["summary"]
    lines = [
        "# Stage42-GR Long Objective State Reconciler",
        "",
        f"- source: `{payload['source']}`",
        f"- result source: `fresh_run` for reconciliation; model/data evidence remains `cached_verified` where stated.",
        f"- git commit: `{payload['git_commit']}`",
        f"- generated at: `{payload['generated_at']}`",
        f"- gate: `{payload['stage42_gr_gate']['passed']} / {payload['stage42_gr_gate']['total']}`",
        f"- verdict: `{payload['stage42_gr_gate']['verdict']}`",
        "",
        "## Current Claim Boundary",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Source / Legal / Conversion State",
        "",
        f"- contract ready now: `{s['contract_ready_now']}`",
        f"- auto-download allowed now: `{s['auto_download_allowed_now']}`",
        f"- package source-claim violations: `{s['source_claim_violations']}`",
        f"- after-terms t50 opportunity: `{s['after_terms_t50_opportunity']}`",
        f"- after-terms t100 opportunity: `{s['after_terms_t100_opportunity']}`",
        "- No download, conversion, training, or evaluation was executed by this reconciler.",
        "",
        "## A-F Objective Reconciliation",
        "",
        "| objective | status | result_source | next action |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload["objective_rows"]:
        lines.append(f"| `{row['objective']}` | `{row['status']}` | `{row['result_source']}` | {row['next_action']} |")
    lines.extend(
        [
            "",
            "## Required User / External Actions",
            "",
            *[f"- {item}" for item in payload["user_action_required"]],
        ]
    )
    return "\n".join(lines) + "\n"


def _render_user_action(payload: Mapping[str, Any]) -> str:
    lines = [
        "# User Action Required: Stage42-GR Long Objective State",
        "",
        "The current blocker is still source/legal confirmation, not model runtime.",
        "",
        "## Required Before Any New External Conversion Claim",
        "",
        *[f"- {item}" for item in payload["user_action_required"]],
        "",
        "## Do Not Claim Yet",
        "",
        "- Do not claim UCY/ETH/TrajNet/AerialMPT are newly legally converted.",
        "- Do not claim OpenTraj MIT covers underlying third-party data.",
        "- Do not claim metric/seconds-level performance.",
        "- Do not claim Stage5C or SMC execution.",
    ]
    return "\n".join(lines) + "\n"


def _render_gate(payload: Mapping[str, Any]) -> str:
    gate = payload["stage42_gr_gate"]
    lines = [
        "# Stage42-GR Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | `{bool(ok)}` |")
    return "\n".join(lines) + "\n"


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "## Stage42-GR Long Objective State Reconciler",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{payload['stage42_gr_gate']['verdict']}`",
        f"- gates: `{payload['stage42_gr_gate']['passed']} / {payload['stage42_gr_gate']['total']}`",
        f"- objectives reconciled: `{s['objectives_reconciled']}`",
        f"- contract ready now: `{s['contract_ready_now']}`",
        f"- auto-download allowed now: `{s['auto_download_allowed_now']}`",
        f"- package source-claim violations: `{s['source_claim_violations']}`",
        f"- after-terms opportunity: t50 `{s['after_terms_t50_opportunity']}`, t100 `{s['after_terms_t100_opportunity']}`",
        "- This is a fresh reconciliation step, not a data/model execution step.",
        "- Current deployable status remains protected dataset-local/raw-frame 2.5D candidate; no true 3D, no foundation, no global metric/seconds-level, no Stage5C, no SMC.",
    ]
    for path in [README_RESULTS, M3W_README]:
        _replace_section(path, "STAGE42_GR_REFRESH", lines)

    state = read_json(RESEARCH_STATE, {})
    state.update(
        {
            "current_stage": "Stage42-GR long objective state reconciler",
            "current_verdict": payload["stage42_gr_gate"]["verdict"],
            "stage42_gr_long_objective_state_reconciler": {
                "source": payload["source"],
                "path": str(REPORT_MD),
                "json": str(REPORT_JSON),
                "gate": payload["stage42_gr_gate"],
                "summary": s,
                "claim_boundary": payload["claim_boundary"],
            },
        }
    )
    write_json(RESEARCH_STATE, state)


def run_stage42_long_objective_state_reconciler(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    inputs = _input_payloads()
    source_summary = _source_summary(inputs)
    rows = _objective_rows(source_summary)
    user_action_required = [
        "Confirm official terms/allowed use for UCY original, ETH/BIWI, TrajNet++, and any AerialMPT/other top-down source.",
        "Confirm local raw source path and source identity; do not use derived cache as raw source proof.",
        "Only after confirmation, run guarded conversion, no-leakage audit, source-CV split, and time/geometry calibration.",
        "Keep existing paper claims bounded to protected dataset-local/raw-frame 2.5D evidence until those steps pass.",
    ]
    payload: dict[str, Any] = {
        "source": SOURCE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_status": _input_status(inputs),
        "source_summary": source_summary,
        "objective_rows": rows,
        "summary": _summary(inputs, source_summary, rows),
        "claim_boundary": CLAIM_BOUNDARY,
        "current_facts": CURRENT_FACTS,
        "user_action_required": user_action_required,
    }
    payload["stage42_gr_gate"] = _gate(payload)
    payload["evidence_hash"] = _combined_hash(
        {
            "source": payload["source"],
            "input_status": payload["input_status"],
            "source_summary": payload["source_summary"],
            "objective_rows": payload["objective_rows"],
            "claim_boundary": payload["claim_boundary"],
        }
    )
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_md(payload).splitlines())
    write_md(USER_ACTION_MD, _render_user_action(payload).splitlines())
    write_md(GATE_MD, _render_gate(payload).splitlines())
    if refresh_readmes:
        _refresh_readmes(payload)
    return payload


__all__ = ["run_stage42_long_objective_state_reconciler"]
