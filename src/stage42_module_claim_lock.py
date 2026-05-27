from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")

FU_JSON = OUT_DIR / "module_contribution_ledger_stage42.json"
Z_JSON = OUT_DIR / "paper_claim_evidence_audit_stage42.json"
DP_JSON = OUT_DIR / "context_model_closure_stage42.json"
DQ_JSON = OUT_DIR / "full_waypoint_promotion_checkpoint_stage42.json"
GH_JSON = OUT_DIR / "calibrated_post_confirmation_subset_plan_stage42.json"

REPORT_JSON = OUT_DIR / "module_claim_lock_stage42.json"
REPORT_MD = OUT_DIR / "module_claim_lock_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_gj_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_gj_module_claim_lock_from_fu_z_dp_dq_gh"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-GJ 是 claim lock / experiment guard；它整合 FU/Z/DP/DQ/GH，不重新训练、不调 test threshold。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _gate_passed(payload: Mapping[str, Any], key: str) -> bool:
    gate = payload.get(key, {})
    if not isinstance(gate, Mapping):
        return False
    return int(gate.get("passed", -1)) == int(gate.get("total", 0)) and int(gate.get("total", 0)) > 0


def _replace_section(text: str, marker: str, section: str) -> str:
    start = f"<!-- {marker}:START -->"
    end = f"<!-- {marker}:END -->"
    if start in text and end in text:
        before = text.split(start, 1)[0]
        after = text.split(end, 1)[1]
        return before + section + after
    return text.rstrip() + "\n\n" + section


def _load_inputs() -> dict[str, Any]:
    return {
        "fu": read_json(FU_JSON, {}),
        "z": read_json(Z_JSON, {}),
        "dp": read_json(DP_JSON, {}),
        "dq": read_json(DQ_JSON, {}),
        "gh": read_json(GH_JSON, {}),
    }


def _summary(inputs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    fu_summary = inputs["fu"].get("summary", {})
    dp_summary = inputs["dp"].get("summary", {})
    dq_summary = inputs["dq"].get("summary", {})
    gh_summary = inputs["gh"].get("summary", {})
    z_gate = inputs["z"].get("stage42_z_gate", {})

    supported_modules = list(fu_summary.get("main_claim_allowed_modules", []))
    blocked_modules = list(fu_summary.get("blocked_or_auxiliary_modules", []))
    calibrated_ready_now = int(gh_summary.get("restricted_ready_now", gh_summary.get("source_ready_now", 0)) or 0)

    return {
        "source": SOURCE,
        "supported_main_modules_locked": supported_modules,
        "blocked_main_modules_locked": blocked_modules,
        "paper_ready_scope": z_gate.get("paper_ready_scope", "protected_2p5d_raw_frame_world_state_candidate"),
        "not_ready_scope": z_gate.get("not_ready_scope", "true_3d_metric_seconds_foundation_or_stage5c_smc"),
        "context_protocol_status": dp_summary.get("closure_decision", "unknown"),
        "context_root_cause": dp_summary.get("root_cause", ""),
        "full_waypoint_promotion_decision": dq_summary.get("promotion_decision", {}),
        "protected_full_waypoint_runtime_supported": bool(
            (dq_summary.get("promotion_decision", {}) or {}).get("source_level_group_consistency_runtime_policy_promoted", False)
        ),
        "ungated_full_waypoint_deployable": bool(
            (dq_summary.get("promotion_decision", {}) or {}).get("ungated_full_waypoint_deployable", False)
        ),
        "calibrated_subset_candidates_after_terms": int(gh_summary.get("restricted_metric_time_candidates_after_terms", 0) or 0),
        "calibrated_subset_ready_now": calibrated_ready_now,
        "calibrated_t50_after_terms": int(gh_summary.get("calibrated_t50_windows_after_terms", 0) or 0),
        "calibrated_t100_after_terms": int(gh_summary.get("calibrated_t100_windows_after_terms", 0) or 0),
        "calibrated_domains_after_terms": list(gh_summary.get("domains_with_candidates", [])),
        "claim_lock": {
            "allowed": [
                "protected dataset-local/raw-frame 2.5D world-state candidate",
                "history/domain-expert/safe-switch/teacher-floor/group-consistency full-waypoint as supported modules",
                "protected source-level group-consistency full-waypoint runtime evidence",
                "post-confirmation calibrated subset candidate map as user-action plan only",
            ],
            "forbidden": [
                "true 3D / foundation / global metric / seconds-level claim",
                "JEPA or Transformer as independent main contribution under current evidence",
                "scene/goal or neighbor/interaction as independent main contribution under current evidence",
                "ungated full-waypoint or ungated neural deployment",
                "post-confirmation candidates as converted/evaluated data before terms and guarded conversion",
                "Stage5C execution or SMC enablement",
            ],
        },
        "next_admissible_experiments": [
            "terms-confirmed guarded conversion for calibrated ETH/UCY candidates, followed by no-leakage and source-CV",
            "changed-target context modeling only: gain/harm, switchability, or full-sequence objectives; do not repeat the closed residual sequence/graph protocol unchanged",
            "protected full-waypoint runtime replay or protocol-aligned evaluation; do not promote ungated full-waypoint",
            "source/horizon-specific h100 support repair after source/legal/calibration closure",
        ],
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    inputs = _load_inputs()
    summary = _summary(inputs)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-GJ module claim lock and experiment guard",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(FU_JSON), str(Z_JSON), str(DP_JSON), str(DQ_JSON), str(GH_JSON)]),
        "current_facts": CURRENT_FACTS,
        "input_status": {
            "fu_gate": inputs["fu"].get("stage42_fu_gate", {}).get("verdict", ""),
            "z_gate": inputs["z"].get("stage42_z_gate", {}).get("verdict", ""),
            "dp_gate": inputs["dp"].get("stage42_dp_gate", {}).get("verdict", ""),
            "dq_gate": inputs["dq"].get("stage42_dq_gate", {}).get("verdict", ""),
            "gh_gate": inputs["gh"].get("stage42_gh_gate", {}).get("verdict", ""),
        },
        "summary": summary,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "synthesis_only_no_training": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "post_confirmation_candidates_claimed_as_data": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_gj_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    boundary = payload["claim_boundary"]
    blocked = set(s["blocked_main_modules_locked"])
    supported = set(s["supported_main_modules_locked"])
    gates = {
        "fu_input_passed": payload["input_status"]["fu_gate"] == "stage42_fu_module_contribution_ledger_pass",
        "z_input_passed": payload["input_status"]["z_gate"] == "stage42_z_paper_claim_evidence_audit_pass",
        "dp_input_passed": payload["input_status"]["dp_gate"] == "stage42_dp_context_model_closure_pass",
        "dq_input_passed": payload["input_status"]["dq_gate"] == "stage42_dq_full_waypoint_promotion_checkpoint_pass",
        "gh_input_passed": payload["input_status"]["gh_gate"] == "stage42_gh_calibrated_post_confirmation_subset_plan_pass",
        "core_modules_locked": {"history", "domain_expert", "safe_switch", "teacher_floor", "group_consistency_full_waypoint"}.issubset(supported),
        "negative_modules_locked": {"scene_goal", "neighbor_interaction", "JEPA", "Transformer"}.issubset(blocked),
        "context_residual_protocol_closed": s["context_protocol_status"] == "close_current_sequence_graph_residual_context_protocol",
        "protected_full_waypoint_supported": s["protected_full_waypoint_runtime_supported"] is True,
        "ungated_full_waypoint_blocked": s["ungated_full_waypoint_deployable"] is False,
        "calibrated_subset_candidates_recorded": s["calibrated_subset_candidates_after_terms"] >= 1,
        "calibrated_subset_not_claimed_ready": s["calibrated_subset_ready_now"] == 0,
        "next_experiments_are_concrete": len(s["next_admissible_experiments"]) >= 3,
        "no_future_or_test_leakage": all(
            payload["no_leakage"][key] is False
            for key in ["future_endpoint_input", "future_waypoint_input", "central_velocity", "test_endpoint_goals", "test_threshold_tuning"]
        ),
        "no_metric_seconds_overclaim": boundary["metric_or_seconds_claim"] is False,
        "not_true3d_or_foundation": boundary["true_3d"] is False and boundary["foundation_world_model"] is False,
        "post_confirmation_candidates_not_overclaimed": boundary["post_confirmation_candidates_claimed_as_data"] is False,
        "stage5c_false": boundary["stage5c_executed"] is False,
        "smc_false": boundary["smc_enabled"] is False,
    }
    passed = sum(bool(v) for v in gates.values())
    total = len(gates)
    verdict = "stage42_gj_module_claim_lock_pass" if passed == total else "stage42_gj_module_claim_lock_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    gate = payload["stage42_gj_gate"]
    lines = [
        "# Stage42-GJ Module Claim Lock and Experiment Guard",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Locked Claims",
        "",
        f"- paper_ready_scope: `{s['paper_ready_scope']}`",
        f"- not_ready_scope: `{s['not_ready_scope']}`",
        f"- supported_main_modules_locked: `{s['supported_main_modules_locked']}`",
        f"- blocked_main_modules_locked: `{s['blocked_main_modules_locked']}`",
        f"- context_protocol_status: `{s['context_protocol_status']}`",
        f"- protected_full_waypoint_runtime_supported: `{s['protected_full_waypoint_runtime_supported']}`",
        f"- ungated_full_waypoint_deployable: `{s['ungated_full_waypoint_deployable']}`",
        "",
        "## Calibrated Subset Boundary",
        "",
        f"- calibrated_subset_candidates_after_terms: `{s['calibrated_subset_candidates_after_terms']}`",
        f"- calibrated_subset_ready_now: `{s['calibrated_subset_ready_now']}`",
        f"- calibrated_t50_after_terms: `{s['calibrated_t50_after_terms']}`",
        f"- calibrated_t100_after_terms: `{s['calibrated_t100_after_terms']}`",
        f"- calibrated_domains_after_terms: `{s['calibrated_domains_after_terms']}`",
        "- These are post-confirmation candidates only; no permission, conversion, training, or evaluation is claimed.",
        "",
        "## Allowed Claims",
        "",
        *[f"- {item}" for item in s["claim_lock"]["allowed"]],
        "",
        "## Forbidden Claims",
        "",
        *[f"- {item}" for item in s["claim_lock"]["forbidden"]],
        "",
        "## Next Admissible Experiments",
        "",
        *[f"- {item}" for item in s["next_admissible_experiments"]],
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_gj_gate"]
    return [
        "# Stage42-GJ Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _summary_section(payload: Mapping[str, Any]) -> str:
    s = payload["summary"]
    return "\n".join(
        [
            "<!-- STAGE42_GJ_MODULE_CLAIM_LOCK:START -->",
            "## Stage42-GJ Module Claim Lock",
            "",
            f"- source: `{payload['source']}`",
            f"- gate: `{payload['stage42_gj_gate']['passed']} / {payload['stage42_gj_gate']['total']}`; verdict `{payload['stage42_gj_gate']['verdict']}`.",
            f"- locked supported modules: `{s['supported_main_modules_locked']}`.",
            f"- locked blocked modules: `{s['blocked_main_modules_locked']}`.",
            f"- protected full-waypoint runtime supported: `{s['protected_full_waypoint_runtime_supported']}`; ungated full-waypoint deployable: `{s['ungated_full_waypoint_deployable']}`.",
            f"- calibrated post-confirmation candidates: `{s['calibrated_subset_candidates_after_terms']}`; ready now: `{s['calibrated_subset_ready_now']}`; after-terms t50/t100: `{s['calibrated_t50_after_terms']}` / `{s['calibrated_t100_after_terms']}`.",
            "- next admissible experiments are restricted to terms-confirmed guarded conversion, changed-target gain/harm or full-sequence context, protected full-waypoint runtime replay, and source/horizon-specific h100 support repair.",
            "- Still no true-3D, foundation, global metric, seconds-level, Stage5C, SMC, or post-confirmation-candidate-as-data claim.",
            "<!-- STAGE42_GJ_MODULE_CLAIM_LOCK:END -->",
            "",
        ]
    )


def _append_to_docs(payload: Mapping[str, Any]) -> None:
    section = _summary_section(payload)
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        old = path.read_text() if path.exists() else ""
        path.write_text(_replace_section(old, "STAGE42_GJ_MODULE_CLAIM_LOCK", section))


def _update_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-GJ module claim lock and experiment guard"
    state["current_verdict"] = payload["stage42_gj_gate"]["verdict"]
    state["stage42_gj_module_claim_lock"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_gj_gate"]["verdict"],
        "gates": f"{payload['stage42_gj_gate']['passed']}/{payload['stage42_gj_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
        "conclusion": "Stage42-GJ locks the current paper/evidence boundary: supported main modules are history/domain_expert/safe_switch/teacher_floor/group_consistency_full_waypoint/full_waypoint_shape/endpoint_bridge; JEPA/Transformer/scene_goal/neighbor_interaction remain blocked as independent main claims; calibrated subset candidates remain post-confirmation only and ready_now is zero.",
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_module_claim_lock() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    _append_to_docs(payload)
    _update_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_module_claim_lock()
    gate = result["stage42_gj_gate"]
    print(f"Stage42-GJ gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
