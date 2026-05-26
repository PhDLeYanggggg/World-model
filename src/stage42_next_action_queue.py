from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import read_json, write_json, write_md
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "next_action_queue_stage42.json"
REPORT_MD = OUT_DIR / "next_action_queue_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_da_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_README = Path("README_M3W_GOAL_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

EVIDENCE_FILES = {
    "paper_freeze": OUT_DIR / "paper_freeze_candidate_manifest_stage42.json",
    "worktree_caveat": OUT_DIR / "worktree_caveat_classifier_stage42.json",
    "a_journal_gap": OUT_DIR / "a_journal_gap_stage42.md",
    "context_forensics": OUT_DIR / "context_contribution_forensics_stage42.json",
    "goal_scene": OUT_DIR / "goal_scene_gated_expert_stage42.json",
    "neighbor_interaction": OUT_DIR / "neighbor_interaction_gated_expert_stage42.json",
    "bridge_shape": OUT_DIR / "common_validation_bridge_shape_composer_stage42.json",
    "proximity_guard": OUT_DIR / "proximity_aware_composer_guard_stage42.json",
    "proximity_ablation": OUT_DIR / "proximity_guard_ablation_stage42.json",
    "source_terms": OUT_DIR / "source_terms_validation_stage42.json",
    "source_time_geometry": OUT_DIR / "source_time_geometry_calibration_stage42.json",
    "t100_gap": OUT_DIR / "t100_data_gap_audit_stage42.json",
}

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-DA 是 next-action evidence queue，不重新训练，不调 threshold，不把计划当完成。",
    "所有下一步动作必须继续区分 fresh_run / cached_verified / not_run。",
    "future endpoints / future waypoints 只能作为 supervised/evaluation labels，不能作为 inference input。",
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


def _exists_map() -> dict[str, bool]:
    return {key: path.exists() for key, path in EVIDENCE_FILES.items()}


def _safe_get(path_key: str, default: Any) -> Any:
    path = EVIDENCE_FILES.get(path_key)
    if path is None:
        return default
    return read_json(path, default) if path.suffix == ".json" else default


def _current_evidence_summary() -> dict[str, Any]:
    freeze = _safe_get("paper_freeze", {})
    caveat = _safe_get("worktree_caveat", {})
    context = _safe_get("context_forensics", {})
    goal = _safe_get("goal_scene", {})
    neighbor = _safe_get("neighbor_interaction", {})
    bridge = _safe_get("bridge_shape", {})
    proximity = _safe_get("proximity_guard", {})
    t100_gap = _safe_get("t100_gap", {})
    return {
        "paper_freeze_status": freeze.get("freeze_status", {}).get("freeze_status", "unknown"),
        "paper_freeze_final_immutable_release": freeze.get("freeze_status", {}).get(
            "final_immutable_release", False
        ),
        "stage42_dirty_files": caveat.get("summary", {}).get("stage42_dirty_files", "unknown"),
        "stage42_substantive_dirty_files": caveat.get("summary", {}).get(
            "stage42_substantive_dirty_files", "unknown"
        ),
        "dominant_mechanism": context.get("conclusion", {}).get(
            "dominant_mechanism", "baseline_family_rollout_context"
        ),
        "goal_scene_rescue_success": goal.get("summary", {}).get("goal_scene_rescue_success", False),
        "neighbor_interaction_rescue_success": neighbor.get("summary", {}).get(
            "neighbor_interaction_rescue_success", False
        ),
        "common_validation_composer_all_improvement": bridge.get("test_eval", {})
        .get("metric_vs_endpoint_ade", {})
        .get("all_improvement", None),
        "proximity_guard_all_improvement": proximity.get("test_eval", {})
        .get("metric_vs_endpoint_ade", {})
        .get("all_improvement", None),
        "proximity_guard_t50_improvement": proximity.get("test_eval", {})
        .get("metric_vs_endpoint_ade", {})
        .get("t50_improvement", None),
        "global_t100_claim_ready": t100_gap.get("summary", {}).get("global_t100_claim_ready", False),
    }


def _actions() -> list[dict[str, Any]]:
    return [
        {
            "id": "DA-1",
            "title": "Close legal/source support for ETH_UCY and TrajNet t100/t50 calibration",
            "priority": 100,
            "status": "not_run_next_action",
            "why_now": "Global t100 and restricted metric/seconds claims are still blocked by independent source support and terms confirmation.",
            "evidence": [
                "outputs/stage42_long_research/t100_data_gap_audit_stage42.md",
                "outputs/stage42_long_research/source_terms_validation_stage42.md",
                "outputs/stage42_long_research/source_time_geometry_calibration_stage42.md",
            ],
            "next_commands": [
                ".venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py",
                ".venv-pytorch/bin/python run_stage42_source_time_geometry_calibration.py",
            ],
            "requires_user_or_external_state": True,
            "blocked_claim_until_done": "global_or_restricted metric/seconds and global t100 deployable claim",
            "success_gate": "official/terms-safe conversion + no-leakage + source-CV positive/easy-safe on ETH_UCY and TrajNet.",
        },
        {
            "id": "DA-2",
            "title": "Train a stronger source-compatible sequence/graph context model beyond baseline-family rollout",
            "priority": 92,
            "status": "not_run_next_action",
            "why_now": "Current context forensics says baseline-family rollout context dominates; history/goal/neighbor/interaction are not independent main drivers yet.",
            "evidence": [
                "outputs/stage42_long_research/context_contribution_forensics_stage42.md",
                "outputs/stage42_long_research/goal_scene_gated_expert_stage42.md",
                "outputs/stage42_long_research/neighbor_interaction_gated_expert_stage42.md",
            ],
            "next_commands": [
                ".venv-pytorch/bin/python run_stage42_source_level_sequence_context.py",
                ".venv-pytorch/bin/python run_stage42_source_level_graph_context.py",
            ],
            "requires_user_or_external_state": False,
            "blocked_claim_until_done": "scene/goal/neighbor/interaction as independent main contribution",
            "success_gate": "retrained context model beats baseline-family control on all/t50/hard with easy degradation <=2%.",
        },
        {
            "id": "DA-3",
            "title": "Promote protected full-waypoint from bridge/composer to learned all-agent sequence dynamics",
            "priority": 88,
            "status": "not_run_next_action",
            "why_now": "Common-validation composer is positive, but endpoint-linear bridge remains the stronger all-ADE floor and ungated full-waypoint is unsafe.",
            "evidence": [
                "outputs/stage42_long_research/full_waypoint_bridge_shape_audit_stage42.md",
                "outputs/stage42_long_research/common_validation_bridge_shape_composer_stage42.md",
                "outputs/stage42_long_research/proximity_guard_ablation_stage42.md",
            ],
            "next_commands": [
                ".venv-pytorch/bin/python run_stage42_full_waypoint_dynamics.py",
                ".venv-pytorch/bin/python run_stage42_common_validation_bridge_shape_composer.py",
            ],
            "requires_user_or_external_state": False,
            "blocked_claim_until_done": "ungated or independently learned full-waypoint world dynamics",
            "success_gate": "learned sequence dynamics improves endpoint-linear/proximity-guard composer on all/t50/hard without proximity/easy regression.",
        },
        {
            "id": "DA-4",
            "title": "Convert paper-freeze candidate into reviewer-replay package",
            "priority": 76,
            "status": "not_run_next_action",
            "why_now": "Stage42-CZ has a clean hash manifest, but a reviewer still needs a minimal replay sequence and immutable archive boundary.",
            "evidence": [
                "outputs/stage42_long_research/paper_freeze_candidate_manifest_stage42.md",
                "outputs/stage42_long_research/evidence_provenance_stage42.md",
                "outputs/stage42_long_research/proximity_guard_batch_replay_stage42.md",
            ],
            "next_commands": [
                ".venv-pytorch/bin/python run_stage42_paper_freeze_candidate_manifest.py",
                ".venv-pytorch/bin/python -m pytest tests/test_stage42_paper_freeze_candidate_manifest.py tests/test_stage42_evidence_provenance_verifier.py",
            ],
            "requires_user_or_external_state": False,
            "blocked_claim_until_done": "paper-ready reproducibility package beyond internal manifest",
            "success_gate": "manifest, replay, and provenance can be regenerated without tracked artifact churn.",
        },
        {
            "id": "DA-5",
            "title": "Audit deployment variants as safety-sensitive vs accuracy-priority policies",
            "priority": 72,
            "status": "not_run_next_action",
            "why_now": "No-guard composer has higher ADE, proximity guard is safer. Paper and deployment must not mix these claims.",
            "evidence": [
                "outputs/stage42_long_research/proximity_aware_composer_guard_stage42.md",
                "outputs/stage42_long_research/proximity_guard_ablation_stage42.md",
            ],
            "next_commands": [
                ".venv-pytorch/bin/python run_stage42_proximity_guard_ablation.py",
                ".venv-pytorch/bin/python run_stage42_runtime_replay_paper_refresh.py",
            ],
            "requires_user_or_external_state": False,
            "blocked_claim_until_done": "single deployment policy claim with explicit risk/accuracy tradeoff",
            "success_gate": "deployment card separates safety-sensitive deployable from diagnostic accuracy-priority variant.",
        },
    ]


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    claim = payload["claim_boundary"]
    actions = payload["next_actions"]
    evidence = payload["evidence_files"]
    gates = {
        "evidence_files_checked": len(evidence) >= 10,
        "paper_freeze_status_recorded": bool(payload["current_evidence"].get("paper_freeze_status")),
        "at_least_five_next_actions": len(actions) >= 5,
        "actions_have_priorities": all(isinstance(row.get("priority"), int) for row in actions),
        "actions_have_success_gates": all(bool(row.get("success_gate")) for row in actions),
        "actions_reference_evidence": all(bool(row.get("evidence")) for row in actions),
        "user_action_blockers_explicit": any(row.get("requires_user_or_external_state") for row in actions),
        "no_not_run_marked_complete": all(row.get("status") != "complete" for row in actions),
        "goal_scene_negative_evidence_preserved": payload["current_evidence"].get("goal_scene_rescue_success") is False,
        "neighbor_negative_evidence_preserved": payload["current_evidence"].get(
            "neighbor_interaction_rescue_success"
        )
        is False,
        "no_metric_seconds_overclaim": claim["metric_or_seconds_claim"] is False,
        "true_3d_overclaim_blocked": claim["true_3d"] is False,
        "foundation_overclaim_blocked": claim["foundation_world_model"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_da_next_action_queue_pass" if passed == total else "stage42_da_next_action_queue_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-DA Next-Action Evidence Queue",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- gate: `{payload['stage42_da_gate']['passed']} / {payload['stage42_da_gate']['total']}`",
        f"- verdict: `{payload['stage42_da_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
    ]
    lines.extend(f"- {fact}" for fact in CURRENT_FACTS)
    lines.extend(
        [
            "",
            "## Current Evidence Snapshot",
            "",
        ]
    )
    for key, value in payload["current_evidence"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Evidence Files Checked", ""])
    for key, exists in payload["evidence_files"].items():
        lines.append(f"- {key}: `{exists}`")
    lines.extend(["", "## Prioritized Next Actions", ""])
    for row in payload["next_actions"]:
        lines.extend(
            [
                f"### {row['id']} - {row['title']}",
                "",
                f"- priority: `{row['priority']}`",
                f"- status: `{row['status']}`",
                f"- why_now: {row['why_now']}",
                f"- requires_user_or_external_state: `{row['requires_user_or_external_state']}`",
                f"- blocked_claim_until_done: `{row['blocked_claim_until_done']}`",
                f"- success_gate: {row['success_gate']}",
                "- evidence:",
            ]
        )
        lines.extend(f"  - `{item}`" for item in row["evidence"])
        lines.append("- next_commands:")
        lines.extend(f"  - `{item}`" for item in row["next_commands"])
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- Stage42-DA does not count any next action as complete.",
            "- It converts current evidence and blockers into an ordered experiment queue.",
            "- The strongest current deployable claim remains the Stage42-CQ/CV proximity-aware guarded composer under Stage37/teacher floor.",
            "- The next substantive research risk is proving independent neural/scene/interaction/full-waypoint contribution beyond baseline-family rollout context.",
        ]
    )
    return lines


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_da_gate"]
    top = payload["next_actions"][0]
    return [
        "## Stage42-DA Next-Action Evidence Queue",
        "",
        "- source: `fresh_synthesis_from_cached_verified_stage42_artifacts`",
        "- role: convert current Stage42 paper gaps into prioritized executable next actions.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- top priority: `{top['id']} {top['title']}`.",
        "- user/external blockers remain explicit; no not_run item is counted complete.",
        "- Current deployable claim remains protected dataset-local/raw-frame 2.5D; no true 3D, no foundation, no metric/seconds-level, no Stage5C, no SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, GOAL_README]:
        _replace_section(path, "STAGE42_DA_NEXT_ACTION_QUEUE", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-DA next-action evidence queue"
    state["current_verdict"] = payload["stage42_da_gate"]["verdict"]
    state["stage42_da_next_action_queue"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_da_gate"]["verdict"],
        "gates": f"{payload['stage42_da_gate']['passed']}/{payload['stage42_da_gate']['total']}",
        "top_priority": payload["next_actions"][0],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_next_action_queue(*, refresh_readmes: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source": "fresh_synthesis_from_cached_verified_stage42_artifacts",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "evidence_files": _exists_map(),
        "current_evidence": _current_evidence_summary(),
        "next_actions": sorted(_actions(), key=lambda row: row["priority"], reverse=True),
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_da_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(
        GATE_MD,
        [
            "# Stage42-DA Gate",
            "",
            f"- gate: `{payload['stage42_da_gate']['passed']} / {payload['stage42_da_gate']['total']}`",
            f"- verdict: `{payload['stage42_da_gate']['verdict']}`",
            "",
            "## Gates",
            "",
            *[
                f"- {key}: `{value}`"
                for key, value in payload["stage42_da_gate"]["gates"].items()
            ],
        ],
    )
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_next_action_queue()
