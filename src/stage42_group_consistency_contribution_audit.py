from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
DY_JSON = OUT_DIR / "explicit_physical_consistency_checkpoint_stage42.json"
DZ_JSON = OUT_DIR / "ucy_supported_group_consistency_stage42.json"
EA_JSON = OUT_DIR / "dual_domain_group_consistency_statistics_stage42.json"
DP_JSON = OUT_DIR / "context_model_closure_stage42.json"

REPORT_JSON = OUT_DIR / "group_consistency_contribution_audit_stage42.json"
REPORT_MD = OUT_DIR / "group_consistency_contribution_audit_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ec_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-EC 是 contribution audit：fresh synthesis from DY/DZ/EA/DP，不重新训练，不调 threshold。",
    "future endpoints / waypoints 只作为 supervised/evaluation labels，不能作为 inference input。",
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
    "stage5c_executed": False,
    "smc_enabled": False,
    "ungated_full_waypoint_deployable": False,
    "global_primary_full_waypoint_replacement_claim_allowed": False,
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _ci_positive_safe(ci: Mapping[str, Any]) -> bool:
    return bool(
        ci["all"]["low"] > 0.0
        and ci["t50"]["low"] > 0.0
        and ci["hard_failure"]["low"] > 0.0
        and ci["easy_degradation"]["high"] <= 0.02
    )


def _stage_gate_passed(payload: Mapping[str, Any], key: str) -> bool:
    gate = payload.get(key, {})
    return bool(gate.get("passed") == gate.get("total") and gate.get("total", 0) > 0)


def _summary(
    dy: Mapping[str, Any],
    dz: Mapping[str, Any],
    ea: Mapping[str, Any],
    dp: Mapping[str, Any],
) -> dict[str, Any]:
    dy_s = dy["summary"]
    dz_s = dz["summary"]
    ea_s = ea["summary"]
    ea_ci = ea["bootstrap_ci"]
    dp_s = dp["summary"]
    return {
        "source": "fresh_synthesis_from_stage42_dy_dz_ea_dp",
        "supported_contributions": {
            "explicit_group_consistency_full_waypoint": {
                "status": "supported_source_level",
                "evidence": "DY source-level promotion plus DZ dual-domain repair plus EA bootstrap.",
                "all": dy_s["group_consistency_all"],
                "t50": dy_s["group_consistency_t50"],
                "t100_raw_frame_diagnostic": dy_s["group_consistency_t100_raw_frame_diagnostic"],
                "hard": dy_s["group_consistency_hard"],
                "easy": dy_s["group_consistency_easy"],
                "near005_base": dy_s["group_consistency_near005_base"],
                "near005_final": dy_s["group_consistency_near005_final"],
            },
            "dual_domain_raw_frame_support": {
                "status": "supported",
                "evidence": "DZ reports UCY and TrajNet positive-safe domains; EA adds 2000-bootstrap positive-safe CIs for both domains.",
                "positive_safe_domains": dz_s["positive_safe_domains"],
                "ucy_all": dz_s["ucy_all"],
                "ucy_t50": dz_s["ucy_t50"],
                "ucy_hard": dz_s["ucy_hard"],
                "trajnet_all": dz_s["trajnet_all"],
                "trajnet_t50": dz_s["trajnet_t50"],
                "trajnet_hard": dz_s["trajnet_hard"],
                "bootstrap_n": ea_s["bootstrap_n"],
                "ci_positive_safe_domains": ea_s["ci_positive_safe_domains"],
            },
            "baseline_family_rollout_context": {
                "status": "dominant_control_not_new_context_claim",
                "all": dp_s["baseline_family_metric"]["all_improvement"],
                "t50": dp_s["baseline_family_metric"]["t50_improvement"],
                "hard": dp_s["baseline_family_metric"]["hard_failure_improvement"],
                "interpretation": "The baseline-family rollout context remains the dominant first-stage control; group consistency should be framed as a source-level physical consistency repair over this protected family.",
            },
        },
        "blocked_or_negative_contributions": {
            "scalar_loss_family_primary": {
                "status": "blocked",
                "best_candidate": dy_s["best_loss_family_candidate"],
                "best_all": dy_s["best_loss_family_all"],
                "best_t50": dy_s["best_loss_family_t50"],
                "best_hard": dy_s["best_loss_family_hard"],
                "any_promotable": dy_s["loss_family_any_promotable_over_stage42_am"],
                "reason": "No scalar loss-family candidate beats Stage42-AM on the required all+hard promotion gate.",
            },
            "current_sequence_graph_residual_context": {
                "status": "closed_current_protocol",
                "positive_context_rows": len(dp_s["positive_context_rows"]),
                "best_delta_all": dp_s["best_delta_all"],
                "best_delta_t50": dp_s["best_delta_t50"],
                "best_delta_hard_failure": dp_s["best_delta_hard_failure"],
                "reason": dp_s["root_cause"],
            },
            "goal_scene_main_claim": {
                "status": "not_supported_under_current_protocols",
                "reason": "Current context closure and prior goal/scene expert runs do not support a main independent contribution claim.",
            },
            "neighbor_interaction_main_claim": {
                "status": "not_supported_under_current_protocols",
                "reason": "Current graph/interaction rows remain below baseline-family control.",
            },
            "ungated_global_full_waypoint_replacement": {
                "status": "blocked",
                "reason": "The supported group-consistency result is source-level protected evidence, not ungated or global primary replacement.",
            },
        },
        "statistical_evidence": {
            "global_all_ci_low": ea_ci["global"]["all"]["low"],
            "global_t50_ci_low": ea_ci["global"]["t50"]["low"],
            "global_hard_ci_low": ea_ci["global"]["hard_failure"]["low"],
            "global_easy_ci_high": ea_ci["global"]["easy_degradation"]["high"],
            "ucy_positive_safe_ci": _ci_positive_safe(ea_ci["by_domain"]["UCY"]),
            "trajnet_positive_safe_ci": _ci_positive_safe(ea_ci["by_domain"]["TrajNet"]),
            "near005_delta_high": ea["near_collision_ci"]["global"]["delta_final_minus_base"]["high"],
        },
        "paper_ready_claim": (
            "A paper can claim protected source-level group-consistency full-waypoint dynamics "
            "with UCY+TrajNet bootstrap-backed raw-frame evidence, while explicitly blocking "
            "scalar-loss, current sequence/graph residual context, ungated/global, metric/seconds, Stage5C, and SMC overclaims."
        ),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    supported = s["supported_contributions"]
    blocked = s["blocked_or_negative_contributions"]
    stats = s["statistical_evidence"]
    claim = payload["claim_boundary"]
    gates = {
        "dy_input_passed": payload["input_gates"]["dy"],
        "dz_input_passed": payload["input_gates"]["dz"],
        "ea_input_passed": payload["input_gates"]["ea"],
        "dp_input_passed": payload["input_gates"]["dp"],
        "group_consistency_supported": supported["explicit_group_consistency_full_waypoint"]["status"] == "supported_source_level",
        "dual_domain_bootstrap_supported": supported["dual_domain_raw_frame_support"]["ci_positive_safe_domains"] >= 2,
        "ucy_ci_positive_safe": stats["ucy_positive_safe_ci"] is True,
        "trajnet_ci_positive_safe": stats["trajnet_positive_safe_ci"] is True,
        "near_collision_repaired": stats["near005_delta_high"] <= 0.0,
        "scalar_loss_family_blocked": blocked["scalar_loss_family_primary"]["any_promotable"] is False,
        "context_residual_protocol_closed": blocked["current_sequence_graph_residual_context"]["positive_context_rows"] == 0,
        "goal_scene_overclaim_blocked": blocked["goal_scene_main_claim"]["status"].startswith("not_supported"),
        "neighbor_interaction_overclaim_blocked": blocked["neighbor_interaction_main_claim"]["status"].startswith("not_supported"),
        "ungated_global_replacement_blocked": blocked["ungated_global_full_waypoint_replacement"]["status"] == "blocked"
        and claim["ungated_full_waypoint_deployable"] is False
        and claim["global_primary_full_waypoint_replacement_claim_allowed"] is False,
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_ec_group_consistency_contribution_audit_pass" if passed == total else "stage42_ec_group_consistency_contribution_audit_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    group = s["supported_contributions"]["explicit_group_consistency_full_waypoint"]
    dual = s["supported_contributions"]["dual_domain_raw_frame_support"]
    blocked = s["blocked_or_negative_contributions"]
    stats = s["statistical_evidence"]
    return [
        "## Stage42-EC Group-Consistency Contribution Audit",
        "",
        "- source: `fresh_synthesis_from_stage42_dy_dz_ea_dp`",
        "- role: converts the latest positive and negative evidence into a contribution/claim matrix.",
        f"- gate: `{payload['stage42_ec_gate']['passed']} / {payload['stage42_ec_gate']['total']}`; verdict `{payload['stage42_ec_gate']['verdict']}`.",
        f"- supported contribution: explicit group-consistency full-waypoint source-level repair, all/t50/t100 raw/hard `{group['all']:.6f}` / `{group['t50']:.6f}` / `{group['t100_raw_frame_diagnostic']:.6f}` / `{group['hard']:.6f}`.",
        f"- dual-domain evidence: UCY all/t50/hard `{dual['ucy_all']:.6f}` / `{dual['ucy_t50']:.6f}` / `{dual['ucy_hard']:.6f}`; TrajNet all/t50/hard `{dual['trajnet_all']:.6f}` / `{dual['trajnet_t50']:.6f}` / `{dual['trajnet_hard']:.6f}`.",
        f"- bootstrap CI lows global all/t50/hard `{stats['global_all_ci_low']:.6f}` / `{stats['global_t50_ci_low']:.6f}` / `{stats['global_hard_ci_low']:.6f}`; easy high `{stats['global_easy_ci_high']:.6f}`.",
        f"- blocked contributions: scalar loss-family primary `{blocked['scalar_loss_family_primary']['status']}`, current sequence/graph residual context `{blocked['current_sequence_graph_residual_context']['status']}`, goal/scene main claim `{blocked['goal_scene_main_claim']['status']}`, neighbor/interaction main claim `{blocked['neighbor_interaction_main_claim']['status']}`.",
        "- claim boundary: supported as protected source-level raw-frame full-waypoint evidence only; no true-3D, foundation, metric/seconds, Stage5C, SMC, or ungated/global primary replacement claim.",
    ]


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-EC Group-Consistency Contribution Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_ec_gate']['passed']} / {payload['stage42_ec_gate']['total']}`",
        f"- verdict: `{payload['stage42_ec_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Supported Contributions",
        "",
        "| contribution | status | key evidence |",
        "| --- | --- | --- |",
    ]
    for name, row in s["supported_contributions"].items():
        evidence = row.get("evidence") or row.get("interpretation") or f"all={_pct(row.get('all'))}, t50={_pct(row.get('t50'))}, hard={_pct(row.get('hard'))}"
        lines.append(f"| `{name}` | `{row['status']}` | {evidence} |")
    lines.extend(
        [
            "",
            "## Blocked Or Negative Contributions",
            "",
            "| contribution | status | reason |",
            "| --- | --- | --- |",
        ]
    )
    for name, row in s["blocked_or_negative_contributions"].items():
        lines.append(f"| `{name}` | `{row['status']}` | {row['reason']} |")
    stats = s["statistical_evidence"]
    lines.extend(
        [
            "",
            "## Statistical Evidence",
            "",
            f"- global all/t50/hard CI lows: `{stats['global_all_ci_low']:.6f}` / `{stats['global_t50_ci_low']:.6f}` / `{stats['global_hard_ci_low']:.6f}`.",
            f"- global easy degradation CI high: `{stats['global_easy_ci_high']:.6f}`.",
            f"- UCY positive-safe CI: `{stats['ucy_positive_safe_ci']}`.",
            f"- TrajNet positive-safe CI: `{stats['trajnet_positive_safe_ci']}`.",
            f"- near@0.05 final-base delta high: `{stats['near005_delta_high']:.6f}`.",
            "",
            "## Paper-Ready Claim",
            "",
            s["paper_ready_claim"],
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
            *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_ec_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ec_gate"]
    return [
        "# Stage42-EC Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, GOAL_SUMMARY]:
        _replace_section(path, "STAGE42_EC_GROUP_CONSISTENCY_CONTRIBUTION_AUDIT", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-EC group-consistency contribution audit"
    state["current_verdict"] = payload["stage42_ec_gate"]["verdict"]
    state["stage42_ec_group_consistency_contribution_audit"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_ec_gate"]["verdict"],
        "gates": f"{payload['stage42_ec_gate']['passed']}/{payload['stage42_ec_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_group_consistency_contribution_audit(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    dy = read_json(DY_JSON, {})
    dz = read_json(DZ_JSON, {})
    ea = read_json(EA_JSON, {})
    dp = read_json(DP_JSON, {})
    payload: dict[str, Any] = {
        "source": "fresh_synthesis_from_stage42_dy_dz_ea_dp",
        "stage": "Stage42-EC",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([DY_JSON, DZ_JSON, EA_JSON, DP_JSON]),
        "current_facts": CURRENT_FACTS,
        "input_gates": {
            "dy": _stage_gate_passed(dy, "stage42_dy_gate"),
            "dz": _stage_gate_passed(dz, "stage42_dz_gate"),
            "ea": _stage_gate_passed(ea, "stage42_ea_gate"),
            "dp": _stage_gate_passed(dp, "stage42_dp_gate"),
        },
        "summary": _summary(dy, dz, ea, dp),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["stage42_ec_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_group_consistency_contribution_audit()
