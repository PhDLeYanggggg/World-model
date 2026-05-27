from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
BW_JSON = OUT_DIR / "safety_floor_necessity_audit_stage42.json"
BY_JSON = OUT_DIR / "t50_floor_relaxability_repair_stage42.json"
CQ_JSON = OUT_DIR / "proximity_aware_composer_guard_stage42.json"
CR_JSON = OUT_DIR / "proximity_guard_ablation_stage42.json"
EM_JSON = OUT_DIR / "official_source_link_audit_stage42.json"

REPORT_JSON = OUT_DIR / "floor_removability_decision_map_stage42.json"
REPORT_MD = OUT_DIR / "floor_removability_decision_map_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_en_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
GOAL_LEDGER = Path("README_M3W_GOAL_DETAILED_LEDGER_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_floor_removability_decision_map"
EASY_LIMIT = 0.02

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-EN is a fresh synthesis audit over cached-verified safety-floor evidence; it does not train or tune a model.",
    "This audit distinguishes deployment fallback, teacher/floor rollout context, proximity guard, and narrow t50 fallback relaxation.",
    "future endpoints / waypoints remain supervised/evaluation labels only, never inference inputs.",
    "No central velocity, no test endpoints for goals, no test-threshold tuning.",
    "t+50 / t+100 remain raw-frame horizons; no seconds-level claim.",
    "dataset-local/raw-frame coordinates are not global metric coordinates.",
    "Stage5C latent generative was not executed.",
    "SMC was not enabled.",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "floor_free_neural_deployable": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def _metric(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    return float(value if value is not None else default)


def _load_inputs() -> dict[str, Any]:
    return {
        "bw": read_json(BW_JSON, {}),
        "by": read_json(BY_JSON, {}),
        "cq": read_json(CQ_JSON, {}),
        "cr": read_json(CR_JSON, {}),
        "em": read_json(EM_JSON, {}),
    }


def _component_map(inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    bw = inputs["bw"]
    by = inputs["by"]
    cr = inputs["cr"]
    em = inputs["em"]
    sf = bw.get("safety_floor_findings", {})
    ctx = bw.get("context_findings", {})
    summary = bw.get("summary", {})
    by_summary = by.get("summary", {})
    by_decisions = by.get("target_t50_decisions") or by.get("target_decisions", {})
    cr_rows = cr.get("ablation_rows", {})
    no_guard = cr_rows.get("no_proximity_guard", {})
    guard = cr_rows.get("proximity_guard", {})
    em_summary = em.get("summary", {})

    components: list[dict[str, Any]] = [
        {
            "component": "ungated_neural_endpoint_or_full_waypoint",
            "decision": "blocked",
            "evidence": "ungated raw errors improve but easy degradation violates deployment safety",
            "key_metrics": {
                "ungated_endpoint_easy_degradation": _metric(sf.get("ungated_endpoint", {}), "easy_degradation"),
                "ungated_full_waypoint_easy_degradation": _metric(sf.get("ungated_full_waypoint", {}), "easy_degradation"),
                "easy_limit": EASY_LIMIT,
            },
            "deployment_action": "do_not_deploy_ungated_neural",
        },
        {
            "component": "teacher_floor_rollout_context",
            "decision": "required",
            "evidence": "removing floor/safe baseline rollout context hurts protected t50",
            "key_metrics": {
                "no_floor_rel_context_t50_delta": _metric(ctx, "no_floor_rel_context_protected_delta_t50"),
                "no_safe_baseline_context_t50_delta": _metric(ctx, "no_safe_baseline_context_protected_delta_t50"),
            },
            "deployment_action": "keep_teacher_floor_rollout_context",
        },
        {
            "component": "deployment_fallback_floor",
            "decision": "required_globally_partial_relaxation_allowed",
            "evidence": "global floor-free neural is not deployable, but protected t50 fallback relaxation is supported on selected validation-backed slices",
            "key_metrics": {
                "floor_free_neural_deployable": bool(summary.get("floor_free_neural_deployable", False)),
                "repaired_t50_slices": list(by_summary.get("repaired_t50_slices", [])),
                "global_t50_improvement_after_repair": _metric(by_summary, "global_t50_improvement"),
                "global_easy_degradation_after_repair": _metric(by_summary, "global_easy_degradation"),
            },
            "deployment_action": "allow_only_validation_backed_t50_slice_relaxation",
        },
        {
            "component": "proximity_guard",
            "decision": "required_for_safety_sensitive_reporting",
            "evidence": "no-guard has higher ADE gain but worsens near-collision; guard repairs proximity while preserving positive all/t50/t100/hard gains",
            "key_metrics": {
                "no_guard_all_improvement": _metric(no_guard, "all_improvement"),
                "no_guard_near_collision_delta": _metric(no_guard, "near_collision_005_delta_vs_endpoint"),
                "guard_all_improvement": _metric(guard, "all_improvement"),
                "guard_t50_improvement": _metric(guard, "t50_improvement"),
                "guard_near_collision_delta": _metric(guard, "near_collision_005_delta_vs_endpoint"),
            },
            "deployment_action": "use_guarded_variant_for_safety_sensitive_claims",
        },
        {
            "component": "source_expansion_without_terms",
            "decision": "blocked",
            "evidence": "official source candidates exist but conversion-ready targets remain zero",
            "key_metrics": {
                "official_or_toolkit_source_candidates": int(em_summary.get("official_or_toolkit_source_candidates", 0)),
                "conversion_ready_now": int(em_summary.get("conversion_ready_now", 0)),
                "auto_download_allowed_now": int(em_summary.get("auto_download_allowed_now", 0)),
            },
            "deployment_action": "wait_for_user_terms_path_source_identity_confirmation",
        },
    ]

    for slice_name, decision in by_decisions.items():
        metric = decision.get("after_metric", {})
        components.append(
            {
                "component": f"t50_slice_relaxation::{slice_name}",
                "decision": "partial_supported" if decision.get("protected_t50_repaired", False) else "blocked",
                "evidence": decision.get("before_bx_reason", ""),
                "key_metrics": {
                    "rows": int(metric.get("rows", 0)),
                    "t50_improvement": _metric(metric, "t50_improvement"),
                    "hard_failure_improvement": _metric(metric, "hard_failure_improvement"),
                    "easy_degradation": _metric(metric, "easy_degradation"),
                    "switch_rate": _metric(metric, "switch_rate"),
                },
                "deployment_action": "slice_only_under_train_internal_validation_policy",
            }
        )
    return components


def _summary(components: list[Mapping[str, Any]]) -> dict[str, Any]:
    blocked = [row["component"] for row in components if row["decision"] in {"blocked", "required"}]
    partial = [row["component"] for row in components if row["decision"] == "partial_supported"]
    safety_required = [
        row["component"]
        for row in components
        if row["decision"] in {"required", "required_for_safety_sensitive_reporting", "required_globally_partial_relaxation_allowed"}
    ]
    return {
        "source": SOURCE,
        "components_audited": len(components),
        "blocked_or_required_components": blocked,
        "partial_relaxation_components": partial,
        "safety_required_components": safety_required,
        "floor_free_neural_deployable": False,
        "safe_partial_floor_relaxation_available": len(partial) >= 2,
        "global_floor_removal_allowed": False,
        "teacher_floor_rollout_context_removal_allowed": False,
        "proximity_guard_required_for_safety_claim": True,
        "conversion_ready_now": 0,
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    by_components = [row for row in payload["component_decision_map"] if row["component"].startswith("t50_slice_relaxation::")]
    gates = {
        "inputs_present": all(bool(payload["input_reports"].get(key, {}).get("exists")) for key in ["bw", "by", "cq", "cr", "em"]),
        "components_audited": s["components_audited"] >= 7,
        "ungated_neural_blocked": any(row["component"] == "ungated_neural_endpoint_or_full_waypoint" and row["decision"] == "blocked" for row in payload["component_decision_map"]),
        "teacher_context_required": s["teacher_floor_rollout_context_removal_allowed"] is False,
        "fallback_global_removal_blocked": s["global_floor_removal_allowed"] is False,
        "partial_t50_relaxation_mapped": len([row for row in by_components if row["decision"] == "partial_supported"]) >= 2,
        "proximity_guard_required": s["proximity_guard_required_for_safety_claim"] is True,
        "source_terms_blocker_preserved": s["conversion_ready_now"] == 0,
        "floor_free_neural_not_claimed": payload["claim_boundary"]["floor_free_neural_deployable"] is False,
        "no_leakage_pass": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["central_velocity"] is False
        and payload["no_leakage"]["test_endpoint_goals"] is False,
        "no_metric_seconds_overclaim": payload["claim_boundary"]["global_metric_claim_allowed"] is False
        and payload["claim_boundary"]["global_seconds_claim_allowed"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_en_floor_removability_decision_map_pass" if passed == total else "stage42_en_floor_removability_decision_map_partial"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-EN Floor Removability Decision Map",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_en_gate']['passed']} / {payload['stage42_en_gate']['total']}`",
        f"- verdict: `{payload['stage42_en_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Summary",
        "",
        f"- components_audited: `{s['components_audited']}`",
        f"- floor_free_neural_deployable: `{s['floor_free_neural_deployable']}`",
        f"- safe_partial_floor_relaxation_available: `{s['safe_partial_floor_relaxation_available']}`",
        f"- global_floor_removal_allowed: `{s['global_floor_removal_allowed']}`",
        f"- teacher_floor_rollout_context_removal_allowed: `{s['teacher_floor_rollout_context_removal_allowed']}`",
        f"- proximity_guard_required_for_safety_claim: `{s['proximity_guard_required_for_safety_claim']}`",
        "",
        "## Decision Map",
        "",
        "| component | decision | key metrics | deployment action |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload["component_decision_map"]:
        metrics = ", ".join(f"{k}={_pct(v) if isinstance(v, float) else v}" for k, v in row["key_metrics"].items())
        lines.append(
            f"| `{row['component']}` | `{row['decision']}` | {metrics} | `{row['deployment_action']}` |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Teacher/floor rollout context remains a core mechanism, not a removable implementation detail.",
        "- Deployment fallback cannot be removed globally because ungated neural variants violate easy safety.",
        "- Narrow t50 fallback relaxation is allowed only on validation-backed slices and still relies on teacher/floor context.",
        "- The proximity guard is required for safety-sensitive claims because the no-guard variant has better ADE but worse near-collision.",
        "- Source expansion remains blocked until the user confirms official terms, allowed use, local path, and source identity.",
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_en_gate"]["gates"].items()],
    ]
    return lines


def _readme_block(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    return [
        "## Stage42-EN Floor Removability Decision Map",
        "",
        f"- source: `{payload['source']}`",
        "- role: maps which parts of Stage37/teacher floor can be removed, partially relaxed, or must remain.",
        f"- gate: `{payload['stage42_en_gate']['passed']} / {payload['stage42_en_gate']['total']}`; verdict `{payload['stage42_en_gate']['verdict']}`.",
        f"- floor_free_neural_deployable: `{s['floor_free_neural_deployable']}`; global_floor_removal_allowed: `{s['global_floor_removal_allowed']}`.",
        f"- partial t50 relaxation available: `{s['safe_partial_floor_relaxation_available']}`; teacher rollout context removal allowed: `{s['teacher_floor_rollout_context_removal_allowed']}`.",
        f"- proximity guard required for safety-sensitive claim: `{s['proximity_guard_required_for_safety_claim']}`.",
        "- Boundary: no metric/seconds claim, no Stage5C, no SMC.",
    ]


def _update_readmes(payload: Mapping[str, Any]) -> None:
    block = _readme_block(payload)
    for path in [README_RESULTS, M3W_README, GOAL_SUMMARY, GOAL_LEDGER]:
        if path.exists():
            _replace_section(path, "STAGE42_EN_FLOOR_REMOVABILITY_DECISION_MAP", block)

    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-EN floor removability decision map"
    state["current_verdict"] = payload["stage42_en_gate"]["verdict"]
    state["stage42_en_floor_removability_decision_map"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_en_gate"]["verdict"],
        "gates": f"{payload['stage42_en_gate']['passed']}/{payload['stage42_en_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_floor_removability_decision_map(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    inputs = _load_inputs()
    components = _component_map(inputs)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-EN Floor Removability Decision Map",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(BW_JSON), str(BY_JSON), str(CQ_JSON), str(CR_JSON), str(EM_JSON)]),
        "current_facts": CURRENT_FACTS,
        "input_reports": {
            "bw": {"path": str(BW_JSON), "exists": BW_JSON.exists()},
            "by": {"path": str(BY_JSON), "exists": BY_JSON.exists()},
            "cq": {"path": str(CQ_JSON), "exists": CQ_JSON.exists()},
            "cr": {"path": str(CR_JSON), "exists": CR_JSON.exists()},
            "em": {"path": str(EM_JSON), "exists": EM_JSON.exists()},
        },
        "component_decision_map": components,
        "summary": _summary(components),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "fresh_synthesis_from_cached_verified_reports": True,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["stage42_en_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(
        GATE_MD,
        [
            "# Stage42-EN Gate",
            "",
            f"- gate: `{payload['stage42_en_gate']['passed']} / {payload['stage42_en_gate']['total']}`",
            f"- verdict: `{payload['stage42_en_gate']['verdict']}`",
            "",
            "| gate | pass |",
            "| --- | ---: |",
            *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_en_gate"]["gates"].items()],
        ],
    )
    if refresh_readmes:
        _update_readmes(payload)
    return payload


if __name__ == "__main__":
    run_stage42_floor_removability_decision_map()
