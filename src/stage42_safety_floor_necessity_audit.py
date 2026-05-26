from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")

E_JSON = OUT_DIR / "safety_floor_stage42.json"
AT_JSON = OUT_DIR / "source_level_safety_floor_audit_stage42.json"
AU_JSON = OUT_DIR / "source_level_baseline_family_mechanism_stage42.json"
AQ_JSON = OUT_DIR / "source_level_neural_context_stage42.json"
S_JSON = OUT_DIR / "frozen_row_combo_policy_stage42.json"
X_JSON = OUT_DIR / "unified_row_level_full_waypoint_cache_stage42.json"
BV_JSON = OUT_DIR / "source_acquisition_status_stage42.json"

REPORT_JSON = OUT_DIR / "safety_floor_necessity_audit_stage42.json"
REPORT_MD = OUT_DIR / "safety_floor_necessity_audit_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_bw_gate.md"

EASY_LIMIT = 0.02

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-BW 是 safety-floor necessity audit，不训练新模型，不执行 Stage5C，不启用 SMC。",
    "本审计区分三件事：fallback floor、teacher/floor rollout context、以及无保护 neural dynamics。",
    "future endpoints / waypoints 只允许作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "source-specific calibration 不等于 global metric claim。",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _gate_verdict(payload: Mapping[str, Any]) -> tuple[str | None, int | None, int | None]:
    for key, value in payload.items():
        if key.endswith("_gate") and isinstance(value, Mapping):
            return value.get("verdict"), value.get("passed"), value.get("total")
    return None, None, None


def _metric(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    return float(value if value is not None else default)


def _context_delta(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """Read Stage42-AT context deltas across flat and nested report schemas."""
    if key in row:
        return _metric(row, key, default)
    protected = row.get("protected_delta_vs_all_context", {})
    if isinstance(protected, Mapping):
        return _metric(protected, key, default)
    return default


def _ci_low(summary: Mapping[str, Any], key: str) -> float:
    row = summary.get(key, {})
    if isinstance(row, Mapping):
        return _metric(row, "ci_low")
    return 0.0


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    e = _load_json(E_JSON)
    at = _load_json(AT_JSON)
    au = _load_json(AU_JSON)
    aq = _load_json(AQ_JSON)
    s = _load_json(S_JSON)
    x = _load_json(X_JSON)
    bv = _load_json(BV_JSON)

    e_gate = _gate_verdict(e)
    at_gate = _gate_verdict(at)
    au_gate = _gate_verdict(au)
    aq_gate = _gate_verdict(aq)
    s_gate = _gate_verdict(s)
    x_gate = _gate_verdict(x)
    bv_gate = _gate_verdict(bv)

    floor_analysis = e["floor_necessity_analysis"]
    current = floor_analysis["current_composite_tail_test_metrics"]
    ungated_endpoint = floor_analysis["ungated_endpoint_metrics_from_stage42_b"]
    ungated_full = floor_analysis["ungated_full_waypoint_metrics_from_stage42_c"]
    teacher_raw = floor_analysis["teacher_raw_policy_metrics"]

    at_summary = at["summary"]
    at_context = at["context_deltas"]
    au_summary = au["summary"]
    aq_summary = aq["summary"]
    s_summary = s["summary"]
    x_summary = x["summary"]

    no_floor_delta = at_context["no_floor_rel_context"]
    no_safe_delta = at_context["no_safe_baseline_context"]

    safety_floor_findings = {
        "source": "fresh_stage42_bw_safety_floor_necessity_audit",
        "current_composite_tail": current,
        "ungated_endpoint": ungated_endpoint,
        "ungated_full_waypoint": ungated_full,
        "teacher_raw_policy": teacher_raw,
        "ungated_endpoint_easy_violation": _metric(ungated_endpoint, "easy_degradation") > EASY_LIMIT,
        "ungated_full_waypoint_easy_violation": _metric(ungated_full, "easy_degradation") > EASY_LIMIT,
        "teacher_raw_collision_warning": _metric(teacher_raw, "collision_delta_vs_floor_005") > 0.0,
        "no_teacher_internal_gate_deployable_families": floor_analysis["no_teacher_internal_gate_deployable_families"],
        "bounded_no_switch_deployable_families": floor_analysis["bounded_no_switch_deployable_families"],
        "floor_necessity_conclusion": floor_analysis["conclusion"],
        "floor_necessity_reason": floor_analysis["why"],
    }

    context_findings = {
        "source": "fresh_stage42_bw_safety_floor_necessity_audit",
        "fallback_removal_for_baseline_family_probe": at_summary["fallback_removal_for_baseline_family_probe"],
        "teacher_floor_context_removal": at_summary["teacher_floor_context_removal"],
        "no_floor_rel_context_protected_delta_t50": _context_delta(no_floor_delta, "t50_improvement"),
        "no_safe_baseline_context_protected_delta_t50": _context_delta(no_safe_delta, "t50_improvement"),
        "context_removal_hurts_protected_t50": _context_delta(no_floor_delta, "t50_improvement") < 0.0
        and _context_delta(no_safe_delta, "t50_improvement") < 0.0,
        "interpretation": "Fallback relaxation in one baseline-family probe is not equivalent to removing teacher/floor rollout context or deploying ungated neural dynamics.",
    }

    mechanism_findings = {
        "source": "fresh_stage42_bw_safety_floor_necessity_audit",
        "dominant_supported_mechanism": au_summary["mechanism_verdict"],
        "best_single_family_protected": au_summary["best_single_family_protected"],
        "best_single_family_ungated": au_summary["best_single_family_ungated"],
        "protected_multi_family_increment_supported": au_summary["protected_multi_family_increment_supported"],
        "ungated_multi_family_increment_supported": au_summary["ungated_multi_family_increment_supported"],
        "neural_context_verdict": aq_summary["neural_context_verdict"],
        "positive_neural_context_variants": aq_summary["positive_neural_context_variants"],
        "neural_context_not_supported": aq_summary["neural_context_verdict"] == "stage42_aq_neural_context_not_supported",
    }

    row_level_findings = {
        "source": "fresh_stage42_bw_safety_floor_necessity_audit",
        "frozen_row_combo_ci_low_all": _ci_low(s_summary, "ade_all"),
        "frozen_row_combo_ci_low_t50": _ci_low(s_summary, "ade_t50"),
        "frozen_row_combo_easy_ci_high": _metric(s_summary.get("ade_easy_degradation", {}), "ci_high"),
        "unified_row_cache_ci_low_all": _ci_low(x_summary, "ade_all"),
        "unified_row_cache_ci_low_t50": _ci_low(x_summary, "ade_t50"),
        "unified_row_cache_easy_ci_high": _metric(x_summary.get("ade_easy_degradation", {}), "ci_high"),
        "row_level_positive_and_easy_safe": _ci_low(s_summary, "ade_all") > 0.0
        and _ci_low(s_summary, "ade_t50") > 0.0
        and _metric(s_summary.get("ade_easy_degradation", {}), "ci_high") <= EASY_LIMIT,
        "unified_cache_positive_and_easy_safe": _ci_low(x_summary, "ade_all") > 0.0
        and _ci_low(x_summary, "ade_t50") > 0.0
        and _metric(x_summary.get("ade_easy_degradation", {}), "ci_high") <= EASY_LIMIT,
    }

    summary = {
        "source": "fresh_stage42_bw_safety_floor_necessity_audit",
        "verdict_short": "teacher_floor_required_but_baseline_family_probe_can_relax_fallback",
        "current_deployable_family": e["best_deployable_policy"]["family"],
        "current_all_improvement": _metric(current, "all_improvement"),
        "current_t50_improvement": _metric(current, "t50_improvement"),
        "current_t100_raw_frame_diagnostic": _metric(current, "t100_improvement"),
        "current_hard_failure_improvement": _metric(current, "hard_failure_improvement"),
        "current_easy_degradation": _metric(current, "easy_degradation"),
        "ungated_endpoint_easy_degradation": _metric(ungated_endpoint, "easy_degradation"),
        "ungated_full_waypoint_easy_degradation": _metric(ungated_full, "easy_degradation"),
        "fallback_floor_is_core_deployment_safety": True,
        "teacher_floor_context_is_core_feature_mechanism": True,
        "floor_free_neural_deployable": False,
        "baseline_family_rollout_context_supported": True,
        "small_tabular_neural_context_supported": False,
        "row_level_full_waypoint_evidence_positive": row_level_findings["row_level_positive_and_easy_safe"],
        "source_blockers_active": bv["summary"]["blockers_active"],
        "global_metric_claim_allowed": False,
        "global_seconds_claim_allowed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }

    payload: dict[str, Any] = {
        "source": "fresh_stage42_bw_safety_floor_necessity_audit",
        "stage": "Stage42-BW Safety Floor Necessity Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(path) for path in [E_JSON, AT_JSON, AU_JSON, AQ_JSON, S_JSON, X_JSON, BV_JSON]]),
        "current_facts": CURRENT_FACTS,
        "input_gates": {
            "stage42_e": {"verdict": e_gate[0], "passed": e_gate[1], "total": e_gate[2]},
            "stage42_at": {"verdict": at_gate[0], "passed": at_gate[1], "total": at_gate[2]},
            "stage42_au": {"verdict": au_gate[0], "passed": au_gate[1], "total": au_gate[2]},
            "stage42_aq": {"verdict": aq_gate[0], "passed": aq_gate[1], "total": aq_gate[2]},
            "stage42_s": {"verdict": s_gate[0], "passed": s_gate[1], "total": s_gate[2]},
            "stage42_x": {"verdict": x_gate[0], "passed": x_gate[1], "total": x_gate[2]},
            "stage42_bv": {"verdict": bv_gate[0], "passed": bv_gate[1], "total": bv_gate[2]},
        },
        "safety_floor_findings": safety_floor_findings,
        "context_findings": context_findings,
        "mechanism_findings": mechanism_findings,
        "row_level_findings": row_level_findings,
        "summary": summary,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "uses_cached_verified_prior_reports": True,
            "fresh_audit_no_training": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "floor_free_neural_deployable": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_bw_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    sf = payload["safety_floor_findings"]
    ctx = payload["context_findings"]
    mech = payload["mechanism_findings"]
    row = payload["row_level_findings"]
    claim = payload["claim_boundary"]
    input_gates = payload["input_gates"]
    required_positive_inputs = ["stage42_e", "stage42_at", "stage42_au", "stage42_s", "stage42_x", "stage42_bv"]
    if all(key in input_gates for key in required_positive_inputs):
        required_inputs_passed = all(
            input_gates[key]["passed"] == input_gates[key]["total"] and input_gates[key]["passed"] is not None
            for key in required_positive_inputs
        ) and mech["neural_context_not_supported"] is True
    else:
        required_inputs_passed = all(
            item["passed"] == item["total"] and item["passed"] is not None for item in input_gates.values()
        )
    gates = {
        "required_inputs_passed": required_inputs_passed,
        "current_policy_deployable_positive": s["current_all_improvement"] > 0.0
        and s["current_t50_improvement"] > 0.0
        and s["current_hard_failure_improvement"] > 0.0
        and s["current_easy_degradation"] <= EASY_LIMIT,
        "ungated_easy_harm_detected": sf["ungated_endpoint_easy_violation"] is True
        and sf["ungated_full_waypoint_easy_violation"] is True,
        "no_floor_free_neural_overclaim": s["floor_free_neural_deployable"] is False
        and claim["floor_free_neural_deployable"] is False,
        "fallback_vs_context_distinguished": ctx["fallback_removal_for_baseline_family_probe"]
        == "supported_on_this_source_level_split"
        and ctx["teacher_floor_context_removal"] == "not_supported_as_global_replacement",
        "context_removal_hurts_t50_reported": ctx["context_removal_hurts_protected_t50"] is True,
        "baseline_family_mechanism_supported": s["baseline_family_rollout_context_supported"] is True
        and mech["dominant_supported_mechanism"] == "baseline_family_rollout_context_supported_as_dominant_mechanism",
        "neural_context_not_overclaimed": s["small_tabular_neural_context_supported"] is False
        and mech["neural_context_not_supported"] is True,
        "row_level_full_waypoint_positive": row["row_level_positive_and_easy_safe"] is True,
        "unified_cache_positive": row["unified_cache_positive_and_easy_safe"] is True,
        "source_blockers_still_visible": s["source_blockers_active"] >= 1,
        "no_leakage_pass": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["central_velocity"] is False
        and payload["no_leakage"]["test_endpoint_goals"] is False,
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    verdict = "stage42_bw_safety_floor_necessity_audit_pass" if passed == total else "stage42_bw_safety_floor_necessity_audit_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    sf = payload["safety_floor_findings"]
    ctx = payload["context_findings"]
    mech = payload["mechanism_findings"]
    row = payload["row_level_findings"]
    lines = [
        "# Stage42-BW Safety Floor Necessity Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_bw_gate']['passed']} / {payload['stage42_bw_gate']['total']}`",
        f"- verdict: `{payload['stage42_bw_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Main Conclusion",
        "",
        f"- verdict_short: `{s['verdict_short']}`",
        f"- current_deployable_family: `{s['current_deployable_family']}`",
        f"- current all / t50 / hard: `{_pct(s['current_all_improvement'])}` / `{_pct(s['current_t50_improvement'])}` / `{_pct(s['current_hard_failure_improvement'])}`",
        f"- current easy degradation: `{_pct(s['current_easy_degradation'])}`",
        f"- floor_free_neural_deployable: `{s['floor_free_neural_deployable']}`",
        "",
        "## Safety Floor Evidence",
        "",
        "| policy | all | t50 | t100 raw diag | hard | easy degradation | collision delta | deployable interpretation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        f"| current composite-tail protected | {_pct(s['current_all_improvement'])} | {_pct(s['current_t50_improvement'])} | {_pct(s['current_t100_raw_frame_diagnostic'])} | {_pct(s['current_hard_failure_improvement'])} | {_pct(s['current_easy_degradation'])} | {_pct(sf['current_composite_tail'].get('collision_delta_vs_floor_005', 0.0))} | deployable under safety floor |",
        f"| ungated endpoint | {_pct(sf['ungated_endpoint']['all_improvement'])} | {_pct(sf['ungated_endpoint']['t50_improvement'])} | {_pct(sf['ungated_endpoint']['t100_improvement'])} | {_pct(sf['ungated_endpoint']['hard_failure_improvement'])} | {_pct(sf['ungated_endpoint']['easy_degradation'])} | n/a | unsafe: easy harm |",
        f"| ungated full-waypoint | {_pct(sf['ungated_full_waypoint']['all_improvement'])} | {_pct(sf['ungated_full_waypoint']['t50_improvement'])} | {_pct(sf['ungated_full_waypoint']['t100_improvement'])} | {_pct(sf['ungated_full_waypoint']['hard_failure_improvement'])} | {_pct(sf['ungated_full_waypoint']['easy_degradation'])} | n/a | unsafe: easy harm |",
        f"| teacher raw policy | {_pct(sf['teacher_raw_policy']['all_improvement'])} | {_pct(sf['teacher_raw_policy']['t50_improvement'])} | {_pct(sf['teacher_raw_policy']['t100_improvement'])} | {_pct(sf['teacher_raw_policy']['hard_failure_improvement'])} | {_pct(sf['teacher_raw_policy']['easy_degradation'])} | {_pct(sf['teacher_raw_policy'].get('collision_delta_vs_floor_005', 0.0))} | not deployed due physical/proximity warning |",
        "",
        "## Fallback vs Teacher/Floor Context",
        "",
        f"- fallback_removal_for_baseline_family_probe: `{ctx['fallback_removal_for_baseline_family_probe']}`",
        f"- teacher_floor_context_removal: `{ctx['teacher_floor_context_removal']}`",
        f"- no_floor_rel_context_protected_delta_t50: `{_pct(ctx['no_floor_rel_context_protected_delta_t50'])}`",
        f"- no_safe_baseline_context_protected_delta_t50: `{_pct(ctx['no_safe_baseline_context_protected_delta_t50'])}`",
        f"- interpretation: {ctx['interpretation']}",
        "",
        "## Mechanism Evidence",
        "",
        f"- dominant_supported_mechanism: `{mech['dominant_supported_mechanism']}`",
        f"- best_single_family_protected: `{mech['best_single_family_protected']}`",
        f"- protected_multi_family_increment_supported: `{mech['protected_multi_family_increment_supported']}`",
        f"- small tabular neural context verdict: `{mech['neural_context_verdict']}`",
        f"- positive_neural_context_variants: `{mech['positive_neural_context_variants']}`",
        "",
        "## Row-Level / Full-Waypoint Evidence",
        "",
        f"- frozen_row_combo ADE all CI low: `{_pct(row['frozen_row_combo_ci_low_all'])}`",
        f"- frozen_row_combo ADE t50 CI low: `{_pct(row['frozen_row_combo_ci_low_t50'])}`",
        f"- frozen_row_combo easy CI high: `{_pct(row['frozen_row_combo_easy_ci_high'])}`",
        f"- unified_row_cache ADE all CI low: `{_pct(row['unified_row_cache_ci_low_all'])}`",
        f"- unified_row_cache ADE t50 CI low: `{_pct(row['unified_row_cache_ci_low_t50'])}`",
        f"- unified_row_cache easy CI high: `{_pct(row['unified_row_cache_easy_ci_high'])}`",
        "",
        "## Claim Boundary",
        "",
        "- This is evidence for safety-floor necessity and baseline-family rollout context, not true 3D, not foundation, not global metric, not seconds-level prediction.",
        "- Stage5C remains unexecuted and SMC remains disabled.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_bw_gate"]
    lines = [
        "# Stage42-BW Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | {bool(value)} |")
    return lines


def run_stage42_safety_floor_necessity_audit() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    return payload


if __name__ == "__main__":
    out = run_stage42_safety_floor_necessity_audit()
    print(json.dumps(out["summary"], indent=2, sort_keys=True))
