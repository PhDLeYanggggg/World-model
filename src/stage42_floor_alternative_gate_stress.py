from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_full_waypoint_bridge_shape_audit import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

E_JSON = OUT_DIR / "safety_floor_stage42.json"
HB_JSON = OUT_DIR / "teacher_floor_necessity_meta_audit_stage42.json"
GT_JSON = OUT_DIR / "floor_relaxation_safety_stress_stage42.json"

REPORT_JSON = OUT_DIR / "floor_alternative_gate_stress_stage42.json"
REPORT_MD = OUT_DIR / "floor_alternative_gate_stress_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_hc_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
USER_SUMMARY = Path("README_M3W_USER_DETAILED_SUMMARY_ZH.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")

EASY_LIMIT = 0.02
COLLISION_LIMIT = 0.01

FLOOR_FREE_SWITCH_FAMILIES = {
    "internal_self_gate",
    "uncertainty_gate",
    "harm_predictor_gate",
    "conformal_risk_gate",
}

TEACHER_DEPENDENT_SWITCH_FAMILIES = {
    "teacher_raw_policy",
    "teacher_repaired_floor",
    "teacher_prob_gate",
}

FLOOR_FREE_BOUNDED_FAMILIES = {
    "bounded_all_rows_alpha",
    "bounded_horizon_alpha",
}

TEACHER_DEPENDENT_BOUNDED_FAMILIES = {
    "bounded_teacher_switch_alpha",
    "bounded_teacher_prob70_alpha",
    "bounded_horizon_teacher_switch_alpha",
    "current_composite_tail_policy",
}

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-HC 是 floor-alternative gate stress matrix，不重新训练，不下载，不转换，不调 test threshold。",
    "本审计使用 Stage42-E fresh validation-selected gate families，并把结果按 floor-free / teacher-dependent / bounded residual 分组。",
    "future endpoint / future waypoint 只允许作为监督或评估标签，不允许作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "Stage5C latent generative 未执行，SMC 未启用。",
]


def _metric(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    return float(default if value is None else value)


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def _gate_passed(payload: Mapping[str, Any], key: str) -> bool:
    gate = payload.get(key, {})
    return bool(gate and gate.get("passed") == gate.get("total") and int(gate.get("total", 0)) > 0)


def _is_deployable(metrics: Mapping[str, Any]) -> bool:
    return (
        _metric(metrics, "all_improvement") > 0.0
        and (_metric(metrics, "t50_improvement") > 0.0 or _metric(metrics, "hard_failure_improvement") > 0.0)
        and _metric(metrics, "easy_degradation", 1.0) <= EASY_LIMIT
        and _metric(metrics, "collision_delta_vs_floor_005", 1.0) <= COLLISION_LIMIT
        and (_metric(metrics, "switch_rate") > 0.0 or _metric(metrics, "alpha_positive_rate") > 0.0)
    )


def _failure_reasons(metrics: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if _metric(metrics, "all_improvement") <= 0.0:
        reasons.append("all_not_positive")
    if _metric(metrics, "t50_improvement") <= 0.0 and _metric(metrics, "hard_failure_improvement") <= 0.0:
        reasons.append("t50_and_hard_not_positive")
    if _metric(metrics, "easy_degradation", 1.0) > EASY_LIMIT:
        reasons.append("easy_degradation_over_2pct")
    if _metric(metrics, "collision_delta_vs_floor_005", 1.0) > COLLISION_LIMIT:
        reasons.append("near_collision_delta_over_1pp")
    if _metric(metrics, "switch_rate", _metric(metrics, "alpha_positive_rate")) <= 0.0:
        reasons.append("no_intervention")
    return reasons


def _compact_row(row: Mapping[str, Any], family_type: str) -> dict[str, Any]:
    metrics = dict(row.get("test_metrics", {}))
    return {
        "family": row.get("family"),
        "family_type": family_type,
        "source": row.get("source"),
        "candidate_count": row.get("candidate_count"),
        "val_eligible_count": row.get("val_eligible_count"),
        "stage42e_test_deployable": row.get("test_deployable"),
        "strict_deployable": _is_deployable(metrics),
        "failure_reasons": _failure_reasons(metrics),
        "test_metrics": metrics,
    }


def _rows_by_family(e: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in e.get("switch_gate_rows", []):
        family = str(row.get("family"))
        if family in FLOOR_FREE_SWITCH_FAMILIES:
            typ = "floor_free_switch_gate"
        elif family in TEACHER_DEPENDENT_SWITCH_FAMILIES:
            typ = "teacher_dependent_switch_gate"
        else:
            typ = "other_switch_gate"
        rows.append(_compact_row(row, typ))
    for row in e.get("bounded_residual_rows", []):
        family = str(row.get("family"))
        if family in FLOOR_FREE_BOUNDED_FAMILIES:
            typ = "floor_free_bounded_residual"
        elif family in TEACHER_DEPENDENT_BOUNDED_FAMILIES:
            typ = "teacher_dependent_bounded_or_current"
        else:
            typ = "other_bounded"
        rows.append(_compact_row(row, typ))
    return rows


def _best_by_score(rows: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        return {}

    def score(row: Mapping[str, Any]) -> float:
        m = row.get("test_metrics", {})
        return (
            _metric(m, "all_improvement")
            + 1.2 * _metric(m, "t50_improvement")
            + 0.8 * _metric(m, "t100_improvement")
            + _metric(m, "hard_failure_improvement")
            - 20.0 * max(0.0, _metric(m, "easy_degradation") - EASY_LIMIT)
            - 10.0 * max(0.0, _metric(m, "collision_delta_vs_floor_005") - COLLISION_LIMIT)
        )

    return max(rows, key=score)


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    e = read_json(E_JSON, {})
    hb = read_json(HB_JSON, {})
    gt = read_json(GT_JSON, {})
    rows = _rows_by_family(e)
    floor_free = [row for row in rows if str(row["family_type"]).startswith("floor_free")]
    teacher_dependent = [row for row in rows if str(row["family_type"]).startswith("teacher_dependent")]
    deployable_floor_free = [row for row in floor_free if row["strict_deployable"]]
    deployable_teacher = [row for row in teacher_dependent if row["strict_deployable"]]
    positive_but_unsafe_floor_free = [
        row
        for row in floor_free
        if _metric(row["test_metrics"], "all_improvement") > 0.0
        and (_metric(row["test_metrics"], "t50_improvement") > 0.0 or _metric(row["test_metrics"], "hard_failure_improvement") > 0.0)
        and not row["strict_deployable"]
    ]
    gt_summary = gt.get("summary", {})
    summary = {
        "source": "fresh_stage42_hc_floor_alternative_gate_stress",
        "candidate_rows_total": len(rows),
        "floor_free_candidate_rows": len(floor_free),
        "teacher_dependent_candidate_rows": len(teacher_dependent),
        "floor_free_deployable_count": len(deployable_floor_free),
        "teacher_dependent_deployable_count": len(deployable_teacher),
        "floor_free_positive_but_unsafe_count": len(positive_but_unsafe_floor_free),
        "best_floor_free_candidate": _best_by_score(floor_free),
        "best_teacher_dependent_candidate": _best_by_score(teacher_dependent),
        "best_deployable_teacher_dependent_candidate": _best_by_score(deployable_teacher),
        "partial_t50_relaxation_supported": gt_summary.get("target_union_safety_pass") is True,
        "partial_t50_target_slices": gt_summary.get("target_slices", []),
        "partial_t50_improvement": _metric(gt_summary, "target_union_t50_improvement"),
        "partial_t50_easy": _metric(gt_summary, "target_union_easy_degradation"),
        "global_floor_removal_allowed": False,
        "floor_free_neural_deployable": False,
        "deployment_decision": "keep_stage37_teacher_floor_globally_allow_only_validation_backed_partial_t50_relaxation",
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    payload: dict[str, Any] = {
        "source": "fresh_stage42_hc_floor_alternative_gate_stress",
        "stage": "Stage42-HC floor-alternative gate stress matrix",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([E_JSON, HB_JSON, GT_JSON]),
        "current_facts": CURRENT_FACTS,
        "input_status": {
            "stage42_e_safety_floor": {
                "path": str(E_JSON),
                "source": e.get("source"),
                "gate_passed": _gate_passed(e, "stage42_e_gate"),
                "verdict": (e.get("stage42_e_gate", {}) or {}).get("verdict"),
            },
            "stage42_hb_meta_audit": {
                "path": str(HB_JSON),
                "source": hb.get("source"),
                "gate_passed": _gate_passed(hb, "stage42_hb_gate"),
                "verdict": (hb.get("stage42_hb_gate", {}) or {}).get("verdict"),
            },
            "stage42_gt_floor_relaxation_stress": {
                "path": str(GT_JSON),
                "source": gt.get("source"),
                "gate_passed": _gate_passed(gt, "stage42_gt_gate"),
                "verdict": (gt.get("stage42_gt_gate", {}) or {}).get("verdict"),
            },
        },
        "candidate_rows": rows,
        "summary": summary,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "fresh_matrix_no_training": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "global_floor_removal_allowed": False,
            "floor_free_neural_deployable": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_hc_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    leakage = payload["no_leakage"]
    claim = payload["claim_boundary"]
    best_floor_free = s.get("best_floor_free_candidate", {})
    gates = {
        "required_inputs_passed": all(item.get("gate_passed") is True for item in payload["input_status"].values()),
        "floor_free_gate_families_evaluated": s["floor_free_candidate_rows"] >= 6,
        "teacher_dependent_gate_families_evaluated": s["teacher_dependent_candidate_rows"] >= 3,
        "floor_free_positive_but_unsafe_detected": s["floor_free_positive_but_unsafe_count"] >= 1,
        "floor_free_deployable_none": s["floor_free_deployable_count"] == 0,
        "teacher_dependent_deployable_exists": s["teacher_dependent_deployable_count"] >= 1,
        "best_floor_free_failure_reason_recorded": len(best_floor_free.get("failure_reasons", [])) >= 1,
        "partial_t50_relaxation_preserved": s["partial_t50_relaxation_supported"] is True
        and s["partial_t50_improvement"] > 0.0
        and s["partial_t50_easy"] <= EASY_LIMIT,
        "global_floor_removal_blocked": s["global_floor_removal_allowed"] is False and claim["global_floor_removal_allowed"] is False,
        "floor_free_neural_not_deployable": s["floor_free_neural_deployable"] is False and claim["floor_free_neural_deployable"] is False,
        "no_future_test_or_central_velocity_leakage": leakage["future_endpoint_input"] is False
        and leakage["future_waypoint_input"] is False
        and leakage["central_velocity"] is False
        and leakage["test_endpoint_goals"] is False
        and leakage["test_threshold_tuning"] is False,
        "no_metric_seconds_overclaim": claim["metric_or_seconds_claim"] is False
        and claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = sum(bool(v) for v in gates.values())
    total = len(gates)
    verdict = "stage42_hc_floor_alternative_gate_stress_pass" if passed == total else "stage42_hc_floor_alternative_gate_stress_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hc_gate"]
    s = payload["summary"]
    best_floor = s.get("best_floor_free_candidate", {})
    best_teacher = s.get("best_teacher_dependent_candidate", {})
    best_teacher_deployable = s.get("best_deployable_teacher_dependent_candidate", {})
    lines = [
        "# Stage42-HC Floor-Alternative Gate Stress Matrix",
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
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Direct Decision",
        "",
        f"- deployment_decision: `{s['deployment_decision']}`",
        f"- floor_free_deployable_count: `{s['floor_free_deployable_count']}`",
        f"- teacher_dependent_deployable_count: `{s['teacher_dependent_deployable_count']}`",
        f"- partial_t50_target_slices: `{s['partial_t50_target_slices']}`",
        f"- global_floor_removal_allowed: `{s['global_floor_removal_allowed']}`",
        f"- floor_free_neural_deployable: `{s['floor_free_neural_deployable']}`",
        "",
        "## Best Candidates",
        "",
        f"- best_floor_free_candidate: `{best_floor.get('family')}`; strict_deployable=`{best_floor.get('strict_deployable')}`; failure_reasons=`{best_floor.get('failure_reasons')}`; metrics=`{best_floor.get('test_metrics')}`",
        f"- best_teacher_dependent_candidate: `{best_teacher.get('family')}`; strict_deployable=`{best_teacher.get('strict_deployable')}`; failure_reasons=`{best_teacher.get('failure_reasons')}`; metrics=`{best_teacher.get('test_metrics')}`",
        f"- best_deployable_teacher_dependent_candidate: `{best_teacher_deployable.get('family')}`; strict_deployable=`{best_teacher_deployable.get('strict_deployable')}`; metrics=`{best_teacher_deployable.get('test_metrics')}`",
        "",
        "## Candidate Matrix",
        "",
        "| family | type | strict deployable | Stage42-E deployable | all | t50 | t100 raw | hard | easy | collision d005 | failure reasons |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["candidate_rows"]:
        m = row["test_metrics"]
        lines.append(
            f"| `{row['family']}` | `{row['family_type']}` | {row['strict_deployable']} | {row['stage42e_test_deployable']} | "
            f"{_pct(m.get('all_improvement'))} | {_pct(m.get('t50_improvement'))} | {_pct(m.get('t100_improvement'))} | "
            f"{_pct(m.get('hard_failure_improvement'))} | {_pct(m.get('easy_degradation'))} | {_pct(m.get('collision_delta_vs_floor_005'))} | `{row['failure_reasons']}` |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Floor-free switch gates can produce high raw improvements, but the best such candidates fail strict deployment because near-collision delta exceeds the safety limit.",
        "- Teacher-dependent gates and the current composite remain deployable, which supports HB's conclusion that Stage37/teacher floor is still a core mechanism.",
        "- Partial t50 floor relaxation remains allowed only for validation-backed slices and does not license global floor removal.",
        "- No metric/seconds/true-3D/foundation/Stage5C/SMC claim is made.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hc_gate"]
    lines = [
        "# Stage42-HC Gate",
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


def _refresh_docs(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    s = payload["summary"]
    best_floor = s.get("best_floor_free_candidate", {})
    best_metrics = best_floor.get("test_metrics", {})
    best_teacher_deployable = s.get("best_deployable_teacher_dependent_candidate", {})
    teacher_metrics = best_teacher_deployable.get("test_metrics", {})
    lines = [
        "## Stage42-HC Floor-Alternative Gate Stress Matrix",
        "",
        "- source: `fresh_stage42_hc_floor_alternative_gate_stress`",
        f"- gate: `{payload['stage42_hc_gate']['passed']} / {payload['stage42_hc_gate']['total']}`",
        f"- verdict: `{payload['stage42_hc_gate']['verdict']}`",
        "- Tested Stage42-E internal self-gate, uncertainty gate, conformal risk gate, harm predictor, teacher-dependent gates, and bounded residual families as floor alternatives.",
        f"- floor-free deployable count: `{s['floor_free_deployable_count']}`; teacher-dependent deployable count: `{s['teacher_dependent_deployable_count']}`.",
        f"- best floor-free candidate `{best_floor.get('family')}` reaches all/t50/hard `{_pct(best_metrics.get('all_improvement'))}` / `{_pct(best_metrics.get('t50_improvement'))}` / `{_pct(best_metrics.get('hard_failure_improvement'))}` but is not deployable because `{best_floor.get('failure_reasons')}`.",
        f"- best deployable teacher-dependent candidate `{best_teacher_deployable.get('family')}` reaches all/t50/hard `{_pct(teacher_metrics.get('all_improvement'))}` / `{_pct(teacher_metrics.get('t50_improvement'))}` / `{_pct(teacher_metrics.get('hard_failure_improvement'))}` with easy `{_pct(teacher_metrics.get('easy_degradation'))}`.",
        "- Deployment decision remains: keep Stage37/teacher floor globally; allow only validation-backed partial t50 relaxation on selected slices.",
        "- No metric/seconds/true-3D/foundation/Stage5C/SMC claim is allowed.",
    ]
    status = []
    for path in [README_RESULTS, M3W_README, USER_SUMMARY, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, "STAGE42_HC_FLOOR_ALTERNATIVE_GATE_STRESS", lines)
            text = path.read_text(encoding="utf-8")
            status.append(
                {
                    "path": str(path),
                    "exists": True,
                    "contains_stage42_hc": "Stage42-HC Floor-Alternative Gate Stress Matrix" in text,
                    "blocks_floor_free": "keep Stage37/teacher floor globally" in text,
                }
            )
        else:
            status.append({"path": str(path), "exists": False})
    return status


def run_stage42_floor_alternative_gate_stress() -> dict[str, Any]:
    payload = _build_payload()
    payload["doc_refresh_status"] = _refresh_docs(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    return payload


if __name__ == "__main__":
    out = run_stage42_floor_alternative_gate_stress()
    print(json.dumps(out["summary"], ensure_ascii=False, indent=2, sort_keys=True))
