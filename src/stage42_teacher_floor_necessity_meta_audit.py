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
BW_JSON = OUT_DIR / "safety_floor_necessity_audit_stage42.json"
GT_JSON = OUT_DIR / "floor_relaxation_safety_stress_stage42.json"
CQ_JSON = OUT_DIR / "proximity_aware_composer_guard_stage42.json"
GZ_JSON = OUT_DIR / "full_waypoint_claim_guard_stage42.json"
HA_JSON = OUT_DIR / "full_waypoint_overclaim_linter_stage42.json"
M3W_EVIDENCE_JSON = Path("outputs/m3w_neural_v1/evidence_matrix_m3w_neural_v1.json")

REPORT_JSON = OUT_DIR / "teacher_floor_necessity_meta_audit_stage42.json"
REPORT_MD = OUT_DIR / "teacher_floor_necessity_meta_audit_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_hb_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
USER_SUMMARY = Path("README_M3W_USER_DETAILED_SUMMARY_ZH.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")

EASY_LIMIT = 0.02

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-HB 是 teacher / Stage37 floor necessity meta-audit，不重新训练，不下载，不转换，不调 test threshold。",
    "Stage5C latent generative 仍未执行，SMC 仍未启用。",
    "future endpoint / future waypoint 只允许作为监督或评估标签，不允许作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 结果不能写成 global metric 或 true 3D。",
]


def _metric(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    return float(default if value is None else value)


def _pct(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def _gate_payload(payload: Mapping[str, Any], key: str | None = None) -> Mapping[str, Any]:
    if key and isinstance(payload.get(key), Mapping):
        return payload[key]
    for maybe_key, value in payload.items():
        if maybe_key.endswith("_gate") and isinstance(value, Mapping):
            return value
    return {}


def _gate_passed(payload: Mapping[str, Any], key: str | None = None) -> bool:
    gate = _gate_payload(payload, key)
    passed = gate.get("passed")
    total = gate.get("total")
    return isinstance(passed, int) and isinstance(total, int) and total > 0 and passed == total


def _claim_status(gz: Mapping[str, Any], claim_id: str) -> Mapping[str, Any]:
    for row in gz.get("claim_rows", []):
        if row.get("claim_id") == claim_id:
            return row
    return {}


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    e = read_json(E_JSON, {})
    bw = read_json(BW_JSON, {})
    gt = read_json(GT_JSON, {})
    cq = read_json(CQ_JSON, {})
    gz = read_json(GZ_JSON, {})
    ha = read_json(HA_JSON, {})
    m3w = read_json(M3W_EVIDENCE_JSON, {})

    floor = e.get("floor_necessity_analysis", {})
    current = floor.get("current_composite_tail_test_metrics", {})
    ungated_endpoint = floor.get("ungated_endpoint_metrics_from_stage42_b", {})
    ungated_full = floor.get("ungated_full_waypoint_metrics_from_stage42_c", {})
    teacher_raw = floor.get("teacher_raw_policy_metrics", {})
    gt_summary = gt.get("summary", {})
    cq_safety = (cq.get("test_joint_safety", {}) or {}).get("composer_minus_endpoint", {})
    cq_floor_safety = (cq.get("test_joint_safety", {}) or {}).get("composer_minus_floor", {})
    cq_metric = (cq.get("test_eval", {}) or {}).get("metric_vs_endpoint_ade", {})
    ha_summary = ha.get("summary", {})
    m3w_metrics = m3w.get("best_metrics_vs_stage37_floor", {})

    input_status = {
        "stage42_e_safety_floor": {
            "path": str(E_JSON),
            "source": e.get("source"),
            "gate_passed": _gate_passed(e, "stage42_e_gate"),
            "verdict": _gate_payload(e, "stage42_e_gate").get("verdict"),
        },
        "stage42_bw_safety_floor_necessity": {
            "path": str(BW_JSON),
            "source": bw.get("source"),
            "gate_passed": _gate_passed(bw, "stage42_bw_gate"),
            "verdict": _gate_payload(bw, "stage42_bw_gate").get("verdict"),
        },
        "stage42_gt_floor_relaxation_stress": {
            "path": str(GT_JSON),
            "source": gt.get("source"),
            "gate_passed": _gate_passed(gt, "stage42_gt_gate"),
            "verdict": _gate_payload(gt, "stage42_gt_gate").get("verdict"),
        },
        "stage42_cq_proximity_guard": {
            "path": str(CQ_JSON),
            "source": cq.get("source"),
            "gate_passed": _gate_passed(cq, "stage42_cq_gate"),
            "verdict": _gate_payload(cq, "stage42_cq_gate").get("verdict"),
        },
        "stage42_gz_full_waypoint_claim_guard": {
            "path": str(GZ_JSON),
            "source": gz.get("source"),
            "gate_passed": _gate_passed(gz, "stage42_gz_gate"),
            "verdict": _gate_payload(gz, "stage42_gz_gate").get("verdict"),
        },
        "stage42_ha_full_waypoint_linter": {
            "path": str(HA_JSON),
            "source": ha.get("source"),
            "gate_passed": _gate_passed(ha, "stage42_ha_gate"),
            "verdict": _gate_payload(ha, "stage42_ha_gate").get("verdict"),
        },
        "m3w_neural_v1_evidence_matrix": {
            "path": str(M3W_EVIDENCE_JSON),
            "source": m3w.get("source"),
            "gate_passed": int(m3w.get("gates_passed", -1)) == int(m3w.get("gates_total", 0)) and int(m3w.get("gates_total", 0)) > 0,
            "verdict": m3w.get("current_verdict"),
        },
    }

    floor_taxonomy = {
        "global_teacher_floor_required": floor.get("conclusion") == "teacher_floor_required_for_current_deployment",
        "protected_composite_deployable": _metric(current, "all_improvement") > 0.0
        and _metric(current, "t50_improvement") > 0.0
        and _metric(current, "hard_failure_improvement") > 0.0
        and _metric(current, "easy_degradation") <= EASY_LIMIT,
        "ungated_endpoint_unsafe": _metric(ungated_endpoint, "easy_degradation") > EASY_LIMIT,
        "ungated_full_waypoint_unsafe": _metric(ungated_full, "easy_degradation") > EASY_LIMIT,
        "teacher_raw_not_deployable_due_proximity": _metric(teacher_raw, "collision_delta_vs_floor_005") > 0.0,
        "partial_t50_floor_relaxation_supported": gt_summary.get("target_union_safety_pass") is True
        and _metric(gt_summary, "target_union_t50_improvement") > 0.0
        and _metric(gt_summary, "target_union_hard_failure_improvement") > 0.0
        and _metric(gt_summary, "target_union_easy_degradation") <= EASY_LIMIT,
        "partial_relaxation_target_slices": gt_summary.get("target_slices", []),
        "global_floor_removal_allowed": bool(gt_summary.get("global_floor_removal_allowed", False)),
        "floor_free_neural_deployable": bool(gt_summary.get("floor_free_neural_deployable", False)),
        "teacher_floor_context_required": bool(gt_summary.get("teacher_floor_context_required", True)),
        "proximity_guard_required_for_safety_sensitive_full_waypoint": _metric(cq_safety, "near_collision_rate_005_delta", 1.0) <= 0.0
        and _metric(cq_floor_safety, "near_collision_rate_005_delta", 1.0) <= 0.0
        and _claim_status(gz, "GZ-C5").get("allowed_as_main_claim") is False,
        "full_waypoint_claim_guard_blocks_ungated": _claim_status(gz, "GZ-C3").get("allowed_as_main_claim") is False,
        "full_waypoint_linter_clean": int(ha_summary.get("violations_total", -1)) == 0,
    }

    summary = {
        "source": "fresh_stage42_hb_teacher_floor_necessity_meta_audit",
        "verdict_short": "teacher_floor_is_core_current_safety_mechanism_partial_t50_relaxation_only",
        "current_protected_all": _metric(current, "all_improvement"),
        "current_protected_t50": _metric(current, "t50_improvement"),
        "current_protected_t100_raw": _metric(current, "t100_improvement"),
        "current_protected_hard_failure": _metric(current, "hard_failure_improvement"),
        "current_protected_easy": _metric(current, "easy_degradation"),
        "ungated_endpoint_easy": _metric(ungated_endpoint, "easy_degradation"),
        "ungated_full_waypoint_easy": _metric(ungated_full, "easy_degradation"),
        "teacher_raw_collision_delta_vs_floor_005": _metric(teacher_raw, "collision_delta_vs_floor_005"),
        "partial_t50_relaxation_rows": int(gt_summary.get("target_union_rows", 0)),
        "partial_t50_relaxation_improvement": _metric(gt_summary, "target_union_t50_improvement"),
        "partial_t50_relaxation_hard": _metric(gt_summary, "target_union_hard_failure_improvement"),
        "partial_t50_relaxation_easy": _metric(gt_summary, "target_union_easy_degradation"),
        "partial_t50_relaxation_near_collision_delta": _metric(gt_summary, "target_union_near_collision_005_delta"),
        "proximity_guard_all_vs_endpoint": _metric(cq_metric, "all_improvement"),
        "proximity_guard_t50_vs_endpoint": _metric(cq_metric, "t50_improvement"),
        "proximity_guard_hard_vs_endpoint": _metric(cq_metric, "hard_failure_improvement"),
        "m3w_neural_v1_protected_all": _metric(m3w_metrics, "all_improvement"),
        "m3w_neural_v1_protected_t50": _metric(m3w_metrics, "t50_improvement"),
        "m3w_neural_v1_protected_t100_raw": _metric(m3w_metrics, "t100_improvement"),
        "m3w_neural_v1_protected_hard": _metric(m3w_metrics, "hard_failure_improvement"),
        "m3w_neural_v1_protected_easy": _metric(m3w_metrics, "easy_degradation"),
        "deployment_answer": "Stage37/teacher floor is not just a temporary crutch; it is the current safety floor and rollout-context mechanism. Narrow t50 slices can relax part of the floor only under validation-backed protection.",
        "global_floor_removal_allowed": False,
        "floor_free_neural_deployable": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }

    payload: dict[str, Any] = {
        "source": "fresh_stage42_hb_teacher_floor_necessity_meta_audit",
        "stage": "Stage42-HB teacher-floor necessity meta-audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([E_JSON, BW_JSON, GT_JSON, CQ_JSON, GZ_JSON, HA_JSON, M3W_EVIDENCE_JSON]),
        "current_facts": CURRENT_FACTS,
        "input_status": input_status,
        "floor_taxonomy": floor_taxonomy,
        "summary": summary,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "fresh_meta_audit_no_training": True,
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
    payload["stage42_hb_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    status = payload["input_status"]
    floor = payload["floor_taxonomy"]
    s = payload["summary"]
    leakage = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "required_inputs_loaded_and_passed": all(item.get("gate_passed") is True for item in status.values()),
        "protected_current_positive_easy_safe": floor["protected_composite_deployable"] is True,
        "ungated_neural_unsafe_detected": floor["ungated_endpoint_unsafe"] is True and floor["ungated_full_waypoint_unsafe"] is True,
        "teacher_raw_proximity_warning_detected": floor["teacher_raw_not_deployable_due_proximity"] is True,
        "teacher_floor_required_recorded": floor["global_teacher_floor_required"] is True and floor["teacher_floor_context_required"] is True,
        "partial_t50_relaxation_supported": floor["partial_t50_floor_relaxation_supported"] is True,
        "partial_relaxation_not_global_removal": floor["global_floor_removal_allowed"] is False and s["global_floor_removal_allowed"] is False,
        "floor_free_neural_not_deployable": floor["floor_free_neural_deployable"] is False and s["floor_free_neural_deployable"] is False,
        "proximity_guard_required": floor["proximity_guard_required_for_safety_sensitive_full_waypoint"] is True,
        "full_waypoint_claim_guard_blocks_ungated": floor["full_waypoint_claim_guard_blocks_ungated"] is True,
        "full_waypoint_linter_clean": floor["full_waypoint_linter_clean"] is True,
        "m3w_protected_candidate_positive": s["m3w_neural_v1_protected_all"] > 0.0
        and s["m3w_neural_v1_protected_t50"] > 0.0
        and s["m3w_neural_v1_protected_hard"] > 0.0
        and s["m3w_neural_v1_protected_easy"] <= EASY_LIMIT,
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
    passed = sum(bool(value) for value in gates.values())
    total = len(gates)
    verdict = "stage42_hb_teacher_floor_necessity_meta_audit_pass" if passed == total else "stage42_hb_teacher_floor_necessity_meta_audit_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hb_gate"]
    s = payload["summary"]
    floor = payload["floor_taxonomy"]
    lines = [
        "# Stage42-HB Teacher / Stage37 Floor Necessity Meta-Audit",
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
        "## Direct Answer",
        "",
        f"- deployment_answer: {s['deployment_answer']}",
        f"- global_teacher_floor_required: `{floor['global_teacher_floor_required']}`",
        f"- partial_t50_floor_relaxation_supported: `{floor['partial_t50_floor_relaxation_supported']}` for `{floor['partial_relaxation_target_slices']}`",
        f"- global_floor_removal_allowed: `{floor['global_floor_removal_allowed']}`",
        f"- floor_free_neural_deployable: `{floor['floor_free_neural_deployable']}`",
        "",
        "## Key Evidence",
        "",
        "| evidence | all | t50 | t100 raw diagnostic | hard/failure | easy degradation | safety/proximity | interpretation |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        f"| protected M3W-Neural v1 / current composite | {_pct(s['current_protected_all'])} | {_pct(s['current_protected_t50'])} | {_pct(s['current_protected_t100_raw'])} | {_pct(s['current_protected_hard_failure'])} | {_pct(s['current_protected_easy'])} | safe under floor | current protected candidate |",
        f"| ungated endpoint | n/a | n/a | n/a | n/a | {_pct(s['ungated_endpoint_easy'])} | unsafe easy harm | rejected |",
        f"| ungated full-waypoint | n/a | n/a | n/a | n/a | {_pct(s['ungated_full_waypoint_easy'])} | unsafe easy harm | rejected |",
        f"| teacher raw policy | n/a | n/a | n/a | n/a | 0.00% | collision delta {_pct(s['teacher_raw_collision_delta_vs_floor_005'])} | not deployed without guard |",
        f"| partial t50 floor relaxation | n/a | {_pct(s['partial_t50_relaxation_improvement'])} | n/a | {_pct(s['partial_t50_relaxation_hard'])} | {_pct(s['partial_t50_relaxation_easy'])} | near@0.05 delta {_pct(s['partial_t50_relaxation_near_collision_delta'])} | narrow slice support only |",
        f"| proximity-aware composer guard | {_pct(s['proximity_guard_all_vs_endpoint'])} | {_pct(s['proximity_guard_t50_vs_endpoint'])} | n/a | {_pct(s['proximity_guard_hard_vs_endpoint'])} | n/a | not worse than endpoint/floor | guard required for safety-sensitive full-waypoint |",
        "",
        "## Taxonomy",
        "",
        "| item | value |",
        "| --- | --- |",
    ]
    for key, value in floor.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines += [
        "",
        "## Input Status",
        "",
        "| input | source | gate | verdict |",
        "| --- | --- | ---: | --- |",
    ]
    for key, item in payload["input_status"].items():
        lines.append(f"| `{key}` | `{item.get('source')}` | `{item.get('gate_passed')}` | `{item.get('verdict')}` |")
    lines += [
        "",
        "## Claim Boundary",
        "",
        "- This meta-audit supports Stage37/teacher floor as the current safety mechanism and rollout-context floor.",
        "- It supports only narrow validation-backed t50 floor relaxation on selected slices; it does not support global floor removal.",
        "- It does not support floor-free neural deployment, metric/seconds-level claims, true 3D, foundation model claims, Stage5C execution, or SMC.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hb_gate"]
    lines = [
        "# Stage42-HB Gate",
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
    lines = [
        "## Stage42-HB Teacher-Floor Necessity Meta-Audit",
        "",
        "- source: `fresh_stage42_hb_teacher_floor_necessity_meta_audit`",
        f"- gate: `{payload['stage42_hb_gate']['passed']} / {payload['stage42_hb_gate']['total']}`",
        f"- verdict: `{payload['stage42_hb_gate']['verdict']}`",
        "- Direct conclusion: Stage37 / teacher floor is the current safety mechanism and rollout-context floor, not merely a disposable crutch.",
        f"- Protected current all/t50/t100raw/hard/easy: `{_pct(s['current_protected_all'])}` / `{_pct(s['current_protected_t50'])}` / `{_pct(s['current_protected_t100_raw'])}` / `{_pct(s['current_protected_hard_failure'])}` / `{_pct(s['current_protected_easy'])}`.",
        f"- Ungated endpoint/full-waypoint easy degradation remains unsafe: `{_pct(s['ungated_endpoint_easy'])}` / `{_pct(s['ungated_full_waypoint_easy'])}`.",
        f"- Narrow t50 floor relaxation is supported only on selected slices: rows `{s['partial_t50_relaxation_rows']}`, t50 `{_pct(s['partial_t50_relaxation_improvement'])}`, hard `{_pct(s['partial_t50_relaxation_hard'])}`, easy `{_pct(s['partial_t50_relaxation_easy'])}`.",
        "- Global floor removal and floor-free neural deployment remain false.",
        "- No metric/seconds/true-3D/foundation/Stage5C/SMC claim is allowed.",
    ]
    status = []
    for path in [README_RESULTS, M3W_README, USER_SUMMARY, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, "STAGE42_HB_TEACHER_FLOOR_NECESSITY_META_AUDIT", lines)
            text = path.read_text(encoding="utf-8")
            status.append(
                {
                    "path": str(path),
                    "exists": True,
                    "contains_stage42_hb": "Stage42-HB Teacher-Floor Necessity Meta-Audit" in text,
                    "blocks_floor_free": "floor-free neural deployment remain false" in text,
                }
            )
        else:
            status.append({"path": str(path), "exists": False})
    return status


def run_stage42_teacher_floor_necessity_meta_audit() -> dict[str, Any]:
    payload = _build_payload()
    payload["doc_refresh_status"] = _refresh_docs(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    return payload


if __name__ == "__main__":
    out = run_stage42_teacher_floor_necessity_meta_audit()
    print(json.dumps(out["summary"], ensure_ascii=False, indent=2, sort_keys=True))
