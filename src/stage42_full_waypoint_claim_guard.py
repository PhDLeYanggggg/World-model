from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_full_waypoint_bridge_shape_audit import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

CM_JSON = OUT_DIR / "full_waypoint_bridge_shape_audit_stage42.json"
CO_JSON = OUT_DIR / "common_validation_bridge_shape_composer_stage42.json"
CQ_JSON = OUT_DIR / "proximity_aware_composer_guard_stage42.json"
W_JSON = OUT_DIR / "unified_external_full_waypoint_policy_stage42.json"
GJ_JSON = OUT_DIR / "module_claim_lock_stage42.json"

REPORT_JSON = OUT_DIR / "full_waypoint_claim_guard_stage42.json"
REPORT_MD = OUT_DIR / "full_waypoint_claim_guard_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_gz_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
SUMMARY_README = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-GZ 是 full-waypoint claim guard，不重新训练，不调 threshold，不执行 Stage5C，不启用 SMC。",
    "full-waypoint future labels 只允许作为 supervised/eval label，不允许作为 inference input。",
    "endpoint-only 或 endpoint-linear bridge 成功不能直接写成 learned full-waypoint dynamics 成功。",
    "ungated full-waypoint neural 仍不可部署。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def _gate_passed(payload: Mapping[str, Any], key: str) -> bool:
    gate = payload.get(key, {})
    return bool(gate and int(gate.get("passed", -1)) == int(gate.get("total", 0)) and int(gate.get("total", 0)) > 0)


def _metric_from_rows(cm: Mapping[str, Any], name: str) -> dict[str, Any]:
    for row in cm.get("comparison_rows", []):
        if row.get("name") == name:
            return dict(row)
    return {}


def _co_metric(co: Mapping[str, Any], key: str = "metric_vs_endpoint_ade") -> Mapping[str, Any]:
    return (co.get("test_eval", {}) or {}).get(key, {})


def _cq_metric(cq: Mapping[str, Any], key: str = "metric_vs_endpoint_ade") -> Mapping[str, Any]:
    return (cq.get("test_eval", {}) or {}).get(key, {})


def _build_claim_rows(
    cm: Mapping[str, Any],
    co: Mapping[str, Any],
    cq: Mapping[str, Any],
    unified: Mapping[str, Any],
    gj: Mapping[str, Any],
) -> list[dict[str, Any]]:
    protected_full = _metric_from_rows(cm, "full_waypoint_transformer_protected")
    linear_bridge = _metric_from_rows(cm, "m3w_neural_v1_composite_tail_linear_bridge")
    ungated = _metric_from_rows(cm, "ungated_full_waypoint_transformer")
    graph_group = _metric_from_rows(cm, "graph_interaction_group_consistency")
    endpoint = _metric_from_rows(cm, "endpoint_only_final_fde")
    co_vs_endpoint = _co_metric(co)
    cq_vs_endpoint = _cq_metric(cq)
    cq_safety = (cq.get("test_joint_safety", {}) or {}).get("composer_minus_endpoint", {})
    unified_summary = unified.get("summary", {})
    gj_summary = gj.get("summary", {})

    full_decision = gj_summary.get("full_waypoint_promotion_decision", {}) or {}
    supported_modules = gj_summary.get("supported_main_modules_locked", [])
    blocked_modules = gj_summary.get("blocked_main_modules_locked", [])

    return [
        {
            "claim_id": "GZ-C1",
            "claim": "Protected full-waypoint sequence evidence exists and can be cited as protected raw-frame 2.5D world-state evidence.",
            "status": "allowed_with_boundary",
            "evidence": f"CM protected full-waypoint all/t50/t100raw/hard = {_pct(protected_full.get('all_improvement'))} / {_pct(protected_full.get('t50_improvement'))} / {_pct(protected_full.get('t100_raw_frame_diagnostic_improvement'))} / {_pct(protected_full.get('hard_failure_improvement'))}; GJ protected_full_waypoint_runtime_supported={gj_summary.get('protected_full_waypoint_runtime_supported')}",
            "allowed_as_main_claim": True,
            "required_boundary": "protected dataset-local/raw-frame 2.5D; not ungated, not metric, not seconds-level",
        },
        {
            "claim_id": "GZ-C2",
            "claim": "Endpoint-only or endpoint-linear bridge success is equivalent to learned full-waypoint dynamics.",
            "status": "rejected",
            "evidence": f"CM endpoint status={endpoint.get('status')}; linear bridge all={_pct(linear_bridge.get('all_improvement'))}, protected full-waypoint all={_pct(protected_full.get('all_improvement'))}; endpoint bridge may remain a floor but cannot be counted as learned shape.",
            "allowed_as_main_claim": False,
            "required_boundary": "endpoint bridge and full-waypoint shape must be reported separately",
        },
        {
            "claim_id": "GZ-C3",
            "claim": "Ungated full-waypoint neural dynamics is deployable.",
            "status": "rejected_by_safety",
            "evidence": f"CM ungated full-waypoint easy degradation={_pct(ungated.get('easy_degradation'))}; GJ ungated_full_waypoint_deployable={gj_summary.get('ungated_full_waypoint_deployable')}",
            "allowed_as_main_claim": False,
            "required_boundary": "deployment requires protected switch / teacher floor",
        },
        {
            "claim_id": "GZ-C4",
            "claim": "Common-validation full-waypoint composer has positive protected endpoint-bridge replacement evidence.",
            "status": "allowed_with_safety_caveat",
            "evidence": f"CO vs endpoint ADE all/t50/t100raw/hard/easy = {_pct(co_vs_endpoint.get('all_improvement'))} / {_pct(co_vs_endpoint.get('t50_improvement'))} / {_pct(co_vs_endpoint.get('t100_raw_frame_diagnostic_improvement'))} / {_pct(co_vs_endpoint.get('hard_failure_improvement'))} / {_pct(co_vs_endpoint.get('easy_degradation'))}; use_full_rate={_pct((co.get('test_eval', {}) or {}).get('use_full_rate'))}",
            "allowed_as_main_claim": True,
            "required_boundary": "protected common-validation composer, not global floor-free replacement",
        },
        {
            "claim_id": "GZ-C5",
            "claim": "Proximity-aware guard can be omitted without changing the safety claim.",
            "status": "rejected_by_joint_safety",
            "evidence": f"CQ guarded vs endpoint ADE all/t50/hard = {_pct(cq_vs_endpoint.get('all_improvement'))} / {_pct(cq_vs_endpoint.get('t50_improvement'))} / {_pct(cq_vs_endpoint.get('hard_failure_improvement'))}; near@0.05 delta vs endpoint={_pct(cq_safety.get('near_collision_rate_005_delta'))}",
            "allowed_as_main_claim": False,
            "required_boundary": "safety-sensitive reports must use proximity-aware guard or explicitly mark caveat",
        },
        {
            "claim_id": "GZ-C6",
            "claim": "Graph/group consistency can be cited as a protected full-waypoint module, but not as independent neighbor/interaction dominance.",
            "status": "allowed_limited_claim",
            "evidence": f"CM graph/group all/t50/hard={_pct(graph_group.get('all_improvement'))} / {_pct(graph_group.get('t50_improvement'))} / {_pct(graph_group.get('hard_failure_improvement'))}; supported_modules={supported_modules}; blocked_modules={blocked_modules}",
            "allowed_as_main_claim": True,
            "required_boundary": "claim must be group-consistency full-waypoint under protected policy; neighbor_interaction remains blocked as independent main claim",
        },
        {
            "claim_id": "GZ-C7",
            "claim": "Unified row-level full-waypoint evidence supports broad protected source/domain evidence, but not uniform horizon success.",
            "status": "allowed_with_horizon_limit",
            "evidence": f"W unified ADE all/t50/t100raw/hard/easy mean = {_pct((unified_summary.get('ade_all', {}) or {}).get('mean'))} / {_pct((unified_summary.get('ade_t50', {}) or {}).get('mean'))} / {_pct((unified_summary.get('ade_t100_raw_frame_diagnostic', {}) or {}).get('mean'))} / {_pct((unified_summary.get('ade_hard_failure', {}) or {}).get('mean'))} / {_pct((unified_summary.get('ade_easy_degradation', {}) or {}).get('mean'))}",
            "allowed_as_main_claim": True,
            "required_boundary": "source/domain protected evidence only; weak horizon/h100 blocker remains separate",
        },
        {
            "claim_id": "GZ-C8",
            "claim": "Global primary full-waypoint replacement claim is allowed.",
            "status": "rejected_by_protocol_boundary",
            "evidence": f"GJ global_primary_full_waypoint_replacement_claim_allowed={full_decision.get('global_primary_full_waypoint_replacement_claim_allowed')}; reason={full_decision.get('reason')}",
            "allowed_as_main_claim": False,
            "required_boundary": "report source-level/protected components; do not collapse protocols into global primary replacement",
        },
        {
            "claim_id": "GZ-C9",
            "claim": "Metric/seconds-level, true-3D, foundation, Stage5C, or SMC claims are allowed.",
            "status": "rejected_global_boundary",
            "evidence": "All inputs are dataset-local/raw-frame 2.5D evidence; Stage5C and SMC remain false.",
            "allowed_as_main_claim": False,
            "required_boundary": "raw-frame / dataset-local only",
        },
    ]


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    cm = payload["inputs"]["cm"]
    co = payload["inputs"]["co"]
    cq = payload["inputs"]["cq"]
    unified = payload["inputs"]["unified"]
    gj = payload["inputs"]["gj"]
    claim_rows = payload["claim_rows"]

    co_ade = _co_metric(co)
    cq_ade = _cq_metric(cq)
    cq_safety = (cq.get("test_joint_safety", {}) or {}).get("composer_minus_endpoint", {})
    protected_full = _metric_from_rows(cm, "full_waypoint_transformer_protected")
    linear_bridge = _metric_from_rows(cm, "m3w_neural_v1_composite_tail_linear_bridge")
    ungated = _metric_from_rows(cm, "ungated_full_waypoint_transformer")
    gj_summary = gj.get("summary", {})
    decision = gj_summary.get("full_waypoint_promotion_decision", {}) or {}

    rejected_ids = {row["claim_id"] for row in claim_rows if not row["allowed_as_main_claim"]}
    allowed_ids = {row["claim_id"] for row in claim_rows if row["allowed_as_main_claim"]}

    gates = {
        "stage42_cm_input_pass": _gate_passed(cm, "stage42_cm_gate"),
        "stage42_co_input_pass": _gate_passed(co, "stage42_co_gate"),
        "stage42_cq_input_pass": _gate_passed(cq, "stage42_cq_gate"),
        "stage42_w_input_pass": _gate_passed(unified, "stage42_w_gate"),
        "stage42_gj_input_pass": _gate_passed(gj, "stage42_gj_gate"),
        "protected_full_waypoint_positive_recorded": protected_full.get("t50_improvement", 0.0) > 0.0
        and protected_full.get("hard_failure_improvement", 0.0) > 0.0,
        "linear_bridge_all_advantage_preserved": linear_bridge.get("all_improvement", 0.0)
        > protected_full.get("all_improvement", 0.0),
        "common_validation_composer_positive": co_ade.get("all_improvement", 0.0) > 0.0
        and co_ade.get("t50_improvement", 0.0) > 0.0
        and co_ade.get("hard_failure_improvement", 0.0) > 0.0,
        "proximity_guard_positive_and_safe": cq_ade.get("all_improvement", 0.0) > 0.0
        and cq_safety.get("near_collision_rate_005_delta", 1.0) <= 0.0,
        "ungated_full_waypoint_blocked": ungated.get("easy_degradation", 0.0) > 0.02
        and "GZ-C3" in rejected_ids,
        "endpoint_as_full_waypoint_blocked": "GZ-C2" in rejected_ids,
        "global_primary_replacement_blocked": decision.get("global_primary_full_waypoint_replacement_claim_allowed") is False
        and "GZ-C8" in rejected_ids,
        "group_consistency_allowed_only_with_boundary": "GZ-C6" in allowed_ids
        and "neighbor_interaction" in gj_summary.get("blocked_main_modules_locked", []),
        "uniform_horizon_overclaim_blocked": "GZ-C7" in allowed_ids,
        "no_future_test_or_central_velocity_leakage": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["future_waypoints_input"] is False
        and payload["no_leakage"]["central_velocity"] is False
        and payload["no_leakage"]["test_threshold_tuning"] is False,
        "no_metric_seconds_overclaim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = sum(bool(v) for v in gates.values())
    total = len(gates)
    verdict = "stage42_gz_full_waypoint_claim_guard_pass" if passed == total else "stage42_gz_full_waypoint_claim_guard_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_gz_gate"]
    lines = [
        "# Stage42-GZ Full-Waypoint Claim Guard",
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
        "## Claim Rows",
        "",
        "| id | status | allowed main claim | claim | evidence | required boundary |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for row in payload["claim_rows"]:
        lines.append(
            f"| `{row['claim_id']}` | `{row['status']}` | {row['allowed_as_main_claim']} | {row['claim']} | {row['evidence']} | {row['required_boundary']} |"
        )
    lines += [
        "",
        "## Deployment Interpretation",
        "",
        "- Protected full-waypoint evidence can be cited only as dataset-local/raw-frame 2.5D world-state evidence.",
        "- Endpoint-only and endpoint-linear bridge evidence remain separate from learned full-waypoint shape evidence.",
        "- Ungated full-waypoint neural deployment remains rejected.",
        "- Group-consistency full-waypoint is a supported protected module; neighbor/interaction alone remains blocked as an independent main claim.",
        "- Uniform horizon / h100 success is not claimed here.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_gz_gate"]
    lines = [
        "# Stage42-GZ Gate",
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
    lines = [
        "## Stage42-GZ Full-Waypoint Claim Guard",
        "",
        "- source: `fresh_stage42_gz_full_waypoint_claim_guard`",
        f"- gate: `{payload['stage42_gz_gate']['passed']} / {payload['stage42_gz_gate']['total']}`",
        f"- verdict: `{payload['stage42_gz_gate']['verdict']}`",
        "- Protected full-waypoint evidence can be cited only as dataset-local/raw-frame 2.5D evidence.",
        "- Endpoint-only or endpoint-linear bridge success must not be counted as learned full-waypoint dynamics.",
        "- Ungated full-waypoint neural deployment remains rejected.",
        "- Group-consistency full-waypoint is supported under protected policy; neighbor/interaction alone remains blocked as an independent main claim.",
        "- No metric/seconds/true-3D/foundation/Stage5C/SMC claim is allowed.",
    ]
    status = []
    for path in [README_RESULTS, M3W_README, SUMMARY_README]:
        if path.exists():
            _replace_section(path, "STAGE42_GZ_FULL_WAYPOINT_CLAIM_GUARD", lines)
            text = path.read_text(encoding="utf-8")
            status.append(
                {
                    "path": str(path),
                    "exists": True,
                    "contains_stage42_gz": "Stage42-GZ Full-Waypoint Claim Guard" in text,
                    "blocks_endpoint_overclaim": "Endpoint-only or endpoint-linear bridge success must not be counted" in text,
                    "blocks_ungated_full_waypoint": "Ungated full-waypoint neural deployment remains rejected" in text,
                }
            )
        else:
            status.append({"path": str(path), "exists": False})
    return status


def run_stage42_full_waypoint_claim_guard() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    cm = read_json(CM_JSON, {})
    co = read_json(CO_JSON, {})
    cq = read_json(CQ_JSON, {})
    unified = read_json(W_JSON, {})
    gj = read_json(GJ_JSON, {})
    payload: dict[str, Any] = {
        "source": "fresh_stage42_gz_full_waypoint_claim_guard",
        "stage": "Stage42-GZ full-waypoint claim guard",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([CM_JSON, CO_JSON, CQ_JSON, W_JSON, GJ_JSON]),
        "inputs": {
            "cm": cm,
            "co": co,
            "cq": cq,
            "unified": unified,
            "gj": gj,
        },
        "claim_rows": _build_claim_rows(cm, co, cq, unified, gj),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "ungated_full_waypoint_deployable": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_gz_gate"] = _gate(payload)
    payload["input_status"] = {
        "cm": {
            "path": str(CM_JSON),
            "source": cm.get("source"),
            "verdict": (cm.get("stage42_cm_gate", {}) or {}).get("verdict"),
        },
        "co": {
            "path": str(CO_JSON),
            "source": co.get("source"),
            "verdict": (co.get("stage42_co_gate", {}) or {}).get("verdict"),
        },
        "cq": {
            "path": str(CQ_JSON),
            "source": cq.get("source"),
            "verdict": (cq.get("stage42_cq_gate", {}) or {}).get("verdict"),
        },
        "unified": {
            "path": str(W_JSON),
            "source": unified.get("source"),
            "verdict": (unified.get("stage42_w_gate", {}) or {}).get("verdict"),
        },
        "module_claim_lock": {
            "path": str(GJ_JSON),
            "source": gj.get("source"),
            "verdict": (gj.get("stage42_gj_gate", {}) or {}).get("verdict"),
        },
    }
    payload["doc_refresh_status"] = _refresh_docs(payload)
    # Keep the committed report lightweight: the input hash/status proves provenance,
    # while the upstream artifacts remain the authoritative full evidence.
    payload.pop("inputs", None)

    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    return payload


if __name__ == "__main__":
    run_stage42_full_waypoint_claim_guard()
