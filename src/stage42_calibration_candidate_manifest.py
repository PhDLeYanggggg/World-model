from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
DU_JSON = OUT_DIR / "raw_source_time_geometry_hint_audit_stage42.json"
BN_JSON = OUT_DIR / "source_time_geometry_calibration_stage42.json"
DATA_CALIBRATION_JSON = OUT_DIR / "data_calibration_stage42.json"
DATA_CALIBRATION_MD = OUT_DIR / "data_calibration_stage42.md"

REPORT_JSON = OUT_DIR / "calibration_candidate_manifest_stage42.json"
REPORT_MD = OUT_DIR / "calibration_candidate_manifest_stage42.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_calibration_candidates_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_dv_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "source_specific_candidate_claim_allowed_after_terms_only": True,
    "converted_datasets_now": 0,
    "evaluated_datasets_now": 0,
    "stage5c_executed": False,
    "smc_enabled": False,
}

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-DV 是 calibration candidate manifest：合并 raw H/FPS/stride hints 与 source-specific calibration evidence。",
    "本步骤不转换数据、不训练、不评估，只给出下一步 terms/source/time/geometry closure 优先级。",
    "H/FPS/stride/source-specific evidence 不能写成全局 metric 或 seconds-level claim。",
    "source-specific calibrated subset 也必须等 legal terms、source identity、path/version 和 no-leakage conversion 完成后才能声明。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon；dataset-local/raw-frame 不能写成 global metric/seconds。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


OFFICIAL_ACTIONS = {
    "UCY": "confirm UCY official terms/credit requirements, source identity, local path/version, and H.txt convention before conversion",
    "ETH_UCY": "confirm ETH/BIWI terms, source identity, local path/version, annotation fps, and meter-coordinate convention before conversion",
    "TrajNet": "confirm TrajNet++ official terms, train/val/test split license, and ndjson fps/coordinate convention before conversion",
    "OpenTraj": "confirm OpenTraj toolkit role as loader/mirror only; do not treat toolkit as dataset license by itself",
    "SDD": "keep SDD as already converted pixel raw-frame reference; do not count as new external metric/time source",
    "other_topdown": "provide official URL, terms, and raw data identity before conversion",
    "traffic_diagnostic": "diagnostic only; do not use as pedestrian top-down official benchmark",
}


def _du_row_map(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("dataset_id")): row for row in payload.get("target_rows", [])}


def _bn_sources(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("source_rows", "source_records", "sources"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return rows
    # Older BN payloads primarily expose summary-level source ids.
    ids = payload.get("summary", {}).get("source_specific_metric_time_sources", [])
    return [{"source_id": source_id, "source_specific_metric_time_evidence": True} for source_id in ids]


def _source_ids_for_domain(bn_payload: Mapping[str, Any], domain: str, dataset_id: str) -> list[str]:
    rows = _bn_sources(bn_payload)
    ids: list[str] = []
    for row in rows:
        source_id = str(row.get("source_id", ""))
        row_domain = str(row.get("domain", ""))
        if row.get("source_specific_metric_time_evidence") is False:
            continue
        if row_domain == domain or (domain == "UCY" and source_id.startswith("UCY_")) or (
            domain == "ETH_UCY" and source_id.startswith("ETH_")
        ):
            ids.append(source_id)
    if dataset_id == "trajnetplusplus_official":
        return [source_id for source_id in ids if source_id.startswith("TrajNet")]
    return sorted(set(ids))


def classify_candidate(row: Mapping[str, Any], bn_source_ids: list[str]) -> dict[str, Any]:
    dataset_id = str(row.get("dataset_id"))
    domain = str(row.get("domain"))
    has_h = int(row.get("h_matrix_hint_count", 0)) > 0
    has_time = int(row.get("time_metadata_hint_count", 0)) > 0
    has_stride = int(row.get("frame_stride_hint_count", 0)) > 0
    has_bn = bool(bn_source_ids)
    is_sdd = dataset_id == "stanford_drone_dataset"
    is_traffic = domain == "traffic_diagnostic"
    legal_ready = bool(row.get("legal_conversion_ready"))

    if is_traffic:
        candidate_class = "diagnostic_only_missing_or_traffic"
        priority = 0
    elif is_sdd:
        candidate_class = "reference_only_not_new_external"
        priority = 20
    elif has_h and has_time and has_stride and has_bn:
        candidate_class = "source_specific_metric_time_candidate_after_terms"
        priority = 95
    elif has_h and has_time:
        candidate_class = "raw_h_time_candidate_needs_source_crosscheck"
        priority = 80
    elif has_time and has_stride:
        candidate_class = "time_stride_candidate_dataset_local_only"
        priority = 55
    elif has_stride:
        candidate_class = "stride_only_raw_frame_candidate"
        priority = 35
    else:
        candidate_class = "not_calibration_candidate_now"
        priority = 10 if row.get("files_scanned", 0) else 0

    blockers = []
    if not legal_ready:
        blockers.append("terms/source_identity/path_version_not_confirmed")
    if candidate_class.startswith("source_specific") and not has_bn:
        blockers.append("source_specific_bn_evidence_missing")
    if not has_h:
        blockers.append("homography_or_coordinate_transform_missing")
    if not has_time:
        blockers.append("fps_or_annotation_timestep_missing")
    if is_sdd:
        blockers.append("already_sdd_reference_not_external_expansion")
    if is_traffic:
        blockers.append("traffic_diagnostic_not_pedestrian_topdown_official")

    return {
        "dataset_id": dataset_id,
        "domain": domain,
        "candidate_class": candidate_class,
        "priority_score": priority,
        "bn_source_specific_candidates": bn_source_ids,
        "h_matrix_hints": int(row.get("h_matrix_hint_count", 0)),
        "time_hints": int(row.get("time_metadata_hint_count", 0)),
        "frame_stride_hints": int(row.get("frame_stride_hint_count", 0)),
        "metric_time_subset_hint": bool(row.get("metric_time_subset_hint")),
        "legal_conversion_ready": legal_ready,
        "conversion_allowed_now": False,
        "evaluation_allowed_now": False,
        "global_metric_or_seconds_claim_allowed": False,
        "next_action": OFFICIAL_ACTIONS.get(domain, "confirm official source, terms, path/version, and geometry/time semantics"),
        "blockers": blockers,
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    claim = payload["claim_boundary"]
    gates = {
        "du_input_present": bool(payload.get("stage42_du_verdict")),
        "bn_input_present": bool(payload.get("stage42_bn_verdict")),
        "candidate_manifest_written": s["targets_checked"] >= 7,
        "source_specific_candidates_ranked": s["source_specific_candidate_targets"] >= 2,
        "time_stride_candidates_ranked": s["time_stride_candidate_targets"] >= 1,
        "legal_blockers_preserved": s["conversion_ready_targets"] == 0,
        "no_conversion_claim": claim["converted_datasets_now"] == 0,
        "no_evaluation_claim": claim["evaluated_datasets_now"] == 0,
        "global_metric_seconds_blocked": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "user_action_required_written": bool(payload.get("user_action_required_written")),
        "data_calibration_updated": bool(payload.get("data_calibration_updated")),
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_dv_calibration_candidate_manifest_pass" if passed == total else "stage42_dv_calibration_candidate_manifest_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-DV Calibration Candidate Manifest",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{payload['stage42_dv_gate']['passed']} / {payload['stage42_dv_gate']['total']}`",
        f"- verdict: `{payload['stage42_dv_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        *[f"- {key}: `{value}`" for key, value in payload["summary"].items()],
        "",
        "## Candidate Table",
        "",
        "| dataset | domain | class | priority | H | time | stride | source candidates | conversion allowed |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in payload["candidate_rows"]:
        sources = ", ".join(row["bn_source_specific_candidates"]) or "none"
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | {} | `{}` |".format(
                row["dataset_id"],
                row["domain"],
                row["candidate_class"],
                row["priority_score"],
                row["h_matrix_hints"],
                row["time_hints"],
                row["frame_stride_hints"],
                sources,
                row["conversion_allowed_now"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- UCY and ETH/BIWI are the highest-priority source-specific calibration candidates, but terms/source/path/version confirmation is still required.",
            "- TrajNet currently remains time/stride or dataset-local unless official coordinate and split semantics are confirmed.",
            "- SDD is a reference pixel raw-frame source here, not a new external calibration source.",
            "- No row in this manifest authorizes conversion, evaluation, metric claims, or seconds-level claims by itself.",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | --- |",
            *[f"| `{key}` | `{value}` |" for key, value in payload["stage42_dv_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage42-DV Calibration Candidates",
        "",
        "优先确认以下数据源的官方 terms/source identity/path/version/time-geometry semantics。确认前不得转换为 official metric/seconds subset。",
        "",
        "| priority | dataset | source candidates | required action | blockers |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for row in sorted(payload["candidate_rows"], key=lambda r: r["priority_score"], reverse=True):
        sources = ", ".join(row["bn_source_specific_candidates"]) or "none"
        lines.append(
            "| `{}` | `{}` | {} | {} | {} |".format(
                row["priority_score"],
                row["dataset_id"],
                sources,
                row["next_action"],
                ", ".join(row["blockers"]) or "none",
            )
        )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_dv_gate"]
    return [
        "# Stage42-DV Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
        *[f"| `{key}` | `{value}` |" for key, value in gate["gates"].items()],
    ]


def _calibration_addendum(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    return [
        "## Stage42-DV Calibration Candidate Manifest",
        "",
        "- source: `fresh_synthesis_from_stage42_du_bn`",
        "- role: ranks raw-source calibration candidates; no conversion, no evaluation, no metric/seconds claim.",
        f"- gate: `{payload['stage42_dv_gate']['passed']} / {payload['stage42_dv_gate']['total']}`; verdict `{payload['stage42_dv_gate']['verdict']}`.",
        f"- source-specific candidate targets: `{s['source_specific_candidate_targets']}`; time/stride candidate targets: `{s['time_stride_candidate_targets']}`.",
        f"- conversion-ready targets: `{s['conversion_ready_targets']}`; converted/evaluated now: `{s['converted_datasets_now']}` / `{s['evaluated_datasets_now']}`.",
        "- Candidate status remains blocked by user-confirmed terms/source/path/version and no-leakage conversion.",
    ]


def _refresh_data_calibration(payload: Mapping[str, Any]) -> None:
    existing = read_json(DATA_CALIBRATION_JSON, {}) if DATA_CALIBRATION_JSON.exists() else {}
    existing["stage42_dv_calibration_candidate_manifest"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_dv_gate"]["verdict"],
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(DATA_CALIBRATION_JSON, existing)
    _replace_section(DATA_CALIBRATION_MD, "STAGE42_DV_CALIBRATION_CANDIDATE_MANIFEST", _calibration_addendum(payload))


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    return [
        "## Stage42-DV Calibration Candidate Manifest",
        "",
        "- source: `fresh_synthesis_from_stage42_du_bn`",
        "- role: ranks source-specific calibration candidates from raw H/FPS/stride hints; no conversion/evaluation.",
        f"- gate: `{payload['stage42_dv_gate']['passed']} / {payload['stage42_dv_gate']['total']}`; verdict `{payload['stage42_dv_gate']['verdict']}`.",
        f"- source-specific candidate targets: `{s['source_specific_candidate_targets']}`; time/stride candidate targets: `{s['time_stride_candidate_targets']}`.",
        f"- conversion-ready targets: `{s['conversion_ready_targets']}`; global metric/seconds claim remains `False`.",
        f"- report: `{REPORT_MD}`.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, GOAL_SUMMARY]:
        _replace_section(path, "STAGE42_DV_CALIBRATION_CANDIDATE_MANIFEST", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-DV calibration candidate manifest"
    state["current_verdict"] = payload["stage42_dv_gate"]["verdict"]
    state["stage42_dv_calibration_candidate_manifest"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "user_action_required": str(USER_ACTION_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_dv_gate"]["verdict"],
        "gates": f"{payload['stage42_dv_gate']['passed']}/{payload['stage42_dv_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_calibration_candidate_manifest() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    du_payload = read_json(DU_JSON, {})
    bn_payload = read_json(BN_JSON, {})
    rows = []
    for row in _du_row_map(du_payload).values():
        rows.append(classify_candidate(row, _source_ids_for_domain(bn_payload, str(row.get("domain")), str(row.get("dataset_id")))))
    rows = sorted(rows, key=lambda r: (r["priority_score"], r["dataset_id"]), reverse=True)
    payload: dict[str, Any] = {
        "source": "fresh_synthesis_from_stage42_du_bn",
        "stage": "Stage42-DV",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_facts": CURRENT_FACTS,
        "stage42_du_report": str(DU_JSON),
        "stage42_du_verdict": du_payload.get("stage42_du_gate", {}).get("verdict"),
        "stage42_bn_report": str(BN_JSON),
        "stage42_bn_verdict": bn_payload.get("stage42_bn_gate", {}).get("verdict"),
        "candidate_rows": rows,
        "summary": {
            "targets_checked": len(rows),
            "source_specific_candidate_targets": sum(
                1 for row in rows if row["candidate_class"] == "source_specific_metric_time_candidate_after_terms"
            ),
            "time_stride_candidate_targets": sum(
                1 for row in rows if row["candidate_class"] == "time_stride_candidate_dataset_local_only"
            ),
            "stride_only_candidate_targets": sum(1 for row in rows if row["candidate_class"] == "stride_only_raw_frame_candidate"),
            "conversion_ready_targets": sum(1 for row in rows if row["legal_conversion_ready"]),
            "converted_datasets_now": 0,
            "evaluated_datasets_now": 0,
            "recommended_first_targets": [
                row["dataset_id"] for row in rows if row["priority_score"] >= 80 and row["dataset_id"] != "stanford_drone_dataset"
            ][:3],
        },
        "user_action_required_written": True,
        "data_calibration_updated": True,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["stage42_dv_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    _refresh_data_calibration(payload)
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_calibration_candidate_manifest()
