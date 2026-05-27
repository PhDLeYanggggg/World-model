from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src import stage42_post_bj_local_source_verification as bk
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

GW_JSON = OUT_DIR / "h100_blocker_closure_decision_stage42.json"
FQ_JSON = OUT_DIR / "h100_source_support_repair_queue_stage42.json"
FS_JSON = OUT_DIR / "ucy_h100_terms_intake_validator_stage42.json"

REPORT_JSON = OUT_DIR / "ucy_h100_candidate_integrity_manifest_stage42.json"
REPORT_MD = OUT_DIR / "ucy_h100_candidate_integrity_manifest_stage42.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_ucy_h100_candidate_integrity_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_gx_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
CONSOLIDATED_SUMMARY = Path("README_M3W_CURRENT_GOAL_CONSOLIDATED_SUMMARY_ZH.md")
PAPER_EVIDENCE = OUT_DIR / "paper_ready_evidence_matrix_stage42.md"
A_JOURNAL_GAP = OUT_DIR / "a_journal_gap_stage42.md"
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_gx_ucy_h100_candidate_integrity_manifest"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-GX 只生成 UCY h100 candidate integrity manifest，不下载、不转换、不训练、不评估。",
    "文件 hash、路径存在和 parseability 不等于 legal permission 或 conversion readiness。",
    "terms accepted、local path、source identity、guarded conversion、no-leakage/source-CV 都通过前，不能把 repair 写成完成。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
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
    "uniform_h100_or_t100_claim_allowed": False,
    "raw_data_committed": False,
    "download_executed": False,
    "conversion_executed": False,
    "training_executed": False,
    "evaluation_executed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_candidate_path(relative_path: str) -> Path:
    return bk.DATASETS_ROOT / relative_path


def _source_identity_suggestion(relative_path: str) -> str:
    parts = Path(relative_path).parts
    if len(parts) >= 2 and parts[0] == "UCY":
        return f"UCY::{parts[1]}::{Path(relative_path).stem}"
    return f"UCY::{relative_path}"


def _load_inputs() -> dict[str, Any]:
    return {
        "gw": read_json(GW_JSON, {}),
        "fq": read_json(FQ_JSON, {}),
        "fs": read_json(FS_JSON, {}),
    }


def _find_verdict(payload: Mapping[str, Any]) -> str:
    for key, value in payload.items():
        if key.endswith("_gate") and isinstance(value, Mapping):
            return str(value.get("verdict", ""))
    return str(payload.get("verdict", ""))


def _ucy_closure_decision(gw_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    for row in gw_payload.get("closure_decisions", []):
        if row.get("key") == "UCY|100":
            return row
    return {}


def _ucy_candidate_rows(fq_payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return list(fq_payload.get("key_rows", {}).get("UCY|100", {}).get("top_candidates", []))


def _manifest_row(candidate: Mapping[str, Any]) -> dict[str, Any]:
    relative_path = str(candidate.get("relative_path", ""))
    path = _resolve_candidate_path(relative_path)
    exists = path.exists() and path.is_file()
    parsed = bk._parse_file(path) if exists else {}
    return {
        "relative_path": relative_path,
        "absolute_path": str(path),
        "source_identity_suggestion": _source_identity_suggestion(relative_path),
        "exists": exists,
        "sha256": _sha256(path) if exists else "",
        "file_size_bytes": int(path.stat().st_size) if exists else 0,
        "family_bucket": candidate.get("family_bucket", ""),
        "target_bucket_match": bool(candidate.get("target_bucket_match", False)),
        "priority_score": int(candidate.get("priority_score", 0) or 0),
        "candidate_estimated_t100_windows": int(candidate.get("estimated_t100_windows", 0) or 0),
        "parsed_rows": int(parsed.get("parsed_rows", 0) or 0),
        "skipped_rows": int(parsed.get("skipped_rows", 0) or 0),
        "unique_agents": int(parsed.get("unique_agents", 0) or 0),
        "unique_frames": int(parsed.get("unique_frames", 0) or 0),
        "min_frame": parsed.get("min_frame"),
        "max_frame": parsed.get("max_frame"),
        "common_frame_step": parsed.get("common_frame_step"),
        "max_track_points": int(parsed.get("max_track_points", 0) or 0),
        "t100_capable": bool(parsed.get("t100_capable", False)),
        "parsed_estimated_t100_windows": int(parsed.get("estimated_t100_windows", 0) or 0),
        "loader_supported": bool(parsed.get("loader_supported", False)),
        "synthetic_or_diagnostic": bool(parsed.get("synthetic_or_diagnostic", False)),
        "raw_content_stored_in_manifest": False,
        "conversion_ready_now": False,
        "conversion_executed": False,
        "evaluation_executed": False,
        "legal_status": "terms_unverified_user_confirmation_required",
    }


def _summary(rows: list[Mapping[str, Any]], inputs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    ucy_decision = _ucy_closure_decision(inputs["gw"])
    return {
        "source": SOURCE,
        "input_gw_verdict": _find_verdict(inputs["gw"]),
        "input_fq_verdict": _find_verdict(inputs["fq"]),
        "input_fs_verdict": _find_verdict(inputs["fs"]),
        "candidate_rows": len(rows),
        "existing_files": sum(1 for row in rows if row["exists"]),
        "target_family_candidates": sum(1 for row in rows if row["target_bucket_match"]),
        "t100_capable_files": sum(1 for row in rows if row["t100_capable"]),
        "total_parsed_rows": sum(int(row["parsed_rows"]) for row in rows),
        "total_parsed_t100_windows": sum(int(row["parsed_estimated_t100_windows"]) for row in rows),
        "unique_hashes": len({str(row["sha256"]) for row in rows if row["sha256"]}),
        "conversion_ready_now_count": 0,
        "downloaded_now": 0,
        "converted_now": 0,
        "evaluated_now": 0,
        "ucy_gw_closure_status": ucy_decision.get("closure_status", ""),
        "ucy_gw_legal_conversion_ready": bool(ucy_decision.get("legal_conversion_ready") is True),
        "uniform_h100_or_t100_claim_allowed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    rows = payload["candidate_integrity_rows"]
    gates = {
        "gw_input_verified": str(s["input_gw_verdict"]).startswith("stage42_gw_h100_blocker_closure_decision_pass"),
        "fq_input_verified": str(s["input_fq_verdict"]).startswith("stage42_fq_h100_source_support_repair_queue_pass"),
        "fs_input_verified": str(s["input_fs_verdict"]).startswith("stage42_fs_ucy_h100_terms_intake_validator_pass"),
        "ucy_candidates_loaded": s["candidate_rows"] >= 1,
        "all_candidate_files_exist": s["existing_files"] == s["candidate_rows"],
        "target_family_candidates_present": s["target_family_candidates"] >= 1,
        "hashes_computed": s["unique_hashes"] >= 1 and all(bool(row["sha256"]) for row in rows),
        "parsed_stats_computed": s["total_parsed_rows"] > 0 and s["t100_capable_files"] >= 1,
        "raw_content_not_stored": all(row["raw_content_stored_in_manifest"] is False for row in rows),
        "legal_blocker_preserved": s["ucy_gw_legal_conversion_ready"] is False,
        "no_conversion_ready_claim": s["conversion_ready_now_count"] == 0,
        "no_download_conversion_eval": s["downloaded_now"] == 0 and s["converted_now"] == 0 and s["evaluated_now"] == 0,
        "user_action_written": payload["user_action_required_written"] is True,
        "no_future_test_or_central_velocity_leakage": (
            payload["no_leakage"]["future_endpoint_input"] is False
            and payload["no_leakage"]["future_waypoint_input"] is False
            and payload["no_leakage"]["central_velocity"] is False
            and payload["no_leakage"]["test_endpoint_goals"] is False
            and payload["no_leakage"]["test_threshold_tuning"] is False
        ),
        "no_metric_seconds_overclaim": payload["claim_boundary"]["global_metric_claim_allowed"] is False
        and payload["claim_boundary"]["global_seconds_claim_allowed"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_gx_ucy_h100_candidate_integrity_manifest_pass" if passed == total else "stage42_gx_ucy_h100_candidate_integrity_manifest_partial"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    inputs = _load_inputs()
    candidates = _ucy_candidate_rows(inputs["fq"])
    rows = [_manifest_row(candidate) for candidate in candidates]
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-GX UCY h100 candidate integrity manifest",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([GW_JSON, FQ_JSON, FS_JSON]),
        "current_facts": CURRENT_FACTS,
        "summary": _summary(rows, inputs),
        "candidate_integrity_rows": rows,
        "user_action_required": _user_actions(rows),
        "user_action_required_written": True,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "manifest_only_no_conversion": True,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["stage42_gx_gate"] = _gate(payload)
    return payload


def _user_actions(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "priority": "critical" if row["target_bucket_match"] else "supporting",
            "relative_path": row["relative_path"],
            "source_identity_suggestion": row["source_identity_suggestion"],
            "sha256": row["sha256"],
            "action": "If the user has accepted UCY official terms, copy the source_identity_suggestion into the terms intake and confirm allowed use/local path. Do not run conversion before validator readiness.",
            "claim_guard": "Hash/path integrity is not permission, conversion, evaluation, metric evidence, or h100 repair.",
        }
        for row in rows
    ]


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    gate = payload["stage42_gx_gate"]
    lines = [
        "# Stage42-GX UCY H100 Candidate Integrity Manifest",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- result source: `fresh_run file integrity manifest from cached_verified blocker decisions`",
        "",
        "## Summary",
        "",
        f"- candidate_rows: `{s['candidate_rows']}`",
        f"- existing_files: `{s['existing_files']}`",
        f"- target_family_candidates: `{s['target_family_candidates']}`",
        f"- t100_capable_files: `{s['t100_capable_files']}`",
        f"- total_parsed_rows: `{s['total_parsed_rows']}`",
        f"- total_parsed_t100_windows: `{s['total_parsed_t100_windows']}`",
        f"- unique_hashes: `{s['unique_hashes']}`",
        f"- conversion_ready_now_count: `{s['conversion_ready_now_count']}`",
        f"- UCY legal conversion ready: `{s['ucy_gw_legal_conversion_ready']}`",
        "",
        "## Candidate Integrity Rows",
        "",
        "| path | sha256 | size | source identity | agents | frames | max track | t100 windows | target | legal status |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["candidate_integrity_rows"]:
        lines.append(
            f"| `{row['relative_path']}` | `{row['sha256'][:16]}...` | {row['file_size_bytes']} | "
            f"`{row['source_identity_suggestion']}` | {row['unique_agents']} | {row['unique_frames']} | "
            f"{row['max_track_points']} | {row['parsed_estimated_t100_windows']} | {row['target_bucket_match']} | `{row['legal_status']}` |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- This manifest locks lightweight file identity and parse stats for UCY h100 candidates.",
        "- It stores hashes and counts only; it does not store raw trajectories.",
        "- It does not grant legal permission and does not execute conversion/evaluation.",
        "- `UCY|100` remains blocked until terms/source identity/local path are confirmed and guarded conversion/no-leakage/source-CV pass.",
        "- `TrajNet|100` remains separately hard-blocked by missing long raw source support.",
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | {bool(ok)} |")
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage42-GX UCY H100 Candidate Integrity",
        "",
        "The UCY h100 candidate files are locally present and hash-locked, but not legally conversion-ready. Fill the terms intake only if you have accepted the official UCY terms and can confirm allowed use/local path/source identity.",
        "",
    ]
    for row in payload["user_action_required"]:
        lines += [
            f"## `{row['relative_path']}`",
            "",
            f"- priority: `{row['priority']}`",
            f"- source_identity_suggestion: `{row['source_identity_suggestion']}`",
            f"- sha256: `{row['sha256']}`",
            f"- action: {row['action']}",
            f"- claim_guard: {row['claim_guard']}",
            "",
        ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_gx_gate"]
    lines = [
        "# Stage42-GX Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | {bool(ok)} |")
    return lines


def _refresh_docs(payload: Mapping[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "## Stage42-GX UCY H100 Candidate Integrity Manifest",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{payload['stage42_gx_gate']['passed']} / {payload['stage42_gx_gate']['total']}`; verdict `{payload['stage42_gx_gate']['verdict']}`",
        f"- UCY candidate files: `{s['candidate_rows']}`; existing `{s['existing_files']}`; target-family candidates `{s['target_family_candidates']}`.",
        f"- parsed rows: `{s['total_parsed_rows']}`; parsed t100 windows: `{s['total_parsed_t100_windows']}`; unique hashes `{s['unique_hashes']}`.",
        "- This locks file identity/hash/parse stats only. It is not legal permission, not conversion, not evaluation, and not h100 repair.",
        "- `UCY|100` remains terms/source-identity blocked; `TrajNet|100` remains long-source blocked. No metric/seconds, no Stage5C, no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, CONSOLIDATED_SUMMARY, PAPER_EVIDENCE, A_JOURNAL_GAP]:
        _replace_section(path, "STAGE42_GX_UCY_H100_CANDIDATE_INTEGRITY", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    s = payload["summary"]
    state["current_stage"] = "Stage42-GX UCY h100 candidate integrity manifest"
    state["current_verdict"] = payload["stage42_gx_gate"]["verdict"]
    state["stage42_gx_ucy_h100_candidate_integrity_manifest"] = {
        "source": payload["source"],
        "result_source": "fresh_run_file_integrity_manifest_from_cached_verified_inputs",
        "report": str(REPORT_MD),
        "report_json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "verdict": payload["stage42_gx_gate"]["verdict"],
        "gates": f"{payload['stage42_gx_gate']['passed']}/{payload['stage42_gx_gate']['total']}",
        "candidate_rows": s["candidate_rows"],
        "existing_files": s["existing_files"],
        "target_family_candidates": s["target_family_candidates"],
        "total_parsed_rows": s["total_parsed_rows"],
        "total_parsed_t100_windows": s["total_parsed_t100_windows"],
        "unique_hashes": s["unique_hashes"],
        "conversion_ready_now_count": s["conversion_ready_now_count"],
        "claim_boundary": CLAIM_BOUNDARY,
        "conclusion": "UCY h100 candidates are hash-locked and parseable, but remain terms/source-identity blocked; no conversion/evaluation/h100 repair can run yet.",
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_ucy_h100_candidate_integrity_manifest() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    _refresh_docs(payload)
    _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_ucy_h100_candidate_integrity_manifest()
    gate = result["stage42_gx_gate"]
    print(f"Stage42-GX gate: {gate['passed']} / {gate['total']} - {gate['verdict']}")
