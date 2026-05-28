from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

DATA_CAL_JSON = OUT_DIR / "data_calibration_stage42.json"
SOURCE_TIME_JSON = OUT_DIR / "source_time_geometry_calibration_stage42.json"
RESTRICTED_GUARD_JSON = OUT_DIR / "restricted_metric_time_post_hk_claim_guard_stage42.json"
CALIBRATION_MANIFEST_JSON = OUT_DIR / "calibration_candidate_manifest_stage42.json"
TERMS_SNAPSHOT_JSON = OUT_DIR / "source_terms_confirmation_intake_calibration_snapshot_stage42.json"

REPORT_JSON = OUT_DIR / "calibration_readiness_reconciliation_stage42.json"
REPORT_MD = OUT_DIR / "calibration_readiness_reconciliation_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_jd_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_calibration_readiness_reconciliation_stage42.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
LEDGER = OUT_DIR / "run_ledger.jsonl"

SOURCE = "fresh_stage42_jd_calibration_readiness_reconciliation"
SECTION = "STAGE42_JD_CALIBRATION_READINESS_RECONCILIATION"

REQUIRED_DATASETS = ["sdd", "opentraj", "eth_ucy", "trajnet", "ucy", "tgsim", "aerialmpt"]

PATH_CANDIDATES = {
    "sdd_raw": [Path("external_data/StanfordDroneDataset")],
    "sdd_converted": [Path("data/stage21_sdd_world_state"), Path("data/stage24_sdd_fast_cache")],
    "opentraj_raw": [Path("external_data/OpenTraj")],
    "opentraj_converted": [Path("data/stage20_world_state/opentraj"), Path("data/stage31_external_feature_store")],
    "eth_ucy_converted": [Path("data/stage20_world_state/eth_ucy_full"), Path("data/stage5b_world_state/eth_ucy")],
    "trajnet_converted": [Path("data/stage20_world_state/trajnet_full"), Path("data/stage5b_world_state/trajnet")],
    "ucy_converted": [Path("data/stage20_world_state/ucy_crowd"), Path("data/stage37_t50_history")],
    "tgsim_converted": [Path("data/stage5_world_state/tgsim"), Path("data/stage5b_world_state/tgsim")],
    "aerialmpt_converted": [Path("data/aerialmpt"), Path("data/stage11_multiagent_episodes/aerialmpt")],
}

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-JD 是 data/time/geometry calibration readiness reconciliation，不训练、不下载、不转换、不评估。",
    "ETH/UCY 等 source-specific metric/time hints 只能写成 candidates；除非 terms、guarded conversion、no-leakage 和 restricted eval 都完成，否则不能写成 metric/seconds result。",
    "SDD 仍是 pixel/raw-frame；TrajNet/OpenTraj 当前仍是 dataset-local/raw-frame 或 source-dependent。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
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


def _gate_pass(payload: Mapping[str, Any], gate_name: str) -> bool:
    gate = payload.get(gate_name, {})
    try:
        return int(gate.get("passed", -1)) == int(gate.get("total", 0)) and int(gate.get("total", 0)) > 0
    except Exception:
        return False


def _path_status() -> dict[str, Any]:
    rows = []
    found = 0
    for name, paths in PATH_CANDIDATES.items():
        existing = [str(path) for path in paths if path.exists()]
        found += int(bool(existing))
        rows.append({"key": name, "candidates": [str(path) for path in paths], "found": bool(existing), "existing": existing})
    return {"rows": rows, "groups_checked": len(rows), "groups_found": found}


def _dataset_rows(data_calibration: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in data_calibration.get("datasets", []):
        name = str(row.get("dataset_id") or row.get("dataset") or row.get("dataset_name") or "").lower()
        rows.append(
            {
                "dataset": name,
                "raw_path_found": bool(row.get("raw_path_found", False)),
                "converted_path_found": bool(row.get("converted_path_found", False)),
                "calibration_state": row.get("calibration_state", "unknown"),
                "metric_claim_allowed": bool(row.get("metric_claim_allowed", False)),
                "seconds_claim_allowed": bool(row.get("seconds_claim_allowed", False)),
                "auto_download_allowed": bool(row.get("auto_download_allowed", False)),
                "requires_terms_or_login_or_application": bool(row.get("requires_terms_or_login_or_application", False)),
                "next_actions": row.get("next_actions", []),
            }
        )
    return rows


def _candidate_rows(source_time: Mapping[str, Any], manifest: Mapping[str, Any], guard: Mapping[str, Any]) -> list[dict[str, Any]]:
    guard_summary = guard.get("summary", {})
    source_specific_ready_now = bool(guard_summary.get("hk_ready_now", False))
    rows = []
    for record in source_time.get("source_records", []):
        homography = record.get("homography", {}) or {}
        timing = record.get("timing", {}) or {}
        rows.append(
            {
                "source_id": record.get("source_id") or record.get("source") or "unknown",
                "domain": record.get("domain", "unknown"),
                "h_parseable": bool(homography.get("parseable", record.get("homography_parseable", record.get("H_parseable", False)))),
                "annotation_fps": timing.get("annotation_fps", record.get("annotation_fps")),
                "timestep_s": timing.get("annotation_timestep_seconds", record.get("timestep_s")),
                "local_claim": record.get("allowed_local_claim", record.get("local_claim", record.get("claim", "unknown"))),
                "claimable_now": False,
                "status": "candidate_after_terms_not_converted_or_evaluated",
                "blocker": "requires user-confirmed terms/source identity/local path, guarded conversion, no-leakage audit, and restricted source-CV/final-test evaluation",
            }
        )
    if not rows:
        for row in manifest.get("candidate_rows", []):
            rows.append(
                {
                    "source_id": row.get("source_id", row.get("target", "unknown")),
                    "domain": row.get("domain", "unknown"),
                    "h_parseable": bool(row.get("homography_hint", False)),
                    "annotation_fps": row.get("annotation_fps"),
                    "timestep_s": row.get("timestep_s"),
                    "local_claim": row.get("candidate_type", "candidate"),
                    "claimable_now": False,
                    "status": "candidate_manifest_only",
                    "blocker": "manifest candidate only; no guarded conversion or restricted evaluation completed",
                }
            )
    for row in rows:
        row["claimable_now"] = bool(source_specific_ready_now and False)
    return rows


def _summary(
    data_calibration: Mapping[str, Any],
    source_time: Mapping[str, Any],
    restricted_guard: Mapping[str, Any],
    manifest: Mapping[str, Any],
    path_status: Mapping[str, Any],
    dataset_rows: list[Mapping[str, Any]],
    candidate_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    data_summary = data_calibration.get("summary", {})
    source_summary = source_time.get("summary", {})
    guard_summary = restricted_guard.get("summary", {})
    manifest_summary = manifest.get("summary", {})
    covered = sorted(set(row["dataset"] for row in dataset_rows))
    required_covered = sorted(set(REQUIRED_DATASETS).intersection(covered))
    return {
        "source": SOURCE,
        "datasets_audited": int(data_summary.get("datasets_audited", len(dataset_rows)) or 0),
        "required_datasets": REQUIRED_DATASETS,
        "required_datasets_covered": required_covered,
        "required_dataset_coverage_complete": set(required_covered) == set(REQUIRED_DATASETS),
        "direct_path_groups_checked": int(path_status["groups_checked"]),
        "direct_path_groups_found": int(path_status["groups_found"]),
        "source_specific_metric_time_hint_sources": list(source_summary.get("source_specific_metric_time_sources", [])),
        "source_specific_candidate_count": int(len(candidate_rows)),
        "manifest_conversion_ready_targets": int(manifest_summary.get("conversion_ready_targets", 0) or 0),
        "restricted_terms_confirmed": bool(guard_summary.get("hk_terms_confirmed", False)),
        "restricted_metric_time_ready_now": bool(guard_summary.get("hk_ready_now", False)),
        "restricted_conversion_ready_targets_now": int(guard_summary.get("hk_conversion_ready_targets_now", 0) or 0),
        "converted_datasets_now": int(manifest_summary.get("converted_datasets_now", 0) or 0),
        "evaluated_datasets_now": int(manifest_summary.get("evaluated_datasets_now", 0) or 0),
        "global_metric_claim_allowed": False,
        "global_seconds_claim_allowed": False,
        "restricted_metric_time_claim_allowed_now": False,
        "stage42_b_external_validation_ready": bool(data_summary.get("stage42_b_external_validation_ready", False)),
        "stage42_c_full_waypoint_prereq_ready": bool(data_summary.get("stage42_c_full_waypoint_prereq_ready", False)),
        "decision": "calibration_hints_reconciled_metric_time_claim_still_blocked",
        "next_action": "Ask user to confirm official source terms/local source identity for ETH/UCY calibrated candidates before guarded conversion; keep all current claims raw-frame/dataset-local.",
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    data_calibration = read_json(DATA_CAL_JSON, {})
    source_time = read_json(SOURCE_TIME_JSON, {})
    restricted_guard = read_json(RESTRICTED_GUARD_JSON, {})
    manifest = read_json(CALIBRATION_MANIFEST_JSON, {})
    terms_snapshot = read_json(TERMS_SNAPSHOT_JSON, {})
    paths = _path_status()
    datasets = _dataset_rows(data_calibration)
    candidates = _candidate_rows(source_time, manifest, restricted_guard)
    summary = _summary(data_calibration, source_time, restricted_guard, manifest, paths, datasets, candidates)
    payload: dict[str, Any] = {
        "stage": "Stage42-JD",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([DATA_CAL_JSON, SOURCE_TIME_JSON, RESTRICTED_GUARD_JSON, CALIBRATION_MANIFEST_JSON, TERMS_SNAPSHOT_JSON]),
        "current_facts": CURRENT_FACTS,
        "input_reports": {
            "data_calibration": str(DATA_CAL_JSON),
            "source_time_geometry_calibration": str(SOURCE_TIME_JSON),
            "restricted_metric_time_claim_guard": str(RESTRICTED_GUARD_JSON),
            "calibration_candidate_manifest": str(CALIBRATION_MANIFEST_JSON),
            "terms_snapshot": str(TERMS_SNAPSHOT_JSON),
        },
        "input_gate_status": {
            "data_calibration_present": DATA_CAL_JSON.exists(),
            "source_time_geometry_gate_passed": _gate_pass(source_time, "stage42_bn_gate"),
            "restricted_metric_time_claim_guard_passed": _gate_pass(restricted_guard, "stage42_hl_gate"),
            "calibration_manifest_gate_passed": _gate_pass(manifest, "stage42_dv_gate"),
        },
        "path_status": paths,
        "dataset_rows": datasets,
        "source_specific_candidate_rows": candidates,
        "terms_snapshot_summary": {
            "exists": bool(terms_snapshot),
            "agent_may_fill_legal_acceptance": bool(terms_snapshot.get("agent_may_fill_legal_acceptance", False)) if isinstance(terms_snapshot, dict) else False,
        },
        "summary": summary,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "download_executed": False,
            "conversion_executed": False,
            "training_executed": False,
            "evaluation_executed": False,
            "claim_reconciliation_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "global_metric_claim": False,
            "global_seconds_claim": False,
            "restricted_metric_time_claim_allowed_now": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_jd_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    input_status = payload["input_gate_status"]
    gates = {
        "data_calibration_present": input_status["data_calibration_present"],
        "source_time_geometry_gate_passed": input_status["source_time_geometry_gate_passed"],
        "restricted_metric_time_guard_passed": input_status["restricted_metric_time_claim_guard_passed"],
        "required_dataset_coverage_complete": s["required_dataset_coverage_complete"],
        "direct_paths_inspected": s["direct_path_groups_checked"] >= 8,
        "direct_paths_found": s["direct_path_groups_found"] >= 6,
        "source_specific_candidates_recorded": s["source_specific_candidate_count"] >= 5,
        "restricted_terms_not_confirmed": s["restricted_terms_confirmed"] is False,
        "restricted_ready_now_false": s["restricted_metric_time_ready_now"] is False,
        "conversion_ready_now_zero": s["restricted_conversion_ready_targets_now"] == 0,
        "converted_now_zero": s["converted_datasets_now"] == 0,
        "evaluated_now_zero": s["evaluated_datasets_now"] == 0,
        "global_metric_blocked": payload["claim_boundary"]["global_metric_claim"] is False,
        "global_seconds_blocked": payload["claim_boundary"]["global_seconds_claim"] is False,
        "restricted_claim_blocked": payload["claim_boundary"]["restricted_metric_time_claim_allowed_now"] is False,
        "external_validation_ready": s["stage42_b_external_validation_ready"] is True,
        "full_waypoint_prereq_ready": s["stage42_c_full_waypoint_prereq_ready"] is True,
        "no_download_conversion_training_eval": payload["no_leakage"]["download_executed"] is False
        and payload["no_leakage"]["conversion_executed"] is False
        and payload["no_leakage"]["training_executed"] is False
        and payload["no_leakage"]["evaluation_executed"] is False,
        "no_future_or_test_leakage": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["central_velocity"] is False
        and payload["no_leakage"]["test_endpoint_goals"] is False
        and payload["no_leakage"]["test_threshold_tuning"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_jd_calibration_readiness_reconciliation_pass" if passed == total else "stage42_jd_calibration_readiness_reconciliation_partial"
    return {"source": "fresh_run", "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jd_gate"]
    s = payload["summary"]
    lines = [
        "# Stage42-JD Calibration Readiness Reconciliation",
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
        "## Summary",
        "",
        f"- decision: `{s['decision']}`",
        f"- required_datasets_covered: `{s['required_datasets_covered']}`",
        f"- direct_path_groups_found: `{s['direct_path_groups_found']} / {s['direct_path_groups_checked']}`",
        f"- source_specific_candidate_count: `{s['source_specific_candidate_count']}`",
        f"- restricted_terms_confirmed: `{s['restricted_terms_confirmed']}`",
        f"- restricted_metric_time_ready_now: `{s['restricted_metric_time_ready_now']}`",
        f"- converted/evaluated restricted datasets now: `{s['converted_datasets_now']} / {s['evaluated_datasets_now']}`",
        f"- global_metric_claim_allowed: `{s['global_metric_claim_allowed']}`",
        f"- global_seconds_claim_allowed: `{s['global_seconds_claim_allowed']}`",
        f"- next_action: {s['next_action']}",
        "",
        "## Dataset Readiness",
        "",
        "| dataset | raw | converted | calibration | metric claim | seconds claim | terms/login/app |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in payload["dataset_rows"]:
        lines.append(
            f"| `{row['dataset']}` | `{row['raw_path_found']}` | `{row['converted_path_found']}` | "
            f"`{row['calibration_state']}` | `{row['metric_claim_allowed']}` | `{row['seconds_claim_allowed']}` | "
            f"`{row['requires_terms_or_login_or_application']}` |"
        )
    lines.extend(
        [
            "",
            "## Direct Local Path Check",
            "",
            "| key | found | existing paths |",
            "| --- | ---: | --- |",
        ]
    )
    for row in payload["path_status"]["rows"]:
        lines.append(f"| `{row['key']}` | `{row['found']}` | `{row['existing']}` |")
    lines.extend(
        [
            "",
            "## Source-Specific Metric/Time Candidates",
            "",
            "| source | domain | H parseable | fps | timestep_s | status | claimable now |",
            "| --- | --- | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for row in payload["source_specific_candidate_rows"]:
        lines.append(
            f"| `{row['source_id']}` | `{row['domain']}` | `{row['h_parseable']}` | "
            f"{row.get('annotation_fps')} | {row.get('timestep_s')} | `{row['status']}` | `{row['claimable_now']}` |"
        )
    lines.extend(
        [
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{payload['no_leakage']}`",
            f"- claim_boundary: `{payload['claim_boundary']}`",
            "",
            "## Interpretation",
            "",
            "- Existing local converted state is enough to continue raw-frame/dataset-local external validation and full-waypoint dynamics work.",
            "- ETH/UCY source-specific H/FPS/timestep hints are valuable, but they remain candidates until user-confirmed terms and guarded conversion/evaluation happen.",
            "- Current paper language must keep global metric/seconds and restricted metric/time claims blocked.",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jd_gate"]
    lines = [
        "# Stage42-JD Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | `{bool(value)}` |")
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    rows = payload["source_specific_candidate_rows"]
    return [
        "# User Action Required: Stage42-JD Calibration Readiness",
        "",
        "The following source-specific metric/time hints cannot be claimed or evaluated as metric/seconds-level M3W results yet.",
        "Please confirm official terms, source identity, and local paths before guarded conversion.",
        "",
        "## Required User Actions",
        "",
        "1. Confirm that the local ETH/BIWI and UCY sources match their official dataset terms.",
        "2. Confirm the exact local path/version to use for each candidate source.",
        "3. Confirm whether the project may run guarded source-specific conversion for those local files.",
        "4. After confirmation, rerun no-leakage conversion and restricted source-CV/final-test evaluation.",
        "",
        "## Candidate Sources",
        "",
        "| source | domain | fps | timestep_s | blocker |",
        "| --- | --- | ---: | ---: | --- |",
        *[
            f"| `{row['source_id']}` | `{row['domain']}` | {row.get('annotation_fps')} | {row.get('timestep_s')} | {row['blocker']} |"
            for row in rows
        ],
        "",
        "Until those actions are complete, M3W remains raw-frame/dataset-local for these sources.",
    ]


def _section_lines(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    gate = payload["stage42_jd_gate"]
    return [
        "## Stage42-JD Calibration Readiness Reconciliation",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict: `{gate['verdict']}`",
        f"- required datasets covered: `{s['required_datasets_covered']}`; direct path groups found `{s['direct_path_groups_found']} / {s['direct_path_groups_checked']}`.",
        f"- source-specific metric/time candidates: `{s['source_specific_candidate_count']}`; ready now: `{s['restricted_metric_time_ready_now']}`.",
        "- conclusion: external validation/full-waypoint work can continue in raw-frame/dataset-local mode, but metric/seconds claims remain blocked until user-confirmed terms, guarded conversion, no-leakage, and restricted evaluation.",
        "- Stage5C not executed; SMC not enabled.",
    ]


def _update_readmes(payload: Mapping[str, Any]) -> None:
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, _section_lines(payload))


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    stage42 = state.setdefault("stage42", {})
    stage42["calibration_readiness_reconciliation"] = {
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "verdict": payload["stage42_jd_gate"]["verdict"],
        "gate": {"passed": payload["stage42_jd_gate"]["passed"], "total": payload["stage42_jd_gate"]["total"]},
        "decision": payload["summary"]["decision"],
        "source_specific_candidate_count": payload["summary"]["source_specific_candidate_count"],
        "restricted_metric_time_ready_now": payload["summary"]["restricted_metric_time_ready_now"],
        "global_metric_claim_allowed": False,
        "global_seconds_claim_allowed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["last_updated_utc"] = payload["generated_at_utc"]
    write_json(RESEARCH_STATE, state)


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(LEDGER.parent)
    import json

    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "stage": "Stage42-JD",
                    "source": payload["source"],
                    "generated_at_utc": payload["generated_at_utc"],
                    "verdict": payload["stage42_jd_gate"]["verdict"],
                    "result": payload["summary"]["decision"],
                    "fresh_run": True,
                    "downloaded": False,
                    "converted": False,
                    "trained": False,
                    "evaluated": False,
                    "stage5c_executed": False,
                    "smc_enabled": False,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def run_stage42_calibration_readiness_reconciliation(*, refresh_readmes: bool = True) -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    if refresh_readmes:
        _update_readmes(payload)
        _update_state(payload)
        _append_ledger(payload)
    return payload


def main() -> None:
    run_stage42_calibration_readiness_reconciliation(refresh_readmes=True)


if __name__ == "__main__":
    main()
