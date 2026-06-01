from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage42_local_calibrated_source_support_intake import _candidate_records as _local_calibrated_records
from src import stage43_full_waypoint_latent_dynamics as m


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_blocked_source_support_acquisition_preflight.json"
REPORT_MD = OUT_DIR / "stage43_blocked_source_support_acquisition_preflight.md"
GATE_MD = OUT_DIR / "stage43_stage_be_blocked_source_support_acquisition_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SOURCE = "fresh_stage43_be_blocked_source_support_acquisition_preflight"
SECTION = "STAGE43_BE_BLOCKED_SOURCE_SUPPORT_ACQUISITION_PREFLIGHT"

STAGE43_BC = OUT_DIR / "stage43_blocked_family_support_scan.json"
STAGE43_BD = OUT_DIR / "stage43_biwi_support_rebuild_preflight.json"
STAGE42_JN = Path("outputs/stage42_long_research/local_calibrated_source_support_intake_stage42.json")


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return read_json(path, {})


def _local_candidate_family(dataset_name: str) -> str:
    lowered = dataset_name.lower()
    if "pets" in lowered:
        return "mot_like"
    if "town" in lowered or "wild" in lowered:
        return "mot_like_or_external_topdown_support"
    return "external_topdown_support"


def _local_candidate_record(row: Mapping[str, Any]) -> dict[str, Any]:
    stats = row.get("stats", {})
    parseable = bool(row.get("parseable", False))
    has_long = int(stats.get("t50_rows", 0)) > 0 or int(stats.get("t100_rows", 0)) > 0
    legal_ready = bool(row.get("legal_auto_convert_allowed", False))
    calibration_count = int(row.get("calibration_file_count", 0))
    blockers: list[str] = []
    if not parseable:
        blockers.append("not_parseable_locally")
    if not has_long:
        blockers.append("no_t50_or_t100_candidate_rows")
    if not legal_ready:
        blockers.append("terms_or_license_not_confirmed_for_benchmark_conversion")
    if calibration_count == 0:
        blockers.append("no_local_calibration_file_found")
    if str(row.get("conversion_status", "")).startswith("not_converted"):
        blockers.append("not_converted_into_stage43_feature_store")
    source_family = _local_candidate_family(str(row.get("dataset_name", "")))
    return {
        "source": "fresh_local_scan",
        "dataset_name": str(row.get("dataset_name", "unknown")),
        "support_family": source_family,
        "local_path": str(row.get("root", "")),
        "parseable": parseable,
        "point_rows": int(stats.get("point_rows", 0)),
        "agent_tracks": int(stats.get("agent_tracks", 0)),
        "t50_candidate_rows": int(stats.get("t50_rows", 0)),
        "t100_candidate_rows": int(stats.get("t100_rows", 0)),
        "calibration_file_count": calibration_count,
        "coordinate_unit": str(row.get("coordinate_unit", "unknown")),
        "metric_status": str(row.get("metric_status", "unverified")),
        "legal_conversion_ready": legal_ready,
        "technical_support_candidate": bool(parseable and has_long),
        "conversion_ready_now": bool(parseable and has_long and legal_ready),
        "blockers": blockers,
        "recommended_action": (
            "record_terms_then_guarded_conversion_preflight"
            if parseable and has_long and not legal_ready
            else "guarded_conversion_preflight"
            if parseable and has_long and legal_ready
            else "acquire_or_repair_source_before_conversion"
        ),
    }


def _blocked_family_records(stage43_bc: Mapping[str, Any], stage43_bd: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for action in stage43_bc.get("blocked_family_actions", []):
        family = str(action.get("family", "unknown"))
        records.append(
            {
                "source": "cached_verified_stage43_bc_raw_scan",
                "family": family,
                "raw_file_count": int(action.get("raw_file_count", 0)),
                "raw_train_dir_files": int(action.get("raw_train_dir_files", 0)),
                "raw_test_dir_files": int(action.get("raw_test_dir_files", 0)),
                "raw_t50_candidate_windows": int(action.get("raw_t50_candidate_windows", 0)),
                "raw_t100_candidate_windows": int(action.get("raw_t100_candidate_windows", 0)),
                "current_train_family_rows": int(action.get("current_train_family_rows", 0)),
                "current_val_family_rows": int(action.get("current_val_family_rows", 0)),
                "current_test_family_rows": int(action.get("current_test_family_rows", 0)),
                "support_candidate_exists_in_raw_scan": bool(action.get("support_candidate_exists_in_raw_scan", False)),
                "repair_training_allowed_now": False,
                "blockers": list(action.get("blockers", [])),
                "recommendation": str(action.get("recommendation", "unknown")),
            }
        )
    if stage43_bd:
        records.append(
            {
                "source": "cached_verified_stage43_bd_biwi_preflight",
                "family": "TrajNet_biwi",
                "raw_file_count": int(stage43_bd.get("summary", {}).get("biwi_source_count_in_feature_store", 0)),
                "raw_train_dir_files": 1,
                "raw_test_dir_files": 1,
                "raw_t50_candidate_windows": int(stage43_bd.get("summary", {}).get("current_t50_test_rows", 0)),
                "raw_t100_candidate_windows": 0,
                "current_train_family_rows": int(stage43_bd.get("summary", {}).get("current_train_rows", 0)),
                "current_val_family_rows": int(stage43_bd.get("summary", {}).get("current_val_rows", 0)),
                "current_test_family_rows": int(stage43_bd.get("summary", {}).get("current_test_rows", 0)),
                "support_candidate_exists_in_raw_scan": True,
                "repair_training_allowed_now": False,
                "blockers": [
                    "biwi_hotel_is_current_stage43_heldout_test_source",
                    "biwi_eth_validation_support_too_small",
                    "no_independent_biwi_like_test_source_after_rebuild",
                ],
                "recommendation": "keep_biwi_floor_only_until_independent_source_support_exists",
            }
        )
    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in records:
        dedup[(str(row["source"]), str(row["family"]))] = row
    return list(dedup.values())


def _family_readiness(blocked: list[Mapping[str, Any]], local: list[Mapping[str, Any]]) -> dict[str, Any]:
    biwi_rows = [row for row in blocked if row.get("family") == "TrajNet_biwi"]
    mot_rows = [row for row in blocked if row.get("family") == "TrajNet_mot"]
    local_mot = [row for row in local if str(row.get("support_family", "")).startswith("mot_like")]
    local_conversion_ready = [row for row in local if row.get("conversion_ready_now")]
    technical_candidates = [row for row in local if row.get("technical_support_candidate")]
    return {
        "TrajNet_biwi": {
            "status": "blocked_until_independent_biwi_like_source_available",
            "technical_candidate_count": int(sum(bool(row.get("support_candidate_exists_in_raw_scan")) for row in biwi_rows)),
            "conversion_ready_count": 0,
            "repair_training_allowed_now": False,
            "reason": "current useful biwi support would reuse the held-out biwi source; no independent source-level train/val/test story yet",
        },
        "TrajNet_mot": {
            "status": "local_topdown_candidates_exist_but_terms_and_conversion_not_closed",
            "technical_candidate_count": int(len(local_mot)),
            "technical_candidate_names": [str(row["dataset_name"]) for row in local_mot],
            "conversion_ready_count": int(sum(bool(row.get("conversion_ready_now")) for row in local_mot)),
            "repair_training_allowed_now": False,
            "reason": "PETS/Town-Center/Wild-Track are support candidates only until terms/source identity/calibration projection and guarded conversion pass",
        },
        "overall": {
            "technical_support_candidate_count": int(len(technical_candidates)),
            "conversion_ready_now_count": int(len(local_conversion_ready)),
            "repair_training_allowed_now_count": 0,
        },
    }


def build_blocked_source_support_acquisition_preflight() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage43_bc = _load(STAGE43_BC)
    stage43_bd = _load(STAGE43_BD)
    stage42_jn_cached = _load(STAGE42_JN)
    blocked_records = _blocked_family_records(stage43_bc, stage43_bd)
    local_records = [_local_candidate_record(row) for row in _local_calibrated_records()]
    readiness = _family_readiness(blocked_records, local_records)
    summary = {
        "blocked_family_count": int(len({str(row.get("family")) for row in blocked_records})),
        "local_candidate_count": int(len(local_records)),
        "local_technical_support_candidate_count": int(
            sum(bool(row["technical_support_candidate"]) for row in local_records)
        ),
        "local_conversion_ready_now_count": int(sum(bool(row["conversion_ready_now"]) for row in local_records)),
        "repair_training_allowed_now_count": 0,
        "biwi_status": readiness["TrajNet_biwi"]["status"],
        "mot_status": readiness["TrajNet_mot"]["status"],
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_support_acquisition_preflight_from_local_candidates_and_stage43_blocker_artifacts",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "input_artifacts": {
            "stage43_bc": str(STAGE43_BC),
            "stage43_bd": str(STAGE43_BD),
            "stage42_jn_cached_if_present": str(STAGE42_JN),
        },
        "input_verdicts": {
            "stage43_bc": stage43_bc.get("stage43_bc_gate", {}).get("verdict"),
            "stage43_bd": stage43_bd.get("stage43_bd_gate", {}).get("verdict"),
            "stage42_jn_cached": stage42_jn_cached.get("stage42_jn_gate", {}).get("verdict"),
        },
        "protocol": {
            "purpose": "source_support_acquisition_preflight_only",
            "conversion_executed": False,
            "training_executed": False,
            "threshold_search_executed": False,
            "test_used_for_training_or_thresholds": False,
            "technical_candidate_is_not_conversion_ready": True,
            "raw_file_counts_are_support_diagnostics": True,
        },
        "blocked_family_records": blocked_records,
        "local_source_candidates": local_records,
        "family_readiness": readiness,
        "summary": summary,
        "next_required_actions": [
            "Keep TrajNet_biwi and TrajNet_mot floor-only in deployable Stage43 policy until source-support gates clear.",
            "For biwi, acquire or locate an independent biwi-like source so train, validation, and test support are source-disjoint.",
            "For mot-like repair, record terms/source identity for PETS/Town-Center/Wild-Track before guarded conversion.",
            "After any conversion, rerun no-leakage, source-level split, strongest baseline, and Stage43 replay before model repair training.",
        ],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_or_loss_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
            "conversion_executed": False,
            "training_executed": False,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "dataset_local_raw_frame_only": True,
            "external_not_run_written_as_success": False,
            "blocked_source_repair_success_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
        "input_hash": _combined_hash([STAGE43_BC, STAGE43_BD, STAGE42_JN]),
    }
    payload["stage43_be_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    readiness = payload["family_readiness"]
    no_leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "stage43_bc_precondition_passed": payload["input_verdicts"]["stage43_bc"]
        == "stage43_bc_blocked_family_support_scan_pass",
        "stage43_bd_precondition_passed": payload["input_verdicts"]["stage43_bd"]
        == "stage43_bd_biwi_support_rebuild_preflight_pass",
        "blocked_families_loaded": summary["blocked_family_count"] >= 2,
        "local_candidate_sources_scanned": summary["local_candidate_count"] >= 3,
        "technical_candidates_separated_from_conversion_ready": summary["local_technical_support_candidate_count"]
        >= summary["local_conversion_ready_now_count"]
        and summary["local_conversion_ready_now_count"] == 0,
        "biwi_independent_support_blocker_preserved": readiness["TrajNet_biwi"]["repair_training_allowed_now"] is False
        and "independent" in readiness["TrajNet_biwi"]["reason"],
        "mot_candidate_terms_blocker_preserved": readiness["TrajNet_mot"]["repair_training_allowed_now"] is False
        and readiness["TrajNet_mot"]["technical_candidate_count"] > 0
        and readiness["TrajNet_mot"]["conversion_ready_count"] == 0,
        "repair_training_still_disallowed": summary["repair_training_allowed_now_count"] == 0,
        "next_actions_recorded": len(payload["next_required_actions"]) >= 4,
        "no_future_or_test_leakage": no_leak["future_endpoint_input"] is False
        and no_leak["future_waypoint_input"] is False
        and no_leak["future_labels_eval_or_loss_only"] is True
        and no_leak["central_velocity_input"] is False
        and no_leak["test_endpoint_goal_construction"] is False
        and no_leak["test_statistics_normalization"] is False
        and no_leak["test_threshold_tuning"] is False
        and no_leak["conversion_executed"] is False
        and no_leak["training_executed"] is False,
        "claim_boundary_not_overstated": claim["true_3d_world_model"] is False
        and claim["foundation_world_model"] is False
        and claim["metric_or_seconds_claim"] is False
        and claim["dataset_local_raw_frame_only"] is True
        and claim["external_not_run_written_as_success"] is False
        and claim["blocked_source_repair_success_claim"] is False,
        "stage5c_and_smc_false": claim["stage5c_executed"] is False and claim["smc_enabled"] is False,
        "long_objective_kept_active": claim["long_objective_complete"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_be_blocked_source_support_acquisition_preflight_pass"
        if passed == total
        else "stage43_be_blocked_source_support_acquisition_preflight_incomplete",
        "stage5c_executed": False,
        "smc_enabled": False,
        "goal_complete": False,
    }


def _render_md(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_be_gate"]
    summary = payload["summary"]
    lines = [
        "# Stage43-BE Blocked Source Support Acquisition Preflight",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- local candidates scanned: `{summary['local_candidate_count']}`",
        f"- technical support candidates: `{summary['local_technical_support_candidate_count']}`",
        f"- conversion-ready now: `{summary['local_conversion_ready_now_count']}`",
        f"- repair training allowed now: `{summary['repair_training_allowed_now_count']}`",
        "",
        "## Blocked Family Readiness",
        "",
        "| family | status | technical candidates | conversion ready | repair training now | reason |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for family in ["TrajNet_biwi", "TrajNet_mot"]:
        row = payload["family_readiness"][family]
        lines.append(
            f"| `{family}` | `{row['status']}` | {row['technical_candidate_count']} | "
            f"{row['conversion_ready_count']} | `{row['repair_training_allowed_now']}` | {row['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Local Source Candidates",
            "",
            "| dataset | family | parseable | rows | tracks | t50 | t100 | calibration files | conversion ready | blockers |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in payload["local_source_candidates"]:
        lines.append(
            f"| `{row['dataset_name']}` | `{row['support_family']}` | `{row['parseable']}` | "
            f"{row['point_rows']} | {row['agent_tracks']} | {row['t50_candidate_rows']} | "
            f"{row['t100_candidate_rows']} | {row['calibration_file_count']} | "
            f"`{row['conversion_ready_now']}` | {', '.join(row['blockers']) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is an acquisition preflight, not a conversion or training result. I found local technical candidates that could help the blocked MOT-like family after terms/source-identity/calibration checks, but none are conversion-ready now. For biwi, the useful local support is still entangled with the current held-out source, so it stays floor-only.",
            "",
            "The next legitimate move is not another selector trial. It is source support closure: confirm terms, lock source identity, run guarded conversion, rebuild source-level splits, then rerun no-leakage and baseline checks before any repair training.",
            "",
            "## Next Required Actions",
            "",
            *[f"- {item}" for item in payload["next_required_actions"]],
            "",
            "## Claim Boundary",
            "",
            "- Dataset-local/raw-frame 2.5D only.",
            "- Technical candidates are not converted benchmark evidence.",
            "- No metric or seconds-level claim.",
            "- No blocked-source repair success claim.",
            "- No Stage5C execution and no SMC.",
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
        ]
    )
    lines.extend([f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()])
    return lines


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_be_gate"]
    summary = payload["summary"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"local_candidates = `{summary['local_candidate_count']}`",
        f"technical_support_candidates = `{summary['local_technical_support_candidate_count']}`",
        f"conversion_ready_now = `{summary['local_conversion_ready_now_count']}`",
        f"repair_training_allowed_now = `{summary['repair_training_allowed_now_count']}`",
        "",
        "I checked the local source-support options for the blocked biwi/mot families. The useful takeaway is not a new model win: biwi still needs an independent held-out source before repair training, while PETS/Town-Center/Wild-Track are technical MOT-like support candidates but still need terms/source-identity/calibration closure before guarded conversion. I am keeping these sources floor-only until those support gates clear.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_be_blocked_source_support_acquisition_preflight"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "summary": summary,
        "family_readiness": payload["family_readiness"],
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_be_blocked_source_support_acquisition_preflight"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-BE",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "summary": summary,
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    write_json(WORLD_GATE_JSON, m._jsonable(payload["stage43_be_gate"]))
    lines = _render_md(payload)
    write_md(REPORT_MD, lines)
    write_md(GATE_MD, lines)
    gate = payload["stage43_be_gate"]
    write_md(
        WORLD_GATE_MD,
        [
            "# Stage43 Current World-Model Gate",
            "",
            f"- source: `{payload['source']}`",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- long objective complete: `{gate['goal_complete']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "## Current Boundary",
            "",
            "- Stage43-P / AZ remains the performance leader and exact replay artifact.",
            "- Stage43-BE found technical support candidates, but no blocked source is conversion-ready or repair-trainable now.",
            "- Blocked biwi/mot source families stay floor-only until terms/source/split gates clear.",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    _update_ledgers(payload)


def run_blocked_source_support_acquisition_preflight() -> dict[str, Any]:
    payload = build_blocked_source_support_acquisition_preflight()
    _write_outputs(payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Preflight local source-support acquisition options for Stage43 blocked source families."
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    build_arg_parser().parse_args(argv)
    payload = run_blocked_source_support_acquisition_preflight()
    gate = payload["stage43_be_gate"]
    summary = payload["summary"]
    print(f"Stage43-BE: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"local_candidates={summary['local_candidate_count']}")
    print(f"conversion_ready_now={summary['local_conversion_ready_now_count']}")
    print(f"repair_training_allowed_now={summary['repair_training_allowed_now_count']}")
    return payload


if __name__ == "__main__":
    main()
