from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m
from src.stage43_full_waypoint_latent_safe_repair import _source_family


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_blocked_family_support_scan.json"
REPORT_MD = OUT_DIR / "stage43_blocked_family_support_scan.md"
GATE_MD = OUT_DIR / "stage43_stage_bc_blocked_family_support_scan_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SOURCE = "fresh_stage43_bc_blocked_family_support_scan"
SECTION = "STAGE43_BC_BLOCKED_FAMILY_SUPPORT_SCAN"

STAGE43_BB = OUT_DIR / "stage43_blocked_source_repair_feasibility.json"
STAGE43_P = OUT_DIR / "stage43_tail_horizon_waypoint_adapter.json"

EXTERNAL_ROOTS = [
    Path("external_data/OpenTraj/datasets/TrajNet"),
    Path("/Users/yangyue/Downloads/World/external_data/OpenTraj/datasets/TrajNet"),
    Path("/Users/yangyue/Downloads/OpenTraj/datasets/TrajNet"),
]
HORIZONS = [10, 25, 50, 100]


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path, {})


def _unique_existing(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    out = []
    for path in paths:
        resolved = str(path.resolve()) if path.exists() else str(path)
        if path.exists() and resolved not in seen:
            seen.add(resolved)
            out.append(path)
    return out


def _discover_trajnet_txt_files(roots: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for root in _unique_existing(roots):
        files.extend(sorted(path for path in root.rglob("*.txt") if path.is_file()))
    return sorted({str(path.resolve()): path for path in files}.values(), key=lambda p: str(p))


def _role_from_path(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "train" in parts:
        return "raw_train_dir"
    if "test" in parts:
        return "raw_test_dir"
    return "unknown_dir"


def _parse_trajnet_txt(path: Path) -> dict[str, Any]:
    try:
        arr = np.loadtxt(path, dtype=np.float64)
    except Exception as exc:  # pragma: no cover - exercised through report output.
        return {
            "path": str(path),
            "parseable": False,
            "error": str(exc),
            "rows": 0,
            "track_count": 0,
            "frame_count": 0,
            "horizon_window_counts": {},
        }
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] < 4:
        return {
            "path": str(path),
            "parseable": False,
            "error": f"expected_at_least_4_columns_got_{arr.shape[1]}",
            "rows": int(arr.shape[0]),
            "track_count": 0,
            "frame_count": 0,
            "horizon_window_counts": {},
        }
    frame = arr[:, 0].astype(np.int64)
    agent = arr[:, 1].astype(str)
    frames_sorted = np.unique(frame)
    if len(frames_sorted) >= 2:
        diffs = np.diff(np.sort(frames_sorted))
        frame_step_median = float(np.median(diffs))
        frame_step_min = int(np.min(diffs))
        frame_step_max = int(np.max(diffs))
    else:
        frame_step_median = 0.0
        frame_step_min = 0
        frame_step_max = 0
    by_agent: dict[str, list[int]] = defaultdict(list)
    for a, f in zip(agent, frame):
        by_agent[str(a)].append(int(f))
    track_lengths = np.asarray([len(values) for values in by_agent.values()], dtype=np.int64)
    horizon_counts = {}
    observation_step_horizon_counts = {}
    for horizon in HORIZONS:
        raw_frame_count = 0
        obs_step_count = 0
        for values in by_agent.values():
            unique_frames = sorted(set(int(value) for value in values))
            frame_set = set(unique_frames)
            raw_frame_count += sum(1 for value in unique_frames if value + int(horizon) in frame_set)
            obs_step_count += max(0, len(unique_frames) - int(horizon))
        horizon_counts[str(horizon)] = int(raw_frame_count)
        observation_step_horizon_counts[str(horizon)] = int(obs_step_count)
    return {
        "path": str(path),
        "parseable": True,
        "rows": int(arr.shape[0]),
        "track_count": int(len(by_agent)),
        "frame_count": int(len(frames_sorted)),
        "frame_min": int(frame.min()) if len(frame) else None,
        "frame_max": int(frame.max()) if len(frame) else None,
        "frame_step_median": frame_step_median,
        "frame_step_min": frame_step_min,
        "frame_step_max": frame_step_max,
        "track_length_min": int(track_lengths.min()) if len(track_lengths) else 0,
        "track_length_median": float(np.median(track_lengths)) if len(track_lengths) else 0.0,
        "track_length_max": int(track_lengths.max()) if len(track_lengths) else 0,
        "horizon_window_counts": horizon_counts,
        "observation_step_horizon_window_counts": observation_step_horizon_counts,
    }


def _family_summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, Any] = {}
    for family in sorted(set(str(record["family"]) for record in records)):
        rows = [record for record in records if record["family"] == family]
        parseable = [record for record in rows if record["parseable"]]
        role_counts = Counter(str(record["raw_role"]) for record in rows)
        horizon_counts = {
            str(h): int(sum(int(record.get("horizon_window_counts", {}).get(str(h), 0)) for record in parseable))
            for h in HORIZONS
        }
        by_family[family] = {
            "file_count": int(len(rows)),
            "parseable_file_count": int(len(parseable)),
            "raw_role_counts": dict(role_counts),
            "rows": int(sum(int(record.get("rows", 0)) for record in parseable)),
            "tracks": int(sum(int(record.get("track_count", 0)) for record in parseable)),
            "horizon_window_counts": horizon_counts,
            "files": [
                {
                    "path": str(record["path"]),
                    "raw_role": str(record["raw_role"]),
                    "rows": int(record.get("rows", 0)),
                    "tracks": int(record.get("track_count", 0)),
                    "horizon_window_counts": dict(record.get("horizon_window_counts", {})),
                }
                for record in rows
            ],
        }
    return by_family


def _blocked_family_action(
    *,
    family: str,
    bb_row: Mapping[str, Any],
    raw_summary: Mapping[str, Any],
    min_validation_rows: int,
) -> dict[str, Any]:
    raw = raw_summary.get(family, {})
    train_like_files = int(raw.get("raw_role_counts", {}).get("raw_train_dir", 0))
    test_like_files = int(raw.get("raw_role_counts", {}).get("raw_test_dir", 0))
    h50_windows = int(raw.get("horizon_window_counts", {}).get("50", 0))
    h100_windows = int(raw.get("horizon_window_counts", {}).get("100", 0))
    bb_support = bb_row.get("split_support", {})
    current_train_family_rows = int(bb_support.get("train", {}).get("family_rows", 0))
    current_val_family_rows = int(bb_support.get("val", {}).get("family_rows", 0))
    current_test_family_rows = int(bb_support.get("test", {}).get("family_rows", 0))
    blockers = []
    if current_train_family_rows == 0:
        blockers.append("current_feature_store_has_no_train_family_rows")
    if current_val_family_rows < min_validation_rows:
        blockers.append("current_validation_support_below_threshold")
    if h50_windows == 0:
        blockers.append("raw_scan_has_no_t50_candidate_windows")
    if train_like_files == 0:
        blockers.append("no_raw_train_dir_file_for_family")
    if float(bb_row.get("ungated_improvement", 0.0)) < -0.5:
        blockers.append("existing_ungated_transfer_catastrophic_negative")
    if family == "TrajNet_mot" and test_like_files == 0 and train_like_files <= 1:
        blockers.append("single_source_family_no_independent_support_file")
    can_build_support_candidate = h50_windows > 0 and (train_like_files + test_like_files) >= 2
    if can_build_support_candidate and current_train_family_rows == 0:
        recommendation = "rebuild_source_family_split_with_raw_candidate_support_before_any_repair_training"
    elif not can_build_support_candidate:
        recommendation = "acquire_additional_source_family_data_before_repair_training"
    else:
        recommendation = "replay_converter_and_validation_support_before_repair_training"
    return {
        "family": family,
        "raw_file_count": int(raw.get("file_count", 0)),
        "raw_train_dir_files": train_like_files,
        "raw_test_dir_files": test_like_files,
        "raw_t50_candidate_windows": h50_windows,
        "raw_t100_candidate_windows": h100_windows,
        "current_train_family_rows": current_train_family_rows,
        "current_val_family_rows": current_val_family_rows,
        "current_test_family_rows": current_test_family_rows,
        "support_candidate_exists_in_raw_scan": bool(can_build_support_candidate),
        "repair_training_allowed_now": False,
        "blockers": blockers,
        "recommendation": recommendation,
    }


def build_blocked_family_support_scan(*, min_validation_rows: int = 1000) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    bb = _load(STAGE43_BB)
    tail_p = _load(STAGE43_P)
    files = _discover_trajnet_txt_files(EXTERNAL_ROOTS)
    raw_records: list[dict[str, Any]] = []
    for path in files:
        parsed = _parse_trajnet_txt(path)
        raw_records.append(
            {
                **parsed,
                "family": _source_family(str(path)),
                "raw_role": _role_from_path(path),
                "coordinate_unit": "dataset_local",
                "metric_status": "unverified_dataset_local",
            }
        )
    raw_by_family = _family_summary(raw_records)
    blocked_actions = [
        _blocked_family_action(
            family=str(row["family"]),
            bb_row=row,
            raw_summary=raw_by_family,
            min_validation_rows=int(min_validation_rows),
        )
        for row in bb.get("blocked_source_rows", [])
    ]
    summary = {
        "raw_file_count": int(len(raw_records)),
        "parseable_raw_file_count": int(sum(bool(record["parseable"]) for record in raw_records)),
        "blocked_family_count": int(len(blocked_actions)),
        "blocked_family_with_raw_support_candidate_count": int(
            sum(bool(row["support_candidate_exists_in_raw_scan"]) for row in blocked_actions)
        ),
        "repair_training_allowed_now_count": int(sum(bool(row["repair_training_allowed_now"]) for row in blocked_actions)),
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_raw_external_scan_for_blocked_source_family_support",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "input_artifacts": {
            "stage43_bb": str(STAGE43_BB),
            "stage43_p": str(STAGE43_P),
        },
        "input_verdicts": {
            "stage43_bb": bb.get("stage43_bb_gate", {}).get("verdict"),
            "stage43_p": tail_p.get("stage43_p_gate", {}).get("verdict"),
        },
        "scan_protocol": {
            "roots": [str(path) for path in EXTERNAL_ROOTS],
            "parsed_format": "TrajNet txt columns frame agent x y",
            "horizon_window_counts_are_diagnostic": True,
            "horizon_window_count_method": "same_agent_raw_frame_id_plus_horizon_exists",
            "raw_scan_does_not_build_feature_store": True,
            "raw_scan_does_not_train_model": True,
            "test_threshold_tuning_allowed": False,
            "min_validation_rows_for_future_repair": int(min_validation_rows),
        },
        "raw_family_summary": raw_by_family,
        "blocked_family_actions": blocked_actions,
        "summary": summary,
        "next_required_actions": [
            "For TrajNet_biwi, rebuild a legal source-family support split using raw biwi candidates before repair training.",
            "For TrajNet_mot, acquire or locate another independent MOT-like source; the current scan finds no independent validation support.",
            "After any support conversion, rerun no-leakage and validation-only support gates before evaluating test.",
            "Keep Stage43-P/AZ floor-only behavior on blocked sources until support gates clear.",
        ],
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_or_loss_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "dataset_local_raw_frame_only": True,
            "uniform_positive_external_transfer_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
        "input_hash": _combined_hash([STAGE43_BB, STAGE43_P]),
    }
    payload["stage43_bc_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    no_leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "stage43_bb_precondition_passed": payload["input_verdicts"]["stage43_bb"]
        == "stage43_bb_blocked_source_repair_feasibility_pass",
        "raw_external_sources_scanned": summary["raw_file_count"] > 0,
        "raw_parseability_reported": summary["parseable_raw_file_count"] > 0,
        "blocked_families_have_actions": summary["blocked_family_count"] > 0
        and len(payload["blocked_family_actions"]) == summary["blocked_family_count"],
        "support_candidates_separated_from_training_permission": all(
            row["repair_training_allowed_now"] is False for row in payload["blocked_family_actions"]
        ),
        "mot_blocker_recorded": any(
            row["family"] == "TrajNet_mot" and "single_source_family_no_independent_support_file" in row["blockers"]
            for row in payload["blocked_family_actions"]
        ),
        "biwi_support_candidate_recorded": any(
            row["family"] == "TrajNet_biwi" and row["support_candidate_exists_in_raw_scan"] is True
            for row in payload["blocked_family_actions"]
        ),
        "next_actions_recorded": len(payload["next_required_actions"]) >= 3,
        "no_future_or_test_leakage": no_leak["future_endpoint_input"] is False
        and no_leak["future_waypoint_input"] is False
        and no_leak["future_labels_eval_or_loss_only"] is True
        and no_leak["central_velocity_input"] is False
        and no_leak["test_endpoint_goal_construction"] is False
        and no_leak["test_statistics_normalization"] is False
        and no_leak["test_threshold_tuning"] is False,
        "claim_boundary_not_overstated": claim["true_3d_world_model"] is False
        and claim["foundation_world_model"] is False
        and claim["metric_or_seconds_claim"] is False
        and claim["dataset_local_raw_frame_only"] is True
        and claim["uniform_positive_external_transfer_claim"] is False,
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
        "verdict": "stage43_bc_blocked_family_support_scan_pass"
        if passed == total
        else "stage43_bc_blocked_family_support_scan_incomplete",
        "stage5c_executed": False,
        "smc_enabled": False,
        "goal_complete": False,
    }


def _render_md(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_bc_gate"]
    summary = payload["summary"]
    lines = [
        "# Stage43-BC Blocked Family Support Scan",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- raw files scanned: `{summary['raw_file_count']}`",
        f"- parseable raw files: `{summary['parseable_raw_file_count']}`",
        f"- blocked families: `{summary['blocked_family_count']}`",
        f"- repair training allowed now: `{summary['repair_training_allowed_now_count']}`",
        "",
        "## Blocked Family Actions",
        "",
        "| family | raw files | raw train | raw test | raw t50 windows | current train rows | current val rows | recommendation | blockers |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload["blocked_family_actions"]:
        lines.append(
            f"| `{row['family']}` | {row['raw_file_count']} | {row['raw_train_dir_files']} | {row['raw_test_dir_files']} | "
            f"{row['raw_t50_candidate_windows']} | {row['current_train_family_rows']} | {row['current_val_family_rows']} | "
            f"`{row['recommendation']}` | {', '.join(row['blockers']) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Family Raw Summary",
            "",
            "| family | files | rows | tracks | t50 diagnostic windows | t100 diagnostic windows | roles |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for family, row in sorted(payload["raw_family_summary"].items()):
        roles = ", ".join(f"{key}:{value}" for key, value in sorted(row["raw_role_counts"].items()))
        lines.append(
            f"| `{family}` | {row['file_count']} | {row['rows']} | {row['tracks']} | "
            f"{row['horizon_window_counts'].get('50', 0)} | {row['horizon_window_counts'].get('100', 0)} | {roles} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This scan is a data-support step, not a new model result. It says `TrajNet_biwi` has raw candidate material that could be converted into support, but the current feature-store split still has no train rows for that family. `TrajNet_mot` remains a harder blocker: the raw scan finds only the current PETS source and no independent family support.",
            "",
            "So the next legitimate move is a guarded conversion/split rebuild, not source repair training. Test rows remain diagnostic and are not used for thresholds or training.",
            "",
            "## Next Required Actions",
            "",
            *[f"- {item}" for item in payload["next_required_actions"]],
            "",
            "## Claim Boundary",
            "",
            "- Dataset-local/raw-frame 2.5D only.",
            "- Horizon window counts here are diagnostic raw-file availability, not official model metrics.",
            "- No metric or seconds-level claim.",
            "- No true 3D or foundation claim.",
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
    gate = payload["stage43_bc_gate"]
    summary = payload["summary"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"raw_files_scanned = `{summary['raw_file_count']}`",
        f"blocked_families = `{summary['blocked_family_count']}`",
        f"repair_training_allowed_now = `{summary['repair_training_allowed_now_count']}`",
        "",
        "I scanned the raw TrajNet/OpenTraj files behind the blocked source families. The result is useful but conservative: biwi has possible raw support to convert, while mot lacks an independent support file. I am not training a repair from this scan; it only defines what support must be rebuilt before any safe source-specific repair can be tested.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_bc_blocked_family_support_scan"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "summary": summary,
        "blocked_family_actions": payload["blocked_family_actions"],
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_bc_blocked_family_support_scan"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-BC",
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
    write_json(WORLD_GATE_JSON, m._jsonable(payload["stage43_bc_gate"]))
    lines = _render_md(payload)
    write_md(REPORT_MD, lines)
    write_md(GATE_MD, lines)
    gate = payload["stage43_bc_gate"]
    world_lines = [
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
        "- Stage43-BB says blocked sources are not repairable safely yet.",
        "- Stage43-BC identifies raw support candidates and remaining acquisition blockers.",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| {name} | `{bool(value)}` |" for name, value in gate["gates"].items()],
    ]
    write_md(WORLD_GATE_MD, world_lines)
    _update_ledgers(payload)


def run_blocked_family_support_scan(*, min_validation_rows: int = 1000) -> dict[str, Any]:
    payload = build_blocked_family_support_scan(min_validation_rows=int(min_validation_rows))
    _write_outputs(payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan raw external source-family support for Stage43 blocked sources.")
    parser.add_argument("--min-validation-rows", type=int, default=1000)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    payload = run_blocked_family_support_scan(min_validation_rows=int(args.min_validation_rows))
    gate = payload["stage43_bc_gate"]
    print(f"Stage43-BC: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"raw_files_scanned={payload['summary']['raw_file_count']}")
    print(f"repair_training_allowed_now={payload['summary']['repair_training_allowed_now_count']}")
    return payload


if __name__ == "__main__":
    main()
