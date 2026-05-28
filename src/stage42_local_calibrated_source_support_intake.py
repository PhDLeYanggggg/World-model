from __future__ import annotations

import csv
import json
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "local_calibrated_source_support_intake_stage42.json"
REPORT_MD = OUT_DIR / "local_calibrated_source_support_intake_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_jn_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_local_calibrated_sources_stage42.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
LEDGER = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE42_JN_LOCAL_CALIBRATED_SOURCE_SUPPORT_INTAKE"
SOURCE = "fresh_stage42_jn_local_calibrated_source_support_intake"

OPENTRAJ = Path("external_data/OpenTraj/datasets")

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-JN searches local OpenTraj-style data for calibrated/source-support candidates after JM left ETH/UCY blocked.",
    "This is parseability and support-readiness evidence, not conversion into the deployable benchmark.",
    "Local calibration files and metric hints do not override license/terms or claim-boundary blockers.",
    "No internet scraping, no automatic download, no future endpoint input, no central velocity, no Stage5C, and no SMC are used.",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _town_center_points(path: Path) -> list[tuple[str, int, int, float, float]]:
    points: list[tuple[str, int, int, float, float]] = []
    if not path.exists():
        return points
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 12:
                continue
            try:
                agent = int(float(row[0]))
                frame = int(float(row[1]))
                body_valid = int(float(row[3])) == 1
                if not body_valid:
                    continue
                left, top, right, bottom = (float(row[8]), float(row[9]), float(row[10]), float(row[11]))
            except ValueError:
                continue
            x = 0.5 * (left + right)
            y = bottom
            points.append(("Town-Center", agent, frame, x, y))
    return points


def _wildtrack_xy(position_id: int) -> tuple[float, float]:
    return -3.0 + 0.025 * (position_id % 480), -9.0 + 0.025 * (position_id // 480)


def _wildtrack_points(root: Path) -> list[tuple[str, int, int, float, float]]:
    points: list[tuple[str, int, int, float, float]] = []
    if not root.exists():
        return points
    for path in sorted(root.glob("*.json")):
        try:
            frame = int(path.stem)
            rows = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            try:
                agent = int(row["personID"])
                x, y = _wildtrack_xy(int(row["positionID"]))
            except Exception:
                continue
            points.append(("Wild-Track", agent, frame, x, y))
    return points


def _pets_points(path: Path) -> list[tuple[str, int, int, float, float]]:
    points: list[tuple[str, int, int, float, float]] = []
    if not path.exists():
        return points
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return points
    for frame_el in root.findall(".//frame"):
        try:
            frame = int(frame_el.attrib["number"])
        except Exception:
            continue
        for obj in frame_el.findall(".//object"):
            try:
                agent = int(obj.attrib["id"])
                box = obj.find("box")
                if box is None:
                    continue
                x = float(box.attrib["xc"])
                y = float(box.attrib["yc"]) + 0.5 * float(box.attrib.get("h", 0.0))
            except Exception:
                continue
            points.append(("PETS-2009", agent, frame, x, y))
    return points


def _track_stats(points: Iterable[tuple[str, int, int, float, float]]) -> dict[str, Any]:
    tracks: dict[tuple[str, int], list[tuple[int, float, float]]] = defaultdict(list)
    for source, agent, frame, x, y in points:
        tracks[(source, agent)].append((int(frame), float(x), float(y)))
    lengths = []
    frame_values = []
    horizon_counts = {10: 0, 25: 0, 50: 0, 100: 0}
    displacement = []
    for rows in tracks.values():
        rows = sorted(rows, key=lambda item: item[0])
        n = len(rows)
        lengths.append(n)
        frame_values.extend([frame for frame, _x, _y in rows])
        for h in horizon_counts:
            horizon_counts[h] += max(0, n - h)
        if n >= 2:
            x0, y0 = rows[0][1], rows[0][2]
            x1, y1 = rows[-1][1], rows[-1][2]
            displacement.append(float(np.hypot(x1 - x0, y1 - y0)))
    lengths_arr = np.asarray(lengths, dtype=np.float64) if lengths else np.asarray([], dtype=np.float64)
    return {
        "point_rows": int(sum(lengths)),
        "agent_tracks": int(len(lengths)),
        "frame_count": int(len(set(frame_values))),
        "frame_min": int(min(frame_values)) if frame_values else None,
        "frame_max": int(max(frame_values)) if frame_values else None,
        "track_length_min": int(np.min(lengths_arr)) if len(lengths_arr) else 0,
        "track_length_median": float(np.median(lengths_arr)) if len(lengths_arr) else 0.0,
        "track_length_max": int(np.max(lengths_arr)) if len(lengths_arr) else 0,
        "t10_rows": int(horizon_counts[10]),
        "t25_rows": int(horizon_counts[25]),
        "t50_rows": int(horizon_counts[50]),
        "t100_rows": int(horizon_counts[100]),
        "median_track_displacement": float(np.median(displacement)) if displacement else 0.0,
    }


def _candidate_records() -> list[dict[str, Any]]:
    town_root = OPENTRAJ / "Town-Center"
    wild_root = OPENTRAJ / "Wild-Track"
    pets_root = OPENTRAJ / "PETS-2009/data"
    candidates = [
        {
            "dataset_name": "Town-Center",
            "domain": "topdown_fixed_camera_pedestrian",
            "root": town_root,
            "points": _town_center_points(town_root / "TownCentre-groundtruth-top.txt"),
            "calibration_files": [town_root / "TownCentre-calibration-ci.txt"],
            "readme": town_root / "README.md",
            "coordinate_unit": "image_pixel_bbox_bottom_center",
            "metric_status": "calibration_file_present_but_world_projection_not_integrated",
            "time_status": "README states 25fps and 4500 labeled frames",
            "license_status": "no_license_information_available_in_local_readme",
        },
        {
            "dataset_name": "Wild-Track",
            "domain": "topdown_multicamera_pedestrian",
            "root": wild_root,
            "points": _wildtrack_points(wild_root / "annotations_positions"),
            "calibration_files": sorted((wild_root / "calibrations").glob("*/*.xml"))[:16],
            "readme": wild_root / "README.md",
            "coordinate_unit": "ground_grid_meter_candidate_2.5cm_spacing",
            "metric_status": "README documents 2.5cm ground grid and camera calibration files",
            "time_status": "frame index available; fps not asserted by this intake",
            "license_status": "no_license_information_available_in_local_readme",
        },
        {
            "dataset_name": "PETS-2009-S2L1",
            "domain": "fixed_camera_pedestrian",
            "root": pets_root,
            "points": _pets_points(pets_root / "annotations/PETS2009-S2L1.xml"),
            "calibration_files": sorted((pets_root / "calibration").glob("*.xml")),
            "readme": pets_root / "README.md",
            "coordinate_unit": "image_pixel_bbox_bottom_center",
            "metric_status": "camera_calibration_files_present_but_ground_projection_not_integrated",
            "time_status": "frame index available; fps not asserted by this intake",
            "license_status": "no_license_information_available_in_local_readme",
        },
    ]
    records = []
    for candidate in candidates:
        stats = _track_stats(candidate.pop("points"))
        calibration_files = [str(path) for path in candidate.pop("calibration_files") if path.exists()]
        readme = candidate.pop("readme")
        root = candidate.pop("root")
        parseable = stats["point_rows"] > 0 and stats["agent_tracks"] > 0
        long_horizon_support = stats["t50_rows"] > 0 or stats["t100_rows"] > 0
        legal_auto_convert_allowed = "no_license_information" not in candidate["license_status"]
        records.append(
            {
                "source": "fresh_run",
                **candidate,
                "root": str(root),
                "root_exists": root.exists(),
                "readme_exists": readme.exists(),
                "readme_license_excerpt": _license_excerpt(_read_text(readme)),
                "calibration_file_count": len(calibration_files),
                "calibration_files_sample": calibration_files[:6],
                "parseable": parseable,
                "stats": stats,
                "long_horizon_support": long_horizon_support,
                "legal_auto_convert_allowed": legal_auto_convert_allowed,
                "usable_next_role": "user_action_required_before_conversion" if not legal_auto_convert_allowed else "candidate_support_conversion",
                "can_help_blocked_eth_ucy": bool(parseable and long_horizon_support),
                "conversion_status": "not_converted_license_or_projection_guard",
                "claim_boundary": "support-candidate only; not deployed, not official, no global metric/seconds claim",
            }
        )
    return records


def _license_excerpt(text: str) -> str:
    if not text:
        return "no_readme_text"
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if "license" in line.lower():
            return " ".join(lines[idx : idx + 3]).strip()[:300]
    return "no_license_section_found"


def _user_actions(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    actions = []
    for row in records:
        if row["legal_auto_convert_allowed"]:
            continue
        actions.append(
            {
                "dataset_name": row["dataset_name"],
                "local_path": row["root"],
                "action": "confirm_dataset_terms_or_provide_official_license_before_conversion",
                "reason": "Local files are parseable and may add source-support coverage, but local README/license information is insufficient for automatic benchmark conversion.",
                "intended_use": "source-support diagnostics for ETH/UCY blocked sources; not global metric/seconds or foundation claim",
            }
        )
    return actions


def _summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    parseable = [row["dataset_name"] for row in records if row["parseable"]]
    long_horizon = [row["dataset_name"] for row in records if row["long_horizon_support"]]
    can_help = [row["dataset_name"] for row in records if row["can_help_blocked_eth_ucy"]]
    auto_convert = [row["dataset_name"] for row in records if row["legal_auto_convert_allowed"]]
    decision = "candidate_sources_found_but_user_terms_required" if can_help and not auto_convert else "candidate_sources_ready_for_guarded_conversion" if auto_convert else "no_local_candidate_support_source"
    return {
        "source": SOURCE,
        "candidate_count": len(records),
        "parseable_candidates": parseable,
        "long_horizon_candidates": long_horizon,
        "can_help_blocked_eth_ucy": can_help,
        "auto_convert_allowed": auto_convert,
        "decision": decision,
        "next_action": "Ask for/record official terms for parseable candidate sources, then run guarded conversion/no-leakage before using them as support.",
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    records = _candidate_records()
    payload: dict[str, Any] = {
        "stage": "Stage42-JN",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(
            [
                "external_data/OpenTraj/datasets/Town-Center/TownCentre-groundtruth-top.txt",
                "external_data/OpenTraj/datasets/Wild-Track/annotations_positions",
                "external_data/OpenTraj/datasets/PETS-2009/data/annotations/PETS2009-S2L1.xml",
            ]
        ),
        "current_facts": CURRENT_FACTS,
        "records": records,
        "summary": _summary(records),
        "user_action_required": _user_actions(records),
        "user_action_required_written": True,
        "no_leakage": {
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "conversion_executed": False,
            "support_intake_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "global_metric_or_seconds_claim": False,
            "raw_frame_dataset_local_main_claim": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_jn_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    records = payload["records"]
    gates = {
        "local_candidates_audited": len(records) >= 3,
        "parseability_checked": all("parseable" in row for row in records),
        "calibration_files_checked": all("calibration_file_count" in row for row in records),
        "horizon_support_estimated": all("t50_rows" in row["stats"] and "t100_rows" in row["stats"] for row in records),
        "support_candidates_identified": len(payload["summary"]["can_help_blocked_eth_ucy"]) > 0,
        "legal_guard_applied": all(
            row["legal_auto_convert_allowed"] or row["usable_next_role"] == "user_action_required_before_conversion"
            for row in records
        ),
        "user_action_written": payload["user_action_required_written"] and len(payload["user_action_required"]) > 0,
        "no_conversion_overclaim": all(row["conversion_status"] != "converted" for row in records),
        "no_leakage_pass": all(
            payload["no_leakage"][key] is False
            for key in ["future_endpoint_input", "central_velocity", "test_endpoint_goals", "test_threshold_tuning", "conversion_executed"]
        )
        and payload["no_leakage"]["support_intake_only"],
        "no_global_metric_seconds_overclaim": payload["claim_boundary"]["global_metric_or_seconds_claim"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    verdict = "stage42_jn_local_calibrated_source_support_intake_pass" if passed == len(gates) else "stage42_jn_local_calibrated_source_support_intake_partial"
    return {"source": "fresh_run", "gates": gates, "passed": passed, "total": len(gates), "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jn_gate"]
    summary = payload["summary"]
    lines = [
        "# Stage42-JN Local Calibrated Source Support Intake",
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
        f"- decision: `{summary['decision']}`",
        f"- parseable_candidates: `{summary['parseable_candidates']}`",
        f"- long_horizon_candidates: `{summary['long_horizon_candidates']}`",
        f"- can_help_blocked_eth_ucy: `{summary['can_help_blocked_eth_ucy']}`",
        f"- auto_convert_allowed: `{summary['auto_convert_allowed']}`",
        f"- next_action: {summary['next_action']}",
        "",
        "## Candidate Sources",
        "",
        "| dataset | parseable | rows | agents | t50 | t100 | calibration files | metric status | legal auto-convert | role |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for row in payload["records"]:
        stats = row["stats"]
        lines.append(
            f"| `{row['dataset_name']}` | `{row['parseable']}` | {stats['point_rows']} | {stats['agent_tracks']} | "
            f"{stats['t50_rows']} | {stats['t100_rows']} | {row['calibration_file_count']} | `{row['metric_status']}` | "
            f"`{row['legal_auto_convert_allowed']}` | `{row['usable_next_role']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Local candidates with calibration or ground-coordinate hints exist, but this stage intentionally does not convert them into the deployable benchmark.",
            "- The immediate value is source-support coverage for ETH/UCY blocked sources after terms/license confirmation and guarded conversion.",
            "- Any metric/time claim must remain source-specific and restricted until conversion, no-leakage, and evaluation pass.",
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{payload['no_leakage']}`",
            f"- claim_boundary: `{payload['claim_boundary']}`",
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Local Calibrated Source Support",
        "",
        "These sources are local and parseable enough for support-candidate planning, but they are not converted or used as benchmark evidence until terms/license and source identity are confirmed.",
        "",
    ]
    for action in payload["user_action_required"]:
        lines.extend(
            [
                f"## {action['dataset_name']}",
                "",
                f"- local_path: `{action['local_path']}`",
                f"- action: `{action['action']}`",
                f"- reason: {action['reason']}",
                f"- intended_use: {action['intended_use']}",
                "",
            ]
        )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jn_gate"]
    lines = [
        "# Stage42-JN Gate",
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


def _section_lines(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["summary"]
    gate = payload["stage42_jn_gate"]
    return [
        "## Stage42-JN Local Calibrated Source Support Intake",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict: `{gate['verdict']}`",
        f"- parseable support candidates: `{summary['parseable_candidates']}`; long-horizon candidates: `{summary['long_horizon_candidates']}`.",
        f"- decision: `{summary['decision']}`; auto_convert_allowed: `{summary['auto_convert_allowed']}`.",
        "- boundary: candidate-source intake only; no conversion, no deployment claim, no global metric/seconds claim, no Stage5C, no SMC.",
    ]


def _update_readmes(payload: Mapping[str, Any]) -> None:
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, _section_lines(payload))


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    stage42 = state.setdefault("stage42", {})
    stage42["local_calibrated_source_support_intake"] = {
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "verdict": payload["stage42_jn_gate"]["verdict"],
        "gate": {"passed": payload["stage42_jn_gate"]["passed"], "total": payload["stage42_jn_gate"]["total"]},
        "decision": payload["summary"]["decision"],
        "parseable_candidates": payload["summary"]["parseable_candidates"],
        "long_horizon_candidates": payload["summary"]["long_horizon_candidates"],
        "auto_convert_allowed": payload["summary"]["auto_convert_allowed"],
        "global_metric_or_seconds_claim": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["last_updated_utc"] = payload["generated_at_utc"]
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(LEDGER.parent)
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "stage": "Stage42-JN",
                    "source": payload["source"],
                    "generated_at_utc": payload["generated_at_utc"],
                    "verdict": payload["stage42_jn_gate"]["verdict"],
                    "result": payload["summary"]["decision"],
                    "fresh_run": True,
                    "downloaded": False,
                    "converted": False,
                    "trained": False,
                    "evaluated": True,
                    "stage5c_executed": False,
                    "smc_enabled": False,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def run_stage42_local_calibrated_source_support_intake(*, refresh_readmes: bool = True) -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    if refresh_readmes:
        _update_readmes(payload)
        _update_state(payload)
        _append_ledger(payload)
    return payload


def main() -> None:
    run_stage42_local_calibrated_source_support_intake(refresh_readmes=True)


if __name__ == "__main__":
    main()
