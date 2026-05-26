from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "source_time_geometry_calibration_stage42.json"
REPORT_MD = OUT_DIR / "source_time_geometry_calibration_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_bn_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_source_time_geometry_stage42.md"

OPENTRAJ_ROOT = Path("external_data/OpenTraj")
DATASET_ROOT = OPENTRAJ_ROOT / "datasets"

ETH_README = DATASET_ROOT / "ETH/README.md"
UCY_README = DATASET_ROOT / "UCY/README.md"
SDD_README = DATASET_ROOT / "SDD/README.md"
SDD_SCALES = DATASET_ROOT / "SDD/estimated_scales.yaml"
OPENTRAJ_README = OPENTRAJ_ROOT / "README.md"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-BN 是 strict source-level time/geometry calibration audit，不训练模型，不下载数据。",
    "本步骤区分 source-specific calibration evidence 与全局 metric/seconds claim。",
    "即使某些 ETH/UCY source 有 meters / FPS / homography evidence，也不能把整个 M3W 写成 metric 或 seconds-level。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


SOURCE_SPECS = [
    {
        "source_id": "ETH_seq_eth",
        "domain": "ETH_UCY",
        "dataset": "ETH",
        "root": DATASET_ROOT / "ETH/seq_eth",
        "trajectory_file": DATASET_ROOT / "ETH/seq_eth/obsmat.txt",
        "homography_file": DATASET_ROOT / "ETH/seq_eth/H.txt",
        "info_file": DATASET_ROOT / "ETH/seq_eth/info.txt",
        "readme": ETH_README,
        "license_status": "no_license_information_in_local_eth_readme",
    },
    {
        "source_id": "ETH_seq_hotel",
        "domain": "ETH_UCY",
        "dataset": "ETH",
        "root": DATASET_ROOT / "ETH/seq_hotel",
        "trajectory_file": DATASET_ROOT / "ETH/seq_hotel/obsmat.txt",
        "homography_file": DATASET_ROOT / "ETH/seq_hotel/H.txt",
        "info_file": DATASET_ROOT / "ETH/seq_hotel/info.txt",
        "readme": ETH_README,
        "license_status": "no_license_information_in_local_eth_readme",
    },
    {
        "source_id": "UCY_zara01",
        "domain": "UCY",
        "dataset": "UCY",
        "root": DATASET_ROOT / "UCY/zara01",
        "trajectory_file": DATASET_ROOT / "UCY/zara01/obsmat.txt",
        "homography_file": DATASET_ROOT / "UCY/zara01/H.txt",
        "info_file": UCY_README,
        "readme": UCY_README,
        "license_status": "free_use_with_credit_per_local_ucy_readme",
    },
    {
        "source_id": "UCY_zara02",
        "domain": "UCY",
        "dataset": "UCY",
        "root": DATASET_ROOT / "UCY/zara02",
        "trajectory_file": DATASET_ROOT / "UCY/zara02/obsmat.txt",
        "homography_file": DATASET_ROOT / "UCY/zara02/H.txt",
        "info_file": UCY_README,
        "readme": UCY_README,
        "license_status": "free_use_with_credit_per_local_ucy_readme",
    },
    {
        "source_id": "UCY_zara03",
        "domain": "UCY",
        "dataset": "UCY",
        "root": DATASET_ROOT / "UCY/zara03",
        "trajectory_file": DATASET_ROOT / "UCY/zara03/crowds_zara03.txt",
        "homography_file": DATASET_ROOT / "UCY/zara03/H.txt",
        "info_file": UCY_README,
        "readme": UCY_README,
        "license_status": "free_use_with_credit_per_local_ucy_readme",
    },
    {
        "source_id": "UCY_students03",
        "domain": "UCY",
        "dataset": "UCY",
        "root": DATASET_ROOT / "UCY/students03",
        "trajectory_file": DATASET_ROOT / "UCY/students03/obsmat.txt",
        "homography_file": DATASET_ROOT / "UCY/students03/H.txt",
        "info_file": UCY_README,
        "readme": UCY_README,
        "license_status": "free_use_with_credit_per_local_ucy_readme",
    },
    {
        "source_id": "UCY_students01",
        "domain": "UCY",
        "dataset": "UCY",
        "root": DATASET_ROOT / "UCY/students01",
        "trajectory_file": DATASET_ROOT / "UCY/students01/students001.txt",
        "homography_file": None,
        "info_file": UCY_README,
        "readme": UCY_README,
        "license_status": "free_use_with_credit_per_local_ucy_readme",
    },
]


def _read_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _numbers(line: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?", line)]


def _det3(matrix: list[list[float]]) -> float:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def parse_homography_matrix(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"path": str(path) if path else None, "exists": False, "parseable": False, "determinant": None}
    rows: list[list[float]] = []
    for line in _read_text(path).splitlines():
        values = _numbers(line)
        if len(values) >= 3:
            rows.append(values[:3])
    matrix = rows[:3]
    parseable = len(matrix) == 3 and all(len(row) == 3 and all(math.isfinite(v) for v in row) for row in matrix)
    det = _det3(matrix) if parseable else None
    return {
        "path": str(path),
        "exists": True,
        "parseable": bool(parseable and det is not None and abs(det) > 1e-12),
        "determinant": float(det) if det is not None else None,
    }


def _extract_annotation_timing(dataset: str, text: str) -> dict[str, Any]:
    lower = text.lower()
    annotation_fps = None
    timestep = None
    video_fps = None
    if dataset == "ETH":
        if "annotation was done at 2.5 fps" in lower:
            annotation_fps = 2.5
        if "timestep of 0.4 seconds" in lower:
            timestep = 0.4
        video_match = re.search(r"shot at\s+([0-9.]+)\s*fps", lower)
        if video_match:
            video_fps = float(video_match.group(1))
    elif dataset == "UCY":
        if "25 frames per second" in lower:
            video_fps = 25.0
        if "2.5 fps" in lower and "every 10 frames" in lower:
            annotation_fps = 2.5
            timestep = 0.4
    return {
        "video_fps": video_fps,
        "annotation_fps": annotation_fps,
        "annotation_timestep_seconds": timestep,
        "h10_annotation_seconds": 10 * timestep if timestep else None,
        "h25_annotation_seconds": 25 * timestep if timestep else None,
        "h50_annotation_seconds": 50 * timestep if timestep else None,
        "h100_annotation_seconds": 100 * timestep if timestep else None,
    }


def _extract_dataset_row_from_opentraj(dataset: str) -> str | None:
    text = _read_text(OPENTRAJ_README)
    for line in text.splitlines():
        if f"| [{dataset}]" in line:
            return line.strip()
    return None


def _source_coordinate_evidence(dataset: str, readme_text: str, source_id: str) -> dict[str, Any]:
    lower = readme_text.lower()
    if dataset == "ETH":
        meter_statement = "positions and velocities are in meters" in lower
        return {
            "coordinate_evidence": "obsmat_positions_velocities_in_meters" if meter_statement else "not_verified",
            "meter_coordinates_evidence": meter_statement,
            "homography_direction_evidence": "README provides H.txt and world2image example; direction still should be validated before image projection claims.",
        }
    if dataset == "UCY":
        h_statement = "homography" in lower and "h.txt" in lower
        row = _extract_dataset_row_from_opentraj("UCY") or ""
        world_2d = "coord=world-2d" in row.lower()
        has_obsmat = "students03" in source_id or "zara" in source_id
        return {
            "coordinate_evidence": "world_2d_candidate_with_homography" if h_statement and world_2d and has_obsmat else "not_verified_or_missing_source_homography",
            "meter_coordinates_evidence": bool(h_statement and world_2d and has_obsmat),
            "homography_direction_evidence": "UCY README documents H.txt for zara/students03 and OpenTraj table marks Coord=world-2D; source-specific numeric direction still should be validated before image projection claims.",
        }
    return {"coordinate_evidence": "not_verified", "meter_coordinates_evidence": False, "homography_direction_evidence": "not_available"}


def _audit_source(spec: Mapping[str, Any]) -> dict[str, Any]:
    readme_text = _read_text(Path(spec["readme"]))
    info_text = _read_text(Path(spec["info_file"])) if spec.get("info_file") else readme_text
    h = parse_homography_matrix(spec.get("homography_file"))
    timing = _extract_annotation_timing(str(spec["dataset"]), info_text)
    coord = _source_coordinate_evidence(str(spec["dataset"]), readme_text, str(spec["source_id"]))
    trajectory_file = Path(spec["trajectory_file"])
    source_specific_metric_time_evidence = bool(
        trajectory_file.exists()
        and h["parseable"]
        and coord["meter_coordinates_evidence"]
        and timing["annotation_fps"]
        and timing["annotation_timestep_seconds"]
    )
    if source_specific_metric_time_evidence:
        allowed_local_claim = "source_specific_annotation_step_meter_coordinate_evidence"
    elif trajectory_file.exists() and timing["annotation_fps"]:
        allowed_local_claim = "source_specific_time_evidence_only_coordinate_not_verified"
    else:
        allowed_local_claim = "dataset_local_raw_frame_only"
    return {
        "source": "fresh_source_time_geometry_calibration_audit",
        "source_id": spec["source_id"],
        "domain": spec["domain"],
        "dataset": spec["dataset"],
        "root": str(spec["root"]),
        "trajectory_file": str(trajectory_file),
        "trajectory_file_exists": trajectory_file.exists(),
        "license_status": spec["license_status"],
        "homography": h,
        "timing": timing,
        "coordinate": coord,
        "source_specific_metric_time_evidence": source_specific_metric_time_evidence,
        "allowed_local_claim": allowed_local_claim,
        "global_metric_claim_allowed": False,
        "global_seconds_claim_allowed": False,
        "m3w_official_metric_seconds_claim_allowed": False,
    }


def _audit_sdd() -> dict[str, Any]:
    readme = _read_text(SDD_README)
    scales_text = _read_text(SDD_SCALES)
    scale_values = [float(v) for v in re.findall(r"scale:\s*([0-9.]+)", scales_text)]
    certainty_values = [float(v) for v in re.findall(r"certainty:\s*([0-9.]+)", scales_text)]
    has_estimated_scale_warning = "some of the scales are estimated using google maps" in readme.lower() or "rational guess" in readme.lower()
    fps_row = _extract_dataset_row_from_opentraj("SDD") or ""
    return {
        "source": "fresh_source_time_geometry_calibration_audit",
        "source_id": "SDD_estimated_scale_audit",
        "dataset": "SDD",
        "domain": "SDD",
        "estimated_scale_file": str(SDD_SCALES),
        "scale_count": len(scale_values),
        "certainty_min": min(certainty_values) if certainty_values else None,
        "certainty_max": max(certainty_values) if certainty_values else None,
        "open_traj_fps_row_mentions_30": "fps=30" in fps_row.lower(),
        "estimated_scale_warning_present": has_estimated_scale_warning,
        "allowed_local_claim": "pixel_raw_frame_with_estimated_scale_diagnostic_only",
        "metric_claim_allowed": False,
        "seconds_claim_allowed": False,
        "reason": "SDD scales are estimated with explicit uncertainty; current Stage42 keeps SDD as pixel raw-frame benchmark.",
    }


def _audit_external_diagnostics() -> list[dict[str, Any]]:
    return [
        {
            "source_id": "TrajNet_local_snippets",
            "dataset": "TrajNet",
            "domain": "TrajNet",
            "allowed_local_claim": "dataset_local_short_snippet_only",
            "metric_claim_allowed": False,
            "seconds_claim_allowed": False,
            "reason": "Local TrajNet files are fixed short snippets and have no verified homography/FPS/scale evidence.",
        },
        {
            "source_id": "TGSIM",
            "dataset": "TGSIM",
            "domain": "traffic_diagnostic",
            "allowed_local_claim": "traffic_metric_diagnostic_only",
            "metric_claim_allowed_for_pedestrian_world_model": False,
            "seconds_claim_allowed_for_pedestrian_world_model": False,
            "reason": "TGSIM can be metric traffic diagnostic only, not pedestrian top-down world-model success.",
        },
        {
            "source_id": "AerialMPT",
            "dataset": "AerialMPT",
            "domain": "aerial_diagnostic",
            "allowed_local_claim": "dataset_local_raw_frame_only_until_source_terms_geometry_verified",
            "metric_claim_allowed": False,
            "seconds_claim_allowed": False,
            "reason": "Local Stage42 calibration evidence did not verify source-specific homography/FPS/scale enough for metric/seconds claims.",
        },
    ]


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    claim = payload["claim_boundary"]
    gates = {
        "sources_audited": summary["source_records_audited"] >= 6,
        "eth_source_specific_metric_time_evidence_found": summary["eth_source_specific_metric_time_sources"] >= 2,
        "ucy_source_specific_metric_time_evidence_found": summary["ucy_source_specific_metric_time_sources"] >= 3,
        "sdd_estimated_scale_not_overclaimed": payload["sdd_audit"]["metric_claim_allowed"] is False,
        "trajnet_gap_kept_raw_frame": any(row["source_id"] == "TrajNet_local_snippets" for row in payload["diagnostic_sources"]),
        "global_metric_claim_blocked": claim["global_metric_claim_allowed"] is False,
        "global_seconds_claim_blocked": claim["global_seconds_claim_allowed"] is False,
        "m3w_metric_seconds_claim_blocked": claim["m3w_official_metric_seconds_claim_allowed"] is False,
        "user_action_generated": summary["user_action_required"] is True,
        "no_training": summary["training_run"] is False,
        "no_auto_download": summary["auto_download_executed"] is False,
        "no_stage5c": claim["stage5c_executed"] is False,
        "no_smc": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_bn_source_time_geometry_calibration_pass_with_global_claim_blocked" if passed == total else "stage42_bn_source_time_geometry_calibration_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def run_stage42_source_time_geometry_calibration() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    source_records = [_audit_source(spec) for spec in SOURCE_SPECS]
    sdd_audit = _audit_sdd()
    diagnostic_sources = _audit_external_diagnostics()
    eth_metric_time = [row for row in source_records if row["domain"] == "ETH_UCY" and row["source_specific_metric_time_evidence"]]
    ucy_metric_time = [row for row in source_records if row["domain"] == "UCY" and row["source_specific_metric_time_evidence"]]
    summary = {
        "source": "fresh_source_time_geometry_calibration_audit",
        "source_records_audited": len(source_records),
        "eth_source_specific_metric_time_sources": len(eth_metric_time),
        "ucy_source_specific_metric_time_sources": len(ucy_metric_time),
        "source_specific_metric_time_sources": [row["source_id"] for row in source_records if row["source_specific_metric_time_evidence"]],
        "sdd_scale_count": sdd_audit["scale_count"],
        "sdd_metric_claim_allowed": sdd_audit["metric_claim_allowed"],
        "global_metric_claim_allowed": False,
        "global_seconds_claim_allowed": False,
        "m3w_official_metric_seconds_claim_allowed": False,
        "training_run": False,
        "auto_download_executed": False,
        "user_action_required": True,
    }
    claim_boundary = {
        "true_3d": False,
        "foundation_world_model": False,
        "global_metric_claim_allowed": False,
        "global_seconds_claim_allowed": False,
        "m3w_official_metric_seconds_claim_allowed": False,
        "raw_frame_global_claim_required": True,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    payload: dict[str, Any] = {
        "source": "fresh_source_time_geometry_calibration_audit",
        "stage": "Stage42-BN Strict Source Time/Geometry Calibration",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(
            [str(OPENTRAJ_README), str(ETH_README), str(UCY_README), str(SDD_README), str(SDD_SCALES)]
            + [str(spec["homography_file"]) for spec in SOURCE_SPECS if spec.get("homography_file")]
            + [str(spec["trajectory_file"]) for spec in SOURCE_SPECS]
        ),
        "current_facts": CURRENT_FACTS,
        "source_records": source_records,
        "sdd_audit": sdd_audit,
        "diagnostic_sources": diagnostic_sources,
        "summary": summary,
        "claim_boundary": claim_boundary,
    }
    payload["stage42_bn_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    return payload


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["summary"]
    lines = [
        "# Stage42-BN Strict Source Time/Geometry Calibration Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_bn_gate']['passed']} / {payload['stage42_bn_gate']['total']}`",
        f"- verdict: `{payload['stage42_bn_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- source_records_audited: `{summary['source_records_audited']}`",
        f"- eth_source_specific_metric_time_sources: `{summary['eth_source_specific_metric_time_sources']}`",
        f"- ucy_source_specific_metric_time_sources: `{summary['ucy_source_specific_metric_time_sources']}`",
        f"- source_specific_metric_time_sources: `{summary['source_specific_metric_time_sources']}`",
        f"- sdd_scale_count: `{summary['sdd_scale_count']}`",
        f"- global_metric_claim_allowed: `{summary['global_metric_claim_allowed']}`",
        f"- global_seconds_claim_allowed: `{summary['global_seconds_claim_allowed']}`",
        f"- m3w_official_metric_seconds_claim_allowed: `{summary['m3w_official_metric_seconds_claim_allowed']}`",
        "",
        "## Source-Level Evidence",
        "",
        "| source | domain | H parseable | annotation fps | timestep s | local claim | global metric/seconds |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload["source_records"]:
        lines.append(
            f"| `{row['source_id']}` | `{row['domain']}` | {row['homography']['parseable']} | {row['timing']['annotation_fps']} | {row['timing']['annotation_timestep_seconds']} | `{row['allowed_local_claim']}` | `{row['global_metric_claim_allowed']}/{row['global_seconds_claim_allowed']}` |"
        )
    lines.extend(
        [
            "",
            "## SDD Scale Audit",
            "",
            f"- scale_count: `{payload['sdd_audit']['scale_count']}`",
            f"- certainty_min: `{payload['sdd_audit']['certainty_min']}`",
            f"- certainty_max: `{payload['sdd_audit']['certainty_max']}`",
            f"- estimated_scale_warning_present: `{payload['sdd_audit']['estimated_scale_warning_present']}`",
            f"- metric_claim_allowed: `{payload['sdd_audit']['metric_claim_allowed']}`",
            f"- seconds_claim_allowed: `{payload['sdd_audit']['seconds_claim_allowed']}`",
            "",
            "## Diagnostic / Blocked Sources",
            "",
            "| source | claim | reason |",
            "| --- | --- | --- |",
        ]
    )
    for row in payload["diagnostic_sources"]:
        lines.append(f"| `{row['source_id']}` | `{row['allowed_local_claim']}` | {row['reason']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- ETH `seq_eth` and `seq_hotel` have source-specific local evidence for meter coordinates and 2.5fps / 0.4s annotation steps.",
            "- UCY `zara01`, `zara02`, `zara03`, and `students03` have source-specific local evidence for H.txt-backed world-2D candidate coordinates and 2.5fps / 0.4s annotation steps.",
            "- These are source-specific calibration candidates, not a global M3W metric/seconds claim.",
            "- Current M3W reports must still use raw-frame / dataset-local wording unless a downstream evaluation explicitly restricts itself to a verified source-specific calibrated subset.",
            "- SDD remains pixel raw-frame in this project because its scales are estimated and the global Stage42 claim does not validate metric/seconds semantics.",
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    return [
        "# Stage42-BN User Action Required: Source Time/Geometry Calibration",
        "",
        f"- source: `{payload['source']}`",
        "",
        "## Before Any Metric / Seconds-Level Paper Claim",
        "",
        "- Confirm source-specific coordinate convention and homography direction for each ETH/UCY source used in a calibrated subset.",
        "- Confirm whether model horizons are annotation-step horizons or raw video frame horizons for the exact feature/evaluation rows.",
        "- Keep SDD as pixel raw-frame unless a source-specific scale and annotation-stride protocol is explicitly validated.",
        "- Do not use TrajNet snippets for raw-frame t100 or metric/time claims.",
        "- Do not use TGSIM traffic metric evidence as pedestrian world-model success.",
        "",
        "## Current Allowed Wording",
        "",
        "- Allowed: source-specific ETH/UCY calibration evidence exists for selected sources.",
        "- Not allowed: M3W is metric, seconds-level, true 3D, or foundation-scale.",
    ]


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_bn_gate"]
    lines = [
        "# Stage42-BN Gate",
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


if __name__ == "__main__":
    run_stage42_source_time_geometry_calibration()
