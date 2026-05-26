from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_data_calibration import CURRENT_FACTS, DATASET_SPECS, OUT_DIR


MAX_SCAN_FILES_PER_DATASET = 1200
MAX_TEXT_BYTES = 256 * 1024

ALWAYS_EVIDENCE_SUFFIXES = {".json", ".md", ".yaml", ".yml"}
EVIDENCE_NAME_TERMS = {
    "homography",
    "calib",
    "calibration",
    "scale",
    "meter",
    "fps",
    "frame_rate",
    "stride",
    "metadata",
    "readme",
    "info",
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _hash_paths(paths: Iterable[str | Path]) -> str:
    h = hashlib.sha256()
    for raw in sorted({str(p) for p in paths}):
        path = Path(raw)
        h.update(str(path).encode("utf-8"))
        h.update(b"\0")
        if path.exists():
            stat = path.stat()
            h.update(str(stat.st_size).encode("ascii"))
            h.update(str(stat.st_mtime_ns).encode("ascii"))
        else:
            h.update(b"missing")
    return h.hexdigest()


def _contains_name_term(path: Path) -> bool:
    lower = path.name.lower()
    return any(term in lower for term in EVIDENCE_NAME_TERMS)


def _is_homography_evidence_file(path: Path) -> bool:
    lower_name = path.name.lower()
    lower_parts = {part.lower() for part in path.parts}
    return (
        lower_name in {"h.txt", "h-old.txt", "h-cam.txt"}
        or re.fullmatch(r"h(?:[-_][a-z0-9]+)?\.txt", lower_name) is not None
        or "homography" in lower_name
        or "calib" in lower_name
        or "calibration" in lower_name
        or bool({"calib", "calibration", "calibrations"} & lower_parts)
    )


def _is_evidence_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    lower_name = path.name.lower()
    lower_parts = {part.lower() for part in path.parts}
    if suffix in ALWAYS_EVIDENCE_SUFFIXES:
        return True
    if lower_name in {"info.txt", "readme.txt", "h.txt", "h-old.txt", "h-cam.txt"}:
        return True
    if re.fullmatch(r"h(?:[-_][a-z0-9]+)?\.txt", lower_name):
        return True
    if suffix in {".txt", ".csv"} and (_contains_name_term(path) or {"calib", "calibration", "calibrations"} & lower_parts):
        return True
    return False


def _candidate_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return [root] if _is_evidence_file(root) else []

    out: list[Path] = []
    for current, _, files in os.walk(root):
        for filename in files:
            path = Path(current) / filename
            if _is_evidence_file(path):
                out.append(path)
            if len(out) >= MAX_SCAN_FILES_PER_DATASET:
                return out
    return out


def _read_small_text(path: Path) -> str:
    try:
        if path.stat().st_size > MAX_TEXT_BYTES:
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _matrix_det_3x3(values: list[list[float]]) -> float:
    a, b, c = values[0]
    d, e, f = values[1]
    g, h, i = values[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def parse_homography_like_matrices(text: str) -> list[dict[str, Any]]:
    """Return parseable 3x3 numeric matrices from small calibration text.

    This only proves that a homography-like file is parseable. It does not prove
    the units, coordinate convention, or source-term validity needed for a metric
    claim.
    """

    rows: list[list[float]] = []
    for line in text.splitlines():
        nums = [float(x) for x in re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?", line)]
        if len(nums) >= 3:
            rows.append(nums[:3])
    matrices: list[dict[str, Any]] = []
    for idx in range(0, max(0, len(rows) - 2)):
        candidate = rows[idx : idx + 3]
        if all(len(row) == 3 and all(math.isfinite(v) for v in row) for row in candidate):
            det = _matrix_det_3x3(candidate)
            if math.isfinite(det) and abs(det) > 1e-12:
                matrices.append({"row_index": idx, "determinant": det})
    return matrices


def _extract_json_evidence(path: Path, text: str) -> dict[str, Any]:
    if path.suffix.lower() != ".json" or not text.strip():
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    keys = {str(k).lower(): v for k, v in payload.items()}
    evidence: dict[str, Any] = {}
    for key in ("coordinate_unit", "metric_status", "whether_metric_coordinates", "dt_note", "frame_rate", "fps", "stride"):
        if key in keys:
            evidence[key] = keys[key]
    if "columns_inferred" in keys:
        evidence["columns_inferred_present"] = True
    if "horizon_audit" in keys:
        evidence["horizon_audit_present"] = True
    return evidence


def scan_evidence_file(path: Path) -> dict[str, Any]:
    text = _read_small_text(path)
    lower = text.lower()
    name_lower = path.name.lower()
    homography_file_candidate = _is_homography_evidence_file(path)
    matrices = parse_homography_like_matrices(text) if homography_file_candidate else []
    fps_terms = re.findall(r"(?:fps|frame[_ -]?rate|frames per second)[^\n\r]{0,40}", lower)
    stride_terms = re.findall(r"(?:stride|dt|delta[_ -]?t|time step|frame step)[^\n\r]{0,40}", lower)
    scale_terms = re.findall(r"(?:meter|metre|meters per pixel|m/pixel|scale)[^\n\r]{0,60}", lower)
    homography_hint = homography_file_candidate or "homography" in lower or name_lower == "h.txt"
    json_evidence = _extract_json_evidence(path, text)
    return {
        "path": str(path),
        "bytes": path.stat().st_size if path.exists() else 0,
        "has_text": bool(text),
        "homography_hint": homography_hint,
        "parseable_homography_matrices": len(matrices),
        "fps_terms": sorted(set(fps_terms))[:6],
        "stride_terms": sorted(set(stride_terms))[:6],
        "scale_terms": sorted(set(scale_terms))[:6],
        "json_evidence": json_evidence,
    }


def _extra_metadata_candidates(dataset_id: str) -> list[str]:
    by_id = {
        "sdd": [
            "data/stage20_raw_index/stanford_drone/metadata.json",
            "outputs/stage30_m3w_verified/time_geometry_raw_audit.json",
            "outputs/reports/stage23_sdd_time_geometry_audit.json",
        ],
        "opentraj": ["data/stage20_raw_index/opentraj/metadata.json"],
        "eth_ucy": [
            "data/stage5b_world_state/eth_ucy/metadata.json",
            "data/stage20_raw_index/eth_ucy_full/metadata.json",
        ],
        "trajnet": [
            "data/stage5b_world_state/trajnet/metadata.json",
            "data/stage20_raw_index/trajnet_full/metadata.json",
        ],
        "ucy": ["data/stage20_raw_index/ucy_crowd/metadata.json"],
        "tgsim": [
            "data/stage5b_world_state/tgsim/metadata.json",
            "data/stage5b_world_state/tgsim_i90/metadata.json",
        ],
        "aerialmpt": [],
    }
    return by_id.get(dataset_id, [])


def _claim_status(dataset_id: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    parseable_h = evidence["parseable_homography_matrix_count"]
    fps_count = evidence["fps_or_frame_rate_evidence_count"]
    stride_count = evidence["stride_or_dt_evidence_count"]
    scale_count = evidence["scale_or_meter_evidence_count"]
    coordinate_units = set(evidence["coordinate_unit_evidence"])
    metric_meta = evidence["metadata_metric_true_count"]

    if dataset_id == "tgsim":
        return {
            "metric_claim_status": "traffic_metric_diagnostic_only",
            "seconds_claim_status": "time_values_dataset_diagnostic_only" if stride_count else "seconds_not_validated",
            "allowed_claim": "traffic diagnostic metric only; not pedestrian/drone world-model success",
            "metric_claim_allowed_for_pedestrian_world_model": False,
            "seconds_claim_allowed_for_official_pedestrian": False,
        }
    if dataset_id == "sdd":
        return {
            "metric_claim_status": "not_allowed_pixel_space",
            "seconds_claim_status": "not_allowed_effective_seconds_unverified",
            "allowed_claim": "pixel raw-frame only",
            "metric_claim_allowed_for_pedestrian_world_model": False,
            "seconds_claim_allowed_for_official_pedestrian": False,
        }

    weak_candidate = parseable_h > 0 or metric_meta > 0 or scale_count > 0 or any("world" in unit for unit in coordinate_units)
    seconds_candidate = fps_count > 0 and stride_count > 0
    return {
        "metric_claim_status": "weak_metric_candidate_requires_manual_validation" if weak_candidate else "not_allowed_no_verified_scale",
        "seconds_claim_status": "effective_seconds_candidate_requires_manual_validation" if seconds_candidate else "not_allowed_no_verified_fps_stride_pair",
        "allowed_claim": "dataset-local raw-frame only until source-specific homography/FPS/scale is manually validated",
        "metric_claim_allowed_for_pedestrian_world_model": False,
        "seconds_claim_allowed_for_official_pedestrian": False,
    }


def audit_dataset_calibration_evidence(spec: Mapping[str, Any]) -> dict[str, Any]:
    dataset_id = str(spec["id"])
    candidate_roots = [Path(p) for p in spec.get("raw_candidates", []) + spec.get("converted_candidates", [])]
    candidate_roots.extend(Path(p) for p in _extra_metadata_candidates(dataset_id))
    checked_roots = [{"path": str(root), "exists": root.exists()} for root in candidate_roots]

    files: list[Path] = []
    for root in candidate_roots:
        files.extend(_candidate_files(root))
        if len(files) >= MAX_SCAN_FILES_PER_DATASET:
            files = files[:MAX_SCAN_FILES_PER_DATASET]
            break
    unique_files = sorted({str(p): p for p in files}.values(), key=lambda p: str(p))
    scans = [scan_evidence_file(path) for path in unique_files]

    coordinate_units: Counter[str] = Counter()
    metadata_metric_true_count = 0
    for scan in scans:
        json_e = scan.get("json_evidence", {})
        if "coordinate_unit" in json_e:
            coordinate_units[str(json_e["coordinate_unit"])] += 1
        if json_e.get("whether_metric_coordinates") is True:
            metadata_metric_true_count += 1

    evidence = {
        "source": "fresh_run",
        "dataset_id": dataset_id,
        "dataset_name": spec["name"],
        "data_role": spec["role"],
        "known_coordinate_unit": spec["known_coordinate_unit"],
        "known_metric_status": spec["known_metric_status"],
        "official_hint": spec["official_hint"],
        "local_paths_checked": checked_roots,
        "evidence_files_scanned": len(scans),
        "evidence_file_samples": [scan["path"] for scan in scans[:24]],
        "homography_like_file_count": sum(1 for scan in scans if scan["homography_hint"]),
        "parseable_homography_matrix_count": sum(int(scan["parseable_homography_matrices"]) for scan in scans),
        "fps_or_frame_rate_evidence_count": sum(1 for scan in scans if scan["fps_terms"]),
        "stride_or_dt_evidence_count": sum(1 for scan in scans if scan["stride_terms"]),
        "scale_or_meter_evidence_count": sum(1 for scan in scans if scan["scale_terms"]),
        "coordinate_unit_evidence": dict(coordinate_units),
        "metadata_metric_true_count": metadata_metric_true_count,
        "notable_evidence": [
            scan
            for scan in scans
            if scan["homography_hint"] or scan["fps_terms"] or scan["stride_terms"] or scan["scale_terms"] or scan["json_evidence"]
        ][:30],
    }
    evidence.update(_claim_status(dataset_id, evidence))
    if evidence["metric_claim_status"].startswith("weak_metric") or evidence["seconds_claim_status"].startswith("effective"):
        evidence["next_action"] = (
            "manual_source_validation_required: confirm official coordinate convention, homography direction, frame rate, annotation stride, and meters-per-pixel before metric/seconds claim"
        )
    elif dataset_id == "tgsim":
        evidence["next_action"] = "keep_as_traffic_diagnostic_only; do not use as pedestrian top-down world-model success"
    else:
        evidence["next_action"] = "keep_dataset_local_raw_frame_claim"
    return evidence


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    datasets = payload["datasets"]
    by_id = {d["dataset_id"]: d for d in datasets}
    gates = [
        ("Stage42-A Prior Audit Present", Path("outputs/stage42_long_research/data_calibration_stage42.json").exists()),
        ("Required Dataset Coverage", len(datasets) >= 7 and {"sdd", "opentraj", "eth_ucy", "trajnet", "ucy", "tgsim", "aerialmpt"}.issubset(by_id)),
        ("Evidence Files Scanned", sum(d["evidence_files_scanned"] for d in datasets) > 0),
        ("Homography/Scale Evidence Separated From Claim", any(d["parseable_homography_matrix_count"] > 0 for d in datasets) and not payload["summary"]["global_metric_claim_allowed"]),
        ("Pedestrian Metric Overclaim Guard", not any(d["metric_claim_allowed_for_pedestrian_world_model"] for d in datasets)),
        ("Pedestrian Seconds Overclaim Guard", not any(d["seconds_claim_allowed_for_official_pedestrian"] for d in datasets)),
        ("TGSIM Diagnostic Boundary", by_id.get("tgsim", {}).get("metric_claim_status") == "traffic_metric_diagnostic_only"),
        ("User Action Generated", bool(payload["user_action_required"])),
        ("Stage5C Execution Gate", payload["claim_boundary"]["stage5c_executed"] is False),
        ("SMC Execution Gate", payload["claim_boundary"]["smc_enabled"] is False),
    ]
    passed = sum(1 for _, ok in gates if ok)
    return {
        "source": "fresh_run",
        "passed": passed,
        "total": len(gates),
        "verdict": "stage42_ad_calibration_evidence_refresh_pass" if passed == len(gates) else "stage42_ad_calibration_evidence_refresh_partial",
        "gates": [{"name": name, "passed": bool(ok)} for name, ok in gates],
    }


def build_calibration_evidence_refresh() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    datasets = [audit_dataset_calibration_evidence(spec) for spec in DATASET_SPECS]
    summary = {
        "datasets_audited": len(datasets),
        "evidence_files_scanned": sum(d["evidence_files_scanned"] for d in datasets),
        "datasets_with_parseable_homography_like_matrices": [
            d["dataset_id"] for d in datasets if d["parseable_homography_matrix_count"] > 0
        ],
        "datasets_with_fps_evidence": [d["dataset_id"] for d in datasets if d["fps_or_frame_rate_evidence_count"] > 0],
        "datasets_with_stride_or_dt_evidence": [d["dataset_id"] for d in datasets if d["stride_or_dt_evidence_count"] > 0],
        "datasets_with_scale_or_meter_evidence": [d["dataset_id"] for d in datasets if d["scale_or_meter_evidence_count"] > 0],
        "global_metric_claim_allowed": False,
        "global_seconds_claim_allowed": False,
        "pedestrian_metric_claim_ready": [],
        "pedestrian_seconds_claim_ready": [],
        "traffic_metric_diagnostic": ["tgsim"],
    }
    user_actions = []
    for d in datasets:
        if (
            d["metric_claim_status"].startswith("weak_metric")
            or d["seconds_claim_status"].startswith("effective")
            or d["dataset_id"] in {"sdd", "aerialmpt"}
        ):
            user_actions.append(
                {
                    "dataset_id": d["dataset_id"],
                    "dataset_name": d["dataset_name"],
                    "official_hint": d["official_hint"],
                    "reason": d["next_action"],
                    "needed_for": "metric/time calibration claim boundary",
                }
            )
    payload = {
        "source": "fresh_run",
        "stage": "Stage42-AD calibration evidence refresh",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "current_facts": CURRENT_FACTS,
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "input_hash": _hash_paths(
            ["outputs/stage42_long_research/data_calibration_stage42.json"]
            + [p for spec in DATASET_SPECS for p in spec.get("raw_candidates", []) + spec.get("converted_candidates", [])]
            + [p for spec in DATASET_SPECS for p in _extra_metadata_candidates(str(spec["id"]))]
        ),
        "summary": summary,
        "datasets": datasets,
        "user_action_required": user_actions,
    }
    payload["stage42_ad_gate"] = _gate(payload)
    write_json(OUT_DIR / "calibration_evidence_refresh_stage42.json", payload)
    write_md(OUT_DIR / "calibration_evidence_refresh_stage42.md", _render_md(payload))
    _write_csv(OUT_DIR / "calibration_evidence_refresh_stage42.csv", payload)
    write_md(OUT_DIR / "stage42_stage_ad_gate.md", _render_gate_md(payload))
    write_md(OUT_DIR / "user_action_required_stage42_calibration.md", _render_user_actions(payload))
    return payload


def _write_csv(path: Path, payload: Mapping[str, Any]) -> None:
    rows = []
    for d in payload["datasets"]:
        rows.append(
            {
                "dataset_id": d["dataset_id"],
                "dataset_name": d["dataset_name"],
                "files_scanned": d["evidence_files_scanned"],
                "homography_like_files": d["homography_like_file_count"],
                "parseable_homography_matrices": d["parseable_homography_matrix_count"],
                "fps_evidence": d["fps_or_frame_rate_evidence_count"],
                "stride_dt_evidence": d["stride_or_dt_evidence_count"],
                "scale_meter_evidence": d["scale_or_meter_evidence_count"],
                "metric_claim_status": d["metric_claim_status"],
                "seconds_claim_status": d["seconds_claim_status"],
                "allowed_claim": d["allowed_claim"],
                "next_action": d["next_action"],
            }
        )
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _render_md(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-AD Calibration Evidence Refresh",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_ad_gate']['passed']} / {payload['stage42_ad_gate']['total']}`",
        f"- verdict: `{payload['stage42_ad_gate']['verdict']}`",
        "",
        "## Current Claim Boundary",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- datasets_audited: `{s['datasets_audited']}`",
        f"- evidence_files_scanned: `{s['evidence_files_scanned']}`",
        f"- datasets_with_parseable_homography_like_matrices: `{', '.join(s['datasets_with_parseable_homography_like_matrices']) or 'none'}`",
        f"- datasets_with_fps_evidence: `{', '.join(s['datasets_with_fps_evidence']) or 'none'}`",
        f"- datasets_with_stride_or_dt_evidence: `{', '.join(s['datasets_with_stride_or_dt_evidence']) or 'none'}`",
        f"- datasets_with_scale_or_meter_evidence: `{', '.join(s['datasets_with_scale_or_meter_evidence']) or 'none'}`",
        f"- global_metric_claim_allowed: `{s['global_metric_claim_allowed']}`",
        f"- global_seconds_claim_allowed: `{s['global_seconds_claim_allowed']}`",
        "",
        "## Dataset Evidence Table",
        "",
        "| dataset | files | parseable H | fps | stride/dt | scale/meter | metric status | seconds status | allowed claim |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for d in payload["datasets"]:
        lines.append(
            f"| `{d['dataset_id']}` | {d['evidence_files_scanned']} | {d['parseable_homography_matrix_count']} | {d['fps_or_frame_rate_evidence_count']} | {d['stride_or_dt_evidence_count']} | {d['scale_or_meter_evidence_count']} | {d['metric_claim_status']} | {d['seconds_claim_status']} | {d['allowed_claim']} |"
        )
    lines.extend(["", "## Per-Dataset Notes", ""])
    for d in payload["datasets"]:
        lines.extend(
            [
                f"### {d['dataset_name']}",
                "",
                f"- dataset_id: `{d['dataset_id']}`",
                f"- data_role: `{d['data_role']}`",
                f"- known_coordinate_unit: `{d['known_coordinate_unit']}`",
                f"- known_metric_status: `{d['known_metric_status']}`",
                f"- local_paths_checked: `{sum(1 for row in d['local_paths_checked'] if row['exists'])} / {len(d['local_paths_checked'])}`",
                f"- evidence_files_scanned: `{d['evidence_files_scanned']}`",
                f"- homography_like_file_count: `{d['homography_like_file_count']}`",
                f"- parseable_homography_matrix_count: `{d['parseable_homography_matrix_count']}`",
                f"- fps_or_frame_rate_evidence_count: `{d['fps_or_frame_rate_evidence_count']}`",
                f"- stride_or_dt_evidence_count: `{d['stride_or_dt_evidence_count']}`",
                f"- scale_or_meter_evidence_count: `{d['scale_or_meter_evidence_count']}`",
                f"- coordinate_unit_evidence: `{d['coordinate_unit_evidence']}`",
                f"- metric_claim_status: `{d['metric_claim_status']}`",
                f"- seconds_claim_status: `{d['seconds_claim_status']}`",
                f"- allowed_claim: `{d['allowed_claim']}`",
                f"- next_action: `{d['next_action']}`",
                "",
            ]
        )
        if d["notable_evidence"]:
            lines.extend(["Notable evidence samples:", ""])
            for sample in d["notable_evidence"][:8]:
                lines.append(
                    f"- `{sample['path']}`: H={sample['parseable_homography_matrices']}, fps={bool(sample['fps_terms'])}, stride/dt={bool(sample['stride_terms'])}, scale/meter={bool(sample['scale_terms'])}, json={bool(sample['json_evidence'])}"
                )
            lines.append("")
    lines.extend(
        [
            "## Conclusion",
            "",
            "Stage42-AD separates calibration evidence existence from claim permission. ETH/UCY and UCY contain parseable homography-like files, and some metadata/text sources contain FPS, dt, or metric hints, but this is still insufficient for global pedestrian metric or seconds-level claims without source-specific validation of coordinate convention, annotation stride, homography direction, and scale. SDD remains pixel raw-frame; external pedestrian datasets remain dataset-local raw-frame; TGSIM remains traffic diagnostic only. Stage5C and SMC remain disabled.",
        ]
    )
    return lines


def _render_gate_md(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ad_gate"]
    lines = [
        "# Stage42-AD Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- gates: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for row in gate["gates"]:
        lines.append(f"| {row['name']} | `{row['passed']}` |")
    return lines


def _render_user_actions(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-AD Calibration User Action Required",
        "",
        "- source: `fresh_run`",
        "- purpose: list evidence still needed before metric/time claims.",
        "",
    ]
    for row in payload["user_action_required"]:
        lines.extend(
            [
                f"## {row['dataset_name']}",
                "",
                f"- dataset_id: `{row['dataset_id']}`",
                f"- official_hint: `{row['official_hint']}`",
                f"- needed_for: `{row['needed_for']}`",
                f"- reason: {row['reason']}",
                "",
            ]
        )
    return lines


def main() -> None:
    build_calibration_evidence_refresh()


if __name__ == "__main__":
    main()
