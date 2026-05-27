from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_proximity_guard_policy_freeze import _replace_section
from src.stage42_raw_source_parseability_dry_run import _iter_existing_raw_files


OUT_DIR = Path("outputs/stage42_long_research")
DS_JSON = OUT_DIR / "source_conversion_readiness_recheck_stage42.json"
DT_JSON = OUT_DIR / "raw_source_parseability_dry_run_stage42.json"
DATA_CALIBRATION_JSON = OUT_DIR / "data_calibration_stage42.json"
DATA_CALIBRATION_MD = OUT_DIR / "data_calibration_stage42.md"

REPORT_JSON = OUT_DIR / "raw_source_time_geometry_hint_audit_stage42.json"
REPORT_MD = OUT_DIR / "raw_source_time_geometry_hint_audit_stage42.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_raw_source_time_geometry_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_du_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "global_t100_deployable_claim_allowed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-DU 是 raw source time/geometry hint audit：只抽取 H/FPS/stride hints，不转换轨迹。",
    "H matrix、FPS、frame stride 线索不是 metric/seconds-level claim。",
    "legal/source blocker 未关闭时，不能把 hints 写成 official metric conversion。",
    "本步骤不下载、不解压 gated 数据、不训练、不评估、不生成 world-state rows。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon；dataset-local/raw-frame 不能写成 global metric/seconds。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _read_text(path: Path, max_bytes: int = 256_000) -> str:
    try:
        with path.open("rb") as f:
            data = f.read(max_bytes)
    except OSError:
        return ""
    return data.decode("utf-8", errors="ignore")


def _parse_h_matrix(text: str) -> dict[str, Any] | None:
    rows: list[list[float]] = []
    for line in text.splitlines():
        parts: list[float] = []
        for token in line.replace(",", " ").split():
            try:
                parts.append(float(token))
            except ValueError:
                continue
        if len(parts) >= 3:
            rows.append(parts[:3])
        if len(rows) == 3:
            break
    if len(rows) != 3:
        return None
    a = rows
    det = (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )
    return {
        "matrix": rows,
        "determinant": det,
        "non_singular_hint": abs(det) > 1e-12,
    }


def _parse_time_metadata(text: str) -> dict[str, Any]:
    fps_values = [float(x) for x in re.findall(r"(?<![A-Za-z])([0-9]+(?:\.[0-9]+)?)\s*fps", text, flags=re.I)]
    timestep_values = [
        float(x)
        for x in re.findall(r"timestep\s+of\s+([0-9]+(?:\.[0-9]+)?)\s*seconds?", text, flags=re.I)
    ]
    annotation_fps = None
    if "annotation" in text.lower() and fps_values:
        annotation_fps = fps_values[-1]
    elif timestep_values:
        annotation_fps = 1.0 / timestep_values[-1] if timestep_values[-1] else None
    return {
        "fps_values": fps_values,
        "timestep_seconds_values": timestep_values,
        "annotation_fps_hint": annotation_fps,
    }


def _parse_ndjson_fps(path: Path, max_lines: int = 50) -> list[float]:
    fps: list[float] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                scene = obj.get("scene")
                if isinstance(scene, dict) and "fps" in scene:
                    try:
                        fps.append(float(scene["fps"]))
                    except (TypeError, ValueError):
                        pass
    except OSError:
        return []
    return fps


def _parse_frame_stride(path: Path, max_lines: int = 5000) -> dict[str, Any] | None:
    frames: list[float] = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                parts = line.replace(",", " ").split()
                if len(parts) < 4:
                    continue
                try:
                    frame = float(parts[0])
                except ValueError:
                    continue
                frames.append(frame)
    except OSError:
        return None
    unique = sorted(set(frames))
    if len(unique) < 3:
        return None
    diffs = [round(unique[i + 1] - unique[i], 6) for i in range(len(unique) - 1) if unique[i + 1] > unique[i]]
    if not diffs:
        return None
    counts = Counter(diffs)
    mode, mode_count = counts.most_common(1)[0]
    return {
        "unique_frames_sampled": len(unique),
        "min_frame": unique[0],
        "max_frame": unique[-1],
        "mode_frame_stride": mode,
        "mode_count": mode_count,
        "stride_values_sample": sorted(counts.keys())[:10],
    }


def _audit_target(ds_row: Mapping[str, Any]) -> dict[str, Any]:
    files = _iter_existing_raw_files(ds_row, limit=180)
    h_hints = []
    time_hints = []
    stride_hints = []
    ndjson_fps_hints = []
    for path in files:
        name = path.name.lower()
        suffix = path.suffix.lower()
        if name in {"h.txt", "h-cam.txt", "h-old.txt"}:
            h = _parse_h_matrix(_read_text(path))
            if h:
                h_hints.append({"path": str(path), **h})
        if suffix in {".txt", ".md"}:
            text = _read_text(path)
            t = _parse_time_metadata(text)
            if t["fps_values"] or t["timestep_seconds_values"]:
                time_hints.append({"path": str(path), **t})
        if suffix == ".ndjson":
            fps = _parse_ndjson_fps(path)
            if fps:
                ndjson_fps_hints.append({"path": str(path), "fps_values": sorted(set(fps))})
        if suffix == ".txt" and name not in {"h.txt", "h-cam.txt", "h-old.txt"}:
            stride = _parse_frame_stride(path)
            if stride:
                stride_hints.append({"path": str(path), **stride})
    any_time = bool(time_hints or ndjson_fps_hints)
    any_h = bool(h_hints)
    metric_time_subset_hint = any_h and any_time
    return {
        "dataset_id": ds_row.get("dataset_id"),
        "domain": ds_row.get("domain"),
        "files_scanned": len(files),
        "h_matrix_hints": h_hints[:12],
        "time_metadata_hints": time_hints[:12],
        "ndjson_fps_hints": ndjson_fps_hints[:12],
        "frame_stride_hints": stride_hints[:20],
        "h_matrix_hint_count": len(h_hints),
        "time_metadata_hint_count": len(time_hints) + len(ndjson_fps_hints),
        "frame_stride_hint_count": len(stride_hints),
        "metric_time_subset_hint": metric_time_subset_hint,
        "legal_conversion_ready": bool(ds_row.get("conversion_ready")),
        "claim_allowed_now": False,
        "reason_claim_not_allowed": "hints_only_and_terms_source_confirmation_missing",
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    claim = payload["claim_boundary"]
    gates = {
        "stage42_ds_input_present": bool(payload.get("stage42_ds_verdict")),
        "stage42_dt_input_present": bool(payload.get("stage42_dt_verdict")),
        "hints_audited": s["targets_checked"] >= 7,
        "h_matrix_hints_found": s["targets_with_h_matrix_hints"] >= 2,
        "time_hints_found": s["targets_with_time_hints"] >= 1,
        "frame_stride_hints_found": s["targets_with_frame_stride_hints"] >= 3,
        "metric_time_subset_hints_separated": s["metric_time_subset_hint_targets"] >= 1,
        "legal_readiness_not_overclaimed": s["legal_conversion_ready_targets"] == 0,
        "no_conversion_or_rows": s["converted_datasets_now"] == 0 and s["world_state_rows_generated"] == 0,
        "no_evaluation_claim": s["evaluated_datasets_now"] == 0,
        "data_calibration_addendum_written": bool(payload.get("data_calibration_addendum_written")),
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_du_raw_source_time_geometry_hint_audit_pass" if passed == total else "stage42_du_raw_source_time_geometry_hint_audit_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-DU Raw Source Time/Geometry Hint Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{payload['stage42_du_gate']['passed']} / {payload['stage42_du_gate']['total']}`",
        f"- verdict: `{payload['stage42_du_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        *[f"- {key}: `{value}`" for key, value in payload["summary"].items()],
        "",
        "## Dataset Hint Table",
        "",
        "| dataset | domain | files | H hints | time hints | stride hints | metric/time subset hint | claim allowed |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["target_rows"]:
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                row["dataset_id"],
                row["domain"],
                row["files_scanned"],
                row["h_matrix_hint_count"],
                row["time_metadata_hint_count"],
                row["frame_stride_hint_count"],
                row["metric_time_subset_hint"],
                row["claim_allowed_now"],
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- H/FPS/stride hints were extracted from local files, but no conversion was performed.",
            "- These hints can guide a future no-leakage conversion only after terms/source confirmation.",
            "- No global metric or seconds-level claim is allowed.",
            "- Stage5C and SMC remain disabled.",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | --- |",
            *[f"| `{key}` | `{value}` |" for key, value in payload["stage42_du_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage42-DU Time/Geometry Hints",
        "",
        "本步骤发现 H/FPS/stride hints，但它们不是 legal conversion 或 metric/seconds claim。继续需要用户确认：",
        "",
        "- terms acceptance and allowed use",
        "- exact raw local source path",
        "- source identity / version",
        "- whether homography direction and coordinate convention are official",
        "- whether annotation FPS/stride can be cited from official documentation",
        "",
        "| dataset | hints found | required action |",
        "| --- | --- | --- |",
    ]
    for row in payload["target_rows"]:
        hints = []
        if row["h_matrix_hint_count"]:
            hints.append(f"H={row['h_matrix_hint_count']}")
        if row["time_metadata_hint_count"]:
            hints.append(f"time={row['time_metadata_hint_count']}")
        if row["frame_stride_hint_count"]:
            hints.append(f"stride={row['frame_stride_hint_count']}")
        lines.append(
            f"| `{row['dataset_id']}` | {', '.join(hints) or 'none'} | confirm source/legal/time-geometry before conversion or metric/seconds claim |"
        )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_du_gate"]
    return [
        "# Stage42-DU Gate",
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
        "## Stage42-DU Raw Source Time/Geometry Hint Addendum",
        "",
        "- source: `fresh_hint_audit_from_local_raw_sources_after_stage42_dt`",
        "- role: H/FPS/stride hint extraction only; no conversion and no metric/seconds claim.",
        f"- gate: `{payload['stage42_du_gate']['passed']} / {payload['stage42_du_gate']['total']}`; verdict `{payload['stage42_du_gate']['verdict']}`.",
        f"- targets checked: `{s['targets_checked']}`; H-hint targets: `{s['targets_with_h_matrix_hints']}`; time-hint targets: `{s['targets_with_time_hints']}`; stride-hint targets: `{s['targets_with_frame_stride_hints']}`.",
        f"- metric/time subset hint targets: `{s['metric_time_subset_hint_targets']}`; legal conversion ready targets: `{s['legal_conversion_ready_targets']}`.",
        "- H/FPS/stride hints remain hints until source/legal confirmation and no-leakage conversion are complete.",
    ]


def _refresh_data_calibration(payload: Mapping[str, Any]) -> None:
    existing = read_json(DATA_CALIBRATION_JSON, {}) if DATA_CALIBRATION_JSON.exists() else {}
    existing["stage42_du_raw_source_time_geometry_hint_audit"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_du_gate"]["verdict"],
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(DATA_CALIBRATION_JSON, existing)
    _replace_section(DATA_CALIBRATION_MD, "STAGE42_DU_RAW_SOURCE_TIME_GEOMETRY_HINT_AUDIT", _calibration_addendum(payload))


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    gate = payload["stage42_du_gate"]
    return [
        "## Stage42-DU Raw Source Time/Geometry Hint Audit",
        "",
        "- source: `fresh_hint_audit_from_local_raw_sources_after_stage42_dt`",
        "- role: extracts H/FPS/stride hints only; no conversion, no evaluation, no metric/seconds claim.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- H-hint targets: `{s['targets_with_h_matrix_hints']}`; time-hint targets: `{s['targets_with_time_hints']}`; stride-hint targets: `{s['targets_with_frame_stride_hints']}`.",
        f"- metric/time subset hint targets: `{s['metric_time_subset_hint_targets']}`; legal conversion ready targets: `{s['legal_conversion_ready_targets']}`.",
        f"- report: `{REPORT_MD}`.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, GOAL_SUMMARY]:
        _replace_section(path, "STAGE42_DU_RAW_SOURCE_TIME_GEOMETRY_HINT_AUDIT", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-DU raw source time/geometry hint audit"
    state["current_verdict"] = payload["stage42_du_gate"]["verdict"]
    state["stage42_du_raw_source_time_geometry_hint_audit"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "user_action_required": str(USER_ACTION_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_du_gate"]["verdict"],
        "gates": f"{payload['stage42_du_gate']['passed']}/{payload['stage42_du_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_raw_source_time_geometry_hint_audit() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ds_payload = read_json(DS_JSON, {})
    dt_payload = read_json(DT_JSON, {})
    rows = [_audit_target(row) for row in ds_payload.get("target_rows", [])]
    payload: dict[str, Any] = {
        "source": "fresh_hint_audit_from_local_raw_sources_after_stage42_dt",
        "stage": "Stage42-DU",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_facts": CURRENT_FACTS,
        "stage42_ds_report": str(DS_JSON),
        "stage42_ds_verdict": ds_payload.get("stage42_ds_gate", {}).get("verdict"),
        "stage42_dt_report": str(DT_JSON),
        "stage42_dt_verdict": dt_payload.get("stage42_dt_gate", {}).get("verdict"),
        "target_rows": rows,
        "summary": {
            "targets_checked": len(rows),
            "files_scanned_total": sum(row["files_scanned"] for row in rows),
            "targets_with_h_matrix_hints": sum(1 for row in rows if row["h_matrix_hint_count"] > 0),
            "targets_with_time_hints": sum(1 for row in rows if row["time_metadata_hint_count"] > 0),
            "targets_with_frame_stride_hints": sum(1 for row in rows if row["frame_stride_hint_count"] > 0),
            "metric_time_subset_hint_targets": sum(1 for row in rows if row["metric_time_subset_hint"]),
            "legal_conversion_ready_targets": sum(1 for row in rows if row["legal_conversion_ready"]),
            "converted_datasets_now": 0,
            "evaluated_datasets_now": 0,
            "world_state_rows_generated": 0,
        },
        "data_calibration_addendum_written": True,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["stage42_du_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    _refresh_data_calibration(payload)
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_raw_source_time_geometry_hint_audit()
