from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
DS_JSON = OUT_DIR / "source_conversion_readiness_recheck_stage42.json"

REPORT_JSON = OUT_DIR / "raw_source_parseability_dry_run_stage42.json"
REPORT_MD = OUT_DIR / "raw_source_parseability_dry_run_stage42.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_raw_source_parseability_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_dt_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

MAX_FILES_PER_TARGET = 80
MAX_LINES_PER_FILE = 20

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
    "Stage42-DT 是 raw source parseability dry-run：只做文件形态和少量样例行解析，不生成转换数据。",
    "本步骤不下载、不解压 gated 数据、不训练、不评估、不生成 world-state rows。",
    "sample parseability 不等于 legal conversion permission，也不等于 official benchmark readiness。",
    "derived cache 不算 raw official source。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon；dataset-local/raw-frame 不能写成 global metric/seconds。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


TEXT_SUFFIXES = {".txt", ".ndjson", ".json", ".xml", ".yaml", ".yml", ".vsp", ".md"}


def _iter_existing_raw_files(row: Mapping[str, Any], limit: int = MAX_FILES_PER_TARGET) -> list[Path]:
    files: list[Path] = []
    for summary in row.get("raw_path_summaries", []):
        if not summary.get("exists"):
            continue
        path = Path(str(summary.get("path")))
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    files.append(child)
                if len(files) >= limit:
                    return files
        if len(files) >= limit:
            break
    return files[:limit]


def _read_sample_lines(path: Path, max_lines: int = MAX_LINES_PER_FILE) -> list[str]:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []
    return [line.strip() for line in lines[:max_lines] if line.strip()]


def _numeric_columns(line: str) -> int:
    count = 0
    for part in line.replace(",", " ").split():
        try:
            float(part)
        except ValueError:
            continue
        count += 1
    return count


def _classify_file(path: Path, sample_lines: Iterable[str]) -> dict[str, Any]:
    name = path.name.lower()
    suffix = path.suffix.lower()
    lines = list(sample_lines)
    numeric_counts = [_numeric_columns(line) for line in lines[:5]]
    text = "\n".join(lines[:10]).lower()
    parser_family = "unknown_or_non_trajectory"
    if suffix == ".ndjson" and any('"scene"' in line or '"track"' in line for line in lines):
        parser_family = "trajnetpp_ndjson_scene_or_track"
    elif suffix == ".xml" and ("<object" in text or "<frame" in text):
        parser_family = "eth_person_xml_tracks"
    elif suffix == ".txt" and name in {"h.txt", "h-cam.txt", "h-old.txt"}:
        parser_family = "homography_matrix_candidate"
    elif suffix in {".txt", ".vsp"} and any(count >= 8 for count in numeric_counts):
        parser_family = "obsmat_like_8col_trajectory"
    elif suffix == ".txt" and any(count >= 4 for count in numeric_counts):
        parser_family = "trajnet_4col_trajectory"
    elif suffix == ".txt" and "fps" in text:
        parser_family = "time_metadata_candidate"
    elif suffix in {".png", ".jpg", ".jpeg"}:
        parser_family = "scene_image_candidate"
    elif suffix in {".avi", ".mp4"}:
        parser_family = "raw_video_candidate"
    elif suffix in {".zip", ".tgz", ".gz", ".tar", ".rar"}:
        parser_family = "archive_requires_explicit_terms_before_extraction"
    row_like = parser_family in {
        "trajnetpp_ndjson_scene_or_track",
        "eth_person_xml_tracks",
        "obsmat_like_8col_trajectory",
        "trajnet_4col_trajectory",
    }
    calibration_like = parser_family in {"homography_matrix_candidate", "time_metadata_candidate"}
    return {
        "path": str(path),
        "suffix": suffix or "<none>",
        "sample_lines_read": len(lines),
        "numeric_column_counts_first5": numeric_counts,
        "parser_family": parser_family,
        "trajectory_like": row_like,
        "calibration_like": calibration_like,
    }


def _summarize_target(row: Mapping[str, Any]) -> dict[str, Any]:
    files = _iter_existing_raw_files(row)
    classified = [_classify_file(path, _read_sample_lines(path)) for path in files]
    families: dict[str, int] = {}
    suffixes: dict[str, int] = {}
    for item in classified:
        families[item["parser_family"]] = families.get(item["parser_family"], 0) + 1
        suffixes[item["suffix"]] = suffixes.get(item["suffix"], 0) + 1
    trajectory_like = [item for item in classified if item["trajectory_like"]]
    calibration_like = [item for item in classified if item["calibration_like"]]
    has_homography_hint = any(item["parser_family"] == "homography_matrix_candidate" for item in classified)
    has_time_hint = any(item["parser_family"] == "time_metadata_candidate" for item in classified)
    dry_run_parseable = bool(trajectory_like)
    legal_conversion_ready = bool(row.get("conversion_ready"))
    return {
        "dataset_id": row.get("dataset_id"),
        "domain": row.get("domain"),
        "source_from_ds": "cached_verified_from_stage42_ds",
        "files_sampled": len(classified),
        "parser_families": dict(sorted(families.items())),
        "suffix_counts": dict(sorted(suffixes.items())),
        "trajectory_like_files_sampled": len(trajectory_like),
        "calibration_like_files_sampled": len(calibration_like),
        "has_homography_hint": has_homography_hint,
        "has_time_hint": has_time_hint,
        "dry_run_parseable": dry_run_parseable,
        "legal_conversion_ready": legal_conversion_ready,
        "conversion_executed": False,
        "evaluation_executed": False,
        "world_state_rows_generated": 0,
        "sampled_files": classified[:20],
        "technical_next_step": _technical_next_step(row, dry_run_parseable, has_homography_hint, has_time_hint),
    }


def _technical_next_step(
    row: Mapping[str, Any],
    dry_run_parseable: bool,
    has_homography_hint: bool,
    has_time_hint: bool,
) -> str:
    if row.get("dataset_id") == "stanford_drone_dataset":
        return "already converted SDD reference; use only for SDD pixel raw-frame work"
    if row.get("domain") == "traffic_diagnostic":
        return "diagnostic traffic source only; not pedestrian topdown official"
    if not dry_run_parseable:
        return "locate raw trajectory files or official extraction instructions after terms confirmation"
    if has_homography_hint or has_time_hint:
        return "after user terms/source confirmation, run no-leakage conversion plus source-specific time/geometry audit"
    return "after user terms/source confirmation, run no-leakage conversion as dataset-local raw-frame source"


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    claim = payload["claim_boundary"]
    gates = {
        "stage42_ds_input_present": bool(payload.get("stage42_ds_verdict")),
        "sample_only_no_conversion": s["world_state_rows_generated"] == 0 and s["converted_datasets_now"] == 0,
        "sample_only_no_evaluation": s["evaluated_datasets_now"] == 0,
        "parseable_sources_identified": s["dry_run_parseable_targets"] >= 3,
        "homography_or_time_hints_reported": s["targets_with_homography_or_time_hints"] >= 1,
        "legal_readiness_not_overclaimed": s["legal_conversion_ready_targets"] == 0,
        "archives_not_extracted": s["archives_extracted_now"] == 0,
        "user_action_present": s["user_action_required_targets"] >= 5,
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_dt_raw_source_parseability_dry_run_pass" if passed == total else "stage42_dt_raw_source_parseability_dry_run_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-DT Raw Source Parseability Dry Run",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{payload['stage42_dt_gate']['passed']} / {payload['stage42_dt_gate']['total']}`",
        f"- verdict: `{payload['stage42_dt_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        *[f"- {key}: `{value}`" for key, value in payload["summary"].items()],
        "",
        "## Target Parseability",
        "",
        "| dataset | domain | files sampled | trajectory-like | calibration-like | H hint | time hint | legal ready | next step |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["target_rows"]:
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | {} |".format(
                row["dataset_id"],
                row["domain"],
                row["files_sampled"],
                row["trajectory_like_files_sampled"],
                row["calibration_like_files_sampled"],
                row["has_homography_hint"],
                row["has_time_hint"],
                row["legal_conversion_ready"],
                row["technical_next_step"],
            )
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This is a sample-only parser preflight. It does not create feature stores, world-state rows, episodes, or benchmarks.",
            "- Legal/source blockers from Stage42-DS remain active; legal conversion ready remains zero.",
            "- Homography/time hints are only hints; they do not authorize metric or seconds-level claims.",
            "- Archives were not extracted.",
            "- Stage5C and SMC remain disabled.",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | --- |",
            *[f"| `{key}` | `{value}` |" for key, value in payload["stage42_dt_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage42-DT Raw Source Parseability",
        "",
        "本步骤只证明若干本地源有 trajectory-like / calibration-like 文件形态。继续转换前仍需用户确认 legal/source 信息。",
        "",
        "| dataset | dry-run status | required action |",
        "| --- | --- | --- |",
    ]
    for row in payload["target_rows"]:
        if row["legal_conversion_ready"]:
            continue
        status = "parseable" if row["dry_run_parseable"] else "not parseable in sample"
        lines.append(f"| `{row['dataset_id']}` | {status} | {row['technical_next_step']}；并提供 terms/source confirmation |")
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_dt_gate"]
    return [
        "# Stage42-DT Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
        *[f"| `{key}` | `{value}` |" for key, value in gate["gates"].items()],
    ]


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    gate = payload["stage42_dt_gate"]
    return [
        "## Stage42-DT Raw Source Parseability Dry Run",
        "",
        "- source: `fresh_sample_only_raw_source_parseability_dry_run`",
        "- role: sample-only technical parser preflight after Stage42-DS; no conversion, no evaluation.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- dry-run parseable targets: `{s['dry_run_parseable_targets']}`; targets with homography/time hints: `{s['targets_with_homography_or_time_hints']}`.",
        f"- legal conversion ready targets: `{s['legal_conversion_ready_targets']}`; generated rows: `{s['world_state_rows_generated']}`.",
        "- Homography/time hints remain hints only; no metric/seconds claim is made.",
        f"- report: `{REPORT_MD}`.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, GOAL_SUMMARY]:
        _replace_section(path, "STAGE42_DT_RAW_SOURCE_PARSEABILITY_DRY_RUN", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-DT raw source parseability dry run"
    state["current_verdict"] = payload["stage42_dt_gate"]["verdict"]
    state["stage42_dt_raw_source_parseability_dry_run"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "user_action_required": str(USER_ACTION_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_dt_gate"]["verdict"],
        "gates": f"{payload['stage42_dt_gate']['passed']}/{payload['stage42_dt_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_raw_source_parseability_dry_run() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ds_payload = read_json(DS_JSON, {})
    rows = [_summarize_target(row) for row in ds_payload.get("target_rows", [])]
    payload: dict[str, Any] = {
        "source": "fresh_sample_only_raw_source_parseability_dry_run",
        "stage": "Stage42-DT",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_facts": CURRENT_FACTS,
        "stage42_ds_report": str(DS_JSON),
        "stage42_ds_verdict": ds_payload.get("stage42_ds_gate", {}).get("verdict"),
        "target_rows": rows,
        "summary": {
            "targets_checked": len(rows),
            "files_sampled_total": sum(row["files_sampled"] for row in rows),
            "dry_run_parseable_targets": sum(1 for row in rows if row["dry_run_parseable"]),
            "targets_with_homography_or_time_hints": sum(
                1 for row in rows if row["has_homography_hint"] or row["has_time_hint"]
            ),
            "legal_conversion_ready_targets": sum(1 for row in rows if row["legal_conversion_ready"]),
            "user_action_required_targets": sum(1 for row in rows if not row["legal_conversion_ready"]),
            "converted_datasets_now": 0,
            "evaluated_datasets_now": 0,
            "world_state_rows_generated": 0,
            "archives_extracted_now": 0,
        },
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["stage42_dt_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_raw_source_parseability_dry_run()

