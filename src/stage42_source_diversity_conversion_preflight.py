from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_independent_t50_source_inventory import _annotate_usage, _current_usage, _parse_track_file
from src.stage42_source_diversity_acquisition_package import OFFICIAL_TARGETS


OUT_DIR = Path("outputs/stage42_long_research")
CD_JSON = OUT_DIR / "source_diversity_acquisition_package_stage42.json"

REPORT_JSON = OUT_DIR / "source_diversity_conversion_preflight_stage42.json"
REPORT_MD = OUT_DIR / "source_diversity_conversion_preflight_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ce_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_source_conversion_preflight_stage42.md"

MAX_FILES_PER_TARGET = 300
TRACK_SUFFIXES = {".txt", ".csv", ".ndjson"}

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-CE 是 local path conversion preflight，不转换数据，不训练模型，不调 threshold。",
    "本轮只检查本地路径结构和 track-like 文件；不绕过 license，不下载数据。",
    "local path found 不等于 legal / converted / evaluated。",
    "alternate representation 不等于 independent held-out source。",
    "future endpoints / waypoints 只允许作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "Stage5C 未执行，SMC 未启用。",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_track_files(paths: Iterable[str], max_files: int = MAX_FILES_PER_TARGET) -> list[Path]:
    files: list[Path] = []
    seen: set[str] = set()
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            continue
        candidates = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        for candidate in candidates:
            if candidate.suffix.lower() not in TRACK_SUFFIXES:
                continue
            resolved = str(candidate.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(candidate)
            if len(files) >= max_files:
                return files
    return files


def _file_rows_for_target(target: Mapping[str, Any], usage: Mapping[str, Any]) -> list[dict[str, Any]]:
    parsed = [_parse_track_file(path) for path in _iter_track_files(target["local_path_candidates"])]
    annotated = _annotate_usage(parsed, usage)
    return sorted(annotated, key=lambda row: (-int(row["estimated_windows"]["t50"]), str(row["path"])))


def _summarize_target(target: Mapping[str, Any], usage: Mapping[str, Any]) -> dict[str, Any]:
    local_paths = []
    for raw in target["local_path_candidates"]:
        path = Path(raw)
        local_paths.append(
            {
                "path": raw,
                "exists": path.exists(),
                "is_dir": path.is_dir(),
                "is_file": path.is_file(),
            }
        )
    rows = _file_rows_for_target(target, usage)
    t50_files = [row for row in rows if row["t50_capable"]]
    t100_files = [row for row in rows if row["t100_capable"]]
    independent = [row for row in rows if row["final_status"] == "candidate_t50_independent_source"]
    alternates = [row for row in rows if row["final_status"] == "alternate_representation_of_current_source"]
    already_used = [row for row in rows if row["final_status"] == "already_in_current_combined_split"]
    diagnostic = [
        row
        for row in rows
        if row["final_status"] in {"diagnostic_or_simulation_only", "diagnostic_sdd_or_stanford_derived"} and row["t50_capable"]
    ]
    terms_blocked = bool(target["requires_manual_terms_acceptance"] or target["requires_login_or_application"])
    schema_possible = bool(rows and any(row["parsed_rows"] >= 10 for row in rows))
    source_cv_preflight_ready = bool(independent and not terms_blocked)
    return {
        "id": target["id"],
        "name": target["name"],
        "priority": target["priority"],
        "target_blocker": target["target_blocker"],
        "official_url": target["official_url"],
        "source_confidence": target["source_confidence"],
        "requires_manual_terms_acceptance": target["requires_manual_terms_acceptance"],
        "requires_login_or_application": target["requires_login_or_application"],
        "auto_download_allowed": target["auto_download_allowed"],
        "local_paths": local_paths,
        "local_path_found": any(row["exists"] for row in local_paths),
        "track_like_files_scanned": len(rows),
        "schema_possible": schema_possible,
        "t50_capable_files": len(t50_files),
        "t100_capable_files": len(t100_files),
        "independent_t50_candidate_files": len(independent),
        "alternate_current_source_files": len(alternates),
        "already_used_files": len(already_used),
        "diagnostic_t50_files": len(diagnostic),
        "legal_terms_blocked": terms_blocked,
        "source_cv_preflight_ready": source_cv_preflight_ready,
        "converted_now": False,
        "evaluated_now": False,
        "top_files": [
            {
                "source_name": row["source_name"],
                "path": row["path"],
                "parsed_rows": row["parsed_rows"],
                "max_track_points": row["max_track_points"],
                "t50_windows": row["estimated_windows"]["t50"],
                "t100_windows": row["estimated_windows"]["t100"],
                "final_status": row["final_status"],
                "next_step": row["final_next_step"],
            }
            for row in rows[:12]
        ],
        "next_action": _next_action(target, independent, terms_blocked, schema_possible),
    }


def _next_action(target: Mapping[str, Any], independent: list[Mapping[str, Any]], terms_blocked: bool, schema_possible: bool) -> str:
    if not schema_possible:
        return "provide a legal local path containing parseable trajectory rows"
    if terms_blocked:
        return "verify/accept official dataset terms before conversion; local parseability is not legal permission"
    if not independent:
        return "rebuild source split or provide independent source; current files are already-used, alternate, short, or diagnostic"
    return "ready for a future conversion/no-leakage/source-CV script, but not converted in Stage42-CE"


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    cd = _load_json(CD_JSON)
    usage = _current_usage()
    target_summaries = [_summarize_target(target, usage) for target in OFFICIAL_TARGETS]
    summary = {
        "source": "fresh_stage42_ce_source_diversity_conversion_preflight",
        "targets_checked": len(target_summaries),
        "targets_with_local_path": sum(1 for row in target_summaries if row["local_path_found"]),
        "targets_with_schema_possible": sum(1 for row in target_summaries if row["schema_possible"]),
        "targets_with_t50_files": sum(1 for row in target_summaries if row["t50_capable_files"] > 0),
        "targets_with_t100_files": sum(1 for row in target_summaries if row["t100_capable_files"] > 0),
        "targets_with_independent_t50_candidates": sum(1 for row in target_summaries if row["independent_t50_candidate_files"] > 0),
        "targets_source_cv_ready_now": sum(1 for row in target_summaries if row["source_cv_preflight_ready"]),
        "converted_datasets_now": 0,
        "evaluated_datasets_now": 0,
        "source_diversity_repair_ready_now": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    payload: dict[str, Any] = {
        "source": "fresh_stage42_ce_source_diversity_conversion_preflight",
        "stage": "Stage42-CE Source Diversity Conversion Preflight",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(CD_JSON)]),
        "current_facts": CURRENT_FACTS,
        "input_reports": {
            "stage42_cd_verdict": cd["stage42_cd_gate"]["verdict"],
            "stage42_cd_source_diversity_repair_ready_now": cd["summary"]["source_diversity_repair_ready_now"],
        },
        "summary": summary,
        "target_summaries": target_summaries,
        "user_action_required": [
            {
                "priority": row["priority"],
                "target": row["name"],
                "official_url": row["official_url"],
                "action": row["next_action"],
            }
            for row in target_summaries
            if not row["source_cv_preflight_ready"]
        ],
        "claim_boundary": {
            "preflight_counted_as_conversion": False,
            "local_path_counted_as_legal_permission": False,
            "converted_dataset_claim": False,
            "evaluated_dataset_claim": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_ce_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    gates = {
        "cd_input_verified": payload["input_reports"]["stage42_cd_verdict"]
        == "stage42_cd_source_diversity_acquisition_package_pass",
        "targets_checked": summary["targets_checked"] >= 5,
        "local_paths_inspected": summary["targets_with_local_path"] >= 1,
        "schema_preflight_done": summary["targets_with_schema_possible"] >= 1,
        "source_cv_not_overclaimed": summary["source_diversity_repair_ready_now"] is False,
        "no_conversion_claim": payload["claim_boundary"]["preflight_counted_as_conversion"] is False
        and payload["claim_boundary"]["converted_dataset_claim"] is False,
        "no_evaluation_claim": payload["claim_boundary"]["evaluated_dataset_claim"] is False,
        "legal_blockers_explicit": any(row["legal_terms_blocked"] for row in payload["target_summaries"]),
        "user_action_written": bool(payload["user_action_required"]),
        "no_metric_seconds_overclaim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_ce_source_diversity_conversion_preflight_pass" if passed == total else "stage42_ce_source_diversity_conversion_preflight_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-CE Source Diversity Conversion Preflight",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_ce_gate']['passed']} / {payload['stage42_ce_gate']['total']}`",
        f"- verdict: `{payload['stage42_ce_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- targets_checked: `{s['targets_checked']}`",
        f"- targets_with_local_path: `{s['targets_with_local_path']}`",
        f"- targets_with_schema_possible: `{s['targets_with_schema_possible']}`",
        f"- targets_with_t50_files: `{s['targets_with_t50_files']}`",
        f"- targets_with_t100_files: `{s['targets_with_t100_files']}`",
        f"- targets_with_independent_t50_candidates: `{s['targets_with_independent_t50_candidates']}`",
        f"- targets_source_cv_ready_now: `{s['targets_source_cv_ready_now']}`",
        f"- converted_datasets_now: `{s['converted_datasets_now']}`",
        f"- evaluated_datasets_now: `{s['evaluated_datasets_now']}`",
        f"- source_diversity_repair_ready_now: `{s['source_diversity_repair_ready_now']}`",
        "",
        "## Target Preflight Table",
        "",
        "| target | local path | schema possible | t50 files | t100 files | independent t50 | legal blocked | source-CV ready | next action |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["target_summaries"]:
        lines.append(
            f"| `{row['id']}` | {row['local_path_found']} | {row['schema_possible']} | {row['t50_capable_files']} | "
            f"{row['t100_capable_files']} | {row['independent_t50_candidate_files']} | {row['legal_terms_blocked']} | "
            f"{row['source_cv_preflight_ready']} | {row['next_action']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Stage42-CE is a local conversion preflight, not conversion.",
        "- Local parseability and local path existence are not legal permission.",
        "- No target is counted as source-diversity repair in this stage.",
        "- Any future conversion must rebuild source-level split and rerun no-leakage/source-CV/final test.",
    ]
    return lines


def _render_user_actions(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage42-CE Conversion Preflight",
        "",
        "The local paths were inspected, but source-diversity repair is still not ready. Required actions:",
        "",
    ]
    for row in payload["user_action_required"]:
        lines += [
            f"## {row['priority'].upper()} - {row['target']}",
            "",
            f"- official_url: {row['official_url']}",
            f"- action: {row['action']}",
            "",
        ]
    lines.append("Do not count local paths, alternate formats, or parseable files as converted/evaluated evidence.")
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ce_gate"]
    lines = [
        "# Stage42-CE Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | {bool(value)} |")
    return lines


def run_stage42_source_diversity_conversion_preflight() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    write_md(USER_ACTION_MD, _render_user_actions(payload))
    return payload


if __name__ == "__main__":
    result = run_stage42_source_diversity_conversion_preflight()
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False, sort_keys=True))
