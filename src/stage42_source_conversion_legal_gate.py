from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
CE_JSON = OUT_DIR / "source_diversity_conversion_preflight_stage42.json"

REPORT_JSON = OUT_DIR / "source_conversion_legal_gate_stage42.json"
REPORT_MD = OUT_DIR / "source_conversion_legal_gate_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_cf_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_source_legal_gate_stage42.md"
CONFIRMATION_TEMPLATE_JSON = OUT_DIR / "source_terms_confirmation_template_stage42.json"

SOURCE = "fresh_stage42_cf_source_conversion_legal_gate"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-CF 是 source conversion legal gate，不下载、不转换、不训练、不评估。",
    "local path found / schema_possible 不等于 legal permission。",
    "terms confirmed 必须是用户或官方条款确认后的显式记录，不能由脚本自动伪造。",
    "future endpoints / waypoints 只允许作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "Stage5C 未执行，SMC 未启用。",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _decide_target(row: Mapping[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if not row["local_path_found"]:
        blockers.append("local_path_missing")
    if not row["schema_possible"]:
        blockers.append("schema_not_parseable")
    if row["legal_terms_blocked"]:
        blockers.append("manual_terms_or_application_required")
    if row["independent_t50_candidate_files"] <= 0:
        blockers.append("no_independent_t50_candidate")
    if row["source_cv_preflight_ready"]:
        blockers.append("requires_explicit_terms_confirmation_before_conversion")

    conversion_allowed_now = False
    if not blockers:
        # This branch is intentionally unreachable for current Stage42-CE evidence.
        # A future stage may allow it only after explicit user/legal confirmation is supplied.
        conversion_allowed_now = True

    return {
        "id": row["id"],
        "name": row["name"],
        "priority": row["priority"],
        "official_url": row["official_url"],
        "local_path_found": row["local_path_found"],
        "schema_possible": row["schema_possible"],
        "t50_capable_files": row["t50_capable_files"],
        "t100_capable_files": row["t100_capable_files"],
        "independent_t50_candidate_files": row["independent_t50_candidate_files"],
        "legal_terms_blocked": row["legal_terms_blocked"],
        "source_cv_preflight_ready": row["source_cv_preflight_ready"],
        "conversion_allowed_now": conversion_allowed_now,
        "converted_now": False,
        "evaluated_now": False,
        "blockers": blockers,
        "required_confirmation_fields": [
            "dataset_id",
            "official_url",
            "terms_accepted_by_user",
            "terms_acceptance_date",
            "allowed_use",
            "local_path",
            "source_identity",
            "notes",
        ],
        "next_action": _next_action(row, blockers),
    }


def _next_action(row: Mapping[str, Any], blockers: list[str]) -> str:
    if "local_path_missing" in blockers:
        return "provide a legal local path for this official source"
    if "schema_not_parseable" in blockers:
        return "provide parseable trajectory files or a loader mapping"
    if "manual_terms_or_application_required" in blockers:
        return "verify/accept official dataset terms and record explicit confirmation before conversion"
    if "no_independent_t50_candidate" in blockers:
        return "provide or isolate an independent t50-capable source split before source-CV"
    if "requires_explicit_terms_confirmation_before_conversion" in blockers:
        return "fill the source terms confirmation template; rerun conversion only after explicit confirmation"
    return "ready only for a future guarded conversion stage"


def _confirmation_template(decisions: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "source": SOURCE,
        "purpose": "Template only. Fill manually after official terms/path verification; do not treat this template as permission.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "terms_confirmation_is_currently_absent": True,
        "datasets": [
            {
                "dataset_id": row["id"],
                "official_url": row["official_url"],
                "terms_accepted_by_user": False,
                "terms_acceptance_date": "",
                "allowed_use": "",
                "local_path": "",
                "source_identity": "",
                "notes": row["next_action"],
            }
            for row in decisions
        ],
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    decisions = payload["target_decisions"]
    summary = payload["summary"]
    gates = {
        "ce_input_verified": payload["input_reports"]["stage42_ce_verdict"]
        == "stage42_ce_source_diversity_conversion_preflight_pass",
        "legal_gate_built": len(decisions) >= 5,
        "confirmation_template_written": payload["confirmation_template"]["terms_confirmation_is_currently_absent"] is True,
        "all_targets_have_decisions": all("conversion_allowed_now" in row for row in decisions),
        "blocked_targets_have_reasons": all(row["blockers"] for row in decisions),
        "no_conversion_allowed_without_confirmation": summary["conversion_allowed_now_count"] == 0,
        "no_conversion_claim": summary["converted_datasets_now"] == 0,
        "no_evaluation_claim": summary["evaluated_datasets_now"] == 0,
        "user_action_written": bool(payload["user_action_required"]),
        "source_cv_not_overclaimed": summary["source_cv_ready_now"] == 0,
        "no_metric_seconds_overclaim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_cf_source_conversion_legal_gate_pass" if passed == total else "stage42_cf_source_conversion_legal_gate_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ce = _load_json(CE_JSON)
    decisions = [_decide_target(row) for row in ce["target_summaries"]]
    summary = {
        "source": SOURCE,
        "targets_checked": len(decisions),
        "local_paths_present": sum(1 for row in decisions if row["local_path_found"]),
        "schema_possible_targets": sum(1 for row in decisions if row["schema_possible"]),
        "targets_with_t50_files": sum(1 for row in decisions if row["t50_capable_files"] > 0),
        "targets_with_t100_files": sum(1 for row in decisions if row["t100_capable_files"] > 0),
        "source_cv_ready_now": sum(1 for row in decisions if row["source_cv_preflight_ready"]),
        "conversion_allowed_now_count": sum(1 for row in decisions if row["conversion_allowed_now"]),
        "converted_datasets_now": 0,
        "evaluated_datasets_now": 0,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    confirmation_template = _confirmation_template(decisions)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-CF Source Conversion Legal Gate",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(CE_JSON)]),
        "current_facts": CURRENT_FACTS,
        "input_reports": {
            "stage42_ce_verdict": ce["stage42_ce_gate"]["verdict"],
            "stage42_ce_source_diversity_repair_ready_now": ce["summary"]["source_diversity_repair_ready_now"],
        },
        "summary": summary,
        "target_decisions": decisions,
        "confirmation_template": confirmation_template,
        "user_action_required": [
            {
                "priority": row["priority"],
                "target": row["name"],
                "official_url": row["official_url"],
                "blockers": row["blockers"],
                "action": row["next_action"],
            }
            for row in decisions
            if not row["conversion_allowed_now"]
        ],
        "claim_boundary": {
            "legal_gate_counted_as_conversion": False,
            "local_path_counted_as_permission": False,
            "converted_dataset_claim": False,
            "evaluated_dataset_claim": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_cf_gate"] = _gate(payload)
    return payload


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-CF Source Conversion Legal Gate",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_cf_gate']['passed']} / {payload['stage42_cf_gate']['total']}`",
        f"- verdict: `{payload['stage42_cf_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- targets_checked: `{s['targets_checked']}`",
        f"- local_paths_present: `{s['local_paths_present']}`",
        f"- schema_possible_targets: `{s['schema_possible_targets']}`",
        f"- targets_with_t50_files: `{s['targets_with_t50_files']}`",
        f"- targets_with_t100_files: `{s['targets_with_t100_files']}`",
        f"- source_cv_ready_now: `{s['source_cv_ready_now']}`",
        f"- conversion_allowed_now_count: `{s['conversion_allowed_now_count']}`",
        f"- converted_datasets_now: `{s['converted_datasets_now']}`",
        f"- evaluated_datasets_now: `{s['evaluated_datasets_now']}`",
        "",
        "## Target Decisions",
        "",
        "| target | local path | schema | t50 files | independent t50 | legal blocked | conversion allowed now | blockers |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["target_decisions"]:
        lines.append(
            f"| `{row['id']}` | {row['local_path_found']} | {row['schema_possible']} | {row['t50_capable_files']} | "
            f"{row['independent_t50_candidate_files']} | {row['legal_terms_blocked']} | {row['conversion_allowed_now']} | "
            f"{', '.join(row['blockers'])} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Stage42-CF intentionally allows zero conversions right now.",
        "- This is a guardrail: future conversion requires explicit terms confirmation plus independent source identity.",
        "- The generated confirmation template is not permission; it is a checklist the user must fill after official terms/path verification.",
        "- No metric/seconds-level, true-3D, Stage5C, or SMC claim is introduced.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_cf_gate"]
    lines = [
        "# Stage42-CF Gate",
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


def _render_user_actions(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage42-CF Source Legal Gate",
        "",
        "No source-diversity target is currently conversion-allowed. Required actions:",
        "",
    ]
    for row in payload["user_action_required"]:
        lines += [
            f"## {row['priority'].upper()} - {row['target']}",
            "",
            f"- official_url: {row['official_url']}",
            f"- blockers: {', '.join(row['blockers'])}",
            f"- action: {row['action']}",
            "",
        ]
    lines += [
        "Fill `outputs/stage42_long_research/source_terms_confirmation_template_stage42.json` only after official terms/path verification.",
        "Do not treat the template, local files, or parseability as permission.",
    ]
    return lines


def run_stage42_source_conversion_legal_gate() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_json(CONFIRMATION_TEMPLATE_JSON, payload["confirmation_template"])
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    write_md(USER_ACTION_MD, _render_user_actions(payload))
    return payload


if __name__ == "__main__":
    result = run_stage42_source_conversion_legal_gate()
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False, sort_keys=True))
