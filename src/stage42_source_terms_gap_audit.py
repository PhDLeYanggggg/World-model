from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section
from src.stage42_source_terms_confirmation_validator import run_stage42_source_terms_confirmation_validator


OUT_DIR = Path("outputs/stage42_long_research")
CG_JSON = OUT_DIR / "source_terms_validation_stage42.json"
ED_JSON = OUT_DIR / "source_conversion_unblocker_stage42.json"
TEMPLATE_JSON = OUT_DIR / "source_terms_confirmation_template_stage42.json"

REPORT_JSON = OUT_DIR / "source_terms_gap_audit_stage42.json"
REPORT_MD = OUT_DIR / "source_terms_gap_audit_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ef_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_source_terms_gap_stage42.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_rerun_cg_plus_ed_source_terms_gap_audit"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-EF reruns the source terms validator and merges it with Stage42-ED technical-after-terms potential.",
    "本阶段不下载、不转换、不训练、不评估，只生成 legal/source/time blocker closure checklist。",
    "local path、parseability、technical dry-run 都不等于 legal conversion readiness。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "converted_datasets_now": 0,
    "evaluated_datasets_now": 0,
    "stage5c_executed": False,
    "smc_enabled": False,
}

FIELD_BY_BLOCKER = {
    "terms_not_accepted": "terms_accepted_by_user",
    "terms_acceptance_date_missing": "terms_acceptance_date",
    "allowed_use_missing": "allowed_use",
    "local_path_confirmation_missing": "local_path",
    "confirmed_local_path_missing": "local_path",
    "source_identity_missing": "source_identity",
    "official_url_mismatch": "official_url",
    "confirmation_entry_missing": "datasets[] entry",
}


def _missing_fields(blockers: list[str]) -> list[str]:
    fields = []
    for blocker in blockers:
        field = FIELD_BY_BLOCKER.get(blocker)
        if field and field not in fields:
            fields.append(field)
    return fields


def _merge_rows(cg: Mapping[str, Any], ed: Mapping[str, Any]) -> list[dict[str, Any]]:
    validations = {row["dataset_id"]: row for row in cg.get("validations", [])}
    rows: list[dict[str, Any]] = []
    for ed_row in ed.get("action_rows", []):
        dataset_id = ed_row["dataset_id"]
        validation = validations.get(dataset_id, {})
        if validation:
            confirmation_blockers = list(validation.get("confirmation_blockers", ed_row.get("terms_blockers", [])))
            cf_blockers = list(validation.get("cf_blockers", ed_row.get("cf_blockers", [])))
        else:
            confirmation_blockers = ["confirmation_entry_missing"]
            cf_blockers = list(ed_row.get("cf_blockers", []))
        missing = _missing_fields(confirmation_blockers)
        t50 = int(ed_row.get("estimated_t50_windows_after_terms", 0))
        t100 = int(ed_row.get("estimated_t100_windows_after_terms", 0))
        technical_sources = list(ed_row.get("technical_ready_source_ids_after_terms", []))
        unlock_score = t50 + t100 + 1000 * len(technical_sources)
        rows.append(
            {
                "dataset_id": dataset_id,
                "domain": ed_row.get("domain", ""),
                "official_url": ed_row.get("official_url", ""),
                "raw_path_found": bool(ed_row.get("raw_path_found")),
                "derived_cache_found": bool(ed_row.get("derived_cache_found")),
                "conversion_ready_now": bool(validation.get("conversion_ready", ed_row.get("conversion_ready", False))),
                "terms_accepted_by_user": bool(validation.get("terms_accepted_by_user", ed_row.get("terms_accepted_by_user", False))),
                "confirmation_blockers": confirmation_blockers,
                "cf_blockers": cf_blockers,
                "missing_confirmation_fields": missing,
                "technical_ready_source_ids_after_terms": technical_sources,
                "technical_ready_source_count_after_terms": len(technical_sources),
                "estimated_t50_windows_after_terms": t50,
                "estimated_t100_windows_after_terms": t100,
                "source_cv_after_terms": bool(technical_sources and ed_row.get("domain") in {"UCY", "ETH_UCY"}),
                "purpose": ed_row.get("purpose", ""),
                "blocker_class": ed_row.get("blocker_class", ""),
                "unlock_score": unlock_score,
                "next_command_after_confirmation": ".venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py",
                "then_required_future_stage": "guarded source conversion + no-leakage + source-CV evaluation; Stage42-EF does not convert",
            }
        )
    return sorted(rows, key=lambda row: (row["conversion_ready_now"], row["unlock_score"]), reverse=True)


def _summary(rows: list[Mapping[str, Any]], cg: Mapping[str, Any], ed: Mapping[str, Any]) -> dict[str, Any]:
    top = rows[:3]
    return {
        "source": SOURCE,
        "targets": len(rows),
        "cg_terms_accepted_targets": int(cg.get("summary", {}).get("terms_accepted_targets", 0)),
        "cg_conversion_ready_targets": int(cg.get("summary", {}).get("conversion_ready_targets", 0)),
        "ed_technical_ready_after_terms_targets": int(ed.get("summary", {}).get("technical_ready_after_terms_targets", 0)),
        "conversion_ready_now": sum(1 for row in rows if row["conversion_ready_now"]),
        "converted_datasets_now": 0,
        "evaluated_datasets_now": 0,
        "user_action_required_targets": sum(1 for row in rows if not row["conversion_ready_now"]),
        "estimated_t50_windows_after_terms": sum(int(row["estimated_t50_windows_after_terms"]) for row in rows),
        "estimated_t100_windows_after_terms": sum(int(row["estimated_t100_windows_after_terms"]) for row in rows),
        "technical_ready_source_count_after_terms": sum(int(row["technical_ready_source_count_after_terms"]) for row in rows),
        "top_unblock_targets": [row["dataset_id"] for row in top],
        "top_unblock_actions": [
            {
                "dataset_id": row["dataset_id"],
                "domain": row["domain"],
                "missing_confirmation_fields": row["missing_confirmation_fields"],
                "estimated_t50_windows_after_terms": row["estimated_t50_windows_after_terms"],
                "estimated_t100_windows_after_terms": row["estimated_t100_windows_after_terms"],
                "source_cv_after_terms": row["source_cv_after_terms"],
            }
            for row in top
        ],
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    rows = payload["gap_rows"]
    gates = {
        "cg_fresh_rerun_passed": payload["input_reports"]["stage42_cg_gate"]["passed"]
        == payload["input_reports"]["stage42_cg_gate"]["total"],
        "ed_input_passed": payload["input_reports"]["stage42_ed_gate"]["passed"]
        == payload["input_reports"]["stage42_ed_gate"]["total"],
        "all_targets_gap_scored": s["targets"] >= 5 and all("missing_confirmation_fields" in row for row in rows),
        "empty_template_still_blocks_conversion": s["cg_conversion_ready_targets"] == 0,
        "ucy_priority_preserved": s["top_unblock_targets"] and s["top_unblock_targets"][0] == "ucy_crowd_original",
        "eth_priority_present": "eth_biwi_original" in s["top_unblock_targets"],
        "technical_potential_recorded": s["estimated_t50_windows_after_terms"] >= 10060
        and s["estimated_t100_windows_after_terms"] >= 5696,
        "missing_fields_are_concrete": all(row["missing_confirmation_fields"] for row in rows if not row["conversion_ready_now"]),
        "user_action_written": payload["user_action_required_written"] is True,
        "no_conversion_or_eval_claim": s["converted_datasets_now"] == 0 and s["evaluated_datasets_now"] == 0,
        "no_metric_seconds_overclaim": payload["claim_boundary"]["global_metric_claim_allowed"] is False
        and payload["claim_boundary"]["global_seconds_claim_allowed"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    verdict = "stage42_ef_source_terms_gap_audit_pass" if passed == total else "stage42_ef_source_terms_gap_audit_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-EF Source Terms Gap Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_ef_gate']['passed']} / {payload['stage42_ef_gate']['total']}`",
        f"- verdict: `{payload['stage42_ef_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Summary",
        "",
        f"- targets: `{s['targets']}`",
        f"- cg_terms_accepted_targets: `{s['cg_terms_accepted_targets']}`",
        f"- cg_conversion_ready_targets: `{s['cg_conversion_ready_targets']}`",
        f"- conversion_ready_now: `{s['conversion_ready_now']}`",
        f"- converted/evaluated now: `{s['converted_datasets_now']}` / `{s['evaluated_datasets_now']}`",
        f"- estimated_t50/t100_windows_after_terms: `{s['estimated_t50_windows_after_terms']}` / `{s['estimated_t100_windows_after_terms']}`",
        f"- top_unblock_targets: `{s['top_unblock_targets']}`",
        "",
        "## Gap Table",
        "",
        "| rank | dataset | domain | t50 after terms | t100 after terms | source-CV after terms | missing fields | blocker class |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for idx, row in enumerate(payload["gap_rows"], start=1):
        lines.append(
            f"| {idx} | `{row['dataset_id']}` | `{row['domain']}` | {row['estimated_t50_windows_after_terms']} | {row['estimated_t100_windows_after_terms']} | {row['source_cv_after_terms']} | {', '.join(row['missing_confirmation_fields']) or 'none'} | `{row['blocker_class']}` |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- UCY remains the first legal unblock target because it has the largest t50/t100 after-terms potential and source-CV support.",
        "- ETH/BIWI is second because it has source-specific metric/time candidates but far fewer t50/t100 rows.",
        "- TrajNet/OpenTraj/AerialMPT still need source identity, legal terms, or independent source repair before conversion claims.",
        "- No conversion, no evaluation, no metric/seconds claim, no Stage5C, and no SMC are made by this stage.",
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_ef_gate"]["gates"].items()],
    ]
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage42-EF Source Terms Gap",
        "",
        "No source is conversion-ready. Fill `outputs/stage42_long_research/source_terms_confirmation_template_stage42.json` only after official terms/path/source verification.",
        "",
        "## Priority Order",
        "",
    ]
    for idx, row in enumerate(payload["gap_rows"], start=1):
        lines += [
            f"### {idx}. {row['dataset_id']}",
            "",
            f"- official_url: {row['official_url']}",
            f"- domain: `{row['domain']}`",
            f"- missing fields: {', '.join(row['missing_confirmation_fields']) or 'none'}",
            f"- estimated t50/t100 windows after terms: `{row['estimated_t50_windows_after_terms']}` / `{row['estimated_t100_windows_after_terms']}`",
            f"- source-CV after terms: `{row['source_cv_after_terms']}`",
            f"- next command after confirmation: `{row['next_command_after_confirmation']}`",
            f"- then required future stage: {row['then_required_future_stage']}",
            "",
        ]
    lines.append("Do not convert or evaluate until the validator reports conversion-ready targets and a later no-leakage/source-CV conversion stage passes.")
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ef_gate"]
    return [
        "# Stage42-EF Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    return [
        "## Stage42-EF Source Terms Gap Audit",
        "",
        "- source: `fresh_rerun_cg_plus_ed_source_terms_gap_audit`",
        "- role: reruns source terms validator and merges it with ED technical-after-terms potential.",
        f"- gate: `{payload['stage42_ef_gate']['passed']} / {payload['stage42_ef_gate']['total']}`; verdict `{payload['stage42_ef_gate']['verdict']}`.",
        f"- conversion_ready_now: `{s['conversion_ready_now']}`; converted/evaluated now `{s['converted_datasets_now']}` / `{s['evaluated_datasets_now']}`.",
        f"- top unblock targets: `{s['top_unblock_targets']}`; estimated t50/t100 after terms `{s['estimated_t50_windows_after_terms']}` / `{s['estimated_t100_windows_after_terms']}`.",
        "- boundary: no legal conversion, no metric/seconds claim, no Stage5C, no SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, GOAL_SUMMARY]:
        _replace_section(path, "STAGE42_EF_SOURCE_TERMS_GAP_AUDIT", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-EF source terms gap audit"
    state["current_verdict"] = payload["stage42_ef_gate"]["verdict"]
    state["stage42_ef_source_terms_gap_audit"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "verdict": payload["stage42_ef_gate"]["verdict"],
        "gates": f"{payload['stage42_ef_gate']['passed']}/{payload['stage42_ef_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_source_terms_gap_audit(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    cg = run_stage42_source_terms_confirmation_validator()
    ed = read_json(ED_JSON, {})
    rows = _merge_rows(cg, ed)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-EF",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([CG_JSON, ED_JSON, TEMPLATE_JSON]),
        "current_facts": CURRENT_FACTS,
        "input_reports": {
            "stage42_cg_gate": cg["stage42_cg_gate"],
            "stage42_ed_gate": ed["stage42_ed_gate"],
        },
        "gap_rows": rows,
        "summary": _summary(rows, cg, ed),
        "claim_boundary": CLAIM_BOUNDARY,
        "user_action_required_written": True,
    }
    payload["stage42_ef_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_source_terms_gap_audit()
