from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
EF_JSON = OUT_DIR / "source_terms_gap_audit_stage42.json"

REPORT_JSON = OUT_DIR / "source_terms_confirmation_intake_package_stage42.json"
REPORT_MD = OUT_DIR / "source_terms_confirmation_intake_package_stage42.md"
SCHEMA_JSON = OUT_DIR / "source_terms_confirmation_schema_stage42.json"
INTAKE_JSON = OUT_DIR / "source_terms_confirmation_intake_template_stage42.json"
USER_ACTION_MD = OUT_DIR / "user_action_required_source_terms_confirmation_intake_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_eh_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_source_terms_confirmation_intake_from_stage42_ef"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-EH turns the Stage42-EF source-terms blocker into a fillable confirmation/intake package.",
    "本阶段不下载、不转换、不训练、不评估。",
    "只有用户确认 official terms、allowed use、local path 和 source identity 后，未来阶段才允许转换。",
    "local path、parseability、technical dry-run 都不等于 legal conversion readiness。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

REQUIRED_FIELDS = [
    "terms_accepted_by_user",
    "terms_acceptance_date",
    "official_terms_url",
    "accepted_terms_version_or_access_date",
    "allowed_use",
    "redistribution_allowed",
    "derived_data_allowed",
    "local_path",
    "source_identity",
    "confirmed_by_user",
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


def _schema() -> dict[str, Any]:
    return {
        "source": SOURCE,
        "purpose": "Manual user confirmation schema. Do not treat a blank or partially filled record as permission.",
        "required_fields": REQUIRED_FIELDS,
        "field_rules": {
            "terms_accepted_by_user": "must be true only after the user manually accepts or verifies official dataset terms",
            "terms_acceptance_date": "YYYY-MM-DD; user-provided",
            "official_terms_url": "official page or official terms/license URL; not a random mirror",
            "accepted_terms_version_or_access_date": "terms version if known, otherwise date accessed by user",
            "allowed_use": "e.g. research_only, academic_noncommercial, commercial_allowed, unknown; user-provided",
            "redistribution_allowed": "true/false/unknown from official terms; user-provided",
            "derived_data_allowed": "true/false/unknown from official terms; user-provided",
            "local_path": "absolute local path to the official/source-identified dataset copy",
            "source_identity": "official archive/file/source name or source-specific subdirectory identity",
            "confirmed_by_user": "short user confirmation string; not filled by the agent",
        },
        "hard_blocks": [
            "terms_accepted_by_user is false",
            "allowed_use is blank or unknown",
            "local_path is blank",
            "source_identity is blank",
            "official_terms_url is blank or not official",
        ],
        "non_claims": [
            "This schema does not grant permission.",
            "This schema does not convert or evaluate data.",
            "This schema does not allow metric/seconds-level claims.",
        ],
    }


def _priority(row: Mapping[str, Any]) -> int:
    return int(row.get("estimated_t50_windows_after_terms", 0)) + int(row.get("estimated_t100_windows_after_terms", 0))


def _intake_rows(ef: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = sorted(ef.get("gap_rows", []), key=_priority, reverse=True)
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        dataset_id = row["dataset_id"]
        official_url = row.get("official_url", "")
        requires_new_official_url = official_url == "user_or_web_verified_official_url_required"
        out.append(
            {
                "priority_rank": idx,
                "dataset_id": dataset_id,
                "domain": row.get("domain", ""),
                "official_url_from_prior_audit": official_url,
                "requires_new_official_url": requires_new_official_url,
                "after_terms_potential": {
                    "estimated_t50_windows": int(row.get("estimated_t50_windows_after_terms", 0)),
                    "estimated_t100_windows": int(row.get("estimated_t100_windows_after_terms", 0)),
                    "source_cv_after_terms": bool(row.get("source_cv_after_terms", False)),
                    "technical_ready_source_ids": list(row.get("technical_ready_source_ids_after_terms", [])),
                },
                "current_blockers": {
                    "blocker_class": row.get("blocker_class", ""),
                    "missing_confirmation_fields": list(row.get("missing_confirmation_fields", [])),
                    "confirmation_blockers": list(row.get("confirmation_blockers", [])),
                    "cf_blockers": list(row.get("cf_blockers", [])),
                },
                "user_confirmation": {
                    "terms_accepted_by_user": False,
                    "terms_acceptance_date": "",
                    "official_terms_url": "" if requires_new_official_url else official_url,
                    "accepted_terms_version_or_access_date": "",
                    "allowed_use": "",
                    "redistribution_allowed": "unknown",
                    "derived_data_allowed": "unknown",
                    "local_path": "",
                    "source_identity": "",
                    "confirmed_by_user": "",
                    "notes": "",
                },
                "allowed_next_command_after_user_fills_fields": ".venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py",
                "future_stage_after_validator_passes": "guarded source conversion + no-leakage + source-CV evaluation",
                "agent_may_fill": False,
                "conversion_ready_now": False,
                "converted_now": False,
                "evaluated_now": False,
            }
        )
    return out


def _summary(rows: list[Mapping[str, Any]], ef: Mapping[str, Any]) -> dict[str, Any]:
    high_priority = [row for row in rows if row["after_terms_potential"]["estimated_t50_windows"] > 0]
    return {
        "source": SOURCE,
        "targets": len(rows),
        "high_priority_after_terms_targets": len(high_priority),
        "conversion_ready_now": 0,
        "converted_datasets_now": 0,
        "evaluated_datasets_now": 0,
        "estimated_t50_windows_after_terms": sum(row["after_terms_potential"]["estimated_t50_windows"] for row in rows),
        "estimated_t100_windows_after_terms": sum(row["after_terms_potential"]["estimated_t100_windows"] for row in rows),
        "top_unblock_targets": [row["dataset_id"] for row in rows[:3]],
        "first_action": "fill outputs/stage42_long_research/source_terms_confirmation_intake_template_stage42.json manually after official terms/path/source verification",
        "validator_command": ".venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py",
        "ef_verdict": ef.get("stage42_ef_gate", {}).get("verdict", ""),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    rows = payload["intake_rows"]
    gates = {
        "ef_input_passed": payload["input_reports"]["stage42_ef_gate"]["passed"]
        == payload["input_reports"]["stage42_ef_gate"]["total"],
        "schema_written": payload["schema_written"] is True,
        "intake_template_written": payload["intake_template_written"] is True,
        "required_fields_present": all(
            all(field in row["user_confirmation"] for field in REQUIRED_FIELDS) for row in rows
        ),
        "ucy_priority_preserved": s["top_unblock_targets"] and s["top_unblock_targets"][0] == "ucy_crowd_original",
        "eth_present": "eth_biwi_original" in s["top_unblock_targets"],
        "all_targets_require_user_confirmation": all(row["agent_may_fill"] is False for row in rows),
        "no_conversion_or_eval_claim": s["converted_datasets_now"] == 0 and s["evaluated_datasets_now"] == 0,
        "legal_blocker_preserved": s["conversion_ready_now"] == 0,
        "validator_command_recorded": bool(s["validator_command"]),
        "user_action_written": payload["user_action_required_written"] is True,
        "no_metric_seconds_overclaim": payload["claim_boundary"]["global_metric_claim_allowed"] is False
        and payload["claim_boundary"]["global_seconds_claim_allowed"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    verdict = "stage42_eh_source_terms_confirmation_intake_pass" if passed == total else "stage42_eh_source_terms_confirmation_intake_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-EH Source Terms Confirmation Intake Package",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_eh_gate']['passed']} / {payload['stage42_eh_gate']['total']}`",
        f"- verdict: `{payload['stage42_eh_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Summary",
        "",
        f"- targets: `{s['targets']}`",
        f"- high_priority_after_terms_targets: `{s['high_priority_after_terms_targets']}`",
        f"- conversion_ready_now: `{s['conversion_ready_now']}`",
        f"- converted/evaluated now: `{s['converted_datasets_now']}` / `{s['evaluated_datasets_now']}`",
        f"- estimated_t50/t100_windows_after_terms: `{s['estimated_t50_windows_after_terms']}` / `{s['estimated_t100_windows_after_terms']}`",
        f"- top_unblock_targets: `{s['top_unblock_targets']}`",
        f"- intake_template: `{INTAKE_JSON}`",
        f"- schema: `{SCHEMA_JSON}`",
        f"- validator_command: `{s['validator_command']}`",
        "",
        "## Intake Table",
        "",
        "| rank | dataset | domain | t50 after terms | t100 after terms | source-CV after terms | user fields required | agent may fill |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in payload["intake_rows"]:
        fields = ", ".join(REQUIRED_FIELDS)
        potential = row["after_terms_potential"]
        lines.append(
            f"| {row['priority_rank']} | `{row['dataset_id']}` | `{row['domain']}` | {potential['estimated_t50_windows']} | {potential['estimated_t100_windows']} | {potential['source_cv_after_terms']} | {fields} | {row['agent_may_fill']} |"
        )
    lines += [
        "",
        "## How To Use",
        "",
        "1. Open `outputs/stage42_long_research/source_terms_confirmation_intake_template_stage42.json`.",
        "2. For each dataset you want to unblock, manually verify the official terms/source page and fill every required field.",
        "3. Do not let the agent infer acceptance, allowed use, or source identity.",
        "4. Run `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py`.",
        "5. Only if the validator reports conversion-ready targets should a future guarded conversion/no-leakage/source-CV stage run.",
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_eh_gate"]["gates"].items()],
    ]
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage42-EH Source Terms Confirmation Intake",
        "",
        "No external source is conversion-ready yet. The agent must not fill legal acceptance fields for you.",
        "",
        "Fill this file after official verification:",
        "",
        f"`{INTAKE_JSON}`",
        "",
        "## Priority Targets",
        "",
    ]
    for row in payload["intake_rows"]:
        potential = row["after_terms_potential"]
        lines += [
            f"### {row['priority_rank']}. {row['dataset_id']}",
            "",
            f"- official_url_from_prior_audit: {row['official_url_from_prior_audit']}",
            f"- domain: `{row['domain']}`",
            f"- estimated t50/t100 after terms: `{potential['estimated_t50_windows']}` / `{potential['estimated_t100_windows']}`",
            f"- source-CV after terms: `{potential['source_cv_after_terms']}`",
            f"- required fields: `{', '.join(REQUIRED_FIELDS)}`",
            f"- next validator command: `{row['allowed_next_command_after_user_fills_fields']}`",
            "",
        ]
    lines += [
        "## Safety Rule",
        "",
        "Do not download, convert, evaluate, or claim metric/seconds-level evidence from these sources until a later validator + no-leakage + source-CV stage passes.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_eh_gate"]
    return [
        "# Stage42-EH Gate",
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
        "## Stage42-EH Source Terms Confirmation Intake Package",
        "",
        "- source: `fresh_source_terms_confirmation_intake_from_stage42_ef`",
        "- role: turns the Stage42-EF source terms blocker into a fillable, auditable confirmation package.",
        f"- gate: `{payload['stage42_eh_gate']['passed']} / {payload['stage42_eh_gate']['total']}`; verdict `{payload['stage42_eh_gate']['verdict']}`.",
        f"- intake template: `{INTAKE_JSON}`; schema: `{SCHEMA_JSON}`.",
        f"- top unblock targets: `{s['top_unblock_targets']}`; after-terms t50/t100 potential `{s['estimated_t50_windows_after_terms']}` / `{s['estimated_t100_windows_after_terms']}`.",
        "- conversion_ready_now remains `0`; this stage does not download, convert, train, evaluate, or make metric/seconds claims.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, GOAL_SUMMARY]:
        _replace_section(path, "STAGE42_EH_SOURCE_TERMS_CONFIRMATION_INTAKE", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-EH source terms confirmation intake package"
    state["current_verdict"] = payload["stage42_eh_gate"]["verdict"]
    state["stage42_eh_source_terms_confirmation_intake"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "schema": str(SCHEMA_JSON),
        "intake_template": str(INTAKE_JSON),
        "user_action_required": str(USER_ACTION_MD),
        "verdict": payload["stage42_eh_gate"]["verdict"],
        "gates": f"{payload['stage42_eh_gate']['passed']}/{payload['stage42_eh_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_source_terms_confirmation_intake(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ef = read_json(EF_JSON, {})
    rows = _intake_rows(ef)
    schema = _schema()
    intake = {
        "source": SOURCE,
        "purpose": "Manual intake template. Fill only after official terms/path/source verification.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "agent_may_fill_legal_acceptance": False,
        "datasets": rows,
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-EH",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([EF_JSON]),
        "current_facts": CURRENT_FACTS,
        "input_reports": {"stage42_ef_gate": ef.get("stage42_ef_gate", {})},
        "schema": schema,
        "intake_rows": rows,
        "summary": _summary(rows, ef),
        "claim_boundary": CLAIM_BOUNDARY,
        "schema_written": True,
        "intake_template_written": True,
        "user_action_required_written": True,
    }
    payload["stage42_eh_gate"] = _gate(payload)
    write_json(SCHEMA_JSON, schema)
    write_json(INTAKE_JSON, intake)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_source_terms_confirmation_intake()
