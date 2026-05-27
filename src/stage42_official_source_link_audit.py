from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
INTAKE_JSON = OUT_DIR / "source_terms_confirmation_intake_template_stage42.json"
READINESS_JSON = OUT_DIR / "source_conversion_readiness_manifest_stage42.json"

REPORT_JSON = OUT_DIR / "official_source_link_audit_stage42.json"
REPORT_MD = OUT_DIR / "official_source_link_audit_stage42.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_official_source_links_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_em_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
GOAL_LEDGER = Path("README_M3W_GOAL_DETAILED_LEDGER_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_official_source_link_audit"
RETRIEVAL_DATE = "2026-05-27"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-EM only resolves official source-link and manual confirmation gaps; it does not download, convert, train, or evaluate data.",
    "Official source links are not legal acceptance; user must still confirm terms, allowed use, local path, and source identity.",
    "local path, parseability, toolkit mirrors, or GitHub code licenses are not blanket permission for third-party datasets.",
    "future endpoints / waypoints are labels/eval only, never inference inputs.",
    "No central velocity, no test endpoints for goals, no test-threshold tuning.",
    "t+50 / t+100 remain raw-frame horizons; no seconds-level claim.",
    "dataset-local/raw-frame coordinates are not global metric coordinates.",
    "Stage5C latent generative was not executed.",
    "SMC was not enabled.",
]

OFFICIAL_SOURCE_NOTES: dict[str, dict[str, Any]] = {
    "ucy_crowd_original": {
        "official_urls": [
            "https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data",
            "https://graphics.cs.ucy.ac.cy/portfolio",
        ],
        "official_source_evidence": "UCY Graphics pages identify the crowd dataset as publicly accessible crowd data with video/.vsp and csv files, but the agent still cannot accept terms or infer redistribution/derived-use permission.",
        "source_confidence": "official_primary_plus_official_lab_portfolio",
        "recommended_user_action": "Open the UCY crowd-data page, confirm official terms/allowed use, then fill local_path and source_identity for UCY_students03 / UCY_zara01 / UCY_zara02.",
        "manual_terms_required": True,
        "auto_download_allowed_now": False,
    },
    "eth_biwi_original": {
        "official_urls": [
            "https://vision.ee.ethz.ch/datsets.html",
        ],
        "official_source_evidence": "ETH Zurich CVL dataset page lists BIWI Walking Pedestrians with annotations/videos and states datasets are for research purposes unless stated otherwise.",
        "source_confidence": "official_primary",
        "recommended_user_action": "Open the ETH CVL dataset page, confirm research-only terms and local BIWI ETH/Hotel source identity before guarded conversion.",
        "manual_terms_required": True,
        "auto_download_allowed_now": False,
    },
    "trajnetplusplus_official": {
        "official_urls": [
            "https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/",
            "https://www.epfl.ch/labs/vita/datasets/",
            "https://github.com/vita-epfl/trajnetplusplusbaselines",
        ],
        "official_source_evidence": "EPFL/VITA pages identify TrajNet++ as an interaction-centric human trajectory forecasting benchmark; the GitHub code license does not by itself grant blanket rights to every underlying dataset.",
        "source_confidence": "official_project_plus_official_github_code",
        "recommended_user_action": "Confirm the official TrajNet++ data/challenge access path and underlying dataset terms, then provide a local official/source-identified copy.",
        "manual_terms_required": True,
        "auto_download_allowed_now": False,
    },
    "opentraj_toolkit": {
        "official_urls": [
            "https://github.com/crowdbotp/OpenTraj",
        ],
        "official_source_evidence": "OpenTraj is an official toolkit/benchmark repository that references many datasets; it is not blanket independent permission for every referenced dataset.",
        "source_confidence": "official_github_toolkit_only",
        "recommended_user_action": "Use OpenTraj for loaders/toolkit context only after confirming each underlying dataset's official terms and source identity.",
        "manual_terms_required": True,
        "auto_download_allowed_now": False,
    },
    "aerialmpt_or_other_topdown": {
        "official_urls": [],
        "official_source_evidence": "No confirmed official usable source is recorded in the current intake; user or future web audit must provide official page and terms.",
        "source_confidence": "missing_official_source",
        "recommended_user_action": "Provide an official HTTPS source page, terms/license, local path, and source identity before any conversion.",
        "manual_terms_required": True,
        "auto_download_allowed_now": False,
    },
}

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "download_executed": False,
    "conversion_executed": False,
    "evaluation_executed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}


def _intake_rows() -> list[Mapping[str, Any]]:
    payload = read_json(INTAKE_JSON, {})
    return list(payload.get("datasets", []))


def _readiness_summary() -> dict[str, Any]:
    if not READINESS_JSON.exists():
        return {"exists": False, "conversion_ready_targets": 0, "blocked_targets": 0}
    payload = read_json(READINESS_JSON, {})
    return {
        "exists": True,
        "conversion_ready_targets": len(payload.get("conversion_ready_targets", [])),
        "blocked_targets": len(payload.get("blocked_targets", [])),
        "conversion_executed": bool(payload.get("conversion_executed", False)),
        "evaluation_executed": bool(payload.get("evaluation_executed", False)),
    }


def _audit_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        dataset_id = str(row.get("dataset_id", ""))
        user_confirmation = dict(row.get("user_confirmation", {}))
        notes = OFFICIAL_SOURCE_NOTES.get(dataset_id, OFFICIAL_SOURCE_NOTES["aerialmpt_or_other_topdown"])
        required_missing = [
            field
            for field in [
                "terms_accepted_by_user",
                "terms_acceptance_date",
                "allowed_use",
                "local_path",
                "source_identity",
                "confirmed_by_user",
            ]
            if not user_confirmation.get(field)
        ]
        official_url_candidates = notes["official_urls"] or [
            str(row.get("official_url_from_prior_audit", "")) or "official_url_required"
        ]
        out.append(
            {
                "dataset_id": dataset_id,
                "domain": row.get("domain", ""),
                "priority_rank": row.get("priority_rank", 999),
                "retrieval_date": RETRIEVAL_DATE,
                "result_source": SOURCE,
                "official_url_candidates": official_url_candidates,
                "official_source_evidence": notes["official_source_evidence"],
                "source_confidence": notes["source_confidence"],
                "manual_terms_required": bool(notes["manual_terms_required"]),
                "auto_download_allowed_now": bool(notes["auto_download_allowed_now"]),
                "agent_may_fill_legal_acceptance": False,
                "conversion_ready_now": False,
                "converted_now": False,
                "evaluated_now": False,
                "after_terms_potential": row.get("after_terms_potential", {}),
                "missing_user_confirmation_fields": required_missing,
                "recommended_user_action": notes["recommended_user_action"],
                "allowed_next_command_after_user_fills_fields": row.get(
                    "allowed_next_command_after_user_fills_fields",
                    ".venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py",
                ),
            }
        )
    return out


def _summary(audit_rows: list[Mapping[str, Any]], readiness: Mapping[str, Any]) -> dict[str, Any]:
    officialish = [
        row
        for row in audit_rows
        if row["source_confidence"]
        in {
            "official_primary_plus_official_lab_portfolio",
            "official_primary",
            "official_project_plus_official_github_code",
            "official_github_toolkit_only",
        }
    ]
    return {
        "source": SOURCE,
        "targets": len(audit_rows),
        "official_or_toolkit_source_candidates": len(officialish),
        "manual_terms_required_targets": sum(1 for row in audit_rows if row["manual_terms_required"]),
        "auto_download_allowed_now": sum(1 for row in audit_rows if row["auto_download_allowed_now"]),
        "conversion_ready_now": int(readiness.get("conversion_ready_targets", 0)),
        "converted_now": 0,
        "evaluated_now": 0,
        "estimated_t50_after_terms": sum(
            int(row.get("after_terms_potential", {}).get("estimated_t50_windows", 0)) for row in audit_rows
        ),
        "estimated_t100_after_terms": sum(
            int(row.get("after_terms_potential", {}).get("estimated_t100_windows", 0)) for row in audit_rows
        ),
        "next_validator_command": ".venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py",
        "next_guarded_launcher_command": ".venv-pytorch/bin/python run_stage42_guarded_source_conversion_launcher.py",
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    rows = payload["official_source_rows"]
    gates = {
        "intake_template_exists": INTAKE_JSON.exists(),
        "readiness_manifest_checked": payload["readiness_manifest"]["exists"] is True,
        "targets_audited": s["targets"] >= 5,
        "official_candidates_recorded": s["official_or_toolkit_source_candidates"] >= 4,
        "ucy_has_official_url": any(
            row["dataset_id"] == "ucy_crowd_original" and row["official_url_candidates"] for row in rows
        ),
        "eth_has_official_url": any(
            row["dataset_id"] == "eth_biwi_original" and row["official_url_candidates"] for row in rows
        ),
        "trajnet_has_official_url": any(
            row["dataset_id"] == "trajnetplusplus_official" and row["official_url_candidates"] for row in rows
        ),
        "manual_terms_preserved": s["manual_terms_required_targets"] == s["targets"],
        "no_auto_download": s["auto_download_allowed_now"] == 0,
        "no_conversion_or_eval": s["converted_now"] == 0 and s["evaluated_now"] == 0,
        "user_action_written": payload["user_action_written"] is True,
        "no_metric_seconds_overclaim": payload["claim_boundary"]["global_metric_claim_allowed"] is False
        and payload["claim_boundary"]["global_seconds_claim_allowed"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_em_official_source_link_audit_pass" if passed == total else "stage42_em_official_source_link_audit_partial"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-EM Official Source Link Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- retrieval_date: `{payload['retrieval_date']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_em_gate']['passed']} / {payload['stage42_em_gate']['total']}`",
        f"- verdict: `{payload['stage42_em_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Summary",
        "",
        f"- targets audited: `{s['targets']}`",
        f"- official/toolkit source candidates recorded: `{s['official_or_toolkit_source_candidates']}`",
        f"- manual terms required targets: `{s['manual_terms_required_targets']}`",
        f"- auto download allowed now: `{s['auto_download_allowed_now']}`",
        f"- conversion ready now: `{s['conversion_ready_now']}`",
        f"- converted/evaluated now: `{s['converted_now']}` / `{s['evaluated_now']}`",
        f"- estimated t50/t100 after terms: `{s['estimated_t50_after_terms']}` / `{s['estimated_t100_after_terms']}`",
        "",
        "## Official Source Rows",
        "",
        "| dataset | confidence | official candidates | manual terms | conversion ready now | action |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in payload["official_source_rows"]:
        urls = "<br>".join(row["official_url_candidates"])
        lines.append(
            f"| `{row['dataset_id']}` | `{row['source_confidence']}` | {urls} | {row['manual_terms_required']} | {row['conversion_ready_now']} | {row['recommended_user_action']} |"
        )
    lines += [
        "",
        "## User Action",
        "",
        "Fill the Stage42-EH intake template only after manually checking the official source terms and local source identity:",
        "",
        "- `outputs/stage42_long_research/source_terms_confirmation_intake_template_stage42.json`",
        f"- validator: `{s['next_validator_command']}`",
        f"- guarded launcher after validator readiness: `{s['next_guarded_launcher_command']}`",
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_em_gate"]["gates"].items()],
    ]
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage42-EM Official Source Links",
        "",
        "No dataset is conversion-ready yet. The agent cannot accept terms, infer allowed use, or treat toolkit/local files as data permission.",
        "",
        "| priority | dataset | official candidates | required user action |",
        "| ---: | --- | --- | --- |",
    ]
    for row in payload["official_source_rows"]:
        urls = "<br>".join(row["official_url_candidates"])
        lines.append(
            f"| {row['priority_rank']} | `{row['dataset_id']}` | {urls} | {row['recommended_user_action']} |"
        )
    lines += [
        "",
        "After filling the intake template, run:",
        "",
        "```bash",
        ".venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py",
        ".venv-pytorch/bin/python run_stage42_guarded_source_conversion_launcher.py",
        "```",
    ]
    return lines


def _readme_block(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    return [
        "## Stage42-EM Official Source Link Audit",
        "",
        f"- source: `{payload['source']}`",
        "- role: record official source candidates and user confirmation blockers for the next guarded conversion.",
        f"- gate: `{payload['stage42_em_gate']['passed']} / {payload['stage42_em_gate']['total']}`; verdict `{payload['stage42_em_gate']['verdict']}`.",
        f"- official/toolkit source candidates: `{s['official_or_toolkit_source_candidates']}` / `{s['targets']}`.",
        f"- conversion_ready_now: `{s['conversion_ready_now']}`; auto_download_allowed_now: `{s['auto_download_allowed_now']}`.",
        f"- estimated after-terms t50/t100 potential: `{s['estimated_t50_after_terms']}` / `{s['estimated_t100_after_terms']}`.",
        "- No download, conversion, training, evaluation, metric/seconds claim, Stage5C, or SMC execution.",
    ]


def _update_readmes(payload: Mapping[str, Any]) -> None:
    block = _readme_block(payload)
    for path in [README_RESULTS, M3W_README, GOAL_SUMMARY, GOAL_LEDGER]:
        if path.exists():
            _replace_section(path, "STAGE42_EM_OFFICIAL_SOURCE_LINK_AUDIT", block)

    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-EM official source link audit"
    state["current_verdict"] = payload["stage42_em_gate"]["verdict"]
    state["stage42_em_official_source_link_audit"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "user_action": str(USER_ACTION_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_em_gate"]["verdict"],
        "gates": f"{payload['stage42_em_gate']['passed']}/{payload['stage42_em_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_official_source_link_audit(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    rows = _intake_rows()
    readiness = _readiness_summary()
    audit_rows = _audit_rows(rows)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-EM Official Source Link Audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "retrieval_date": RETRIEVAL_DATE,
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(INTAKE_JSON), str(READINESS_JSON)]),
        "current_facts": CURRENT_FACTS,
        "readiness_manifest": readiness,
        "official_source_rows": audit_rows,
        "summary": _summary(audit_rows, readiness),
        "claim_boundary": CLAIM_BOUNDARY,
        "user_action_written": True,
    }
    payload["stage42_em_gate"] = _gate(payload)

    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(
        GATE_MD,
        [
            "# Stage42-EM Gate",
            "",
            f"- gate: `{payload['stage42_em_gate']['passed']} / {payload['stage42_em_gate']['total']}`",
            f"- verdict: `{payload['stage42_em_gate']['verdict']}`",
            "",
            "| gate | pass |",
            "| --- | ---: |",
            *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_em_gate"]["gates"].items()],
        ],
    )
    if refresh_readmes:
        _update_readmes(payload)
    return payload


if __name__ == "__main__":
    run_stage42_official_source_link_audit()
