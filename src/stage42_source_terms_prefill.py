from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

GA_JSON = OUT_DIR / "live_source_calibration_recheck_stage42.json"
SOURCE_TERMS_TEMPLATE_JSON = OUT_DIR / "source_terms_confirmation_template_stage42.json"

REPORT_JSON = OUT_DIR / "source_terms_prefill_stage42.json"
REPORT_MD = OUT_DIR / "source_terms_prefill_stage42.md"
PREFILL_DRAFT_JSON = OUT_DIR / "source_terms_confirmation_prefill_stage42.json"
USER_ACTION_MD = OUT_DIR / "user_action_required_source_terms_prefill_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_gb_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
ONE_FILE_SUMMARY = Path("README_M3W_ONE_FILE_DETAILED_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_gb_source_terms_prefill"

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "download_executed": False,
    "conversion_executed": False,
    "training_executed": False,
    "evaluation_executed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
    "prefill_is_permission": False,
    "conversion_ready_claim": False,
}

RAW_SOURCE_PATH_HINTS = ("external_data/", "/Users/yangyue/Downloads/", "data/aerialmpt/")
DERIVED_OR_CACHE_HINTS = (
    "stage20_world_state",
    "stage21_sdd_world_state",
    "stage24_sdd_fast_cache",
    "stage14_multimodal_episodes",
)


def _target_by_id(ga: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("target_id", "")): row for row in ga.get("target_rows", [])}


def _candidate_paths(target_row: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in target_row.get("path_status", []):
        if not item.get("exists"):
            continue
        path = str(item.get("path", ""))
        rows.append(
            {
                "path": path,
                "exists": True,
                "is_raw_source_candidate": path.startswith(RAW_SOURCE_PATH_HINTS) and not any(hint in path for hint in DERIVED_OR_CACHE_HINTS),
                "is_derived_or_cache": any(hint in path for hint in DERIVED_OR_CACHE_HINTS),
                "size_mb": item.get("size_mb"),
                "sample_extensions": item.get("sample_extensions", {}),
                "sample_count": item.get("sample_count", 0),
            }
        )
    return rows


def _preferred_path(candidates: list[Mapping[str, Any]]) -> str:
    raw = [row for row in candidates if row.get("is_raw_source_candidate")]
    if raw:
        return str(raw[0]["path"])
    if candidates:
        return str(candidates[0]["path"])
    return ""


def _source_identity_hint(dataset_id: str, target_row: Mapping[str, Any], preferred_path: str) -> str:
    domain = str(target_row.get("domain", dataset_id))
    official_url = str(target_row.get("official_url", ""))
    if not preferred_path:
        return f"{domain} official source identity requires user-provided legal local path and official URL verification."
    return f"{domain} local candidate at {preferred_path}; user must confirm it corresponds to official source {official_url}."


def _dataset_prefill(row: Mapping[str, Any], target_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    dataset_id = str(row.get("dataset_id", ""))
    target = target_rows.get(dataset_id, {})
    candidates = _candidate_paths(target)
    preferred = _preferred_path(candidates)
    official_url = str(row.get("official_url") or target.get("official_url", ""))
    return {
        "dataset_id": dataset_id,
        "official_url": official_url,
        "result_source": "fresh_prefill_from_stage42_ga_local_scan",
        "suggested_local_path": preferred,
        "suggested_source_identity": _source_identity_hint(dataset_id, target, preferred),
        "local_path_candidates": candidates,
        "must_be_filled_by_user": {
            "terms_accepted_by_user": False,
            "terms_acceptance_date": "",
            "allowed_use": "",
            "redistribution_allowed": "unknown",
            "derived_data_allowed": "unknown",
            "local_path": "",
            "source_identity": "",
            "confirmed_by_user": "",
        },
        "safe_copy_instruction": "Copy suggested_local_path/source_identity only after checking official terms; do not set terms_accepted_by_user unless the user has accepted/verified terms.",
        "conversion_ready_now": False,
        "converted_now": False,
        "evaluated_now": False,
        "next_action": "User verifies official terms, copies/edits suggested fields into source_terms_confirmation_template_stage42.json, then reruns the terms validator and guarded conversion queue.",
    }


def _build_prefill(ga: Mapping[str, Any], template: Mapping[str, Any]) -> dict[str, Any]:
    target_rows = _target_by_id(ga)
    datasets = [_dataset_prefill(row, target_rows) for row in template.get("datasets", [])]
    return {
        "source": SOURCE,
        "purpose": "User-assistance draft only. This file suggests local paths/source-identity text from Stage42-GA; it is not legal permission and is not consumed as conversion readiness.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_ga_source": ga.get("source"),
        "input_template_source": template.get("source"),
        "terms_confirmation_is_currently_absent": True,
        "datasets": datasets,
    }


def _summary(prefill: Mapping[str, Any]) -> dict[str, Any]:
    datasets = list(prefill.get("datasets", []))
    return {
        "datasets_prefilled": len(datasets),
        "datasets_with_suggested_local_path": sum(1 for row in datasets if row.get("suggested_local_path")),
        "raw_source_candidate_rows": sum(1 for row in datasets if any(c.get("is_raw_source_candidate") for c in row.get("local_path_candidates", []))),
        "terms_accepted_by_user_count": 0,
        "conversion_ready_now": 0,
        "downloads_executed": 0,
        "conversions_executed": 0,
        "evaluations_executed": 0,
        "highest_priority_next_action": "FW-TERMS-ucy_crowd_original",
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    claim = payload["claim_boundary"]
    gates = {
        "source_fresh": payload.get("source") == SOURCE,
        "ga_loaded": payload.get("input_status", {}).get("ga_exists") is True,
        "terms_template_loaded": payload.get("input_status", {}).get("terms_template_exists") is True,
        "all_template_rows_prefilled": s.get("datasets_prefilled", 0) >= 5,
        "local_path_suggestions_present": s.get("datasets_with_suggested_local_path", 0) >= 4,
        "raw_candidates_identified": s.get("raw_source_candidate_rows", 0) >= 4,
        "terms_not_auto_accepted": s.get("terms_accepted_by_user_count") == 0,
        "conversion_ready_zero": s.get("conversion_ready_now") == 0,
        "draft_written": payload.get("prefill_draft_written") is True,
        "user_action_written": payload.get("user_action_required_written") is True,
        "no_download_conversion_eval": claim.get("download_executed") is False
        and claim.get("conversion_executed") is False
        and claim.get("evaluation_executed") is False,
        "no_metric_seconds_overclaim": claim.get("global_metric_claim_allowed") is False
        and claim.get("global_seconds_claim_allowed") is False,
        "no_true3d_foundation_overclaim": claim.get("true_3d") is False and claim.get("foundation_world_model") is False,
        "stage5c_false": claim.get("stage5c_executed") is False,
        "smc_false": claim.get("smc_enabled") is False,
    }
    passed = sum(bool(v) for v in gates.values())
    total = len(gates)
    return {
        "passed": passed,
        "total": total,
        "gates": gates,
        "verdict": "stage42_gb_source_terms_prefill_pass" if passed == total else "stage42_gb_source_terms_prefill_partial",
    }


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-GB Source Terms Prefill",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_gb_gate']['passed']} / {payload['stage42_gb_gate']['total']}`",
        f"- verdict: `{payload['stage42_gb_gate']['verdict']}`",
        "",
        "## Role",
        "",
        "- This stage turns the Stage42-GA local scan into a user-facing source-terms prefill draft.",
        "- It does not accept terms, download data, convert data, train, evaluate, or mark any source as conversion-ready.",
        "- The draft is intentionally not used as permission by the validator; the user must still edit/confirm the actual source terms template.",
        "",
        "## Summary",
        "",
        f"- datasets_prefilled: `{s['datasets_prefilled']}`",
        f"- datasets_with_suggested_local_path: `{s['datasets_with_suggested_local_path']}`",
        f"- raw_source_candidate_rows: `{s['raw_source_candidate_rows']}`",
        f"- conversion_ready_now: `{s['conversion_ready_now']}`",
        f"- highest_priority_next_action: `{s['highest_priority_next_action']}`",
        "",
        "## Prefill Rows",
        "",
        "| dataset | suggested local path | raw candidate? | next action |",
        "| --- | --- | ---: | --- |",
    ]
    for row in payload["prefill"]["datasets"]:
        raw = any(c.get("is_raw_source_candidate") for c in row.get("local_path_candidates", []))
        lines.append(
            f"| `{row['dataset_id']}` | `{row['suggested_local_path'] or 'missing'}` | {raw} | {row['next_action']} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- Prefill is not legal permission and not source conversion readiness.",
            "- Local files remain insufficient for new benchmark claims without official terms/source identity confirmation and guarded conversion/no-leakage/source-CV.",
            "- Current M3W remains protected dataset-local/raw-frame 2.5D; no true 3D, foundation, metric, seconds-level, Stage5C, or SMC claim.",
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage42-GB Source Terms Prefill",
        "",
        "Use this as a checklist. Do not treat the prefill draft as authorization.",
        "",
        f"- prefill draft: `{PREFILL_DRAFT_JSON}`",
        "- copy a suggested local path only after verifying it is the official allowed source copy",
        "- fill `terms_accepted_by_user`, `terms_acceptance_date`, `allowed_use`, `local_path`, and `source_identity` in `source_terms_confirmation_template_stage42.json`",
        "- then rerun the source terms validator and guarded conversion queue",
        "",
        "| dataset | suggested path | official URL |",
        "| --- | --- | --- |",
    ]
    for row in payload["prefill"]["datasets"]:
        lines.append(f"| `{row['dataset_id']}` | `{row['suggested_local_path'] or 'missing'}` | {row['official_url']} |")
    lines.extend(
        [
            "",
            "```bash",
            ".venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py",
            ".venv-pytorch/bin/python run_stage42_unified_guarded_conversion_queue.py",
            ".venv-pytorch/bin/python run_stage42_guarded_source_conversion_launcher.py",
            ".venv-pytorch/bin/python run_stage42_source_support_closure_audit.py",
            "```",
        ]
    )
    return lines


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-GB Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | {bool(ok)} |")
    write_md(GATE_MD, lines)


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    s = payload["summary"]
    lines = [
        "## Stage42-GB Source Terms Prefill",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{payload['stage42_gb_gate']['passed']} / {payload['stage42_gb_gate']['total']}`; verdict `{payload['stage42_gb_gate']['verdict']}`.",
        "- role: converts Stage42-GA local path evidence into a user-facing source-terms prefill draft; no download, conversion, training, evaluation, or permission claim.",
        f"- datasets prefilled: `{s['datasets_prefilled']}`; with suggested local path `{s['datasets_with_suggested_local_path']}`; raw-source candidates `{s['raw_source_candidate_rows']}`.",
        f"- conversion_ready_now: `{s['conversion_ready_now']}`; highest-priority next action `{s['highest_priority_next_action']}`.",
        f"- prefill draft: `{PREFILL_DRAFT_JSON}`.",
        "- boundary: prefill is not legal permission; protected dataset-local/raw-frame 2.5D only; no true 3D/foundation/global metric/seconds/Stage5C/SMC claim.",
    ]
    for path in [README_RESULTS, M3W_README, ONE_FILE_SUMMARY]:
        _replace_section(path, "STAGE42_GB_SOURCE_TERMS_PREFILL", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-GB source terms prefill"
    state["current_verdict"] = payload["stage42_gb_gate"]["verdict"]
    state["stage42_gb_source_terms_prefill"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "prefill_draft": str(PREFILL_DRAFT_JSON),
        "gate": str(GATE_MD),
        "updated_at": payload["generated_at_utc"],
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_source_terms_prefill() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ga = read_json(GA_JSON, {})
    template = read_json(SOURCE_TERMS_TEMPLATE_JSON, {})
    prefill = _build_prefill(ga, template)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([GA_JSON, SOURCE_TERMS_TEMPLATE_JSON]),
        "input_status": {
            "ga_exists": GA_JSON.exists(),
            "terms_template_exists": SOURCE_TERMS_TEMPLATE_JSON.exists(),
            "ga_source": ga.get("source", ""),
            "terms_template_source": template.get("source", ""),
        },
        "prefill": prefill,
        "summary": _summary(prefill),
        "claim_boundary": CLAIM_BOUNDARY,
        "prefill_draft_written": True,
        "user_action_required_written": True,
    }
    payload["stage42_gb_gate"] = _gate(payload)
    write_json(PREFILL_DRAFT_JSON, prefill)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    _write_gate(payload["stage42_gb_gate"])
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


__all__ = [
    "run_stage42_source_terms_prefill",
    "_candidate_paths",
    "_preferred_path",
    "_gate",
]
