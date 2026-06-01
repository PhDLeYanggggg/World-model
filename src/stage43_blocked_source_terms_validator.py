from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m


OUT_DIR = m.OUT_DIR
BF_TEMPLATE_JSON = OUT_DIR / "stage43_blocked_source_terms_identity_template.json"
BF_PACKET_JSON = OUT_DIR / "stage43_blocked_source_terms_identity_packet.json"

REPORT_JSON = OUT_DIR / "stage43_blocked_source_terms_validation.json"
REPORT_MD = OUT_DIR / "stage43_blocked_source_terms_validation.md"
MANIFEST_JSON = OUT_DIR / "stage43_blocked_source_guarded_conversion_manifest.json"
MANIFEST_MD = OUT_DIR / "stage43_blocked_source_guarded_conversion_manifest.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_stage43_blocked_source_terms_validation.md"
GATE_MD = OUT_DIR / "stage43_stage_bg_blocked_source_terms_validation_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SOURCE = "fresh_stage43_bg_blocked_source_terms_validator"
SECTION = "STAGE43_BG_BLOCKED_SOURCE_TERMS_VALIDATOR"

SAFE_ALLOWED_USE_VALUES = {
    "research_only",
    "academic_research",
    "academic_noncommercial",
    "noncommercial_research",
    "research_and_education",
    "commercial_allowed",
}


def _manual(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("manual_fields_required", {})
    return value if isinstance(value, Mapping) else {}


def _present(value: Any) -> bool:
    return bool(str(value or "").strip())


def _http_url(value: Any) -> bool:
    return str(value or "").strip().startswith(("http://", "https://"))


def _iso8601_like(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def _validate_dataset_row(row: Mapping[str, Any]) -> dict[str, Any]:
    manual = _manual(row)
    blockers: list[str] = []
    warnings: list[str] = []

    dataset_name = str(row.get("dataset_name", "")).strip()
    if not dataset_name:
        blockers.append("dataset_name_missing")
    if row.get("technical_support_candidate") is not True:
        blockers.append("technical_support_candidate_false")

    local_path = Path(str(row.get("local_path", "")).strip())
    if not str(row.get("local_path", "")).strip():
        blockers.append("local_path_missing")
    elif not local_path.exists():
        blockers.append("local_path_not_found")

    preferred_url = str(row.get("preferred_official_url", "")).strip()
    official_candidates = [str(url).strip() for url in row.get("official_url_candidates", []) if str(url).strip()]
    if manual.get("official_url_confirmed") is not True:
        blockers.append("official_url_not_confirmed_by_user")
    if preferred_url and preferred_url not in official_candidates:
        warnings.append("preferred_official_url_not_in_candidate_list")
    if preferred_url and not _http_url(preferred_url):
        blockers.append("preferred_official_url_not_http")

    if not _present(manual.get("official_terms_url")):
        blockers.append("official_terms_url_missing")
    elif not _http_url(manual.get("official_terms_url")):
        blockers.append("official_terms_url_not_http")

    if not _present(manual.get("license_name")):
        blockers.append("license_name_missing")
    if manual.get("terms_accepted_by_user") is not True:
        blockers.append("terms_not_accepted_by_user")
    if not _present(manual.get("accepted_by_user")):
        blockers.append("accepted_by_user_missing")
    if not _present(manual.get("accepted_at_utc")):
        blockers.append("accepted_at_utc_missing")
    elif not _iso8601_like(manual.get("accepted_at_utc")):
        blockers.append("accepted_at_utc_invalid_iso8601")

    allowed_use = str(manual.get("allowed_use", "")).strip().lower()
    if not allowed_use:
        blockers.append("allowed_use_missing")
    elif allowed_use in {"unknown", "unspecified", "not_sure"}:
        blockers.append("allowed_use_unknown")
    elif allowed_use not in SAFE_ALLOWED_USE_VALUES:
        warnings.append("allowed_use_not_in_known_safe_set_manual_review_required")

    if manual.get("source_identity_confirmed") is not True:
        blockers.append("source_identity_not_confirmed_by_user")
    if manual.get("calibration_projection_scope_confirmed") is not True:
        blockers.append("calibration_projection_scope_not_confirmed_by_user")
    if manual.get("conversion_scope_confirmed") is not True:
        blockers.append("conversion_scope_not_confirmed_by_user")
    if manual.get("can_use_for_stage43_support") is not True:
        blockers.append("can_use_for_stage43_support_false")

    if str(row.get("source_confidence", "")).lower() in {"low", "unknown"}:
        warnings.append("source_confidence_requires_extra_review")
    metric_status = str(row.get("metric_status", "")).lower()
    if "unverified" in metric_status or "not_integrated" in metric_status:
        warnings.append("metric_or_projection_not_verified_for_claims")

    ready = not blockers
    return {
        "dataset_name": dataset_name,
        "source": "fresh_validation_from_stage43_bf_template",
        "local_path": str(row.get("local_path", "")).strip(),
        "preferred_official_url": preferred_url,
        "source_confidence": str(row.get("source_confidence", "unknown")),
        "support_family": str(row.get("support_family", "unknown")),
        "technical_support_candidate": bool(row.get("technical_support_candidate", False)),
        "point_rows": int(row.get("point_rows", 0)),
        "agent_tracks": int(row.get("agent_tracks", 0)),
        "t50_candidate_rows": int(row.get("t50_candidate_rows", 0)),
        "t100_candidate_rows": int(row.get("t100_candidate_rows", 0)),
        "coordinate_unit": str(row.get("coordinate_unit", "unknown")),
        "metric_status": str(row.get("metric_status", "unverified")),
        "manual_terms_accepted": manual.get("terms_accepted_by_user") is True,
        "manual_source_identity_confirmed": manual.get("source_identity_confirmed") is True,
        "manual_conversion_scope_confirmed": manual.get("conversion_scope_confirmed") is True,
        "manual_can_use_for_stage43_support": manual.get("can_use_for_stage43_support") is True,
        "blockers": blockers,
        "warnings": warnings,
        "ready_for_guarded_conversion_preflight": ready,
        "conversion_executed_now": False,
        "training_allowed_now": False,
        "evaluated_now": False,
        "next_action": "eligible_for_guarded_conversion_preflight" if ready else "fill_source_terms_identity_template_and_revalidate",
    }


def _validate_biwi(template: Mapping[str, Any]) -> dict[str, Any]:
    row = template.get("biwi_independent_source", {})
    manual = row.get("manual_fields_required", {}) if isinstance(row, Mapping) else {}
    blockers = list(row.get("blockers", [])) if isinstance(row, Mapping) else ["biwi_packet_missing"]
    if not _present(manual.get("new_independent_source_path")):
        blockers.append("new_independent_source_path_missing")
    if manual.get("official_url_confirmed") is not True:
        blockers.append("official_url_not_confirmed")
    if manual.get("terms_accepted_by_user") is not True:
        blockers.append("terms_not_accepted")
    if manual.get("source_identity_confirmed") is not True:
        blockers.append("source_identity_not_confirmed")
    if manual.get("heldout_source_disjoint_from_train_val") is not True:
        blockers.append("heldout_source_disjoint_from_train_val_not_confirmed")
    return {
        "family": "TrajNet_biwi",
        "status": row.get("status", "missing") if isinstance(row, Mapping) else "missing",
        "repair_training_allowed_now": False,
        "blockers": sorted(set(str(item) for item in blockers if str(item))),
        "ready_for_repair_training_preflight": False,
        "next_action": "locate_independent_source_and_confirm_terms_before_biwi_repair",
    }


def _build_manifest(validations: list[Mapping[str, Any]], biwi: Mapping[str, Any]) -> dict[str, Any]:
    ready = [dict(row) for row in validations if row["ready_for_guarded_conversion_preflight"]]
    blocked = [dict(row) for row in validations if not row["ready_for_guarded_conversion_preflight"]]
    return {
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Future guarded conversion manifest for Stage43 blocked source support. This manifest is not permission, conversion, training, or evaluation.",
        "ready_for_guarded_conversion_preflight": ready,
        "blocked_datasets": blocked,
        "biwi_independent_source": dict(biwi),
        "conversion_executed": False,
        "training_executed": False,
        "evaluation_executed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def build_blocked_source_terms_validation() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    bf_packet = read_json(BF_PACKET_JSON, {})
    template = read_json(BF_TEMPLATE_JSON, {})
    validations = [_validate_dataset_row(row) for row in template.get("datasets", [])]
    biwi = _validate_biwi(template)
    manifest = _build_manifest(validations, biwi)
    summary = {
        "datasets_validated": len(validations),
        "manual_terms_accepted_rows": sum(1 for row in validations if row["manual_terms_accepted"]),
        "ready_for_guarded_conversion_preflight_rows": sum(
            1 for row in validations if row["ready_for_guarded_conversion_preflight"]
        ),
        "blocked_rows": [row["dataset_name"] for row in validations if not row["ready_for_guarded_conversion_preflight"]],
        "conversion_executed_now": 0,
        "training_allowed_now": 0,
        "evaluated_now": 0,
        "biwi_ready_for_repair_training_preflight": False,
        "decision": "blocked_until_source_terms_identity_template_confirmed",
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_validation_of_stage43_bf_terms_identity_template",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "input_artifacts": {
            "stage43_bf_packet": str(BF_PACKET_JSON),
            "stage43_bf_template": str(BF_TEMPLATE_JSON),
        },
        "input_verdicts": {
            "stage43_bf": bf_packet.get("stage43_bf_gate", {}).get("verdict"),
            "template_source": template.get("source"),
        },
        "input_hash": _combined_hash([BF_PACKET_JSON, BF_TEMPLATE_JSON]),
        "summary": summary,
        "validations": validations,
        "biwi_validation": biwi,
        "manifest": manifest,
        "no_leakage_and_execution": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_eval_or_loss_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
            "download_executed": False,
            "conversion_executed": False,
            "training_executed": False,
            "evaluation_executed": False,
        },
        "claim_boundary": {
            "validator_is_permission": False,
            "manifest_is_conversion": False,
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "converted_external_support_source": False,
            "blocked_source_repair_success_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
    }
    payload["stage43_bg_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    no_leak = payload["no_leakage_and_execution"]
    claim = payload["claim_boundary"]
    gates = {
        "stage43_bf_precondition_passed": payload["input_verdicts"]["stage43_bf"]
        == "stage43_bf_blocked_source_terms_identity_packet_pass",
        "template_loaded": payload["input_verdicts"]["template_source"]
        == "fresh_stage43_bf_blocked_source_terms_identity_packet",
        "datasets_validated": summary["datasets_validated"] >= 3,
        "all_rows_have_blocker_status": all(isinstance(row.get("blockers"), list) for row in payload["validations"]),
        "blank_template_blocks_conversion": summary["manual_terms_accepted_rows"] == 0
        and summary["ready_for_guarded_conversion_preflight_rows"] == 0,
        "manifest_written": "ready_for_guarded_conversion_preflight" in payload["manifest"],
        "conversion_training_eval_zero": summary["conversion_executed_now"] == 0
        and summary["training_allowed_now"] == 0
        and summary["evaluated_now"] == 0,
        "biwi_repair_still_blocked": payload["biwi_validation"]["repair_training_allowed_now"] is False
        and summary["biwi_ready_for_repair_training_preflight"] is False,
        "no_future_or_test_leakage": no_leak["future_endpoint_input"] is False
        and no_leak["future_waypoint_input"] is False
        and no_leak["future_labels_eval_or_loss_only"] is True
        and no_leak["central_velocity_input"] is False
        and no_leak["test_endpoint_goal_construction"] is False
        and no_leak["test_statistics_normalization"] is False
        and no_leak["test_threshold_tuning"] is False,
        "no_execution": no_leak["download_executed"] is False
        and no_leak["conversion_executed"] is False
        and no_leak["training_executed"] is False
        and no_leak["evaluation_executed"] is False,
        "claim_boundary_not_overstated": claim["validator_is_permission"] is False
        and claim["manifest_is_conversion"] is False
        and claim["true_3d_world_model"] is False
        and claim["foundation_world_model"] is False
        and claim["metric_or_seconds_claim"] is False
        and claim["converted_external_support_source"] is False
        and claim["blocked_source_repair_success_claim"] is False,
        "stage5c_and_smc_false": claim["stage5c_executed"] is False and claim["smc_enabled"] is False,
        "long_objective_kept_active": claim["long_objective_complete"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_bg_blocked_source_terms_validation_pass"
        if passed == total
        else "stage43_bg_blocked_source_terms_validation_incomplete",
        "stage5c_executed": False,
        "smc_enabled": False,
        "goal_complete": False,
    }


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_bg_gate"]
    summary = payload["summary"]
    lines = [
        "# Stage43-BG Blocked Source Terms Validator",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- datasets_validated: `{summary['datasets_validated']}`",
        f"- ready_for_guarded_conversion_preflight_rows: `{summary['ready_for_guarded_conversion_preflight_rows']}`",
        f"- training_allowed_now: `{summary['training_allowed_now']}`",
        "",
        "## Validation Table",
        "",
        "| dataset | accepted | ready | t50 | t100 | blockers | warnings |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload["validations"]:
        lines.append(
            f"| `{row['dataset_name']}` | `{row['manual_terms_accepted']}` | `{row['ready_for_guarded_conversion_preflight']}` | "
            f"{row['t50_candidate_rows']} | {row['t100_candidate_rows']} | `{row['blockers']}` | `{row['warnings']}` |"
        )
    lines.extend(
        [
            "",
            "## Biwi Independent Source",
            "",
            f"- ready_for_repair_training_preflight: `{payload['biwi_validation']['ready_for_repair_training_preflight']}`",
            f"- repair_training_allowed_now: `{payload['biwi_validation']['repair_training_allowed_now']}`",
            f"- blockers: `{payload['biwi_validation']['blockers']}`",
            "",
            "## Interpretation",
            "",
            "The validator is doing the right thing: the Stage43-BF template is still blank, so every local candidate remains blocked. This keeps PETS, Town-Center, Wild-Track, and the biwi family out of conversion/training until source identity, terms, scope, and independent-source checks are actually closed.",
            "",
            "## Claim Boundary",
            "",
            "- Validator output is not permission.",
            "- Manifest output is not conversion.",
            "- No download, conversion, training, threshold tuning, or evaluation is executed.",
            "- Dataset-local/raw-frame 2.5D only; no metric or seconds-level claim.",
            "- Stage5C remains false and SMC remains false.",
        ]
    )
    return lines


def _render_manifest(manifest: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage43-BG Blocked Source Guarded Conversion Manifest",
        "",
        f"- source: `{manifest['source']}`",
        f"- generated_at_utc: `{manifest['generated_at_utc']}`",
        f"- ready_count: `{len(manifest['ready_for_guarded_conversion_preflight'])}`",
        f"- blocked_count: `{len(manifest['blocked_datasets'])}`",
        f"- conversion_executed: `{manifest['conversion_executed']}`",
        f"- training_executed: `{manifest['training_executed']}`",
        f"- evaluation_executed: `{manifest['evaluation_executed']}`",
        "",
        "This manifest is a future input checklist only. It is not permission and not a data conversion.",
        "",
        "## Ready Rows",
        "",
    ]
    if manifest["ready_for_guarded_conversion_preflight"]:
        for row in manifest["ready_for_guarded_conversion_preflight"]:
            lines.append(f"- `{row['dataset_name']}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Blocked Rows", ""])
    for row in manifest["blocked_datasets"]:
        lines.append(f"- `{row['dataset_name']}`: `{row['blockers']}`")
    lines.extend(["", "## Biwi", "", f"- blockers: `{manifest['biwi_independent_source']['blockers']}`"])
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage43 Blocked Source Terms Validation",
        "",
        f"- template: `{BF_TEMPLATE_JSON}`",
        f"- validation_report: `{REPORT_MD}`",
        f"- manifest: `{MANIFEST_MD}`",
        "",
        "Fill the template manually only after checking official source identity and terms. The agent must not fill acceptance fields for the user.",
        "",
    ]
    for row in payload["validations"]:
        if not row["ready_for_guarded_conversion_preflight"]:
            lines.extend(
                [
                    f"## {row['dataset_name']}",
                    "",
                    f"- preferred_official_url_hint: `{row['preferred_official_url']}`",
                    f"- blockers: `{row['blockers']}`",
                    f"- warnings: `{row['warnings']}`",
                    "- required: official URL confirmation, terms URL, license/use scope, accepted-by user/date, source identity, calibration projection scope, conversion scope, and Stage43 support permission.",
                    "",
                ]
            )
    lines.extend(
        [
            "## TrajNet_biwi",
            "",
            f"- blockers: `{payload['biwi_validation']['blockers']}`",
            "- required: an independent biwi-like source disjoint from held-out test source before any repair training.",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_bg_gate"]
    lines = [
        "# Stage43-BG Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | passed |",
        "| --- | ---: |",
    ]
    lines.extend([f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()])
    return lines


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bg_gate"]
    summary = payload["summary"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"datasets_validated = `{summary['datasets_validated']}`",
        f"ready_for_guarded_conversion_preflight_rows = `{summary['ready_for_guarded_conversion_preflight_rows']}`",
        f"training_allowed_now = `{summary['training_allowed_now']}`",
        "",
        "I validated the Stage43-BF terms/source template as-is. The result is intentionally blocked: no candidate source has user-confirmed terms, source identity, calibration scope, conversion scope, or Stage43 support permission yet. The manifest is useful for the next guarded conversion step, but it is not permission and it does not convert or train anything.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_bg_blocked_source_terms_validator"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "summary": summary,
        "report": str(REPORT_MD),
        "manifest": str(MANIFEST_MD),
        "user_action": str(USER_ACTION_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_bg_blocked_source_terms_validator"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-BG",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "summary": summary,
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    write_json(MANIFEST_JSON, m._jsonable(payload["manifest"]))
    write_json(WORLD_GATE_JSON, m._jsonable(payload["stage43_bg_gate"]))
    write_md(REPORT_MD, _render_report(payload))
    write_md(MANIFEST_MD, _render_manifest(payload["manifest"]))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    gate = payload["stage43_bg_gate"]
    write_md(
        WORLD_GATE_MD,
        [
            "# Stage43 Current World-Model Gate",
            "",
            f"- source: `{payload['source']}`",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- long objective complete: `{gate['goal_complete']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "## Current Boundary",
            "",
            "- Stage43-P / AZ remains the performance leader and exact replay artifact.",
            "- Stage43-BG validates the blocked-source terms template; it does not accept terms, convert data, or train.",
            "- PETS, Town-Center, Wild-Track, and biwi stay floor-only until source/terms/split/conversion gates clear.",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    _update_ledgers(payload)


def run_blocked_source_terms_validator() -> dict[str, Any]:
    payload = build_blocked_source_terms_validation()
    _write_outputs(payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description="Validate Stage43 blocked source terms/source-identity template without conversion or training."
    )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    build_arg_parser().parse_args(argv)
    payload = run_blocked_source_terms_validator()
    gate = payload["stage43_bg_gate"]
    summary = payload["summary"]
    print(f"Stage43-BG: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"datasets_validated={summary['datasets_validated']}")
    print(f"ready_for_guarded_conversion_preflight_rows={summary['ready_for_guarded_conversion_preflight_rows']}")
    print(f"training_allowed_now={summary['training_allowed_now']}")
    return payload


if __name__ == "__main__":
    main()
