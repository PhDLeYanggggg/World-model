from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")
CF_JSON = OUT_DIR / "source_conversion_legal_gate_stage42.json"
CONFIRMATION_JSON = OUT_DIR / "source_terms_confirmation_template_stage42.json"

REPORT_JSON = OUT_DIR / "source_terms_validation_stage42.json"
REPORT_MD = OUT_DIR / "source_terms_validation_stage42.md"
MANIFEST_JSON = OUT_DIR / "source_conversion_readiness_manifest_stage42.json"
MANIFEST_MD = OUT_DIR / "source_conversion_readiness_manifest_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_cg_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_source_terms_validation_stage42.md"

SOURCE = "fresh_stage42_cg_source_terms_confirmation_validator"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-CG 只验证 source terms confirmation，不下载、不转换、不训练、不评估。",
    "空白模板、local path、parseability 都不等于 legal permission。",
    "conversion_ready 需要 terms accepted、allowed_use、local_path、source_identity 和 CF source-CV blockers 同时通过。",
    "future endpoints / waypoints 只允许作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "Stage5C 未执行，SMC 未启用。",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _confirmation_by_id(template: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("dataset_id", "")): row for row in template.get("datasets", [])}


def _validate_confirmation(decision: Mapping[str, Any], confirmation: Mapping[str, Any] | None) -> dict[str, Any]:
    blockers = list(decision["blockers"])
    confirmation_blockers: list[str] = []
    if not confirmation:
        confirmation_blockers.append("confirmation_entry_missing")
    else:
        if confirmation.get("official_url") != decision["official_url"]:
            confirmation_blockers.append("official_url_mismatch")
        if confirmation.get("terms_accepted_by_user") is not True:
            confirmation_blockers.append("terms_not_accepted")
        if not str(confirmation.get("terms_acceptance_date", "")).strip():
            confirmation_blockers.append("terms_acceptance_date_missing")
        if not str(confirmation.get("allowed_use", "")).strip():
            confirmation_blockers.append("allowed_use_missing")
        local_path = Path(str(confirmation.get("local_path", "")))
        if not str(confirmation.get("local_path", "")).strip():
            confirmation_blockers.append("local_path_confirmation_missing")
        elif not local_path.exists():
            confirmation_blockers.append("confirmed_local_path_missing")
        if not str(confirmation.get("source_identity", "")).strip():
            confirmation_blockers.append("source_identity_missing")

    conversion_ready = not blockers and not confirmation_blockers
    return {
        "dataset_id": decision["id"],
        "name": decision["name"],
        "official_url": decision["official_url"],
        "cf_blockers": blockers,
        "confirmation_blockers": confirmation_blockers,
        "terms_accepted_by_user": bool(confirmation and confirmation.get("terms_accepted_by_user") is True),
        "confirmed_local_path": str(confirmation.get("local_path", "")) if confirmation else "",
        "source_identity": str(confirmation.get("source_identity", "")) if confirmation else "",
        "conversion_ready": conversion_ready,
        "conversion_allowed_now": False,
        "converted_now": False,
        "evaluated_now": False,
        "next_action": _next_action(blockers, confirmation_blockers),
    }


def _next_action(cf_blockers: list[str], confirmation_blockers: list[str]) -> str:
    if confirmation_blockers:
        return "fill explicit official terms/path/source-identity confirmation before conversion"
    if cf_blockers:
        return "resolve source-CV blockers before conversion"
    return "ready for a future guarded conversion stage; Stage42-CG still does not convert"


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    gates = {
        "cf_input_verified": payload["input_reports"]["stage42_cf_verdict"] == "stage42_cf_source_conversion_legal_gate_pass",
        "confirmation_template_loaded": payload["input_reports"]["confirmation_template_source"]
        == "fresh_stage42_cf_source_conversion_legal_gate",
        "all_targets_validated": s["targets_validated"] >= 5,
        "readiness_manifest_written": "manifest" in payload,
        "empty_template_blocks_conversion": s["conversion_ready_targets"] == 0,
        "no_conversion_claim": s["converted_datasets_now"] == 0,
        "no_evaluation_claim": s["evaluated_datasets_now"] == 0,
        "user_action_written": bool(payload["user_action_required"]),
        "no_metric_seconds_overclaim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_cg_source_terms_confirmation_validator_pass" if passed == total else "stage42_cg_source_terms_confirmation_validator_partial"
    return {"source": payload["source"], "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    cf = _load_json(CF_JSON)
    template = _load_json(CONFIRMATION_JSON)
    confirmations = _confirmation_by_id(template)
    validations = [_validate_confirmation(row, confirmations.get(row["id"])) for row in cf["target_decisions"]]
    summary = {
        "source": SOURCE,
        "targets_validated": len(validations),
        "terms_accepted_targets": sum(1 for row in validations if row["terms_accepted_by_user"]),
        "conversion_ready_targets": sum(1 for row in validations if row["conversion_ready"]),
        "conversion_allowed_now_count": 0,
        "converted_datasets_now": 0,
        "evaluated_datasets_now": 0,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    manifest = {
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "conversion_ready_targets": [row for row in validations if row["conversion_ready"]],
        "blocked_targets": [row for row in validations if not row["conversion_ready"]],
        "conversion_executed": False,
        "evaluation_executed": False,
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-CG Source Terms Confirmation Validator",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(CF_JSON), str(CONFIRMATION_JSON)]),
        "current_facts": CURRENT_FACTS,
        "input_reports": {
            "stage42_cf_verdict": cf["stage42_cf_gate"]["verdict"],
            "confirmation_template_source": template.get("source", ""),
            "terms_confirmation_is_currently_absent": template.get("terms_confirmation_is_currently_absent", None),
        },
        "summary": summary,
        "validations": validations,
        "manifest": manifest,
        "user_action_required": [
            {
                "dataset_id": row["dataset_id"],
                "official_url": row["official_url"],
                "cf_blockers": row["cf_blockers"],
                "confirmation_blockers": row["confirmation_blockers"],
                "action": row["next_action"],
            }
            for row in validations
            if not row["conversion_ready"]
        ],
        "claim_boundary": {
            "validator_counted_as_conversion": False,
            "converted_dataset_claim": False,
            "evaluated_dataset_claim": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_cg_gate"] = _gate(payload)
    return payload


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-CG Source Terms Confirmation Validator",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_cg_gate']['passed']} / {payload['stage42_cg_gate']['total']}`",
        f"- verdict: `{payload['stage42_cg_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        f"- targets_validated: `{s['targets_validated']}`",
        f"- terms_accepted_targets: `{s['terms_accepted_targets']}`",
        f"- conversion_ready_targets: `{s['conversion_ready_targets']}`",
        f"- conversion_allowed_now_count: `{s['conversion_allowed_now_count']}`",
        f"- converted_datasets_now: `{s['converted_datasets_now']}`",
        f"- evaluated_datasets_now: `{s['evaluated_datasets_now']}`",
        "",
        "## Validation Table",
        "",
        "| dataset | terms accepted | conversion ready | CF blockers | confirmation blockers |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in payload["validations"]:
        lines.append(
            f"| `{row['dataset_id']}` | {row['terms_accepted_by_user']} | {row['conversion_ready']} | "
            f"{', '.join(row['cf_blockers']) or 'none'} | {', '.join(row['confirmation_blockers']) or 'none'} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Stage42-CG validates terms-confirmation readiness; it still performs no conversion.",
        "- The current CF-generated template is blank, so every source remains blocked.",
        "- Future conversion must use a filled confirmation file plus a separate no-leakage/source-CV conversion stage.",
    ]
    return lines


def _render_manifest(payload: Mapping[str, Any]) -> list[str]:
    manifest = payload["manifest"]
    return [
        "# Stage42-CG Source Conversion Readiness Manifest",
        "",
        f"- source: `{manifest['source']}`",
        f"- conversion_ready_targets: `{len(manifest['conversion_ready_targets'])}`",
        f"- blocked_targets: `{len(manifest['blocked_targets'])}`",
        f"- conversion_executed: `{manifest['conversion_executed']}`",
        f"- evaluation_executed: `{manifest['evaluation_executed']}`",
        "",
        "Current manifest blocks all conversion. It is a readiness artifact only.",
    ]


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_cg_gate"]
    lines = [
        "# Stage42-CG Gate",
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
        "# User Action Required: Stage42-CG Terms Validation",
        "",
        "No target is conversion-ready. Required actions:",
        "",
    ]
    for row in payload["user_action_required"]:
        lines += [
            f"## {row['dataset_id']}",
            "",
            f"- official_url: {row['official_url']}",
            f"- CF blockers: {', '.join(row['cf_blockers']) or 'none'}",
            f"- confirmation blockers: {', '.join(row['confirmation_blockers']) or 'none'}",
            f"- action: {row['action']}",
            "",
        ]
    lines.append("Do not convert or evaluate any source until this validator reports conversion_ready targets and a later no-leakage conversion gate passes.")
    return lines


def run_stage42_source_terms_confirmation_validator() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_json(MANIFEST_JSON, payload["manifest"])
    write_md(REPORT_MD, _render_report(payload))
    write_md(MANIFEST_MD, _render_manifest(payload))
    write_md(GATE_MD, _render_gate(payload))
    write_md(USER_ACTION_MD, _render_user_actions(payload))
    return payload


if __name__ == "__main__":
    result = run_stage42_source_terms_confirmation_validator()
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False, sort_keys=True))
