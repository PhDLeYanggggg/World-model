from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
TERMS_JSON = OUT_DIR / "source_terms_validation_stage42.json"
TIME_JSON = OUT_DIR / "source_time_geometry_calibration_stage42.json"
CLOSURE_JSON = OUT_DIR / "source_support_closure_audit_stage42.json"

REPORT_JSON = OUT_DIR / "source_legal_time_action_package_stage42.json"
REPORT_MD = OUT_DIR / "source_legal_time_action_package_stage42.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_source_legal_time_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_do_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
RETRO_README = Path("README_M3W_GOAL_RETROSPECTIVE_CURRENT_ZH.md")
GOAL_SUMMARY_README = Path("README_M3W_CURRENT_GOAL_SUMMARY_ZH.md")
GOAL_RESULTS_README = Path("README_M3W_GOAL_RESULTS_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "m3w_official_metric_seconds_claim_allowed": False,
    "global_t100_deployable_claim_allowed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-DO 是 DA-1 source/legal/time closure action package，不训练模型、不下载数据、不转换数据。",
    "本步骤复核 Stage42-CG terms validator 与 Stage42-BN time/geometry calibration，并生成明确 user_action_required。",
    "local path、parseability、H/FPS evidence 不等于 legal conversion permission。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon；只有 source-specific calibrated subset 可以在未来单独声明。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _gate_passed(payload: Mapping[str, Any], name: str) -> bool:
    gate = payload.get(name, {})
    return bool(gate) and gate.get("passed") == gate.get("total")


def _domain_calibration(time_payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    by_domain: dict[str, dict[str, Any]] = {}
    for row in time_payload.get("source_records", []):
        domain = str(row.get("domain", "unknown"))
        bucket = by_domain.setdefault(
            domain,
            {
                "source_specific_metric_time_sources": [],
                "source_specific_time_only_sources": [],
                "blocked_or_diagnostic_sources": [],
            },
        )
        if row.get("source_specific_metric_time_evidence"):
            bucket["source_specific_metric_time_sources"].append(row.get("source_id"))
        elif row.get("timing", {}).get("annotation_fps"):
            bucket["source_specific_time_only_sources"].append(row.get("source_id"))
        else:
            bucket["blocked_or_diagnostic_sources"].append(row.get("source_id"))
    return by_domain


def _build_action_rows(
    terms_payload: Mapping[str, Any],
    time_payload: Mapping[str, Any],
    closure_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    calibration = _domain_calibration(time_payload)
    closure_by_domain = {str(row.get("domain")): row for row in closure_payload.get("domain_status", [])}
    rows = []
    for row in terms_payload.get("validations", []):
        dataset_id = str(row.get("dataset_id"))
        if dataset_id.startswith("eth"):
            domain = "ETH_UCY"
        elif dataset_id.startswith("ucy"):
            domain = "UCY"
        elif dataset_id.startswith("trajnet"):
            domain = "TrajNet"
        elif dataset_id.startswith("opentraj"):
            domain = "OpenTraj"
        else:
            domain = "other_topdown"
        closure = closure_by_domain.get(domain, {})
        rows.append(
            {
                "dataset_id": dataset_id,
                "domain": domain,
                "official_url": row.get("official_url"),
                "terms_accepted": row.get("terms_accepted_by_user", False),
                "conversion_ready": row.get("conversion_ready", False),
                "conversion_allowed_now": row.get("conversion_allowed_now", False),
                "confirmed_local_path": row.get("confirmed_local_path", ""),
                "source_identity": row.get("source_identity", ""),
                "terms_blockers": row.get("confirmation_blockers", []),
                "cf_blockers": row.get("cf_blockers", []),
                "domain_claim_status": closure.get("claim_status", "not_closed"),
                "domain_blockers": closure.get("blockers", []),
                "source_specific_metric_time_sources": calibration.get(domain, {}).get(
                    "source_specific_metric_time_sources", []
                ),
                "source_specific_time_only_sources": calibration.get(domain, {}).get(
                    "source_specific_time_only_sources", []
                ),
                "required_user_action": _required_user_action(row, closure),
            }
        )
    return rows


def _required_user_action(terms_row: Mapping[str, Any], closure_row: Mapping[str, Any]) -> str:
    blockers = list(terms_row.get("confirmation_blockers", [])) + list(terms_row.get("cf_blockers", []))
    if terms_row.get("conversion_ready"):
        return "ready_for_future_no_leakage_conversion_stage"
    if any("terms" in str(b) or "allowed_use" in str(b) for b in blockers):
        return "accept/confirm official terms, allowed use, acceptance date, local path, and source identity"
    if closure_row.get("blockers"):
        return str(closure_row.get("next_action", "resolve domain closure blockers"))
    return "provide legal local source path and independent source split evidence"


def _input_summary(path: Path, gate_name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    gate = payload.get(gate_name, {})
    return {
        "path": str(path),
        "sha256": _sha256(path),
        gate_name: {
            "passed": gate.get("passed"),
            "total": gate.get("total"),
            "verdict": gate.get("verdict"),
        },
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    claim = payload["claim_boundary"]
    gates = {
        "terms_validator_passed": payload["input_summaries"]["terms"]["stage42_cg_gate"]["passed"]
        == payload["input_summaries"]["terms"]["stage42_cg_gate"]["total"],
        "time_geometry_passed": payload["input_summaries"]["time_geometry"]["stage42_bn_gate"]["passed"]
        == payload["input_summaries"]["time_geometry"]["stage42_bn_gate"]["total"],
        "closure_audit_passed": payload["input_summaries"]["closure"]["stage42_dd_gate"]["passed"]
        == payload["input_summaries"]["closure"]["stage42_dd_gate"]["total"],
        "zero_conversion_ready_recorded": s["conversion_ready_targets"] == 0,
        "global_metric_seconds_blocked": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "global_t100_blocked": claim["global_t100_deployable_claim_allowed"] is False,
        "user_action_rows_present": len(payload["user_action_rows"]) >= 5,
        "official_urls_present": all(bool(row["official_url"]) for row in payload["user_action_rows"]),
        "source_specific_candidates_reported": s["source_specific_metric_time_sources_count"] >= 1,
        "no_conversion_claim": s["converted_datasets_now"] == 0,
        "no_evaluation_claim": s["evaluated_datasets_now"] == 0,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_do_source_legal_time_action_package_pass" if passed == total else "stage42_do_source_legal_time_action_package_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# Stage42-DO Source Legal/Time Action Package",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- gate: `{payload['stage42_do_gate']['passed']} / {payload['stage42_do_gate']['total']}`",
        f"- verdict: `{payload['stage42_do_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Summary",
        "",
        *[f"- {key}: `{value}`" for key, value in payload["summary"].items()],
        "",
        "## User Action Rows",
        "",
        "| dataset | domain | conversion ready | source-specific metric/time candidates | official URL | required action |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for row in payload["user_action_rows"]:
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | {} | {} |".format(
                row["dataset_id"],
                row["domain"],
                row["conversion_ready"],
                ", ".join(map(str, row["source_specific_metric_time_sources"])) or "none",
                row["official_url"],
                row["required_user_action"],
            )
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "- Current source-specific calibration candidates do not permit global metric/seconds claims.",
            "- No source is conversion-ready because terms/path/source-identity confirmation remains missing.",
            "- No dataset is converted or evaluated by Stage42-DO.",
            "- t+100 remains raw-frame diagnostic and not globally deployable.",
            "- Stage5C remains unexecuted and SMC remains disabled.",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | --- |",
            *[f"| `{key}` | `{value}` |" for key, value in payload["stage42_do_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage42-DO Source Legal/Time Closure",
        "",
        "No external source is conversion-ready yet. To unlock restricted source-specific metric/time or t+100 claims, fill the confirmation fields below and rerun the guarded conversion/no-leakage/source-CV path.",
        "",
    ]
    for row in payload["user_action_rows"]:
        lines.extend(
            [
                f"## {row['dataset_id']}",
                "",
                f"- official_url: {row['official_url']}",
                f"- domain: `{row['domain']}`",
                f"- conversion_ready_now: `{row['conversion_ready']}`",
                f"- source_specific_metric_time_candidates: `{row['source_specific_metric_time_sources']}`",
                f"- terms_blockers: `{row['terms_blockers']}`",
                f"- cf_blockers: `{row['cf_blockers']}`",
                f"- domain_blockers: `{row['domain_blockers']}`",
                f"- required_action: {row['required_user_action']}",
                "",
            ]
        )
    lines.extend(
        [
            "Do not convert, evaluate, or make metric/seconds/t100 deployment claims until the validator reports conversion-ready sources and a later no-leakage/source-CV gate passes.",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_do_gate"]
    return [
        "# Stage42-DO Gate",
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
    gate = payload["stage42_do_gate"]
    return [
        "## Stage42-DO Source Legal/Time Action Package",
        "",
        "- source: `fresh_synthesis_from_stage42_cg_bn_dd_after_da1_rerun`",
        "- role: closes the current DA-1 pass as an honest blocker/action package, not as conversion or evaluation.",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- conversion-ready targets: `{s['conversion_ready_targets']}`; converted/evaluated now: `{s['converted_datasets_now']}` / `{s['evaluated_datasets_now']}`.",
        f"- source-specific metric/time candidate count: `{s['source_specific_metric_time_sources_count']}`.",
        "- global metric/seconds/t100 deployable claims remain blocked; Stage5C and SMC remain disabled.",
        f"- user action file: `{USER_ACTION_MD}`.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, RETRO_README, GOAL_SUMMARY_README, GOAL_RESULTS_README]:
        _replace_section(path, "STAGE42_DO_SOURCE_LEGAL_TIME_ACTION_PACKAGE", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-DO source legal/time action package"
    state["current_verdict"] = payload["stage42_do_gate"]["verdict"]
    state["stage42_do_source_legal_time_action_package"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "user_action_required": str(USER_ACTION_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_do_gate"]["verdict"],
        "gates": f"{payload['stage42_do_gate']['passed']}/{payload['stage42_do_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_source_legal_time_action_package() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    terms = read_json(TERMS_JSON, {})
    time_payload = read_json(TIME_JSON, {})
    closure = read_json(CLOSURE_JSON, {})
    user_action_rows = _build_action_rows(terms, time_payload, closure)
    conversion_ready = [row for row in user_action_rows if row["conversion_ready"]]
    source_specific = sorted(
        {
            source
            for row in user_action_rows
            for source in row.get("source_specific_metric_time_sources", [])
            if source
        }
    )
    payload: dict[str, Any] = {
        "source": "fresh_synthesis_from_stage42_cg_bn_dd_after_da1_rerun",
        "stage": "Stage42-DO",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_facts": CURRENT_FACTS,
        "input_summaries": {
            "terms": _input_summary(TERMS_JSON, "stage42_cg_gate", terms),
            "time_geometry": _input_summary(TIME_JSON, "stage42_bn_gate", time_payload),
            "closure": _input_summary(CLOSURE_JSON, "stage42_dd_gate", closure),
        },
        "summary": {
            "targets_checked": len(user_action_rows),
            "conversion_ready_targets": len(conversion_ready),
            "conversion_ready_ids": [row["dataset_id"] for row in conversion_ready],
            "converted_datasets_now": 0,
            "evaluated_datasets_now": 0,
            "source_specific_metric_time_sources_count": len(source_specific),
            "source_specific_metric_time_sources": source_specific,
            "global_metric_seconds_claim_allowed": False,
            "global_t100_deployable_claim_allowed": False,
        },
        "user_action_rows": user_action_rows,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["stage42_do_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_source_legal_time_action_package()
