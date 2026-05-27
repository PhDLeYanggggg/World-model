from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
GO_JSON = OUT_DIR / "official_source_terms_live_verifier_stage42.json"

REPORT_JSON = OUT_DIR / "source_terms_paper_claim_guard_stage42.json"
REPORT_MD = OUT_DIR / "source_terms_paper_claim_guard_stage42.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_source_terms_paper_claim_guard_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_gp_gate.md"

DATA_CARD = OUT_DIR / "data_card_stage42.md"
A_JOURNAL_GAP = OUT_DIR / "a_journal_gap_stage42.md"
METHOD_DRAFT = OUT_DIR / "method_draft_stage42.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
ONE_FILE_SUMMARY = Path("README_M3W_ONE_FILE_DETAILED_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_gp_source_terms_paper_claim_guard"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-GP 将 GO 官方 source/terms live audit 写入 paper package claim guard；不下载、不转换、不训练、不评估。",
    "OpenTraj toolkit MIT 许可不能写成 ETH/UCY/TrajNet/AerialMPT 底层数据许可。",
    "用户必须亲自确认 official terms、allowed use、local path、source identity；agent 不能代填 acceptance。",
    "future endpoints / waypoints 只允许作为 supervised/evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon；不能写成 seconds-level，除非未来 source-specific guard 通过。",
    "dataset-local/raw-frame 不能写成 global metric；restricted source-specific metric/time subset 也必须等 legal conversion 后再审计。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "restricted_metric_time_claim_allowed_now": False,
    "auto_download_allowed_now": False,
    "converted_dataset_claim_allowed": False,
    "download_executed": False,
    "conversion_executed": False,
    "training_executed": False,
    "evaluation_executed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}

UNSAFE_SOURCE_CLAIMS = [
    "OpenTraj MIT covers underlying dataset",
    "OpenTraj MIT permits ETH/UCY data reuse",
    "auto-download allowed now: 1",
    "converted dataset claim allowed: true",
    "restricted metric time claim allowed now: true",
    "global metric claim allowed: true",
    "global seconds claim allowed: true",
]


def _audit_rows(go_payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [row for row in go_payload.get("audit_rows", []) if isinstance(row, Mapping)]


def _paper_guard_rows(go_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _audit_rows(go_payload):
        rows.append(
            {
                "dataset_id": row.get("dataset_id", ""),
                "priority_rank": row.get("priority_rank"),
                "official_source_live_status": row.get("official_source_live_status", ""),
                "terms_status": row.get("terms_status", ""),
                "auto_download_allowed_now": bool(row.get("auto_download_allowed_now")),
                "underlying_data_license_confirmed": bool(row.get("underlying_data_license_confirmed")),
                "contract_conversion_ready_now": bool(row.get("contract_conversion_ready_now")),
                "paper_claim_status": "blocked_until_user_terms_path_source_confirmation"
                if not row.get("contract_conversion_ready_now")
                else "eligible_for_future_guarded_conversion_only",
                "allowed_paper_wording": _allowed_wording(row),
                "disallowed_paper_wording": _disallowed_wording(row),
            }
        )
    return rows


def _allowed_wording(row: Mapping[str, Any]) -> str:
    dataset = row.get("dataset_id", "")
    if dataset == "opentraj_toolkit":
        return "OpenTraj is used only as toolkit/metadata/source-discovery evidence; underlying dataset terms are separate."
    return (
        f"{dataset} is a post-confirmation source candidate with official/source terms still requiring user confirmation; "
        "it is not counted as converted or evaluated data."
    )


def _disallowed_wording(row: Mapping[str, Any]) -> str:
    dataset = row.get("dataset_id", "")
    if dataset == "opentraj_toolkit":
        return "Do not write that OpenTraj MIT license grants permission for ETH/UCY/TrajNet underlying data."
    return f"Do not write that {dataset} has been legally converted, evaluated, auto-downloaded, or metric/seconds-calibrated."


def _summary(rows: list[Mapping[str, Any]], go_payload: Mapping[str, Any]) -> dict[str, Any]:
    go_summary = go_payload.get("summary", {})
    return {
        "source": SOURCE,
        "go_source": go_payload.get("source", ""),
        "go_verdict": go_payload.get("stage42_go_gate", {}).get("verdict", ""),
        "datasets_guarded": len(rows),
        "underlying_data_license_confirmed": sum(1 for row in rows if row.get("underlying_data_license_confirmed")),
        "auto_download_allowed_now": sum(1 for row in rows if row.get("auto_download_allowed_now")),
        "contract_ready_now": sum(1 for row in rows if row.get("contract_conversion_ready_now")),
        "total_t50_after_terms": go_summary.get("total_t50_after_terms", 0),
        "total_t100_after_terms": go_summary.get("total_t100_after_terms", 0),
        "paper_files_refreshed": [str(DATA_CARD), str(A_JOURNAL_GAP), str(METHOD_DRAFT)],
        "download_executed": False,
        "conversion_executed": False,
        "training_executed": False,
        "evaluation_executed": False,
        "next_required_action": "paper claims must keep source/legal blocker wording until user confirmation and guarded conversion pass",
    }


def _paper_guard_section(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "## Source / Terms Claim Guard",
        "",
        "- source: `fresh_stage42_gp_source_terms_paper_claim_guard`",
        f"- upstream GO verdict: `{s['go_verdict']}`",
        f"- datasets guarded: `{s['datasets_guarded']}`",
        f"- underlying data licenses confirmed now: `{s['underlying_data_license_confirmed']}`",
        f"- auto-download allowed now: `{s['auto_download_allowed_now']}`",
        f"- contract-ready sources now: `{s['contract_ready_now']}`",
        "- OpenTraj toolkit/code metadata may be cited separately from underlying ETH/UCY/TrajNet dataset permission.",
        "- UCY / ETH-BIWI / TrajNet / AerialMPT remain source candidates until user-confirmed terms, path, source identity, guarded conversion, no-leakage audit, and source-CV pass.",
        "- No result in this paper package may say these candidates are newly converted/evaluated data yet.",
        "- No result may claim global metric or seconds-level prediction from these sources yet.",
        "",
        "| dataset | allowed wording | disallowed wording |",
        "| --- | --- | --- |",
    ]
    for row in payload["paper_guard_rows"]:
        lines.append(f"| `{row['dataset_id']}` | {row['allowed_paper_wording']} | {row['disallowed_paper_wording']} |")
    return lines


def _scan_refreshed_blocks(paths: list[Path]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for path in paths:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        lower = text.lower()
        for phrase in UNSAFE_SOURCE_CLAIMS:
            if phrase.lower() in lower:
                violations.append({"file": str(path), "phrase": phrase})
    return {"violations": violations, "violation_count": len(violations)}


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    scan = payload["claim_scan"]
    c = payload["claim_boundary"]
    gates = {
        "source_fresh": payload.get("source") == SOURCE,
        "go_loaded": payload.get("input_status", {}).get("go_exists") is True,
        "go_gate_passed": payload.get("go_gate", {}).get("passed") == payload.get("go_gate", {}).get("total"),
        "paper_rows_built": s["datasets_guarded"] >= 5,
        "paper_files_refreshed": all(Path(path).exists() for path in s["paper_files_refreshed"]),
        "no_unsafe_source_claims": scan["violation_count"] == 0,
        "no_license_or_auto_download_claim": s["underlying_data_license_confirmed"] == 0
        and s["auto_download_allowed_now"] == 0
        and s["contract_ready_now"] == 0,
        "no_download_conversion_training_eval": not (
            s["download_executed"] or s["conversion_executed"] or s["training_executed"] or s["evaluation_executed"]
        ),
        "user_action_written": payload.get("user_action_required_written") is True,
        "no_metric_seconds_overclaim": c["global_metric_claim_allowed"] is False
        and c["global_seconds_claim_allowed"] is False
        and c["restricted_metric_time_claim_allowed_now"] is False,
        "stage5c_false": c["stage5c_executed"] is False,
        "smc_false": c["smc_enabled"] is False,
    }
    passed = sum(bool(value) for value in gates.values())
    total = len(gates)
    verdict = "stage42_gp_source_terms_paper_claim_guard_pass" if passed == total else "stage42_gp_source_terms_paper_claim_guard_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-GP Source Terms Paper Claim Guard",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_gp_gate']['passed']} / {payload['stage42_gp_gate']['total']}`",
        f"- verdict: `{payload['stage42_gp_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Summary",
        "",
        *[f"- {key}: `{value}`" for key, value in s.items()],
        "",
        "## Guard Rows",
        "",
        "| dataset | paper claim status | allowed wording | disallowed wording |",
        "| --- | --- | --- | --- |",
    ]
    for row in payload["paper_guard_rows"]:
        lines.append(
            f"| `{row['dataset_id']}` | `{row['paper_claim_status']}` | {row['allowed_paper_wording']} | {row['disallowed_paper_wording']} |"
        )
    lines.extend(
        [
            "",
            "## Claim Scan",
            "",
            f"- unsafe source claim violations: `{payload['claim_scan']['violation_count']}`",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
            *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_gp_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    return [
        "# User Action Required: Stage42-GP Source Terms Paper Claim Guard",
        "",
        "Paper/package claims have been refreshed to keep source terms blockers explicit.",
        "",
        "Before any claim can change from candidate/opportunity to converted/evaluated data, the user must:",
        "",
        "1. Confirm official terms and accepted version/access date.",
        "2. Confirm allowed use, redistribution, and derived-data status.",
        "3. Confirm local path and source identity.",
        "4. Rerun validator, source conversion contract, guarded conversion harness, source-specific converter, no-leakage, and source-CV.",
        "",
        "Do not write that OpenTraj MIT covers underlying datasets, and do not write metric/seconds-level claims from unconverted source candidates.",
    ]


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_gp_gate"]
    return [
        "# Stage42-GP Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _refresh_paper_files(payload: Mapping[str, Any]) -> None:
    lines = _paper_guard_section(payload)
    for path in [DATA_CARD, A_JOURNAL_GAP, METHOD_DRAFT]:
        _replace_section(path, "STAGE42_GP_SOURCE_TERMS_PAPER_CLAIM_GUARD", lines)


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    return [
        "## Stage42-GP Source Terms Paper Claim Guard",
        "",
        "- source: `fresh_stage42_gp_source_terms_paper_claim_guard`",
        "- role: writes the GO source/terms blocker into data card, method draft, and A-journal gap so paper claims cannot overrun legal/source evidence.",
        f"- gate: `{payload['stage42_gp_gate']['passed']} / {payload['stage42_gp_gate']['total']}`; verdict `{payload['stage42_gp_gate']['verdict']}`.",
        f"- paper files refreshed: `{len(s['paper_files_refreshed'])}`; unsafe source-claim violations: `{payload['claim_scan']['violation_count']}`.",
        "- No source is license-confirmed, auto-downloadable, conversion-ready, converted, trained, or evaluated by this step.",
        "- Boundary: source/terms paper guard only; not converted data, not metric/seconds evidence, not Stage5C, and not SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, ONE_FILE_SUMMARY]:
        _replace_section(path, "STAGE42_GP_SOURCE_TERMS_PAPER_CLAIM_GUARD", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-GP source terms paper claim guard"
    state["current_verdict"] = payload["stage42_gp_gate"]["verdict"]
    state["stage42_gp_source_terms_paper_claim_guard"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "verdict": payload["stage42_gp_gate"]["verdict"],
        "gates": f"{payload['stage42_gp_gate']['passed']}/{payload['stage42_gp_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_source_terms_paper_claim_guard(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    go_payload = read_json(GO_JSON, {})
    rows = _paper_guard_rows(go_payload)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-GP",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([GO_JSON]),
        "input_status": {"go_exists": GO_JSON.exists()},
        "go_gate": go_payload.get("stage42_go_gate", {}),
        "current_facts": CURRENT_FACTS,
        "paper_guard_rows": rows,
        "summary": _summary(rows, go_payload),
        "claim_boundary": CLAIM_BOUNDARY,
        "user_action_required_written": True,
    }
    _refresh_paper_files(payload)
    payload["claim_scan"] = _scan_refreshed_blocks([DATA_CARD, A_JOURNAL_GAP, METHOD_DRAFT])
    payload["stage42_gp_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_source_terms_paper_claim_guard()
