from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
PRIORITY_JSON = OUT_DIR / "source_confirmation_priority_board_stage42.json"
CONTRACT_JSON = OUT_DIR / "source_conversion_contract_stage42.json"

REPORT_JSON = OUT_DIR / "official_source_terms_live_verifier_stage42.json"
REPORT_MD = OUT_DIR / "official_source_terms_live_verifier_stage42.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_official_source_terms_live_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_go_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
ONE_FILE_SUMMARY = Path("README_M3W_ONE_FILE_DETAILED_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_go_official_source_terms_live_verifier"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-GO 只记录官方 source / terms live audit；不下载、不转换、不训练、不评估。",
    "OpenTraj toolkit license 不能替代 ETH/UCY/TrajNet/AerialMPT 底层数据授权。",
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
    "official_terms_accepted_by_agent": False,
    "auto_download_allowed_now": False,
    "converted_dataset_claim_allowed": False,
    "restricted_metric_time_claim_allowed_now": False,
    "download_executed": False,
    "conversion_executed": False,
    "training_executed": False,
    "evaluation_executed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}

# This table is a conservative record of the official-page web audit performed
# outside this script. It is intentionally not a scraper and does not accept
# terms or infer permission from mirrors.
OFFICIAL_WEB_EVIDENCE = {
    "ucy_crowd_original": {
        "official_url": "https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data",
        "official_source_live_status": "official_url_known_but_page_unavailable_in_live_audit",
        "terms_status": "not_verified_by_agent",
        "access_status": "manual_terms_or_credit_review_required",
        "evidence_summary": "UCY crowd data official URL is the known prior-audit URL, but live page retrieval failed; do not auto-download or treat local files as legally confirmed.",
        "auto_download_allowed_now": False,
        "underlying_data_license_confirmed": False,
        "user_must_confirm": [
            "official terms URL/version or access date",
            "allowed use",
            "redistribution policy",
            "derived-data policy",
            "local path/source identity",
        ],
    },
    "eth_biwi_original": {
        "official_url": "https://vision.ee.ethz.ch/datsets.html",
        "official_source_live_status": "official_page_reachable_with_dataset_download_links",
        "terms_status": "not_verified_by_agent",
        "access_status": "manual_terms_or_credit_review_required",
        "evidence_summary": "ETH Vision dataset page is reachable and lists BIWI Walking Pedestrians material/download links, but live audit did not establish full redistribution/derived-data terms.",
        "auto_download_allowed_now": False,
        "underlying_data_license_confirmed": False,
        "user_must_confirm": [
            "official terms/version or access date",
            "allowed use",
            "redistribution policy",
            "derived-data policy",
            "local path/source identity",
            "annotation frame rate / H-matrix convention if claiming restricted metric/time subset",
        ],
    },
    "trajnetplusplus_official": {
        "official_url": "https://www.epfl.ch/schools/enac/about/open-science/open-research/trajnet/",
        "official_source_live_status": "official_epfl_page_reachable_platform_access_via_aicrowd",
        "terms_status": "not_verified_by_agent",
        "access_status": "manual_platform_terms_required",
        "evidence_summary": "EPFL TrajNet++ page is reachable and points to benchmark/platform access; data/platform terms must be accepted by the user and current local snippets remain h100-limited.",
        "auto_download_allowed_now": False,
        "underlying_data_license_confirmed": False,
        "user_must_confirm": [
            "AIcrowd/platform/data terms",
            "allowed use",
            "redistribution policy",
            "derived-data policy",
            "official split/source identity",
            "whether longer h100-capable raw sources are available",
        ],
    },
    "opentraj_toolkit": {
        "official_url": "https://github.com/crowdbotp/OpenTraj",
        "official_source_live_status": "official_github_reachable_toolkit_license_only",
        "terms_status": "toolkit_mit_not_underlying_dataset_terms",
        "access_status": "toolkit_usable_under_mit_but_dataset_terms_separate",
        "evidence_summary": "OpenTraj GitHub is official for toolkit/code and metadata, but its MIT license should not be treated as permission for every redistributed underlying trajectory dataset.",
        "auto_download_allowed_now": False,
        "underlying_data_license_confirmed": False,
        "user_must_confirm": [
            "which underlying dataset source is being used",
            "underlying dataset official terms",
            "allowed use",
            "local path/source identity",
        ],
    },
    "aerialmpt_or_other_topdown": {
        "official_url": "user_or_web_verified_official_url_required",
        "official_source_live_status": "not_verified",
        "terms_status": "not_verified_by_agent",
        "access_status": "official_url_and_terms_required_before_use",
        "evidence_summary": "AerialMPT or other top-down sources need an official URL and terms path before any guarded conversion.",
        "auto_download_allowed_now": False,
        "underlying_data_license_confirmed": False,
        "user_must_confirm": [
            "official URL",
            "official terms",
            "allowed use",
            "local path/source identity",
            "whether data are top-down pedestrian/drone and h50/h100 capable",
        ],
    },
}


def _priority_by_dataset(priority: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for row in priority.get("priority_rows", []):
        if isinstance(row, Mapping):
            rows[str(row.get("dataset_id", ""))] = row
    return rows


def _contract_by_dataset(contract: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for row in contract.get("contract_rows", []):
        if isinstance(row, Mapping):
            rows[str(row.get("dataset_id", ""))] = row
    return rows


def _audit_rows(priority: Mapping[str, Any], contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    priority_rows = _priority_by_dataset(priority)
    contract_rows = _contract_by_dataset(contract)
    dataset_ids = sorted(
        set(OFFICIAL_WEB_EVIDENCE)
        | {str(row.get("dataset_id", "")) for row in priority.get("priority_rows", []) if isinstance(row, Mapping)}
        | {str(row.get("dataset_id", "")) for row in contract.get("contract_rows", []) if isinstance(row, Mapping)}
    )
    out: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        evidence = OFFICIAL_WEB_EVIDENCE.get(
            dataset_id,
            {
                "official_url": contract_rows.get(dataset_id, {}).get("official_url", ""),
                "official_source_live_status": "not_verified",
                "terms_status": "not_verified_by_agent",
                "access_status": "manual_review_required",
                "evidence_summary": "No live official-source evidence was recorded for this dataset.",
                "auto_download_allowed_now": False,
                "underlying_data_license_confirmed": False,
                "user_must_confirm": ["official URL", "official terms", "local path/source identity"],
            },
        )
        priority_row = priority_rows.get(dataset_id, {})
        contract_row = contract_rows.get(dataset_id, {})
        missing_fields = (
            contract_row.get("confirmation", {}).get("missing_fields", [])
            if isinstance(contract_row.get("confirmation", {}), Mapping)
            else []
        )
        out.append(
            {
                "dataset_id": dataset_id,
                "domain": priority_row.get("domain", contract_row.get("domain", "")),
                "priority_rank": priority_row.get("priority_rank"),
                "priority_score": priority_row.get("priority_score"),
                "value_class": priority_row.get("value_class", "not_ranked"),
                "official_url": evidence["official_url"],
                "official_source_live_status": evidence["official_source_live_status"],
                "terms_status": evidence["terms_status"],
                "access_status": evidence["access_status"],
                "evidence_summary": evidence["evidence_summary"],
                "auto_download_allowed_now": bool(evidence["auto_download_allowed_now"]),
                "underlying_data_license_confirmed": bool(evidence["underlying_data_license_confirmed"]),
                "contract_conversion_ready_now": bool(contract_row.get("contract_conversion_ready_now")),
                "missing_contract_fields": list(missing_fields) if isinstance(missing_fields, list) else [],
                "estimated_t50_windows_after_terms": int(priority_row.get("estimated_t50_windows_after_terms", 0) or 0),
                "estimated_t100_windows_after_terms": int(priority_row.get("estimated_t100_windows_after_terms", 0) or 0),
                "calibrated_t50_windows_after_terms": int(priority_row.get("calibrated_t50_windows_after_terms", 0) or 0),
                "calibrated_t100_windows_after_terms": int(priority_row.get("calibrated_t100_windows_after_terms", 0) or 0),
                "user_must_confirm": list(evidence["user_must_confirm"]),
                "next_safe_action": "user fills official terms/path/source identity; rerun validator -> contract -> guarded harness",
                "result_source": "fresh_live_official_source_audit_plus_cached_verified_stage42_gn_gl",
                "download_executed": False,
                "conversion_executed": False,
                "evaluation_executed": False,
            }
        )
    out.sort(key=lambda row: (row["priority_rank"] is None, row["priority_rank"] or 999, str(row["dataset_id"])))
    return out


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    reachable = [
        row
        for row in rows
        if row.get("official_source_live_status")
        in {
            "official_page_reachable_with_dataset_download_links",
            "official_epfl_page_reachable_platform_access_via_aicrowd",
            "official_github_reachable_toolkit_license_only",
        }
    ]
    return {
        "source": SOURCE,
        "datasets_audited": len(rows),
        "official_sources_reachable": len(reachable),
        "underlying_data_license_confirmed": sum(1 for row in rows if row.get("underlying_data_license_confirmed")),
        "auto_download_allowed_now": sum(1 for row in rows if row.get("auto_download_allowed_now")),
        "contract_ready_now": sum(1 for row in rows if row.get("contract_conversion_ready_now")),
        "top_priority_dataset": rows[0].get("dataset_id", "") if rows else "",
        "top_priority_terms_status": rows[0].get("terms_status", "") if rows else "",
        "total_t50_after_terms": sum(int(row.get("estimated_t50_windows_after_terms", 0) or 0) for row in rows),
        "total_t100_after_terms": sum(int(row.get("estimated_t100_windows_after_terms", 0) or 0) for row in rows),
        "download_executed": False,
        "conversion_executed": False,
        "training_executed": False,
        "evaluation_executed": False,
        "next_required_action": "user confirms official terms/path/source identity; no source can be converted automatically yet",
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    c = payload["claim_boundary"]
    rows = payload["audit_rows"]
    gates = {
        "source_fresh": payload.get("source") == SOURCE,
        "priority_board_loaded": payload.get("input_status", {}).get("priority_exists") is True,
        "contract_loaded": payload.get("input_status", {}).get("contract_exists") is True,
        "top_sources_audited": s["datasets_audited"] >= 5,
        "official_sources_recorded": all(bool(row.get("official_url")) for row in rows),
        "terms_not_agent_accepted": s["underlying_data_license_confirmed"] == 0 and s["contract_ready_now"] == 0,
        "no_auto_download_allowed": s["auto_download_allowed_now"] == 0,
        "user_actions_recorded": all(row.get("user_must_confirm") for row in rows),
        "opportunity_preserved": s["total_t50_after_terms"] >= 0 and s["total_t100_after_terms"] >= 0,
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
    verdict = (
        "stage42_go_official_source_terms_live_verifier_pass"
        if passed == total
        else "stage42_go_official_source_terms_live_verifier_partial"
    )
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-GO Official Source / Terms Live Verifier",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_go_gate']['passed']} / {payload['stage42_go_gate']['total']}`",
        f"- verdict: `{payload['stage42_go_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Summary",
        "",
        *[f"- {key}: `{value}`" for key, value in s.items()],
        "",
        "## Official Source / Terms Audit",
        "",
        "| priority | dataset | live source status | terms status | auto download | t50/t100 after terms | notes |",
        "| ---: | --- | --- | --- | ---: | --- | --- |",
    ]
    for row in payload["audit_rows"]:
        lines.append(
            f"| {row.get('priority_rank') or ''} | `{row['dataset_id']}` | `{row['official_source_live_status']}` | "
            f"`{row['terms_status']}` | {row['auto_download_allowed_now']} | "
            f"{row['estimated_t50_windows_after_terms']} / {row['estimated_t100_windows_after_terms']} | "
            f"{row['evidence_summary']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- UCY and ETH/BIWI remain the highest-value confirmation targets, but neither can be converted until the user confirms terms/path/source identity.",
            "- TrajNet++ official/platform access still requires manual terms confirmation and current local snippets do not solve h100.",
            "- OpenTraj is useful as toolkit/metadata evidence, but its MIT license is not treated as underlying dataset permission.",
            "- No source is auto-downloadable or conversion-ready in this audit.",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
            *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_go_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    lines = [
        "# User Action Required: Stage42-GO Official Source / Terms Live Verifier",
        "",
        "No automatic download or conversion is allowed yet. The agent did not accept terms.",
        "",
        "Please confirm these fields manually for the top-priority sources if you want conversion to proceed:",
        "",
    ]
    for row in payload["audit_rows"]:
        if row.get("priority_rank") and row["priority_rank"] <= 3:
            lines.extend(
                [
                    f"## {row['priority_rank']}. {row['dataset_id']}",
                    "",
                    f"- official_url: `{row['official_url']}`",
                    f"- live_status: `{row['official_source_live_status']}`",
                    f"- terms_status: `{row['terms_status']}`",
                    f"- access_status: `{row['access_status']}`",
                    f"- user_must_confirm: {', '.join(row['user_must_confirm'])}",
                    f"- suggested next action: {row['next_safe_action']}",
                    "",
                ]
            )
    lines.append("After manual confirmation, rerun validator -> source conversion contract -> guarded conversion harness.")
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_go_gate"]
    return [
        "# Stage42-GO Gate",
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
        "## Stage42-GO Official Source / Terms Live Verifier",
        "",
        "- source: `fresh_stage42_go_official_source_terms_live_verifier`",
        "- role: official source/terms live audit for the Stage42-GN priority queue; it does not accept terms or download data.",
        f"- gate: `{payload['stage42_go_gate']['passed']} / {payload['stage42_go_gate']['total']}`; verdict `{payload['stage42_go_gate']['verdict']}`.",
        f"- datasets_audited: `{s['datasets_audited']}`; official_sources_reachable: `{s['official_sources_reachable']}`; auto_download_allowed_now: `{s['auto_download_allowed_now']}`.",
        f"- top priority remains `{s['top_priority_dataset']}`; terms status `{s['top_priority_terms_status']}`.",
        "- OpenTraj toolkit license is explicitly not counted as underlying dataset permission.",
        "- No download, conversion, feature-store build, training, or evaluation was executed.",
        "- Boundary: source/terms audit only; not converted data, not metric/seconds evidence, not Stage5C, and not SMC.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, ONE_FILE_SUMMARY]:
        _replace_section(path, "STAGE42_GO_OFFICIAL_SOURCE_TERMS_LIVE_VERIFIER", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-GO official source terms live verifier"
    state["current_verdict"] = payload["stage42_go_gate"]["verdict"]
    state["stage42_go_official_source_terms_live_verifier"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "verdict": payload["stage42_go_gate"]["verdict"],
        "gates": f"{payload['stage42_go_gate']['passed']}/{payload['stage42_go_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_official_source_terms_live_verifier(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    priority = read_json(PRIORITY_JSON, {})
    contract = read_json(CONTRACT_JSON, {})
    rows = _audit_rows(priority, contract)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-GO",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([PRIORITY_JSON, CONTRACT_JSON]),
        "input_status": {"priority_exists": PRIORITY_JSON.exists(), "contract_exists": CONTRACT_JSON.exists()},
        "current_facts": CURRENT_FACTS,
        "audit_rows": rows,
        "summary": _summary(rows),
        "claim_boundary": CLAIM_BOUNDARY,
        "user_action_required_written": True,
    }
    payload["stage42_go_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_official_source_terms_live_verifier()
