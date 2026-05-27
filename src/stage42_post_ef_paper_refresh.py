from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
EB_JSON = OUT_DIR / "paper_package_post_ea_refresh_stage42.json"
EC_JSON = OUT_DIR / "group_consistency_contribution_audit_stage42.json"
EE_JSON = OUT_DIR / "context_switchability_materiality_audit_stage42.json"
EF_JSON = OUT_DIR / "source_terms_gap_audit_stage42.json"

REPORT_JSON = OUT_DIR / "paper_package_post_ef_refresh_stage42.json"
REPORT_MD = OUT_DIR / "paper_package_post_ef_refresh_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_eg_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

PAPER_FILES = [
    OUT_DIR / "paper_outline_stage42.md",
    OUT_DIR / "method_draft_stage42.md",
    OUT_DIR / "experiment_tables_stage42.md",
    OUT_DIR / "ablation_tables_stage42.md",
    OUT_DIR / "failure_taxonomy_stage42.md",
    OUT_DIR / "model_card_stage42.md",
    OUT_DIR / "data_card_stage42.md",
    OUT_DIR / "reproducibility_stage42.md",
    OUT_DIR / "a_journal_gap_stage42.md",
]

SOURCE = "fresh_paper_refresh_from_stage42_eb_ec_ee_ef"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-EG is a post-EE/EF paper claim refresh; it does not train, convert, download, or tune thresholds.",
    "本阶段把 context materiality negative result 和 source terms gap 写入 paper claim/gap matrix。",
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
    "context_main_claim_allowed": False,
    "source_conversion_claim_allowed": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _paper_claim_matrix(ec: Mapping[str, Any], ee: Mapping[str, Any], ef: Mapping[str, Any]) -> list[dict[str, Any]]:
    ec_summary = ec["summary"]
    ee_summary = ee["summary"]
    ef_summary = ef["summary"]
    return [
        {
            "claim": "protected_source_level_group_consistency_full_waypoint",
            "status": "supported_source_level",
            "evidence": "Stage42-EC/DY/DZ/EA supports explicit group-consistency full-waypoint source-level repair with dual-domain bootstrap.",
            "main_claim_allowed": True,
            "boundary": "protected, source-level, dataset-local/raw-frame only; not global ungated replacement",
            "key_numbers": {
                "global_all_ci_low": ec_summary["statistical_evidence"]["global_all_ci_low"],
                "global_t50_ci_low": ec_summary["statistical_evidence"]["global_t50_ci_low"],
                "global_hard_ci_low": ec_summary["statistical_evidence"]["global_hard_ci_low"],
            },
        },
        {
            "claim": "current_context_switchability_scene_goal_neighbor_interaction",
            "status": "blocked_materiality_too_small",
            "evidence": "Stage42-EE fresh-reruns Stage42-DC and finds selected context deltas far below 1pp materiality threshold.",
            "main_claim_allowed": False,
            "boundary": "may be discussed as negative evidence and future work only",
            "key_numbers": {
                "delta_all": ee_summary["selected_delta_all"],
                "delta_t50": ee_summary["selected_delta_t50"],
                "delta_hard": ee_summary["selected_delta_hard"],
                "material_delta_threshold": ee_summary["material_delta_threshold"],
            },
        },
        {
            "claim": "source_conversion_metric_time_expansion",
            "status": "blocked_until_terms_confirmation",
            "evidence": "Stage42-EF reruns source terms validation and records conversion_ready_now=0 with concrete missing fields.",
            "main_claim_allowed": False,
            "boundary": "technical-after-terms potential can be reported, but no conversion/evaluation/metric-time claim is allowed now",
            "key_numbers": {
                "conversion_ready_now": ef_summary["conversion_ready_now"],
                "estimated_t50_windows_after_terms": ef_summary["estimated_t50_windows_after_terms"],
                "estimated_t100_windows_after_terms": ef_summary["estimated_t100_windows_after_terms"],
                "top_unblock_targets": ef_summary["top_unblock_targets"],
            },
        },
        {
            "claim": "global_metric_or_seconds_level_world_model",
            "status": "forbidden",
            "evidence": "Metric/time calibration remains source-specific and legally blocked for new conversion; SDD/external remain raw-frame/dataset-local.",
            "main_claim_allowed": False,
            "boundary": "no global metric, no seconds-level horizon, no true-3D/foundation claim",
            "key_numbers": {},
        },
    ]


def _summary(eb: Mapping[str, Any], ec: Mapping[str, Any], ee: Mapping[str, Any], ef: Mapping[str, Any]) -> dict[str, Any]:
    matrix = _paper_claim_matrix(ec, ee, ef)
    supported = [row["claim"] for row in matrix if row["main_claim_allowed"]]
    blocked = [row["claim"] for row in matrix if not row["main_claim_allowed"]]
    ee_s = ee["summary"]
    ef_s = ef["summary"]
    return {
        "source": SOURCE,
        "supported_main_claims": supported,
        "blocked_or_diagnostic_claims": blocked,
        "context_materiality": {
            "selected_candidate": ee_s["selected_candidate"],
            "material_context_contribution": ee_s["material_context_contribution"],
            "delta_all": ee_s["selected_delta_all"],
            "delta_t50": ee_s["selected_delta_t50"],
            "delta_hard": ee_s["selected_delta_hard"],
            "threshold": ee_s["material_delta_threshold"],
        },
        "source_terms_gap": {
            "conversion_ready_now": ef_s["conversion_ready_now"],
            "converted_datasets_now": ef_s["converted_datasets_now"],
            "evaluated_datasets_now": ef_s["evaluated_datasets_now"],
            "estimated_t50_windows_after_terms": ef_s["estimated_t50_windows_after_terms"],
            "estimated_t100_windows_after_terms": ef_s["estimated_t100_windows_after_terms"],
            "top_unblock_targets": ef_s["top_unblock_targets"],
        },
        "paper_claim_matrix": matrix,
        "paper_verdict": {
            "paper_package_refreshed_after_ee_ef": True,
            "group_consistency_main_claim_allowed": "protected_source_level_group_consistency_full_waypoint" in supported,
            "context_main_claim_allowed": False,
            "source_conversion_claim_allowed": False,
            "metric_seconds_claim_allowed": False,
            "foundation_claim_allowed": False,
            "stage5c_execution_allowed": False,
            "smc_allowed": False,
            "a_journal_candidate_status": "protected_2p5d_candidate_package_strengthened_but_not_foundation_or_metric_ready",
        },
        "eb_verdict": eb.get("stage42_eb_gate", {}).get("verdict", ""),
    }


def _refresh_lines(summary: Mapping[str, Any]) -> list[str]:
    context = summary["context_materiality"]
    source = summary["source_terms_gap"]
    return [
        "## Stage42-EG Post-EE/EF Paper Claim Refresh",
        "",
        "- source: `fresh_paper_refresh_from_stage42_eb_ec_ee_ef`",
        "- role: integrate context materiality and source terms gap evidence into the paper claim/gap matrix.",
        "- This is a paper-package refresh, not new training, conversion, download, or threshold tuning.",
        "",
        "### Main Claim Boundary After EE/EF",
        "",
        "- Supported main claim: protected source-level group-consistency full-waypoint dynamics with dual-domain bootstrap evidence.",
        f"- Context main claim remains blocked: selected `{context['selected_candidate']}` deltas all/t50/hard `{context['delta_all']:.6f}` / `{context['delta_t50']:.6f}` / `{context['delta_hard']:.6f}`, below threshold `{context['threshold']}`.",
        f"- Source conversion remains blocked: conversion_ready_now `{source['conversion_ready_now']}`, converted/evaluated now `{source['converted_datasets_now']}` / `{source['evaluated_datasets_now']}`.",
        f"- Source unlock potential after terms: t50/t100 `{source['estimated_t50_windows_after_terms']}` / `{source['estimated_t100_windows_after_terms']}`, top targets `{source['top_unblock_targets']}`.",
        "- Still forbidden: true 3D, foundation model, global metric/seconds-level claims, Stage5C execution, and SMC readiness.",
    ]


def _refresh_paper_files(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    lines = _refresh_lines(summary)
    status: list[dict[str, Any]] = []
    for path in PAPER_FILES:
        _replace_section(path, "STAGE42_EG_POST_EE_EF_PAPER_REFRESH", lines)
        text = path.read_text(encoding="utf-8")
        status.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "contains_stage42_eg": "Stage42-EG Post-EE/EF Paper Claim Refresh" in text,
                "contains_context_blocker": "Context main claim remains blocked" in text,
                "contains_source_terms_blocker": "Source conversion remains blocked" in text,
                "contains_group_claim": "protected source-level group-consistency full-waypoint dynamics" in text,
                "contains_non_claims": "Still forbidden: true 3D" in text and "Stage5C execution" in text and "SMC readiness" in text,
            }
        )
    return status


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["paper_refresh_summary"]
    gates = {
        "eb_input_passed": payload["inputs"]["stage42_eb"].get("stage42_eb_gate", {}).get("passed")
        == payload["inputs"]["stage42_eb"].get("stage42_eb_gate", {}).get("total"),
        "ec_input_passed": payload["inputs"]["stage42_ec"].get("stage42_ec_gate", {}).get("passed")
        == payload["inputs"]["stage42_ec"].get("stage42_ec_gate", {}).get("total"),
        "ee_input_passed": payload["inputs"]["stage42_ee"].get("stage42_ee_gate", {}).get("passed")
        == payload["inputs"]["stage42_ee"].get("stage42_ee_gate", {}).get("total"),
        "ef_input_passed": payload["inputs"]["stage42_ef"].get("stage42_ef_gate", {}).get("passed")
        == payload["inputs"]["stage42_ef"].get("stage42_ef_gate", {}).get("total"),
        "paper_files_refreshed": all(row["contains_stage42_eg"] for row in payload["paper_file_status"]),
        "group_consistency_claim_preserved": s["paper_verdict"]["group_consistency_main_claim_allowed"] is True,
        "context_main_claim_blocked": s["paper_verdict"]["context_main_claim_allowed"] is False
        and s["context_materiality"]["material_context_contribution"] is False,
        "source_conversion_claim_blocked": s["paper_verdict"]["source_conversion_claim_allowed"] is False
        and s["source_terms_gap"]["conversion_ready_now"] == 0,
        "metric_seconds_overclaim_blocked": s["paper_verdict"]["metric_seconds_claim_allowed"] is False,
        "foundation_overclaim_blocked": s["paper_verdict"]["foundation_claim_allowed"] is False,
        "stage5c_false": s["paper_verdict"]["stage5c_execution_allowed"] is False,
        "smc_false": s["paper_verdict"]["smc_allowed"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_eg_post_ee_ef_paper_refresh_pass" if passed == total else "stage42_eg_post_ee_ef_paper_refresh_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["paper_refresh_summary"]
    gate = payload["stage42_eg_gate"]
    lines = [
        "# Stage42-EG Post-EE/EF Paper Claim Refresh",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Claim Matrix",
        "",
        "| claim | status | main claim allowed | evidence | boundary |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for row in s["paper_claim_matrix"]:
        lines.append(
            f"| `{row['claim']}` | `{row['status']}` | {row['main_claim_allowed']} | {row['evidence']} | {row['boundary']} |"
        )
    lines += [
        "",
        "## Summary",
        "",
        f"- supported_main_claims: `{s['supported_main_claims']}`",
        f"- blocked_or_diagnostic_claims: `{s['blocked_or_diagnostic_claims']}`",
        f"- context materiality delta all/t50/hard: `{s['context_materiality']['delta_all']:.6f}` / `{s['context_materiality']['delta_t50']:.6f}` / `{s['context_materiality']['delta_hard']:.6f}`",
        f"- source conversion_ready_now: `{s['source_terms_gap']['conversion_ready_now']}`",
        f"- source t50/t100 after terms potential: `{s['source_terms_gap']['estimated_t50_windows_after_terms']}` / `{s['source_terms_gap']['estimated_t100_windows_after_terms']}`",
        "",
        "## Paper File Status",
        "",
        "| file | refreshed | context blocker | source blocker | group claim | non-claims |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["paper_file_status"]:
        lines.append(
            f"| `{row['path']}` | {row['contains_stage42_eg']} | {row['contains_context_blocker']} | {row['contains_source_terms_blocker']} | {row['contains_group_claim']} | {row['contains_non_claims']} |"
        )
    lines += [
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_eg_gate"]
    return [
        "# Stage42-EG Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload["paper_refresh_summary"])
    for path in [README_RESULTS, M3W_README, GOAL_SUMMARY]:
        _replace_section(path, "STAGE42_EG_POST_EE_EF_PAPER_REFRESH", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-EG post-EE/EF paper claim refresh"
    state["current_verdict"] = payload["stage42_eg_gate"]["verdict"]
    state["stage42_eg_post_ee_ef_paper_refresh"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_eg_gate"]["verdict"],
        "gates": f"{payload['stage42_eg_gate']['passed']}/{payload['stage42_eg_gate']['total']}",
        "summary": payload["paper_refresh_summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_post_ef_paper_refresh(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    eb = read_json(EB_JSON, {})
    ec = read_json(EC_JSON, {})
    ee = read_json(EE_JSON, {})
    ef = read_json(EF_JSON, {})
    summary = _summary(eb, ec, ee, ef)
    paper_status = _refresh_paper_files(summary)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-EG",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([EB_JSON, EC_JSON, EE_JSON, EF_JSON]),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "stage42_eb": {"stage42_eb_gate": eb.get("stage42_eb_gate", {})},
            "stage42_ec": {"stage42_ec_gate": ec.get("stage42_ec_gate", {})},
            "stage42_ee": {"stage42_ee_gate": ee.get("stage42_ee_gate", {})},
            "stage42_ef": {"stage42_ef_gate": ef.get("stage42_ef_gate", {})},
        },
        "paper_refresh_summary": summary,
        "paper_file_status": paper_status,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["stage42_eg_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_post_ef_paper_refresh()
