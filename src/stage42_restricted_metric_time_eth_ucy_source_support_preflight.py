from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_full_waypoint_bridge_shape_audit import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

HJ_JSON = OUT_DIR / "restricted_metric_time_source_cv_preflight_stage42.json"
BL_JSON = OUT_DIR / "eth_person_xml_t100_conversion_stage42.json"
CG_JSON = OUT_DIR / "source_terms_validation_stage42.json"

REPORT_JSON = OUT_DIR / "restricted_metric_time_eth_ucy_source_support_stage42.json"
REPORT_MD = OUT_DIR / "restricted_metric_time_eth_ucy_source_support_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_hk_gate.md"
USER_ACTION_MD = OUT_DIR / "user_action_required_restricted_metric_time_eth_ucy_source_support_stage42.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
USER_SUMMARY = Path("README_M3W_USER_DETAILED_SUMMARY_ZH.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
GOAL_SUMMARY = Path("README_M3W_GOAL_FULL_SUMMARY_ZH.md")
CURRENT_SUMMARY = Path("README_M3W_CURRENT_DETAILED_SUMMARY_2026_05_27_ZH.md")
A_JOURNAL_GAP = OUT_DIR / "a_journal_gap_stage42.md"
RESEARCH_STATE = Path("research_state.json")

SOURCE = "fresh_stage42_hk_restricted_metric_time_eth_ucy_source_support_preflight"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-HK 是 ETH_UCY restricted metric/time source-support augmentation preflight，不训练、不转换、不下载、不调 threshold。",
    "本阶段合并 Stage42-HJ 的 ETH_UCY blocker 与 Stage42-BL 的 ETH-Person XML technical dry-run。",
    "ETH-Person XML local files 仍是 terms-unverified；technical source support 不等于 official converted/evaluated data。",
    "future endpoint / future waypoint 只可作为 supervised/evaluation labels，不可作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 建 goals，不使用 test metrics 调 threshold。",
    "restricted seconds/metric wording 仍需 user terms confirmation、guarded conversion、no-leakage、source-CV、final test。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _gate_passed(payload: Mapping[str, Any], key: str) -> bool:
    gate = payload.get(key, {})
    return bool(gate and gate.get("passed") == gate.get("total") and int(gate.get("total", 0)) > 0)


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    hj = read_json(HJ_JSON, {})
    bl = read_json(BL_JSON, {})
    cg = read_json(CG_JSON, {})

    hj_summary = dict(hj.get("summary", {}))
    bl_summary = dict(bl.get("summary", {}))
    bl_source_cv = dict(bl.get("source_cv", {}))
    windows_by_source = dict(bl_source_cv.get("windows_by_source", {}))
    candidate_sources = list(bl.get("candidate_sources", []))
    eth_sources = [row for row in candidate_sources if row.get("domain") == "ETH_UCY" and row.get("t100_capable")]
    eth_person_sources = [row for row in eth_sources if str(row.get("source_id", "")).startswith("ETH-Person_")]
    eth_seq_sources = [row for row in eth_sources if str(row.get("source_id", "")).startswith("ETH_seq_")]
    independent_keys = sorted({str(row.get("independent_key")) for row in eth_sources if row.get("independent_key")})

    total_t50 = int(sum(int(windows_by_source.get(str(row.get("source_id")), {}).get("t50_windows", 0)) for row in eth_sources))
    total_t100 = int(sum(int(windows_by_source.get(str(row.get("source_id")), {}).get("t100_windows", 0)) for row in eth_sources))
    fold_count = int(bl_summary.get("source_cv_folds", 0))
    technical_safe_positive = bool(bl_summary.get("technical_t100_all_folds_safe_positive"))
    terms_confirmed = bool(bl_summary.get("license_terms_confirmed"))
    cg_ready_count = int(cg.get("summary", {}).get("conversion_ready_targets", 0) or 0)

    summary = {
        "source": SOURCE,
        "hj_verdict": hj.get("stage42_hj_gate", {}).get("verdict"),
        "bl_verdict": bl.get("stage42_bl_gate", {}).get("verdict"),
        "cg_verdict": cg.get("stage42_cg_gate", {}).get("verdict"),
        "hj_eth_ucy_blocked_after_terms": "ETH_UCY" in hj_summary.get("domains_source_cv_blocked_after_terms", []),
        "eth_person_xml_candidate_sources": len(eth_person_sources),
        "eth_seq_candidate_sources": len(eth_seq_sources),
        "augmented_eth_ucy_sources_after_terms": len(eth_sources),
        "augmented_eth_ucy_independent_keys": len(independent_keys),
        "augmented_eth_ucy_t50_windows_after_terms": total_t50,
        "augmented_eth_ucy_t100_windows_after_terms": total_t100,
        "augmented_eth_ucy_source_cv_folds_after_terms": fold_count,
        "augmented_eth_ucy_source_cv_feasible_after_terms": len(independent_keys) >= 2 and total_t100 > 0,
        "augmented_eth_ucy_robust_source_cv_feasible_after_terms": len(independent_keys) >= 3 and fold_count >= 3,
        "cached_bl_technical_t100_safe_positive": technical_safe_positive,
        "cached_bl_technical_t100_mean_improvement_vs_fallback": bl_summary.get("technical_t100_mean_improvement_vs_fallback"),
        "terms_confirmed": terms_confirmed,
        "conversion_ready_targets_now": cg_ready_count,
        "restricted_metric_time_ready_now": False,
        "conversion_executed": False,
        "evaluation_executed": False,
        "training_run": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-HK Restricted Metric/Time ETH_UCY Source-Support Augmentation Preflight",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([HJ_JSON, BL_JSON, CG_JSON]),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "hj_gate_passed": _gate_passed(hj, "stage42_hj_gate"),
            "bl_gate_passed": _gate_passed(bl, "stage42_bl_gate"),
            "cg_gate_passed": _gate_passed(cg, "stage42_cg_gate"),
        },
        "eth_ucy_augmented_sources": [
            {
                "source": "cached_verified_from_stage42_bl_terms_unverified_dry_run",
                "source_id": row.get("source_id"),
                "independent_key": row.get("independent_key"),
                "relative_path": row.get("relative_path"),
                "parsed_rows": row.get("parsed_rows"),
                "unique_agents": row.get("unique_agents"),
                "max_track_points": row.get("max_track_points"),
                "t50_windows": windows_by_source.get(str(row.get("source_id")), {}).get("t50_windows", 0),
                "t100_windows": windows_by_source.get(str(row.get("source_id")), {}).get("t100_windows", 0),
                "license_status": row.get("license_status"),
                "usable_after_terms": bool(row.get("t100_capable")),
                "ready_now": False,
            }
            for row in eth_sources
        ],
        "summary": summary,
        "claim_boundary": {
            "eth_person_xml_terms_unverified": not terms_confirmed,
            "source_support_preflight_is_conversion": False,
            "source_support_preflight_is_evaluation": False,
            "restricted_metric_time_claim_allowed_now": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "user_action_required": [
            "Confirm ETH/BIWI and ETH-Person/OpenTraj local source terms, local paths, and source identity.",
            "Rerun `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py` with explicit confirmation fields.",
            "Only after terms confirmation, run guarded ETH_UCY restricted metric/time conversion, no-leakage, source-CV, and final test.",
        ],
    }
    payload["stage42_hk_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    claim = payload["claim_boundary"]
    gates = {
        "source_fresh": payload.get("source") == SOURCE,
        "hj_input_passed": payload["inputs"]["hj_gate_passed"] is True,
        "bl_input_passed": payload["inputs"]["bl_gate_passed"] is True,
        "cg_input_passed": payload["inputs"]["cg_gate_passed"] is True,
        "hj_eth_ucy_blocker_detected": summary["hj_eth_ucy_blocked_after_terms"] is True,
        "eth_person_xml_candidates_present": summary["eth_person_xml_candidate_sources"] >= 4,
        "augmented_sources_enough": summary["augmented_eth_ucy_independent_keys"] >= 5,
        "augmented_t50_t100_windows_present": summary["augmented_eth_ucy_t50_windows_after_terms"] > 0
        and summary["augmented_eth_ucy_t100_windows_after_terms"] > 0,
        "augmented_eth_ucy_source_cv_feasible_after_terms": summary["augmented_eth_ucy_source_cv_feasible_after_terms"] is True,
        "augmented_eth_ucy_robust_source_cv_feasible_after_terms": summary["augmented_eth_ucy_robust_source_cv_feasible_after_terms"] is True,
        "cached_bl_technical_support_positive_recorded": summary["cached_bl_technical_t100_safe_positive"] is True,
        "terms_still_block_conversion_now": summary["terms_confirmed"] is False
        and summary["conversion_ready_targets_now"] == 0,
        "no_conversion_or_evaluation_claim": claim["source_support_preflight_is_conversion"] is False
        and claim["source_support_preflight_is_evaluation"] is False,
        "global_metric_seconds_blocked": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = (
        "stage42_hk_eth_ucy_source_support_preflight_pass_terms_blocked"
        if passed == total
        else "stage42_hk_eth_ucy_source_support_preflight_partial"
    )
    return {"source": payload.get("source", SOURCE), "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hk_gate"]
    s = payload["summary"]
    lines = [
        "# Stage42-HK ETH_UCY Restricted Metric/Time Source-Support Preflight",
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
        "## Summary",
        "",
        f"- HJ ETH_UCY blocker detected: `{s['hj_eth_ucy_blocked_after_terms']}`",
        f"- ETH-Person XML candidate sources: `{s['eth_person_xml_candidate_sources']}`",
        f"- augmented ETH_UCY independent sources after terms: `{s['augmented_eth_ucy_independent_keys']}`",
        f"- augmented ETH_UCY source-CV feasible after terms: `{s['augmented_eth_ucy_source_cv_feasible_after_terms']}`",
        f"- augmented ETH_UCY robust source-CV feasible after terms: `{s['augmented_eth_ucy_robust_source_cv_feasible_after_terms']}`",
        f"- augmented t50/t100 windows after terms: `{s['augmented_eth_ucy_t50_windows_after_terms']}` / `{s['augmented_eth_ucy_t100_windows_after_terms']}`",
        f"- cached BL technical t100 safe-positive: `{s['cached_bl_technical_t100_safe_positive']}`",
        f"- cached BL technical t100 mean improvement vs fallback: `{s['cached_bl_technical_t100_mean_improvement_vs_fallback']}`",
        f"- conversion ready targets now: `{s['conversion_ready_targets_now']}`",
        "",
        "## Augmented ETH_UCY Sources",
        "",
        "| source | path | rows | agents | max track | t50 | t100 | ready now |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["eth_ucy_augmented_sources"]:
        lines.append(
            f"| `{row['source_id']}` | `{row['relative_path']}` | {row['parsed_rows']} | "
            f"{row['unique_agents']} | {row['max_track_points']} | {row['t50_windows']} | "
            f"{row['t100_windows']} | {row['ready_now']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- HJ correctly records ETH_UCY as blocked under the original HI source list because ETH_seq_hotel has no t100 windows.",
        "- HK shows the local ETH-Person XML technical dry-run can supply enough ETH_UCY independent t100-capable sources after terms.",
        "- This narrows the ETH_UCY blocker from raw source support to user-confirmed terms plus guarded conversion/evaluation.",
        "- No conversion, no official evaluation, no model training, no metric/seconds claim, no Stage5C, and no SMC occurred.",
        "",
        "## Gate",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hk_gate"]
    return [
        "# Stage42-HK Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{key}` | {bool(value)} |" for key, value in gate["gates"].items()],
    ]


def _render_user_action(payload: Mapping[str, Any]) -> list[str]:
    return [
        "# User Action Required: Stage42-HK ETH_UCY Source Support",
        "",
        "- ETH_UCY source support is technically repairable after terms using local ETH-Person XML candidates plus ETH_seq_eth.",
        "- Confirm official/source terms, local path identity, and allowed research use for ETH/BIWI / ETH-Person local sources.",
        "- Rerun `.venv-pytorch/bin/python run_stage42_source_terms_confirmation_validator.py` with explicit confirmation fields.",
        "- Then run guarded restricted metric/time conversion, no-leakage, source-CV, and final test.",
        "",
        "Do not claim metric/seconds-level or official converted/evaluated ETH_UCY results from this preflight alone.",
    ]


def _refresh_lines(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_hk_gate"]
    s = payload["summary"]
    return [
        "## Stage42-HK ETH_UCY Restricted Metric/Time Source-Support Preflight",
        "",
        "- source: `fresh_stage42_hk_restricted_metric_time_eth_ucy_source_support_preflight`",
        f"- verdict: `{gate['verdict']}`",
        f"- gates: `{gate['passed']} / {gate['total']}`",
        f"- augmented ETH_UCY independent sources after terms: `{s['augmented_eth_ucy_independent_keys']}`.",
        f"- augmented ETH_UCY t50/t100 windows after terms: `{s['augmented_eth_ucy_t50_windows_after_terms']}` / `{s['augmented_eth_ucy_t100_windows_after_terms']}`.",
        f"- cached BL technical t100 safe-positive: `{s['cached_bl_technical_t100_safe_positive']}`; ready now: `{s['restricted_metric_time_ready_now']}`.",
        "- conclusion: ETH_UCY source-CV blocker is technically repairable after terms using ETH-Person XML candidates, but conversion/evaluation and metric/seconds claims remain blocked until user-confirmed terms and guarded rerun.",
    ]


def _refresh_a_journal_gap(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    lines[0] = "## Stage42-HK ETH_UCY Source-Support Preflight Refresh"
    _replace_section(A_JOURNAL_GAP, "STAGE42_HK_ETH_UCY_SOURCE_SUPPORT_PREFLIGHT", lines)


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, USER_SUMMARY, WORK_SUMMARY, GOAL_SUMMARY, CURRENT_SUMMARY]:
        _replace_section(path, "STAGE42_HK_ETH_UCY_SOURCE_SUPPORT_PREFLIGHT", lines)
    _refresh_a_journal_gap(payload)


def _refresh_research_state(payload: Mapping[str, Any], *, verification: Mapping[str, Any] | None = None) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-HK ETH_UCY restricted metric/time source-support preflight"
    state["current_verdict"] = payload["stage42_hk_gate"]["verdict"]
    state["stage42_hk_eth_ucy_restricted_metric_time_source_support_preflight"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "user_action_required": str(USER_ACTION_MD),
        "verdict": payload["stage42_hk_gate"]["verdict"],
        "gates": f"{payload['stage42_hk_gate']['passed']}/{payload['stage42_hk_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
        "verification": dict(verification or {"status": "pending"}),
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_restricted_metric_time_eth_ucy_source_support_preflight(
    *,
    refresh_readmes: bool = True,
    verification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    write_md(USER_ACTION_MD, _render_user_action(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload, verification=verification)
    return payload


if __name__ == "__main__":
    run_stage42_restricted_metric_time_eth_ucy_source_support_preflight()
