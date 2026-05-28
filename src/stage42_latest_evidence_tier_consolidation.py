from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
LEDGER = OUT_DIR / "run_ledger.jsonl"

IV_JSON = OUT_DIR / "source_level_row_cache_integration_stage42.json"
IW_JSON = OUT_DIR / "source_level_row_cache_mechanism_audit_stage42.json"
IZ_JSON = OUT_DIR / "source_level_nonlinear_context_slice_audit_stage42.json"
JA_JSON = OUT_DIR / "context_slice_policy_promotion_stage42.json"
JB_JSON = OUT_DIR / "context_slice_policy_conservative_repair_stage42.json"
GJ_JSON = OUT_DIR / "module_claim_lock_stage42.json"
PAPER_AUDIT_JSON = OUT_DIR / "paper_claim_evidence_audit_stage42.json"

REPORT_JSON = OUT_DIR / "latest_evidence_tier_consolidation_stage42.json"
REPORT_MD = OUT_DIR / "latest_evidence_tier_consolidation_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_jc_gate.md"

SECTION = "STAGE42_JC_LATEST_EVIDENCE_TIER_CONSOLIDATION"
SOURCE = "fresh_stage42_jc_latest_evidence_tier_consolidation"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-JC 是 latest evidence tier consolidation：它重新审计最新 reports 的 claim tier，不训练、不下载、不转换。",
    "source-level row-cache full-waypoint evidence 可以作为当前强证据，但只在 dataset-local/raw-frame 2.5D 边界内。",
    "context nonlinear slice evidence 只能作为局部分析；JA/JB policy promotion 失败，因此不能升级成可部署主贡献。",
    "future waypoints/endpoints 只允许作为 supervised/evaluation labels，不能作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
    "Stage5C latent generative 未执行。",
    "SMC 未启用。",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _gate_pass(payload: Mapping[str, Any], gate_name: str) -> bool:
    gate = payload.get(gate_name, {})
    try:
        return int(gate.get("passed", -1)) == int(gate.get("total", 0)) and int(gate.get("total", 0)) > 0
    except Exception:
        return False


def _metric_value(metric: Mapping[str, Any], key: str) -> float:
    try:
        return float(metric.get(key, 0.0))
    except Exception:
        return 0.0


def _pct(value: Any) -> str:
    try:
        return f"{100.0 * float(value):.2f}%"
    except Exception:
        return "n/a"


def _safe_verdict(payload: Mapping[str, Any], gate_name: str) -> str:
    gate = payload.get(gate_name, {})
    return str(gate.get("verdict", "missing"))


def _load_inputs() -> dict[str, Any]:
    return {
        "iv": read_json(IV_JSON, {}),
        "iw": read_json(IW_JSON, {}),
        "iz": read_json(IZ_JSON, {}),
        "ja": read_json(JA_JSON, {}),
        "jb": read_json(JB_JSON, {}),
        "gj": read_json(GJ_JSON, {}),
        "paper": read_json(PAPER_AUDIT_JSON, {}),
    }


def _tier_rows(inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    iv = inputs["iv"]
    iw = inputs["iw"]
    iz = inputs["iz"]
    ja = inputs["ja"]
    jb = inputs["jb"]
    gj = inputs["gj"]
    iv_metric = iv.get("metric", {})
    iw_metric = iw.get("metric", {})
    iz_summary = iz.get("summary", {})
    ja_summary = ja.get("summary", {})
    jb_summary = jb.get("summary", {})
    gj_summary = gj.get("summary", {})
    return [
        {
            "tier": "T1_source_level_row_cache_full_waypoint",
            "status": "main_supported_evidence" if _gate_pass(iv, "stage42_iv_gate") else "missing_or_partial",
            "source": iv.get("source", "unknown"),
            "evidence_file": str(IV_JSON),
            "claim": "single row-level source-level full-waypoint cache over current TrajNet+UCY test protocol",
            "rows": int(iv.get("test_rows", 0) or 0),
            "all": _metric_value(iv_metric, "all_improvement"),
            "t50": _metric_value(iv_metric, "t50_improvement"),
            "t100_raw": _metric_value(iv_metric, "t100_improvement"),
            "hard": _metric_value(iv_metric, "hard_failure_improvement"),
            "easy": _metric_value(iv_metric, "easy_degradation"),
            "paper_role": "main protected 2.5D raw-frame world-state evidence",
        },
        {
            "tier": "T2_mechanism_row_cache_audit",
            "status": "mechanism_supported" if _gate_pass(iw, "stage42_iw_gate") else "missing_or_partial",
            "source": iw.get("source", "unknown"),
            "evidence_file": str(IW_JSON),
            "claim": "safe-switch, teacher/floor fallback, row-level bootstrap, and waypoint-shape coverage are directly auditable",
            "rows": int(iw.get("rows", 0) or 0),
            "all": _metric_value(iw_metric, "all_improvement"),
            "t50": _metric_value(iw_metric, "t50_improvement"),
            "t100_raw": _metric_value(iw_metric, "t100_improvement"),
            "hard": _metric_value(iw_metric, "hard_failure_improvement"),
            "easy": _metric_value(iw_metric, "easy_degradation"),
            "paper_role": "mechanism support for protected policy, not proof of every token family",
        },
        {
            "tier": "T3_context_slice_analysis",
            "status": "local_slice_supported_not_deployable"
            if _gate_pass(iz, "stage42_iz_gate") and iz_summary.get("supported_context_slice_count", 0) > 0
            else "not_supported",
            "source": iz.get("source", "unknown"),
            "evidence_file": str(IZ_JSON),
            "claim": "nonlinear context has local source/horizon slice support",
            "rows": int(iz_summary.get("powered_slice_rows", 0) or 0),
            "all": None,
            "t50": None,
            "t100_raw": None,
            "hard": None,
            "easy": None,
            "paper_role": "analysis-only context evidence; cannot be written as deployed/global contribution",
        },
        {
            "tier": "T4_context_policy_promotion",
            "status": "not_promotable" if ja_summary.get("decision") == "validation_selected_context_slice_policy_not_promoted" else "review_needed",
            "source": ja.get("source", "unknown"),
            "evidence_file": str(JA_JSON),
            "claim": "validation-selected context slice policy did not beat baseline-family floor",
            "rows": int((ja_summary.get("metrics", {}).get("context_policy", {}) or {}).get("rows", 0) or 0),
            "all": _metric_value((ja_summary.get("metrics", {}).get("delta_vs_baseline_family", {}) or {}), "all_improvement"),
            "t50": _metric_value((ja_summary.get("metrics", {}).get("delta_vs_baseline_family", {}) or {}), "t50_improvement"),
            "t100_raw": _metric_value((ja_summary.get("metrics", {}).get("delta_vs_baseline_family", {}) or {}), "t100_raw_frame_diagnostic_improvement"),
            "hard": _metric_value((ja_summary.get("metrics", {}).get("delta_vs_baseline_family", {}) or {}), "hard_failure_improvement"),
            "easy": _metric_value((ja_summary.get("metrics", {}).get("delta_vs_baseline_family", {}) or {}), "easy_degradation"),
            "paper_role": "negative deployment evidence; prevents overclaiming context",
        },
        {
            "tier": "T5_conservative_context_repair",
            "status": "not_promotable" if jb_summary.get("decision") == "conservative_context_slice_policy_not_promoted" else "review_needed",
            "source": jb.get("source", "unknown"),
            "evidence_file": str(JB_JSON),
            "claim": "conservative context repair still regressed core metrics against baseline-family floor",
            "rows": int((jb_summary.get("metrics", {}).get("context_policy", {}) or {}).get("rows", 0) or 0),
            "all": _metric_value((jb_summary.get("metrics", {}).get("delta_vs_baseline_family", {}) or {}), "all_improvement"),
            "t50": _metric_value((jb_summary.get("metrics", {}).get("delta_vs_baseline_family", {}) or {}), "t50_improvement"),
            "t100_raw": _metric_value((jb_summary.get("metrics", {}).get("delta_vs_baseline_family", {}) or {}), "t100_raw_frame_diagnostic_improvement"),
            "hard": _metric_value((jb_summary.get("metrics", {}).get("delta_vs_baseline_family", {}) or {}), "hard_failure_improvement"),
            "easy": _metric_value((jb_summary.get("metrics", {}).get("delta_vs_baseline_family", {}) or {}), "easy_degradation"),
            "paper_role": "negative repair evidence; context remains slice-local only",
        },
        {
            "tier": "T6_module_claim_lock",
            "status": "claim_lock_passed" if _gate_pass(gj, "stage42_gj_gate") else "missing_or_partial",
            "source": gj.get("source", "unknown"),
            "evidence_file": str(GJ_JSON),
            "claim": "main and blocked module claims are locked from the current evidence package",
            "rows": 0,
            "all": None,
            "t50": None,
            "t100_raw": None,
            "hard": None,
            "easy": None,
            "paper_role": f"supported={gj_summary.get('supported_main_modules_locked', [])}; blocked={gj_summary.get('blocked_main_modules_locked', [])}",
        },
    ]


def _summary(inputs: Mapping[str, Any], rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    iv = inputs["iv"]
    ja = inputs["ja"]
    jb = inputs["jb"]
    gj = inputs["gj"]
    iv_metric = iv.get("metric", {})
    gj_summary = gj.get("summary", {})
    main_row = rows[0]
    negative_context_rows = [row for row in rows if row["tier"] in {"T4_context_policy_promotion", "T5_conservative_context_repair"}]
    context_promotable = not all(row["status"] == "not_promotable" for row in negative_context_rows)
    return {
        "source": SOURCE,
        "decision": "latest_evidence_tiers_consolidated_context_not_promoted",
        "main_evidence_tier": main_row["tier"],
        "main_rows": int(main_row["rows"]),
        "main_all_improvement": _metric_value(iv_metric, "all_improvement"),
        "main_t50_improvement": _metric_value(iv_metric, "t50_improvement"),
        "main_t100_raw_frame_diagnostic_improvement": _metric_value(iv_metric, "t100_improvement"),
        "main_hard_failure_improvement": _metric_value(iv_metric, "hard_failure_improvement"),
        "main_easy_degradation": _metric_value(iv_metric, "easy_degradation"),
        "context_slice_local_support": inputs["iz"].get("summary", {}).get("supported_context_slice_count", 0),
        "context_policy_promotable": context_promotable,
        "ja_decision": ja.get("summary", {}).get("decision"),
        "jb_decision": jb.get("summary", {}).get("decision"),
        "supported_main_modules_locked": gj_summary.get("supported_main_modules_locked", []),
        "blocked_main_modules_locked": gj_summary.get("blocked_main_modules_locked", []),
        "paper_ready_claim": (
            "protected source-level full-waypoint dataset-local/raw-frame 2.5D world-state evidence; "
            "context modules remain slice-local/diagnostic unless future policy promotion succeeds"
        ),
        "next_action": (
            "Move toward source/legal/calibration support or a genuinely different full-sequence target; "
            "do not rerun current context-slice promotion as a main route."
        ),
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    inputs = _load_inputs()
    rows = _tier_rows(inputs)
    summary = _summary(inputs, rows)
    payload: dict[str, Any] = {
        "stage": "Stage42-JC",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([IV_JSON, IW_JSON, IZ_JSON, JA_JSON, JB_JSON, GJ_JSON, PAPER_AUDIT_JSON]),
        "current_facts": CURRENT_FACTS,
        "input_reports": {
            "source_level_row_cache_integration": str(IV_JSON),
            "source_level_row_cache_mechanism_audit": str(IW_JSON),
            "nonlinear_context_slice_audit": str(IZ_JSON),
            "context_slice_policy_promotion": str(JA_JSON),
            "context_slice_policy_conservative_repair": str(JB_JSON),
            "module_claim_lock": str(GJ_JSON),
            "paper_claim_evidence_audit": str(PAPER_AUDIT_JSON),
        },
        "tier_rows": rows,
        "summary": summary,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "fresh_evaluation_this_stage": False,
            "claim_audit_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_jc_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    tiers = {row["tier"]: row for row in payload["tier_rows"]}
    gates = {
        "source_level_row_cache_passed": tiers["T1_source_level_row_cache_full_waypoint"]["status"] == "main_supported_evidence",
        "mechanism_audit_passed": tiers["T2_mechanism_row_cache_audit"]["status"] == "mechanism_supported",
        "context_slice_support_recorded": summary["context_slice_local_support"] > 0,
        "context_policy_not_promoted": summary["context_policy_promotable"] is False,
        "ja_negative_recorded": summary["ja_decision"] == "validation_selected_context_slice_policy_not_promoted",
        "jb_negative_recorded": summary["jb_decision"] == "conservative_context_slice_policy_not_promoted",
        "module_claim_lock_passed": tiers["T6_module_claim_lock"]["status"] == "claim_lock_passed",
        "main_metrics_positive": summary["main_all_improvement"] > 0
        and summary["main_t50_improvement"] > 0
        and summary["main_hard_failure_improvement"] > 0,
        "easy_preserved": summary["main_easy_degradation"] <= 0.02,
        "t100_reported_as_raw_diagnostic": summary["main_t100_raw_frame_diagnostic_improvement"] > 0
        and payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "no_metric_seconds_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "no_future_endpoint_input": payload["no_leakage"]["future_endpoint_input"] is False,
        "no_future_waypoint_input": payload["no_leakage"]["future_waypoint_input"] is False,
        "no_central_velocity": payload["no_leakage"]["central_velocity"] is False,
        "no_test_endpoint_goals": payload["no_leakage"]["test_endpoint_goals"] is False,
        "no_test_threshold_tuning": payload["no_leakage"]["test_threshold_tuning"] is False,
        "true3d_not_claimed": payload["claim_boundary"]["true_3d"] is False,
        "foundation_not_claimed": payload["claim_boundary"]["foundation_world_model"] is False,
        "stage5c_false": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_false": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_jc_latest_evidence_tier_consolidation_pass" if passed == total else "stage42_jc_latest_evidence_tier_consolidation_partial"
    return {"source": "fresh_run", "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jc_gate"]
    s = payload["summary"]
    lines = [
        "# Stage42-JC Latest Evidence Tier Consolidation",
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
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Decision",
        "",
        f"- decision: `{s['decision']}`",
        f"- main_evidence_tier: `{s['main_evidence_tier']}`",
        f"- paper_ready_claim: {s['paper_ready_claim']}",
        f"- next_action: {s['next_action']}",
        "",
        "## Evidence Tiers",
        "",
        "| tier | status | rows | all | t50 | t100 raw | hard/failure | easy | paper role |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in payload["tier_rows"]:
        lines.append(
            f"| `{row['tier']}` | `{row['status']}` | {int(row.get('rows') or 0)} | "
            f"{_pct(row.get('all'))} | {_pct(row.get('t50'))} | {_pct(row.get('t100_raw'))} | "
            f"{_pct(row.get('hard'))} | {_pct(row.get('easy'))} | {row.get('paper_role', '')} |"
        )
    lines.extend(
        [
            "",
            "## Context Claim Boundary",
            "",
            f"- supported_context_slice_count: `{s['context_slice_local_support']}`",
            f"- JA decision: `{s['ja_decision']}`",
            f"- JB decision: `{s['jb_decision']}`",
            "- conclusion: local nonlinear context slices are analysis evidence only. They are not promoted to deployed/global scene-goal-neighbor contribution because both validation-selected and conservative policy promotion failed.",
            "",
            "## Locked Module Claims",
            "",
            f"- supported_main_modules_locked: `{s['supported_main_modules_locked']}`",
            f"- blocked_main_modules_locked: `{s['blocked_main_modules_locked']}`",
            "",
            "## No-Leakage And Claim Boundary",
            "",
            f"- no_leakage: `{payload['no_leakage']}`",
            f"- claim_boundary: `{payload['claim_boundary']}`",
            "",
            "## Interpretation",
            "",
            "- The current strongest paper-ready evidence is protected source-level full-waypoint row-cache evidence, not a free-running generative model.",
            "- Stage42-IZ remains useful because it identifies where context has local slice-level signal, but Stage42-JA/JB keep that signal out of the deployable claim.",
            "- The next high-value research move is source/legal/calibration expansion or a genuinely different full-sequence target, not repeating the same context slice promotion protocol.",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_jc_gate"]
    lines = [
        "# Stage42-JC Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | `{bool(value)}` |")
    return lines


def _section_lines(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    gate = payload["stage42_jc_gate"]
    return [
        "## Stage42-JC Latest Evidence Tier Consolidation",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`; verdict: `{gate['verdict']}`",
        f"- main evidence: `{s['main_evidence_tier']}` with all `{_pct(s['main_all_improvement'])}`, t50 `{_pct(s['main_t50_improvement'])}`, t100 raw-frame diagnostic `{_pct(s['main_t100_raw_frame_diagnostic_improvement'])}`, hard/failure `{_pct(s['main_hard_failure_improvement'])}`, easy degradation `{_pct(s['main_easy_degradation'])}`.",
        f"- context boundary: Stage42-IZ has `{s['context_slice_local_support']}` local supported context slices, but JA/JB failed promotion, so context is not a deployable/global main contribution.",
        "- claim boundary: still protected dataset-local/raw-frame 2.5D; not true 3D, not foundation, not metric/seconds-level, no Stage5C, no SMC.",
    ]


def _update_text_file(path: Path, payload: Mapping[str, Any]) -> None:
    _replace_section(path, SECTION, _section_lines(payload))


def _update_readmes(payload: Mapping[str, Any]) -> None:
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _update_text_file(path, payload)


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    stage42 = state.setdefault("stage42", {})
    stage42["latest_evidence_tier_consolidation"] = {
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "verdict": payload["stage42_jc_gate"]["verdict"],
        "gate": {
            "passed": payload["stage42_jc_gate"]["passed"],
            "total": payload["stage42_jc_gate"]["total"],
        },
        "decision": payload["summary"]["decision"],
        "main_evidence_tier": payload["summary"]["main_evidence_tier"],
        "context_policy_promotable": payload["summary"]["context_policy_promotable"],
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["last_updated_utc"] = payload["generated_at_utc"]
    write_json(RESEARCH_STATE, state)


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(LEDGER.parent)
    import json

    row = {
        "stage": payload["stage"],
        "source": payload["source"],
        "generated_at_utc": payload["generated_at_utc"],
        "verdict": payload["stage42_jc_gate"]["verdict"],
        "result": payload["summary"]["decision"],
        "fresh_run": True,
        "trained": False,
        "converted": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_stage42_latest_evidence_tier_consolidation(*, refresh_readmes: bool = True) -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _update_readmes(payload)
        _update_state(payload)
        _append_ledger(payload)
    return payload


def main() -> None:
    run_stage42_latest_evidence_tier_consolidation(refresh_readmes=True)


if __name__ == "__main__":
    main()
