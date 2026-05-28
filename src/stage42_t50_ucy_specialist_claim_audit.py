from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_t50_ensemble_source_robustness as s42ij
from src import stage42_t50_ensemble_ucy_specialist_integration as s42ik
from src import stage42_t50_gain_harm_ensemble_repair as s42ii
from src import stage42_unified_row_level_full_waypoint_cache as s42x
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
REPORT_JSON = OUT_DIR / "t50_ucy_specialist_claim_audit_stage42.json"
REPORT_MD = OUT_DIR / "t50_ucy_specialist_claim_audit_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_il_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
MASTER_README = Path("README_M3W_MASTER_SUMMARY_2026_05_27_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE42_IL_T50_UCY_SPECIALIST_CLAIM_AUDIT"
SOURCE = "fresh_stage42_il_t50_ucy_specialist_claim_audit"
EPS = 1e-8
UNCHANGED_TOL = 1e-6


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _domain_metric(row: Mapping[str, Any], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def _source_by_filename(source_rows: list[Mapping[str, Any]], needle: str) -> Mapping[str, Any]:
    matches = [row for row in source_rows if needle in str(row.get("source_file", ""))]
    if len(matches) != 1:
        raise ValueError(f"Expected one source row containing {needle!r}, got {len(matches)}.")
    return matches[0]


def _max_abs_non_ucy_delta(ii: Mapping[str, Any], ik: Mapping[str, Any]) -> dict[str, Any]:
    ii_domains = (ii.get("bootstrap", {}) or {}).get("by_domain", {}) or {}
    ik_domains = ik.get("by_domain", {}) or {}
    rows: list[dict[str, Any]] = []
    max_abs = 0.0
    for domain in ["ETH_UCY", "TrajNet"]:
        ii_row = ii_domains.get(domain, {})
        ik_row = ik_domains.get(domain, {})
        for metric in ["all_improvement", "t50_improvement", "hard_failure_improvement", "easy_degradation", "switch_rate"]:
            delta = _domain_metric(ik_row, metric) - _domain_metric(ii_row, metric)
            max_abs = max(max_abs, abs(delta))
            rows.append({"domain": domain, "metric": metric, "delta": float(delta)})
    return {"max_abs_delta": float(max_abs), "rows": rows}


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ii = read_json(s42ii.REPORT_JSON, {})
    ij = read_json(s42ij.REPORT_JSON, {})
    ik = read_json(s42ik.REPORT_JSON, {})
    x = read_json(s42x.REPORT_JSON, {})
    if not ii or not ij or not ik or not x:
        raise FileNotFoundError("Stage42-II/IJ/IK/X reports are required for Stage42-IL.")

    ii_summary = ii.get("summary", {})
    ij_summary = ij.get("summary", {})
    ik_summary = ik.get("summary", {})
    non_ucy_delta = _max_abs_non_ucy_delta(ii, ik)
    ij_ucy = _source_by_filename(ij.get("source_rows", []), "crowds_zara03.txt")
    ik_ucy = _source_by_filename(ik.get("source_rows", []), "crowds_zara03.txt")
    global_delta = {
        "ade_all_delta_vs_stage42ii": float(ik_summary.get("ade_all", 0.0) - ii_summary.get("ade_all", 0.0)),
        "ade_t50_delta_vs_stage42ii": float(ik_summary.get("ade_t50", 0.0) - ii_summary.get("ade_t50", 0.0)),
        "ade_t100_raw_delta_vs_stage42ii": float(ik_summary.get("ade_t100_raw_frame_diagnostic", 0.0) - ii_summary.get("ade_t100_raw_frame_diagnostic", 0.0)),
        "ade_hard_delta_vs_stage42ii": float(ik_summary.get("ade_hard_failure", 0.0) - ii_summary.get("ade_hard_failure", 0.0)),
        "easy_degradation_delta_vs_stage42ii": float(ik_summary.get("ade_easy_degradation", 0.0) - ii_summary.get("ade_easy_degradation", 0.0)),
        "switch_rate_delta_vs_stage42ii": float(ik_summary.get("switch_rate", 0.0) - ii_summary.get("switch_rate", 0.0)),
    }
    ucy_delta = {
        "before_rows": int(ij_ucy.get("rows", 0)),
        "after_rows": int(ik_ucy.get("rows", 0)),
        "before_t50": float(ij_ucy.get("t50_improvement", 0.0)),
        "after_t50": float(ik_ucy.get("t50_improvement", 0.0)),
        "delta_t50": float(ik_ucy.get("t50_improvement", 0.0) - ij_ucy.get("t50_improvement", 0.0)),
        "before_all": float(ij_ucy.get("all_improvement", 0.0)),
        "after_all": float(ik_ucy.get("all_improvement", 0.0)),
        "before_hard": float(ij_ucy.get("hard_failure_improvement", 0.0)),
        "after_hard": float(ik_ucy.get("hard_failure_improvement", 0.0)),
        "after_easy_degradation": float(ik_ucy.get("easy_degradation", 0.0)),
    }
    supported_claims = [
        {
            "claim": "Stage42-IK repairs the Stage42-II/IJ UCY fallback-only t50 weak source under a row-aligned source specialist.",
            "status": "supported_fresh_composition_eval",
            "evidence": f"UCY t50 {ucy_delta['before_t50']:.6f} -> {ucy_delta['after_t50']:.6f}; alignment rows {ik.get('alignment', {}).get('stage42x_ucy_rows')}",
        },
        {
            "claim": "The Stage42-II non-UCY ensemble decisions are unchanged by the IK composition.",
            "status": "supported_fresh_audit",
            "evidence": f"max_abs_non_ucy_domain_metric_delta={non_ucy_delta['max_abs_delta']:.12f}",
        },
        {
            "claim": "All powered t50 source files are nonnegative/positive after IK.",
            "status": "supported_fresh_audit",
            "evidence": f"positive_powered_t50_source_count={ik_summary.get('positive_powered_t50_source_count')}/{ik_summary.get('powered_t50_source_count')}",
        },
        {
            "claim": "IK improves the global Stage42-II ensemble while preserving easy cases.",
            "status": "supported_fresh_audit",
            "evidence": f"all_delta={global_delta['ade_all_delta_vs_stage42ii']:.6f}; t50_delta={global_delta['ade_t50_delta_vs_stage42ii']:.6f}; easy_degradation={ik_summary.get('ade_easy_degradation'):.6f}",
        },
    ]
    blocked_claims = [
        {
            "claim": "IK proves a new independent external-domain generalization result.",
            "status": "blocked",
            "reason": "IK is a source-specialist composition using cached-verified row-aligned UCY full-waypoint branch evidence.",
        },
        {
            "claim": "IK is new training.",
            "status": "blocked",
            "reason": "IK source labels mark new_training=not_run; it is a fresh composition/evaluation audit.",
        },
        {
            "claim": "IK allows metric or seconds-level claims.",
            "status": "blocked",
            "reason": "claim boundary remains dataset-local/raw-frame only.",
        },
        {
            "claim": "IK permits Stage5C or SMC.",
            "status": "blocked",
            "reason": "Stage5C and SMC remain false in all relevant artifacts.",
        },
    ]
    summary = {
        "stage42ii_verdict": (ii.get("stage42_ii_gate", {}) or {}).get("verdict"),
        "stage42ij_verdict": (ij.get("stage42_ij_gate", {}) or {}).get("verdict"),
        "stage42ik_verdict": (ik.get("stage42_ik_gate", {}) or {}).get("verdict"),
        "stage42x_verdict": (x.get("stage42_x_gate", {}) or {}).get("verdict"),
        "global_delta": global_delta,
        "ucy_delta": ucy_delta,
        "non_ucy_max_abs_delta": non_ucy_delta["max_abs_delta"],
        "all_powered_t50_sources_positive": int(ik_summary.get("positive_powered_t50_source_count", 0)) == int(ik_summary.get("powered_t50_source_count", -1)),
        "supported_claim_count": len(supported_claims),
        "blocked_claim_count": len(blocked_claims),
        "new_training": "not_run",
        "source_scope": "source_specialist_composition",
        "metric_seconds_claim_allowed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    payload: dict[str, Any] = {
        "stage": "Stage42-IL",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([s42ii.REPORT_JSON, s42ij.REPORT_JSON, s42ik.REPORT_JSON, s42x.REPORT_JSON]),
        "purpose": "Audit the Stage42-IK UCY source-specialist composition so the positive result is captured with clear delta evidence and strict claim boundaries.",
        "summary": summary,
        "non_ucy_delta_rows": non_ucy_delta["rows"],
        "supported_claims": supported_claims,
        "blocked_claims": blocked_claims,
        "source_labels": {
            "stage42ii": "cached_verified",
            "stage42ij": "cached_verified",
            "stage42ik": "fresh_run_composition_eval",
            "stage42x": "cached_verified_row_aligned_full_waypoint_cache",
            "stage42il": "fresh_run_claim_delta_audit",
            "new_training": "not_run",
        },
        "no_leakage": ik.get("no_leakage", {}),
        "claim_boundary": ik.get("claim_boundary", {}),
    }
    payload["stage42_il_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    claim = payload["claim_boundary"]
    no_leak = payload["no_leakage"]
    ucy = s["ucy_delta"]
    global_delta = s["global_delta"]
    gates = {
        "stage42ii_passed": s["stage42ii_verdict"] == "stage42_ii_ensemble_repair_stabilizes_t50",
        "stage42ij_passed": s["stage42ij_verdict"] == "stage42_ij_t50_ensemble_source_robustness_pass",
        "stage42ik_passed": s["stage42ik_verdict"] == "stage42_ik_ucy_specialist_integration_pass",
        "stage42x_passed": s["stage42x_verdict"] == "stage42_x_unified_row_level_full_waypoint_cache_pass",
        "ucy_was_fallback_before": abs(ucy["before_t50"]) <= EPS,
        "ucy_t50_repaired": ucy["after_t50"] > 0.0 and ucy["delta_t50"] > 0.0,
        "non_ucy_unchanged": s["non_ucy_max_abs_delta"] <= UNCHANGED_TOL,
        "global_delta_positive": global_delta["ade_all_delta_vs_stage42ii"] > 0.0 and global_delta["ade_t50_delta_vs_stage42ii"] > 0.0,
        "easy_not_worse": global_delta["easy_degradation_delta_vs_stage42ii"] <= EPS,
        "all_powered_sources_positive": s["all_powered_t50_sources_positive"] is True,
        "supported_and_blocked_claims_present": s["supported_claim_count"] >= 4 and s["blocked_claim_count"] >= 4,
        "no_future_or_test_leakage": no_leak.get("future_endpoint_input") is False
        and no_leak.get("future_waypoints_input") is False
        and no_leak.get("central_velocity") is False
        and no_leak.get("test_endpoint_goals") is False
        and no_leak.get("test_threshold_tuning") is False,
        "no_metric_seconds_overclaim": claim.get("metric_or_seconds_claim") is False,
        "scope_not_overclaimed": claim.get("source_specialist_claim_only") is True and claim.get("independent_new_domain_claim") is False,
        "stage5c_false": claim.get("stage5c_executed") is False,
        "smc_false": claim.get("smc_enabled") is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_il_ucy_specialist_claim_audit_pass" if passed == total else "stage42_il_ucy_specialist_claim_audit_partial"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _write_report(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_il_gate"]
    s = payload["summary"]
    gd = s["global_delta"]
    ud = s["ucy_delta"]
    lines = [
        "# Stage42-IL T50 UCY Specialist Claim Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Purpose",
        "",
        "Stage42-IK repaired the UCY fallback-only weak source in the Stage42-II/IJ t+50 ensemble. Stage42-IL turns that into a paper-safe evidence record: it measures the exact delta, verifies non-UCY rows remain unchanged, and states what claims are allowed or blocked.",
        "",
        "## Delta Versus Stage42-II",
        "",
        "| metric | delta |",
        "| --- | ---: |",
        f"| ADE all | {gd['ade_all_delta_vs_stage42ii']:.6f} |",
        f"| ADE t50 | {gd['ade_t50_delta_vs_stage42ii']:.6f} |",
        f"| ADE t100 raw diagnostic | {gd['ade_t100_raw_delta_vs_stage42ii']:.6f} |",
        f"| ADE hard/failure | {gd['ade_hard_delta_vs_stage42ii']:.6f} |",
        f"| easy degradation | {gd['easy_degradation_delta_vs_stage42ii']:.6f} |",
        f"| switch rate | {gd['switch_rate_delta_vs_stage42ii']:.6f} |",
        "",
        "## UCY Weak Source Repair",
        "",
        "| item | value |",
        "| --- | ---: |",
        f"| rows before / after | {ud['before_rows']} / {ud['after_rows']} |",
        f"| t50 before | {ud['before_t50']:.6f} |",
        f"| t50 after | {ud['after_t50']:.6f} |",
        f"| t50 delta | {ud['delta_t50']:.6f} |",
        f"| all after | {ud['after_all']:.6f} |",
        f"| hard after | {ud['after_hard']:.6f} |",
        f"| easy degradation after | {ud['after_easy_degradation']:.6f} |",
        "",
        f"- non_ucy_max_abs_delta: `{s['non_ucy_max_abs_delta']:.12f}`",
        "",
        "## Supported Claims",
        "",
        "| claim | status | evidence |",
        "| --- | --- | --- |",
    ]
    for row in payload["supported_claims"]:
        lines.append(f"| {row['claim']} | `{row['status']}` | {row['evidence']} |")
    lines.extend(["", "## Blocked Claims", "", "| claim | status | reason |", "| --- | --- | --- |"])
    for row in payload["blocked_claims"]:
        lines.append(f"| {row['claim']} | `{row['status']}` | {row['reason']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- IK can be used as source-specialist composition evidence for repairing the UCY t50 weak source.",
            "- IK should not be written as new independent-domain generalization, new training, metric/seconds calibration, Stage5C, or SMC.",
            "- This strengthens the external validation ledger by removing a fallback-only powered source while keeping the claim boundary narrow.",
        ]
    )
    write_md(REPORT_MD, lines)


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-IL Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | `{bool(ok)}` |")
    write_md(GATE_MD, lines)


def _refresh_readmes_and_state(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_il_gate"]
    s = payload["summary"]
    gd = s["global_delta"]
    ud = s["ucy_delta"]
    lines = [
        "## Stage42-IL T50 UCY Specialist Claim Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gates: `{gate['passed']} / {gate['total']}`",
        f"- Stage42-IK vs Stage42-II delta all/t50/hard: `{gd['ade_all_delta_vs_stage42ii']:.6f}` / `{gd['ade_t50_delta_vs_stage42ii']:.6f}` / `{gd['ade_hard_delta_vs_stage42ii']:.6f}`",
        f"- UCY t50 before/after: `{ud['before_t50']:.6f}` -> `{ud['after_t50']:.6f}`",
        f"- non-UCY max abs metric delta: `{s['non_ucy_max_abs_delta']:.12f}`",
        "- boundary: claim audit only; IK is source-specialist composition evidence, not independent-domain, metric/seconds, Stage5C, or SMC evidence.",
    ]
    for path in [README_RESULTS, M3W_README, MASTER_README]:
        _replace_section(path, SECTION, lines)
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage42_il_t50_ucy_specialist_claim_audit"
    state["current_verdict"] = gate["verdict"]
    stage42 = state.setdefault("stage42", {})
    stage42["stage_il_t50_ucy_specialist_claim_audit"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "verdict": gate["verdict"],
        "gates": f"{gate['passed']}/{gate['total']}",
        "summary": s,
        "claim_boundary": payload["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, GATE_MD]:
        sp = str(path)
        if sp not in reports:
            reports.append(sp)
    write_json(RESEARCH_STATE, _jsonable(state))


def run_stage42_t50_ucy_specialist_claim_audit() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, _jsonable(payload))
    _write_report(payload)
    _write_gate(payload["stage42_il_gate"])
    _refresh_readmes_and_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_t50_ucy_specialist_claim_audit()
    print(json.dumps(_jsonable(result["stage42_il_gate"]), ensure_ascii=False, indent=2))
