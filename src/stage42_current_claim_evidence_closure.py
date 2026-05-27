from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

MASTER_README = Path("README_M3W_MASTER_SUMMARY_2026_05_27_ZH.md")
README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
RESEARCH_STATE = Path("research_state.json")

PAPER_MATRIX_MD = OUT_DIR / "paper_ready_evidence_matrix_stage42.md"
MODULE_LOCK_JSON = OUT_DIR / "module_claim_lock_stage42.json"
CLAIM_LINTER_JSON = OUT_DIR / "claim_boundary_linter_stage42.json"
CONTEXT_CLOSURE_MD = OUT_DIR / "context_model_closure_stage42.md"
AJOURNAL_GAP_MD = OUT_DIR / "a_journal_gap_stage42.md"
T100_REPLAY_JSON = OUT_DIR / "t100_runtime_row_cache_replay_stage42.json"
SOURCE_TERMS_DRY_RUN_JSON = OUT_DIR / "source_terms_ia_bridged_validator_dry_run_stage42.json"

REPORT_JSON = OUT_DIR / "current_claim_evidence_closure_stage42.json"
REPORT_MD = OUT_DIR / "current_claim_evidence_closure_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ic_gate.md"

SECTION = "STAGE42_IC_CURRENT_CLAIM_EVIDENCE_CLOSURE"
SOURCE = "fresh_stage42_ic_current_claim_evidence_closure"

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-IC 是 claim/evidence closure，不重新训练、不下载、不转换、不评估。",
    "本阶段只把已有 fresh/cached_verified evidence 映射成当前可写/不可写 claim 闭环。",
    "future endpoints / waypoints 只作为 supervised/evaluation labels，不作为 inference input。",
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


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def _gate_pass(gate: Mapping[str, Any]) -> bool:
    try:
        return int(gate.get("passed", -1)) == int(gate.get("total", 0)) and int(gate.get("total", 0)) > 0
    except Exception:
        return False


def _text_has(path: Path, needles: list[str]) -> bool:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return all(needle in text for needle in needles)


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    module_lock = read_json(MODULE_LOCK_JSON, {})
    linter = read_json(CLAIM_LINTER_JSON, {})
    replay = read_json(T100_REPLAY_JSON, {})
    source_terms = read_json(SOURCE_TERMS_DRY_RUN_JSON, {})

    module_summary = module_lock.get("summary", {})
    replay_metric = ((replay.get("runtime_batch_replay", {}) or {}).get("metric", {}) or {})
    source_summary = source_terms.get("summary", {})
    linter_summary = linter.get("summary", linter)

    supported_claims = [
        {
            "claim": "protected dataset-local/raw-frame 2.5D multi-agent world-state candidate",
            "status": "supported",
            "evidence": "Stage42 paper matrix + module claim lock + M3W master summary",
            "paper_use": "main framing only with strict boundary",
        },
        {
            "claim": "Stage26 SDD cost-aware selector remains the SDD deployable baseline",
            "status": "supported_cached_verified",
            "evidence": "M3W master summary records t50 +14.58%, hard/failure +11.23%, easy degradation 1.81%",
            "paper_use": "baseline and historical development evidence",
        },
        {
            "claim": "Stage37 external t50 safe selector is deployable for dataset-local raw-frame external t50 transfer",
            "status": "supported_cached_verified",
            "evidence": "M3W master summary records all +13.48%, t50 +8.46%, t50 CI [+7.69%, +9.15%], hard/failure +15.54%, easy 0.041%",
            "paper_use": "external safety floor / comparison baseline",
        },
        {
            "claim": "M3W-Neural v1 is a protected neural world-state candidate, not ungated neural deployment",
            "status": "supported_cached_verified",
            "evidence": "M3W-Neural README and master summary record all +21.03%, t50 +13.65%, t100 raw +14.69%, hard/failure +20.38%, easy 0.00%",
            "paper_use": "protected neural candidate evidence",
        },
        {
            "claim": "Stage42 protected full-waypoint/group-consistency policies are current source-level world-state evidence",
            "status": "supported",
            "evidence": "module claim lock supports group_consistency_full_waypoint, full_waypoint_shape, endpoint_bridge",
            "paper_use": "main Stage42 world-state evidence",
        },
        {
            "claim": "Stage42-HV provides row-level batch replay for the t100 easy-guard runtime policy",
            "status": "supported_cached_verified",
            "evidence": f"rows={replay_metric.get('rows')}; all={_pct(replay_metric.get('all_improvement'))}; t50={_pct(replay_metric.get('t50_improvement'))}; t100raw={_pct(replay_metric.get('t100_raw_frame_diagnostic_improvement'))}; hard={_pct(replay_metric.get('hard_failure_improvement'))}; t100 easy={_pct(replay_metric.get('t100_easy_degradation'))}",
            "paper_use": "runtime/replay evidence, raw-frame diagnostic only",
        },
    ]

    blocked_claims = [
        {
            "claim": "true 3D or foundation world model",
            "status": "blocked",
            "reason": "claim boundary explicitly false in module lock, linter, and replay artifacts",
        },
        {
            "claim": "global metric or seconds-level performance",
            "status": "blocked",
            "reason": "restricted metric/time/source terms ready candidates remain zero; calibration/source confirmation incomplete",
        },
        {
            "claim": "Stage5C latent generative execution or SMC readiness",
            "status": "blocked",
            "reason": "all current artifacts keep Stage5C false and SMC false",
        },
        {
            "claim": "JEPA or Transformer as independent main contribution",
            "status": "blocked_or_diagnostic",
            "reason": f"module lock blocked modules: {module_summary.get('blocked_main_modules_locked', [])}",
        },
        {
            "claim": "scene/goal or neighbor/interaction as independent main contribution",
            "status": "blocked_or_diagnostic",
            "reason": "Stage42-CJ/CK and context closure select baseline-family control / close current residual context protocol",
        },
        {
            "claim": "new external converted/evaluated metric-time data from HZ/IA/IB",
            "status": "blocked",
            "reason": f"IB conversion_ready_targets={source_summary.get('conversion_ready_targets')}; converted={source_summary.get('converted_datasets_now')}; evaluated={source_summary.get('evaluated_datasets_now')}",
        },
        {
            "claim": "t100 seconds-level long-horizon prediction",
            "status": "blocked",
            "reason": "Stage42-HV t100 is exact row-level raw-frame replay, not verified seconds-level calibration",
        },
    ]

    evidence_files = [
        MASTER_README,
        PAPER_MATRIX_MD,
        MODULE_LOCK_JSON,
        CLAIM_LINTER_JSON,
        CONTEXT_CLOSURE_MD,
        AJOURNAL_GAP_MD,
        T100_REPLAY_JSON,
        SOURCE_TERMS_DRY_RUN_JSON,
    ]
    payload = {
        "stage": "Stage42-IC",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash(evidence_files),
        "current_facts": CURRENT_FACTS,
        "evidence_files": [str(path) for path in evidence_files],
        "supported_claims": supported_claims,
        "blocked_claims": blocked_claims,
        "summary": {
            "supported_claim_count": len(supported_claims),
            "blocked_claim_count": len(blocked_claims),
            "module_lock_verdict": (module_lock.get("stage42_gj_gate", {}) or {}).get("verdict"),
            "module_lock_gate_passed": _gate_pass(module_lock.get("stage42_gj_gate", {}) or {}),
            "claim_linter_violations": int(linter_summary.get("violations_total", 0) or 0),
            "t100_row_replay_rows": int(replay_metric.get("rows", 0) or 0),
            "t100_row_replay_gate_passed": _gate_pass(replay.get("stage42_hv_gate", {}) or {}),
            "source_terms_conversion_ready": int(source_summary.get("conversion_ready_targets", 0) or 0),
            "source_terms_converted_now": int(source_summary.get("converted_datasets_now", 0) or 0),
            "source_terms_evaluated_now": int(source_summary.get("evaluated_datasets_now", 0) or 0),
            "metric_seconds_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "raw_frame_dataset_local_only": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "actions": {
            "downloaded": False,
            "converted": False,
            "trained": False,
            "evaluated": False,
            "summary_closure_only": True,
        },
        "next_actions": [
            "Use this closure as the paper/package claim map for the next Stage42 manuscript refresh.",
            "Do not run guarded conversion until user-confirmed source terms/local path/source identity exist.",
            "If modeling continues without new legal sources, prioritize gain/harm or switchability targets rather than repeating closed residual context protocols.",
        ],
    }
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    gates = {
        "master_summary_exists": MASTER_README.exists(),
        "paper_matrix_exists": PAPER_MATRIX_MD.exists(),
        "module_lock_passed": bool(s["module_lock_gate_passed"]),
        "claim_linter_clean": s["claim_linter_violations"] == 0,
        "context_closure_recorded": _text_has(CONTEXT_CLOSURE_MD, ["close_current_sequence_graph_residual_context_protocol"]),
        "a_journal_gap_recorded": _text_has(AJOURNAL_GAP_MD, ["not yet", "foundation"]) or AJOURNAL_GAP_MD.exists(),
        "supported_claims_present": s["supported_claim_count"] >= 5,
        "blocked_claims_present": s["blocked_claim_count"] >= 6,
        "t100_row_replay_gate_passed": bool(s["t100_row_replay_gate_passed"]),
        "t100_replay_large_enough": s["t100_row_replay_rows"] >= 40000,
        "source_terms_still_block_conversion": s["source_terms_conversion_ready"] == 0,
        "no_conversion_claim": s["source_terms_converted_now"] == 0,
        "no_evaluation_claim": s["source_terms_evaluated_now"] == 0,
        "no_metric_seconds_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_not_executed": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_not_enabled": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_ic_current_claim_evidence_closure_pass" if passed == total else "stage42_ic_current_claim_evidence_closure_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _claim_table(rows: list[Mapping[str, Any]], *, blocked: bool = False) -> list[str]:
    if blocked:
        lines = ["| claim | status | reason |", "| --- | --- | --- |"]
        for row in rows:
            lines.append(f"| {row['claim']} | `{row['status']}` | {row['reason']} |")
        return lines
    lines = ["| claim | status | evidence | paper use |", "| --- | --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['claim']} | `{row['status']}` | {row['evidence']} | {row['paper_use']} |")
    return lines


def _write_reports(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> None:
    payload_for_json = dict(payload)
    payload_for_json["stage42_ic_gate"] = gate
    write_json(REPORT_JSON, payload_for_json)
    write_md(
        REPORT_MD,
        [
            "# Stage42-IC Current Claim / Evidence Closure",
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
            "## Supported Claims",
            "",
            *_claim_table(list(payload["supported_claims"])),
            "",
            "## Blocked / Diagnostic Claims",
            "",
            *_claim_table(list(payload["blocked_claims"]), blocked=True),
            "",
            "## Summary",
            "",
            *[f"- {key}: `{value}`" for key, value in payload["summary"].items()],
            "",
            "## Next Actions",
            "",
            *[f"- {item}" for item in payload["next_actions"]],
        ],
    )
    write_md(
        GATE_MD,
        [
            "# Stage42-IC Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            "",
            "| gate | pass |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(ok)}` |" for name, ok in gate["gates"].items()],
        ],
    )


def _refresh_lines(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    return [
        "## Stage42-IC Current Claim / Evidence Closure",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`; gates `{gate['passed']} / {gate['total']}`.",
        f"- supported claims: `{s['supported_claim_count']}`; blocked/diagnostic claims: `{s['blocked_claim_count']}`.",
        f"- t100 row replay rows: `{s['t100_row_replay_rows']}`; source terms conversion-ready now: `{s['source_terms_conversion_ready']}`.",
        "- IC closes the current paper-package claim map: supported claims remain protected dataset-local/raw-frame 2.5D, while true-3D/foundation/metric-seconds/Stage5C/SMC and JEPA/Transformer independent-main claims remain blocked.",
        "- This is not new training, download, conversion, or evaluation; it is a claim/evidence closure over existing fresh/cached_verified artifacts.",
    ]


def _refresh_readmes_and_state(payload: Mapping[str, Any], gate: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload, gate)
    for path in [README_RESULTS, M3W_README, MASTER_README]:
        _replace_section(path, SECTION, lines)
    state = read_json(RESEARCH_STATE, {})
    state["stage42_ic_current_claim_evidence_closure"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "verdict": gate["verdict"],
        "gates": f"{gate['passed']}/{gate['total']}",
        "summary": payload["summary"],
        "summary_closure_only": True,
        "new_training_or_conversion": False,
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_current_claim_evidence_closure() -> dict[str, Any]:
    payload = _build_payload()
    gate = _gate(payload)
    _write_reports(payload, gate)
    _refresh_readmes_and_state(payload, gate)
    payload = dict(payload)
    payload["stage42_ic_gate"] = gate
    write_json(REPORT_JSON, payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_current_claim_evidence_closure()
    gate = result["stage42_ic_gate"]
    print(f"Stage42-IC current claim/evidence closure: {gate['verdict']} ({gate['passed']}/{gate['total']})")
