from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")

FU_JSON = OUT_DIR / "module_contribution_ledger_stage42.json"
FV_JSON = OUT_DIR / "claim_boundary_linter_stage42.json"
FW_JSON = OUT_DIR / "source_action_consolidator_stage42.json"
DM_JSON = OUT_DIR / "reviewer_replay_package_stage42.json"
FX_JSON = OUT_DIR / "objective_coverage_audit_stage42.json"
FY_JSON = OUT_DIR / "horizon_retry_decision_map_stage42.json"

REPORT_JSON = OUT_DIR / "paper_package_fxfy_refresh_stage42.json"
REPORT_MD = OUT_DIR / "paper_package_fxfy_refresh_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_fz_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
ONE_FILE_SUMMARY = Path("README_M3W_ONE_FILE_DETAILED_SUMMARY_ZH.md")
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

SOURCE = "fresh_stage42_paper_package_fxfy_refresh"
MARKER = "STAGE42_FZ_FXFY_PAPER_PACKAGE_REFRESH"

CLAIM_BOUNDARY = {
    "true_3d": False,
    "foundation_world_model": False,
    "global_metric_claim_allowed": False,
    "global_seconds_claim_allowed": False,
    "download_executed": False,
    "conversion_executed": False,
    "training_executed": False,
    "threshold_tuned_on_test": False,
    "stage5c_executed": False,
    "smc_enabled": False,
}

OVERCLAIM_CHECKS = {
    "true_3d": ["is a true 3D world model", "is now a true 3D world model", "已成为 true 3D"],
    "foundation": ["is a foundation world model", "foundation-track success is achieved", "已成为 foundation world model"],
    "metric_seconds": [
        "metric predictor is supported",
        "seconds-level horizon is supported",
        "global metric prediction is achieved",
        "秒级 t+100 已证明",
        "米级预测已证明",
    ],
    "stage5c": ["Stage5C executed: true", "Stage5C has been executed", "Stage5C 已执行: true"],
    "smc": ["SMC enabled: true", "SMC has been enabled", "SMC 已启用: true"],
    "jepa_main": ["JEPA is the main contribution", "JEPA 主贡献已证明"],
    "transformer_main": ["Transformer is the main contribution", "Transformer 主贡献已证明"],
}


def _passed_gate(payload: Mapping[str, Any], key: str) -> bool:
    gate = payload.get(key, {})
    return bool(gate) and gate.get("passed") == gate.get("total") and int(gate.get("total", 0) or 0) > 0


def _load_inputs() -> dict[str, Any]:
    return {
        "module_ledger": read_json(FU_JSON, {}),
        "claim_linter": read_json(FV_JSON, {}),
        "source_action": read_json(FW_JSON, {}),
        "reviewer_replay": read_json(DM_JSON, {}),
        "objective_coverage": read_json(FX_JSON, {}),
        "horizon_retry": read_json(FY_JSON, {}),
    }


def _compact_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "module_ledger": {
            "gate": inputs.get("module_ledger", {}).get("stage42_fu_gate", {}),
            "summary": inputs.get("module_ledger", {}).get("summary", {}),
        },
        "claim_linter": {
            "gate": inputs.get("claim_linter", {}).get("stage42_fv_gate", {}),
            "summary": inputs.get("claim_linter", {}).get("summary", {}),
        },
        "source_action": {
            "gate": inputs.get("source_action", {}).get("stage42_fw_gate", {}),
            "summary": inputs.get("source_action", {}).get("summary", {}),
        },
        "reviewer_replay": {
            "gate": inputs.get("reviewer_replay", {}).get("stage42_dm_gate", {}),
        },
        "objective_coverage": {
            "gate": inputs.get("objective_coverage", {}).get("stage42_fx_gate", {}),
            "summary": inputs.get("objective_coverage", {}).get("summary", {}),
        },
        "horizon_retry": {
            "gate": inputs.get("horizon_retry", {}).get("stage42_fy_gate", {}),
            "summary": inputs.get("horizon_retry", {}).get("summary", {}),
            "decisions": inputs.get("horizon_retry", {}).get("horizon_retry_decisions", []),
        },
    }


def _gate_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "module_ledger": {"stage42_fu_gate": inputs.get("module_ledger", {}).get("stage42_fu_gate", {})},
        "claim_linter": {"stage42_fv_gate": inputs.get("claim_linter", {}).get("stage42_fv_gate", {})},
        "source_action": {"stage42_fw_gate": inputs.get("source_action", {}).get("stage42_fw_gate", {})},
        "reviewer_replay": {"stage42_dm_gate": inputs.get("reviewer_replay", {}).get("stage42_dm_gate", {})},
        "objective_coverage": {"stage42_fx_gate": inputs.get("objective_coverage", {}).get("stage42_fx_gate", {})},
        "horizon_retry": {"stage42_fy_gate": inputs.get("horizon_retry", {}).get("stage42_fy_gate", {})},
    }


def _fxfy_summary(inputs: Mapping[str, Any]) -> dict[str, Any]:
    fu = inputs.get("module_ledger", {})
    fw = inputs.get("source_action", {})
    fx = inputs.get("objective_coverage", {})
    fy = inputs.get("horizon_retry", {})
    return {
        "supported_core_claims": fu.get("summary", {}).get("paper_claim_core", []),
        "blocked_main_claims": fu.get("summary", {}).get("paper_claim_blocked", []),
        "objective_status_counts": fx.get("summary", {}).get("status_counts", {}),
        "blocked_objectives": fx.get("summary", {}).get("blocked_objectives", []),
        "partial_objectives": fx.get("summary", {}).get("partial_objectives", []),
        "passed_objectives": fx.get("summary", {}).get("passed_objectives", []),
        "goal_complete": fx.get("summary", {}).get("goal_complete", False),
        "highest_priority_next_action": fx.get("summary", {}).get("highest_priority_next_action")
        or fw.get("summary", {}).get("highest_priority_blocker"),
        "weak_horizons": fy.get("summary", {}).get("weak_horizons", []),
        "stop_repeat_modeling_now": fy.get("summary", {}).get("stop_repeat_modeling_now", False),
        "uniform_horizon_claim_allowed": fy.get("summary", {}).get("uniform_horizon_claim_allowed", True),
        "conversion_ready_now": fw.get("summary", {}).get("conversion_ready_now"),
        "claim_ready_after_this_stage": fw.get("summary", {}).get("claim_ready_after_this_stage"),
    }


def _paper_section_lines(summary: Mapping[str, Any]) -> list[str]:
    return [
        "## Stage42-FZ FX/FY Evidence Package Refresh",
        "",
        "- source: `fresh_stage42_paper_package_fxfy_refresh`",
        "- result type: paper-package refresh over verified Stage42-FX/FY/FU/FV/FW/DM evidence; no training, no download, no conversion, no threshold tuning.",
        "- current model status: protected dataset-local/raw-frame 2.5D multi-agent world-state candidate.",
        "- not claimed: true 3D, foundation world model, global metric predictor, seconds-level horizon, ungated neural dynamics, Stage5C execution, SMC readiness.",
        f"- supported current core claims: `{summary.get('supported_core_claims')}`.",
        f"- blocked main claims: `{summary.get('blocked_main_claims')}`.",
        f"- objective coverage: blocked `{summary.get('blocked_objectives')}`, partial `{summary.get('partial_objectives')}`, passed `{summary.get('passed_objectives')}`, goal_complete `{summary.get('goal_complete')}`.",
        f"- source/action status: conversion_ready_now `{summary.get('conversion_ready_now')}`, claim_ready_after_this_stage `{summary.get('claim_ready_after_this_stage')}`, highest_priority_next_action `{summary.get('highest_priority_next_action')}`.",
        f"- horizon retry decision: weak_horizons `{summary.get('weak_horizons')}`, stop_repeat_modeling_now `{summary.get('stop_repeat_modeling_now')}`, uniform_horizon_claim_allowed `{summary.get('uniform_horizon_claim_allowed')}`.",
        "- paper implication: keep protected 2.5D safe-switch/group-consistency claim; do not repeat same-feature h100 retries until legal/source/guarded-conversion support exists.",
    ]


def _refresh_paper_files(summary: Mapping[str, Any]) -> None:
    section = _paper_section_lines(summary)
    per_file_extra = {
        "paper_outline_stage42.md": [
            "",
            "### Updated Outline Note",
            "",
            "The paper outline should foreground protected safe-switch/group-consistency evidence and explicitly separate supported modules from blocked auxiliary modules.",
        ],
        "method_draft_stage42.md": [
            "",
            "### Updated Method Boundary",
            "",
            "Method claims must describe Stage37/teacher-floor protection, domain expert routing, history features, and group-consistency full-waypoint evidence as protected deployment mechanisms.",
        ],
        "experiment_tables_stage42.md": [
            "",
            "### Updated Experiment Boundary",
            "",
            "Experiment tables should include FX requirement coverage and FY weak-horizon decision rows; h100 weak slices remain blocked rather than silently removed.",
        ],
        "ablation_tables_stage42.md": [
            "",
            "### Updated Ablation Boundary",
            "",
            "Ablation claims should follow Stage42-FU: history/domain expert/safe switch/teacher floor/group-consistency full-waypoint are supported; JEPA/Transformer/scene-goal/neighbor-interaction are not independent main claims.",
        ],
        "failure_taxonomy_stage42.md": [
            "",
            "### Updated Failure Taxonomy",
            "",
            "Remaining weak horizons are attributed to low-margin ambiguity plus source/legal/support blockers, especially TrajNet|100 and UCY|100; repeating same-feature model retries is not justified.",
        ],
        "model_card_stage42.md": [
            "",
            "### Updated Model-Card Boundary",
            "",
            "Deployable use remains protected/fallback-safe. Ungated neural dynamics, Stage5C, and SMC are not enabled.",
        ],
        "data_card_stage42.md": [
            "",
            "### Updated Data-Card Boundary",
            "",
            "Data claims remain dataset-local/raw-frame. Legal/source actions are required before new h100 conversion or uniform horizon claims.",
        ],
        "reproducibility_stage42.md": [
            "",
            "### Updated Reproducibility Note",
            "",
            "FZ is reproducible with `.venv-pytorch/bin/python run_stage42_paper_package_fxfy_refresh.py` and does not require raw data or training.",
        ],
        "a_journal_gap_stage42.md": [
            "",
            "### Updated A-Journal Gap",
            "",
            "The package is paper-candidate evidence for protected 2.5D world-state modeling, but A-journal/foundation-level claims still need legal source closure, broader domain conversion, metric/time calibration, and robust h100 support.",
        ],
    }
    for path in PAPER_FILES:
        _replace_section(path, MARKER, [*section, *per_file_extra.get(path.name, [])])


def _paper_status() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in PAPER_FILES:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        rows.append(
            {
                "file": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "contains_fz_marker": f"{MARKER}:START" in text and f"{MARKER}:END" in text,
            }
        )
    return rows


def _scan_overclaims(paths: list[Path]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        lower_text = text.lower()
        for check, phrases in OVERCLAIM_CHECKS.items():
            for phrase in phrases:
                if phrase.lower() in lower_text:
                    violations.append({"file": str(path), "check": check, "phrase": phrase})
    return violations


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    inputs = payload.get("inputs", {})
    summary = payload.get("summary", {})
    claim = payload.get("claim_boundary", {})
    gates = {
        "source_fresh": payload.get("source") == SOURCE,
        "fu_gate_passed": _passed_gate(inputs.get("module_ledger", {}), "stage42_fu_gate"),
        "fv_gate_passed": _passed_gate(inputs.get("claim_linter", {}), "stage42_fv_gate"),
        "fw_gate_passed": _passed_gate(inputs.get("source_action", {}), "stage42_fw_gate"),
        "dm_gate_passed": _passed_gate(inputs.get("reviewer_replay", {}), "stage42_dm_gate"),
        "fx_gate_passed": _passed_gate(inputs.get("objective_coverage", {}), "stage42_fx_gate"),
        "fy_gate_passed": _passed_gate(inputs.get("horizon_retry", {}), "stage42_fy_gate"),
        "all_paper_files_refreshed": all(row.get("contains_fz_marker") for row in payload.get("paper_file_status", [])),
        "goal_not_marked_complete": summary.get("goal_complete") is False,
        "data_blocker_preserved": "A" in summary.get("blocked_objectives", []),
        "weak_horizon_blocker_preserved": set(summary.get("weak_horizons", [])) >= {"TrajNet|100", "UCY|100"},
        "stop_repeat_retry_recorded": summary.get("stop_repeat_modeling_now") is True,
        "uniform_horizon_not_claimed": summary.get("uniform_horizon_claim_allowed") is False,
        "no_overclaim_violations": len(payload.get("overclaim_violations", [])) == 0,
        "no_download_conversion_training_eval": claim.get("download_executed") is False
        and claim.get("conversion_executed") is False
        and claim.get("training_executed") is False,
        "no_test_threshold_tuning": claim.get("threshold_tuned_on_test") is False,
        "no_metric_seconds_overclaim": claim.get("global_metric_claim_allowed") is False
        and claim.get("global_seconds_claim_allowed") is False,
        "no_true3d_foundation_overclaim": claim.get("true_3d") is False and claim.get("foundation_world_model") is False,
        "stage5c_false": claim.get("stage5c_executed") is False,
        "smc_false": claim.get("smc_enabled") is False,
    }
    passed = sum(bool(v) for v in gates.values())
    total = len(gates)
    return {
        "passed": passed,
        "total": total,
        "gates": gates,
        "verdict": "stage42_fz_paper_package_fxfy_refresh_pass" if passed == total else "stage42_fz_paper_package_fxfy_refresh_partial",
    }


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["summary"]
    lines = [
        "# Stage42-FZ Paper Package FX/FY Refresh",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_fz_gate']['passed']} / {payload['stage42_fz_gate']['total']}`",
        f"- verdict: `{payload['stage42_fz_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        "- 当前不是 true 3D world model。",
        "- 当前不是 large-scale foundation world model。",
        "- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
        "- FZ 只刷新 paper package；不训练、不下载、不转换、不调 test threshold。",
        "- Stage5C latent generative 未执行；SMC 未启用。",
        "",
        "## Summary",
        "",
        f"- supported_core_claims: `{summary['supported_core_claims']}`",
        f"- blocked_main_claims: `{summary['blocked_main_claims']}`",
        f"- objective_status_counts: `{summary['objective_status_counts']}`",
        f"- blocked_objectives: `{summary['blocked_objectives']}`",
        f"- partial_objectives: `{summary['partial_objectives']}`",
        f"- passed_objectives: `{summary['passed_objectives']}`",
        f"- weak_horizons: `{summary['weak_horizons']}`",
        f"- stop_repeat_modeling_now: `{summary['stop_repeat_modeling_now']}`",
        f"- uniform_horizon_claim_allowed: `{summary['uniform_horizon_claim_allowed']}`",
        f"- highest_priority_next_action: `{summary['highest_priority_next_action']}`",
        "",
        "## Paper Files",
        "",
        "| file | exists | contains FZ marker | size bytes |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in payload["paper_file_status"]:
        lines.append(f"| `{row['file']}` | {row['exists']} | {row['contains_fz_marker']} | {row['size_bytes']} |")
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
        ]
    )
    for gate, ok in payload["stage42_fz_gate"]["gates"].items():
        lines.append(f"| `{gate}` | {bool(ok)} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- FX/FY are now explicitly represented in the paper package.",
            "- The long goal remains active and incomplete; objective A remains blocked by source/legal conversion support.",
            "- Uniform horizon robustness remains blocked by TrajNet|100 and UCY|100; repeating the same-feature model retry is explicitly discouraged.",
        ]
    )
    return lines


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-FZ Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | {bool(ok)} |")
    write_md(GATE_MD, lines)


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "## Stage42-FZ Paper Package FX/FY Refresh",
        "",
        f"- source: `{payload['source']}`",
        f"- gate: `{payload['stage42_fz_gate']['passed']} / {payload['stage42_fz_gate']['total']}`; verdict `{payload['stage42_fz_gate']['verdict']}`.",
        "- role: paper-package refresh over Stage42-FX objective coverage and Stage42-FY horizon retry decision map; no training, no download, no conversion, no test-threshold tuning.",
        f"- supported core claims: `{summary['supported_core_claims']}`.",
        f"- blocked main claims: `{summary['blocked_main_claims']}`.",
        f"- objective status: blocked `{summary['blocked_objectives']}`, partial `{summary['partial_objectives']}`, passed `{summary['passed_objectives']}`, goal_complete `{summary['goal_complete']}`.",
        f"- weak horizons: `{summary['weak_horizons']}`; stop_repeat_modeling_now `{summary['stop_repeat_modeling_now']}`; uniform_horizon_claim_allowed `{summary['uniform_horizon_claim_allowed']}`.",
        f"- highest-priority next action: `{summary['highest_priority_next_action']}`.",
        "- boundary: protected dataset-local/raw-frame 2.5D only; no true 3D/foundation/global metric/seconds/Stage5C/SMC claim.",
    ]
    for path in [README_RESULTS, M3W_README, ONE_FILE_SUMMARY]:
        _replace_section(path, "STAGE42_FZ_PAPER_PACKAGE_FXFY_REFRESH", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "Stage42-FZ paper package FX/FY refresh"
    state["current_verdict"] = payload["stage42_fz_gate"]["verdict"]
    state["stage42_fz_paper_package_fxfy_refresh"] = {
        "source": payload["source"],
        "path": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "updated_at": payload["generated_at_utc"],
        "gate": payload["stage42_fz_gate"],
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
        "goal_complete": False,
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_paper_package_fxfy_refresh() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    inputs = _load_inputs()
    summary = _fxfy_summary(inputs)
    _refresh_paper_files(summary)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([FU_JSON, FV_JSON, FW_JSON, DM_JSON, FX_JSON, FY_JSON]),
        "inputs": _gate_inputs(inputs),
        "input_summary": _compact_inputs(inputs),
        "summary": summary,
        "paper_file_status": _paper_status(),
        "overclaim_violations": _scan_overclaims(PAPER_FILES + [README_RESULTS, M3W_README, ONE_FILE_SUMMARY]),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["stage42_fz_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    _write_gate(payload["stage42_fz_gate"])
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


__all__ = [
    "run_stage42_paper_package_fxfy_refresh",
    "_fxfy_summary",
    "_gate",
    "_scan_overclaims",
]
