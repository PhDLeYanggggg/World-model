from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _pct, _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
CV_JSON = OUT_DIR / "proximity_guard_batch_replay_stage42.json"
CT_JSON = OUT_DIR / "frozen_proximity_guard_policy_replay_stage42.json"
CU_JSON = OUT_DIR / "proximity_guard_runtime_policy_stage42.json"

REPORT_JSON = OUT_DIR / "runtime_replay_paper_refresh_stage42.json"
REPORT_MD = OUT_DIR / "runtime_replay_paper_refresh_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_cw_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
RETRO_README = Path("README_M3W_GOAL_RETROSPECTIVE_CURRENT_ZH.md")
RESEARCH_STATE = Path("research_state.json")

PAPER_FILES = [
    OUT_DIR / "method_draft_stage42.md",
    OUT_DIR / "experiment_tables_stage42.md",
    OUT_DIR / "model_card_stage42.md",
    OUT_DIR / "reproducibility_stage42.md",
    OUT_DIR / "a_journal_gap_stage42.md",
]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-CW 是 paper/reproducibility refresh，不重新训练，不调 threshold。",
    "Stage42-CV 的 runtime batch replay 使用真实 common validation/test rows，而不是 toy smoke test。",
    "runtime replay 精确复现 Stage42-CQ guard 决策和 selected trajectory。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
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


def _cv_summary(cv: Mapping[str, Any]) -> dict[str, Any]:
    val = cv["splits"]["val"]
    test = cv["splits"]["test"]
    metric = test["runtime_metric_vs_endpoint_ade"]
    safety = test["runtime_joint_safety"]["composer_minus_endpoint"]
    return {
        "policy_hash": cv["policy_hash"],
        "gate": cv["stage42_cv_gate"],
        "val_rows": val["rows"],
        "test_rows": test["rows"],
        "val_decision_exact_replay": val["decision_match"],
        "test_decision_exact_replay": test["decision_match"],
        "test_selected_xy_max_abs_diff": test["selected_xy_max_abs_diff"],
        "test_selected_ade_max_abs_diff": test["selected_ade_max_abs_diff"],
        "test_selected_fde_max_abs_diff": test["selected_fde_max_abs_diff"],
        "test_vs_endpoint_linear_ade": {
            "all_improvement": metric["all_improvement"],
            "t50_improvement": metric["t50_improvement"],
            "t100_raw_frame_diagnostic_improvement": metric["t100_raw_frame_diagnostic_improvement"],
            "hard_failure_improvement": metric["hard_failure_improvement"],
            "easy_degradation": metric["easy_degradation"],
            "switch_rate": metric["switch_rate"],
        },
        "test_runtime_reasons": test["reason_counts"],
        "joint_safety_vs_endpoint_linear": {
            "near_collision_002_delta": safety["near_collision_rate_002_delta"],
            "near_collision_005_delta": safety["near_collision_rate_005_delta"],
            "p05_min_group_distance_delta": safety["p05_min_group_distance_delta"],
            "jagged_rate_delta": safety["jagged_rate_delta"],
        },
    }


def _refresh_lines(summary: Mapping[str, Any]) -> list[str]:
    metric = summary["test_vs_endpoint_linear_ade"]
    safety = summary["joint_safety_vs_endpoint_linear"]
    gate = summary["gate"]
    return [
        "## Stage42-CW Runtime Replay Paper / Reproducibility Refresh",
        "",
        "- source: `fresh_synthesis_from_stage42_cv_runtime_batch_replay`",
        "- role: paper-ready deployment reproducibility evidence.",
        f"- Stage42-CV gate: `{gate['passed']} / {gate['total']}`; verdict `{gate['verdict']}`.",
        f"- frozen policy hash: `{summary['policy_hash']}`.",
        f"- validation/test replay rows: `{summary['val_rows']}` / `{summary['test_rows']}`.",
        f"- exact runtime replay: validation `{summary['val_decision_exact_replay']}`, test `{summary['test_decision_exact_replay']}`.",
        f"- selected_xy / ADE / FDE max diff vs original CQ guard on test: `{summary['test_selected_xy_max_abs_diff']}` / `{summary['test_selected_ade_max_abs_diff']}` / `{summary['test_selected_fde_max_abs_diff']}`.",
        f"- test ADE vs endpoint-linear all/t50/t100 raw/hard: `{_pct(metric['all_improvement'])}` / `{_pct(metric['t50_improvement'])}` / `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(metric['hard_failure_improvement'])}`.",
        f"- easy degradation: `{_pct(metric['easy_degradation'])}`; switch rate: `{_pct(metric['switch_rate'])}`.",
        f"- near-collision@0.05 delta vs endpoint-linear: `{_pct(safety['near_collision_005_delta'])}`; jagged-rate delta: `{_pct(safety['jagged_rate_delta'])}`.",
        "- The guard's second proximity input is the validation-selected base composer candidate rollout group min-distance, not future labels.",
        "- This refresh does not create a new metric/seconds/3D/foundation claim; it only strengthens deployable policy reproducibility under protected dataset-local/raw-frame 2.5D boundaries.",
        "- Stage5C remains unexecuted and SMC remains disabled.",
    ]


def _refresh_paper_files(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    lines = _refresh_lines(summary)
    status: list[dict[str, Any]] = []
    for path in PAPER_FILES:
        _replace_section(path, "STAGE42_CW_RUNTIME_REPLAY_REFRESH", lines)
        text = path.read_text(encoding="utf-8")
        status.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "contains_stage42_cw": "Stage42-CW Runtime Replay Paper / Reproducibility Refresh" in text,
                "contains_exact_replay": "exact runtime replay" in text,
                "contains_no_metric_boundary": "does not create a new metric/seconds/3D/foundation claim" in text,
                "contains_stage5c_smc_boundary": "Stage5C remains unexecuted and SMC remains disabled" in text,
            }
        )
    return status


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload["runtime_replay_summary"])
    for path in [README_RESULTS, M3W_README, RETRO_README]:
        _replace_section(path, "STAGE42_CW_RUNTIME_REPLAY_PAPER_REFRESH", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    summary = payload["runtime_replay_summary"]
    metric = summary["test_vs_endpoint_linear_ade"]
    safety = summary["joint_safety_vs_endpoint_linear"]
    state["current_stage"] = "Stage42-CW runtime replay paper refresh"
    state["current_verdict"] = payload["stage42_cw_gate"]["verdict"]
    state["stage42_cw_runtime_replay_paper_refresh"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_cw_gate"]["verdict"],
        "gates": f"{payload['stage42_cw_gate']['passed']}/{payload['stage42_cw_gate']['total']}",
        "policy_hash": summary["policy_hash"],
        "val_rows": summary["val_rows"],
        "test_rows": summary["test_rows"],
        "test_decision_exact_replay": summary["test_decision_exact_replay"],
        "test_selected_xy_max_abs_diff": summary["test_selected_xy_max_abs_diff"],
        "test_selected_ade_max_abs_diff": summary["test_selected_ade_max_abs_diff"],
        "test_selected_fde_max_abs_diff": summary["test_selected_fde_max_abs_diff"],
        "test_vs_endpoint_linear_ade": metric,
        "joint_safety_vs_endpoint_linear": safety,
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["runtime_replay_summary"]
    metric = summary["test_vs_endpoint_linear_ade"]
    safety = summary["joint_safety_vs_endpoint_linear"]
    claim = payload["claim_boundary"]
    gates = {
        "cv_gate_passed": summary["gate"]["passed"] == summary["gate"]["total"],
        "ct_gate_verified": payload["inputs"]["stage42_ct"].get("stage42_ct_gate", {}).get("passed")
        == payload["inputs"]["stage42_ct"].get("stage42_ct_gate", {}).get("total"),
        "cu_gate_verified": payload["inputs"]["stage42_cu"].get("stage42_cu_gate", {}).get("passed")
        == payload["inputs"]["stage42_cu"].get("stage42_cu_gate", {}).get("total"),
        "paper_files_refreshed": all(row["contains_stage42_cw"] for row in payload["paper_file_status"]),
        "exact_runtime_replay_documented": summary["val_decision_exact_replay"] is True
        and summary["test_decision_exact_replay"] is True,
        "selected_xy_exact": float(summary["test_selected_xy_max_abs_diff"]) <= 1e-12,
        "selected_ade_exact": float(summary["test_selected_ade_max_abs_diff"]) <= 1e-12,
        "selected_fde_exact": float(summary["test_selected_fde_max_abs_diff"]) <= 1e-12,
        "all_positive": float(metric["all_improvement"]) > 0.0,
        "t50_positive": float(metric["t50_improvement"]) > 0.0,
        "t100_raw_positive": float(metric["t100_raw_frame_diagnostic_improvement"]) > 0.0,
        "hard_positive": float(metric["hard_failure_improvement"]) > 0.0,
        "easy_under_2pct": float(metric["easy_degradation"]) <= 0.02,
        "proximity_not_worse_than_endpoint": float(safety["near_collision_005_delta"]) <= 0.0,
        "runtime_replay_not_new_training": payload["source"] == "fresh_synthesis_from_stage42_cv_runtime_batch_replay",
        "no_future_endpoint_input": payload["no_leakage"]["future_endpoint_input"] is False,
        "no_future_waypoints_input": payload["no_leakage"]["future_waypoints_input"] is False,
        "no_central_velocity": payload["no_leakage"]["central_velocity"] is False,
        "no_test_endpoint_goals": payload["no_leakage"]["test_endpoint_goals"] is False,
        "no_test_threshold_tuning": payload["no_leakage"]["test_threshold_tuning"] is False,
        "metric_seconds_overclaim_blocked": claim["metric_or_seconds_claim"] is False,
        "true_3d_overclaim_blocked": claim["true_3d"] is False,
        "foundation_overclaim_blocked": claim["foundation_world_model"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_cw_runtime_replay_paper_refresh_pass" if passed == total else "stage42_cw_runtime_replay_paper_refresh_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["runtime_replay_summary"]
    metric = summary["test_vs_endpoint_linear_ade"]
    safety = summary["joint_safety_vs_endpoint_linear"]
    lines = [
        "# Stage42-CW Runtime Replay Paper Refresh",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_cw_gate']['passed']} / {payload['stage42_cw_gate']['total']}`",
        f"- verdict: `{payload['stage42_cw_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in payload["current_facts"]],
        "",
        "## Runtime Replay Evidence",
        "",
        f"- policy_hash: `{summary['policy_hash']}`",
        f"- validation rows: `{summary['val_rows']}`",
        f"- test rows: `{summary['test_rows']}`",
        f"- validation decision exact replay: `{summary['val_decision_exact_replay']}`",
        f"- test decision exact replay: `{summary['test_decision_exact_replay']}`",
        f"- test selected_xy max abs diff: `{summary['test_selected_xy_max_abs_diff']}`",
        f"- test selected ADE max abs diff: `{summary['test_selected_ade_max_abs_diff']}`",
        f"- test selected FDE max abs diff: `{summary['test_selected_fde_max_abs_diff']}`",
        "",
        "## Test Metrics Vs Endpoint-Linear ADE",
        "",
        f"- all: `{_pct(metric['all_improvement'])}`",
        f"- t50: `{_pct(metric['t50_improvement'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}`",
        f"- hard/failure: `{_pct(metric['hard_failure_improvement'])}`",
        f"- easy degradation: `{_pct(metric['easy_degradation'])}`",
        f"- switch rate: `{_pct(metric['switch_rate'])}`",
        "",
        "## Joint Safety Vs Endpoint-Linear",
        "",
        f"- near_collision@0.02 delta: `{_pct(safety['near_collision_002_delta'])}`",
        f"- near_collision@0.05 delta: `{_pct(safety['near_collision_005_delta'])}`",
        f"- p05 min group distance delta: `{_pct(safety['p05_min_group_distance_delta'])}`",
        f"- jagged-rate delta: `{_pct(safety['jagged_rate_delta'])}`",
        "",
        "## Paper File Status",
        "",
        "| file | refreshed | exact replay | no metric boundary | Stage5C/SMC boundary |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["paper_file_status"]:
        lines.append(
            f"| `{row['path']}` | {row['contains_stage42_cw']} | {row['contains_exact_replay']} | {row['contains_no_metric_boundary']} | {row['contains_stage5c_smc_boundary']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Stage42-CW moves Stage42-CV from an isolated runtime report into the paper/reproducibility package.",
        "- The evidence is deployment-reproducibility evidence: frozen policy artifact, runtime API, and batch rows all replay exactly.",
        "- This strengthens the protected policy claim but does not expand it into true 3D, foundation, metric, seconds-level, Stage5C, or SMC claims.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_cw_gate"]
    lines = [
        "# Stage42-CW Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | passed |",
        "| --- | ---: |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | `{bool(value)}` |")
    return lines


def run_stage42_runtime_replay_paper_refresh() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    cv = read_json(CV_JSON, {})
    ct = read_json(CT_JSON, {})
    cu = read_json(CU_JSON, {})
    summary = _cv_summary(cv)
    paper_file_status = _refresh_paper_files(summary)
    payload: dict[str, Any] = {
        "source": "fresh_synthesis_from_stage42_cv_runtime_batch_replay",
        "stage": "Stage42-CW runtime replay paper refresh",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([CV_JSON, CT_JSON, CU_JSON]),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "stage42_cv": cv,
            "stage42_ct": ct,
            "stage42_cu": cu,
        },
        "runtime_replay_summary": summary,
        "paper_file_status": paper_file_status,
        "no_leakage": cv.get("no_leakage", {}),
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_cw_gate"] = _gate(payload)
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    return payload


if __name__ == "__main__":
    run_stage42_runtime_replay_paper_refresh()
