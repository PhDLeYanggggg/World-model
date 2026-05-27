from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
DP_JSON = OUT_DIR / "context_model_closure_stage42.json"
DQ_JSON = OUT_DIR / "full_waypoint_promotion_checkpoint_stage42.json"
DN_JSON = OUT_DIR / "deployment_variant_card_stage42.json"
DO_JSON = OUT_DIR / "source_legal_time_action_package_stage42.json"

REPORT_JSON = OUT_DIR / "paper_package_post_dq_refresh_stage42.json"
REPORT_MD = OUT_DIR / "paper_package_post_dq_refresh_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_dr_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
RETRO_README = Path("README_M3W_GOAL_RETROSPECTIVE_CURRENT_ZH.md")
CURRENT_GOAL_README = Path("README_M3W_CURRENT_GOAL_SUMMARY_ZH.md")
GOAL_RESULTS_README = Path("README_M3W_GOAL_RESULTS_SUMMARY_ZH.md")
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

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-DR 是 paper-ready evidence refresh，不重新训练，不调 threshold，不把 paper update 当模型成功。",
    "本阶段同步 Stage42-DP context closure 和 Stage42-DQ full-waypoint promotion checkpoint。",
    "future endpoints / waypoints 只作为 supervised/evaluation labels，不能作为 inference input。",
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
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _summary(dp: Mapping[str, Any], dq: Mapping[str, Any], dn: Mapping[str, Any], do: Mapping[str, Any]) -> dict[str, Any]:
    dp_summary = dp["summary"]
    dq_summary = dq["summary"]
    runtime = dq_summary["group_consistency_runtime_replay_vs_train_horizon_causal_floor"]
    safety = dq_summary["group_consistency_safety"]
    do_summary = do.get("summary", {})
    return {
        "source": "fresh_paper_refresh_from_stage42_dp_dq_dn_do",
        "context_closure": {
            "decision": dp_summary["closure_decision"],
            "best_delta_all": dp_summary["best_delta_all"],
            "best_delta_t50": dp_summary["best_delta_t50"],
            "best_delta_hard_failure": dp_summary["best_delta_hard_failure"],
            "positive_context_rows": [row["variant"] for row in dp_summary["positive_context_rows"]],
            "claim": "sequence/graph residual context is closed under the current protocol; context remains auxiliary/diagnostic until target/data/model changes.",
        },
        "full_waypoint_checkpoint": {
            "runtime_all_improvement": runtime["all_improvement"],
            "runtime_t50_improvement": runtime["t50_improvement"],
            "runtime_t100_raw_frame_diagnostic_improvement": runtime["t100_raw_frame_diagnostic_improvement"],
            "runtime_hard_failure_improvement": runtime["hard_failure_improvement"],
            "runtime_easy_degradation": runtime["easy_degradation"],
            "runtime_switch_rate": runtime["switch_rate"],
            "switch_exact_match": safety["switch_exact_match"],
            "selected_xy_max_abs_diff": safety["selected_xy_max_abs_diff"],
            "runtime_base_near_005": safety["runtime_base_near_005"],
            "runtime_final_near_005": safety["runtime_final_near_005"],
            "runtime_floor_near_005": safety["runtime_floor_near_005"],
            "claim": "protected source-level group-consistency full-waypoint runtime policy is supported; ungated/global primary full-waypoint replacement is not supported.",
        },
        "deployment_variant_boundary": {
            "safety_sensitive_default": dn.get("recommended_policy", {}).get("safety_sensitive_default"),
            "accuracy_priority_diagnostic": dn.get("recommended_policy", {}).get("accuracy_priority_diagnostic"),
            "source_level_full_waypoint_runtime_candidate": dn.get("recommended_policy", {}).get(
                "source_level_full_waypoint_runtime_candidate"
            ),
            "baseline_mixing_caveat": bool(dn.get("baseline_mixing_caveat", True)),
        },
        "source_legal_time_boundary": {
            "conversion_ready_targets": do_summary.get("conversion_ready_targets", 0),
            "converted_datasets_now": do_summary.get("converted_datasets_now", 0),
            "evaluated_datasets_now": do_summary.get("evaluated_datasets_now", 0),
            "global_metric_seconds_claim_allowed": bool(do_summary.get("global_metric_seconds_claim_allowed", False)),
            "global_t100_deployable_claim_allowed": bool(do_summary.get("global_t100_deployable_claim_allowed", False)),
        },
        "paper_verdict": {
            "paper_ready_evidence_package_strengthened": True,
            "context_main_claim_allowed": False,
            "full_waypoint_runtime_evidence_allowed": True,
            "ungated_neural_or_full_waypoint_deployment_allowed": False,
            "metric_seconds_claim_allowed": False,
            "foundation_claim_allowed": False,
            "stage5c_execution_allowed": False,
            "smc_allowed": False,
        },
    }


def _refresh_lines(summary: Mapping[str, Any]) -> list[str]:
    ctx = summary["context_closure"]
    fw = summary["full_waypoint_checkpoint"]
    deploy = summary["deployment_variant_boundary"]
    src = summary["source_legal_time_boundary"]
    return [
        "## Stage42-DR Post-DP/DQ Paper Evidence Refresh",
        "",
        "- source: `fresh_paper_refresh_from_stage42_dp_dq_dn_do`",
        "- role: synchronize paper-ready evidence after the fresh context-closure and full-waypoint-promotion checkpoints.",
        "- This is not new training and not a threshold search; it updates claim hygiene and paper artifacts.",
        "",
        "### Context Claim Boundary",
        "",
        f"- closure decision: `{ctx['decision']}`.",
        f"- best context deltas vs baseline-family control all/t50/hard: `{_pct(ctx['best_delta_all'])}` / `{_pct(ctx['best_delta_t50'])}` / `{_pct(ctx['best_delta_hard_failure'])}`.",
        f"- positive context rows: `{ctx['positive_context_rows']}`.",
        "- Paper wording: sequence/graph/neighbor/goal context remains auxiliary or diagnostic under the current residual protocol, not an independent main contribution.",
        "",
        "### Full-Waypoint Runtime Evidence",
        "",
        f"- runtime all/t50/t100 raw/hard vs train-horizon causal floor: `{_pct(fw['runtime_all_improvement'])}` / `{_pct(fw['runtime_t50_improvement'])}` / `{_pct(fw['runtime_t100_raw_frame_diagnostic_improvement'])}` / `{_pct(fw['runtime_hard_failure_improvement'])}`.",
        f"- runtime easy degradation: `{_pct(fw['runtime_easy_degradation'])}`; switch rate: `{_pct(fw['runtime_switch_rate'])}`.",
        f"- exact replay: switch `{fw['switch_exact_match']}`, selected_xy max abs diff `{fw['selected_xy_max_abs_diff']}`.",
        f"- near@0.05 base/final/floor: `{_pct(fw['runtime_base_near_005'])}` / `{_pct(fw['runtime_final_near_005'])}` / `{_pct(fw['runtime_floor_near_005'])}`.",
        "- Paper wording: protected source-level group-consistency full-waypoint runtime policy is valid evidence, but ungated full-waypoint and global primary replacement remain blocked.",
        "",
        "### Deployment Variant Boundary",
        "",
        f"- safety-sensitive default: `{deploy['safety_sensitive_default']}`.",
        f"- accuracy-priority diagnostic: `{deploy['accuracy_priority_diagnostic']}`.",
        f"- source-level full-waypoint runtime candidate: `{deploy['source_level_full_waypoint_runtime_candidate']}`.",
        f"- baseline mixing caveat: `{deploy['baseline_mixing_caveat']}`.",
        "",
        "### Source / Time / Metric Boundary",
        "",
        f"- conversion-ready targets: `{src['conversion_ready_targets']}`; converted now: `{src['converted_datasets_now']}`; evaluated now: `{src['evaluated_datasets_now']}`.",
        f"- global metric/seconds claim allowed: `{src['global_metric_seconds_claim_allowed']}`.",
        f"- global t100 deployable claim allowed: `{src['global_t100_deployable_claim_allowed']}`.",
        "- Paper wording: dataset-local/raw-frame only unless future source/legal/time calibration closes the blocker.",
        "",
        "### Non-Claims",
        "",
        "- Do not claim true 3D.",
        "- Do not claim foundation world model.",
        "- Do not claim global metric or seconds-level prediction.",
        "- Do not claim Stage5C execution.",
        "- Do not claim SMC readiness.",
    ]


def _refresh_paper_files(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    lines = _refresh_lines(summary)
    status: list[dict[str, Any]] = []
    for path in PAPER_FILES:
        _replace_section(path, "STAGE42_DR_POST_DQ_PAPER_REFRESH", lines)
        text = path.read_text(encoding="utf-8")
        status.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "contains_stage42_dr": "Stage42-DR Post-DP/DQ Paper Evidence Refresh" in text,
                "contains_context_closure": "sequence/graph/neighbor/goal context remains auxiliary or diagnostic" in text,
                "contains_full_waypoint_runtime": "group-consistency full-waypoint runtime policy is valid evidence" in text,
                "contains_source_time_boundary": "dataset-local/raw-frame only" in text,
                "contains_stage5c_smc_boundary": "Do not claim Stage5C execution" in text
                and "Do not claim SMC readiness" in text,
            }
        )
    return status


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["paper_refresh_summary"]
    paper = summary["paper_verdict"]
    ctx = summary["context_closure"]
    fw = summary["full_waypoint_checkpoint"]
    src = summary["source_legal_time_boundary"]
    claim = payload["claim_boundary"]
    gates = {
        "dp_gate_passed": payload["inputs"]["stage42_dp"].get("stage42_dp_gate", {}).get("passed")
        == payload["inputs"]["stage42_dp"].get("stage42_dp_gate", {}).get("total"),
        "dq_gate_passed": payload["inputs"]["stage42_dq"].get("stage42_dq_gate", {}).get("passed")
        == payload["inputs"]["stage42_dq"].get("stage42_dq_gate", {}).get("total"),
        "dn_gate_passed": payload["inputs"]["stage42_dn"].get("stage42_dn_gate", {}).get("passed")
        == payload["inputs"]["stage42_dn"].get("stage42_dn_gate", {}).get("total"),
        "do_gate_passed": payload["inputs"]["stage42_do"].get("stage42_do_gate", {}).get("passed")
        == payload["inputs"]["stage42_do"].get("stage42_do_gate", {}).get("total"),
        "paper_files_refreshed": all(row["contains_stage42_dr"] for row in payload["paper_file_status"]),
        "context_main_claim_blocked": paper["context_main_claim_allowed"] is False
        and ctx["decision"] == "close_current_sequence_graph_residual_context_protocol",
        "full_waypoint_runtime_evidence_allowed": paper["full_waypoint_runtime_evidence_allowed"] is True
        and fw["switch_exact_match"] is True
        and float(fw["selected_xy_max_abs_diff"]) == 0.0,
        "ungated_deployment_blocked": paper["ungated_neural_or_full_waypoint_deployment_allowed"] is False,
        "source_time_metric_blocked": src["global_metric_seconds_claim_allowed"] is False
        and src["global_t100_deployable_claim_allowed"] is False,
        "no_metric_seconds_overclaim": claim["metric_or_seconds_claim"] is False,
        "true_3d_overclaim_blocked": claim["true_3d"] is False,
        "foundation_overclaim_blocked": claim["foundation_world_model"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_dr_post_dq_paper_refresh_pass" if passed == total else "stage42_dr_post_dq_paper_refresh_partial"
    return {"source": payload.get("source", "unknown"), "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    summary = payload["paper_refresh_summary"]
    gate = payload["stage42_dr_gate"]
    lines = [
        "# Stage42-DR Post-DP/DQ Paper Evidence Refresh",
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
    ]
    lines.extend(f"- {fact}" for fact in CURRENT_FACTS)
    lines.extend(["", "## Refresh Content", ""])
    lines.extend(_refresh_lines(summary))
    lines.extend(
        [
            "",
            "## Paper File Status",
            "",
            "| file | refreshed | context closure | full-waypoint runtime | source/time boundary | Stage5C/SMC boundary |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["paper_file_status"]:
        lines.append(
            f"| `{row['path']}` | {row['contains_stage42_dr']} | {row['contains_context_closure']} | {row['contains_full_waypoint_runtime']} | {row['contains_source_time_boundary']} | {row['contains_stage5c_smc_boundary']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-DR strengthens the paper package after fresh DP/DQ evidence.",
            "- It keeps negative context evidence visible rather than hiding it.",
            "- It keeps protected full-waypoint runtime evidence visible without overclaiming ungated/global replacement.",
            "- It keeps source/legal/time blockers explicit for metric, seconds-level, and global t100 claims.",
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_dr_gate"]
    lines = [
        "# Stage42-DR Gate",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    lines.extend(f"| `{name}` | {bool(value)} |" for name, value in gate["gates"].items())
    return lines


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload["paper_refresh_summary"])
    for path in [README_RESULTS, M3W_README, RETRO_README, CURRENT_GOAL_README, GOAL_RESULTS_README]:
        _replace_section(path, "STAGE42_DR_POST_DQ_PAPER_REFRESH", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    gate = payload["stage42_dr_gate"]
    state["current_stage"] = "Stage42-DR post-DP/DQ paper evidence refresh"
    state["current_verdict"] = gate["verdict"]
    state["stage42_dr_post_dq_paper_refresh"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": gate["verdict"],
        "gates": f"{gate['passed']}/{gate['total']}",
        "paper_refresh_summary": payload["paper_refresh_summary"],
        "paper_files": [row["path"] for row in payload["paper_file_status"]],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_post_dq_paper_refresh(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    dp = read_json(DP_JSON, {})
    dq = read_json(DQ_JSON, {})
    dn = read_json(DN_JSON, {})
    do = read_json(DO_JSON, {})
    input_hash = _combined_hash([DP_JSON, DQ_JSON, DN_JSON, DO_JSON])
    claim_boundary = {
        "true_3d": False,
        "foundation_world_model": False,
        "metric_or_seconds_claim": False,
        "raw_frame_dataset_local_only": True,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    payload: dict[str, Any] = {
        "source": "fresh_paper_refresh_from_stage42_dp_dq_dn_do",
        "stage": "Stage42-DR",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": input_hash,
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "stage42_dp": dp,
            "stage42_dq": dq,
            "stage42_dn": dn,
            "stage42_do": do,
        },
        "paper_refresh_summary": _summary(dp, dq, dn, do),
        "claim_boundary": claim_boundary,
    }
    payload["paper_file_status"] = _refresh_paper_files(payload["paper_refresh_summary"])
    payload["stage42_dr_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_post_dq_paper_refresh()
