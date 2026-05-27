from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_context_switchability_gate import run_stage42_context_switchability_gate
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
DC_JSON = OUT_DIR / "context_switchability_gate_stage42.json"

REPORT_JSON = OUT_DIR / "context_switchability_materiality_audit_stage42.json"
REPORT_MD = OUT_DIR / "context_switchability_materiality_audit_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_ee_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
GOAL_SUMMARY = Path("README_M3W_TARGET_WORK_SUMMARY_ZH.md")
RESEARCH_STATE = Path("research_state.json")

MATERIAL_DELTA = 0.01

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-EE fresh-runs Stage42-DC context switchability and applies a materiality gate.",
    "目标是防止把微小 context 增量包装成 scene/goal/neighbor/interaction 主贡献。",
    "future endpoints / waypoints 只作为 supervised/evaluation labels，不能作为 inference input。",
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
    "stage5c_executed": False,
    "smc_enabled": False,
}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _candidate_delta_rows(dc: Mapping[str, Any]) -> list[dict[str, Any]]:
    base = dc["baseline_family_control"]["protected_metric"]
    rows: list[dict[str, Any]] = []
    for name, row in dc["candidate_results"].items():
        metric = row["test_metric"]
        rows.append(
            {
                "candidate": name,
                "all": metric["all_improvement"],
                "t50": metric["t50_improvement"],
                "hard": metric["hard_failure_improvement"],
                "easy": metric["easy_degradation"],
                "switch_rate": metric["switch_rate"],
                "delta_all": float(metric["all_improvement"]) - float(base["all_improvement"]),
                "delta_t50": float(metric["t50_improvement"]) - float(base["t50_improvement"]),
                "delta_hard": float(metric["hard_failure_improvement"]) - float(base["hard_failure_improvement"]),
                "delta_easy": float(metric["easy_degradation"]) - float(base["easy_degradation"]),
            }
        )
    return sorted(rows, key=lambda row: (row["delta_all"] + row["delta_hard"] + row["delta_t50"]), reverse=True)


def _summary(dc: Mapping[str, Any]) -> dict[str, Any]:
    selected = dc["selected_context_switchability_policy"]
    delta = selected["delta_vs_baseline_family_control"]
    rows = _candidate_delta_rows(dc)
    best_any = max(
        rows,
        key=lambda row: max(row["delta_all"], row["delta_t50"], row["delta_hard"]),
    )
    material_axes = {
        "all": delta["all_improvement"] >= MATERIAL_DELTA,
        "t50": delta["t50_improvement"] >= MATERIAL_DELTA,
        "hard": delta["hard_failure_improvement"] >= MATERIAL_DELTA,
    }
    return {
        "source": "fresh_rerun_stage42_dc_context_switchability_materiality",
        "material_delta_threshold": MATERIAL_DELTA,
        "selected_candidate": selected["selected_candidate"],
        "dc_decision": selected["decision"],
        "dc_context_switchability_supported": selected["context_switchability_supported"],
        "selected_delta_all": delta["all_improvement"],
        "selected_delta_t50": delta["t50_improvement"],
        "selected_delta_hard": delta["hard_failure_improvement"],
        "selected_delta_easy": delta["easy_degradation"],
        "selected_switch_rate_delta": delta["switch_rate"],
        "material_axes": material_axes,
        "material_context_contribution": any(material_axes.values()),
        "best_candidate_by_any_delta": best_any,
        "candidate_delta_rows": rows,
        "decision": "context_switchability_materiality_blocked"
        if not any(material_axes.values())
        else "context_switchability_materiality_supported",
        "root_cause": (
            "Gain/harm context switchability can slightly adjust the baseline-family policy, but the best validation-selected "
            "test deltas are far below the 1 percentage-point materiality threshold and t50 is not improved."
        ),
        "next_action": (
            "Do not repeat the current context switchability protocol. If context is revisited, change the target or data support: "
            "source-specific calibrated conversion, graph-neural interaction tokens, or full-waypoint group-consistency objectives."
        ),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    dc_gate = payload["dc_rerun"].get("stage42_dc_gate", {})
    claim = payload["claim_boundary"]
    gates = {
        "dc_fresh_rerun_passed": dc_gate.get("passed") == dc_gate.get("total"),
        "multiple_context_candidates_checked": len(s["candidate_delta_rows"]) >= 5,
        "materiality_threshold_recorded": s["material_delta_threshold"] == MATERIAL_DELTA,
        "selected_delta_recorded": all(
            key in s
            for key in [
                "selected_delta_all",
                "selected_delta_t50",
                "selected_delta_hard",
                "selected_delta_easy",
            ]
        ),
        "context_materiality_decision_recorded": s["decision"] in {
            "context_switchability_materiality_blocked",
            "context_switchability_materiality_supported",
        },
        "micro_delta_not_overclaimed": s["material_context_contribution"] is False
        and claim["context_main_claim_allowed"] is False,
        "root_cause_written": bool(s["root_cause"]),
        "next_action_written": bool(s["next_action"]),
        "no_future_or_test_leakage": all(
            [
                payload["no_leakage"]["future_endpoint_input"] is False,
                payload["no_leakage"]["future_waypoint_input"] is False,
                payload["no_leakage"]["central_velocity"] is False,
                payload["no_leakage"]["test_endpoint_goals"] is False,
                payload["no_leakage"]["test_threshold_tuning"] is False,
            ]
        ),
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_ee_context_switchability_materiality_audit_pass" if passed == total else "stage42_ee_context_switchability_materiality_audit_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-EE Context Switchability Materiality Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_ee_gate']['passed']} / {payload['stage42_ee_gate']['total']}`",
        f"- verdict: `{payload['stage42_ee_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Materiality Summary",
        "",
        f"- material_delta_threshold: `{s['material_delta_threshold']}`",
        f"- selected_candidate: `{s['selected_candidate']}`",
        f"- selected_delta_all/t50/hard/easy: `{s['selected_delta_all']:.6f}` / `{s['selected_delta_t50']:.6f}` / `{s['selected_delta_hard']:.6f}` / `{s['selected_delta_easy']:.6f}`",
        f"- material_context_contribution: `{s['material_context_contribution']}`",
        f"- decision: `{s['decision']}`",
        f"- root_cause: {s['root_cause']}",
        "",
        "## Candidate Deltas Vs Baseline-Family Control",
        "",
        "| candidate | all | t50 | hard | easy | delta all | delta t50 | delta hard | delta easy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in s["candidate_delta_rows"]:
        lines.append(
            f"| `{row['candidate']}` | {row['all']:.6f} | {row['t50']:.6f} | {row['hard']:.6f} | {row['easy']:.6f} | {row['delta_all']:.6f} | {row['delta_t50']:.6f} | {row['delta_hard']:.6f} | {row['delta_easy']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Stage42-EE is a fresh rerun/materiality audit of the gain-harm context target, not a new data conversion or Stage5C run.",
            "- The current context switchability route remains useful as a negative result: it prevents overclaiming tiny graph/goal/neighbor increments as a main contribution.",
            "- The next context attempt must change target/data support rather than repeat residual or current gain-harm switchability.",
            "",
            "## Gate",
            "",
            "| gate | pass |",
            "| --- | ---: |",
            *[f"| `{key}` | {bool(value)} |" for key, value in payload["stage42_ee_gate"]["gates"].items()],
        ]
    )
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_ee_gate"]
    return [
        "# Stage42-EE Gate",
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
        "## Stage42-EE Context Switchability Materiality Audit",
        "",
        "- source: `fresh_rerun_stage42_dc_context_switchability_materiality`",
        "- role: fresh-reruns gain/harm context switchability and applies a 1pp materiality threshold.",
        f"- gate: `{payload['stage42_ee_gate']['passed']} / {payload['stage42_ee_gate']['total']}`; verdict `{payload['stage42_ee_gate']['verdict']}`.",
        f"- selected context candidate `{s['selected_candidate']}` delta all/t50/hard/easy `{s['selected_delta_all']:.6f}` / `{s['selected_delta_t50']:.6f}` / `{s['selected_delta_hard']:.6f}` / `{s['selected_delta_easy']:.6f}`.",
        f"- material_context_contribution: `{s['material_context_contribution']}`; decision `{s['decision']}`.",
        "- boundary: current context switchability has micro-deltas only, so scene/goal/neighbor/interaction main claims remain blocked under this protocol.",
    ]


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    lines = _refresh_lines(payload)
    for path in [README_RESULTS, M3W_README, GOAL_SUMMARY]:
        _replace_section(path, "STAGE42_EE_CONTEXT_SWITCHABILITY_MATERIALITY", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state["current_stage"] = "Stage42-EE context switchability materiality audit"
    state["current_verdict"] = payload["stage42_ee_gate"]["verdict"]
    state["stage42_ee_context_switchability_materiality"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_ee_gate"]["verdict"],
        "gates": f"{payload['stage42_ee_gate']['passed']}/{payload['stage42_ee_gate']['total']}",
        "summary": payload["summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, state)


def run_stage42_context_switchability_materiality_audit(*, refresh_readmes: bool = True) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    dc = run_stage42_context_switchability_gate(refresh_readmes=False)
    payload: dict[str, Any] = {
        "source": "fresh_rerun_stage42_dc_context_switchability_materiality",
        "stage": "Stage42-EE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([DC_JSON]),
        "current_facts": CURRENT_FACTS,
        "dc_rerun": {
            "source": dc.get("source"),
            "stage42_dc_gate": dc.get("stage42_dc_gate"),
            "selected_context_switchability_policy": dc.get("selected_context_switchability_policy"),
        },
        "summary": _summary(dc),
        "no_leakage": dc.get("no_leakage", {}),
        "claim_boundary": CLAIM_BOUNDARY,
    }
    payload["stage42_ee_gate"] = _gate(payload)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    if refresh_readmes:
        _refresh_readmes(payload)
        _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    run_stage42_context_switchability_materiality_audit()
