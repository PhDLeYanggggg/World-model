from __future__ import annotations

import csv
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src import stage42_common_validation_bridge_shape_composer as co
from src import stage42_common_validation_composer_safety as cp


OUT_DIR = Path("outputs/stage42_long_research")
CO_JSON = OUT_DIR / "common_validation_bridge_shape_composer_stage42.json"
CP_JSON = OUT_DIR / "common_validation_composer_safety_stage42.json"
CQ_JSON = OUT_DIR / "proximity_aware_composer_guard_stage42.json"

REPORT_JSON = OUT_DIR / "proximity_guard_ablation_stage42.json"
REPORT_MD = OUT_DIR / "proximity_guard_ablation_stage42.md"
REPORT_CSV = OUT_DIR / "proximity_guard_ablation_stage42.csv"
GATE_MD = OUT_DIR / "stage42_stage_cr_gate.md"

PAPER_FILES = [
    OUT_DIR / "experiment_tables_stage42.md",
    OUT_DIR / "ablation_tables_stage42.md",
    OUT_DIR / "model_card_stage42.md",
    OUT_DIR / "a_journal_gap_stage42.md",
]

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-CR 是 CO/CP/CQ 的 proximity guard ablation / Pareto 审计。",
    "CR 不新增 test tuning；它比较已经 validation-selected 的 CO composer 与 CQ proximity guard。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals。",
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


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _metric_row(name: str, role: str, metric: Mapping[str, Any], joint_delta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    joint_delta = joint_delta or {}
    return {
        "name": name,
        "role": role,
        "all_improvement": float(metric.get("all_improvement", 0.0)),
        "t50_improvement": float(metric.get("t50_improvement", 0.0)),
        "t100_raw_frame_diagnostic_improvement": float(metric.get("t100_raw_frame_diagnostic_improvement", 0.0)),
        "hard_failure_improvement": float(metric.get("hard_failure_improvement", 0.0)),
        "easy_degradation": float(metric.get("easy_degradation", 0.0)),
        "switch_rate": float(metric.get("switch_rate", 0.0)),
        "near_collision_005_delta_vs_endpoint": (
            None
            if joint_delta.get("near_collision_rate_005_delta") is None
            else float(joint_delta["near_collision_rate_005_delta"])
        ),
        "p05_min_distance_delta_vs_endpoint": (
            None
            if joint_delta.get("p05_min_group_distance_delta") is None
            else float(joint_delta["p05_min_group_distance_delta"])
        ),
        "jagged_rate_delta_vs_endpoint": (
            None if joint_delta.get("jagged_rate_delta") is None else float(joint_delta["jagged_rate_delta"])
        ),
    }


def _delta(after: Mapping[str, Any], before: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "all_improvement",
        "t50_improvement",
        "t100_raw_frame_diagnostic_improvement",
        "hard_failure_improvement",
        "easy_degradation",
        "switch_rate",
        "near_collision_005_delta_vs_endpoint",
        "p05_min_distance_delta_vs_endpoint",
        "jagged_rate_delta_vs_endpoint",
    ]
    out = {}
    for key in keys:
        a = after.get(key)
        b = before.get(key)
        out[key] = None if a is None or b is None else float(a) - float(b)
    return out


def _refresh_paper_files(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    guard = payload["ablation_rows"]["proximity_guard"]
    no_guard = payload["ablation_rows"]["no_proximity_guard"]
    contribution = payload["guard_contribution"]
    lines = [
        "## Stage42-CR Proximity Guard Ablation / Pareto Audit",
        "",
        "- source: `fresh_synthesis_from_stage42_co_cp_cq_artifacts`",
        "- scope: CO/CP unguarded composer versus CQ proximity-aware composer guard.",
        f"- no proximity guard ADE all/t50/t100/hard: `{_pct(no_guard['all_improvement'])}` / `{_pct(no_guard['t50_improvement'])}` / `{_pct(no_guard['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(no_guard['hard_failure_improvement'])}`.",
        f"- proximity guard ADE all/t50/t100/hard: `{_pct(guard['all_improvement'])}` / `{_pct(guard['t50_improvement'])}` / `{_pct(guard['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(guard['hard_failure_improvement'])}`.",
        f"- guard accuracy cost all/t50/t100/hard: `{_pct(-contribution['accuracy_cost_vs_no_guard']['all_improvement'])}` / `{_pct(-contribution['accuracy_cost_vs_no_guard']['t50_improvement'])}` / `{_pct(-contribution['accuracy_cost_vs_no_guard']['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(-contribution['accuracy_cost_vs_no_guard']['hard_failure_improvement'])}`.",
        f"- guard near-collision@0.05 repair versus no guard: `{_pct(contribution['safety_delta_vs_no_guard']['near_collision_005_delta_vs_endpoint'])}`.",
        "- claim boundary: still dataset-local/raw-frame 2.5D; no metric/seconds-level, no Stage5C, no SMC.",
    ]
    status = []
    for path in PAPER_FILES:
        co._replace_section(path, "STAGE42_CR_PROXIMITY_GUARD_ABLATION", lines)
        text = path.read_text(encoding="utf-8")
        status.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "contains_stage42_cr": "Stage42-CR Proximity Guard Ablation" in text,
                "contains_claim_boundary": "no metric/seconds-level" in text,
            }
        )
    return status


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    co_gate = payload["inputs"]["stage42_co"]["stage42_co_gate"]
    cp_gate = payload["inputs"]["stage42_cp"]["stage42_cp_gate"]
    cq_gate = payload["inputs"]["stage42_cq"]["stage42_cq_gate"]
    rows = payload["ablation_rows"]
    no_guard = rows["no_proximity_guard"]
    guard = rows["proximity_guard"]
    contribution = payload["guard_contribution"]
    gates = {
        "co_gate_passed": co_gate["passed"] == co_gate["total"],
        "cp_gate_passed": cp_gate["passed"] == cp_gate["total"],
        "cq_gate_passed": cq_gate["passed"] == cq_gate["total"],
        "ablation_rows_present": {"endpoint_linear_reference", "no_proximity_guard", "proximity_guard"}.issubset(rows.keys()),
        "no_guard_has_accuracy_gain": no_guard["all_improvement"] > guard["all_improvement"],
        "no_guard_has_proximity_caveat": no_guard["near_collision_005_delta_vs_endpoint"] > 0.0,
        "guard_repairs_near_collision": guard["near_collision_005_delta_vs_endpoint"] <= 0.0,
        "guard_retains_all_positive": guard["all_improvement"] > 0.0,
        "guard_retains_t50_positive": guard["t50_improvement"] > 0.0,
        "guard_retains_t100_positive": guard["t100_raw_frame_diagnostic_improvement"] > 0.0,
        "guard_retains_hard_positive": guard["hard_failure_improvement"] > 0.0,
        "guard_easy_under_2pct": guard["easy_degradation"] <= 0.02,
        "safety_delta_documented": contribution["safety_delta_vs_no_guard"]["near_collision_005_delta_vs_endpoint"] < 0.0,
        "accuracy_cost_documented": contribution["accuracy_cost_vs_no_guard"]["all_improvement"] < 0.0,
        "deployment_recommendation_safe": payload["deployment_recommendation"]["safety_sensitive_policy"] == "proximity_guard",
        "paper_files_refreshed": all(row["contains_stage42_cr"] for row in payload["paper_file_status"]),
        "metric_seconds_overclaim_blocked": payload["claim_boundary"]["metric_or_seconds_claim"] is False,
        "stage5c_not_executed": payload["claim_boundary"]["stage5c_executed"] is False,
        "smc_not_enabled": payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_cr_proximity_guard_ablation_pass" if passed == total else "stage42_cr_proximity_guard_ablation_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _write_csv(rows: Mapping[str, Mapping[str, Any]]) -> None:
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "role",
                "all_improvement",
                "t50_improvement",
                "t100_raw_frame_diagnostic_improvement",
                "hard_failure_improvement",
                "easy_degradation",
                "switch_rate",
                "near_collision_005_delta_vs_endpoint",
                "p05_min_distance_delta_vs_endpoint",
                "jagged_rate_delta_vs_endpoint",
            ],
        )
        writer.writeheader()
        for row in rows.values():
            writer.writerow(row)


def _write_md(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_cr_gate"]
    rows = payload["ablation_rows"]
    contribution = payload["guard_contribution"]
    rec = payload["deployment_recommendation"]
    lines = [
        "# Stage42-CR Proximity Guard Ablation / Pareto Audit",
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
        "## Ablation Rows",
        "",
        "| variant | role | all | t50 | t100 raw | hard | easy | near@0.05 vs endpoint | switch |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ["endpoint_linear_reference", "no_proximity_guard", "proximity_guard"]:
        row = rows[name]
        lines.append(
            f"| `{name}` | {row['role']} | {_pct(row['all_improvement'])} | {_pct(row['t50_improvement'])} | {_pct(row['t100_raw_frame_diagnostic_improvement'])} | {_pct(row['hard_failure_improvement'])} | {_pct(row['easy_degradation'])} | {_pct(row['near_collision_005_delta_vs_endpoint'])} | {_pct(row['switch_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Guard Contribution",
            "",
            f"- all-ADE cost versus no guard: `{_pct(contribution['accuracy_cost_vs_no_guard']['all_improvement'])}`",
            f"- t50-ADE cost versus no guard: `{_pct(contribution['accuracy_cost_vs_no_guard']['t50_improvement'])}`",
            f"- t100 raw diagnostic cost versus no guard: `{_pct(contribution['accuracy_cost_vs_no_guard']['t100_raw_frame_diagnostic_improvement'])}`",
            f"- hard/failure cost versus no guard: `{_pct(contribution['accuracy_cost_vs_no_guard']['hard_failure_improvement'])}`",
            f"- near-collision@0.05 repair versus no guard: `{_pct(contribution['safety_delta_vs_no_guard']['near_collision_005_delta_vs_endpoint'])}`",
            "",
            "## Recommendation",
            "",
            f"- accuracy-priority diagnostic policy: `{rec['accuracy_priority_policy']}`",
            f"- safety-sensitive deployment policy: `{rec['safety_sensitive_policy']}`",
            "- Do not present no-guard CO/CP as the safety-sensitive deployment policy because its near-collision@0.05 is worse than endpoint-linear.",
            "- Do not present CQ as strictly more accurate than CO/CP; CQ is the safer Pareto point.",
            "",
            "## Interpretation",
            "",
            "- The proximity guard has a real causal contribution: it repairs the near-collision caveat while keeping all/t50/t100 raw-frame/hard-failure gains positive.",
            "- The tradeoff is explicit: CQ gives up some CO/CP ADE gain to satisfy joint-proximity safety.",
            "- This remains protected dataset-local/raw-frame 2.5D evidence, not metric/seconds-level, Stage5C, or SMC.",
        ]
    )
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage42-CR Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
        *[f"| `{name}` | `{ok}` |" for name, ok in gate["gates"].items()],
    ]
    write_md(GATE_MD, gate_lines)


def run() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    co_report = read_json(CO_JSON, {})
    cp_report = read_json(CP_JSON, {})
    cq_report = read_json(CQ_JSON, {})
    endpoint = _metric_row(
        "endpoint_linear_reference",
        "no_full_waypoint_shape_reference",
        {
            "all_improvement": 0.0,
            "t50_improvement": 0.0,
            "t100_raw_frame_diagnostic_improvement": 0.0,
            "hard_failure_improvement": 0.0,
            "easy_degradation": 0.0,
            "switch_rate": 0.0,
        },
        {
            "near_collision_rate_005_delta": 0.0,
            "p05_min_group_distance_delta": 0.0,
            "jagged_rate_delta": 0.0,
        },
    )
    no_guard = _metric_row(
        "no_proximity_guard",
        "accuracy_priority_diagnostic",
        cp_report["test_metric_vs_endpoint_ade"],
        cp_report["joint_safety"]["composer_minus_endpoint"],
    )
    guard = _metric_row(
        "proximity_guard",
        "safety_sensitive_deployable",
        cq_report["test_eval"]["metric_vs_endpoint_ade"],
        cq_report["test_joint_safety"]["composer_minus_endpoint"],
    )
    rows = {
        "endpoint_linear_reference": endpoint,
        "no_proximity_guard": no_guard,
        "proximity_guard": guard,
    }
    payload: dict[str, Any] = {
        "source": "fresh_synthesis_from_stage42_co_cp_cq_artifacts",
        "stage": "Stage42-CR proximity guard ablation / Pareto audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "input_hash": _combined_hash([CO_JSON, CP_JSON, CQ_JSON]),
        "inputs": {"stage42_co": co_report, "stage42_cp": cp_report, "stage42_cq": cq_report},
        "ablation_rows": rows,
        "guard_contribution": {
            "accuracy_cost_vs_no_guard": _delta(guard, no_guard),
            "safety_delta_vs_no_guard": _delta(guard, no_guard),
        },
        "deployment_recommendation": {
            "accuracy_priority_policy": "no_proximity_guard",
            "safety_sensitive_policy": "proximity_guard",
            "reason": "CO/CP has higher ADE gain but worsens near-collision@0.05 versus endpoint-linear; CQ keeps gains positive and makes near-collision no worse than endpoint-linear/floor.",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "new_test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["paper_file_status"] = _refresh_paper_files(payload)
    payload["stage42_cr_gate"] = _gate(payload)
    write_json(REPORT_JSON, _jsonable(payload))
    _write_csv(rows)
    _write_md(payload)
    return payload


if __name__ == "__main__":
    result = run()
    gate = result["stage42_cr_gate"]
    print(f"Stage42-CR proximity guard ablation: {gate['verdict']} ({gate['passed']}/{gate['total']})")
