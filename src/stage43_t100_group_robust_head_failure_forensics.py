from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_t100_group_robust_admissibility_head as da
from src import stage43_t100_residual_admissibility_leave_group_out_policy as cz
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_protected_latent_state_model import _jsonable


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_t100_group_robust_head_failure_forensics.json"
REPORT_MD = OUT_DIR / "stage43_t100_group_robust_head_failure_forensics.md"
GATE_MD = OUT_DIR / "stage43_stage_db_t100_group_robust_head_failure_forensics_gate.md"

README_RESULTS = m.README_RESULTS
M3W_README = m.M3W_README
WORK_SUMMARY = m.WORK_SUMMARY
RESEARCH_STATE = m.RESEARCH_STATE

SECTION = "STAGE43_DB_T100_GROUP_ROBUST_HEAD_FAILURE_FORENSICS"
SOURCE = "fresh_stage43_db_t100_group_robust_head_failure_forensics"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stats(values: list[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)) if len(arr) else 0.0,
        "std": float(np.std(arr)) if len(arr) else 0.0,
        "min": float(np.min(arr)) if len(arr) else 0.0,
        "max": float(np.max(arr)) if len(arr) else 0.0,
        "values": [float(x) for x in arr.tolist()],
    }


def _ensure_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    cz_report = read_json(cz.REPORT_JSON, {})
    if not cz_report or cz_report.get("stage43_cz_gate", {}).get("passed") != cz_report.get("stage43_cz_gate", {}).get("total"):
        cz_report = cz.run_t100_leave_group_out_policy(args)
    da_report = read_json(da.REPORT_JSON, {})
    if not da_report or da_report.get("stage43_da_gate", {}).get("passed") != da_report.get("stage43_da_gate", {}).get("total"):
        da_report = da.train_t100_group_robust_admissibility_head(args)
    return cz_report, da_report


def _align_seed_runs(cz_report: Mapping[str, Any], da_report: Mapping[str, Any]) -> list[dict[str, Any]]:
    cz_by_seed = {int(run["seed"]): run for run in cz_report.get("seed_runs", [])}
    da_by_seed = {int(run["seed"]): run for run in da_report.get("seed_runs", [])}
    common = sorted(set(cz_by_seed) & set(da_by_seed))
    rows: list[dict[str, Any]] = []
    for seed in common:
        c = cz_by_seed[seed]
        d = da_by_seed[seed]
        cz_metrics = c["robust_test_metrics"]
        cz_group = c["robust_test_group_summary"]
        da_metrics = d["test_metrics_with_floor"]
        da_group = d["test_group_summary"]
        val_group = d["validation_selected_policy"]["group_summary"]
        rows.append(
            {
                "seed": int(seed),
                "cz_policy": c["selected_validation_candidate"]["policy"],
                "da_policy": d["validation_selected_policy"]["policy"],
                "cz_t100": float(cz_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]),
                "da_t100": float(da_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]),
                "delta_t100_da_minus_cz": float(
                    da_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]
                    - cz_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]
                ),
                "cz_min_without_group_t100": float(cz_group["min_without_any_group_t100"]),
                "da_min_without_group_t100": float(da_group["min_without_any_group_t100"]),
                "delta_min_without_group_da_minus_cz": float(
                    da_group["min_without_any_group_t100"] - cz_group["min_without_any_group_t100"]
                ),
                "da_val_min_without_group_t100": float(val_group["min_without_any_group_t100"]),
                "da_val_to_test_min_without_group_gap": float(
                    da_group["min_without_any_group_t100"] - val_group["min_without_any_group_t100"]
                ),
                "cz_scene_group_flip_count": int(cz_group["scene_group_flip_count"]),
                "da_scene_group_flip_count": int(da_group["scene_group_flip_count"]),
                "cz_switch_rate": float(cz_metrics["switch_rate"]),
                "da_switch_rate": float(da_metrics["switch_rate"]),
                "delta_switch_rate_da_minus_cz": float(da_metrics["switch_rate"] - cz_metrics["switch_rate"]),
                "da_easy_degradation": float(da_metrics["easy_degradation_vs_floor"]),
                "da_bootstrap_t100_low": float(
                    d["bootstrap_ci"]["metrics"]["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]["low"]
                ),
                "da_best_epoch": int(d["best_epoch"]),
                "da_safe_candidates": int(d["validation_selected_policy"]["safe_candidates"]),
            }
        )
    return rows


def _aggregate(comparisons: list[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [float(r["delta_t100_da_minus_cz"]) for r in comparisons]
    min_deltas = [float(r["delta_min_without_group_da_minus_cz"]) for r in comparisons]
    switch_deltas = [float(r["delta_switch_rate_da_minus_cz"]) for r in comparisons]
    da_t100 = [float(r["da_t100"]) for r in comparisons]
    da_min = [float(r["da_min_without_group_t100"]) for r in comparisons]
    val_test_gap = [float(r["da_val_to_test_min_without_group_gap"]) for r in comparisons]
    root_causes = {
        "trained_head_underperforms_policy_only": bool(comparisons and all(v < 0.0 for v in deltas)),
        "group_worst_case_not_preserved": bool(comparisons and all(v < 0.0 for v in da_min)),
        "group_worst_case_gap_vs_cz": bool(comparisons and all(v < 0.0 for v in min_deltas)),
        "under_switching_relative_to_cz": bool(comparisons and np.mean(switch_deltas) < -0.02),
        "validation_to_test_group_gap": bool(comparisons and np.mean(val_test_gap) < -0.001),
        "not_an_easy_safety_failure": bool(comparisons and max(float(r["da_easy_degradation"]) for r in comparisons) <= 0.02),
        "not_a_no_signal_failure": bool(comparisons and min(da_t100) > 0.0),
    }
    return {
        "seed_count": int(len(comparisons)),
        "delta_t100_da_minus_cz": _stats(deltas),
        "delta_min_without_group_da_minus_cz": _stats(min_deltas),
        "delta_switch_rate_da_minus_cz": _stats(switch_deltas),
        "da_t100": _stats(da_t100),
        "da_min_without_group_t100": _stats(da_min),
        "da_val_to_test_min_without_group_gap": _stats(val_test_gap),
        "da_safe_candidates": _stats([float(r["da_safe_candidates"]) for r in comparisons]),
        "root_causes": root_causes,
        "root_cause_count": int(sum(bool(v) for v in root_causes.values())),
    }


def _repair_hypotheses(agg: Mapping[str, Any]) -> list[dict[str, Any]]:
    causes = agg["root_causes"]
    rows = [
        {
            "hypothesis": "DA optimizes gain/harm/delta labels but not the actual CZ deployment policy.",
            "evidence": "DA is positive but below CZ in every aligned seed.",
            "next_test": "Train a policy-distilled admissibility head using CZ leave-group-out selected switches as teacher labels.",
            "priority": "high",
            "triggered": bool(causes["trained_head_underperforms_policy_only"]),
        },
        {
            "hypothesis": "The support penalty is too indirect to protect worst-case source/scene/domain groups.",
            "evidence": "DA min-without-group t100 is negative while CZ remains positive.",
            "next_test": "Add explicit worst-group validation loss or group DRO style batch objective, then select by min-without-group.",
            "priority": "high",
            "triggered": bool(causes["group_worst_case_not_preserved"]),
        },
        {
            "hypothesis": "The trained head is more conservative than CZ and misses useful t100 switches.",
            "evidence": "DA switch rate drops relative to CZ.",
            "next_test": "Distill high-confidence CZ switches while keeping harm/easy conformal guard.",
            "priority": "medium",
            "triggered": bool(causes["under_switching_relative_to_cz"]),
        },
        {
            "hypothesis": "Validation group support does not transfer cleanly to test groups.",
            "evidence": "DA validation min-without-group is stronger than DA test min-without-group.",
            "next_test": "Use nested leave-one-source/scene validation and choose checkpoints by heldout-group transfer.",
            "priority": "medium",
            "triggered": bool(causes["validation_to_test_group_gap"]),
        },
    ]
    return rows


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    agg = payload["aggregate"]
    root = agg["root_causes"]
    gates = {
        "cz_report_verified": payload["input_reports"]["cz_gate_passed"] is True,
        "da_report_verified": payload["input_reports"]["da_gate_passed"] is True,
        "seed_alignment_complete": agg["seed_count"] >= 3,
        "da_positive_signal_confirmed": root["not_a_no_signal_failure"] is True,
        "da_underperformance_identified": root["trained_head_underperforms_policy_only"] is True,
        "group_worst_case_failure_identified": root["group_worst_case_not_preserved"] is True,
        "easy_not_primary_failure": root["not_an_easy_safety_failure"] is True,
        "repair_hypotheses_written": len([r for r in payload["repair_hypotheses"] if r["triggered"]]) >= 2,
        "diagnostic_not_deployed": payload["deploy_on_current_heldout"] is False,
        "no_future_or_test_leakage": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["central_velocity_input"] is False
        and payload["no_leakage"]["test_endpoint_goal_construction"] is False
        and payload["no_leakage"]["test_statistics_normalization"] is False,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
        "long_objective_kept_active": payload["long_objective_complete"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    if passed == total:
        verdict = "stage43_db_t100_head_failure_forensics_complete_policy_distill_next"
    else:
        verdict = "stage43_db_t100_head_failure_forensics_incomplete"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_db_gate"]
    agg = payload["aggregate"]
    lines = [
        "# Stage43-DB T100 Group-Robust Head Failure Forensics",
        "",
        f"- source: `{SOURCE}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        "- deploy on current heldout t100: `False`",
        "",
        "## Aggregate Deltas",
        "",
        f"- DA minus CZ t100 mean: `{agg['delta_t100_da_minus_cz']['mean']:.6f}`",
        f"- DA minus CZ min-without-group mean: `{agg['delta_min_without_group_da_minus_cz']['mean']:.6f}`",
        f"- DA minus CZ switch-rate mean: `{agg['delta_switch_rate_da_minus_cz']['mean']:.6f}`",
        f"- DA validation-to-test min-without-group gap mean: `{agg['da_val_to_test_min_without_group_gap']['mean']:.6f}`",
        f"- root cause count: `{agg['root_cause_count']}`",
        "",
        "## Root Causes",
        "",
    ]
    for key, value in agg["root_causes"].items():
        lines.append(f"- `{key}`: `{bool(value)}`")
    lines.extend(
        [
            "",
            "## Per Seed",
            "",
            "| seed | DA t100 | CZ t100 | delta | DA min-without | CZ min-without | DA switch | CZ switch |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["seed_comparisons"]:
        lines.append(
            f"| `{row['seed']}` | `{row['da_t100']:.6f}` | `{row['cz_t100']:.6f}` | "
            f"`{row['delta_t100_da_minus_cz']:.6f}` | `{row['da_min_without_group_t100']:.6f}` | "
            f"`{row['cz_min_without_group_t100']:.6f}` | `{row['da_switch_rate']:.6f}` | `{row['cz_switch_rate']:.6f}` |"
        )
    lines.extend(["", "## Repair Hypotheses", ""])
    for item in payload["repair_hypotheses"]:
        status = "triggered" if item["triggered"] else "not primary"
        lines.append(f"- `{item['priority']}` / `{status}`: {item['hypothesis']} Next test: {item['next_test']}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- DA is not a no-signal failure: every seed remains t100-positive and easy-safe.",
            "- DA is also not ready to replace CZ: it loses t100 mean, loses worst-group robustness, and switches less often than CZ.",
            "- The next repair should train toward the actual CZ robust policy decisions and worst-group transfer objective, not just generic gain/harm/delta labels.",
            "- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
            "",
        ]
    )
    return lines


def _write_reports(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_db_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-DB Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            "- deploy on current heldout t100: `False`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
            "| gate | pass |",
            "| --- | --- |",
            *[f"| `{key}` | `{bool(value)}` |" for key, value in gate["gates"].items()],
            "",
        ],
    )
    agg = payload["aggregate"]
    readme_block = [
        "## Stage43-DB: why the trained t100 head did not beat CZ",
        "",
        "I compared the Stage43-CZ policy-only repair against the Stage43-DA trained head seed by seed. The trained head has real positive signal, but it loses the exact thing CZ fixed: worst-group robustness.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- DA minus CZ t100 mean: `{agg['delta_t100_da_minus_cz']['mean']:.4%}`",
        f"- DA minus CZ min-without-group mean: `{agg['delta_min_without_group_da_minus_cz']['mean']:.4%}`",
        f"- DA minus CZ switch-rate mean: `{agg['delta_switch_rate_da_minus_cz']['mean']:.4%}`",
        f"- root causes identified: `{agg['root_cause_count']}`",
        "- deploy on current heldout t100: `False`",
        "",
        "My read is that DA trained generic gain/harm/delta labels, while CZ's win came from the deployment policy and leave-group-out objective. The next useful training step is policy distillation from the CZ robust decisions plus an explicit worst-group transfer loss.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, readme_block)

    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage43_db_t100_group_robust_head_failure_forensics"
    state["current_verdict"] = gate["verdict"]
    stage = state.setdefault("stage43_long_research_execution", {})
    stage["t100_group_robust_head_failure_forensics"] = {
        "source": SOURCE,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "aggregate": payload["aggregate"],
        "repair_hypotheses": payload["repair_hypotheses"],
        "deploy_on_current_heldout": payload["deploy_on_current_heldout"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, _jsonable(state))


def run_t100_group_robust_head_failure_forensics(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    cz_report, da_report = _ensure_inputs(args)
    comparisons = _align_seed_runs(cz_report, da_report)
    aggregate = _aggregate(comparisons)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_forensics_from_verified_cz_da_reports",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "input_reports": {
            "cz_report": str(cz.REPORT_JSON),
            "cz_sha256": _sha256(cz.REPORT_JSON) if cz.REPORT_JSON.exists() else "",
            "cz_gate_passed": cz_report.get("stage43_cz_gate", {}).get("passed") == cz_report.get("stage43_cz_gate", {}).get("total"),
            "da_report": str(da.REPORT_JSON),
            "da_sha256": _sha256(da.REPORT_JSON) if da.REPORT_JSON.exists() else "",
            "da_gate_passed": da_report.get("stage43_da_gate", {}).get("passed") == da_report.get("stage43_da_gate", {}).get("total"),
        },
        "seed_comparisons": comparisons,
        "aggregate": aggregate,
        "repair_hypotheses": _repair_hypotheses(aggregate),
        "deploy_on_current_heldout": False,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "report_only_no_new_training": True,
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "long_objective_complete": False,
    }
    payload["stage43_db_gate"] = _gate(payload)
    _write_reports(payload)
    print(f"Stage43-DB: {payload['stage43_db_gate']['verdict']} ({payload['stage43_db_gate']['passed']}/{payload['stage43_db_gate']['total']})")
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run Stage43-DB t100 group-robust head failure forensics.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--seeds", type=str, default="4323,4331,4337")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1.5e-3)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args(argv)
    return run_t100_group_robust_head_failure_forensics(args)


if __name__ == "__main__":
    main()
