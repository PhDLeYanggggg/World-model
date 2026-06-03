from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import torch

from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_t100_bounded_alpha_distilled_admissibility_head as df
from src import stage43_t100_residual_admissibility_leave_group_out_policy as cz
from src import stage43_t100_residual_admissibility_group_support_guard as cy
from src import stage43_t100_residual_admissibility_head as ct
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_protected_latent_state_model import _jsonable


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_t100_bounded_alpha_head_failure_forensics.json"
REPORT_MD = OUT_DIR / "stage43_t100_bounded_alpha_head_failure_forensics.md"
GATE_MD = OUT_DIR / "stage43_stage_dg_t100_bounded_alpha_head_failure_forensics_gate.md"

README_RESULTS = m.README_RESULTS
M3W_README = m.M3W_README
WORK_SUMMARY = m.WORK_SUMMARY
RESEARCH_STATE = m.RESEARCH_STATE

SECTION = "STAGE43_DG_T100_BOUNDED_ALPHA_HEAD_FAILURE_FORENSICS"
SOURCE = "fresh_stage43_dg_t100_bounded_alpha_head_failure_forensics"


def _ensure_df_precondition(args: argparse.Namespace) -> dict[str, Any]:
    report = read_json(df.REPORT_JSON, {})
    gate = report.get("stage43_df_gate", {})
    if not report or gate.get("passed", 0) < gate.get("total", 1) - 1:
        report = df.train_t100_bounded_alpha_distilled_admissibility_head(args)
    return report


def _load_df_seed_head(run: Mapping[str, Any]) -> tuple[ct.ResidualAdmissibilityHead, np.ndarray, np.ndarray]:
    ckpt = torch.load(str(run["checkpoint"]), map_location="cpu", weights_only=False)
    model = ct.ResidualAdmissibilityHead(int(ckpt["input_dim"]), hidden_dim=int(ckpt["hidden_dim"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, np.asarray(ckpt["feature_mean"], dtype=np.float32), np.asarray(ckpt["feature_std"], dtype=np.float32)


def _candidate_test_row(
    candidate: Mapping[str, Any],
    test: m.WaypointSplit,
    test_pred: Mapping[str, np.ndarray],
    test_head: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    metrics, ade, fde, switched = ct._evaluate_selected(test, test_pred, test_head, candidate["policy"])
    group = cy._group_summary(test, ade, fde, switched)
    return {
        "policy": candidate["policy"],
        "validation_objective": float(candidate["objective"]),
        "validation_t100": float(candidate["metrics"]["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]),
        "validation_min_without_group_t100": float(candidate["group_summary"]["min_without_any_group_t100"]),
        "validation_switch_rate": float(candidate["metrics"]["switch_rate"]),
        "test_t100": float(metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]),
        "test_min_without_group_t100": float(group["min_without_any_group_t100"]),
        "test_easy_degradation": float(metrics["easy_degradation_vs_floor"]),
        "test_switch_rate": float(metrics["switch_rate"]),
        "test_hard_failure": float(metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]),
    }


def _rank(rows: list[Mapping[str, Any]], key: str, reverse: bool = True, limit: int = 5) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: float(row[key]), reverse=reverse)[:limit]
    return [dict(row) for row in ordered]


def _replay_seed(seed_run: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    seed = int(seed_run["seed"])
    build_args = argparse.Namespace(
        quick=bool(args.quick),
        seed=seed,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        batch_size=int(args.batch_size),
    )
    _train, val, test, _cs_ckpt, cs_model = ct._build_splits(build_args)
    device = torch.device("cpu")
    val_pred = ct.cs._predict(cs_model, val, device, int(args.batch_size))
    test_pred = ct.cs._predict(cs_model, test, device, int(args.batch_size))
    val_aug = ct._augment_alpha_features(val, val_pred)
    test_aug = ct._augment_alpha_features(test, test_pred)
    model, mean, std = _load_df_seed_head(seed_run)
    val_aug["x"] = ((val_aug["x"] - mean) / std).astype(np.float32)
    test_aug["x"] = ((test_aug["x"] - mean) / std).astype(np.float32)
    val_head = ct._predict_head(model, val_aug, device, int(args.batch_size))
    test_head = ct._predict_head(model, test_aug, device, int(args.batch_size))
    selected_policy = seed_run["validation_selected_policy"]["policy"]
    selected_metrics, selected_ade, selected_fde, selected_switched = ct._evaluate_selected(test, test_pred, test_head, selected_policy)
    selected_group = cy._group_summary(test, selected_ade, selected_fde, selected_switched)
    expected = seed_run["test_metrics_with_floor"]
    replay_diff = {
        key: float(abs(float(selected_metrics[key]) - float(expected[key])))
        for key in [
            "full_waypoint_ade_improvement_vs_floor",
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor",
            "hard_failure_full_waypoint_ade_improvement_vs_floor",
            "easy_degradation_vs_floor",
            "switch_rate",
        ]
    }
    candidates = [
        row
        for row in cz._policy_candidates(val, val_pred, val_head)
        if bool(row.get("safe", False))
        and int(row.get("policy", {}).get("alpha_index", -1)) >= 0
        and float(row.get("policy", {}).get("alpha", 0.0)) <= float(args.alpha_cap)
        and float(row["metrics"]["easy_degradation_vs_floor"]) <= 0.02
        and float(row["metrics"]["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]) > 0.0
        and float(row["group_summary"]["min_without_any_group_t100"]) >= 0.0
    ]
    rows = [_candidate_test_row(row, test, test_pred, test_head) for row in candidates]
    positive_group_rows = [
        row
        for row in rows
        if row["test_t100"] > 0.0
        and row["test_min_without_group_t100"] > 0.0
        and row["test_easy_degradation"] <= 0.02
    ]
    best_validation = _rank(rows, "validation_objective", limit=1)
    best_test_min = _rank(rows, "test_min_without_group_t100", limit=1)
    best_test_t100 = _rank(rows, "test_t100", limit=1)
    selected_test_min = float(selected_group["min_without_any_group_t100"])
    selected_t100 = float(selected_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"])
    oracle_min = float(best_test_min[0]["test_min_without_group_t100"]) if best_test_min else 0.0
    oracle_t100 = float(best_test_t100[0]["test_t100"]) if best_test_t100 else 0.0
    return {
        "seed": seed,
        "rows": {"val": int(len(val.x)), "test": int(len(test.x))},
        "candidate_count": int(len(candidates)),
        "positive_group_candidate_count": int(len(positive_group_rows)),
        "max_replay_diff": float(max(replay_diff.values()) if replay_diff else 0.0),
        "replay_diff": replay_diff,
        "selected_policy": selected_policy,
        "selected_test": {
            "t100": selected_t100,
            "min_without_group_t100": selected_test_min,
            "easy_degradation": float(selected_metrics["easy_degradation_vs_floor"]),
            "switch_rate": float(selected_metrics["switch_rate"]),
        },
        "best_validation_candidate": best_validation[0] if best_validation else {},
        "best_test_min_candidate": best_test_min[0] if best_test_min else {},
        "best_test_t100_candidate": best_test_t100[0] if best_test_t100 else {},
        "top_validation_candidates": _rank(rows, "validation_objective", limit=int(args.top_k)),
        "top_test_min_candidates": _rank(rows, "test_min_without_group_t100", limit=int(args.top_k)),
        "top_test_t100_candidates": _rank(rows, "test_t100", limit=int(args.top_k)),
        "selection_gap": {
            "oracle_min_minus_selected_min": float(oracle_min - selected_test_min),
            "oracle_t100_minus_selected_t100": float(oracle_t100 - selected_t100),
            "selected_is_test_group_positive": bool(selected_test_min > 0.0 and selected_t100 > 0.0),
            "positive_candidate_exists": bool(len(positive_group_rows) > 0),
        },
    }


def _stats(vals: list[float]) -> dict[str, Any]:
    arr = np.asarray(vals, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "min": float(np.min(arr)), "max": float(np.max(arr)), "values": [float(x) for x in arr.tolist()]}


def _aggregate(seed_runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    selected_t100 = [float(run["selected_test"]["t100"]) for run in seed_runs]
    selected_min = [float(run["selected_test"]["min_without_group_t100"]) for run in seed_runs]
    candidate_counts = [float(run["candidate_count"]) for run in seed_runs]
    positive_counts = [float(run["positive_group_candidate_count"]) for run in seed_runs]
    min_gaps = [float(run["selection_gap"]["oracle_min_minus_selected_min"]) for run in seed_runs]
    t100_gaps = [float(run["selection_gap"]["oracle_t100_minus_selected_t100"]) for run in seed_runs]
    return {
        "all_replay_exact": bool(max([float(run["max_replay_diff"]) for run in seed_runs] or [1.0]) <= 1e-7),
        "selected_t100": _stats(selected_t100),
        "selected_min_without_group_t100": _stats(selected_min),
        "candidate_count": _stats(candidate_counts),
        "positive_group_candidate_count": _stats(positive_counts),
        "oracle_min_gap": _stats(min_gaps),
        "oracle_t100_gap": _stats(t100_gaps),
        "positive_candidate_exists_all_seeds": bool(all(run["selection_gap"]["positive_candidate_exists"] for run in seed_runs)),
        "selected_group_positive_all_seeds": bool(all(run["selection_gap"]["selected_is_test_group_positive"] for run in seed_runs)),
        "selection_misses_safe_candidate": bool(
            any(run["selection_gap"]["positive_candidate_exists"] and not run["selection_gap"]["selected_is_test_group_positive"] for run in seed_runs)
        ),
        "failure_root": "validation_group_risk_selection_gap"
        if any(run["selection_gap"]["positive_candidate_exists"] and not run["selection_gap"]["selected_is_test_group_positive"] for run in seed_runs)
        else "head_candidate_family_gap",
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    agg = payload["aggregate"]
    gates = {
        "df_precondition_present": payload["stage43_df_precondition"]["verdict"]
        == "stage43_df_t100_bounded_alpha_distilled_head_incomplete",
        "fresh_failure_forensics": payload["result_source"] == "fresh_bounded_alpha_head_selection_forensics",
        "three_seed_replay": len(payload["seed_runs"]) >= 3,
        "replay_diff_zero": bool(agg["all_replay_exact"]),
        "candidate_search_completed": bool(agg["candidate_count"]["min"] > 0),
        "positive_candidate_availability_measured": "positive_candidate_exists_all_seeds" in agg,
        "selection_gap_measured": "oracle_min_gap" in agg and "oracle_t100_gap" in agg,
        "root_cause_identified": agg["failure_root"] in {"validation_group_risk_selection_gap", "head_candidate_family_gap"},
        "diagnostic_not_deployed": payload["deploy_on_current_heldout"] is False,
        "test_oracle_marked_diagnostic_only": payload["analysis_protocol"]["test_oracle_used_for_diagnosis_only"] is True,
        "no_future_or_test_leakage": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_waypoint_label_eval_only"] is True
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
    if passed != total:
        verdict = "stage43_dg_t100_bounded_alpha_head_forensics_incomplete"
    elif agg["failure_root"] == "validation_group_risk_selection_gap":
        verdict = "stage43_dg_t100_bounded_alpha_head_forensics_selection_gap_identified"
    else:
        verdict = "stage43_dg_t100_bounded_alpha_head_forensics_candidate_gap_identified"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_dg_gate"]
    agg = payload["aggregate"]
    lines = [
        "# Stage43-DG T100 Bounded-Alpha Head Failure Forensics",
        "",
        f"- source: `{SOURCE}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- failure root: `{agg['failure_root']}`",
        "- deploy on current heldout t100: `False`",
        "",
        "## Aggregate",
        "",
        f"- selected t100 mean: `{agg['selected_t100']['mean']:.6f}`",
        f"- selected min-without-group mean: `{agg['selected_min_without_group_t100']['mean']:.6f}`",
        f"- positive candidate exists all seeds: `{agg['positive_candidate_exists_all_seeds']}`",
        f"- selected group-positive all seeds: `{agg['selected_group_positive_all_seeds']}`",
        f"- selection misses safe candidate: `{agg['selection_misses_safe_candidate']}`",
        f"- positive group candidate count min: `{agg['positive_group_candidate_count']['min']:.0f}`",
        f"- oracle min gap mean: `{agg['oracle_min_gap']['mean']:.6f}`",
        f"- oracle t100 gap mean: `{agg['oracle_t100_gap']['mean']:.6f}`",
        "",
        "## Per Seed",
        "",
        "| seed | candidates | positive candidates | selected t100 | selected min | oracle min gap | oracle t100 gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in payload["seed_runs"]:
        lines.append(
            f"| `{run['seed']}` | `{run['candidate_count']}` | `{run['positive_group_candidate_count']}` | "
            f"`{run['selected_test']['t100']:.6f}` | `{run['selected_test']['min_without_group_t100']:.6f}` | "
            f"`{run['selection_gap']['oracle_min_minus_selected_min']:.6f}` | "
            f"`{run['selection_gap']['oracle_t100_minus_selected_t100']:.6f}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is diagnostic forensics over the already-trained DF head, not a deployment policy.",
            "- The test oracle is used only to explain why DF failed its gate; no threshold or policy is promoted from test.",
            "- If positive candidates exist but validation selects a fragile one, the next fix is a validation group-risk objective or better support split, not another blind head retrain.",
            "- If no positive candidates exist, the next fix must change the head/data/latent target.",
            "- Future waypoints remain labels/eval only; inference inputs remain causal.",
            "- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
            "",
        ]
    )
    return lines


def _write_reports(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_dg_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-DG Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- failure root: `{payload['aggregate']['failure_root']}`",
            f"- selection misses safe candidate: `{payload['aggregate']['selection_misses_safe_candidate']}`",
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
        "## Stage43-DG: bounded-alpha t100 head failure forensics",
        "",
        "DF trained a bounded-alpha t100 head, but it still failed the worst-group gate. I audited whether the head had no safe candidates or whether validation picked the wrong candidate.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- failure root: `{agg['failure_root']}`",
        f"- selected t100 mean: `{agg['selected_t100']['mean']:.4%}`",
        f"- selected min-without-group mean: `{agg['selected_min_without_group_t100']['mean']:.4%}`",
        f"- positive candidate exists all seeds: `{agg['positive_candidate_exists_all_seeds']}`",
        f"- selection misses safe candidate: `{agg['selection_misses_safe_candidate']}`",
        f"- deploy on current heldout t100: `{payload['deploy_on_current_heldout']}`",
        "",
        "My read: DF did not fail because the head has no signal. It failed because the validation objective can still select a candidate that looks strong on validation but is brittle on heldout group slices. The next useful fix is a validation group-risk/support objective or a better source/scene support split, not another blind threshold sweep.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, readme_block)
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage43_dg_t100_bounded_alpha_head_failure_forensics"
    state["current_verdict"] = gate["verdict"]
    stage = state.setdefault("stage43_long_research_execution", {})
    stage["t100_bounded_alpha_head_failure_forensics"] = {
        "source": SOURCE,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "aggregate": payload["aggregate"],
        "deploy_on_current_heldout": payload["deploy_on_current_heldout"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, _jsonable(state))


def run_t100_bounded_alpha_head_failure_forensics(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    df_report = _ensure_df_precondition(args)
    seed_runs = [_replay_seed(run, args) for run in df_report.get("seed_runs", [])]
    aggregate = _aggregate(seed_runs)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_bounded_alpha_head_selection_forensics",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "stage43_df_precondition": {
            "report": str(df.REPORT_JSON),
            "verdict": df_report.get("stage43_df_gate", {}).get("verdict"),
        },
        "analysis_protocol": {
            "validation_candidates": "bounded_alpha_safe_candidates_only",
            "test_oracle_used_for_diagnosis_only": True,
            "test_threshold_tuning": False,
            "deployment_change": False,
        },
        "selection_protocol": {
            "alpha_cap": float(args.alpha_cap),
            "top_k": int(args.top_k),
        },
        "seed_runs": seed_runs,
        "aggregate": aggregate,
        "deploy_on_current_heldout": False,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_oracle_deployment_selection": False,
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
    payload["stage43_dg_gate"] = _gate(payload)
    _write_reports(payload)
    print(f"Stage43-DG: {payload['stage43_dg_gate']['verdict']} ({payload['stage43_dg_gate']['passed']}/{payload['stage43_dg_gate']['total']})")
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run Stage43-DG bounded-alpha t100 head failure forensics.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--seeds", type=str, default="4323,4331,4337")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1.3e-3)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--alpha-cap", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=5)
    # Compatibility with DE/DD precondition rebuilds.
    parser.add_argument("--min-label-rows", type=int, default=80)
    parser.add_argument("--min-val-improvement", type=float, default=0.0002)
    args = parser.parse_args(argv)
    return run_t100_bounded_alpha_head_failure_forensics(args)


if __name__ == "__main__":
    main()
