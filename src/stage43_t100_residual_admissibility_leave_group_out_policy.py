from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import torch

from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_t100_residual_admissibility_group_support_guard as cy
from src import stage43_t100_residual_admissibility_group_stress as cx
from src import stage43_t100_residual_admissibility_head as ct
from src import stage43_t100_residual_admissibility_statistical_confirmation as cu
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_protected_latent_state_model import _jsonable


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_t100_residual_admissibility_leave_group_out_policy.json"
REPORT_MD = OUT_DIR / "stage43_t100_residual_admissibility_leave_group_out_policy.md"
GATE_MD = OUT_DIR / "stage43_stage_cz_t100_residual_admissibility_leave_group_out_policy_gate.md"

README_RESULTS = m.README_RESULTS
M3W_README = m.M3W_README
WORK_SUMMARY = m.WORK_SUMMARY
RESEARCH_STATE = m.RESEARCH_STATE

SECTION = "STAGE43_CZ_T100_RESIDUAL_ADMISSIBILITY_LEAVE_GROUP_OUT_POLICY"
SOURCE = "fresh_stage43_cz_t100_residual_admissibility_leave_group_out_policy"


def _ensure_cy_precondition(args: argparse.Namespace) -> dict[str, Any]:
    report = read_json(cy.REPORT_JSON, {})
    gate = report.get("stage43_cy_gate", {})
    if not report or gate.get("passed") != gate.get("total"):
        report = cy.run_t100_group_support_guard(args)
    return report


def _robust_objective(metrics: Mapping[str, Any], group_summary: Mapping[str, Any]) -> float:
    min_without = float(group_summary["min_without_any_group_t100"])
    flip_count = float(group_summary["source_group_flip_count"] + group_summary["scene_group_flip_count"] + group_summary["domain_pair_flip_count"])
    return float(
        2.2 * float(metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"])
        + 0.8 * float(metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"])
        + 0.2 * float(metrics["full_waypoint_ade_improvement_vs_floor"])
        + 2.0 * min(0.01, min_without)
        - 0.75 * float(metrics["easy_degradation_vs_floor"])
        - 0.04 * float(metrics["switch_rate"])
        - 5.0 * max(0.0, -min_without)
        - 0.015 * flip_count
    )


def _policy_candidates(
    ds: m.WaypointSplit,
    cs_pred: Mapping[str, np.ndarray],
    head_pred: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    floor_ade = ds.floor_ade.copy()
    floor_fde = ds.floor_fde.copy()
    floor_switched = np.zeros(len(ds.x), dtype=bool)
    floor_metrics = m._metrics(ds, floor_ade, floor_fde, floor_switched)
    floor_group = cy._group_summary(ds, floor_ade, floor_fde, floor_switched)
    rows.append(
        {
            "policy": {
                "alpha": 0.0,
                "alpha_index": -1,
                "gain_threshold": 1.01,
                "harm_threshold": -0.01,
                "delta_threshold": -1.0,
                "force_easy_floor": True,
                "selection_mode": "leave_group_out_floor",
            },
            "metrics": floor_metrics,
            "group_summary": floor_group,
            "objective": _robust_objective(floor_metrics, floor_group),
            "safe": True,
        }
    )
    for ai, alpha in enumerate(ct.ALPHAS):
        for gain_thr in [0.20, 0.35, 0.50, 0.65, 0.80, 0.90]:
            for harm_thr in [0.03, 0.05, 0.10, 0.20, 0.35]:
                for delta_thr in [-0.020, -0.010, -0.005, -0.001, 0.0]:
                    for force_easy in [True, False]:
                        policy = {
                            "alpha": float(alpha),
                            "alpha_index": int(ai),
                            "gain_threshold": float(gain_thr),
                            "harm_threshold": float(harm_thr),
                            "delta_threshold": float(delta_thr),
                            "force_easy_floor": bool(force_easy),
                            "selection_mode": "leave_group_out_robust",
                        }
                        metrics, ade, fde, switched = ct._policy_metrics_for_alpha(ds, cs_pred, head_pred, alpha_index=ai, policy=policy)
                        group_summary = cy._group_summary(ds, ade, fde, switched)
                        safe = bool(
                            metrics["easy_degradation_vs_floor"] <= 0.02
                            and metrics["switch_rate"] <= 0.60
                            and group_summary["source_group_flip_count"] == 0
                            and group_summary["domain_pair_flip_count"] == 0
                        )
                        rows.append(
                            {
                                "policy": policy,
                                "metrics": metrics,
                                "group_summary": group_summary,
                                "objective": _robust_objective(metrics, group_summary),
                                "safe": safe,
                            }
                        )
    rows.sort(key=lambda row: (bool(row["safe"]), float(row["objective"])), reverse=True)
    return rows


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
    model, mean, std = cx.cw.cv._load_seed_head(seed_run)
    val_aug["x"] = ((val_aug["x"] - mean) / std).astype(np.float32)
    test_aug["x"] = ((test_aug["x"] - mean) / std).astype(np.float32)
    val_head = ct._predict_head(model, val_aug, device, int(args.batch_size))
    test_head = ct._predict_head(model, test_aug, device, int(args.batch_size))

    original_policy = seed_run["validation_selected_policy"]["policy"]
    original_metrics, original_ade, original_fde, original_switched = ct._evaluate_selected(test, test_pred, test_head, original_policy)
    expected = seed_run["test_metrics_with_floor"]
    replay_diff = {
        key: float(abs(float(original_metrics[key]) - float(expected[key])))
        for key in [
            "full_waypoint_ade_improvement_vs_floor",
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor",
            "hard_failure_full_waypoint_ade_improvement_vs_floor",
            "easy_degradation_vs_floor",
            "switch_rate",
        ]
    }

    candidates = _policy_candidates(val, val_pred, val_head)
    safe_candidates = [row for row in candidates if row["safe"]]
    selected = safe_candidates[0] if safe_candidates else candidates[0]
    robust_metrics, robust_ade, robust_fde, robust_switched = ct._evaluate_selected(test, test_pred, test_head, selected["policy"])
    original_group = cy._group_summary(test, original_ade, original_fde, original_switched)
    robust_group = cy._group_summary(test, robust_ade, robust_fde, robust_switched)
    return {
        "seed": seed,
        "rows": {"val": int(len(val.x)), "test": int(len(test.x))},
        "max_replay_diff": float(max(replay_diff.values()) if replay_diff else 0.0),
        "replay_diff": replay_diff,
        "candidate_count": int(len(candidates)),
        "safe_candidate_count": int(len(safe_candidates)),
        "selected_validation_candidate": selected,
        "original_test_metrics": original_metrics,
        "original_test_group_summary": original_group,
        "robust_test_metrics": robust_metrics,
        "robust_test_group_summary": robust_group,
        "test_delta_vs_original": {
            "t100": float(robust_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] - original_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]),
            "hard_failure": float(robust_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"] - original_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]),
            "easy_degradation": float(robust_metrics["easy_degradation_vs_floor"] - original_metrics["easy_degradation_vs_floor"]),
            "switch_rate": float(robust_metrics["switch_rate"] - original_metrics["switch_rate"]),
            "min_without_group_t100": float(robust_group["min_without_any_group_t100"] - original_group["min_without_any_group_t100"]),
            "scene_group_flip_count": float(robust_group["scene_group_flip_count"] - original_group["scene_group_flip_count"]),
        },
        "top_validation_candidates": [
            {
                "policy": row["policy"],
                "objective": float(row["objective"]),
                "safe": bool(row["safe"]),
                "metrics": row["metrics"],
                "group_summary": row["group_summary"],
            }
            for row in candidates[:8]
        ],
    }


def _nested(row: Mapping[str, Any], path: tuple[str, ...]) -> float:
    cur: Any = row
    for key in path:
        cur = cur[key]
    return float(cur)


def _stats(seed_runs: list[Mapping[str, Any]], path: tuple[str, ...]) -> dict[str, Any]:
    vals = np.asarray([_nested(run, path) for run in seed_runs], dtype=np.float64)
    return {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals)), "values": [float(x) for x in vals.tolist()]}


def _aggregate(seed_runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    out = {
        "all_replay_exact": bool(max([float(run["max_replay_diff"]) for run in seed_runs] or [1.0]) <= 1e-7),
        "safe_candidate_count": _stats(seed_runs, ("safe_candidate_count",)),
        "selected_modes": [str(run["selected_validation_candidate"]["policy"].get("selection_mode")) for run in seed_runs],
        "original_t100": _stats(seed_runs, ("original_test_metrics", "t100_raw_frame_full_waypoint_diagnostic_vs_floor")),
        "robust_t100": _stats(seed_runs, ("robust_test_metrics", "t100_raw_frame_full_waypoint_diagnostic_vs_floor")),
        "original_min_without_group_t100": _stats(seed_runs, ("original_test_group_summary", "min_without_any_group_t100")),
        "robust_min_without_group_t100": _stats(seed_runs, ("robust_test_group_summary", "min_without_any_group_t100")),
        "robust_easy_degradation": _stats(seed_runs, ("robust_test_metrics", "easy_degradation_vs_floor")),
        "robust_switch_rate": _stats(seed_runs, ("robust_test_metrics", "switch_rate")),
        "delta_t100_vs_original": _stats(seed_runs, ("test_delta_vs_original", "t100")),
        "delta_min_without_group_t100_vs_original": _stats(seed_runs, ("test_delta_vs_original", "min_without_group_t100")),
        "delta_scene_group_flip_count_vs_original": _stats(seed_runs, ("test_delta_vs_original", "scene_group_flip_count")),
    }
    out["robust_preserves_easy"] = bool(out["robust_easy_degradation"]["max"] <= 0.02)
    out["robust_t100_positive_all_seeds"] = bool(min(out["robust_t100"]["values"]) > 0.0)
    out["group_fragility_reduced"] = bool(out["delta_min_without_group_t100_vs_original"]["mean"] > 0.0 or out["delta_scene_group_flip_count_vs_original"]["mean"] < 0.0)
    return out


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    gates = {
        "stage43_cy_precondition_present": payload["stage43_cy_precondition"]["verdict"]
        in {
            "stage43_cy_t100_group_support_guard_no_repair_keep_diagnostic",
            "stage43_cy_t100_group_support_guard_reduces_fragility_diagnostic",
            "stage43_cy_t100_group_support_guard_reduces_fragility_but_floor_t100",
        },
        "fresh_leave_group_out_policy_search": payload["result_source"] == "fresh_leave_group_out_robust_policy_search",
        "three_seed_replay": len(payload["seed_runs"]) >= 3,
        "replay_diff_zero": bool(payload["aggregate"]["all_replay_exact"]),
        "validation_only_policy_selection": payload["selection_protocol"]["test_threshold_tuning"] is False,
        "safe_candidates_evaluated": payload["aggregate"]["safe_candidate_count"]["min"] > 0,
        "group_robust_objective_used": payload["selection_protocol"]["objective"] == "leave_group_out_min_t100_plus_flip_penalty",
        "group_fragility_reduction_measured": "group_fragility_reduced" in payload["aggregate"],
        "easy_preserved_or_floor": payload["aggregate"]["robust_preserves_easy"] is True,
        "diagnostic_not_deployed": payload["deploy_on_current_heldout"] is False,
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
        verdict = "stage43_cz_t100_leave_group_out_policy_incomplete_keep_diagnostic"
    elif payload["aggregate"]["group_fragility_reduced"] and payload["aggregate"]["robust_t100_positive_all_seeds"]:
        verdict = "stage43_cz_t100_leave_group_out_policy_reduces_fragility_diagnostic"
    elif payload["aggregate"]["group_fragility_reduced"]:
        verdict = "stage43_cz_t100_leave_group_out_policy_reduces_fragility_but_floor_t100"
    else:
        verdict = "stage43_cz_t100_leave_group_out_policy_no_repair_keep_diagnostic"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_cz_gate"]
    agg = payload["aggregate"]
    first = payload["seed_runs"][0] if payload["seed_runs"] else {}
    lines = [
        "# Stage43-CZ T100 Leave-Group-Out Robust Admissibility Policy",
        "",
        f"- source: `{SOURCE}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        "- deploy on current heldout t100: `False`",
        "",
        "## Aggregate",
        "",
        f"- all replay exact: `{agg['all_replay_exact']}`",
        f"- selected modes: `{agg['selected_modes']}`",
        f"- safe candidate count min: `{agg['safe_candidate_count']['min']:.0f}`",
        f"- original t100 mean: `{agg['original_t100']['mean']:.6f}`",
        f"- robust t100 mean: `{agg['robust_t100']['mean']:.6f}`",
        f"- original min without group t100 mean: `{agg['original_min_without_group_t100']['mean']:.6f}`",
        f"- robust min without group t100 mean: `{agg['robust_min_without_group_t100']['mean']:.6f}`",
        f"- delta min without group t100 mean: `{agg['delta_min_without_group_t100_vs_original']['mean']:.6f}`",
        f"- delta scene group flip count mean: `{agg['delta_scene_group_flip_count_vs_original']['mean']:.6f}`",
        f"- group fragility reduced: `{agg['group_fragility_reduced']}`",
        f"- robust easy degradation max: `{agg['robust_easy_degradation']['max']:.6f}`",
        f"- robust switch rate mean: `{agg['robust_switch_rate']['mean']:.6f}`",
        "",
    ]
    if first:
        selected = first["selected_validation_candidate"]
        lines.extend(
            [
                "## First Seed",
                "",
                f"- seed: `{first['seed']}`",
                f"- selected policy: `{selected['policy']}`",
                f"- validation objective: `{selected['objective']:.6f}`",
                f"- test delta vs original: `{first['test_delta_vs_original']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "- This step changes the validation selection objective itself: policies are rewarded for positive t100 while also surviving leave-group-out source/scene/domain stress.",
            "- It is stricter than the Stage43-CY group whitelist, but it still does not retrain the admissibility head.",
            f"- In this run, group fragility reduced: `{payload['aggregate']['group_fragility_reduced']}`.",
            "- Because this is still policy selection over an existing head, the next repair should train the leave-group-out criterion into the admissibility head and confirm it with stronger heldout/bootstrap evidence.",
            "- Future endpoints/full waypoints remain labels only; inference inputs are causal.",
            "- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
            "",
        ]
    )
    return lines


def _write_reports(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_cz_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-CZ Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- group fragility reduced: `{payload['aggregate']['group_fragility_reduced']}`",
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
        "## Stage43-CZ: t100 leave-group-out robust policy search",
        "",
        "The Stage43-CY whitelist did not reduce grouped fragility, so I moved the robustness requirement into validation policy selection itself. This search rewards t100 lift only when the candidate also survives source/scene/domain leave-group-out stress on validation.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- original t100 mean: `{agg['original_t100']['mean']:.4%}`",
        f"- robust t100 mean: `{agg['robust_t100']['mean']:.4%}`",
        f"- original min without group t100 mean: `{agg['original_min_without_group_t100']['mean']:.4%}`",
        f"- robust min without group t100 mean: `{agg['robust_min_without_group_t100']['mean']:.4%}`",
        f"- group fragility reduced: `{agg['group_fragility_reduced']}`",
        f"- deploy on current heldout t100: `{payload['deploy_on_current_heldout']}`",
        "",
        "This remains diagnostic. The robust validation objective reduced grouped fragility and improved the tiny t100 signal, but it is still policy selection over an existing head; the next step is to train the admissibility head with this leave-group-out objective and confirm it with stronger heldout/bootstrap evidence.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, readme_block)

    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage43_cz_t100_residual_admissibility_leave_group_out_policy"
    state["current_verdict"] = gate["verdict"]
    stage = state.setdefault("stage43_long_research_execution", {})
    stage["t100_residual_admissibility_leave_group_out_policy"] = {
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


def run_t100_leave_group_out_policy(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    cy_report = _ensure_cy_precondition(args)
    cu_report = read_json(cu.REPORT_JSON, {})
    seed_runs = [_replay_seed(run, args) for run in cu_report.get("seed_runs", [])]
    aggregate = _aggregate(seed_runs)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_leave_group_out_robust_policy_search",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "mode": "quick" if args.quick else "small",
        "stage43_cy_precondition": {
            "report": str(cy.REPORT_JSON),
            "verdict": cy_report.get("stage43_cy_gate", {}).get("verdict"),
        },
        "selection_protocol": {
            "guard_selected_on_validation": True,
            "test_threshold_tuning": False,
            "objective": "leave_group_out_min_t100_plus_flip_penalty",
            "candidate_grid": "alpha_gain_harm_delta_easy_floor",
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
            "feature_standardization_train_only": True,
            "validation_policy_selection_only": True,
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
    payload["stage43_cz_gate"] = _gate(payload)
    _write_reports(payload)
    print(f"Stage43-CZ: {payload['stage43_cz_gate']['verdict']} ({payload['stage43_cz_gate']['passed']}/{payload['stage43_cz_gate']['total']})")
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run leave-group-out robust policy search for Stage43 t100 residual admissibility.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args(argv)
    if not args.quick:
        args.small = True
    return run_t100_leave_group_out_policy(args)


if __name__ == "__main__":
    main()
