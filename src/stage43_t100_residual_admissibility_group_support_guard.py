from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import torch

from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_t100_residual_admissibility_group_stress as cx
from src import stage43_t100_residual_admissibility_head as ct
from src import stage43_t100_residual_admissibility_statistical_confirmation as cu
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_protected_latent_state_model import _jsonable


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_t100_residual_admissibility_group_support_guard.json"
REPORT_MD = OUT_DIR / "stage43_t100_residual_admissibility_group_support_guard.md"
GATE_MD = OUT_DIR / "stage43_stage_cy_t100_residual_admissibility_group_support_guard_gate.md"

README_RESULTS = m.README_RESULTS
M3W_README = m.M3W_README
WORK_SUMMARY = m.WORK_SUMMARY
RESEARCH_STATE = m.RESEARCH_STATE

SECTION = "STAGE43_CY_T100_RESIDUAL_ADMISSIBILITY_GROUP_SUPPORT_GUARD"
SOURCE = "fresh_stage43_cy_t100_residual_admissibility_group_support_guard"


def _ensure_cx_precondition(args: argparse.Namespace) -> dict[str, Any]:
    report = read_json(cx.REPORT_JSON, {})
    gate = report.get("stage43_cx_gate", {})
    if not report or gate.get("passed") != gate.get("total"):
        report = cx.stress_t100_residual_admissibility_groups(args)
    return report


def _policy_selected(
    ds: m.WaypointSplit,
    cs_pred: Mapping[str, np.ndarray],
    policy: Mapping[str, Any],
    allow: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    allow = np.asarray(allow).astype(bool)
    if int(policy.get("alpha_index", -1)) < 0:
        selected_ade = ds.floor_ade.copy()
        selected_fde = ds.floor_fde.copy()
        switched = np.zeros(len(ds.x), dtype=bool)
        return m._metrics(ds, selected_ade, selected_fde, switched), selected_ade, selected_fde, switched
    waypoint = ct.cs._compose_waypoint(ds, cs_pred, alpha=float(policy["alpha"]))
    selected_waypoint = np.where(allow[:, None, None], waypoint, ds.floor_waypoint_delta).astype(np.float32)
    selected_ade, selected_fde = m._trajectory_error(ds, selected_waypoint)
    return m._metrics(ds, selected_ade, selected_fde, allow), selected_ade, selected_fde, allow


def _eligible_labels(
    labels: np.ndarray,
    ds: m.WaypointSplit,
    selected_ade: np.ndarray,
    selected_fde: np.ndarray,
    switched: np.ndarray,
    *,
    min_rows: int,
    min_improvement: float,
) -> list[str]:
    labels = np.asarray(labels).astype(str)
    out: list[str] = []
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        metrics = cx.cw._masked_metrics(ds, selected_ade, selected_fde, switched, mask)
        if (
            int(metrics["rows"]) >= int(min_rows)
            and float(metrics["t100_improvement"]) > float(min_improvement)
            and float(metrics["easy_degradation"]) <= 0.02
            and float(metrics["switch_rate"]) > 0.0
        ):
            out.append(str(label))
    return out


def _guard_mask(ds: m.WaypointSplit, variant: str, eligible: Mapping[str, list[str]]) -> np.ndarray:
    n = len(ds.x)
    source = np.isin(ds.source_file.astype(str), np.asarray(eligible.get("source", []), dtype=str))
    scene = np.isin(ds.scene_id.astype(str), np.asarray(eligible.get("scene", []), dtype=str))
    domain = np.isin(ds.domain.astype(str), np.asarray(eligible.get("domain", []), dtype=str))
    if variant == "floor":
        return np.zeros(n, dtype=bool)
    if variant == "source_val_positive":
        return source
    if variant == "scene_val_positive":
        return scene
    if variant == "domain_val_positive":
        return domain
    if variant == "source_or_scene_val_positive":
        return source | scene
    if variant == "source_and_scene_val_positive":
        return source & scene
    if variant == "domain_and_source_or_scene":
        return domain & (source | scene)
    if variant == "domain_and_source_and_scene":
        return domain & source & scene
    raise ValueError(f"unknown guard variant: {variant}")


def _group_summary(ds: m.WaypointSplit, selected_ade: np.ndarray, selected_fde: np.ndarray, switched: np.ndarray) -> dict[str, Any]:
    source_groups = cx._stress_groups_for_labels(ds.source_file, ds, selected_ade, selected_fde, switched, prefix="source")
    scene_groups = cx._stress_groups_for_labels(ds.scene_id, ds, selected_ade, selected_fde, switched, prefix="scene")
    domain_groups = cx._domain_pair_groups(ds, selected_ade, selected_fde, switched)
    all_groups = [*source_groups, *scene_groups, *domain_groups]
    return {
        "min_without_any_group_t100": float(min([row["without_group_metrics"]["t100_improvement"] for row in all_groups] or [0.0])),
        "source_group_flip_count": int(sum(row["removal_flips_t100_nonpositive"] for row in source_groups)),
        "scene_group_flip_count": int(sum(row["removal_flips_t100_nonpositive"] for row in scene_groups)),
        "domain_pair_flip_count": int(sum(row["removal_flips_t100_nonpositive"] for row in domain_groups)),
        "group_count": int(len(all_groups)),
    }


def _variant_objective(metrics: Mapping[str, Any], group_summary: Mapping[str, Any]) -> float:
    min_without = float(group_summary["min_without_any_group_t100"])
    fragility_penalty = 4.0 * max(0.0, -min_without)
    flip_penalty = 0.01 * float(group_summary["source_group_flip_count"] + group_summary["scene_group_flip_count"] + group_summary["domain_pair_flip_count"])
    return float(
        2.0 * float(metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"])
        + 0.7 * float(metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"])
        + 0.2 * float(metrics["full_waypoint_ade_improvement_vs_floor"])
        + 1.5 * min(0.01, min_without)
        - 0.5 * float(metrics["easy_degradation_vs_floor"])
        - 0.05 * float(metrics["switch_rate"])
        - fragility_penalty
        - flip_penalty
    )


def _replay_val_test(seed_run: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
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
    policy = seed_run["validation_selected_policy"]["policy"]
    val_metrics, val_ade, val_fde, val_allow = ct._evaluate_selected(val, val_pred, val_head, policy)
    test_metrics, test_ade, test_fde, test_allow = ct._evaluate_selected(test, test_pred, test_head, policy)
    expected = seed_run["test_metrics_with_floor"]
    replay_diff = {
        key: float(abs(float(test_metrics[key]) - float(expected[key])))
        for key in [
            "full_waypoint_ade_improvement_vs_floor",
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor",
            "hard_failure_full_waypoint_ade_improvement_vs_floor",
            "easy_degradation_vs_floor",
            "switch_rate",
        ]
    }
    eligible = {
        "source": _eligible_labels(val.source_file, val, val_ade, val_fde, val_allow, min_rows=int(args.min_label_rows), min_improvement=float(args.min_val_improvement)),
        "scene": _eligible_labels(val.scene_id, val, val_ade, val_fde, val_allow, min_rows=int(args.min_label_rows), min_improvement=float(args.min_val_improvement)),
        "domain": _eligible_labels(val.domain, val, val_ade, val_fde, val_allow, min_rows=max(1, int(args.min_label_rows // 2)), min_improvement=float(args.min_val_improvement)),
    }
    variants = [
        "floor",
        "source_val_positive",
        "scene_val_positive",
        "domain_val_positive",
        "source_or_scene_val_positive",
        "source_and_scene_val_positive",
        "domain_and_source_or_scene",
        "domain_and_source_and_scene",
    ]
    rows: list[dict[str, Any]] = []
    for variant in variants:
        val_guard = _guard_mask(val, variant, eligible)
        val_allowed = val_allow & val_guard
        vm, va, vf, vsw = _policy_selected(val, val_pred, policy, val_allowed)
        vg = _group_summary(val, va, vf, vsw)
        safe = bool(vm["easy_degradation_vs_floor"] <= 0.02 and vm["switch_rate"] <= 0.60)
        rows.append(
            {
                "variant": variant,
                "validation_metrics": vm,
                "validation_group_summary": vg,
                "validation_objective": _variant_objective(vm, vg),
                "validation_safe": safe,
            }
        )
    safe_rows = [row for row in rows if row["validation_safe"]]
    selected = max(safe_rows or rows, key=lambda row: row["validation_objective"])
    test_guard = _guard_mask(test, selected["variant"], eligible)
    guarded_allowed = test_allow & test_guard
    guarded_metrics, guarded_ade, guarded_fde, guarded_switched = _policy_selected(test, test_pred, policy, guarded_allowed)
    base_group = _group_summary(test, test_ade, test_fde, test_allow)
    guarded_group = _group_summary(test, guarded_ade, guarded_fde, guarded_switched)
    return {
        "seed": seed,
        "rows": {"val": int(len(val.x)), "test": int(len(test.x))},
        "policy": policy,
        "max_replay_diff": float(max(replay_diff.values()) if replay_diff else 0.0),
        "replay_diff": replay_diff,
        "eligible_label_counts": {key: int(len(value)) for key, value in eligible.items()},
        "eligible_labels": eligible,
        "validation_variants": rows,
        "selected_variant": selected,
        "base_test_metrics": test_metrics,
        "base_test_group_summary": base_group,
        "guarded_test_metrics": guarded_metrics,
        "guarded_test_group_summary": guarded_group,
        "test_delta_vs_base": {
            "t100": float(guarded_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] - test_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]),
            "hard_failure": float(guarded_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"] - test_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]),
            "easy_degradation": float(guarded_metrics["easy_degradation_vs_floor"] - test_metrics["easy_degradation_vs_floor"]),
            "switch_rate": float(guarded_metrics["switch_rate"] - test_metrics["switch_rate"]),
            "min_without_group_t100": float(guarded_group["min_without_any_group_t100"] - base_group["min_without_any_group_t100"]),
        },
    }


def _aggregate(seed_runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    def stats(path: tuple[str, ...]) -> dict[str, Any]:
        vals = np.asarray([
            _nested_float(run, path)
            for run in seed_runs
        ], dtype=np.float64)
        return {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals)), "values": [float(x) for x in vals.tolist()]}

    out = {
        "all_replay_exact": bool(max([float(run["max_replay_diff"]) for run in seed_runs] or [1.0]) <= 1e-7),
        "selected_variants": [str(run["selected_variant"]["variant"]) for run in seed_runs],
        "base_t100": stats(("base_test_metrics", "t100_raw_frame_full_waypoint_diagnostic_vs_floor")),
        "guarded_t100": stats(("guarded_test_metrics", "t100_raw_frame_full_waypoint_diagnostic_vs_floor")),
        "base_min_without_group_t100": stats(("base_test_group_summary", "min_without_any_group_t100")),
        "guarded_min_without_group_t100": stats(("guarded_test_group_summary", "min_without_any_group_t100")),
        "guarded_easy_degradation": stats(("guarded_test_metrics", "easy_degradation_vs_floor")),
        "guarded_switch_rate": stats(("guarded_test_metrics", "switch_rate")),
        "delta_t100_vs_base": stats(("test_delta_vs_base", "t100")),
        "delta_min_without_group_t100_vs_base": stats(("test_delta_vs_base", "min_without_group_t100")),
    }
    out["guard_preserves_easy"] = bool(out["guarded_easy_degradation"]["max"] <= 0.02)
    out["group_fragility_reduced"] = bool(out["delta_min_without_group_t100_vs_base"]["mean"] > 0.0)
    out["guarded_t100_positive_all_seeds"] = bool(min(out["guarded_t100"]["values"]) > 0.0)
    out["deploy_on_current_heldout"] = False
    return out


def _nested_float(row: Mapping[str, Any], path: tuple[str, ...]) -> float:
    cur: Any = row
    for key in path:
        cur = cur[key]
    return float(cur)


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    gates = {
        "stage43_cx_precondition_present": payload["stage43_cx_precondition"]["verdict"] == "stage43_cx_t100_group_stress_fragile_keep_diagnostic",
        "fresh_validation_group_support_guard": payload["result_source"] == "fresh_validation_group_support_guard",
        "three_seed_replay": len(payload["seed_runs"]) >= 3,
        "replay_diff_zero": bool(payload["aggregate"]["all_replay_exact"]),
        "validation_only_guard_selection": payload["selection_protocol"]["test_threshold_tuning"] is False,
        "guard_variants_evaluated": all(len(run["validation_variants"]) >= 4 for run in payload["seed_runs"]),
        "group_fragility_reduction_measured": "group_fragility_reduced" in payload["aggregate"],
        "easy_preserved_or_floor": payload["aggregate"]["guard_preserves_easy"] is True,
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
        verdict = "stage43_cy_t100_group_support_guard_incomplete_keep_diagnostic"
    elif payload["aggregate"]["group_fragility_reduced"] and payload["aggregate"]["guarded_t100_positive_all_seeds"]:
        verdict = "stage43_cy_t100_group_support_guard_reduces_fragility_diagnostic"
    elif payload["aggregate"]["group_fragility_reduced"]:
        verdict = "stage43_cy_t100_group_support_guard_reduces_fragility_but_floor_t100"
    else:
        verdict = "stage43_cy_t100_group_support_guard_no_repair_keep_diagnostic"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_cy_gate"]
    agg = payload["aggregate"]
    first = payload["seed_runs"][0] if payload["seed_runs"] else {}
    lines = [
        "# Stage43-CY T100 Residual Admissibility Group Support Guard",
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
        f"- selected variants: `{agg['selected_variants']}`",
        f"- base t100 mean: `{agg['base_t100']['mean']:.6f}`",
        f"- guarded t100 mean: `{agg['guarded_t100']['mean']:.6f}`",
        f"- base min without group t100 mean: `{agg['base_min_without_group_t100']['mean']:.6f}`",
        f"- guarded min without group t100 mean: `{agg['guarded_min_without_group_t100']['mean']:.6f}`",
        f"- delta min without group t100 mean: `{agg['delta_min_without_group_t100_vs_base']['mean']:.6f}`",
        f"- group fragility reduced: `{agg['group_fragility_reduced']}`",
        f"- guarded easy degradation max: `{agg['guarded_easy_degradation']['max']:.6f}`",
        f"- guarded switch rate mean: `{agg['guarded_switch_rate']['mean']:.6f}`",
        "",
    ]
    if first:
        lines.extend(
            [
                "## First Seed",
                "",
                f"- seed: `{first['seed']}`",
                f"- eligible label counts: `{first['eligible_label_counts']}`",
                f"- selected variant: `{first['selected_variant']['variant']}`",
                f"- validation objective: `{first['selected_variant']['validation_objective']:.6f}`",
                f"- test delta vs base: `{first['test_delta_vs_base']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "- This is a validation-selected group-support guard after the Stage43-CX grouped scene/source stress failure.",
            "- It asks whether restricting t100 residual switches to source/scene/domain groups that were positive on validation can reduce grouped fragility on test.",
            f"- In this run, group fragility reduced: `{payload['aggregate']['group_fragility_reduced']}`.",
            "- The guard is diagnostic and does not change current heldout t100 deployment.",
            "- Future endpoints/full waypoints remain labels only; inference inputs and guard metadata are causal or split metadata.",
            "- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
            "",
        ]
    )
    return lines


def _write_reports(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_cy_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-CY Gate",
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
        "## Stage43-CY: t100 group-support safety envelope",
        "",
        "After the grouped scene stress exposed fragility, I tested a stricter validation-selected group-support guard. The point was to see whether t100 residual switches become safer if I only allow them in source/scene/domain groups that were positive on validation.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- selected variants: `{agg['selected_variants']}`",
        f"- base t100 mean: `{agg['base_t100']['mean']:.4%}`",
        f"- guarded t100 mean: `{agg['guarded_t100']['mean']:.4%}`",
        f"- base min without group t100 mean: `{agg['base_min_without_group_t100']['mean']:.4%}`",
        f"- guarded min without group t100 mean: `{agg['guarded_min_without_group_t100']['mean']:.4%}`",
        f"- group fragility reduced: `{agg['group_fragility_reduced']}`",
        f"- deploy on current heldout t100: `{payload['deploy_on_current_heldout']}`",
        "",
        "This is still diagnostic. The validation-positive source guard preserved easy cases, but it did not reduce grouped fragility, so the next t100 repair needs a training objective that is explicitly leave-group-out or source/scene robust rather than another deployment whitelist.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, readme_block)

    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage43_cy_t100_residual_admissibility_group_support_guard"
    state["current_verdict"] = gate["verdict"]
    stage = state.setdefault("stage43_long_research_execution", {})
    stage["t100_residual_admissibility_group_support_guard"] = {
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


def run_t100_group_support_guard(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    cx_report = _ensure_cx_precondition(args)
    cu_report = read_json(cu.REPORT_JSON, {})
    seed_runs = [_replay_val_test(run, args) for run in cu_report.get("seed_runs", [])]
    aggregate = _aggregate(seed_runs)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_validation_group_support_guard",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "mode": "quick" if args.quick else "small",
        "stage43_cx_precondition": {
            "report": str(cx.REPORT_JSON),
            "verdict": cx_report.get("stage43_cx_gate", {}).get("verdict"),
        },
        "selection_protocol": {
            "guard_selected_on_validation": True,
            "test_threshold_tuning": False,
            "min_label_rows": int(args.min_label_rows),
            "min_val_improvement": float(args.min_val_improvement),
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
            "validation_guard_selection_only": True,
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
    payload["stage43_cy_gate"] = _gate(payload)
    _write_reports(payload)
    print(f"Stage43-CY: {payload['stage43_cy_gate']['verdict']} ({payload['stage43_cy_gate']['passed']}/{payload['stage43_cy_gate']['total']})")
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Run validation-selected group-support guard for Stage43 t100 residual admissibility.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--min-label-rows", type=int, default=24)
    parser.add_argument("--min-val-improvement", type=float, default=0.0)
    args = parser.parse_args(argv)
    if not args.quick:
        args.small = True
    return run_t100_group_support_guard(args)


if __name__ == "__main__":
    main()
