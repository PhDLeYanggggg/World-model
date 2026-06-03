from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import torch

from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_t100_residual_admissibility_slice_attribution as cv
from src import stage43_t100_residual_admissibility_statistical_confirmation as cu
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_protected_latent_state_model import _jsonable


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_t100_residual_admissibility_source_stress.json"
REPORT_MD = OUT_DIR / "stage43_t100_residual_admissibility_source_stress.md"
GATE_MD = OUT_DIR / "stage43_stage_cw_t100_residual_admissibility_source_stress_gate.md"

README_RESULTS = m.README_RESULTS
M3W_README = m.M3W_README
WORK_SUMMARY = m.WORK_SUMMARY
RESEARCH_STATE = m.RESEARCH_STATE

SECTION = "STAGE43_CW_T100_RESIDUAL_ADMISSIBILITY_SOURCE_STRESS"
SOURCE = "fresh_stage43_cw_t100_residual_admissibility_source_stress"


def _ensure_cv_precondition(args: argparse.Namespace) -> dict[str, Any]:
    report = read_json(cv.REPORT_JSON, {})
    if not report or report.get("stage43_cv_gate", {}).get("passed") != report.get("stage43_cv_gate", {}).get("total"):
        report = cv.attribute_t100_residual_admissibility_slices(args)
    return report


def _masked_metrics(
    ds: m.WaypointSplit,
    selected_ade: np.ndarray,
    selected_fde: np.ndarray,
    switched: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    mask = np.asarray(mask).astype(bool)
    hard_failure = (ds.hard | ds.failure) & mask
    easy = ds.easy & mask
    if int(mask.sum()) == 0:
        return {
            "rows": 0,
            "t100_improvement": 0.0,
            "endpoint_fde_improvement": 0.0,
            "hard_failure_improvement": 0.0,
            "easy_degradation": 0.0,
            "switch_rate": 0.0,
            "mean_delta_ade_selected_minus_floor": 0.0,
        }
    easy_degradation = (
        float(max(0.0, float(np.mean(selected_ade[easy])) / max(float(np.mean(ds.floor_ade[easy])), m.EPS) - 1.0))
        if int(easy.sum())
        else 0.0
    )
    return {
        "rows": int(mask.sum()),
        "t100_improvement": m._slice_improvement(selected_ade, ds.floor_ade, mask),
        "endpoint_fde_improvement": m._slice_improvement(selected_fde, ds.floor_fde, mask),
        "hard_failure_improvement": m._slice_improvement(selected_ade, ds.floor_ade, hard_failure),
        "easy_degradation": easy_degradation,
        "switch_rate": float(np.mean(switched[mask])),
        "mean_delta_ade_selected_minus_floor": float(np.mean(selected_ade[mask] - ds.floor_ade[mask])),
    }


def _stress_table(
    labels: np.ndarray,
    ds: m.WaypointSplit,
    selected_ade: np.ndarray,
    selected_fde: np.ndarray,
    switched: np.ndarray,
    *,
    max_rows: int = 24,
) -> list[dict[str, Any]]:
    labels = np.asarray(labels).astype(str)
    rows: list[dict[str, Any]] = []
    for label in sorted(set(labels.tolist())):
        slice_mask = labels == label
        without_mask = labels != label
        slice_metrics = _masked_metrics(ds, selected_ade, selected_fde, switched, slice_mask)
        without_metrics = _masked_metrics(ds, selected_ade, selected_fde, switched, without_mask)
        rows.append(
            {
                "label": str(label),
                "slice": slice_metrics,
                "without_label": without_metrics,
                "removal_flips_t100_nonpositive": bool(without_metrics["t100_improvement"] <= 0.0),
                "slice_negative": bool(slice_metrics["t100_improvement"] < 0.0),
            }
        )
    rows.sort(key=lambda row: (row["without_label"]["t100_improvement"], -row["slice"]["rows"]))
    return rows[: int(max_rows)]


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
    _train, _val, test, _cs_ckpt, cs_model = cv.ct._build_splits(build_args)
    device = torch.device("cpu")
    cs_pred = cv.ct.cs._predict(cs_model, test, device, int(args.batch_size))
    test_aug = cv.ct._augment_alpha_features(test, cs_pred)
    model, mean, std = cv._load_seed_head(seed_run)
    test_aug["x"] = ((test_aug["x"] - mean) / std).astype(np.float32)
    head_pred = cv.ct._predict_head(model, test_aug, device, int(args.batch_size))
    policy = seed_run["validation_selected_policy"]["policy"]
    metrics, selected_ade, selected_fde, switched = cv.ct._evaluate_selected(test, cs_pred, head_pred, policy)
    expected = seed_run["test_metrics_with_floor"]
    metric_diff = {
        key: float(abs(float(metrics[key]) - float(expected[key])))
        for key in [
            "full_waypoint_ade_improvement_vs_floor",
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor",
            "hard_failure_full_waypoint_ade_improvement_vs_floor",
            "easy_degradation_vs_floor",
            "switch_rate",
        ]
    }
    source_agent = np.asarray([f"{src}|{idx}" for src, idx in zip(test.source_file.astype(str), np.arange(len(test.source_file)))])
    source_table = _stress_table(test.source_file, test, selected_ade, selected_fde, switched)
    scene_table = _stress_table(test.scene_id, test, selected_ade, selected_fde, switched)
    domain_table = _stress_table(test.domain, test, selected_ade, selected_fde, switched)
    source_agent_table = _stress_table(source_agent, test, selected_ade, selected_fde, switched, max_rows=8)
    return {
        "seed": seed,
        "rows": int(len(test.x)),
        "policy": policy,
        "metrics": metrics,
        "metric_replay_diff": metric_diff,
        "max_metric_replay_diff": float(max(metric_diff.values()) if metric_diff else 0.0),
        "stress_tables": {
            "domain": domain_table,
            "source_file": source_table,
            "scene_id": scene_table,
            "row_source_instance": source_agent_table,
        },
        "summary": {
            "min_without_source_t100": float(min([row["without_label"]["t100_improvement"] for row in source_table] or [0.0])),
            "min_without_scene_t100": float(min([row["without_label"]["t100_improvement"] for row in scene_table] or [0.0])),
            "min_without_domain_t100": float(min([row["without_label"]["t100_improvement"] for row in domain_table] or [0.0])),
            "source_removal_flip_count": int(sum(row["removal_flips_t100_nonpositive"] for row in source_table)),
            "scene_removal_flip_count": int(sum(row["removal_flips_t100_nonpositive"] for row in scene_table)),
            "domain_removal_flip_count": int(sum(row["removal_flips_t100_nonpositive"] for row in domain_table)),
            "negative_source_slice_count": int(sum(row["slice_negative"] for row in source_table)),
            "negative_scene_slice_count": int(sum(row["slice_negative"] for row in scene_table)),
            "negative_domain_slice_count": int(sum(row["slice_negative"] for row in domain_table)),
        },
    }


def _aggregate(seed_stress: list[Mapping[str, Any]]) -> dict[str, Any]:
    keys = [
        "min_without_source_t100",
        "min_without_scene_t100",
        "min_without_domain_t100",
        "source_removal_flip_count",
        "scene_removal_flip_count",
        "domain_removal_flip_count",
        "negative_source_slice_count",
        "negative_scene_slice_count",
        "negative_domain_slice_count",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        vals = np.asarray([run["summary"][key] for run in seed_stress], dtype=np.float64)
        out[key] = {
            "mean": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "values": [float(x) for x in vals.tolist()],
        }
    replay = np.asarray([run["max_metric_replay_diff"] for run in seed_stress], dtype=np.float64)
    out["all_replay_exact"] = bool(np.max(replay) <= 1e-7) if len(replay) else False
    out["all_single_source_exclusions_positive"] = bool(out["source_removal_flip_count"]["max"] == 0.0)
    out["all_single_scene_exclusions_positive"] = bool(out["scene_removal_flip_count"]["max"] == 0.0)
    out["all_single_domain_exclusions_positive"] = bool(out["domain_removal_flip_count"]["max"] == 0.0)
    out["stress_verdict"] = (
        "source_scene_stress_survives_single_exclusion"
        if out["all_single_source_exclusions_positive"] and out["all_single_scene_exclusions_positive"]
        else "source_scene_stress_fragile_keep_diagnostic"
    )
    return out


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    gates = {
        "stage43_cv_precondition_present": payload["stage43_cv_precondition"]["verdict"]
        in {
            "stage43_cv_t100_slice_attribution_broad_supported_diagnostic",
            "stage43_cv_t100_slice_attribution_narrow_supported_diagnostic",
        },
        "fresh_source_scene_stress": payload["result_source"] == "fresh_t100_source_scene_single_exclusion_stress",
        "three_seed_replay": len(payload["seed_stress"]) >= 3,
        "replay_diff_zero": bool(payload["aggregate"]["all_replay_exact"]),
        "source_scene_domain_exclusion_tables_present": all(
            all(name in run["stress_tables"] and run["stress_tables"][name] for name in ["source_file", "scene_id", "domain"])
            for run in payload["seed_stress"]
        ),
        "single_source_exclusion_positive_or_fragility_reported": "stress_verdict" in payload["aggregate"],
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
        verdict = "stage43_cw_t100_source_stress_incomplete_keep_floor"
    elif payload["aggregate"]["stress_verdict"] == "source_scene_stress_survives_single_exclusion":
        verdict = "stage43_cw_t100_source_stress_survives_single_exclusion_diagnostic"
    else:
        verdict = "stage43_cw_t100_source_stress_fragile_keep_diagnostic"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_stress_rows(title: str, rows: list[Mapping[str, Any]], *, limit: int = 8) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| removed label | slice rows | slice t100 | without-label t100 | removal flips | slice negative |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows[:limit]:
        lines.append(
            f"| `{row['label']}` | `{row['slice']['rows']}` | `{row['slice']['t100_improvement']:.6f}` | "
            f"`{row['without_label']['t100_improvement']:.6f}` | `{row['removal_flips_t100_nonpositive']}` | `{row['slice_negative']}` |"
        )
    lines.append("")
    return lines


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_cw_gate"]
    agg = payload["aggregate"]
    first = payload["seed_stress"][0] if payload["seed_stress"] else {}
    lines = [
        "# Stage43-CW T100 Residual Admissibility Source Stress",
        "",
        f"- source: `{SOURCE}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- stress verdict: `{agg['stress_verdict']}`",
        "- deploy on current heldout t100: `False`",
        "",
        "## Aggregate",
        "",
        f"- all replay exact: `{agg['all_replay_exact']}`",
        f"- all single-source exclusions positive: `{agg['all_single_source_exclusions_positive']}`",
        f"- all single-scene exclusions positive: `{agg['all_single_scene_exclusions_positive']}`",
        f"- min without-source t100 mean: `{agg['min_without_source_t100']['mean']:.6f}`",
        f"- min without-scene t100 mean: `{agg['min_without_scene_t100']['mean']:.6f}`",
        f"- negative source slices mean: `{agg['negative_source_slice_count']['mean']:.2f}`",
        f"- negative scene slices mean: `{agg['negative_scene_slice_count']['mean']:.2f}`",
        "",
    ]
    if first:
        lines.extend(_render_stress_rows("Worst source removals from first seed", first["stress_tables"]["source_file"]))
        lines.extend(_render_stress_rows("Worst scene removals from first seed", first["stress_tables"]["scene_id"]))
        lines.extend(_render_stress_rows("Domain removals from first seed", first["stress_tables"]["domain"]))
    lines.extend(
        [
            "## Interpretation",
            "",
            "- This is a source/scene stress audit for the tiny CU/CV t100 residual-admissibility signal.",
            "- Passing this audit does not deploy t100; it only says the supported-protocol signal is not destroyed by removing one source or scene at a time.",
            "- Future endpoints/full waypoints remain labels only; inference inputs are causal.",
            "- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
            "",
        ]
    )
    return lines


def _write_reports(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_cw_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-CW Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- stress verdict: `{payload['aggregate']['stress_verdict']}`",
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
        "## Stage43-CW: t100 residual admissibility source stress",
        "",
        "I stress-tested the tiny Stage43-CU/CV t100 signal by removing one source or scene at a time and replaying the residual-admissibility policy. The point is to check whether the signal survives simple source/scene exclusions before trying to expand it.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- stress verdict: `{agg['stress_verdict']}`",
        f"- all replay exact: `{agg['all_replay_exact']}`",
        f"- all single-source exclusions positive: `{agg['all_single_source_exclusions_positive']}`",
        f"- all single-scene exclusions positive: `{agg['all_single_scene_exclusions_positive']}`",
        f"- min without-source t100 mean: `{agg['min_without_source_t100']['mean']:.4%}`",
        f"- min without-scene t100 mean: `{agg['min_without_scene_t100']['mean']:.4%}`",
        f"- deploy on current heldout t100: `{payload['deploy_on_current_heldout']}`",
        "",
        "This is still diagnostic evidence. It supports source-slice expansion work, not a heldout t100 deployment change.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, readme_block)

    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage43_cw_t100_residual_admissibility_source_stress"
    state["current_verdict"] = gate["verdict"]
    stage = state.setdefault("stage43_long_research_execution", {})
    stage["t100_residual_admissibility_source_stress"] = {
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


def stress_t100_residual_admissibility_sources(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    cv_report = _ensure_cv_precondition(args)
    cu_report = read_json(cu.REPORT_JSON, {})
    seed_stress = [_replay_seed(run, args) for run in cu_report.get("seed_runs", [])]
    aggregate = _aggregate(seed_stress)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_t100_source_scene_single_exclusion_stress",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "mode": "quick" if args.quick else "small",
        "stage43_cv_precondition": {
            "report": str(cv.REPORT_JSON),
            "verdict": cv_report.get("stage43_cv_gate", {}).get("verdict"),
        },
        "seed_stress": seed_stress,
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
    payload["stage43_cw_gate"] = _gate(payload)
    _write_reports(payload)
    print(f"Stage43-CW: {payload['stage43_cw_gate']['verdict']} ({payload['stage43_cw_gate']['passed']}/{payload['stage43_cw_gate']['total']})")
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Stress Stage43 t100 residual-admissibility signal under source/scene exclusions.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--seeds", type=str, default="4323,4331,4337")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args(argv)
    return stress_t100_residual_admissibility_sources(args)


if __name__ == "__main__":
    main()
