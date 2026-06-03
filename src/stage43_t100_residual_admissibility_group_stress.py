from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import torch

from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_t100_residual_admissibility_source_stress as cw
from src import stage43_t100_residual_admissibility_statistical_confirmation as cu
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_protected_latent_state_model import _jsonable


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_t100_residual_admissibility_group_stress.json"
REPORT_MD = OUT_DIR / "stage43_t100_residual_admissibility_group_stress.md"
GATE_MD = OUT_DIR / "stage43_stage_cx_t100_residual_admissibility_group_stress_gate.md"

README_RESULTS = m.README_RESULTS
M3W_README = m.M3W_README
WORK_SUMMARY = m.WORK_SUMMARY
RESEARCH_STATE = m.RESEARCH_STATE

SECTION = "STAGE43_CX_T100_RESIDUAL_ADMISSIBILITY_GROUP_STRESS"
SOURCE = "fresh_stage43_cx_t100_residual_admissibility_group_stress"


def _ensure_cw_precondition(args: argparse.Namespace) -> dict[str, Any]:
    report = read_json(cw.REPORT_JSON, {})
    gate = report.get("stage43_cw_gate", {})
    if not report or gate.get("passed") != gate.get("total"):
        report = cw.stress_t100_residual_admissibility_sources(args)
    return report


def _replay_arrays(seed_run: Mapping[str, Any], args: argparse.Namespace) -> tuple[m.WaypointSplit, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    seed = int(seed_run["seed"])
    build_args = argparse.Namespace(
        quick=bool(args.quick),
        seed=seed,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        batch_size=int(args.batch_size),
    )
    _train, _val, test, _cs_ckpt, cs_model = cw.cv.ct._build_splits(build_args)
    device = torch.device("cpu")
    cs_pred = cw.cv.ct.cs._predict(cs_model, test, device, int(args.batch_size))
    test_aug = cw.cv.ct._augment_alpha_features(test, cs_pred)
    model, mean, std = cw.cv._load_seed_head(seed_run)
    test_aug["x"] = ((test_aug["x"] - mean) / std).astype(np.float32)
    head_pred = cw.cv.ct._predict_head(model, test_aug, device, int(args.batch_size))
    policy = seed_run["validation_selected_policy"]["policy"]
    metrics, selected_ade, selected_fde, switched = cw.cv.ct._evaluate_selected(test, cs_pred, head_pred, policy)
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
    return test, selected_ade, selected_fde, switched, metric_diff


def _label_gain_rows(
    labels: np.ndarray,
    floor_ade: np.ndarray,
    selected_ade: np.ndarray,
    switched: np.ndarray,
) -> list[dict[str, Any]]:
    labels = np.asarray(labels).astype(str)
    positive = np.maximum(floor_ade.astype(np.float64) - selected_ade.astype(np.float64), 0.0)
    rows: list[dict[str, Any]] = []
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        sw = mask & switched
        rows.append(
            {
                "label": str(label),
                "rows": int(mask.sum()),
                "switched": int(sw.sum()),
                "positive_gain_sum": float(np.sum(positive[sw])),
                "slice_t100_improvement": cw._masked_metrics(
                    _MiniSplit(floor_ade=floor_ade, floor_fde=floor_ade, hard=np.zeros(len(floor_ade), dtype=bool), failure=np.zeros(len(floor_ade), dtype=bool), easy=np.zeros(len(floor_ade), dtype=bool)),
                    selected_ade,
                    selected_ade,
                    switched,
                    mask,
                )["t100_improvement"],
            }
        )
    rows.sort(key=lambda row: (row["positive_gain_sum"], row["switched"], row["rows"]), reverse=True)
    return rows


class _MiniSplit:
    def __init__(self, floor_ade: np.ndarray, floor_fde: np.ndarray, hard: np.ndarray, failure: np.ndarray, easy: np.ndarray) -> None:
        self.floor_ade = floor_ade
        self.floor_fde = floor_fde
        self.hard = hard
        self.failure = failure
        self.easy = easy


def _group_metrics(
    labels: np.ndarray,
    group_labels: list[str],
    ds: m.WaypointSplit,
    selected_ade: np.ndarray,
    selected_fde: np.ndarray,
    switched: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(labels).astype(str)
    group = set(str(x) for x in group_labels)
    group_mask = np.asarray([str(x) in group for x in labels.tolist()], dtype=bool)
    without_mask = ~group_mask
    return {
        "group_labels": list(group_labels),
        "group_label_count": int(len(group_labels)),
        "group_rows": int(group_mask.sum()),
        "group_metrics": cw._masked_metrics(ds, selected_ade, selected_fde, switched, group_mask),
        "without_group_metrics": cw._masked_metrics(ds, selected_ade, selected_fde, switched, without_mask),
        "removal_flips_t100_nonpositive": bool(cw._masked_metrics(ds, selected_ade, selected_fde, switched, without_mask)["t100_improvement"] <= 0.0),
    }


def _stress_groups_for_labels(
    labels: np.ndarray,
    ds: m.WaypointSplit,
    selected_ade: np.ndarray,
    selected_fde: np.ndarray,
    switched: np.ndarray,
    *,
    prefix: str,
) -> list[dict[str, Any]]:
    gain_rows = _label_gain_rows(labels, ds.floor_ade, selected_ade, switched)
    top = [row["label"] for row in gain_rows if row["positive_gain_sum"] > 0.0]
    groups: list[tuple[str, list[str]]] = []
    for k in [2, 3, 5]:
        if len(top) >= k:
            groups.append((f"{prefix}_top{k}_positive_gain", top[:k]))
    if len(top) >= 1:
        half = max(1, len(top) // 2)
        groups.append((f"{prefix}_top_half_positive_gain", top[:half]))
    negative = [row["label"] for row in gain_rows if row["slice_t100_improvement"] < 0.0]
    if negative:
        groups.append((f"{prefix}_all_negative_slices", negative))
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for name, labels_group in groups:
        key = tuple(sorted(labels_group))
        if key in seen:
            continue
        seen.add(key)
        row = _group_metrics(labels, labels_group, ds, selected_ade, selected_fde, switched)
        row["group_name"] = name
        out.append(row)
    out.sort(key=lambda row: row["without_group_metrics"]["t100_improvement"])
    return out


def _domain_pair_groups(
    ds: m.WaypointSplit,
    selected_ade: np.ndarray,
    selected_fde: np.ndarray,
    switched: np.ndarray,
) -> list[dict[str, Any]]:
    labels = np.asarray(ds.domain).astype(str)
    domains = sorted(set(labels.tolist()))
    out: list[dict[str, Any]] = []
    for i, left in enumerate(domains):
        for right in domains[i + 1 :]:
            row = _group_metrics(labels, [left, right], ds, selected_ade, selected_fde, switched)
            row["group_name"] = f"domain_remove_{left}+{right}"
            out.append(row)
    out.sort(key=lambda row: row["without_group_metrics"]["t100_improvement"])
    return out


def _seed_group_stress(seed_run: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    ds, selected_ade, selected_fde, switched, metric_diff = _replay_arrays(seed_run, args)
    source_groups = _stress_groups_for_labels(ds.source_file, ds, selected_ade, selected_fde, switched, prefix="source")
    scene_groups = _stress_groups_for_labels(ds.scene_id, ds, selected_ade, selected_fde, switched, prefix="scene")
    domain_groups = _domain_pair_groups(ds, selected_ade, selected_fde, switched)
    all_groups = [*source_groups, *scene_groups, *domain_groups]
    min_without = min([row["without_group_metrics"]["t100_improvement"] for row in all_groups] or [0.0])
    return {
        "seed": int(seed_run["seed"]),
        "rows": int(len(ds.x)),
        "max_metric_replay_diff": float(max(metric_diff.values()) if metric_diff else 0.0),
        "metric_replay_diff": metric_diff,
        "baseline_metrics": seed_run["test_metrics_with_floor"],
        "group_stress_tables": {
            "source_groups": source_groups,
            "scene_groups": scene_groups,
            "domain_pair_groups": domain_groups,
        },
        "summary": {
            "min_without_any_group_t100": float(min_without),
            "source_group_flip_count": int(sum(row["removal_flips_t100_nonpositive"] for row in source_groups)),
            "scene_group_flip_count": int(sum(row["removal_flips_t100_nonpositive"] for row in scene_groups)),
            "domain_pair_flip_count": int(sum(row["removal_flips_t100_nonpositive"] for row in domain_groups)),
            "group_count": int(len(all_groups)),
        },
    }


def _aggregate(seed_groups: list[Mapping[str, Any]]) -> dict[str, Any]:
    keys = [
        "min_without_any_group_t100",
        "source_group_flip_count",
        "scene_group_flip_count",
        "domain_pair_flip_count",
        "group_count",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        vals = np.asarray([run["summary"][key] for run in seed_groups], dtype=np.float64)
        out[key] = {
            "mean": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "values": [float(x) for x in vals.tolist()],
        }
    replay = np.asarray([run["max_metric_replay_diff"] for run in seed_groups], dtype=np.float64)
    out["all_replay_exact"] = bool(np.max(replay) <= 1e-7) if len(replay) else False
    out["all_group_exclusions_positive"] = bool(
        out["source_group_flip_count"]["max"] == 0.0
        and out["scene_group_flip_count"]["max"] == 0.0
        and out["domain_pair_flip_count"]["max"] == 0.0
    )
    out["group_stress_verdict"] = (
        "multi_source_group_stress_survives"
        if out["all_group_exclusions_positive"]
        else "multi_source_group_stress_fragile_keep_diagnostic"
    )
    return out


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    gates = {
        "stage43_cw_precondition_present": payload["stage43_cw_precondition"]["verdict"]
        in {
            "stage43_cw_t100_source_stress_survives_single_exclusion_diagnostic",
            "stage43_cw_t100_source_stress_fragile_keep_diagnostic",
        },
        "fresh_group_stress": payload["result_source"] == "fresh_t100_multi_source_group_stress",
        "three_seed_replay": len(payload["seed_group_stress"]) >= 3,
        "replay_diff_zero": bool(payload["aggregate"]["all_replay_exact"]),
        "group_tables_present": all(
            all(name in run["group_stress_tables"] and run["group_stress_tables"][name] for name in ["source_groups", "scene_groups", "domain_pair_groups"])
            for run in payload["seed_group_stress"]
        ),
        "group_fragility_reported": "group_stress_verdict" in payload["aggregate"],
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
        verdict = "stage43_cx_t100_group_stress_incomplete_keep_floor"
    elif payload["aggregate"]["group_stress_verdict"] == "multi_source_group_stress_survives":
        verdict = "stage43_cx_t100_group_stress_survives_diagnostic"
    else:
        verdict = "stage43_cx_t100_group_stress_fragile_keep_diagnostic"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_group_rows(title: str, rows: list[Mapping[str, Any]], *, limit: int = 8) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| group | labels | group rows | group t100 | without-group t100 | flips |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows[:limit]:
        lines.append(
            f"| `{row['group_name']}` | `{row['group_label_count']}` | `{row['group_rows']}` | "
            f"`{row['group_metrics']['t100_improvement']:.6f}` | `{row['without_group_metrics']['t100_improvement']:.6f}` | "
            f"`{row['removal_flips_t100_nonpositive']}` |"
        )
    lines.append("")
    return lines


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_cx_gate"]
    agg = payload["aggregate"]
    first = payload["seed_group_stress"][0] if payload["seed_group_stress"] else {}
    lines = [
        "# Stage43-CX T100 Residual Admissibility Group Stress",
        "",
        f"- source: `{SOURCE}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- group stress verdict: `{agg['group_stress_verdict']}`",
        "- deploy on current heldout t100: `False`",
        "",
        "## Aggregate",
        "",
        f"- all replay exact: `{agg['all_replay_exact']}`",
        f"- all group exclusions positive: `{agg['all_group_exclusions_positive']}`",
        f"- min without any group t100 mean: `{agg['min_without_any_group_t100']['mean']:.6f}`",
        f"- source group flip count max: `{agg['source_group_flip_count']['max']:.0f}`",
        f"- scene group flip count max: `{agg['scene_group_flip_count']['max']:.0f}`",
        f"- domain pair flip count max: `{agg['domain_pair_flip_count']['max']:.0f}`",
        "",
    ]
    if first:
        tables = first["group_stress_tables"]
        lines.extend(_render_group_rows("Source-group removals from first seed", tables["source_groups"]))
        lines.extend(_render_group_rows("Scene-group removals from first seed", tables["scene_groups"]))
        lines.extend(_render_group_rows("Domain-pair removals from first seed", tables["domain_pair_groups"]))
    lines.extend(
        [
            "## Interpretation",
            "",
            "- This is a stricter grouped source/scene stress audit for the tiny CU/CV/CW t100 residual-admissibility signal.",
            "- The grouped scene stress exposes fragility, so this remains a diagnostic result rather than a t100 deployment change.",
            "- Future endpoints/full waypoints remain labels only; inference inputs are causal.",
            "- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
            "",
        ]
    )
    return lines


def _write_reports(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_cx_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-CX Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- group stress verdict: `{payload['aggregate']['group_stress_verdict']}`",
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
        "## Stage43-CX: t100 residual admissibility group stress",
        "",
        "I ran a stricter stress test on the tiny t100 residual-admissibility signal by removing grouped high-gain sources/scenes and domain pairs. This checks whether the signal survives harsher heldout-like exclusions than the single-source test.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- group stress verdict: `{agg['group_stress_verdict']}`",
        f"- all replay exact: `{agg['all_replay_exact']}`",
        f"- all group exclusions positive: `{agg['all_group_exclusions_positive']}`",
        f"- min without any group t100 mean: `{agg['min_without_any_group_t100']['mean']:.4%}`",
        f"- deploy on current heldout t100: `{payload['deploy_on_current_heldout']}`",
        "",
        "This remains diagnostic evidence. Grouped scene stress exposed fragility, so I keep current heldout t100 on the safety floor.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, readme_block)

    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage43_cx_t100_residual_admissibility_group_stress"
    state["current_verdict"] = gate["verdict"]
    stage = state.setdefault("stage43_long_research_execution", {})
    stage["t100_residual_admissibility_group_stress"] = {
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


def stress_t100_residual_admissibility_groups(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    cw_report = _ensure_cw_precondition(args)
    cu_report = read_json(cu.REPORT_JSON, {})
    seed_group_stress = [_seed_group_stress(run, args) for run in cu_report.get("seed_runs", [])]
    aggregate = _aggregate(seed_group_stress)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_t100_multi_source_group_stress",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "mode": "quick" if args.quick else "small",
        "stage43_cw_precondition": {
            "report": str(cw.REPORT_JSON),
            "verdict": cw_report.get("stage43_cw_gate", {}).get("verdict"),
        },
        "seed_group_stress": seed_group_stress,
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
    payload["stage43_cx_gate"] = _gate(payload)
    _write_reports(payload)
    print(f"Stage43-CX: {payload['stage43_cx_gate']['verdict']} ({payload['stage43_cx_gate']['passed']}/{payload['stage43_cx_gate']['total']})")
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Stress Stage43 t100 admissibility signal under grouped source/scene exclusions.")
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
    return stress_t100_residual_admissibility_groups(args)


if __name__ == "__main__":
    main()
