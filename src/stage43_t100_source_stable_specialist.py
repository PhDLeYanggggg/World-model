from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import fields
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_full_waypoint_latent_dynamics import (
    OUT_DIR,
    README_RESULTS,
    RESEARCH_STATE,
    WORK_SUMMARY,
    M3W_README,
    WaypointSplit,
    _bootstrap_ci,
    _build_split,
    _git_commit,
    _jsonable,
    _metrics,
    _trajectory_error,
)
from src.stage43_full_waypoint_latent_robustness_audit import _pct
from src.stage43_t100_source_coverage_preflight import _short_source
from src.stage43_tail_horizon_waypoint_adapter import (
    _easy_degradation,
    _model_hash,
    _predict_waypoint,
    _ridge_fit,
    _slice_improvement,
    _standardize,
    _target_matrix,
)


REPORT_JSON = OUT_DIR / "stage43_t100_source_stable_specialist.json"
REPORT_MD = OUT_DIR / "stage43_t100_source_stable_specialist.md"
GATE_MD = OUT_DIR / "stage43_stage_t_t100_source_stable_specialist_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

STAGE43_S_JSON = OUT_DIR / "stage43_t100_source_coverage_preflight.json"
SECTION = "STAGE43_T_T100_SOURCE_STABLE_SPECIALIST"
SOURCE = "fresh_stage43_t_t100_source_stable_specialist"
FAMILY = "TrajNet_crowds"


def _concat_splits(splits: list[WaypointSplit]) -> WaypointSplit:
    values: dict[str, Any] = {}
    for field in fields(WaypointSplit):
        name = field.name
        parts = [getattr(ds, name) for ds in splits]
        if name == "split":
            values[name] = "source_pool"
        elif name == "feature_names":
            values[name] = parts[0]
        else:
            values[name] = np.concatenate(parts, axis=0)
    return WaypointSplit(**values)


def _subset(ds: WaypointSplit, source_names: list[str], split: str, *, horizon: int = 100) -> WaypointSplit:
    short = np.asarray([_short_source(value) for value in ds.source_file])
    mask = np.isin(short, np.asarray(source_names, dtype=str)) & (ds.horizon.astype(np.int64) == int(horizon))
    values: dict[str, Any] = {}
    for field in fields(WaypointSplit):
        name = field.name
        value = getattr(ds, name)
        if name == "split":
            values[name] = split
        elif name == "feature_names":
            values[name] = value
        else:
            values[name] = value[mask]
    return WaypointSplit(**values)


def _source_table(ds: WaypointSplit, candidate_ade: np.ndarray | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    short = np.asarray([_short_source(value) for value in ds.source_file])
    for source in sorted(set(short.tolist())):
        mask = short == source
        row = {
            "rows": int(mask.sum()),
            "mean_floor_ade": float(np.mean(ds.floor_ade[mask])) if int(mask.sum()) else 0.0,
            "easy_rows": int(ds.easy[mask].sum()),
            "hard_failure_rows": int((ds.hard[mask] | ds.failure[mask]).sum()),
        }
        if candidate_ade is not None and int(mask.sum()):
            row["full_waypoint_ade_improvement_vs_floor"] = _slice_improvement(candidate_ade, ds.floor_ade, mask)
            row["easy_degradation_vs_floor"] = _easy_degradation(ds, candidate_ade, mask)
        out[source] = row
    return out


def _candidate_eval(
    train: WaypointSplit,
    val: WaypointSplit,
    *,
    target: str,
    l2: float,
    max_easy_degradation: float,
) -> dict[str, Any]:
    weight = _ridge_fit(train.x, _target_matrix(train, target), float(l2))
    pred = _predict_waypoint(val, weight, target)
    ade, fde = _trajectory_error(val, pred)
    switch = np.ones(len(val.x), dtype=bool)
    metrics = _metrics(val, ade, fde, switch)
    improvement = metrics["full_waypoint_ade_improvement_vs_floor"]
    easy = metrics["easy_degradation_vs_floor"]
    source_table = _source_table(val, ade)
    source_rows = list(source_table.values())
    min_source_improvement = min(
        [float(row.get("full_waypoint_ade_improvement_vs_floor", 0.0)) for row in source_rows],
        default=0.0,
    )
    max_source_easy = max(
        [float(row.get("easy_degradation_vs_floor", 0.0)) for row in source_rows],
        default=0.0,
    )
    source_safe = min_source_improvement > 0.0 and max_source_easy <= float(max_easy_degradation)
    objective = float(
        improvement
        + 0.5 * min_source_improvement
        - 20.0 * max(0.0, easy - float(max_easy_degradation))
        - 50.0 * max(0.0, max_source_easy - float(max_easy_degradation))
        - 10.0 * max(0.0, -min_source_improvement)
    )
    return {
        "target": target,
        "l2": float(l2),
        "weight": weight,
        "model_hash": _model_hash(weight, l2=float(l2), target=target, train_filter="source_stable_h100"),
        "validation_metrics": metrics,
        "validation_source_table": source_table,
        "validation_source_safety": {
            "min_source_improvement": float(min_source_improvement),
            "max_source_easy_degradation": float(max_source_easy),
            "source_safe": bool(source_safe),
        },
        "objective": objective,
    }


def run_t100_source_stable_specialist(
    *,
    seed: int = 461,
    l2_grid: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0),
    target_grid: tuple[str, ...] = ("residual", "direct"),
    max_easy_degradation: float = 0.02,
    bootstrap: int = 1000,
) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage43s = read_json(STAGE43_S_JSON, {})
    proposal = stage43s.get("family_summary", {}).get(FAMILY, {}).get("source_level_split_proposal", {})
    old_splits = [_build_split(split, max_rows=None, seed=int(seed)) for split in ["train", "val", "test"]]
    pool = _concat_splits(old_splits)
    train = _subset(pool, proposal.get("train_sources", []), "train", horizon=100)
    val = _subset(pool, proposal.get("val_sources", []), "val", horizon=100)
    test = _subset(pool, proposal.get("test_sources", []), "test", horizon=100)
    feature_mean, feature_std = _standardize(train, val, test)
    candidates = [
        _candidate_eval(train, val, target=target, l2=l2, max_easy_degradation=float(max_easy_degradation))
        for target in target_grid
        for l2 in l2_grid
    ]
    candidates.sort(key=lambda row: row["objective"], reverse=True)
    best = candidates[0]
    pred = _predict_waypoint(test, best["weight"], best["target"])
    candidate_ade, candidate_fde = _trajectory_error(test, pred)
    switch = np.ones(len(test.x), dtype=bool)
    candidate_metrics = _metrics(test, candidate_ade, candidate_fde, switch)
    bootstrap_ci = _bootstrap_ci(test, candidate_ade, candidate_fde, n=int(bootstrap), seed=int(seed) + 4600)
    easy_safe = candidate_metrics["easy_degradation_vs_floor"] <= float(max_easy_degradation)
    validation_source_safe = bool(best["validation_source_safety"]["source_safe"])
    positive = candidate_metrics["full_waypoint_ade_improvement_vs_floor"] > 0.0
    deploy = bool(positive and easy_safe and validation_source_safe)
    deployment_ade = candidate_ade if deploy else test.floor_ade
    deployment_fde = candidate_fde if deploy else test.floor_fde
    deployment_switch = switch if deploy else np.zeros(len(test.x), dtype=bool)
    deployment_metrics = _metrics(test, deployment_ade, deployment_fde, deployment_switch)
    candidate_rows = []
    for row in candidates:
        candidate_rows.append(
            {
                "target": row["target"],
                "l2": row["l2"],
                "model_hash": row["model_hash"],
                "objective": row["objective"],
                "validation_metrics": row["validation_metrics"],
                "validation_source_table": row["validation_source_table"],
                "validation_source_safety": row["validation_source_safety"],
            }
        )
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_source_stable_trajnet_crowds_h100_specialist",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "stage43_s_precondition": {
            "verdict": stage43s.get("stage43_s_gate", {}).get("verdict"),
            "feasible_families": stage43s.get("preflight_summary", {}).get("feasible_families", []),
        },
        "training_protocol": {
            "family": FAMILY,
            "selection_data": "source_level_validation_only",
            "test_threshold_tuning": False,
            "num_workers": 0,
            "seed": int(seed),
            "feature_mean_hash": hashlib.sha256(feature_mean.tobytes()).hexdigest(),
            "feature_std_hash": hashlib.sha256(feature_std.tobytes()).hexdigest(),
            "future_waypoints_as_labels_only": True,
            "horizon": 100,
            "max_easy_degradation": float(max_easy_degradation),
        },
        "source_level_split": {
            "proposal_source": str(STAGE43_S_JSON),
            "train_sources": proposal.get("train_sources", []),
            "val_sources": proposal.get("val_sources", []),
            "test_sources": proposal.get("test_sources", []),
            "train_rows": int(len(train.x)),
            "val_rows": int(len(val.x)),
            "test_rows": int(len(test.x)),
            "train_source_table": _source_table(train),
            "val_source_table": _source_table(val),
            "test_source_table": _source_table(test),
        },
        "candidate_search": {
            "l2_grid": list(map(float, l2_grid)),
            "target_grid": list(target_grid),
            "candidate_count": int(len(candidates)),
            "candidates": candidate_rows,
        },
        "selected_specialist": {
            "target": best["target"],
            "l2": best["l2"],
            "model_hash": best["model_hash"],
            "validation_metrics": best["validation_metrics"],
            "validation_source_table": best["validation_source_table"],
            "validation_source_safety": best["validation_source_safety"],
        },
        "source_stable_h100_test_metrics": candidate_metrics,
        "source_stable_h100_test_source_table": _source_table(test, candidate_ade),
        "bootstrap_ci": bootstrap_ci,
        "deployment_metrics": deployment_metrics,
        "deployment": {
            "deploy_source_stable_h100_specialist": deploy,
            "positive_candidate": bool(positive),
            "easy_safe_candidate": bool(easy_safe),
            "reason": "deployable_source_stable_h100_specialist"
            if deploy
            else "validation_source_safety_failed"
            if not validation_source_safe
            else "candidate_positive_but_easy_harm_exceeds_guard"
            if positive
            else "candidate_nonpositive",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "uniform_t100_success": False,
        },
    }
    payload["stage43_t_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    candidate = payload["source_stable_h100_test_metrics"]
    deployment = payload["deployment"]
    gates = {
        "stage43_s_precondition_passed": payload["stage43_s_precondition"]["verdict"]
        == "stage43_s_t100_source_coverage_preflight_pass",
        "feasible_family_used": FAMILY in payload["stage43_s_precondition"]["feasible_families"],
        "source_level_split_built": payload["source_level_split"]["train_rows"] > 0
        and payload["source_level_split"]["val_rows"] > 0
        and payload["source_level_split"]["test_rows"] > 0,
        "validation_selected_only": payload["training_protocol"]["selection_data"] == "source_level_validation_only",
        "validation_source_safe": payload["selected_specialist"]["validation_source_safety"]["source_safe"] is True,
        "no_test_threshold_tuning": payload["training_protocol"]["test_threshold_tuning"] is False
        and payload["no_leakage"]["test_threshold_tuning"] is False,
        "future_waypoints_label_only": payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_waypoint_label_eval_only"] is True,
        "h100_test_evaluated": payload["source_stable_h100_test_metrics"]["rows"] > 0,
        "positive_or_blocker_reported": candidate["full_waypoint_ade_improvement_vs_floor"] > 0.0
        or deployment["reason"] == "candidate_nonpositive",
        "unsafe_candidate_not_deployed": (
            candidate["easy_degradation_vs_floor"] <= payload["training_protocol"]["max_easy_degradation"]
            and deployment["deploy_source_stable_h100_specialist"] is True
        )
        or (
            candidate["easy_degradation_vs_floor"] > payload["training_protocol"]["max_easy_degradation"]
            and deployment["deploy_source_stable_h100_specialist"] is False
        ),
        "deployment_easy_safe": payload["deployment_metrics"]["easy_degradation_vs_floor"] <= 0.02,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    if passed == total and deployment["deploy_source_stable_h100_specialist"]:
        verdict = "stage43_t_source_stable_h100_specialist_deployable"
    elif passed == total:
        verdict = "stage43_t_source_stable_h100_specialist_positive_but_not_safe"
    else:
        verdict = "stage43_t_source_stable_h100_specialist_incomplete"
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": verdict,
        "deploy_source_stable_h100_specialist": bool(deployment["deploy_source_stable_h100_specialist"] and passed == total),
        "positive_h100_dynamics_signal": bool(candidate["full_waypoint_ade_improvement_vs_floor"] > 0.0),
        "easy_safe": bool(candidate["easy_degradation_vs_floor"] <= payload["training_protocol"]["max_easy_degradation"]),
        "validation_source_safe": bool(payload["selected_specialist"]["validation_source_safety"]["source_safe"]),
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(dict(payload)))
    gate = payload["stage43_t_gate"]
    candidate = payload["source_stable_h100_test_metrics"]
    deployment = payload["deployment_metrics"]
    ci = payload["bootstrap_ci"]["metrics"]
    selected = payload["selected_specialist"]
    lines = [
        "# Stage43-T Source-Stable H100 Specialist",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- deploy source-stable h100 specialist: `{gate['deploy_source_stable_h100_specialist']}`",
        f"- positive h100 dynamics signal: `{gate['positive_h100_dynamics_signal']}`",
        f"- validation source safe: `{gate['validation_source_safe']}`",
        f"- easy safe: `{gate['easy_safe']}`",
        "",
        "## Source-Level Split",
        "",
        f"- train rows: `{payload['source_level_split']['train_rows']}`",
        f"- val rows: `{payload['source_level_split']['val_rows']}`",
        f"- test rows: `{payload['source_level_split']['test_rows']}`",
        f"- train sources: `{', '.join(payload['source_level_split']['train_sources'])}`",
        f"- val sources: `{', '.join(payload['source_level_split']['val_sources'])}`",
        f"- test sources: `{', '.join(payload['source_level_split']['test_sources'])}`",
        "",
        "## Selected Specialist",
        "",
        f"- target: `{selected['target']}`",
        f"- l2: `{selected['l2']}`",
        f"- validation ADE lift: `{_pct(selected['validation_metrics']['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- validation easy degradation: `{_pct(selected['validation_metrics']['easy_degradation_vs_floor'])}`",
        f"- min validation source lift: `{_pct(selected['validation_source_safety']['min_source_improvement'])}`",
        f"- max validation source easy degradation: `{_pct(selected['validation_source_safety']['max_source_easy_degradation'])}`",
        "",
        "## Held-Out H100 Candidate Test Metrics",
        "",
        f"- rows: `{candidate['rows']}`",
        f"- full-waypoint ADE improvement: `{_pct(candidate['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- endpoint FDE improvement: `{_pct(candidate['endpoint_fde_improvement_vs_floor'])}`",
        f"- hard/failure ADE improvement: `{_pct(candidate['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(candidate['easy_degradation_vs_floor'])}`",
        f"- bootstrap ADE CI: `[{_pct(ci['full_waypoint_ade_improvement_vs_floor']['low'])}, {_pct(ci['full_waypoint_ade_improvement_vs_floor']['high'])}]`",
        "",
        "## Deployment Metrics",
        "",
        f"- deployment ADE improvement: `{_pct(deployment['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- deployment easy degradation: `{_pct(deployment['easy_degradation_vs_floor'])}`",
        f"- deployment reason: `{payload['deployment']['reason']}`",
        "",
        "## Interpretation",
        "",
        "Stage43-T tests whether the only h100 family with enough source coverage, TrajNet_crowds, can support a source-stable long-horizon specialist. Deployment requires validation-source safety and held-out easy preservation; otherwise Stage43-P/R remain the safety floor and t100 remains fallback-only outside diagnostic research.",
        "",
        "Claim boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; no Stage5C execution; no SMC.",
    ]
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage43-T Gate",
        "",
        f"verdict: `{gate['verdict']}`",
        f"passed: `{gate['passed']} / {gate['total']}`",
        f"deploy_source_stable_h100_specialist: `{gate['deploy_source_stable_h100_specialist']}`",
        f"positive_h100_dynamics_signal: `{gate['positive_h100_dynamics_signal']}`",
        f"easy_safe: `{gate['easy_safe']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    gate_lines.extend([f"| {name} | `{value}` |" for name, value in gate["gates"].items()])
    write_md(GATE_MD, gate_lines)
    _refresh_readmes(payload)
    _update_state(payload)
    _append_ledger(payload)


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_t_gate"]
    candidate = payload["source_stable_h100_test_metrics"]
    deployment = payload["deployment_metrics"]
    lines = [
        "## Stage43-T source-stable h100 specialist",
        "",
        f"Result source: `{payload['result_source']}`. This trains a source-stable h100 specialist only for the feasible TrajNet_crowds family, using source-level train/val/test split from Stage43-S.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- positive h100 dynamics signal: `{gate['positive_h100_dynamics_signal']}`",
        f"- validation source safe: `{gate['validation_source_safe']}`",
        f"- easy safe: `{gate['easy_safe']}`",
        f"- deployed: `{gate['deploy_source_stable_h100_specialist']}`",
        f"- held-out h100 ADE improvement: `{_pct(candidate['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- held-out h100 easy degradation: `{_pct(candidate['easy_degradation_vs_floor'])}`",
        f"- deployment ADE improvement: `{_pct(deployment['full_waypoint_ade_improvement_vs_floor'])}`",
        "",
        "Boundary: this is a source-stable h100 family trial, not a uniform t100 claim. Deployment is allowed only when validation-source safety and held-out easy preservation both pass; otherwise Stage43-P/R remain the safety floor and t100 remains fallback-only.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, lines)


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    gate = payload["stage43_t_gate"]
    state["stage43_t_t100_source_stable_specialist"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "selected_specialist": payload["selected_specialist"],
        "source_stable_h100_test_metrics": payload["source_stable_h100_test_metrics"],
        "deployment": payload["deployment"],
        "deployment_metrics": payload["deployment_metrics"],
        "claim_boundary": payload["claim_boundary"],
    }
    state["current_stage"] = "stage43_t_t100_source_stable_specialist"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(OUT_DIR)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable({"event": "stage43_t_t100_source_stable_specialist", "payload": payload}), ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-T source-stable h100 specialist.")
    parser.add_argument("--seed", type=int, default=461)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--max-easy-degradation", type=float, default=0.02)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = run_t100_source_stable_specialist(
        seed=int(args.seed),
        bootstrap=int(args.bootstrap),
        max_easy_degradation=float(args.max_easy_degradation),
    )
    gate = result["stage43_t_gate"]
    print(f"Stage43-T: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    return result


if __name__ == "__main__":
    main()
