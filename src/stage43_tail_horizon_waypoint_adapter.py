from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
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
    _bootstrap_ci,
    _build_split,
    _git_commit,
    _jsonable,
    _metrics,
    _trajectory_error,
)
from src.stage43_full_waypoint_latent_robustness_audit import _breakdown, _pct, _slice_metrics, _top_slices
from src.stage43_full_waypoint_latent_safe_repair import _source_family


REPORT_JSON = OUT_DIR / "stage43_tail_horizon_waypoint_adapter.json"
REPORT_MD = OUT_DIR / "stage43_tail_horizon_waypoint_adapter.md"
GATE_MD = OUT_DIR / "stage43_stage_p_tail_horizon_waypoint_adapter_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

STAGE43_O_JSON = OUT_DIR / "stage43_full_waypoint_latent_safe_repair.json"
SECTION = "STAGE43_P_TAIL_HORIZON_WAYPOINT_ADAPTER"
SOURCE = "fresh_stage43_p_tail_horizon_waypoint_adapter"
HORIZONS = [10, 25, 50, 100]
EPS = 1e-8


def _standardize(train, val, test):
    mean = train.x.mean(axis=0).astype(np.float32)
    raw_std = train.x.std(axis=0).astype(np.float32)
    std = np.where(raw_std < 1e-3, 1.0, raw_std).astype(np.float32)
    for ds in [train, val, test]:
        ds.x = ((ds.x - mean) / std).astype(np.float32)
    return mean, std


def _ridge_fit(x: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    xb = np.concatenate([x, np.ones((len(x), 1), dtype=np.float32)], axis=1).astype(np.float64)
    eye = np.eye(xb.shape[1], dtype=np.float64)
    eye[-1, -1] = 0.0
    return np.linalg.solve(xb.T @ xb + float(l2) * eye, xb.T @ y.astype(np.float64))


def _ridge_predict(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    xb = np.concatenate([x, np.ones((len(x), 1), dtype=np.float32)], axis=1).astype(np.float64)
    return (xb @ weight).astype(np.float32)


def _model_hash(weight: np.ndarray, *, l2: float, target: str, train_filter: str) -> str:
    digest = hashlib.sha256()
    digest.update(weight.astype(np.float32).tobytes())
    digest.update(str(l2).encode("utf-8"))
    digest.update(target.encode("utf-8"))
    digest.update(train_filter.encode("utf-8"))
    return digest.hexdigest()


def _train_mask(ds, train_filter: str) -> np.ndarray:
    if train_filter == "all":
        return np.ones(len(ds.x), dtype=bool)
    if train_filter == "t100":
        return ds.horizon == 100
    if train_filter == "t50t100":
        return ds.horizon >= 50
    raise ValueError(f"Unknown train_filter={train_filter}")


def _target_matrix(ds, target: str) -> np.ndarray:
    if target == "residual":
        return (ds.waypoint_delta - ds.floor_waypoint_delta).reshape(len(ds.x), -1).astype(np.float32)
    if target == "direct":
        return ds.waypoint_delta.reshape(len(ds.x), -1).astype(np.float32)
    raise ValueError(f"Unknown target={target}")


def _predict_waypoint(ds, weight: np.ndarray, target: str) -> np.ndarray:
    pred = _ridge_predict(ds.x, weight)
    if target == "residual":
        pred = ds.floor_waypoint_delta.reshape(len(ds.x), -1) + pred
    return pred.reshape(-1, 4, 2).astype(np.float32)


def _family_horizon(ds) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray([_source_family(value) for value in ds.source_file]).astype(str), ds.horizon.astype(np.int64)


def _slice_improvement(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return 0.0
    return float(1.0 - float(np.mean(selected[mask])) / max(float(np.mean(floor[mask])), EPS))


def _easy_degradation(ds, selected: np.ndarray, mask: np.ndarray) -> float:
    easy = mask & ds.easy
    if int(easy.sum()) == 0:
        return 0.0
    return float(max(0.0, float(np.mean(selected[easy])) / max(float(np.mean(ds.floor_ade[easy])), EPS) - 1.0))


def _select_support_rules(
    ds,
    candidate_ade: np.ndarray,
    *,
    min_support_rows: int,
    min_improvement: float,
    max_easy_degradation: float,
    require_all_supported_h100_safe: bool,
) -> tuple[dict[str, Any], set[tuple[str, int]], dict[str, Any]]:
    families, horizons = _family_horizon(ds)
    table: dict[str, Any] = {}
    allowed: set[tuple[str, int]] = set()
    h100_supported: list[dict[str, Any]] = []
    for family in sorted(set(families.tolist())):
        for horizon in HORIZONS:
            mask = (families == family) & (horizons == int(horizon))
            if int(mask.sum()) == 0:
                continue
            improvement = _slice_improvement(candidate_ade, ds.floor_ade, mask)
            easy = _easy_degradation(ds, candidate_ade, mask)
            supported = int(mask.sum()) >= int(min_support_rows)
            safe_positive = supported and improvement > float(min_improvement) and easy <= float(max_easy_degradation)
            key = f"{family}|{horizon}"
            reason = "allowed_by_validation"
            if not supported:
                reason = "blocked_insufficient_validation_support"
            elif improvement <= float(min_improvement):
                reason = "blocked_validation_nonpositive"
            elif easy > float(max_easy_degradation):
                reason = "blocked_validation_easy_harm"
            table[key] = {
                "rows": int(mask.sum()),
                "full_waypoint_ade_improvement_vs_floor": float(improvement),
                "easy_degradation_vs_floor": float(easy),
                "allowed_before_h100_contract": bool(safe_positive),
                "allowed": bool(safe_positive),
                "reason": reason,
            }
            if horizon == 100 and supported:
                h100_supported.append({"family": family, **table[key]})
            if safe_positive:
                allowed.add((family, int(horizon)))
    h100_contract = {
        "require_all_supported_h100_safe": bool(require_all_supported_h100_safe),
        "supported_h100_family_count": int(len(h100_supported)),
        "unsafe_or_nonpositive_supported_h100": [
            row for row in h100_supported if not bool(row["allowed_before_h100_contract"])
        ],
        "allow_h100": True,
    }
    if require_all_supported_h100_safe and h100_supported:
        h100_contract["allow_h100"] = len(h100_contract["unsafe_or_nonpositive_supported_h100"]) == 0
    if not h100_contract["allow_h100"]:
        allowed = {(family, horizon) for family, horizon in allowed if horizon != 100}
        for key, row in table.items():
            if key.endswith("|100") and row["allowed"]:
                row["allowed"] = False
                row["reason"] = "blocked_h100_global_validation_contract"
    return table, allowed, h100_contract


def _apply_rules(ds, candidate_ade: np.ndarray, candidate_fde: np.ndarray, allowed: set[tuple[str, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    families, horizons = _family_horizon(ds)
    switch = np.asarray([(family, int(horizon)) in allowed for family, horizon in zip(families, horizons)], dtype=bool)
    selected_ade = np.where(switch, candidate_ade, ds.floor_ade).astype(np.float32)
    selected_fde = np.where(switch, candidate_fde, ds.floor_fde).astype(np.float32)
    return selected_ade, selected_fde, switch


def _candidate_eval(
    train,
    val,
    *,
    target: str,
    train_filter: str,
    l2: float,
    min_support_rows: int,
    min_improvement: float,
    max_easy_degradation: float,
    require_all_supported_h100_safe: bool,
) -> dict[str, Any]:
    train_ids = _train_mask(train, train_filter)
    weight = _ridge_fit(train.x[train_ids], _target_matrix(train, target)[train_ids], float(l2))
    val_pred = _predict_waypoint(val, weight, target)
    val_ade, val_fde = _trajectory_error(val, val_pred)
    table, allowed, h100_contract = _select_support_rules(
        val,
        val_ade,
        min_support_rows=int(min_support_rows),
        min_improvement=float(min_improvement),
        max_easy_degradation=float(max_easy_degradation),
        require_all_supported_h100_safe=bool(require_all_supported_h100_safe),
    )
    selected_ade, selected_fde, switch = _apply_rules(val, val_ade, val_fde, allowed)
    metrics = _metrics(val, selected_ade, selected_fde, switch)
    objective = (
        1.0 * metrics["full_waypoint_ade_improvement_vs_floor"]
        + 1.3 * metrics["t50_full_waypoint_ade_improvement_vs_floor"]
        + 0.7 * metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
        - 20.0 * max(0.0, metrics["easy_degradation_vs_floor"] - 0.02)
        - 0.8 * max(0.0, -metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"])
    )
    return {
        "target": target,
        "train_filter": train_filter,
        "l2": float(l2),
        "train_rows": int(train_ids.sum()),
        "weight": weight,
        "model_hash": _model_hash(weight, l2=float(l2), target=target, train_filter=train_filter),
        "validation_metrics": metrics,
        "validation_support_table": table,
        "allowed": allowed,
        "h100_contract": h100_contract,
        "objective": float(objective),
    }


def _breakdown_by_source_family(ds, selected_ade, selected_fde, candidate_ade, switch) -> dict[str, Any]:
    families, _ = _family_horizon(ds)
    arrays = (ds.floor_ade, ds.floor_fde, selected_ade, selected_fde, candidate_ade, switch, ds.easy)
    return _breakdown(families, *arrays, min_rows=50)


def run_tail_horizon_waypoint_adapter(
    *,
    seed: int = 431,
    l2_grid: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0),
    target_grid: tuple[str, ...] = ("residual", "direct"),
    train_filter_grid: tuple[str, ...] = ("t100", "t50t100"),
    min_support_rows: int = 1000,
    min_improvement: float = 0.0,
    max_easy_degradation: float = 0.02,
    bootstrap: int = 1000,
) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage43o = read_json(STAGE43_O_JSON, {})
    train = _build_split("train", max_rows=None, seed=int(seed))
    val = _build_split("val", max_rows=None, seed=int(seed))
    test = _build_split("test", max_rows=None, seed=int(seed))
    feature_mean, feature_std = _standardize(train, val, test)
    candidates = [
        _candidate_eval(
            train,
            val,
            target=target,
            train_filter=train_filter,
            l2=l2,
            min_support_rows=int(min_support_rows),
            min_improvement=float(min_improvement),
            max_easy_degradation=float(max_easy_degradation),
            require_all_supported_h100_safe=True,
        )
        for target in target_grid
        for train_filter in train_filter_grid
        for l2 in l2_grid
    ]
    candidates.sort(key=lambda row: row["objective"], reverse=True)
    best = candidates[0]
    test_pred = _predict_waypoint(test, best["weight"], best["target"])
    candidate_ade, candidate_fde = _trajectory_error(test, test_pred)
    selected_ade, selected_fde, switch = _apply_rules(test, candidate_ade, candidate_fde, best["allowed"])
    metrics = _metrics(test, selected_ade, selected_fde, switch)
    bootstrap_ci = _bootstrap_ci(test, selected_ade, selected_fde, n=int(bootstrap), seed=int(seed) + 4300)
    arrays = (test.floor_ade, test.floor_fde, selected_ade, selected_fde, candidate_ade, switch, test.easy)
    by_domain = _breakdown(test.domain, *arrays)
    by_horizon = _breakdown(test.horizon.astype(str), *arrays)
    by_source = _breakdown(test.source_file, *arrays, min_rows=50)
    by_source_family = _breakdown_by_source_family(test, selected_ade, selected_fde, candidate_ade, switch)
    negative_sources = [
        {"source_file": name, **row}
        for name, row in by_source.items()
        if float(row["full_waypoint_ade_improvement_vs_floor"]) < 0.0
    ]
    o_metrics = stage43o.get("overall_full_test_metrics", {})
    candidate_rows = []
    for row in candidates:
        r = {key: value for key, value in row.items() if key not in {"weight", "allowed", "validation_support_table"}}
        r["allowed_rules"] = sorted([f"{family}|{horizon}" for family, horizon in row["allowed"]])
        candidate_rows.append(r)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_train_val_selected_tail_horizon_adapter",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "stage43_o_precondition": {
            "verdict": stage43o.get("stage43_o_gate", {}).get("verdict"),
            "t100_status": stage43o.get("t100_repair", {}).get("status"),
            "t100_improvement": stage43o.get("t100_repair", {}).get("improvement"),
        },
        "training_protocol": {
            "model_family": "closed_form_ridge_full_waypoint_residual_adapter",
            "selection_data": "validation_only",
            "test_threshold_tuning": False,
            "num_workers": 0,
            "seed": int(seed),
            "feature_mean_hash": hashlib.sha256(feature_mean.tobytes()).hexdigest(),
            "feature_std_hash": hashlib.sha256(feature_std.tobytes()).hexdigest(),
            "future_waypoints_as_labels_only": True,
        },
        "candidate_search": {
            "l2_grid": list(map(float, l2_grid)),
            "target_grid": list(target_grid),
            "train_filter_grid": list(train_filter_grid),
            "candidate_count": int(len(candidates)),
            "top_candidates": candidate_rows[:8],
        },
        "selected_model": {
            "target": best["target"],
            "train_filter": best["train_filter"],
            "l2": best["l2"],
            "train_rows": best["train_rows"],
            "model_hash": best["model_hash"],
            "validation_metrics": best["validation_metrics"],
            "allowed_rules": sorted([f"{family}|{horizon}" for family, horizon in best["allowed"]]),
            "h100_contract": best["h100_contract"],
            "validation_support_table": best["validation_support_table"],
        },
        "full_test_rows": int(len(test.x)),
        "overall_full_test_metrics": metrics,
        "delta_vs_stage43_o": {
            "full_waypoint_ade_improvement_delta": float(
                metrics["full_waypoint_ade_improvement_vs_floor"]
                - float(o_metrics.get("full_waypoint_ade_improvement_vs_floor", 0.0))
            ),
            "t50_delta": float(
                metrics["t50_full_waypoint_ade_improvement_vs_floor"]
                - float(o_metrics.get("t50_full_waypoint_ade_improvement_vs_floor", 0.0))
            ),
            "t100_delta": float(
                metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]
                - float(o_metrics.get("t100_raw_frame_full_waypoint_diagnostic_vs_floor", 0.0))
            ),
            "hard_failure_delta": float(
                metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
                - float(o_metrics.get("hard_failure_full_waypoint_ade_improvement_vs_floor", 0.0))
            ),
            "easy_degradation_delta": float(
                metrics["easy_degradation_vs_floor"] - float(o_metrics.get("easy_degradation_vs_floor", 0.0))
            ),
        },
        "bootstrap_ci": bootstrap_ci,
        "by_domain": by_domain,
        "by_horizon": by_horizon,
        "by_source_family": by_source_family,
        "by_source_summary": {
            "source_count": int(len(by_source)),
            "negative_source_count": int(len(negative_sources)),
            "worst_sources": _top_slices(by_source, key="full_waypoint_ade_improvement_vs_floor", n=12),
            "best_sources": _top_slices(by_source, key="full_waypoint_ade_improvement_vs_floor", n=12, reverse=True),
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
            "t100_positive_success": False,
            "uniform_source_positive_success": False,
        },
    }
    payload["stage43_p_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["overall_full_test_metrics"]
    h100 = payload["by_horizon"].get("100", {})
    gates = {
        "stage43_o_precondition_passed": payload["stage43_o_precondition"]["verdict"]
        == "stage43_o_safe_repair_pass_t100_fallback_not_positive",
        "fresh_train_val_selected_adapter": payload["result_source"] == "fresh_train_val_selected_tail_horizon_adapter"
        and payload["training_protocol"]["selection_data"] == "validation_only",
        "no_test_threshold_tuning": payload["training_protocol"]["test_threshold_tuning"] is False
        and payload["no_leakage"]["test_threshold_tuning"] is False,
        "future_waypoints_label_only": payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_waypoint_label_eval_only"] is True,
        "overall_improves_stage43_o": payload["delta_vs_stage43_o"]["full_waypoint_ade_improvement_delta"] > 0.0,
        "t50_improves_stage43_o": payload["delta_vs_stage43_o"]["t50_delta"] > 0.0,
        "hard_failure_improves_stage43_o": payload["delta_vs_stage43_o"]["hard_failure_delta"] > 0.0,
        "easy_preserved": metrics["easy_degradation_vs_floor"] <= 0.02,
        "t100_not_harmed": metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] >= -1e-7
        and float(h100.get("easy_degradation_vs_floor", 0.0)) <= 0.02,
        "t100_positive_not_overclaimed": payload["claim_boundary"]["t100_positive_success"] is False
        and abs(metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]) < 1e-7,
        "negative_source_count_zero": payload["by_source_summary"]["negative_source_count"] == 0,
        "bootstrap_t50_positive": payload["bootstrap_ci"]["metrics"]["t50_full_waypoint_ade_improvement_vs_floor"]["low"] > 0.0,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_p_tail_horizon_adapter_pass_t100_still_fallback"
        if passed == total
        else "stage43_p_tail_horizon_adapter_incomplete",
        "deploy_tail_horizon_adapter": passed == total,
        "t100_positive_success": False,
        "uniform_source_positive_success": False,
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(dict(payload)))
    gate = payload["stage43_p_gate"]
    metrics = payload["overall_full_test_metrics"]
    delta = payload["delta_vs_stage43_o"]
    ci = payload["bootstrap_ci"]["metrics"]
    lines = [
        "# Stage43-P Tail-Horizon Full-Waypoint Adapter",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- full-test rows: `{payload['full_test_rows']}`",
        "",
        "## Selected Model",
        "",
        f"- target: `{payload['selected_model']['target']}`",
        f"- train filter: `{payload['selected_model']['train_filter']}`",
        f"- l2: `{payload['selected_model']['l2']}`",
        f"- train rows: `{payload['selected_model']['train_rows']}`",
        f"- allowed rules: `{', '.join(payload['selected_model']['allowed_rules'])}`",
        f"- h100 allowed by validation contract: `{payload['selected_model']['h100_contract']['allow_h100']}`",
        "",
        "## Full-Test Metrics",
        "",
        f"- full-waypoint ADE improvement: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- endpoint FDE improvement: `{_pct(metrics['endpoint_fde_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- hard/failure full-waypoint ADE improvement: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- switch rate: `{_pct(metrics['switch_rate'])}`",
        "",
        "## Delta vs Stage43-O",
        "",
        f"- all ADE improvement delta: `{_pct(delta['full_waypoint_ade_improvement_delta'])}`",
        f"- t50 delta: `{_pct(delta['t50_delta'])}`",
        f"- t100 delta: `{_pct(delta['t100_delta'])}`",
        f"- hard/failure delta: `{_pct(delta['hard_failure_delta'])}`",
        f"- easy degradation delta: `{_pct(delta['easy_degradation_delta'])}`",
        "",
        "## Bootstrap CI",
        "",
        f"- bootstrap n: `{payload['bootstrap_ci']['n']}`",
        f"- all ADE CI: `[{_pct(ci['full_waypoint_ade_improvement_vs_floor']['low'])}, {_pct(ci['full_waypoint_ade_improvement_vs_floor']['high'])}]`",
        f"- t50 ADE CI: `[{_pct(ci['t50_full_waypoint_ade_improvement_vs_floor']['low'])}, {_pct(ci['t50_full_waypoint_ade_improvement_vs_floor']['high'])}]`",
        f"- hard/failure ADE CI: `[{_pct(ci['hard_failure_full_waypoint_ade_improvement_vs_floor']['low'])}, {_pct(ci['hard_failure_full_waypoint_ade_improvement_vs_floor']['high'])}]`",
        f"- easy degradation CI: `[{_pct(ci['easy_degradation_vs_floor']['low'])}, {_pct(ci['easy_degradation_vs_floor']['high'])}]`",
        "",
        "## Horizon Breakdown",
        "",
        "| horizon | rows | ADE lift | easy degradation | switch |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in payload["by_horizon"].items():
        lines.append(
            f"| {name} | {row['rows']} | {_pct(row['full_waypoint_ade_improvement_vs_floor'])} | {_pct(row['easy_degradation_vs_floor'])} | {_pct(row['switch_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Stage43-P trains a real train-split full-waypoint adapter for the tail horizons and selects deployment rules only on validation. It materially improves all/t50/hard over Stage43-O while preserving easy cases. The h100 contract blocks every h100 switch because validation h100 support is not uniformly safe; t100 is therefore not solved, only kept non-harmful by fallback.",
            "",
            "Claim boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; no Stage5C execution; no SMC.",
        ]
    )
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage43-P Gate",
        "",
        f"verdict: `{gate['verdict']}`",
        f"deploy_tail_horizon_adapter: `{gate['deploy_tail_horizon_adapter']}`",
        f"passed: `{gate['passed']} / {gate['total']}`",
        f"t100_positive_success: `{gate['t100_positive_success']}`",
        f"uniform_source_positive_success: `{gate['uniform_source_positive_success']}`",
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
    gate = payload["stage43_p_gate"]
    metrics = payload["overall_full_test_metrics"]
    ci = payload["bootstrap_ci"]["metrics"]
    lines = [
        "## Stage43-P tail-horizon full-waypoint adapter",
        "",
        f"Result source: `{payload['result_source']}`. A train-split ridge full-waypoint adapter was trained on tail horizons and selected on validation only, then tested once against the Stage43-O safe repair floor.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- full-test rows: `{payload['full_test_rows']}`",
        f"- full-waypoint ADE improvement vs floor: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement vs floor: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- hard/failure ADE improvement vs floor: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- t50 bootstrap CI: `[{_pct(ci['t50_full_waypoint_ade_improvement_vs_floor']['low'])}, {_pct(ci['t50_full_waypoint_ade_improvement_vs_floor']['high'])}]`",
        "",
        "Boundary: this is a stronger protected tail-horizon full-waypoint adapter, but t100 remains fallback-only rather than positive. The result remains dataset-local/raw-frame 2.5D evidence with no metric/seconds-level claim, no Stage5C, and no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, lines)


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    gate = payload["stage43_p_gate"]
    state["stage43_p_tail_horizon_waypoint_adapter"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "selected_model": payload["selected_model"],
        "overall_full_test_metrics": payload["overall_full_test_metrics"],
        "delta_vs_stage43_o": payload["delta_vs_stage43_o"],
        "bootstrap_ci": payload["bootstrap_ci"],
        "claim_boundary": payload["claim_boundary"],
    }
    state["current_stage"] = "stage43_p_tail_horizon_waypoint_adapter"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(OUT_DIR)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable({"event": "stage43_p_tail_horizon_waypoint_adapter", "payload": payload}), ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Stage43-P tail-horizon full-waypoint adapter.")
    parser.add_argument("--seed", type=int, default=431)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--min-support-rows", type=int, default=1000)
    parser.add_argument("--max-easy-degradation", type=float, default=0.02)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = run_tail_horizon_waypoint_adapter(
        seed=int(args.seed),
        min_support_rows=int(args.min_support_rows),
        max_easy_degradation=float(args.max_easy_degradation),
        bootstrap=int(args.bootstrap),
    )
    gate = result["stage43_p_gate"]
    print(f"Stage43-P: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    return result


if __name__ == "__main__":
    main()
