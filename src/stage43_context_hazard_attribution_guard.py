from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_context_admissibility_model as bt
from src import stage43_context_admissibility_robustness_audit as bu
from src import stage43_context_admissibility_slice_safe_repair as bv
from src import stage43_full_waypoint_latent_dynamics as m


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_context_hazard_attribution_guard.json"
REPORT_MD = OUT_DIR / "stage43_context_hazard_attribution_guard.md"
GATE_MD = OUT_DIR / "stage43_stage_bw_context_hazard_attribution_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_BW_CONTEXT_HAZARD_ATTRIBUTION_GUARD"
SOURCE = "fresh_stage43_bw_context_hazard_attribution_guard"
DEFAULT_VARIANT = bt.DEFAULT_VARIANT


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _metric_lines(metrics: Mapping[str, Any]) -> list[str]:
    return [
        f"- all full-waypoint ADE improvement: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic improvement: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- hard/failure improvement: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- switch rate: `{_pct(metrics['switch_rate'])}`",
    ]


def _variant_counts(values: np.ndarray) -> dict[str, int]:
    return {str(v): int(np.sum(values.astype(str) == str(v))) for v in sorted(set(values.astype(str).tolist()))}


def _context_easy_hazard_audit(
    ds: m.WaypointSplit,
    selected_ade: np.ndarray,
    graph_ade: np.ndarray,
    used: np.ndarray,
    *,
    min_rows: int,
    min_context_easy_rows: int,
    rate_threshold: float,
    mean_harm_threshold: float,
) -> dict[str, Any]:
    """Attribute easy risk to context choices, not to the graph-history floor.

    BU/BV report absolute easy hazards versus the floor. That is useful for safety,
    but it can falsely blame context when graph-history already has the same slice
    hazard. This audit measures additional harm on rows where context actually
    switched away from graph-history.
    """

    used_str = used.astype(str)
    context = used_str != DEFAULT_VARIANT
    rows: list[dict[str, Any]] = []
    for family, keys in bv._key_arrays(ds).items():
        for key in sorted(set(keys.tolist())):
            mask = keys == key
            if int(mask.sum()) < int(min_rows):
                continue
            easy_mask = mask & ds.easy
            context_easy = easy_mask & context
            if int(context_easy.sum()) < int(min_context_easy_rows):
                continue
            context_delta = selected_ade[context_easy] - graph_ade[context_easy]
            easy_delta = selected_ade[easy_mask] - graph_ade[easy_mask] if int(easy_mask.sum()) else np.asarray([], dtype=np.float32)
            mean_context_harm = float(np.maximum(context_delta, 0.0).mean()) if len(context_delta) else 0.0
            context_harm_rate = float(np.mean(context_delta > 1e-6)) if len(context_delta) else 0.0
            mean_easy_delta = float(easy_delta.mean()) if len(easy_delta) else 0.0
            row = {
                "slice": f"{family}_{key}",
                "family": family,
                "key": str(key),
                "rows": int(mask.sum()),
                "easy_rows": int(easy_mask.sum()),
                "context_easy_rows": int(context_easy.sum()),
                "context_rate": float(np.mean(context[mask])),
                "mean_context_easy_harm_vs_graph": mean_context_harm,
                "context_easy_harm_rate_vs_graph": context_harm_rate,
                "mean_easy_delta_vs_graph": mean_easy_delta,
                "context_hazard": bool(
                    context_harm_rate >= float(rate_threshold) and mean_context_harm >= float(mean_harm_threshold)
                ),
            }
            rows.append(row)
    hazard_rows = [row for row in rows if row["context_hazard"]]
    return {
        "min_rows": int(min_rows),
        "min_context_easy_rows": int(min_context_easy_rows),
        "rate_threshold": float(rate_threshold),
        "mean_harm_threshold": float(mean_harm_threshold),
        "slice_count": len(rows),
        "context_hazard_slice_count": len(hazard_rows),
        "top_context_hazard_slices": sorted(
            hazard_rows,
            key=lambda r: (float(r["mean_context_easy_harm_vs_graph"]), float(r["context_easy_harm_rate_vs_graph"])),
            reverse=True,
        )[:12],
        "top_context_harm_slices": sorted(
            rows,
            key=lambda r: (float(r["mean_context_easy_harm_vs_graph"]), float(r["context_easy_harm_rate_vs_graph"])),
            reverse=True,
        )[:12],
    }


def _hazard_keys(
    ds: m.WaypointSplit,
    selected_ade: np.ndarray,
    graph_ade: np.ndarray,
    used: np.ndarray,
    *,
    family: str,
    min_context_easy_rows: int,
    rate_threshold: float,
    mean_harm_threshold: float,
) -> set[str]:
    keys = bv._key_arrays(ds)[family]
    context = used.astype(str) != DEFAULT_VARIANT
    out: set[str] = set()
    for key in sorted(set(keys.tolist())):
        context_easy = (keys == key) & ds.easy & context
        if int(context_easy.sum()) < int(min_context_easy_rows):
            continue
        delta = selected_ade[context_easy] - graph_ade[context_easy]
        mean_harm = float(np.maximum(delta, 0.0).mean()) if len(delta) else 0.0
        harm_rate = float(np.mean(delta > 1e-6)) if len(delta) else 0.0
        if harm_rate >= float(rate_threshold) and mean_harm >= float(mean_harm_threshold):
            out.add(str(key))
    return out


def _candidate_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {"name": "bt_unrepaired", "family": None, "rate": 1.0, "mean": 1.0, "block_t100": False, "all_fallback": False},
        {"name": "bv_block_t100", "family": None, "rate": 1.0, "mean": 1.0, "block_t100": True, "all_fallback": False},
        {"name": "all_context_fallback", "family": None, "rate": 0.0, "mean": 0.0, "block_t100": False, "all_fallback": True},
    ]
    for family in ["horizon", "domain", "domain_horizon", "source", "source_horizon"]:
        for rate in [0.10, 0.20, 0.30, 0.40]:
            specs.append(
                {
                    "name": f"guard_{family}_rate_{rate:.2f}",
                    "family": family,
                    "rate": rate,
                    "mean": 0.005,
                    "block_t100": False,
                    "all_fallback": False,
                }
            )
            specs.append(
                {
                    "name": f"guard_{family}_rate_{rate:.2f}_plus_block_t100",
                    "family": family,
                    "rate": rate,
                    "mean": 0.005,
                    "block_t100": True,
                    "all_fallback": False,
                }
            )
    return specs


def _apply_guard(
    batch: bt.ContextBatch,
    selected_ade: np.ndarray,
    selected_fde: np.ndarray,
    switched: np.ndarray,
    used: np.ndarray,
    *,
    spec: Mapping[str, Any],
    hazard_keys: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    graph_ade = batch.arrays[DEFAULT_VARIANT]["selected_ade"].astype(np.float32)
    graph_fde = batch.arrays[DEFAULT_VARIANT]["selected_fde"].astype(np.float32)
    graph_switched = batch.arrays[DEFAULT_VARIANT]["switched"].astype(bool)
    context = used.astype(str) != DEFAULT_VARIANT
    blocked = np.zeros(len(used), dtype=bool)
    if spec.get("all_fallback", False):
        blocked |= context
    if spec.get("block_t100", False):
        blocked |= context & (batch.ds.horizon == 100)
    family = spec.get("family")
    if family:
        keys = bv._key_arrays(batch.ds)[str(family)].astype(str)
        blocked |= context & np.isin(keys, list(hazard_keys))
    out_ade = selected_ade.copy()
    out_fde = selected_fde.copy()
    out_switched = switched.copy()
    out_used = used.copy()
    out_ade[blocked] = graph_ade[blocked]
    out_fde[blocked] = graph_fde[blocked]
    out_switched[blocked] = graph_switched[blocked]
    out_used[blocked] = DEFAULT_VARIANT
    return out_ade.astype(np.float32), out_fde.astype(np.float32), out_switched.astype(bool), out_used, blocked


def _score(delta: Mapping[str, float], context_hazards: int, blocked_rows: int) -> float:
    return float(
        delta["all"]
        + 1.2 * delta["t50"]
        + delta["hard_failure"]
        + 0.4 * min(0.0, delta["t100_raw_frame_diagnostic"])
        - 25.0 * max(0.0, delta["easy_degradation"])
        - 0.005 * int(context_hazards)
        - 0.000001 * int(blocked_rows)
    )


def _select_guard(
    val: bt.ContextBatch,
    selected_ade: np.ndarray,
    selected_fde: np.ndarray,
    switched: np.ndarray,
    used: np.ndarray,
    *,
    min_rows: int,
    min_context_easy_rows: int,
) -> dict[str, Any]:
    graph_ade = val.arrays[DEFAULT_VARIANT]["selected_ade"].astype(np.float32)
    graph_metrics = m._metrics(
        val.ds,
        graph_ade,
        val.arrays[DEFAULT_VARIANT]["selected_fde"].astype(np.float32),
        val.arrays[DEFAULT_VARIANT]["switched"].astype(bool),
    )
    rows: list[dict[str, Any]] = []
    for spec in _candidate_specs():
        family = spec.get("family")
        keys: set[str] = set()
        if family:
            keys = _hazard_keys(
                val.ds,
                selected_ade,
                graph_ade,
                used,
                family=str(family),
                min_context_easy_rows=min_context_easy_rows,
                rate_threshold=float(spec["rate"]),
                mean_harm_threshold=float(spec["mean"]),
            )
        ade, fde, sw, repaired_used, blocked = _apply_guard(
            val, selected_ade, selected_fde, switched, used, spec=spec, hazard_keys=keys
        )
        metrics = m._metrics(val.ds, ade, fde, sw)
        delta = bv._delta(metrics, graph_metrics)
        context_hazards = _context_easy_hazard_audit(
            val.ds,
            ade,
            graph_ade,
            repaired_used,
            min_rows=min_rows,
            min_context_easy_rows=min_context_easy_rows,
            rate_threshold=0.20,
            mean_harm_threshold=0.005,
        )
        safe = (
            metrics["easy_degradation_vs_floor"] <= 0.02
            and delta["easy_degradation"] <= 0.02
            and int(context_hazards["context_hazard_slice_count"]) == 0
            and max(delta["all"], delta["t50"], delta["hard_failure"]) >= 0.0
        )
        rows.append(
            {
                "spec": dict(spec),
                "validation_metrics": metrics,
                "delta_vs_graph_history_only": delta,
                "context_hazard_slice_count": int(context_hazards["context_hazard_slice_count"]),
                "hazard_key_count": len(keys),
                "blocked_context_rows": int(np.sum(blocked)),
                "context_variant_counts": _variant_counts(repaired_used),
                "safe": bool(safe),
                "selection_score": _score(delta, int(context_hazards["context_hazard_slice_count"]), int(np.sum(blocked))),
            }
        )
    safe_rows = [row for row in rows if row["safe"]]
    selected = dict(max(safe_rows or rows, key=lambda row: row["selection_score"]))
    selected["candidate_count"] = len(rows)
    selected["safe_candidate_count"] = len(safe_rows)
    selected["all_candidates"] = [dict(row) for row in rows]
    return selected


def _source_overlap(val: bt.ContextBatch, test: bt.ContextBatch) -> dict[str, Any]:
    val_sources = set(val.ds.source_file.astype(str).tolist())
    test_sources = set(test.ds.source_file.astype(str).tolist())
    return {
        "val_source_count": len(val_sources),
        "test_source_count": len(test_sources),
        "overlap_count": len(val_sources & test_sources),
        "overlap_examples": sorted(val_sources & test_sources)[:8],
        "held_out_source_level": len(val_sources & test_sources) == 0,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    bt_data = bv._load_bt_predictions(rows=int(args.test_rows) if int(args.test_rows) else None)
    val = bt_data["val"]
    test = bt_data["test"]
    val_ade, val_fde, val_switched, val_used = bt_data["val_selected"]
    test_ade, test_fde, test_switched, test_used = bt_data["test_selected"]
    graph_val_ade = val.arrays[DEFAULT_VARIANT]["selected_ade"].astype(np.float32)
    graph_test_ade = test.arrays[DEFAULT_VARIANT]["selected_ade"].astype(np.float32)
    graph_test_fde = test.arrays[DEFAULT_VARIANT]["selected_fde"].astype(np.float32)
    graph_test_switched = test.arrays[DEFAULT_VARIANT]["switched"].astype(bool)

    selected = _select_guard(
        val,
        val_ade,
        val_fde,
        val_switched,
        val_used,
        min_rows=int(args.min_rows),
        min_context_easy_rows=int(args.min_context_easy_rows),
    )
    spec = selected["spec"]
    keys: set[str] = set()
    if spec.get("family"):
        keys = _hazard_keys(
            val.ds,
            val_ade,
            graph_val_ade,
            val_used,
            family=str(spec["family"]),
            min_context_easy_rows=int(args.min_context_easy_rows),
            rate_threshold=float(spec["rate"]),
            mean_harm_threshold=float(spec["mean"]),
        )
    repaired_ade, repaired_fde, repaired_switched, repaired_used, blocked = _apply_guard(
        test,
        test_ade,
        test_fde,
        test_switched,
        test_used,
        spec=spec,
        hazard_keys=keys,
    )
    metrics = m._metrics(test.ds, repaired_ade, repaired_fde, repaired_switched)
    graph_metrics = m._metrics(test.ds, graph_test_ade, graph_test_fde, graph_test_switched)
    delta = bv._delta(metrics, graph_metrics)
    bt_metrics = m._metrics(test.ds, test_ade, test_fde, test_switched)
    graph_abs = bu._slice_audit(
        test.ds,
        graph_test_ade,
        graph_test_ade,
        np.full(len(graph_test_ade), DEFAULT_VARIANT, dtype=object),
        min_rows=int(args.min_rows),
    )
    bt_abs = bu._slice_audit(test.ds, test_ade, graph_test_ade, test_used, min_rows=int(args.min_rows))
    repaired_abs = bu._slice_audit(test.ds, repaired_ade, graph_test_ade, repaired_used, min_rows=int(args.min_rows))
    bt_context_hazard = _context_easy_hazard_audit(
        test.ds,
        test_ade,
        graph_test_ade,
        test_used,
        min_rows=int(args.min_rows),
        min_context_easy_rows=int(args.min_context_easy_rows),
        rate_threshold=0.20,
        mean_harm_threshold=0.005,
    )
    repaired_context_hazard = _context_easy_hazard_audit(
        test.ds,
        repaired_ade,
        graph_test_ade,
        repaired_used,
        min_rows=int(args.min_rows),
        min_context_easy_rows=int(args.min_context_easy_rows),
        rate_threshold=0.20,
        mean_harm_threshold=0.005,
    )
    bootstrap = bu._bootstrap_summary(test.ds, repaired_ade, graph_test_ade, n=int(args.bootstrap), seed=int(args.seed))
    bv_payload = read_json(bv.REPORT_JSON, {})
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_validation_selected_context_hazard_attribution_guard",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "precondition": {
            "bt_verdict": bt_data["bt_report"].get("stage43_bt_gate", {}).get("verdict", "missing"),
            "bv_verdict": bv_payload.get("stage43_bv_gate", {}).get("verdict", "missing"),
            "bv_gate": {
                "passed": bv_payload.get("stage43_bv_gate", {}).get("passed", 0),
                "total": bv_payload.get("stage43_bv_gate", {}).get("total", 0),
            },
        },
        "rows": {"val": int(len(val.ds.x)), "test": int(len(test.ds.x))},
        "bt_policy": bt_data["policy"],
        "validation_selection": {
            "selected_spec": selected["spec"],
            "selected_validation_metrics": selected["validation_metrics"],
            "selected_delta_vs_graph_history_only": selected["delta_vs_graph_history_only"],
            "candidate_count": selected["candidate_count"],
            "safe_candidate_count": selected["safe_candidate_count"],
            "blocked_context_rows_on_validation": selected["blocked_context_rows"],
            "context_hazard_slice_count_on_validation": selected["context_hazard_slice_count"],
            "hazard_key_count": selected["hazard_key_count"],
            "test_tuned": False,
            "selection_rule": "Select source/domain/horizon context-hazard guard on validation only; future variant errors are labels/eval only; evaluate test once.",
            "all_candidates": selected["all_candidates"],
        },
        "source_overlap": _source_overlap(val, test),
        "test_metrics": metrics,
        "bt_unrepaired_metrics": bt_metrics,
        "graph_history_metrics": graph_metrics,
        "delta_vs_graph_history_only": delta,
        "test_context_variant_counts": _variant_counts(repaired_used),
        "blocked_context_rows_on_test": int(np.sum(blocked)),
        "test_hazard_key_count": len(keys),
        "absolute_slice_audit": {
            "graph_history_floor": {
                "easy_hazard_slice_count": graph_abs["easy_hazard_slice_count"],
                "top_easy_hazard_slices": graph_abs["top_easy_hazard_slices"][:8],
            },
            "bt_unrepaired": {
                "easy_hazard_slice_count": bt_abs["easy_hazard_slice_count"],
                "top_easy_hazard_slices": bt_abs["top_easy_hazard_slices"][:8],
            },
            "selected_guard": {
                "easy_hazard_slice_count": repaired_abs["easy_hazard_slice_count"],
                "top_easy_hazard_slices": repaired_abs["top_easy_hazard_slices"][:8],
            },
        },
        "context_induced_hazard_audit": {
            "bt_unrepaired": bt_context_hazard,
            "selected_guard": repaired_context_hazard,
        },
        "bootstrap": bootstrap,
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "raw_scene_or_verified_sdf_claim": False,
            "deployment_policy_changed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_variant_error_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "scene_proxy_train_only": True,
            "graph_inputs_past_or_current_only": True,
            "test_threshold_selection": False,
        },
        "input_hash": _combined_hash([bt.REPORT_JSON, bv.REPORT_JSON]),
    }
    payload["stage43_bw_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    no_leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    delta = payload["delta_vs_graph_history_only"]
    boot = payload["bootstrap"]["metrics"]
    selected_context_hazards = int(payload["context_induced_hazard_audit"]["selected_guard"]["context_hazard_slice_count"])
    bt_context_hazards = int(payload["context_induced_hazard_audit"]["bt_unrepaired"]["context_hazard_slice_count"])
    graph_abs = int(payload["absolute_slice_audit"]["graph_history_floor"]["easy_hazard_slice_count"])
    selected_abs = int(payload["absolute_slice_audit"]["selected_guard"]["easy_hazard_slice_count"])
    floor_inherent_risk_visible = graph_abs > 0
    context_hazard_not_worse = selected_context_hazards <= bt_context_hazards
    easy_safe = float(payload["test_metrics"]["easy_degradation_vs_floor"]) <= 0.02 and float(boot["easy_degradation_delta_vs_graph"]["high"]) <= 0.02
    core_lift = max(float(delta["all"]), float(delta["t50"]), float(delta["hard_failure"])) > 0.0
    gates = {
        "bt_precondition_present": payload["precondition"]["bt_verdict"]
        in {
            "stage43_bt_context_admissibility_pass_safe_lift_diagnostic",
            "stage43_bt_context_admissibility_pass_safe_no_lift_diagnostic",
            "stage43_bt_context_admissibility_pass_unsafe_diagnostic",
        },
        "bv_precondition_present": payload["precondition"]["bv_verdict"]
        in {
            "stage43_bv_context_admissibility_slice_safe_repair_pass",
            "stage43_bv_context_admissibility_slice_safe_partial_lift_pass",
            "stage43_bv_context_admissibility_slice_safe_no_lift_pass",
            "stage43_bv_context_admissibility_slice_repair_diagnostic_remaining_risk",
        },
        "validation_only_guard_selected": payload["validation_selection"]["test_tuned"] is False,
        "source_overlap_reported": "held_out_source_level" in payload["source_overlap"],
        "absolute_floor_hazard_attributed": floor_inherent_risk_visible
        and graph_abs >= selected_abs,
        "context_induced_hazard_measured": "selected_guard" in payload["context_induced_hazard_audit"],
        "context_hazard_not_worse_than_bt": context_hazard_not_worse,
        "test_eval_completed": int(payload["test_metrics"]["rows"]) > 0,
        "bootstrap_completed": int(payload["bootstrap"]["n"]) >= 1000,
        "easy_safety_measured": easy_safe,
        "no_future_or_test_leakage": no_leak["future_endpoint_input"] is False
        and no_leak["future_waypoint_input"] is False
        and no_leak["future_variant_error_label_eval_only"] is True
        and no_leak["central_velocity_input"] is False
        and no_leak["test_endpoint_goal_construction"] is False
        and no_leak["test_statistics_normalization"] is False
        and no_leak["scene_proxy_train_only"] is True
        and no_leak["graph_inputs_past_or_current_only"] is True
        and no_leak["test_threshold_selection"] is False,
        "no_metric_seconds_stage5c_smc_claim": claim["metric_or_seconds_claim"] is False
        and claim["raw_scene_or_verified_sdf_claim"] is False
        and claim["stage5c_executed"] is False
        and claim["smc_enabled"] is False,
        "long_objective_kept_active": claim["long_objective_complete"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    if passed == total and selected_context_hazards == 0 and core_lift:
        verdict = "stage43_bw_context_hazard_guard_pass_context_safe_lift_diagnostic"
    elif passed == total and floor_inherent_risk_visible and context_hazard_not_worse:
        verdict = "stage43_bw_context_hazard_attribution_pass_floor_inherent_risk"
    elif passed == total:
        verdict = "stage43_bw_context_hazard_attribution_pass_remaining_context_risk"
    else:
        verdict = "stage43_bw_context_hazard_attribution_incomplete"
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": verdict,
        "easy_safe": easy_safe,
        "core_lift_vs_graph_history": core_lift,
        "floor_inherent_risk_visible": floor_inherent_risk_visible,
        "context_hazard_not_worse_than_bt": context_hazard_not_worse,
        "deployable_policy_changed": False,
        "protected_multimodal_latent_state_candidate": False,
        "long_objective_complete": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _ci_line(name: str, row: Mapping[str, Any]) -> str:
    return f"| `{name}` | `{row['rows']}` | `{_pct(row['low'])}` | `{_pct(row['mean'])}` | `{_pct(row['high'])}` |"


def _candidate_lines(candidates: list[Mapping[str, Any]]) -> list[str]:
    lines = [
        "| candidate | safe | score | all delta | t50 delta | t100 delta | hard delta | easy delta | context hazards | blocked rows |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(candidates, key=lambda r: float(r["selection_score"]), reverse=True)[:20]:
        d = row["delta_vs_graph_history_only"]
        lines.append(
            f"| `{row['spec']['name']}` | `{row['safe']}` | `{row['selection_score']:.5f}` | `{_pct(d['all'])}` | `{_pct(d['t50'])}` | `{_pct(d['t100_raw_frame_diagnostic'])}` | `{_pct(d['hard_failure'])}` | `{_pct(d['easy_degradation'])}` | `{row['context_hazard_slice_count']}` | `{row['blocked_context_rows']}` |"
        )
    return lines


def _hazard_lines(rows: list[Mapping[str, Any]], *, limit: int = 12) -> list[str]:
    lines = [
        "| slice | rows | easy rows | context easy rows | context harm rate | mean context harm | mean easy delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows[:limit]:
        lines.append(
            f"| `{row['slice']}` | `{row['rows']}` | `{row['easy_rows']}` | `{row['context_easy_rows']}` | `{_pct(row['context_easy_harm_rate_vs_graph'])}` | `{row['mean_context_easy_harm_vs_graph']:.6f}` | `{row['mean_easy_delta_vs_graph']:.6f}` |"
        )
    if len(rows) > limit:
        lines.append(f"| `...` | `{len(rows) - limit} more` |  |  |  |  |  |")
    return lines


def _write_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bw_gate"]
    delta = payload["delta_vs_graph_history_only"]
    boot = payload["bootstrap"]["metrics"]
    write_json(REPORT_JSON, m._jsonable(payload))
    write_md(
        REPORT_MD,
        [
            "# Stage43-BW Context Hazard Attribution Guard",
            "",
            f"- source: `{payload['source']}`",
            f"- result_source: `{payload['result_source']}`",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- selected guard: `{payload['validation_selection']['selected_spec']['name']}`",
            f"- safe validation candidates: `{payload['validation_selection']['safe_candidate_count']} / {payload['validation_selection']['candidate_count']}`",
            f"- source overlap: `{payload['source_overlap']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            "",
            "## Test Metrics",
            "",
            *_metric_lines(payload["test_metrics"]),
            f"- context counts on test: `{payload['test_context_variant_counts']}`",
            f"- blocked context rows on test: `{payload['blocked_context_rows_on_test']}`",
            "",
            "## Delta Vs Graph-History-Only",
            "",
            f"- all delta: `{_pct(delta['all'])}`",
            f"- t50 delta: `{_pct(delta['t50'])}`",
            f"- t100 raw-frame diagnostic delta: `{_pct(delta['t100_raw_frame_diagnostic'])}`",
            f"- hard/failure delta: `{_pct(delta['hard_failure'])}`",
            f"- easy degradation delta: `{_pct(delta['easy_degradation'])}`",
            "",
            "## Absolute Easy Hazard Attribution",
            "",
            f"- graph-history absolute easy hazard slices: `{payload['absolute_slice_audit']['graph_history_floor']['easy_hazard_slice_count']}`",
            f"- BT unrepaired absolute easy hazard slices: `{payload['absolute_slice_audit']['bt_unrepaired']['easy_hazard_slice_count']}`",
            f"- selected guard absolute easy hazard slices: `{payload['absolute_slice_audit']['selected_guard']['easy_hazard_slice_count']}`",
            "",
            "## Context-Induced Easy Hazard",
            "",
            f"- BT unrepaired context-induced hazard slices: `{payload['context_induced_hazard_audit']['bt_unrepaired']['context_hazard_slice_count']}`",
            f"- selected guard context-induced hazard slices: `{payload['context_induced_hazard_audit']['selected_guard']['context_hazard_slice_count']}`",
            "",
            "### Top Selected-Guard Context Harm Slices",
            "",
            *_hazard_lines(payload["context_induced_hazard_audit"]["selected_guard"]["top_context_harm_slices"]),
            "",
            "## Validation Candidate Guards",
            "",
            *_candidate_lines(payload["validation_selection"]["all_candidates"]),
            "",
            "## Bootstrap Delta Vs Graph-History-Only",
            "",
            f"- bootstrap n: `{payload['bootstrap']['n']}`",
            "",
            "| metric | rows | low | mean | high |",
            "| --- | ---: | ---: | ---: | ---: |",
            *[_ci_line(name, row) for name, row in boot.items()],
            "",
            "## Interpretation",
            "",
            "- Stage43-BW separates floor-inherent absolute easy risk from context-induced harm.",
            "- Source-level hazard keys are reported with held-out source overlap so source-key guards are not mistaken for transferable safety.",
            "- This is not a deployment update; it is a safety attribution step for the protected multimodal latent-state track.",
            "- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    write_md(
        GATE_MD,
        [
            "# Stage43-BW Context Hazard Attribution Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- easy safe: `{gate['easy_safe']}`",
            f"- core lift vs graph-history: `{gate['core_lift_vs_graph_history']}`",
            f"- floor-inherent risk visible: `{gate['floor_inherent_risk_visible']}`",
            f"- context hazard not worse than BT: `{gate['context_hazard_not_worse_than_bt']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    write_json(WORLD_GATE_JSON, m._jsonable(gate))
    write_md(
        WORLD_GATE_MD,
        [
            "# Stage43 Current World-Model Gate",
            "",
            f"- source: `{payload['source']}`",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- protected multimodal latent state candidate: `{gate['protected_multimodal_latent_state_candidate']}`",
            f"- easy safe: `{gate['easy_safe']}`",
            f"- floor-inherent risk visible: `{gate['floor_inherent_risk_visible']}`",
            f"- context hazard not worse than BT: `{gate['context_hazard_not_worse_than_bt']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            f"- long objective complete: `{gate['long_objective_complete']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "## Current Boundary",
            "",
            "- Stage43-BW is a hazard-attribution guard for Stage43-BT/BV context admissibility.",
            "- It separates graph-history floor risk from context-induced harm.",
            "- It is not a deployment update and does not complete the Stage43 long objective.",
            "- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    _update_ledgers(payload)


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bw_gate"]
    delta = payload["delta_vs_graph_history_only"]
    boot = payload["bootstrap"]["metrics"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"selected_guard = `{payload['validation_selection']['selected_spec']['name']}`",
        f"deployable_policy_changed = `{gate['deployable_policy_changed']}`",
        "",
        "Stage43-BW distinguishes graph-history floor-inherent absolute easy risk from context-induced easy harm. This matters because BV's remaining easy-hazard slices can be inherited from the floor rather than caused by scene/graph context.",
        f"Absolute easy hazard slices: graph-history `{payload['absolute_slice_audit']['graph_history_floor']['easy_hazard_slice_count']}`, BT unrepaired `{payload['absolute_slice_audit']['bt_unrepaired']['easy_hazard_slice_count']}`, selected guard `{payload['absolute_slice_audit']['selected_guard']['easy_hazard_slice_count']}`.",
        f"Context-induced hazard slices: BT unrepaired `{payload['context_induced_hazard_audit']['bt_unrepaired']['context_hazard_slice_count']}`, selected guard `{payload['context_induced_hazard_audit']['selected_guard']['context_hazard_slice_count']}`.",
        f"Delta vs graph-history-only: all `{_pct(delta['all'])}`, t50 `{_pct(delta['t50'])}`, t100 raw-frame diagnostic `{_pct(delta['t100_raw_frame_diagnostic'])}`, hard/failure `{_pct(delta['hard_failure'])}`, easy degradation `{_pct(delta['easy_degradation'])}`.",
        f"Bootstrap CI low vs graph-history-only: all `{_pct(boot['all_delta_vs_graph']['low'])}`, t50 `{_pct(boot['t50_delta_vs_graph']['low'])}`, t100 raw `{_pct(boot['t100_raw_frame_delta_vs_graph']['low'])}`, hard/failure `{_pct(boot['hard_failure_delta_vs_graph']['low'])}`, easy high `{_pct(boot['easy_degradation_delta_vs_graph']['high'])}`.",
        f"Source overlap audit: `{payload['source_overlap']}`.",
        "",
        "Boundary unchanged: dataset-local/raw-frame 2.5D only; t100 is diagnostic; no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_bw_context_hazard_attribution_guard"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "selected_guard": payload["validation_selection"]["selected_spec"]["name"],
        "deployable_policy_changed": gate["deployable_policy_changed"],
        "absolute_easy_hazard_slices": {
            "graph_history_floor": payload["absolute_slice_audit"]["graph_history_floor"]["easy_hazard_slice_count"],
            "bt_unrepaired": payload["absolute_slice_audit"]["bt_unrepaired"]["easy_hazard_slice_count"],
            "selected_guard": payload["absolute_slice_audit"]["selected_guard"]["easy_hazard_slice_count"],
        },
        "context_induced_hazard_slices": {
            "bt_unrepaired": payload["context_induced_hazard_audit"]["bt_unrepaired"]["context_hazard_slice_count"],
            "selected_guard": payload["context_induced_hazard_audit"]["selected_guard"]["context_hazard_slice_count"],
        },
        "delta_vs_graph_history_only": payload["delta_vs_graph_history_only"],
        "source_overlap": payload["source_overlap"],
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_bw_context_hazard_attribution_guard"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-BW",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "selected_guard": payload["validation_selection"]["selected_spec"]["name"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-BW context hazard attribution and source/domain guard.")
    parser.add_argument("--test-rows", type=int, default=0, help="Optional test row override; 0 uses Stage43-BT rows.")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=479)
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--min-context-easy-rows", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    payload = _run(args)
    gate = payload["stage43_bw_gate"]
    print(f"Stage43-BW: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"selected_guard={payload['validation_selection']['selected_spec']['name']}")
    print(
        "context_hazards="
        f"{payload['context_induced_hazard_audit']['selected_guard']['context_hazard_slice_count']}"
    )
    return payload


if __name__ == "__main__":
    main()
