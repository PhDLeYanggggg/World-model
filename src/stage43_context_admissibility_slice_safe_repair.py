from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_context_admissibility_model as bt
from src import stage43_context_admissibility_robustness_audit as bu


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_context_admissibility_slice_safe_repair.json"
REPORT_MD = OUT_DIR / "stage43_context_admissibility_slice_safe_repair.md"
GATE_MD = OUT_DIR / "stage43_stage_bv_context_admissibility_slice_safe_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_BV_CONTEXT_ADMISSIBILITY_SLICE_SAFE_REPAIR"
SOURCE = "fresh_stage43_bv_context_admissibility_slice_safe_repair"
DEFAULT_VARIANT = bt.DEFAULT_VARIANT
EPS = 1e-8


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _key_arrays(ds: m.WaypointSplit) -> dict[str, np.ndarray]:
    domain = ds.domain.astype(str)
    source = ds.source_file.astype(str)
    horizon = ds.horizon.astype(int).astype(str)
    return {
        "horizon": np.char.add("horizon:", horizon),
        "domain": np.char.add("domain:", domain),
        "domain_horizon": np.char.add(np.char.add(np.char.add("domain_horizon:", domain), "|h:"), horizon),
        "source": np.char.add("source:", source),
        "source_horizon": np.char.add(np.char.add(np.char.add("source_horizon:", source), "|h:"), horizon),
    }


def _slice_table(
    ds: m.WaypointSplit,
    selected_ade: np.ndarray,
    graph_ade: np.ndarray,
    used: np.ndarray,
    *,
    min_rows: int,
    min_delta: float,
    easy_limit: float,
) -> dict[str, Any]:
    keys = _key_arrays(ds)
    tables: dict[str, dict[str, Any]] = {}
    for family, arr in keys.items():
        table: dict[str, Any] = {}
        for key in sorted(set(arr.tolist())):
            mask = arr == key
            if int(mask.sum()) < int(min_rows):
                continue
            selected_imp = bu._slice_improvement(selected_ade, ds.floor_ade, mask)
            graph_imp = bu._slice_improvement(graph_ade, ds.floor_ade, mask)
            delta = float(selected_imp - graph_imp)
            easy_deg = bu._easy_degradation(selected_ade[mask], ds.floor_ade[mask], ds.easy[mask])
            context_rate = float(np.mean(used[mask].astype(str) != DEFAULT_VARIANT)) if int(mask.sum()) else 0.0
            table[str(key)] = {
                "key": str(key),
                "family": family,
                "rows": int(mask.sum()),
                "easy_rows": int(np.sum(ds.easy[mask])),
                "selected_improvement": selected_imp,
                "graph_history_improvement": graph_imp,
                "delta_vs_graph": delta,
                "easy_degradation": easy_deg,
                "context_rate": context_rate,
                "safe": bool(delta >= float(min_delta) and easy_deg <= float(easy_limit)),
                "unsafe": bool(delta < -float(min_delta) or easy_deg > float(easy_limit)),
            }
        tables[family] = table
    summary = {
        family: {
            "keys": len(table),
            "safe_keys": int(sum(1 for row in table.values() if row["safe"])),
            "unsafe_keys": int(sum(1 for row in table.values() if row["unsafe"])),
        }
        for family, table in tables.items()
    }
    return {"min_rows": int(min_rows), "min_delta": float(min_delta), "easy_limit": float(easy_limit), "families": tables, "summary": summary}


def _is_safe(table: Mapping[str, Any], family: str, key: str) -> bool:
    row = table["families"].get(family, {}).get(key)
    return bool(row and row.get("safe", False))


def _is_unsafe(table: Mapping[str, Any], family: str, key: str) -> bool:
    row = table["families"].get(family, {}).get(key)
    return bool(row and row.get("unsafe", False))


def _block_mask(ds: m.WaypointSplit, used: np.ndarray, table: Mapping[str, Any], mode: str) -> np.ndarray:
    keys = _key_arrays(ds)
    context = used.astype(str) != DEFAULT_VARIANT
    blocked = np.zeros(len(used), dtype=bool)
    if mode == "bt_unrepaired":
        return blocked
    if mode == "all_fallback":
        return context
    if mode == "block_t100":
        return context & (ds.horizon == 100)
    for i in range(len(used)):
        if not context[i]:
            continue
        h_key = str(keys["horizon"][i])
        dh_key = str(keys["domain_horizon"][i])
        sh_key = str(keys["source_horizon"][i])
        d_key = str(keys["domain"][i])
        s_key = str(keys["source"][i])
        if mode == "block_unsafe_horizon":
            blocked[i] = _is_unsafe(table, "horizon", h_key)
        elif mode == "block_unsafe_domain_horizon":
            blocked[i] = _is_unsafe(table, "domain_horizon", dh_key) or _is_unsafe(table, "horizon", h_key)
        elif mode == "block_unsafe_source_horizon":
            blocked[i] = _is_unsafe(table, "source_horizon", sh_key) or _is_unsafe(table, "domain_horizon", dh_key) or _is_unsafe(table, "horizon", h_key)
        elif mode == "hierarchical_any_unsafe":
            blocked[i] = (
                _is_unsafe(table, "source_horizon", sh_key)
                or _is_unsafe(table, "source", s_key)
                or _is_unsafe(table, "domain_horizon", dh_key)
                or _is_unsafe(table, "domain", d_key)
                or _is_unsafe(table, "horizon", h_key)
            )
        elif mode == "require_domain_or_source_safe":
            blocked[i] = not (_is_safe(table, "source_horizon", sh_key) or _is_safe(table, "domain_horizon", dh_key) or _is_safe(table, "domain", d_key))
            blocked[i] = blocked[i] or _is_unsafe(table, "horizon", h_key)
        elif mode == "strict_safe_no_t100":
            blocked[i] = ds.horizon[i] == 100 or not (
                _is_safe(table, "source_horizon", sh_key) or (_is_safe(table, "domain_horizon", dh_key) and not _is_unsafe(table, "source", s_key))
            )
        elif mode == "source_or_domain_safe_and_horizon_safe":
            blocked[i] = not (_is_safe(table, "source_horizon", sh_key) or _is_safe(table, "domain_horizon", dh_key))
            blocked[i] = blocked[i] or _is_unsafe(table, "horizon", h_key)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return blocked


def _apply_repair(
    batch: bt.ContextBatch,
    selected_ade: np.ndarray,
    selected_fde: np.ndarray,
    switched: np.ndarray,
    used: np.ndarray,
    table: Mapping[str, Any],
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    graph_ade = batch.arrays[DEFAULT_VARIANT]["selected_ade"].astype(np.float32)
    graph_fde = batch.arrays[DEFAULT_VARIANT]["selected_fde"].astype(np.float32)
    graph_switched = batch.arrays[DEFAULT_VARIANT]["switched"].astype(bool)
    blocked = _block_mask(batch.ds, used, table, mode)
    out_ade = selected_ade.copy()
    out_fde = selected_fde.copy()
    out_switched = switched.copy()
    out_used = used.copy()
    out_ade[blocked] = graph_ade[blocked]
    out_fde[blocked] = graph_fde[blocked]
    out_switched[blocked] = graph_switched[blocked]
    out_used[blocked] = DEFAULT_VARIANT
    return out_ade.astype(np.float32), out_fde.astype(np.float32), out_switched.astype(bool), out_used, blocked


def _delta(metrics: Mapping[str, Any], graph: Mapping[str, Any]) -> dict[str, float]:
    return {
        "all": float(metrics["full_waypoint_ade_improvement_vs_floor"] - graph["full_waypoint_ade_improvement_vs_floor"]),
        "t50": float(metrics["t50_full_waypoint_ade_improvement_vs_floor"] - graph["t50_full_waypoint_ade_improvement_vs_floor"]),
        "t100_raw_frame_diagnostic": float(
            metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] - graph["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]
        ),
        "hard_failure": float(metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"] - graph["hard_failure_full_waypoint_ade_improvement_vs_floor"]),
        "easy_degradation": float(metrics["easy_degradation_vs_floor"] - graph["easy_degradation_vs_floor"]),
    }


def _variant_counts(values: np.ndarray) -> dict[str, int]:
    return {str(v): int(np.sum(values.astype(str) == str(v))) for v in sorted(set(values.astype(str).tolist()))}


def _candidate_modes() -> list[str]:
    return [
        "bt_unrepaired",
        "all_fallback",
        "block_t100",
        "block_unsafe_horizon",
        "block_unsafe_domain_horizon",
        "block_unsafe_source_horizon",
        "hierarchical_any_unsafe",
        "require_domain_or_source_safe",
        "source_or_domain_safe_and_horizon_safe",
        "strict_safe_no_t100",
    ]


def _score(delta: Mapping[str, float], slice_easy_hazards: int, context_rate: float) -> float:
    return float(
        delta["all"]
        + 1.2 * delta["t50"]
        + 1.0 * delta["hard_failure"]
        + 0.5 * min(0.0, delta["t100_raw_frame_diagnostic"])
        - 25.0 * max(0.0, delta["easy_degradation"])
        - 0.01 * context_rate
        - 0.01 * int(slice_easy_hazards)
    )


def _select_repair(
    val: bt.ContextBatch,
    bt_selected_ade: np.ndarray,
    bt_selected_fde: np.ndarray,
    bt_switched: np.ndarray,
    bt_used: np.ndarray,
    table: Mapping[str, Any],
    *,
    min_rows: int,
) -> dict[str, Any]:
    graph_metrics = m._metrics(
        val.ds,
        val.arrays[DEFAULT_VARIANT]["selected_ade"].astype(np.float32),
        val.arrays[DEFAULT_VARIANT]["selected_fde"].astype(np.float32),
        val.arrays[DEFAULT_VARIANT]["switched"].astype(bool),
    )
    candidates: list[dict[str, Any]] = []
    for mode in _candidate_modes():
        ade, fde, switched, used, blocked = _apply_repair(val, bt_selected_ade, bt_selected_fde, bt_switched, bt_used, table, mode)
        metrics = m._metrics(val.ds, ade, fde, switched)
        delta = _delta(metrics, graph_metrics)
        slices = bu._slice_audit(val.ds, ade, val.arrays[DEFAULT_VARIANT]["selected_ade"].astype(np.float32), used, min_rows=min_rows)
        context_rate = float(np.mean(used.astype(str) != DEFAULT_VARIANT))
        safe = (
            metrics["easy_degradation_vs_floor"] <= 0.02
            and delta["easy_degradation"] <= 0.02
            and int(slices["easy_hazard_slice_count"]) == 0
            and delta["all"] >= -0.001
            and delta["t50"] >= -0.001
            and delta["hard_failure"] >= -0.001
        )
        candidates.append(
            {
                "mode": mode,
                "validation_metrics": metrics,
                "delta_vs_graph_history_only": delta,
                "context_variant_counts": _variant_counts(used),
                "context_rate": context_rate,
                "blocked_context_rows": int(np.sum(blocked)),
                "slice_easy_hazard_count": int(slices["easy_hazard_slice_count"]),
                "safe": bool(safe),
                "selection_score": _score(delta, int(slices["easy_hazard_slice_count"]), context_rate),
            }
        )
    safe_candidates = [row for row in candidates if row["safe"]]
    selected_base = max(safe_candidates or candidates, key=lambda row: row["selection_score"])
    selected = dict(selected_base)
    selected["safe_candidate_count"] = len(safe_candidates)
    selected["candidate_count"] = len(candidates)
    selected["all_candidates"] = [dict(row) for row in candidates]
    return selected


def _load_bt_predictions(rows: int | None = None) -> dict[str, Any]:
    report = read_json(bt.REPORT_JSON, {})
    if not report:
        raise FileNotFoundError(bt.REPORT_JSON)
    train_rows = int(report.get("rows", {}).get("train", 20000))
    val_rows = int(report.get("rows", {}).get("val", 12000))
    test_rows = int(rows or report.get("rows", {}).get("test", 12000))
    train = bt._load_context("train", rows=train_rows)
    val = bt._load_context("val", rows=val_rows)
    test = bt._load_context("test", rows=test_rows)
    bt._standardize_batches(train, val, test)
    ckpt_path = Path(report["model"]["checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = bt.ContextAdmissibilityNet(int(ckpt["input_dim"]), len(bt.CONTEXT_VARIANTS), hidden_dim=int(ckpt.get("hidden_dim", 96)))
    model.load_state_dict(ckpt["model_state"])
    val_pred = bt._predict_model(model, val.ds.x, batch_size=4096)
    test_pred = bt._predict_model(model, test.ds.x, batch_size=4096)
    policy = report["validation_selection"]["selected_policy"]
    val_selected = bt._apply_policy(val, val_pred, policy)
    test_selected = bt._apply_policy(test, test_pred, policy)
    return {
        "bt_report": report,
        "train": train,
        "val": val,
        "test": test,
        "policy": policy,
        "val_selected": val_selected,
        "test_selected": test_selected,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": m._sha256(ckpt_path),
    }


def _bootstrap(ds: m.WaypointSplit, selected_ade: np.ndarray, graph_ade: np.ndarray, *, n: int, seed: int) -> dict[str, Any]:
    return bu._bootstrap_summary(ds, selected_ade, graph_ade, n=n, seed=seed)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    bt_data = _load_bt_predictions(rows=int(args.test_rows) if args.test_rows else None)
    val = bt_data["val"]
    test = bt_data["test"]
    val_ade, val_fde, val_switched, val_used = bt_data["val_selected"]
    test_ade, test_fde, test_switched, test_used = bt_data["test_selected"]
    table = _slice_table(
        val.ds,
        val_ade,
        val.arrays[DEFAULT_VARIANT]["selected_ade"].astype(np.float32),
        val_used,
        min_rows=int(args.min_rows),
        min_delta=float(args.min_delta),
        easy_limit=float(args.easy_limit),
    )
    selected_policy = _select_repair(val, val_ade, val_fde, val_switched, val_used, table, min_rows=int(args.min_rows))
    repaired_ade, repaired_fde, repaired_switched, repaired_used, blocked = _apply_repair(
        test,
        test_ade,
        test_fde,
        test_switched,
        test_used,
        table,
        selected_policy["mode"],
    )
    graph_ade = test.arrays[DEFAULT_VARIANT]["selected_ade"].astype(np.float32)
    graph_fde = test.arrays[DEFAULT_VARIANT]["selected_fde"].astype(np.float32)
    graph_switched = test.arrays[DEFAULT_VARIANT]["switched"].astype(bool)
    metrics = m._metrics(test.ds, repaired_ade, repaired_fde, repaired_switched)
    graph_metrics = m._metrics(test.ds, graph_ade, graph_fde, graph_switched)
    delta = _delta(metrics, graph_metrics)
    bootstrap = _bootstrap(test.ds, repaired_ade, graph_ade, n=int(args.bootstrap), seed=int(args.seed))
    slices = bu._slice_audit(test.ds, repaired_ade, graph_ade, repaired_used, min_rows=int(args.min_rows))
    bu_payload = read_json(bu.REPORT_JSON, {})
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_validation_selected_slice_safe_context_repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "precondition": {
            "bt_verdict": bt_data["bt_report"].get("stage43_bt_gate", {}).get("verdict", "missing"),
            "bu_verdict": bu_payload.get("stage43_bu_gate", {}).get("verdict", "missing"),
            "bu_gate": {
                "passed": bu_payload.get("stage43_bu_gate", {}).get("passed", 0),
                "total": bu_payload.get("stage43_bu_gate", {}).get("total", 0),
            },
        },
        "rows": {"val": int(len(val.ds.x)), "test": int(len(test.ds.x))},
        "bt_policy": bt_data["policy"],
        "checkpoint": {"path": bt_data["checkpoint"], "sha256": bt_data["checkpoint_sha256"], "committed": False},
        "validation_slice_table": table,
        "validation_selection": {
            "selected_mode": selected_policy["mode"],
            "selected_validation_metrics": selected_policy["validation_metrics"],
            "selected_delta_vs_graph_history_only": selected_policy["delta_vs_graph_history_only"],
            "selected_context_variant_counts": selected_policy["context_variant_counts"],
            "safe_candidate_count": selected_policy["safe_candidate_count"],
            "candidate_count": selected_policy["candidate_count"],
            "blocked_context_rows_on_validation": selected_policy["blocked_context_rows"],
            "slice_easy_hazard_count_on_validation": selected_policy["slice_easy_hazard_count"],
            "test_tuned": False,
            "selection_rule": "Build validation slice safety table from BT predictions; choose a repair mode on validation only; evaluate test once.",
            "all_candidates": selected_policy["all_candidates"],
        },
        "test_metrics": metrics,
        "graph_history_metrics": graph_metrics,
        "delta_vs_graph_history_only": delta,
        "test_context_variant_counts": _variant_counts(repaired_used),
        "blocked_context_rows_on_test": int(np.sum(blocked)),
        "bootstrap": bootstrap,
        "slice_audit": slices,
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
        "input_hash": _combined_hash([bt.REPORT_JSON, bu.REPORT_JSON]),
    }
    payload["stage43_bv_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    gate_bu_passed = int(payload["precondition"]["bu_gate"]["passed"]) == int(payload["precondition"]["bu_gate"]["total"])
    no_leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    boot = payload["bootstrap"]["metrics"]
    delta = payload["delta_vs_graph_history_only"]
    t100_low = float(boot["t100_raw_frame_delta_vs_graph"]["low"])
    t100_high = float(boot["t100_raw_frame_delta_vs_graph"]["high"])
    easy_high = float(boot["easy_degradation_delta_vs_graph"]["high"])
    slice_easy_safe = int(payload["slice_audit"].get("easy_hazard_slice_count", 0)) == 0
    easy_safe = float(payload["test_metrics"]["easy_degradation_vs_floor"]) <= 0.02 and easy_high <= 0.02
    core_lift = max(float(delta["all"]), float(delta["t50"]), float(delta["hard_failure"])) > 0.0
    t100_ci_crosses_zero = t100_low <= 0.0 <= t100_high
    gates = {
        "bt_precondition_passed": payload["precondition"]["bt_verdict"]
        in {
            "stage43_bt_context_admissibility_pass_safe_lift_diagnostic",
            "stage43_bt_context_admissibility_pass_safe_no_lift_diagnostic",
            "stage43_bt_context_admissibility_pass_unsafe_diagnostic",
        },
        "bu_precondition_passed": gate_bu_passed
        and payload["precondition"]["bu_verdict"]
        in {
            "stage43_bu_context_admissibility_robust_lift_pass",
            "stage43_bu_context_admissibility_partial_robust_lift_pass",
            "stage43_bu_context_admissibility_fragile_lift_diagnostic_pass",
        },
        "validation_slice_table_built": bool(payload["validation_slice_table"]["summary"]),
        "validation_only_repair_selected": payload["validation_selection"]["test_tuned"] is False,
        "test_eval_completed": int(payload["test_metrics"]["rows"]) > 0,
        "bootstrap_completed": int(payload["bootstrap"]["n"]) >= 1000,
        "slice_audit_completed": int(payload["slice_audit"]["slice_count"]) > 0,
        "t50_t100_reported": "t50_delta_vs_graph" in boot and "t100_raw_frame_delta_vs_graph" in boot,
        "easy_safety_measured": "easy_degradation_delta_vs_graph" in boot,
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
    if passed == total and easy_safe and slice_easy_safe and core_lift and t100_low > 0.0:
        verdict = "stage43_bv_context_admissibility_slice_safe_repair_pass"
    elif passed == total and easy_safe and slice_easy_safe and core_lift:
        verdict = "stage43_bv_context_admissibility_slice_safe_partial_lift_pass"
    elif passed == total and easy_safe and slice_easy_safe:
        verdict = "stage43_bv_context_admissibility_slice_safe_no_lift_pass"
    elif passed == total:
        verdict = "stage43_bv_context_admissibility_slice_repair_diagnostic_remaining_risk"
    else:
        verdict = "stage43_bv_context_admissibility_slice_repair_incomplete"
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": verdict,
        "easy_safe": easy_safe,
        "slice_easy_safe": slice_easy_safe,
        "core_lift_vs_graph_history": core_lift,
        "t100_bootstrap_robust": t100_low > 0.0,
        "t100_ci_crosses_zero": t100_ci_crosses_zero,
        "deployable_policy_changed": False,
        "protected_multimodal_latent_state_candidate": passed == total and easy_safe and slice_easy_safe and core_lift,
        "long_objective_complete": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _metric_lines(metrics: Mapping[str, Any]) -> list[str]:
    return [
        f"- all full-waypoint ADE improvement: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic improvement: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- hard/failure improvement: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- switch rate: `{_pct(metrics['switch_rate'])}`",
    ]


def _ci_line(name: str, row: Mapping[str, Any]) -> str:
    return f"| `{name}` | `{row['rows']}` | `{_pct(row['low'])}` | `{_pct(row['mean'])}` | `{_pct(row['high'])}` |"


def _candidate_lines(candidates: list[Mapping[str, Any]]) -> list[str]:
    lines = ["| mode | safe | score | all delta | t50 delta | t100 delta | hard delta | easy delta | hazards | context rate |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in sorted(candidates, key=lambda r: float(r["selection_score"]), reverse=True):
        d = row["delta_vs_graph_history_only"]
        lines.append(
            f"| `{row['mode']}` | `{row['safe']}` | `{row['selection_score']:.5f}` | `{_pct(d['all'])}` | `{_pct(d['t50'])}` | `{_pct(d['t100_raw_frame_diagnostic'])}` | `{_pct(d['hard_failure'])}` | `{_pct(d['easy_degradation'])}` | `{row['slice_easy_hazard_count']}` | `{_pct(row['context_rate'])}` |"
        )
    return lines


def _slice_table_lines(rows: list[Mapping[str, Any]], limit: int = 12) -> list[str]:
    lines = ["| slice | rows | delta vs graph | context rate | easy degradation |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows[:limit]:
        lines.append(
            f"| `{row['slice']}` | `{row['rows']}` | `{_pct(row['delta_vs_graph'])}` | `{_pct(row['context_rate'])}` | `{_pct(row['easy_degradation'])}` |"
        )
    if len(rows) > limit:
        lines.append(f"| `...` | `{len(rows) - limit} more` |  |  |  |")
    return lines


def _write_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bv_gate"]
    boot = payload["bootstrap"]["metrics"]
    delta = payload["delta_vs_graph_history_only"]
    write_json(REPORT_JSON, m._jsonable(payload))
    write_md(
        REPORT_MD,
        [
            "# Stage43-BV Context Admissibility Slice-Safe Repair",
            "",
            f"- source: `{payload['source']}`",
            f"- result_source: `{payload['result_source']}`",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- selected repair mode: `{payload['validation_selection']['selected_mode']}`",
            f"- safe validation candidates: `{payload['validation_selection']['safe_candidate_count']} / {payload['validation_selection']['candidate_count']}`",
            f"- easy safe: `{gate['easy_safe']}`",
            f"- slice easy safe: `{gate['slice_easy_safe']}`",
            f"- t100 bootstrap robust: `{gate['t100_bootstrap_robust']}`",
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
            "## Validation Candidate Repair Policies",
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
            "## Test Slice Audit",
            "",
            f"- slice count: `{payload['slice_audit']['slice_count']}`",
            f"- positive slice count: `{payload['slice_audit']['positive_slice_count']}`",
            f"- negative slice count: `{payload['slice_audit']['negative_slice_count']}`",
            f"- easy hazard slice count: `{payload['slice_audit']['easy_hazard_slice_count']}`",
            f"- core weak slices: `{[row['slice'] for row in payload['slice_audit']['core_weak_slices']]}`",
            "",
            "### Top Easy Hazard Slices",
            "",
            *_slice_table_lines(payload["slice_audit"]["top_easy_hazard_slices"]),
            "",
            "### Top Negative Slices",
            "",
            *_slice_table_lines(payload["slice_audit"]["top_negative_slices"]),
            "",
            "## Interpretation",
            "",
            "- Stage43-BV uses validation slice evidence to repair Stage43-BT context admissibility hazards.",
            "- It does not retrain the BT MLP and does not tune on test.",
            "- Future variant errors are validation/eval labels only, not inference inputs.",
            "- This is a safety repair / diagnostic step; it is not a deployment update unless gates support it.",
            "- Dataset-local/raw-frame 2.5D only; t100 remains raw-frame diagnostic; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
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
            "# Stage43-BV Context Admissibility Slice-Safe Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- easy safe: `{gate['easy_safe']}`",
            f"- slice easy safe: `{gate['slice_easy_safe']}`",
            f"- core lift vs graph-history: `{gate['core_lift_vs_graph_history']}`",
            f"- t100 bootstrap robust: `{gate['t100_bootstrap_robust']}`",
            f"- t100 CI crosses zero: `{gate['t100_ci_crosses_zero']}`",
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
            f"- slice easy safe: `{gate['slice_easy_safe']}`",
            f"- core lift vs graph-history: `{gate['core_lift_vs_graph_history']}`",
            f"- t100 bootstrap robust: `{gate['t100_bootstrap_robust']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            f"- long objective complete: `{gate['long_objective_complete']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "## Current Boundary",
            "",
            "- Stage43-BV is a validation-only slice-safety repair for Stage43-BT context admissibility.",
            "- It is not a deployment update unless it safely improves the graph-history floor.",
            "- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    _update_ledgers(payload)


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bv_gate"]
    delta = payload["delta_vs_graph_history_only"]
    boot = payload["bootstrap"]["metrics"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"selected_repair_mode = `{payload['validation_selection']['selected_mode']}`",
        f"easy_safe = `{gate['easy_safe']}`",
        f"slice_easy_safe = `{gate['slice_easy_safe']}`",
        f"core_lift_vs_graph_history = `{gate['core_lift_vs_graph_history']}`",
        f"t100_bootstrap_robust = `{gate['t100_bootstrap_robust']}`",
        f"deployable_policy_changed = `{gate['deployable_policy_changed']}`",
        "",
        "Stage43-BV applies a validation-only slice-safety repair to Stage43-BT context admissibility. It blocks context on validation-identified unsafe source/domain/horizon slices and evaluates test once.",
        f"Delta vs graph-history-only: all `{_pct(delta['all'])}`, t50 `{_pct(delta['t50'])}`, t100 raw-frame diagnostic `{_pct(delta['t100_raw_frame_diagnostic'])}`, hard/failure `{_pct(delta['hard_failure'])}`, easy degradation `{_pct(delta['easy_degradation'])}`.",
        f"Bootstrap CI low vs graph-history-only: all `{_pct(boot['all_delta_vs_graph']['low'])}`, t50 `{_pct(boot['t50_delta_vs_graph']['low'])}`, t100 raw `{_pct(boot['t100_raw_frame_delta_vs_graph']['low'])}`, hard/failure `{_pct(boot['hard_failure_delta_vs_graph']['low'])}`, easy high `{_pct(boot['easy_degradation_delta_vs_graph']['high'])}`.",
        f"Slice audit: positive `{payload['slice_audit']['positive_slice_count']}`, negative `{payload['slice_audit']['negative_slice_count']}`, easy hazards `{payload['slice_audit']['easy_hazard_slice_count']}`, core weak `{[row['slice'] for row in payload['slice_audit']['core_weak_slices']]}`.",
        "",
        "Boundary unchanged: dataset-local/raw-frame 2.5D only; t100 is diagnostic; no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_bv_context_admissibility_slice_safe_repair"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "selected_repair_mode": payload["validation_selection"]["selected_mode"],
        "easy_safe": gate["easy_safe"],
        "slice_easy_safe": gate["slice_easy_safe"],
        "core_lift_vs_graph_history": gate["core_lift_vs_graph_history"],
        "t100_bootstrap_robust": gate["t100_bootstrap_robust"],
        "deployable_policy_changed": gate["deployable_policy_changed"],
        "delta_vs_graph_history_only": payload["delta_vs_graph_history_only"],
        "bootstrap_summary": payload["bootstrap"]["metrics"],
        "slice_summary": {
            "slice_count": payload["slice_audit"]["slice_count"],
            "positive_slice_count": payload["slice_audit"]["positive_slice_count"],
            "negative_slice_count": payload["slice_audit"]["negative_slice_count"],
            "easy_hazard_slice_count": payload["slice_audit"]["easy_hazard_slice_count"],
            "core_weak_slices": payload["slice_audit"]["core_weak_slices"],
        },
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_bv_context_admissibility_slice_safe_repair"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-BV",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "selected_repair_mode": payload["validation_selection"]["selected_mode"],
                        "easy_safe": gate["easy_safe"],
                        "slice_easy_safe": gate["slice_easy_safe"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-BV validation-only slice-safe context admissibility repair.")
    parser.add_argument("--test-rows", type=int, default=0, help="Optional override for replay test rows; 0 uses Stage43-BT rows.")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=467)
    parser.add_argument("--min-rows", type=int, default=100)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--easy-limit", type=float, default=0.02)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    payload = _run(args)
    gate = payload["stage43_bv_gate"]
    print(f"Stage43-BV: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"selected_repair_mode={payload['validation_selection']['selected_mode']}")
    print(f"easy_safe={gate['easy_safe']}")
    print(f"slice_easy_safe={gate['slice_easy_safe']}")
    return payload


if __name__ == "__main__":
    main()
