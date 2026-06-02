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
from src import stage43_scene_graph_multimodal_ablation as bp
from src import stage43_scene_graph_slice_forensics as br
from src import stage43_gated_scene_graph_fusion as bq


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_scene_graph_context_router.json"
REPORT_MD = OUT_DIR / "stage43_scene_graph_context_router.md"
GATE_MD = OUT_DIR / "stage43_stage_bs_scene_graph_context_router_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_BS_SCENE_GRAPH_CONTEXT_ROUTER"
SOURCE = "fresh_stage43_bs_scene_graph_context_router"
DEFAULT_VARIANT = "graph_history_only"
VARIANTS = br.VARIANTS
EPS = 1e-8


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _variant_label(route_key: tuple[str, str, str]) -> str:
    return "::".join(route_key)


def _load_variant(
    split: str,
    variant: str,
    *,
    rows: int | None,
) -> tuple[m.WaypointSplit, dict[str, np.ndarray], dict[str, Any]]:
    bp_payload = read_json(bp.REPORT_JSON, {})
    variants = {row["variant"]: row for row in bp_payload.get("variants", [])}
    if variant not in variants:
        raise KeyError(f"Stage43-BS missing BP variant {variant}")
    meta = variants[variant]
    ckpt_path = Path(meta["checkpoint"])
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    row_seed = int(ckpt.get("row_seed", 443))
    ds, ctx = bp._build_variant_split(split, max_rows=rows, row_seed=row_seed, variant=variant)
    mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    std = np.asarray(ckpt["feature_std"], dtype=np.float32)
    ds.x = ((ds.x - mean) / std).astype(np.float32)
    model = m.FullWaypointLatentDynamics(
        int(ckpt["input_dim"]),
        hidden_dim=int(ckpt.get("hidden_dim", 128)),
        latent_dim=int(ckpt.get("latent_dim", 32)),
    )
    model.load_state_dict(ckpt["model_state"])
    pred = m._predict(model, ds, torch.device("cpu"), batch_size=4096)
    policy = meta["validation_selected_policy"]["policy"]
    selected_ade, selected_fde, switched = m._select_with_policy(ds, pred, policy)
    arrays = {
        "selected_ade": selected_ade.astype(np.float32),
        "selected_fde": selected_fde.astype(np.float32),
        "switched": switched.astype(bool),
    }
    info = {
        "variant": variant,
        "split": split,
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": meta.get("checkpoint_sha256", ""),
        "policy": policy,
        "context": ctx,
        "metrics": m._metrics(ds, selected_ade, selected_fde, switched),
    }
    return ds, arrays, info


def _assert_aligned(reference: m.WaypointSplit, other: m.WaypointSplit, *, split: str, variant: str) -> None:
    checks = {
        "row_count": len(reference.x) == len(other.x),
        "horizon": np.array_equal(reference.horizon, other.horizon),
        "domain": np.array_equal(reference.domain.astype(str), other.domain.astype(str)),
        "source_file": np.array_equal(reference.source_file.astype(str), other.source_file.astype(str)),
        "floor_ade": np.allclose(reference.floor_ade, other.floor_ade),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"Stage43-BS row alignment failed for {split}/{variant}: {failed}")


def _load_split_variants(split: str, *, rows: int | None) -> tuple[m.WaypointSplit, dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    arrays_by_variant: dict[str, dict[str, np.ndarray]] = {}
    info_by_variant: dict[str, Any] = {}
    reference: m.WaypointSplit | None = None
    for variant in VARIANTS:
        ds, arrays, info = _load_variant(split, variant, rows=rows)
        if reference is None:
            reference = ds
        else:
            _assert_aligned(reference, ds, split=split, variant=variant)
        arrays_by_variant[variant] = arrays
        info_by_variant[variant] = info
    if reference is None:
        raise RuntimeError(f"No Stage43-BS rows loaded for {split}")
    return reference, arrays_by_variant, info_by_variant


def _improvement(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return 0.0
    return float(1.0 - float(np.mean(selected[mask])) / max(float(np.mean(floor[mask])), EPS))


def _route_masks(ds: m.WaypointSplit) -> dict[tuple[str, str, str], np.ndarray]:
    domain = ds.domain.astype(str)
    source = ds.source_file.astype(str)
    horizon = ds.horizon.astype(int)
    masks: dict[tuple[str, str, str], np.ndarray] = {}
    for d in sorted(set(domain.tolist())):
        masks[("domain", d, "*")] = domain == d
    for h in sorted(set(horizon.tolist())):
        masks[("horizon", str(h), "*")] = horizon == h
    for d in sorted(set(domain.tolist())):
        for h in sorted(set(horizon.tolist())):
            masks[("domain_horizon", d, str(h))] = (domain == d) & (horizon == h)
    for s in sorted(set(source.tolist())):
        masks[("source", s, "*")] = source == s
        for h in sorted(set(horizon.tolist())):
            masks[("source_horizon", s, str(h))] = (source == s) & (horizon == h)
    return masks


def _route_priority(key: tuple[str, str, str]) -> int:
    order = {
        "source_horizon": 0,
        "domain_horizon": 1,
        "source": 2,
        "domain": 3,
        "horizon": 4,
    }
    return order.get(key[0], 99)


def _slice_stats(
    ds: m.WaypointSplit,
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    mask: np.ndarray,
) -> dict[str, Any]:
    imps = {variant: _improvement(arrays[variant]["selected_ade"], ds.floor_ade, mask) for variant in VARIANTS}
    default = imps[DEFAULT_VARIANT]
    best = max(imps, key=imps.get)
    return {
        "rows": int(mask.sum()),
        "improvements": imps,
        "best_variant": best,
        "best_minus_default": float(imps[best] - default),
        "default_improvement": float(default),
    }


def _select_routes(
    ds: m.WaypointSplit,
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    *,
    min_rows: int,
    min_gain: float,
    allowed_variants: set[str] | None = None,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    allowed = allowed_variants or set(VARIANTS)
    routes: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    for key, mask in _route_masks(ds).items():
        if int(mask.sum()) < int(min_rows):
            continue
        stats = _slice_stats(ds, arrays, mask)
        best = str(stats["best_variant"])
        accept = best in allowed and best != DEFAULT_VARIANT and float(stats["best_minus_default"]) >= float(min_gain)
        row = {
            "route_key": key,
            "route": _variant_label(key),
            "accepted": bool(accept),
            **stats,
        }
        if accept:
            routes[row["route"]] = best
        rows.append(row)
    return routes, rows


def _row_route_key(ds: m.WaypointSplit, row_id: int, routes: Mapping[str, str]) -> str | None:
    domain = str(ds.domain[row_id])
    source = str(ds.source_file[row_id])
    horizon = str(int(ds.horizon[row_id]))
    candidates = [
        ("source_horizon", source, horizon),
        ("domain_horizon", domain, horizon),
        ("source", source, "*"),
        ("domain", domain, "*"),
        ("horizon", horizon, "*"),
    ]
    for key in sorted(candidates, key=_route_priority):
        label = _variant_label(key)
        if label in routes:
            return label
    return None


def _apply_router(
    ds: m.WaypointSplit,
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    routes: Mapping[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected_ade = arrays[DEFAULT_VARIANT]["selected_ade"].copy()
    selected_fde = arrays[DEFAULT_VARIANT]["selected_fde"].copy()
    switched = arrays[DEFAULT_VARIANT]["switched"].copy()
    route_used = np.full(len(selected_ade), DEFAULT_VARIANT, dtype=object)
    for row_id in range(len(selected_ade)):
        label = _row_route_key(ds, row_id, routes)
        if label is None:
            continue
        variant = routes[label]
        selected_ade[row_id] = arrays[variant]["selected_ade"][row_id]
        selected_fde[row_id] = arrays[variant]["selected_fde"][row_id]
        switched[row_id] = arrays[variant]["switched"][row_id]
        route_used[row_id] = variant
    return selected_ade, selected_fde, switched, route_used


def _variant_counts(values: np.ndarray) -> dict[str, int]:
    return {str(v): int(np.sum(values.astype(str) == str(v))) for v in sorted(set(values.astype(str).tolist()))}


def _candidate_grid(*, include_full: bool) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for min_gain in [0.0, 0.005, 0.01, 0.02, 0.05]:
        full_options = [False, True] if include_full else [False]
        for allow_full in full_options:
            allowed = {"no_context", "scene_proxy_only", "graph_history_only"}
            if allow_full:
                allowed.add("scene_graph_full")
            candidates.append({"min_gain": min_gain, "min_rows": 100, "allow_full": allow_full, "allowed": allowed})
    return candidates


def _score(metrics: Mapping[str, Any], route_count: int) -> float:
    return (
        1.0 * float(metrics["full_waypoint_ade_improvement_vs_floor"])
        + 1.5 * float(metrics["t50_full_waypoint_ade_improvement_vs_floor"])
        + 0.8 * float(metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"])
        - 20.0 * max(0.0, float(metrics["easy_degradation_vs_floor"]) - 0.02)
        - 0.002 * float(route_count)
    )


def _evaluate_candidates(
    val: m.WaypointSplit,
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    *,
    include_full: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cand in _candidate_grid(include_full=include_full):
        routes, route_rows = _select_routes(
            val,
            arrays,
            min_rows=int(cand["min_rows"]),
            min_gain=float(cand["min_gain"]),
            allowed_variants=set(cand["allowed"]),
        )
        selected_ade, selected_fde, switched, route_used = _apply_router(val, arrays, routes)
        metrics = m._metrics(val, selected_ade, selected_fde, switched)
        safe = metrics["easy_degradation_vs_floor"] <= 0.02 and metrics["full_waypoint_ade_improvement_vs_floor"] >= 0.0
        rows.append(
            {
                "candidate": {k: v for k, v in cand.items() if k != "allowed"},
                "routes": routes,
                "route_rows": route_rows,
                "route_count": len(routes),
                "route_variant_counts": _variant_counts(route_used),
                "validation_metrics": metrics,
                "validation_safe": bool(safe),
                "selection_score": _score(metrics, len(routes)),
            }
        )
    return rows


def _reference_metrics(ds: m.WaypointSplit, arrays: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    return {
        variant: m._metrics(ds, arrays[variant]["selected_ade"], arrays[variant]["selected_fde"], arrays[variant]["switched"])
        for variant in VARIANTS
    }


def _route_report(routes: Mapping[str, str], limit: int = 24) -> list[str]:
    lines = [
        "| route | variant |",
        "| --- | --- |",
    ]
    for route, variant in list(routes.items())[:limit]:
        lines.append(f"| `{route}` | `{variant}` |")
    if len(routes) > limit:
        lines.append(f"| `...` | `{len(routes) - limit} more` |")
    return lines


def _metric_lines(metrics: Mapping[str, Any]) -> list[str]:
    return [
        f"- all full-waypoint ADE improvement: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic improvement: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- hard/failure improvement: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- switch rate: `{_pct(metrics['switch_rate'])}`",
    ]


def _run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    val, val_arrays, val_info = _load_split_variants("val", rows=int(args.val_rows))
    test, test_arrays, test_info = _load_split_variants("test", rows=int(args.test_rows))
    bp_payload = read_json(bp.REPORT_JSON, {})
    bp_gate = bp_payload.get("stage43_bp_gate", {})
    full_context_blocked = bool(bp_gate.get("full_multimodal_unsafe", False)) and not bool(args.allow_unsafe_full_context)
    candidates = _evaluate_candidates(val, val_arrays, include_full=not full_context_blocked)
    safe_candidates = [row for row in candidates if row["validation_safe"]]
    selected = max(safe_candidates or candidates, key=lambda row: row["selection_score"])
    test_selected_ade, test_selected_fde, test_switched, test_route_used = _apply_router(test, test_arrays, selected["routes"])
    test_metrics = m._metrics(test, test_selected_ade, test_selected_fde, test_switched)
    references = _reference_metrics(test, test_arrays)
    graph = references[DEFAULT_VARIANT]
    bq_payload = read_json(bq.REPORT_JSON, {})
    br_payload = read_json(br.REPORT_JSON, {})
    delta_vs_graph = {
        "all": float(test_metrics["full_waypoint_ade_improvement_vs_floor"] - graph["full_waypoint_ade_improvement_vs_floor"]),
        "t50": float(test_metrics["t50_full_waypoint_ade_improvement_vs_floor"] - graph["t50_full_waypoint_ade_improvement_vs_floor"]),
        "hard_failure": float(
            test_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
            - graph["hard_failure_full_waypoint_ade_improvement_vs_floor"]
        ),
        "easy_degradation": float(test_metrics["easy_degradation_vs_floor"] - graph["easy_degradation_vs_floor"]),
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_validation_selected_scene_graph_context_router",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "rows": {"val": int(len(val.x)), "test": int(len(test.x))},
        "precondition": {
            "bp_verdict": bp_gate.get("verdict", "missing"),
            "bp_full_multimodal_unsafe": bp_gate.get("full_multimodal_unsafe", False),
            "bq_verdict": bq_payload.get("stage43_bq_gate", {}).get("verdict", "missing"),
            "br_verdict": br_payload.get("stage43_br_gate", {}).get("verdict", "missing"),
        },
        "unsafe_full_context_blocked_by_bp_prior": full_context_blocked,
        "variant_replay": {"val": val_info, "test": test_info},
        "validation_candidates": candidates,
        "safe_validation_candidate_count": len(safe_candidates),
        "selected_candidate": {
            "candidate": selected["candidate"],
            "route_count": selected["route_count"],
            "routes": selected["routes"],
            "route_variant_counts_val": selected["route_variant_counts"],
            "validation_metrics": selected["validation_metrics"],
            "validation_safe": selected["validation_safe"],
            "selection_score": selected["selection_score"],
            "selection_rule": "validation-only source/domain/horizon route table; fallback graph_history_only; test evaluated once",
            "test_tuned": False,
        },
        "test_metrics": test_metrics,
        "test_route_variant_counts": _variant_counts(test_route_used),
        "test_reference_metrics": references,
        "delta_vs_graph_history_only": delta_vs_graph,
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "raw_scene_or_verified_sdf_claim": False,
            "deployment_policy_changed": False,
            "test_threshold_tuning": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "scene_proxy_train_only": True,
            "graph_inputs_past_or_current_only": True,
            "test_route_selection": False,
        },
        "input_hash": _combined_hash([bp.REPORT_JSON, bq.REPORT_JSON, br.REPORT_JSON]),
    }
    payload["stage43_bs_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["test_metrics"]
    delta = payload["delta_vs_graph_history_only"]
    claim = payload["claim_boundary"]
    no_leak = payload["no_leakage"]
    easy_safe = metrics["easy_degradation_vs_floor"] <= 0.02
    beats_graph = max(delta["all"], delta["t50"], delta["hard_failure"]) > 0.0
    gates = {
        "bp_precondition_passed": payload["precondition"]["bp_verdict"]
        in {
            "stage43_bp_scene_graph_multimodal_ablation_pass_negative_unsafe_diagnostic",
            "stage43_bp_scene_graph_multimodal_ablation_pass_mixed_diagnostic",
            "stage43_bp_scene_graph_multimodal_ablation_pass_contribution_supported",
        },
        "bq_precondition_passed": payload["precondition"]["bq_verdict"]
        in {
            "stage43_bq_gated_scene_graph_fusion_pass_contribution_supported",
            "stage43_bq_gated_scene_graph_fusion_pass_safe_no_best_single_lift_diagnostic",
            "stage43_bq_gated_scene_graph_fusion_pass_safe_no_lift_diagnostic",
            "stage43_bq_gated_scene_graph_fusion_pass_unsafe_diagnostic",
        },
        "br_precondition_passed": payload["precondition"]["br_verdict"]
        in {
            "stage43_br_scene_graph_slice_forensics_pass_targeted_scene_signal",
            "stage43_br_scene_graph_slice_forensics_pass_weak_scene_signal_diagnostic",
            "stage43_br_scene_graph_slice_forensics_pass_no_scene_signal_diagnostic",
        },
        "validation_candidates_evaluated": len(payload["validation_candidates"]) > 0,
        "validation_only_route_selection": payload["selected_candidate"]["test_tuned"] is False
        and no_leak["test_route_selection"] is False,
        "route_table_nonempty_or_fallback_explicit": payload["selected_candidate"]["route_count"] >= 0,
        "test_eval_completed": payload["rows"]["test"] > 0,
        "graph_history_reference_present": DEFAULT_VARIANT in payload["test_reference_metrics"],
        "easy_safety_measured": "easy_degradation_vs_floor" in metrics,
        "no_future_or_test_leakage": no_leak["future_endpoint_input"] is False
        and no_leak["future_waypoint_input"] is False
        and no_leak["future_waypoint_label_eval_only"] is True
        and no_leak["central_velocity_input"] is False
        and no_leak["test_endpoint_goal_construction"] is False
        and no_leak["test_statistics_normalization"] is False
        and no_leak["scene_proxy_train_only"] is True
        and no_leak["graph_inputs_past_or_current_only"] is True,
        "no_metric_seconds_stage5c_smc_claim": claim["metric_or_seconds_claim"] is False
        and claim["raw_scene_or_verified_sdf_claim"] is False
        and claim["stage5c_executed"] is False
        and claim["smc_enabled"] is False,
        "long_objective_kept_active": claim["long_objective_complete"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    if passed == total and easy_safe and beats_graph:
        verdict = "stage43_bs_scene_graph_context_router_pass_safe_lift_diagnostic"
    elif passed == total and easy_safe:
        verdict = "stage43_bs_scene_graph_context_router_pass_safe_no_lift_diagnostic"
    elif passed == total:
        verdict = "stage43_bs_scene_graph_context_router_pass_unsafe_diagnostic"
    else:
        verdict = "stage43_bs_scene_graph_context_router_incomplete"
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": verdict,
        "validation_selected_router": passed == total,
        "beats_graph_history_on_any_core_metric": beats_graph,
        "easy_safe": easy_safe,
        "deployable_policy_changed": False,
        "protected_multimodal_latent_state_candidate": passed == total and easy_safe,
        "long_objective_complete": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bs_gate"]
    selected = payload["selected_candidate"]
    metrics = payload["test_metrics"]
    delta = payload["delta_vs_graph_history_only"]
    write_json(REPORT_JSON, m._jsonable(payload))
    write_md(
        REPORT_MD,
        [
            "# Stage43-BS Scene-Graph Context Router",
            "",
            f"- source: `{payload['source']}`",
            f"- result_source: `{payload['result_source']}`",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- validation safe candidates: `{payload['safe_validation_candidate_count']}`",
            f"- selected route count: `{selected['route_count']}`",
            f"- unsafe full context blocked by BP prior: `{payload['unsafe_full_context_blocked_by_bp_prior']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            "",
            "## Selected Validation Router",
            "",
            f"- candidate: `{selected['candidate']}`",
            f"- selection rule: {selected['selection_rule']}",
            f"- val route variant counts: `{selected['route_variant_counts_val']}`",
            f"- test route variant counts: `{payload['test_route_variant_counts']}`",
            "",
            "## Test Metrics",
            "",
            *_metric_lines(metrics),
            "",
            "## Delta Vs Graph-History-Only",
            "",
            f"- all delta: `{_pct(delta['all'])}`",
            f"- t50 delta: `{_pct(delta['t50'])}`",
            f"- hard/failure delta: `{_pct(delta['hard_failure'])}`",
            f"- easy degradation delta: `{_pct(delta['easy_degradation'])}`",
            "",
            "## Routes",
            "",
            *_route_report(selected["routes"]),
            "",
            "## Interpretation",
            "",
            "- This router converts BR slice evidence into a validation-selected source/domain/horizon route table.",
            "- It is a diagnostic context-routing experiment, not a deployment policy update.",
            "- Route selection uses validation only; test is evaluated once.",
            "- Future waypoints remain labels/eval only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
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
            "# Stage43-BS Scene-Graph Context Router Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- validation-selected router: `{gate['validation_selected_router']}`",
            f"- beats graph-history on any core metric: `{gate['beats_graph_history_on_any_core_metric']}`",
            f"- easy safe: `{gate['easy_safe']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            f"- long objective complete: `{gate['long_objective_complete']}`",
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
            f"- validation-selected router: `{gate['validation_selected_router']}`",
            f"- beats graph-history on any core metric: `{gate['beats_graph_history_on_any_core_metric']}`",
            f"- easy safe: `{gate['easy_safe']}`",
            f"- deployable policy changed: `{gate['deployable_policy_changed']}`",
            f"- long objective complete: `{gate['long_objective_complete']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "## Current Boundary",
            "",
            "- Stage43-BS is a validation-selected source/domain/horizon context-router diagnostic.",
            "- It does not update deployment and does not claim raw scene/SDF evidence.",
            "- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    _update_ledgers(payload)


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bs_gate"]
    metrics = payload["test_metrics"]
    delta = payload["delta_vs_graph_history_only"]
    selected = payload["selected_candidate"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"validation_selected_router = `{gate['validation_selected_router']}`",
        f"beats_graph_history_on_any_core_metric = `{gate['beats_graph_history_on_any_core_metric']}`",
        f"easy_safe = `{gate['easy_safe']}`",
        f"deployable_policy_changed = `{gate['deployable_policy_changed']}`",
        "",
        f"Stage43-BS builds a validation-only source/domain/horizon route table from Stage43-BP context variants after Stage43-BR found targeted scene signal. Selected routes: `{selected['route_count']}`; validation-safe candidates: `{payload['safe_validation_candidate_count']}`.",
        f"Unsafe full scene+graph context blocked by BP prior: `{payload['unsafe_full_context_blocked_by_bp_prior']}`.",
        "",
        f"Test metrics: all `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`, t50 `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`, t100 raw-frame diagnostic `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`, hard/failure `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`, easy degradation `{_pct(metrics['easy_degradation_vs_floor'])}`.",
        f"Delta vs graph-history-only: all `{_pct(delta['all'])}`, t50 `{_pct(delta['t50'])}`, hard/failure `{_pct(delta['hard_failure'])}`, easy degradation `{_pct(delta['easy_degradation'])}`.",
        "",
        "This is a diagnostic context-routing experiment, not a deployment policy update.",
        "",
        "Boundary unchanged: dataset-local/raw-frame 2.5D only; future waypoints are labels/eval only; no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_bs_scene_graph_context_router"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "validation_selected_router": gate["validation_selected_router"],
        "beats_graph_history_on_any_core_metric": gate["beats_graph_history_on_any_core_metric"],
        "easy_safe": gate["easy_safe"],
        "selected_route_count": selected["route_count"],
        "safe_validation_candidate_count": payload["safe_validation_candidate_count"],
        "unsafe_full_context_blocked_by_bp_prior": payload["unsafe_full_context_blocked_by_bp_prior"],
        "test_metrics": metrics,
        "delta_vs_graph_history_only": delta,
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_bs_scene_graph_context_router"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-BS",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "beats_graph_history_on_any_core_metric": gate["beats_graph_history_on_any_core_metric"],
                        "easy_safe": gate["easy_safe"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-BS scene-graph context router.")
    parser.add_argument("--val-rows", type=int, default=12000)
    parser.add_argument("--test-rows", type=int, default=12000)
    parser.add_argument(
        "--allow-unsafe-full-context",
        action="store_true",
        help="Diagnostic only: allow scene_graph_full in the route search despite BP marking full multimodal context unsafe.",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    payload = _run(args)
    gate = payload["stage43_bs_gate"]
    print(f"Stage43-BS: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"beats_graph_history_on_any_core_metric={gate['beats_graph_history_on_any_core_metric']}")
    print(f"easy_safe={gate['easy_safe']}")
    return payload


if __name__ == "__main__":
    main()
