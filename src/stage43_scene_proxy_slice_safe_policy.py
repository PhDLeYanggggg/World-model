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
from src import stage43_full_waypoint_latent_dynamics as m
from src import stage43_scene_proxy_guarded_latent_policy as ac
from src import stage43_scene_proxy_guarded_robustness_audit as ad


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_scene_proxy_slice_safe_policy.json"
REPORT_MD = OUT_DIR / "stage43_scene_proxy_slice_safe_policy.md"
GATE_MD = OUT_DIR / "stage43_stage_ae_scene_proxy_slice_safe_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_AE_SCENE_PROXY_SLICE_SAFE_POLICY"
SOURCE = "fresh_stage43_ae_scene_proxy_slice_safe_policy"

ROUTE_FLOOR = 0
ROUTE_STAGE43_M = 1
ROUTE_STAGE43_AB = 2

POLICY_FAMILIES = [
    "ab_non_h100_m_else",
    "ab_non_h100_floor_h100",
    "ab_h25_h50_m_h10_floor_h100",
    "ab_h25_h50_floor_else",
    "domain_safe_v1",
    "domain_safe_v2",
    "h50_ab_m_h25_floor_else",
    "stage43_m_non_h100_floor_h100",
]


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _route_codes(ds: m.WaypointSplit, family: str) -> np.ndarray:
    horizon = ds.horizon.astype(np.int64)
    domain = ds.domain.astype(str)
    non_h100 = horizon != 100
    h10 = horizon == 10
    h25 = horizon == 25
    h50 = horizon == 50
    trajnet = domain == "TrajNet"
    route = np.full(len(ds.x), ROUTE_FLOOR, dtype=np.int8)
    if family == "ab_non_h100_m_else":
        route[:] = ROUTE_STAGE43_M
        route[non_h100] = ROUTE_STAGE43_AB
    elif family == "ab_non_h100_floor_h100":
        route[non_h100] = ROUTE_STAGE43_AB
    elif family == "ab_h25_h50_m_h10_floor_h100":
        route[h10] = ROUTE_STAGE43_M
        route[h25 | h50] = ROUTE_STAGE43_AB
    elif family == "ab_h25_h50_floor_else":
        route[h25 | h50] = ROUTE_STAGE43_AB
    elif family == "domain_safe_v1":
        route[(~trajnet) & non_h100] = ROUTE_STAGE43_AB
        route[trajnet & (h25 | h50)] = ROUTE_STAGE43_M
    elif family == "domain_safe_v2":
        route[(~trajnet) & non_h100] = ROUTE_STAGE43_AB
        route[trajnet & non_h100] = ROUTE_STAGE43_M
    elif family == "h50_ab_m_h25_floor_else":
        route[h25] = ROUTE_STAGE43_M
        route[h50] = ROUTE_STAGE43_AB
    elif family == "stage43_m_non_h100_floor_h100":
        route[non_h100] = ROUTE_STAGE43_M
    else:
        raise KeyError(f"Unknown Stage43-AE policy family: {family}")
    return route


def _candidate_pack(
    ds: m.WaypointSplit,
    pack: Mapping[str, Any],
    *,
    route: np.ndarray,
    ab_allowed: np.ndarray,
    ab_reject_fallback: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    final_route = route.copy()
    rejected_ab = (final_route == ROUTE_STAGE43_AB) & (~ab_allowed)
    if ab_reject_fallback == "stage43_m":
        final_route[rejected_ab] = ROUTE_STAGE43_M
    elif ab_reject_fallback == "floor":
        final_route[rejected_ab] = ROUTE_FLOOR
    else:
        raise KeyError(f"Unknown ab reject fallback: {ab_reject_fallback}")
    selected_ade = ds.floor_ade.astype(np.float32).copy()
    selected_fde = ds.floor_fde.astype(np.float32).copy()
    m_mask = final_route == ROUTE_STAGE43_M
    ab_mask = final_route == ROUTE_STAGE43_AB
    selected_ade[m_mask] = pack["m_ade"][m_mask]
    selected_fde[m_mask] = pack["m_fde"][m_mask]
    selected_ade[ab_mask] = pack["ab_ade"][ab_mask]
    selected_fde[ab_mask] = pack["ab_fde"][ab_mask]
    switched = final_route != ROUTE_FLOOR
    return selected_ade, selected_fde, switched.astype(bool), final_route


def _select_with_policy(pack: Mapping[str, Any], policy: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ds: m.WaypointSplit = pack["ds"]
    pred = pack["pred_ab"]
    route = _route_codes(ds, str(policy["family"]))
    ab_allowed = (
        (pred["gain"] >= float(policy["gain_threshold"]))
        & (pred["harm"] <= float(policy["harm_threshold"]))
        & (pred["failure"] >= float(policy["failure_threshold"]))
    )
    return _candidate_pack(
        ds,
        pack,
        route=route,
        ab_allowed=ab_allowed,
        ab_reject_fallback=str(policy["ab_reject_fallback"]),
    )


def _route_rates(ds: m.WaypointSplit, final_route: np.ndarray) -> dict[str, float]:
    h100 = ds.horizon == 100
    h10 = ds.horizon == 10
    return {
        "floor_rate": float(np.mean(final_route == ROUTE_FLOOR)),
        "stage43_m_rate": float(np.mean(final_route == ROUTE_STAGE43_M)),
        "stage43_ab_rate": float(np.mean(final_route == ROUTE_STAGE43_AB)),
        "h10_floor_rate": float(np.mean(final_route[h10] == ROUTE_FLOOR)) if int(h10.sum()) else 0.0,
        "h10_stage43_ab_rate": float(np.mean(final_route[h10] == ROUTE_STAGE43_AB)) if int(h10.sum()) else 0.0,
        "h100_floor_rate": float(np.mean(final_route[h100] == ROUTE_FLOOR)) if int(h100.sum()) else 0.0,
        "h100_stage43_ab_rate": float(np.mean(final_route[h100] == ROUTE_STAGE43_AB)) if int(h100.sum()) else 0.0,
        "trajnet_floor_rate": float(np.mean(final_route[ds.domain.astype(str) == "TrajNet"] == ROUTE_FLOOR))
        if int((ds.domain.astype(str) == "TrajNet").sum())
        else 0.0,
        "trajnet_stage43_m_rate": float(np.mean(final_route[ds.domain.astype(str) == "TrajNet"] == ROUTE_STAGE43_M))
        if int((ds.domain.astype(str) == "TrajNet").sum())
        else 0.0,
    }


def _eval_policy(pack: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    ds: m.WaypointSplit = pack["ds"]
    selected_ade, selected_fde, switched, final_route = _select_with_policy(pack, policy)
    metrics = m._metrics(ds, selected_ade, selected_fde, switched)
    metrics.update(_route_rates(ds, final_route))
    diagnostics = _slice_diagnostics(ds, selected_ade, selected_fde, switched, final_route)
    return {
        "metrics": metrics,
        "diagnostics": diagnostics,
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "switched": switched,
        "final_route": final_route,
    }


def _slice_metrics(ds: m.WaypointSplit, selected_ade: np.ndarray, selected_fde: np.ndarray, switched: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    sub = ad._subset(ds, mask)
    return m._metrics(sub, selected_ade[mask], selected_fde[mask], switched[mask])


def _slice_diagnostics(
    ds: m.WaypointSplit,
    selected_ade: np.ndarray,
    selected_fde: np.ndarray,
    switched: np.ndarray,
    final_route: np.ndarray,
) -> dict[str, Any]:
    horizon = ds.horizon.astype(np.int64)
    domain = ds.domain.astype(str)
    diag: dict[str, Any] = {"domains": {}, "horizons": {}}
    for dom in sorted(set(domain.tolist())):
        mask = domain == dom
        row = _slice_metrics(ds, selected_ade, selected_fde, switched, mask)
        row.update(_route_rates(ad._subset(ds, mask), final_route[mask]))
        diag["domains"][dom] = row
    for h in [10, 25, 50, 100]:
        mask = horizon == h
        if int(mask.sum()) == 0:
            continue
        row = _slice_metrics(ds, selected_ade, selected_fde, switched, mask)
        row.update(_route_rates(ad._subset(ds, mask), final_route[mask]))
        diag["horizons"][str(h)] = row
    diag["max_domain_easy_degradation"] = max((row["easy_degradation_vs_floor"] for row in diag["domains"].values()), default=0.0)
    diag["min_domain_all_improvement"] = min((row["full_waypoint_ade_improvement_vs_floor"] for row in diag["domains"].values()), default=0.0)
    diag["max_horizon_easy_degradation"] = max((row["easy_degradation_vs_floor"] for row in diag["horizons"].values()), default=0.0)
    diag["h10_easy_degradation"] = diag["horizons"].get("10", {}).get("easy_degradation_vs_floor", 0.0)
    diag["h100_easy_degradation"] = diag["horizons"].get("100", {}).get("easy_degradation_vs_floor", 0.0)
    return diag


def _search_policy(val_pack: Mapping[str, Any]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    fallback_best: dict[str, Any] | None = None
    for family in POLICY_FAMILIES:
        for fallback in ["stage43_m", "floor"]:
            for gain in [0.0, 0.25, 0.45, 0.55, 0.65, 0.75, 0.85]:
                for harm in [0.15, 0.25, 0.35, 0.50, 0.75, 1.00]:
                    for failure in [0.0, 0.10, 0.20, 0.35, 0.50]:
                        policy = {
                            "family": family,
                            "gain_threshold": gain,
                            "harm_threshold": harm,
                            "failure_threshold": failure,
                            "ab_reject_fallback": fallback,
                            "selected_on": "validation_only",
                            "test_threshold_tuning": False,
                            "uses_easy_label_at_inference": False,
                        }
                        out = _eval_policy(val_pack, policy)
                        metrics = out["metrics"]
                        diag = out["diagnostics"]
                        t100_penalty = max(0.0, -metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] - 0.002)
                        easy_penalty = max(0.0, metrics["easy_degradation_vs_floor"] - 0.02)
                        domain_easy_penalty = max(0.0, diag["max_domain_easy_degradation"] - 0.02)
                        horizon_easy_penalty = max(0.0, diag["max_horizon_easy_degradation"] - 0.02)
                        domain_positive_penalty = max(0.0, -diag["min_domain_all_improvement"])
                        objective = (
                            1.0 * metrics["full_waypoint_ade_improvement_vs_floor"]
                            + 1.7 * metrics["t50_full_waypoint_ade_improvement_vs_floor"]
                            + 1.0 * metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
                            + 0.5 * metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]
                            - 25.0 * easy_penalty
                            - 20.0 * domain_easy_penalty
                            - 15.0 * horizon_easy_penalty
                            - 15.0 * t100_penalty
                            - 5.0 * domain_positive_penalty
                            - 0.01 * metrics["stage43_ab_rate"]
                        )
                        row = {
                            "policy": policy,
                            "metrics": metrics,
                            "diagnostics": diag,
                            "objective": float(objective),
                        }
                        if fallback_best is None or row["objective"] > fallback_best["objective"]:
                            fallback_best = row
                        structural_guard = (
                            metrics["h10_floor_rate"] >= 0.99
                            and metrics["h100_floor_rate"] >= 0.99
                            and metrics["floor_rate"] > 0.0
                            and metrics["stage43_m_rate"] > 0.0
                            and metrics["stage43_ab_rate"] > 0.0
                        )
                        if structural_guard and (best is None or row["objective"] > best["objective"]):
                            best = row
    if best is None:
        assert fallback_best is not None
        fallback_best["policy"]["structural_guard_unavailable"] = True
        return fallback_best
    best["policy"]["stage43_ad_structural_caveat_guard"] = True
    return best


def _baseline_pack(pack: Mapping[str, Any], name: str) -> dict[str, Any]:
    ds: m.WaypointSplit = pack["ds"]
    if name == "floor":
        selected_ade, selected_fde, switched = ds.floor_ade, ds.floor_fde, np.zeros(len(ds.x), dtype=bool)
        route = np.full(len(ds.x), ROUTE_FLOOR, dtype=np.int8)
    elif name == "stage43_m":
        selected_ade, selected_fde, switched = pack["m_ade"], pack["m_fde"], pack["m_switched"]
        route = np.where(switched, ROUTE_STAGE43_M, ROUTE_FLOOR).astype(np.int8)
    elif name == "stage43_ab_all":
        selected_ade, selected_fde, switched = pack["ab_ade"], pack["ab_fde"], np.ones(len(ds.x), dtype=bool)
        route = np.full(len(ds.x), ROUTE_STAGE43_AB, dtype=np.int8)
    else:
        raise KeyError(name)
    metrics = m._metrics(ds, selected_ade, selected_fde, switched)
    metrics.update(_route_rates(ds, route))
    return {
        "metrics": metrics,
        "diagnostics": _slice_diagnostics(ds, selected_ade, selected_fde, switched, route),
    }


def _stage43_ac_pack(pack: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    ds: m.WaypointSplit = pack["ds"]
    selected_ade, selected_fde, switched, ab_allowed = ac._select_guarded(pack, policy)
    route = np.where(ab_allowed, ROUTE_STAGE43_AB, np.where(pack["m_switched"], ROUTE_STAGE43_M, ROUTE_FLOOR)).astype(np.int8)
    metrics = m._metrics(ds, selected_ade, selected_fde, switched)
    metrics.update(_route_rates(ds, route))
    return {
        "metrics": metrics,
        "diagnostics": _slice_diagnostics(ds, selected_ade, selected_fde, switched, route),
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    seed = int(args.seed)
    rows = ac._max_rows("medium" if args.medium else "quick" if args.quick else "small")
    stage43_m_report = read_json(m.REPORT_JSON, {})
    stage43_ab_report = read_json(ac.ab.REPORT_JSON, {})
    stage43_ac_report = read_json(ac.REPORT_JSON, {})
    stage43_ad_report = read_json(ad.REPORT_JSON, {})
    m_model, m_ckpt = ac._load_model(Path(stage43_m_report["checkpoint"]))
    ab_model, ab_ckpt = ac._load_model(Path(stage43_ab_report["checkpoint"]))
    val_pack = ac._replay_split(
        "val",
        max_rows=rows["val"],
        seed=seed,
        batch_size=int(args.batch_size),
        m_model=m_model,
        m_ckpt=m_ckpt,
        ab_model=ab_model,
        ab_ckpt=ab_ckpt,
        m_policy=stage43_m_report["validation_selected_policy"]["policy"],
    )
    test_pack = ac._replay_split(
        "test",
        max_rows=rows["test"],
        seed=seed,
        batch_size=int(args.batch_size),
        m_model=m_model,
        m_ckpt=m_ckpt,
        ab_model=ab_model,
        ab_ckpt=ab_ckpt,
        m_policy=stage43_m_report["validation_selected_policy"]["policy"],
    )
    best = _search_policy(val_pack)
    test_eval = _eval_policy(test_pack, best["policy"])
    floor = _baseline_pack(test_pack, "floor")
    stage43_m = _baseline_pack(test_pack, "stage43_m")
    stage43_ab = _baseline_pack(test_pack, "stage43_ab_all")
    stage43_ac_eval = _stage43_ac_pack(test_pack, stage43_ac_report["validation_selected_policy"]["policy"])
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_validation_selected_slice_safe_three_route_policy",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "mode": "medium" if args.medium else "quick" if args.quick else "small",
        "stage43_ac_verdict": stage43_ac_report.get("stage43_ac_gate", {}).get("verdict"),
        "stage43_ad_verdict": stage43_ad_report.get("stage43_ad_gate", {}).get("verdict"),
        "data_rows": {"val": len(val_pack["ds"].x), "test": len(test_pack["ds"].x)},
        "validation_selected_policy": best,
        "test_metrics_slice_safe": test_eval["metrics"],
        "test_diagnostics_slice_safe": test_eval["diagnostics"],
        "baselines": {
            "floor": floor,
            "stage43_m": stage43_m,
            "stage43_ab_all": stage43_ab,
            "stage43_ac": {
                "metrics": stage43_ac_eval["metrics"],
                "diagnostics": stage43_ac_eval["diagnostics"],
            },
        },
        "delta_vs_stage43_m": ac._delta_metrics(test_eval["metrics"], stage43_m["metrics"]),
        "delta_vs_stage43_ac": ac._delta_metrics(test_eval["metrics"], stage43_ac_eval["metrics"]),
        "delta_vs_stage43_ab_all": ac._delta_metrics(test_eval["metrics"], stage43_ab["metrics"]),
        "bootstrap_ci": m._bootstrap_ci(
            test_pack["ds"],
            test_eval["selected_ade"],
            test_eval["selected_fde"],
            n=int(args.bootstrap),
            seed=seed + 4000,
        ),
        "route_counts": {
            "floor": int(np.sum(test_eval["final_route"] == ROUTE_FLOOR)),
            "stage43_m": int(np.sum(test_eval["final_route"] == ROUTE_STAGE43_M)),
            "stage43_ab": int(np.sum(test_eval["final_route"] == ROUTE_STAGE43_AB)),
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
            "uses_easy_label_at_inference": False,
            "scene_proxy_train_only": True,
            "stage43_ad_caveat_slices_guarded_by_structure": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "t100_raw_frame_diagnostic_only": True,
            "slice_safe_not_uniform_horizon_success": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "input_hash": _combined_hash([ac.REPORT_JSON, ad.REPORT_JSON, m.REPORT_JSON, ac.ab.REPORT_JSON, m._cache_path("val"), m._cache_path("test")]),
    }
    payload["stage43_ae_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["test_metrics_slice_safe"]
    diag = payload["test_diagnostics_slice_safe"]
    delta_m = payload["delta_vs_stage43_m"]
    gates = {
        "stage43_ac_available": payload["stage43_ac_verdict"] == "stage43_ac_guarded_scene_proxy_latent_candidate",
        "stage43_ad_caveat_audit_available": payload["stage43_ad_verdict"]
        in {"stage43_ad_guarded_scene_proxy_caveated_audit_pass", "stage43_ad_guarded_scene_proxy_robust_with_caveats"},
        "fresh_validation_selected_policy": payload["result_source"] == "fresh_validation_selected_slice_safe_three_route_policy"
        and payload["validation_selected_policy"]["policy"]["selected_on"] == "validation_only"
        and payload["validation_selected_policy"]["policy"]["test_threshold_tuning"] is False,
        "three_route_policy_used": payload["route_counts"]["floor"] > 0
        and payload["route_counts"]["stage43_m"] > 0
        and payload["route_counts"]["stage43_ab"] > 0,
        "stage43_ad_structural_caveat_guard": metrics["h10_floor_rate"] >= 0.99
        and metrics["h100_floor_rate"] >= 0.99
        and payload["validation_selected_policy"]["policy"].get("stage43_ad_structural_caveat_guard") is True,
        "overall_easy_preserved": metrics["easy_degradation_vs_floor"] <= 0.02,
        "domain_easy_preserved": diag["max_domain_easy_degradation"] <= 0.02,
        "horizon_easy_preserved": diag["max_horizon_easy_degradation"] <= 0.02,
        "all_powered_domains_positive": diag["min_domain_all_improvement"] > 0.0,
        "core_lift_vs_stage43_m": delta_m["full_waypoint_ade_improvement_vs_floor"] > 0.0
        or delta_m["t50_full_waypoint_ade_improvement_vs_floor"] > 0.0
        or delta_m["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0,
        "t50_still_positive": metrics["t50_full_waypoint_ade_improvement_vs_floor"] > 0.0,
        "t100_guarded_to_nonnegative_floor": metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] >= -0.002
        and metrics["h100_floor_rate"] >= 0.99,
        "no_future_or_test_leakage": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_waypoint_label_eval_only"] is True
        and payload["no_leakage"]["central_velocity_input"] is False
        and payload["no_leakage"]["test_endpoint_goal_construction"] is False
        and payload["no_leakage"]["test_statistics_normalization"] is False
        and payload["no_leakage"]["test_threshold_tuning"] is False
        and payload["no_leakage"]["uses_easy_label_at_inference"] is False,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    deploy = bool(passed == total)
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_ae_slice_safe_scene_proxy_candidate"
        if deploy
        else "stage43_ae_slice_safe_scene_proxy_diagnostic_keep_ac",
        "deploy_slice_safe_scene_proxy": deploy,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    gate = payload["stage43_ae_gate"]
    metrics = payload["test_metrics_slice_safe"]
    diag = payload["test_diagnostics_slice_safe"]
    delta_m = payload["delta_vs_stage43_m"]
    delta_ac = payload["delta_vs_stage43_ac"]
    policy = payload["validation_selected_policy"]["policy"]
    lines = [
        "# Stage43-AE Scene-Proxy Slice-Safe Policy",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- deploy slice-safe scene proxy: `{gate['deploy_slice_safe_scene_proxy']}`",
        "",
        "## Selected Policy",
        "",
        f"- policy: `{policy}`",
        f"- validation objective: `{payload['validation_selected_policy']['objective']:.6f}`",
        f"- route counts: `{payload['route_counts']}`",
        "",
        "## Test Metrics",
        "",
        f"- full-waypoint ADE vs floor: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`; delta vs Stage43-M: `{_pct(delta_m['full_waypoint_ade_improvement_vs_floor'])}`; delta vs AC: `{_pct(delta_ac['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50 ADE vs floor: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`; delta vs Stage43-M: `{_pct(delta_m['t50_full_waypoint_ade_improvement_vs_floor'])}`; delta vs AC: `{_pct(delta_ac['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- hard/failure vs floor: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`; delta vs Stage43-M: `{_pct(delta_m['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`; delta vs AC: `{_pct(delta_ac['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- easy degradation overall/domain/horizon max: `{_pct(metrics['easy_degradation_vs_floor'])}` / `{_pct(diag['max_domain_easy_degradation'])}` / `{_pct(diag['max_horizon_easy_degradation'])}`",
        f"- min domain all improvement: `{_pct(diag['min_domain_all_improvement'])}`",
        "",
        "## Domain Diagnostics",
        "",
        "| domain | all | t50 | hard | easy | floor | Stage43-M | Stage43-AB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for domain, row in sorted(diag["domains"].items()):
        lines.append(
            f"| `{domain}` | `{_pct(row['full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(row['t50_full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(row['hard_failure_full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(row['easy_degradation_vs_floor'])}` | `{_pct(row['floor_rate'])}` | `{_pct(row['stage43_m_rate'])}` | `{_pct(row['stage43_ab_rate'])}` |"
        )
    lines.extend(
        [
            "",
            "## Horizon Diagnostics",
            "",
            "| horizon | all | t50 | t100 | hard | easy | floor | Stage43-M | Stage43-AB |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for horizon, row in sorted(diag["horizons"].items(), key=lambda kv: int(kv[0])):
        lines.append(
            f"| `{horizon}` | `{_pct(row['full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(row['t50_full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(row['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}` | `{_pct(row['hard_failure_full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(row['easy_degradation_vs_floor'])}` | `{_pct(row['floor_rate'])}` | `{_pct(row['stage43_m_rate'])}` | `{_pct(row['stage43_ab_rate'])}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Stage43-AE adds a validation-selected three-route safety policy: original floor, Stage43-M protected latent policy, and Stage43-AB scene-proxy latent policy. This directly addresses the Stage43-AD TrajNet/h10/h100 easy-safety caveats without using the easy label as an inference input.",
            "",
            "## Boundary",
            "",
            "- This is still dataset-local/raw-frame 2.5D evidence.",
            "- t100 is guarded to the floor and remains diagnostic, not solved.",
            "- No future endpoint/waypoint input, no central velocity, no test endpoint goals, no test threshold tuning.",
            "- No metric/seconds claim, no Stage5C, no SMC.",
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ]
    )
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage43-AE Scene-Proxy Slice-Safe Policy Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- deploy slice-safe scene proxy: `{gate['deploy_slice_safe_scene_proxy']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ],
    )
    _update_text_outputs(payload)


def _update_text_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_ae_gate"]
    metrics = payload["test_metrics_slice_safe"]
    diag = payload["test_diagnostics_slice_safe"]
    delta = payload["delta_vs_stage43_m"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"deploy_slice_safe_scene_proxy = `{gate['deploy_slice_safe_scene_proxy']}`",
        "",
        f"full_waypoint_ade_vs_floor = `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`; delta_vs_stage43_m = `{_pct(delta['full_waypoint_ade_improvement_vs_floor'])}`",
        f"t50_full_waypoint_ade_vs_floor = `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`; delta_vs_stage43_m = `{_pct(delta['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"hard_failure_vs_floor = `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`; delta_vs_stage43_m = `{_pct(delta['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"t100_raw_frame_diagnostic = `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"max_domain_easy_degradation = `{_pct(diag['max_domain_easy_degradation'])}`",
        f"max_horizon_easy_degradation = `{_pct(diag['max_horizon_easy_degradation'])}`",
        "",
        "Stage43-AE is the slice-safe repair after the AD caveat audit. It uses a validation-selected three-route policy over floor, Stage43-M, and Stage43-AB, so weak easy slices can fall all the way back to the floor rather than only to Stage43-M.",
        "",
        "Boundary unchanged: dataset-local/raw-frame 2.5D only; t100 remains diagnostic; no metric/seconds claim, no Stage5C, no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_ae_scene_proxy_slice_safe_policy"] = {
        "source": payload["source"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "deploy_slice_safe_scene_proxy": gate["deploy_slice_safe_scene_proxy"],
        "metrics": payload["test_metrics_slice_safe"],
        "diagnostics": payload["test_diagnostics_slice_safe"],
        "delta_vs_stage43_m": payload["delta_vs_stage43_m"],
        "delta_vs_stage43_ac": payload["delta_vs_stage43_ac"],
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_ae_scene_proxy_slice_safe_policy"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    ensure_dir(LEDGER_JSONL.parent)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-AE",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "deploy_slice_safe_scene_proxy": gate["deploy_slice_safe_scene_proxy"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select a slice-safe three-route Stage43 scene-proxy latent policy.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true")
    group.add_argument("--small", action="store_true")
    group.add_argument("--medium", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=431)
    parser.add_argument("--bootstrap", type=int, default=1000)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    if not args.quick and not args.small and not args.medium:
        args.small = True
    result = _run(args)
    gate = result["stage43_ae_gate"]
    print(f"Stage43-AE: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"deploy_slice_safe_scene_proxy={gate['deploy_slice_safe_scene_proxy']}")
    return result


if __name__ == "__main__":
    main()
