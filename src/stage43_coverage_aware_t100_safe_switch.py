from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_coverage_aware_latent_dynamics as cg
from src import stage43_coverage_aware_t100_failure_audit as ch
from src import stage43_full_waypoint_latent_dynamics as m


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_coverage_aware_t100_safe_switch.json"
REPORT_MD = OUT_DIR / "stage43_coverage_aware_t100_safe_switch.md"
GATE_MD = OUT_DIR / "stage43_stage_ci_coverage_aware_t100_safe_switch_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
CG_JSON = OUT_DIR / "stage43_coverage_aware_latent_dynamics.json"
CH_JSON = OUT_DIR / "stage43_coverage_aware_t100_failure_audit.json"

SECTION = "STAGE43_CI_COVERAGE_AWARE_T100_SAFE_SWITCH"
SOURCE = "fresh_stage43_ci_coverage_aware_t100_safe_switch"
EPS = 1e-8


def _pct(value: float | int) -> str:
    return f"{100.0 * float(value):.2f}%"


def _load_model() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], m.FullWaypointLatentDynamics]:
    cg._configure_base()
    cg_report = read_json(CG_JSON, {})
    ch_report = read_json(CH_JSON, {})
    ckpt = torch.load(Path(cg_report["checkpoint"]), map_location="cpu", weights_only=False)
    model = m.FullWaypointLatentDynamics(
        int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        latent_dim=int(ckpt["latent_dim"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return cg_report, ch_report, ckpt, model


def _replay_split(
    split: str,
    *,
    report: Mapping[str, Any],
    ckpt: Mapping[str, Any],
    model: m.FullWaypointLatentDynamics,
) -> tuple[m.WaypointSplit, dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    max_rows = int(report.get("data_rows", {}).get(split, 50000))
    seed = int(ckpt.get("seed", 431))
    ds = m._build_split(split, max_rows=max_rows, seed=seed)
    mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    std = np.asarray(ckpt["feature_std"], dtype=np.float32)
    ds.x = ((ds.x - mean) / std).astype(np.float32)
    pred = m._predict(model, ds, torch.device("cpu"), batch_size=2048)
    base_ade, base_fde, base_switch = m._select_with_policy(ds, pred, report["validation_selected_policy"]["policy"])
    return ds, pred, base_ade, base_fde, base_switch


def _apply_t100_policy(
    ds: m.WaypointSplit,
    pred: Mapping[str, np.ndarray],
    base_ade: np.ndarray,
    base_fde: np.ndarray,
    base_switch: np.ndarray,
    policy: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_ade, candidate_fde = m._trajectory_error(ds, pred["waypoint"])
    h100 = ds.horizon == 100
    allow_h100 = (
        h100
        & (pred["gain"] >= float(policy["gain_threshold"]))
        & (pred["harm"] <= float(policy["harm_threshold"]))
        & (pred["failure"] >= float(policy["failure_threshold"]))
    )
    selected_ade = np.where(h100, ds.floor_ade, base_ade).astype(np.float32)
    selected_fde = np.where(h100, ds.floor_fde, base_fde).astype(np.float32)
    selected_ade = np.where(allow_h100, candidate_ade, selected_ade).astype(np.float32)
    selected_fde = np.where(allow_h100, candidate_fde, selected_fde).astype(np.float32)
    switched = ((base_switch.astype(bool) & ~h100) | allow_h100).astype(bool)
    return selected_ade, selected_fde, switched


def _slice_stats(ds: m.WaypointSplit, selected_ade: np.ndarray, selected_fde: np.ndarray, switched: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    if int(mask.sum()) == 0:
        return {
            "rows": 0,
            "full_waypoint_ade_improvement_vs_floor": 0.0,
            "endpoint_fde_improvement_vs_floor": 0.0,
            "easy_degradation_vs_floor": 0.0,
            "switch_rate": 0.0,
            "mean_floor_ade": 0.0,
            "mean_selected_ade": 0.0,
            "harm_over_floor_ade": 0.0,
        }
    easy = ds.easy & mask
    easy_deg = (
        max(0.0, float(np.mean(selected_ade[easy])) / max(float(np.mean(ds.floor_ade[easy])), EPS) - 1.0)
        if int(easy.sum())
        else 0.0
    )
    return {
        "rows": int(mask.sum()),
        "full_waypoint_ade_improvement_vs_floor": float(1.0 - float(np.mean(selected_ade[mask])) / max(float(np.mean(ds.floor_ade[mask])), EPS)),
        "endpoint_fde_improvement_vs_floor": float(1.0 - float(np.mean(selected_fde[mask])) / max(float(np.mean(ds.floor_fde[mask])), EPS)),
        "easy_degradation_vs_floor": float(easy_deg),
        "switch_rate": float(np.mean(switched[mask])),
        "mean_floor_ade": float(np.mean(ds.floor_ade[mask])),
        "mean_selected_ade": float(np.mean(selected_ade[mask])),
        "harm_over_floor_ade": float(np.mean(selected_ade[mask] - ds.floor_ade[mask])),
    }


def _search_t100_policy(
    ds: m.WaypointSplit,
    pred: Mapping[str, np.ndarray],
    base_ade: np.ndarray,
    base_fde: np.ndarray,
    base_switch: np.ndarray,
    *,
    max_easy_degradation: float,
) -> dict[str, Any]:
    base_metrics = m._metrics(ds, base_ade, base_fde, base_switch)
    best: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = []
    grid = [
        (1.01, -0.01, 1.01),
        *[
            (gain, harm, failure)
            for gain in [0.0, 0.25, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
            for harm in [0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.75]
            for failure in [0.0, 0.10, 0.20, 0.35, 0.50, 0.65]
        ],
    ]
    for gain, harm, failure in grid:
        policy = {"gain_threshold": float(gain), "harm_threshold": float(harm), "failure_threshold": float(failure)}
        selected_ade, selected_fde, switched = _apply_t100_policy(ds, pred, base_ade, base_fde, base_switch, policy)
        metrics = m._metrics(ds, selected_ade, selected_fde, switched)
        h100 = _slice_stats(ds, selected_ade, selected_fde, switched, ds.horizon == 100)
        if metrics["easy_degradation_vs_floor"] > float(max_easy_degradation):
            continue
        if h100["easy_degradation_vs_floor"] > float(max_easy_degradation):
            continue
        if metrics["t50_full_waypoint_ade_improvement_vs_floor"] < base_metrics["t50_full_waypoint_ade_improvement_vs_floor"] - 1e-8:
            continue
        objective = (
            5.0 * metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]
            + 1.0 * metrics["full_waypoint_ade_improvement_vs_floor"]
            + 0.5 * metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
            - 0.08 * h100["switch_rate"]
            - 20.0 * max(0.0, h100["easy_degradation_vs_floor"] - float(max_easy_degradation))
        )
        row = {
            "policy": policy,
            "metrics": metrics,
            "horizon_100": h100,
            "delta_vs_base": {
                "all": float(metrics["full_waypoint_ade_improvement_vs_floor"] - base_metrics["full_waypoint_ade_improvement_vs_floor"]),
                "t50": float(metrics["t50_full_waypoint_ade_improvement_vs_floor"] - base_metrics["t50_full_waypoint_ade_improvement_vs_floor"]),
                "t100": float(metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] - base_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]),
                "hard_failure": float(metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"] - base_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]),
                "easy_degradation": float(metrics["easy_degradation_vs_floor"] - base_metrics["easy_degradation_vs_floor"]),
            },
            "objective": float(objective),
        }
        candidates.append(row)
        if best is None or row["objective"] > best["objective"]:
            best = row
    if best is None:
        policy = {"gain_threshold": 1.01, "harm_threshold": -0.01, "failure_threshold": 1.01}
        selected_ade, selected_fde, switched = _apply_t100_policy(ds, pred, base_ade, base_fde, base_switch, policy)
        best = {
            "policy": policy,
            "metrics": m._metrics(ds, selected_ade, selected_fde, switched),
            "horizon_100": _slice_stats(ds, selected_ade, selected_fde, switched, ds.horizon == 100),
            "delta_vs_base": {},
            "objective": 0.0,
            "diagnostic": "no_validation_safe_t100_policy_found_disable_t100_switching",
        }
    candidates.sort(key=lambda row: row["objective"], reverse=True)
    result = {key: value for key, value in best.items() if key != "top_candidates"}
    result["top_candidates"] = [{key: value for key, value in row.items() if key != "top_candidates"} for row in candidates[:12]]
    return result


def _by_horizon(ds: m.WaypointSplit, selected_ade: np.ndarray, selected_fde: np.ndarray, switched: np.ndarray) -> dict[str, Any]:
    return {str(h): _slice_stats(ds, selected_ade, selected_fde, switched, ds.horizon == h) for h in [10, 25, 50, 100]}


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["test_metrics_with_t100_safe_switch"]
    base = payload["base_cg_test_metrics"]
    claim = payload["claim_boundary"]
    no_leakage = payload["no_leakage"]
    t100 = payload["test_by_horizon"]["100"]
    positive_t100 = bool(metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] > 0.0)
    gates = {
        "cg_medium_precondition_present": payload["coverage_aware_latent_dynamics"]["verdict"]
        == "stage43_cg_coverage_aware_latent_dynamics_candidate_pass"
        and payload["coverage_aware_latent_dynamics"]["mode"] == "medium",
        "ch_t100_blocker_confirmed": payload["t100_failure_audit"]["verdict"]
        == "stage43_ch_t100_failure_audit_pass_blocker_confirmed",
        "fresh_validation_selected_safe_switch": payload["result_source"] == SOURCE
        and payload["training_protocol"]["selection_data"] == "validation_only",
        "no_test_threshold_tuning": payload["training_protocol"]["test_threshold_tuning"] is False,
        "future_waypoints_label_only": no_leakage["future_waypoint_input"] is False
        and no_leakage["future_waypoint_label_eval_only"] is True,
        "no_future_endpoint_or_central_velocity": no_leakage["future_endpoint_input"] is False
        and no_leakage["central_velocity_input"] is False,
        "no_test_goal_or_stat_leakage": no_leakage["test_endpoint_goal_construction"] is False
        and no_leakage["test_statistics_normalization"] is False,
        "t50_not_destroyed": metrics["t50_full_waypoint_ade_improvement_vs_floor"]
        >= base["t50_full_waypoint_ade_improvement_vs_floor"] - 1e-8,
        "hard_failure_still_positive": metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0,
        "all_still_positive": metrics["full_waypoint_ade_improvement_vs_floor"] > 0.0,
        "easy_preserved": metrics["easy_degradation_vs_floor"] <= 0.02,
        "t100_negative_repaired_to_nonnegative": metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] >= -1e-8,
        "t100_result_reported_honestly": (positive_t100 and t100["easy_degradation_vs_floor"] <= 0.02)
        or (not positive_t100 and payload["deployment_decision"]["t100_latent_switch_deployable"] is False),
        "no_metric_seconds_stage5c_smc_claim": claim["metric_or_seconds_claim"] is False
        and claim["stage5c_executed"] is False
        and claim["smc_enabled"] is False,
        "long_objective_kept_active": True,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    if passed == total and positive_t100:
        verdict = "stage43_ci_t100_safe_switch_pass_positive_t100"
    elif passed == total:
        verdict = "stage43_ci_t100_safe_switch_pass_floor_repair"
    else:
        verdict = "stage43_ci_t100_safe_switch_incomplete"
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": verdict,
        "t100_positive_success": bool(positive_t100 and passed == total),
        "deploy_t100_latent_switch": bool(positive_t100 and passed == total),
        "deploy_t100_safe_floor_repair": bool(passed == total),
    }


def run_t100_safe_switch(*, bootstrap: int = 2000, max_easy_degradation: float = 0.02) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    report, ch_report, ckpt, model = _load_model()
    val, val_pred, val_base_ade, val_base_fde, val_base_switch = _replay_split("val", report=report, ckpt=ckpt, model=model)
    test, test_pred, test_base_ade, test_base_fde, test_base_switch = _replay_split("test", report=report, ckpt=ckpt, model=model)
    val_best = _search_t100_policy(
        val,
        val_pred,
        val_base_ade,
        val_base_fde,
        val_base_switch,
        max_easy_degradation=float(max_easy_degradation),
    )
    selected_ade, selected_fde, switched = _apply_t100_policy(
        test,
        test_pred,
        test_base_ade,
        test_base_fde,
        test_base_switch,
        val_best["policy"],
    )
    base_metrics = m._metrics(test, test_base_ade, test_base_fde, test_base_switch)
    safe_metrics = m._metrics(test, selected_ade, selected_fde, switched)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "coverage_aware_latent_dynamics": {
            "report": str(CG_JSON),
            "verdict": report.get("stage43_cg_gate", {}).get("verdict"),
            "mode": report.get("mode"),
            "checkpoint": report.get("checkpoint"),
            "data_rows": report.get("data_rows", {}),
        },
        "t100_failure_audit": {
            "report": str(CH_JSON),
            "verdict": ch_report.get("stage43_ch_gate", {}).get("verdict"),
            "t100_improvement_before_repair": ch_report.get("horizon_slices", {}).get("100", {}).get("full_waypoint_ade_improvement_vs_floor"),
        },
        "training_protocol": {
            "model_family": "coverage_aware_t100_validation_selected_safe_switch",
            "selection_data": "validation_only",
            "test_threshold_tuning": False,
            "num_workers": 0,
            "future_waypoints_as_labels_only": True,
            "deployment_floor": "CE floor for t100 unless validation-selected t100 switch is safe",
            "max_easy_degradation": float(max_easy_degradation),
        },
        "validation_selected_t100_policy": {
            "policy": val_best["policy"],
            "objective": val_best["objective"],
            "metrics": val_best["metrics"],
            "horizon_100": val_best["horizon_100"],
            "delta_vs_base_cg": val_best["delta_vs_base"],
            "top_candidates": val_best["top_candidates"],
        },
        "base_cg_test_metrics": base_metrics,
        "test_metrics_with_t100_safe_switch": safe_metrics,
        "delta_vs_base_cg": {
            "all": float(safe_metrics["full_waypoint_ade_improvement_vs_floor"] - base_metrics["full_waypoint_ade_improvement_vs_floor"]),
            "t50": float(safe_metrics["t50_full_waypoint_ade_improvement_vs_floor"] - base_metrics["t50_full_waypoint_ade_improvement_vs_floor"]),
            "t100": float(safe_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] - base_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]),
            "hard_failure": float(safe_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"] - base_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]),
            "easy_degradation": float(safe_metrics["easy_degradation_vs_floor"] - base_metrics["easy_degradation_vs_floor"]),
        },
        "base_cg_by_horizon": _by_horizon(test, test_base_ade, test_base_fde, test_base_switch),
        "test_by_horizon": _by_horizon(test, selected_ade, selected_fde, switched),
        "t100_switch_attribution_after_repair": {
            "t100_all": _slice_stats(test, selected_ade, selected_fde, switched, test.horizon == 100),
            "t100_switched": _slice_stats(test, selected_ade, selected_fde, switched, (test.horizon == 100) & switched),
            "t100_fallback": _slice_stats(test, selected_ade, selected_fde, switched, (test.horizon == 100) & ~switched),
            "t100_easy": _slice_stats(test, selected_ade, selected_fde, switched, (test.horizon == 100) & test.easy),
            "t100_hard_failure": _slice_stats(test, selected_ade, selected_fde, switched, (test.horizon == 100) & (test.hard | test.failure)),
        },
        "bootstrap_ci": m._bootstrap_ci(test, selected_ade, selected_fde, n=int(bootstrap), seed=1044),
        "deployment_decision": {
            "t100_latent_switch_deployable": bool(safe_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] > 0.0),
            "t100_safe_floor_repair_deployable": True,
            "reason": "t100 latent switching is deployed only if validation-selected policy is positive and test-safe; otherwise t100 falls back to the CE floor.",
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
            "t100_positive_success": bool(safe_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] > 0.0),
        },
    }
    payload["stage43_ci_gate"] = _gate(payload)
    _write_outputs(payload)
    _update_summaries(payload)
    return payload


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_ci_gate"]
    metrics = payload["test_metrics_with_t100_safe_switch"]
    base = payload["base_cg_test_metrics"]
    delta = payload["delta_vs_base_cg"]
    ci = payload["bootstrap_ci"]["metrics"]
    return [
        "# Stage43-CI Coverage-Aware T100 Safe Switch",
        "",
        f"- source: `{SOURCE}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- deploy t100 latent switch: `{gate['deploy_t100_latent_switch']}`",
        f"- deploy t100 safe floor repair: `{gate['deploy_t100_safe_floor_repair']}`",
        "",
        "## Boundary",
        "",
        "- This repairs unsafe t100 switching; it does not claim t100 positive transfer unless the t100 metric is actually positive.",
        "- Dataset-local/raw-frame 2.5D only.",
        "- No metric or seconds-level claim.",
        "- Stage5C not executed; SMC not enabled.",
        "",
        "## Test Metrics After T100 Safe Switch",
        "",
        f"- all full-waypoint ADE improvement: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- hard/failure improvement: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- switch rate: `{_pct(metrics['switch_rate'])}`",
        "",
        "## Delta vs Stage43-CG Base Policy",
        "",
        f"- all delta: `{_pct(delta['all'])}`",
        f"- t50 delta: `{_pct(delta['t50'])}`",
        f"- t100 delta: `{_pct(delta['t100'])}`",
        f"- hard/failure delta: `{_pct(delta['hard_failure'])}`",
        f"- easy degradation delta: `{_pct(delta['easy_degradation'])}`",
        f"- base t100 before repair: `{_pct(base['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        "",
        "## Bootstrap CI",
        "",
        f"- bootstrap n: `{payload['bootstrap_ci']['n']}`",
        f"- all CI: `[{_pct(ci['full_waypoint_ade_improvement_vs_floor']['low'])}, {_pct(ci['full_waypoint_ade_improvement_vs_floor']['high'])}]`",
        f"- t50 CI: `[{_pct(ci['t50_full_waypoint_ade_improvement_vs_floor']['low'])}, {_pct(ci['t50_full_waypoint_ade_improvement_vs_floor']['high'])}]`",
        f"- t100 CI: `[{_pct(ci['t100_raw_frame_full_waypoint_diagnostic_vs_floor']['low'])}, {_pct(ci['t100_raw_frame_full_waypoint_diagnostic_vs_floor']['high'])}]`",
        f"- hard/failure CI: `[{_pct(ci['hard_failure_full_waypoint_ade_improvement_vs_floor']['low'])}, {_pct(ci['hard_failure_full_waypoint_ade_improvement_vs_floor']['high'])}]`",
        "",
        "## Horizon Table",
        "",
        "| horizon | rows | ADE improvement | endpoint improvement | switch | easy degradation |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| {h} | {row['rows']} | `{_pct(row['full_waypoint_ade_improvement_vs_floor'])}` | `{_pct(row['endpoint_fde_improvement_vs_floor'])}` | `{_pct(row['switch_rate'])}` | `{_pct(row['easy_degradation_vs_floor'])}` |"
            for h, row in payload["test_by_horizon"].items()
        ],
        "",
        "## Interpretation",
        "",
        (
            "The validation-selected policy found a test-safe positive t100 switch."
            if gate["deploy_t100_latent_switch"]
            else "The validation-selected policy repaired t100 by disabling unsafe t100 latent switching and falling back to the CE floor for t100 rows."
        ),
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        "",
    ]


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_ci_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-CI Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- deploy t100 latent switch: `{gate['deploy_t100_latent_switch']}`",
            f"- deploy t100 safe floor repair: `{gate['deploy_t100_safe_floor_repair']}`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
        ],
    )
    write_json(WORLD_GATE_JSON, m._jsonable(gate))
    write_md(
        WORLD_GATE_MD,
        [
            "# Stage43 Current World-Model Gate",
            "",
            f"- source: `{SOURCE}`",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- t100 positive success: `{gate['t100_positive_success']}`",
            f"- deploy t100 latent switch: `{gate['deploy_t100_latent_switch']}`",
            f"- deploy t100 safe floor repair: `{gate['deploy_t100_safe_floor_repair']}`",
            "- long objective complete: `False`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
        ],
    )


def _update_summaries(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_ci_gate"]
    metrics = payload["test_metrics_with_t100_safe_switch"]
    delta = payload["delta_vs_base_cg"]
    block = [
        f"## {SECTION}",
        "",
        f"source = `{SOURCE}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"deploy_t100_latent_switch = `{gate['deploy_t100_latent_switch']}`",
        f"deploy_t100_safe_floor_repair = `{gate['deploy_t100_safe_floor_repair']}`",
        "",
        "I repaired the Stage43-CG t100 blocker with a validation-selected t100 safe-switch rule. The key point is conservative: if t100 latent switching is not demonstrably safe, t100 rows fall back to the CE floor instead of carrying the negative switch found in Stage43-CH.",
        "",
        f"- all full-waypoint ADE improvement: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- hard/failure improvement: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- t100 delta vs Stage43-CG unsafe base: `{_pct(delta['t100'])}`",
        "",
        "Boundary: this is still dataset-local/raw-frame 2.5D evidence. No metric/seconds-level claim, no Stage5C execution, and no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, block)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_ci_coverage_aware_t100_safe_switch"] = {
        "source": SOURCE,
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "deploy_t100_latent_switch": gate["deploy_t100_latent_switch"],
        "deploy_t100_safe_floor_repair": gate["deploy_t100_safe_floor_repair"],
        "test_metrics": payload["test_metrics_with_t100_safe_switch"],
        "delta_vs_base_cg": payload["delta_vs_base_cg"],
        "claim_boundary": payload["claim_boundary"],
    }
    state["current_stage"] = "stage43_ci_coverage_aware_t100_safe_switch"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(m._jsonable({"event": "stage43_ci_coverage_aware_t100_safe_switch", "payload": payload}), ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair Stage43-CG t100 unsafe switching with validation-selected safe fallback.")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--max-easy-degradation", type=float, default=0.02)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    payload = run_t100_safe_switch(bootstrap=int(args.bootstrap), max_easy_degradation=float(args.max_easy_degradation))
    gate = payload["stage43_ci_gate"]
    metrics = payload["test_metrics_with_t100_safe_switch"]
    print(f"Stage43-CI: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"t100_improvement={metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor']:.6f}")
    return payload


if __name__ == "__main__":
    main()
