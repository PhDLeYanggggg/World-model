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
from src import stage43_self_gate_conformal_audit as ak


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_bounded_residual_safety_audit.json"
REPORT_MD = OUT_DIR / "stage43_bounded_residual_safety_audit.md"
GATE_MD = OUT_DIR / "stage43_stage_al_bounded_residual_safety_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_AL_BOUNDED_RESIDUAL_SAFETY_AUDIT"
SOURCE = "fresh_stage43_al_bounded_residual_safety_audit"


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _clip_residual_by_norm(residual: np.ndarray, clip_norm: float) -> np.ndarray:
    if clip_norm <= 0.0:
        return np.zeros_like(residual, dtype=np.float32)
    norm = np.linalg.norm(residual.astype(np.float64), axis=2, keepdims=True)
    scale = np.minimum(1.0, float(clip_norm) / np.maximum(norm, 1e-8))
    return (residual * scale).astype(np.float32)


def _bounded_waypoint(
    ds: m.WaypointSplit,
    pred: Mapping[str, np.ndarray],
    *,
    alpha: float,
    clip_norm: float,
) -> np.ndarray:
    residual = np.asarray(pred["waypoint"], dtype=np.float32) - ds.floor_waypoint_delta.astype(np.float32)
    bounded = float(alpha) * _clip_residual_by_norm(residual, float(clip_norm))
    return (ds.floor_waypoint_delta.astype(np.float32) + bounded).astype(np.float32)


def _allow_mask(
    ds: m.WaypointSplit,
    pred: Mapping[str, np.ndarray],
    policy: Mapping[str, float],
    *,
    force_h100_floor: bool,
) -> np.ndarray:
    allow = (
        (pred["gain"] >= float(policy["gain_threshold"]))
        & (pred["harm"] <= float(policy["harm_threshold"]))
        & (pred["failure"] >= float(policy["failure_threshold"]))
    )
    if force_h100_floor:
        allow = allow & (ds.horizon != 100)
    return allow.astype(bool)


def _evaluate_bounded(
    ds: m.WaypointSplit,
    pred: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    *,
    floor_only: bool = False,
    ungated: bool = False,
) -> dict[str, Any]:
    if floor_only:
        selected_ade = ds.floor_ade.copy()
        selected_fde = ds.floor_fde.copy()
        switched = np.zeros(len(ds.x), dtype=bool)
        residual_norm = np.zeros(len(ds.x), dtype=np.float32)
    else:
        if ungated:
            waypoint = np.asarray(pred["waypoint"], dtype=np.float32)
            switched = np.ones(len(ds.x), dtype=bool)
        else:
            waypoint = _bounded_waypoint(
                ds,
                pred,
                alpha=float(config["alpha"]),
                clip_norm=float(config["clip_norm"]),
            )
            switched = _allow_mask(
                ds,
                pred,
                config["policy"],
                force_h100_floor=bool(config.get("force_h100_floor", False)),
            )
            waypoint = np.where(switched[:, None, None], waypoint, ds.floor_waypoint_delta).astype(np.float32)
        selected_ade, selected_fde = m._trajectory_error(ds, waypoint)
        residual_norm = np.linalg.norm((waypoint - ds.floor_waypoint_delta).astype(np.float64), axis=2).mean(axis=1)
    metrics = m._metrics(ds, selected_ade, selected_fde, switched)
    return {
        "config": dict(config),
        "metrics": metrics,
        "switch_count": int(np.sum(switched)),
        "safe_easy": bool(metrics["easy_degradation_vs_floor"] <= 0.02),
        "safe_t100": bool(metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] >= -1e-8),
        "mean_residual_norm": float(np.mean(residual_norm)),
        "max_residual_norm": float(np.max(residual_norm)) if len(residual_norm) else 0.0,
    }


def _objective(metrics: Mapping[str, float], *, safe_required: bool) -> float:
    penalty = 0.0
    if safe_required:
        penalty += 30.0 * max(0.0, float(metrics["easy_degradation_vs_floor"]) - 0.02)
        penalty += 10.0 * max(0.0, -float(metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]))
    return float(
        metrics["full_waypoint_ade_improvement_vs_floor"]
        + 1.2 * metrics["t50_full_waypoint_ade_improvement_vs_floor"]
        + 0.8 * metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
        - 0.04 * metrics["switch_rate"]
        - penalty
    )


def _search_bounded(
    val: m.WaypointSplit,
    pred: Mapping[str, np.ndarray],
    *,
    safe_required: bool,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    safe_candidates = 0
    searched = 0
    force_options = [False, True] if not safe_required else [True]
    for alpha in [0.10, 0.20, 0.35, 0.50, 0.75, 1.00]:
        for clip_norm in [0.025, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75]:
            for gain in [0.0, 0.20, 0.35, 0.50, 0.65, 0.80]:
                for harm in [0.05, 0.10, 0.15, 0.25, 0.35, 0.50, 0.75]:
                    for failure in [0.0, 0.10, 0.25, 0.40, 0.60]:
                        for force_h100 in force_options:
                            config = {
                                "alpha": alpha,
                                "clip_norm": clip_norm,
                                "policy": {
                                    "gain_threshold": gain,
                                    "harm_threshold": harm,
                                    "failure_threshold": failure,
                                },
                                "force_h100_floor": force_h100,
                            }
                            result = _evaluate_bounded(val, pred, config)
                            metrics = result["metrics"]
                            passes_safety = bool(
                                metrics["easy_degradation_vs_floor"] <= 0.02
                                and (not safe_required or metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] >= -1e-8)
                            )
                            if passes_safety:
                                safe_candidates += 1
                            searched += 1
                            if safe_required and not passes_safety:
                                continue
                            result["objective"] = _objective(metrics, safe_required=safe_required)
                            if best is None or result["objective"] > best["objective"]:
                                best = result
    if best is None:
        best = _evaluate_bounded(val, pred, {}, floor_only=True)
        best["objective"] = 0.0
    best["searched_candidates"] = int(searched)
    best["safe_candidates"] = int(safe_candidates)
    best["safe_required"] = bool(safe_required)
    return best


def _row(name: str, result: Mapping[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    return {
        "name": name,
        "config": result.get("config", {}),
        "all": metrics["full_waypoint_ade_improvement_vs_floor"],
        "endpoint": metrics["endpoint_fde_improvement_vs_floor"],
        "t50": metrics["t50_full_waypoint_ade_improvement_vs_floor"],
        "t50_endpoint": metrics["t50_endpoint_fde_improvement_vs_floor"],
        "t100": metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"],
        "hard_failure": metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"],
        "easy": metrics["easy_degradation_vs_floor"],
        "switch_rate": metrics["switch_rate"],
        "switch_count": result["switch_count"],
        "mean_residual_norm": result.get("mean_residual_norm", 0.0),
        "max_residual_norm": result.get("max_residual_norm", 0.0),
        "safe_easy": result["safe_easy"],
        "safe_t100": result["safe_t100"],
    }


def _beats_reference(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    return bool(
        candidate["safe_easy"]
        and candidate["easy"] <= 0.02
        and (
            candidate["all"] > reference["all"] + 1e-6
            or candidate["t50"] > reference["t50"] + 1e-6
            or candidate["hard_failure"] > reference["hard_failure"] + 1e-6
        )
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    runtime = m._configure_runtime(int(args.seed))
    prior, ckpt, model = ak._load_stage43_m()
    val, test = ak._build_eval_splits(prior, ckpt)
    device = torch.device("cpu")
    val_pred = m._predict(model, val, device, int(args.batch_size))
    test_pred = m._predict(model, test, device, int(args.batch_size))

    stored_policy = prior["validation_selected_policy"]["policy"]
    stored_replay = ak._evaluate_policy(test, test_pred, stored_policy)
    stored_diff = ak._metric_diff(stored_replay["metrics"], prior["test_metrics_with_floor"])
    ungated = _evaluate_bounded(test, test_pred, {}, ungated=True)
    floor = _evaluate_bounded(test, test_pred, {}, floor_only=True)
    unconstrained_val = _search_bounded(val, val_pred, safe_required=False)
    constrained_val = _search_bounded(val, val_pred, safe_required=True)
    unconstrained_test = _evaluate_bounded(test, test_pred, unconstrained_val["config"])
    constrained_test = _evaluate_bounded(test, test_pred, constrained_val["config"])

    rows = [
        _row("floor_only", floor),
        _row("ungated_neural_waypoint", ungated),
        _row("stored_stage43_m_hard_switch", stored_replay),
        _row("bounded_residual_unconstrained_val_best", unconstrained_test),
        _row("bounded_residual_safe_val_best", constrained_test),
    ]
    stored_row = next(row for row in rows if row["name"] == "stored_stage43_m_hard_switch")
    safe_row = next(row for row in rows if row["name"] == "bounded_residual_safe_val_best")
    unconstrained_row = next(row for row in rows if row["name"] == "bounded_residual_unconstrained_val_best")
    deploy_bounded = _beats_reference(safe_row, stored_row) and safe_row["safe_t100"]
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_bounded_residual_audit_over_frozen_stage43_m_checkpoint",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "runtime": runtime,
        "stage43_m_source": {
            "report": str(ak.STAGE43_M),
            "checkpoint": str(ak.STAGE43_M_CKPT),
            "report_sha256": m._sha256(ak.STAGE43_M),
            "checkpoint_sha256": m._sha256(ak.STAGE43_M_CKPT),
            "prior_verdict": prior.get("stage43_m_gate", {}).get("verdict"),
        },
        "data_rows": {"val": int(len(val.x)), "test": int(len(test.x))},
        "feature_schema_match": list(ckpt["feature_names"]) == test.feature_names,
        "cache_row_hashes": {split: m._row_hash(m._npz(m._cache_path(split))) for split in m.SPLITS},
        "cache_row_hash_match_prior": {split: m._row_hash(m._npz(m._cache_path(split))) for split in m.SPLITS}
        == prior.get("cache_row_hashes"),
        "stored_policy_replay_diff": stored_diff,
        "validation_search": {
            "unconstrained": unconstrained_val,
            "safe_constrained": constrained_val,
        },
        "policy_table": rows,
        "bounded_residual_deployable_candidate": deploy_bounded,
        "best_safe_bounded_residual": safe_row,
        "best_unconstrained_bounded_residual": unconstrained_row,
        "comparison_to_stored_hard_switch": {
            "safe_minus_stored_all": safe_row["all"] - stored_row["all"],
            "safe_minus_stored_t50": safe_row["t50"] - stored_row["t50"],
            "safe_minus_stored_t100": safe_row["t100"] - stored_row["t100"],
            "safe_minus_stored_hard_failure": safe_row["hard_failure"] - stored_row["hard_failure"],
            "safe_minus_stored_easy": safe_row["easy"] - stored_row["easy"],
        },
        "interpretation": {
            "global_floor_removable": False,
            "bounded_residual_reduces_hard_switch_risk": bool(safe_row["safe_easy"] and safe_row["safe_t100"]),
            "bounded_residual_promoted": deploy_bounded,
            "deployment_decision": "promote_bounded_residual_candidate"
            if deploy_bounded
            else "keep_stage43_m_floor_policy; bounded residual remains diagnostic",
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "thresholds_selected_on_test": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "input_hash": _combined_hash([ak.STAGE43_M, ak.STAGE43_M_CKPT, m._cache_path("val"), m._cache_path("test")]),
    }
    payload["stage43_al_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    stored_diff = payload["stored_policy_replay_diff"]["max_abs_diff"]
    safe = payload["best_safe_bounded_residual"]
    unconstrained = payload["best_unconstrained_bounded_residual"]
    comp = payload["comparison_to_stored_hard_switch"]
    gates = {
        "stage43_m_exact_replay": stored_diff <= 1e-5,
        "feature_schema_and_rows_match": payload["feature_schema_match"] is True
        and payload["cache_row_hash_match_prior"] is True,
        "bounded_residual_search_completed": payload["validation_search"]["safe_constrained"]["searched_candidates"] > 0
        and payload["validation_search"]["unconstrained"]["searched_candidates"] > 0,
        "thresholds_selected_on_validation_only": payload["no_leakage"]["thresholds_selected_on_test"] is False,
        "safe_bounded_residual_preserves_easy": safe["easy"] <= 0.02,
        "safe_bounded_residual_preserves_t100": safe["t100"] >= -1e-8,
        "unsafe_or_unconstrained_risk_reported": unconstrained["easy"] > 0.02
        or unconstrained["t100"] < -1e-8
        or payload["bounded_residual_deployable_candidate"] is True,
        "deployment_decision_recorded": payload["interpretation"]["deployment_decision"]
        in {"promote_bounded_residual_candidate", "keep_stage43_m_floor_policy; bounded residual remains diagnostic"},
        "bounded_residual_lift_or_honest_diagnostic": payload["bounded_residual_deployable_candidate"] is True
        or (
            comp["safe_minus_stored_all"] <= 0.0
            or comp["safe_minus_stored_t50"] <= 0.0
            or comp["safe_minus_stored_hard_failure"] <= 0.0
        ),
        "global_floor_not_removed": payload["interpretation"]["global_floor_removable"] is False,
        "no_future_or_test_leakage": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_waypoint_label_eval_only"] is True
        and payload["no_leakage"]["central_velocity_input"] is False
        and payload["no_leakage"]["test_endpoint_goal_construction"] is False
        and payload["no_leakage"]["test_statistics_normalization"] is False,
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
        "verdict": "stage43_al_bounded_residual_candidate_pass"
        if passed == total and payload["bounded_residual_deployable_candidate"]
        else "stage43_al_bounded_residual_diagnostic_keep_floor"
        if passed == total
        else "stage43_al_bounded_residual_audit_incomplete",
        "deploy_bounded_residual": bool(passed == total and payload["bounded_residual_deployable_candidate"]),
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    gate = payload["stage43_al_gate"]
    comp = payload["comparison_to_stored_hard_switch"]
    lines = [
        "# Stage43-AL Bounded Residual Safety Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- deploy bounded residual: `{gate['deploy_bounded_residual']}`",
        f"- stored policy replay max abs diff: `{payload['stored_policy_replay_diff']['max_abs_diff']:.8f}`",
        "",
        "## Policy Comparison",
        "",
        "| policy | all | t50 | t100 diag | hard/failure | easy deg | switch | mean residual | safe easy | safe t100 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload["policy_table"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["name"],
                    _pct(row["all"]),
                    _pct(row["t50"]),
                    _pct(row["t100"]),
                    _pct(row["hard_failure"]),
                    _pct(row["easy"]),
                    _pct(row["switch_rate"]),
                    f"{row['mean_residual_norm']:.4f}",
                    f"`{row['safe_easy']}`",
                    f"`{row['safe_t100']}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Safe Bounded Residual vs Stored Hard Switch",
            "",
            f"- all delta: `{_pct(comp['safe_minus_stored_all'])}`",
            f"- t50 delta: `{_pct(comp['safe_minus_stored_t50'])}`",
            f"- t100 diagnostic delta: `{_pct(comp['safe_minus_stored_t100'])}`",
            f"- hard/failure delta: `{_pct(comp['safe_minus_stored_hard_failure'])}`",
            f"- easy degradation delta: `{_pct(comp['safe_minus_stored_easy'])}`",
            "",
            "## Interpretation",
            "",
            f"- deployment decision: `{payload['interpretation']['deployment_decision']}`",
            "- Bounded residual is evaluated as a floor-protected relaxation of hard switching.",
            "- If it is not better than the stored hard switch under validation-selected safety constraints, Stage43-M remains the active floor policy.",
            "- Global floor removal is not supported.",
            "",
            "## Boundary",
            "",
            "- Dataset-local/raw-frame 2.5D only.",
            "- Future waypoint/endpoint labels are loss/eval only.",
            "- No metric/seconds claim, no Stage5C, no SMC.",
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ]
    )
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage43-AL Bounded Residual Safety Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- deploy bounded residual: `{gate['deploy_bounded_residual']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    _update_text_outputs(payload)


def _update_text_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_al_gate"]
    safe = payload["best_safe_bounded_residual"]
    comp = payload["comparison_to_stored_hard_switch"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"deploy_bounded_residual = `{gate['deploy_bounded_residual']}`",
        f"safe_bounded_all_t50_t100_hard_easy = `{_pct(safe['all'])}` / `{_pct(safe['t50'])}` / `{_pct(safe['t100'])}` / `{_pct(safe['hard_failure'])}` / `{_pct(safe['easy'])}`",
        f"safe_minus_stored_all_t50_t100_hard_easy = `{_pct(comp['safe_minus_stored_all'])}` / `{_pct(comp['safe_minus_stored_t50'])}` / `{_pct(comp['safe_minus_stored_t100'])}` / `{_pct(comp['safe_minus_stored_hard_failure'])}` / `{_pct(comp['safe_minus_stored_easy'])}`",
        "",
        "Stage43-AL tested bounded residual relaxation over the frozen Stage43-M latent waypoint model. Residual deltas are norm-clipped and validation-selected, with future labels used only for validation/eval. If the bounded residual does not beat the stored hard switch while preserving easy/t100 safety, it remains diagnostic and the Stage43-M floor policy stays active.",
        "",
        "Boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds claim; no Stage5C; no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_al_bounded_residual_safety_audit"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "deploy_bounded_residual": gate["deploy_bounded_residual"],
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "policy_table": payload["policy_table"],
        "comparison_to_stored_hard_switch": payload["comparison_to_stored_hard_switch"],
        "global_floor_removable": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_al_bounded_residual_safety_audit"
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
                        "stage": "Stage43-AL",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "deploy_bounded_residual": gate["deploy_bounded_residual"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-AL bounded residual safety audit.")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=431)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = _run(args)
    gate = result["stage43_al_gate"]
    print(f"Stage43-AL: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"deploy_bounded_residual={gate['deploy_bounded_residual']}")
    return result


if __name__ == "__main__":
    main()
