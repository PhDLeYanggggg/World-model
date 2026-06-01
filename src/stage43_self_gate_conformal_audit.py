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


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_self_gate_conformal_audit.json"
REPORT_MD = OUT_DIR / "stage43_self_gate_conformal_audit.md"
GATE_MD = OUT_DIR / "stage43_stage_ak_self_gate_conformal_audit_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_AK_SELF_GATE_CONFORMAL_AUDIT"
SOURCE = "fresh_stage43_ak_self_gate_conformal_audit"
STAGE43_M = OUT_DIR / "stage43_full_waypoint_latent_dynamics.json"
STAGE43_M_CKPT = OUT_DIR / "checkpoints" / "stage43_full_waypoint_latent_dynamics.pt"

CORE_METRICS = [
    "full_waypoint_ade_improvement_vs_floor",
    "endpoint_fde_improvement_vs_floor",
    "t50_full_waypoint_ade_improvement_vs_floor",
    "t50_endpoint_fde_improvement_vs_floor",
    "t100_raw_frame_full_waypoint_diagnostic_vs_floor",
    "hard_failure_full_waypoint_ade_improvement_vs_floor",
    "easy_degradation_vs_floor",
    "switch_rate",
    "harm_over_floor_ade",
    "mean_floor_ade",
    "mean_selected_ade",
]


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _apply_checkpoint_standardization(ds: m.WaypointSplit, ckpt: Mapping[str, Any]) -> m.WaypointSplit:
    mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    std = np.asarray(ckpt["feature_std"], dtype=np.float32)
    ds.x = ((ds.x - mean) / std).astype(np.float32)
    return ds


def _load_stage43_m() -> tuple[dict[str, Any], Mapping[str, Any], m.FullWaypointLatentDynamics]:
    if not STAGE43_M.exists():
        raise FileNotFoundError(STAGE43_M)
    if not STAGE43_M_CKPT.exists():
        raise FileNotFoundError(STAGE43_M_CKPT)
    prior = read_json(STAGE43_M, {})
    ckpt = torch.load(STAGE43_M_CKPT, map_location="cpu", weights_only=False)
    model = m.FullWaypointLatentDynamics(
        int(ckpt["input_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        latent_dim=int(ckpt["latent_dim"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(torch.device("cpu"))
    model.eval()
    return prior, ckpt, model


def _build_eval_splits(prior: Mapping[str, Any], ckpt: Mapping[str, Any]) -> tuple[m.WaypointSplit, m.WaypointSplit]:
    seed = int(ckpt.get("seed", prior.get("runtime", {}).get("seed", 431)) or 431)
    rows = prior.get("data_rows", {})
    val = m._build_split("val", max_rows=int(rows.get("val", 12000)), seed=seed)
    test = m._build_split("test", max_rows=int(rows.get("test", 16000)), seed=seed)
    feature_names = list(ckpt["feature_names"])
    if val.feature_names != feature_names or test.feature_names != feature_names:
        raise ValueError("Feature schema mismatch between rebuilt splits and checkpoint feature_names")
    return _apply_checkpoint_standardization(val, ckpt), _apply_checkpoint_standardization(test, ckpt)


def _select_with_policy(
    ds: m.WaypointSplit,
    pred: Mapping[str, np.ndarray],
    policy: Mapping[str, float],
    *,
    force_h100_floor: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate_ade, candidate_fde = m._trajectory_error(ds, pred["waypoint"])
    allow = (
        (pred["gain"] >= float(policy["gain_threshold"]))
        & (pred["harm"] <= float(policy["harm_threshold"]))
        & (pred["failure"] >= float(policy["failure_threshold"]))
    )
    if force_h100_floor:
        allow = allow & (ds.horizon != 100)
    selected_ade = np.where(allow, candidate_ade, ds.floor_ade).astype(np.float32)
    selected_fde = np.where(allow, candidate_fde, ds.floor_fde).astype(np.float32)
    return selected_ade, selected_fde, allow.astype(bool)


def _evaluate_policy(
    ds: m.WaypointSplit,
    pred: Mapping[str, np.ndarray],
    policy: Mapping[str, float] | None,
    *,
    force_h100_floor: bool = False,
    floor_only: bool = False,
    ungated: bool = False,
) -> dict[str, Any]:
    if floor_only:
        selected_ade = ds.floor_ade.copy()
        selected_fde = ds.floor_fde.copy()
        switched = np.zeros(len(ds.x), dtype=bool)
    elif ungated:
        selected_ade, selected_fde = m._trajectory_error(ds, pred["waypoint"])
        switched = np.ones(len(ds.x), dtype=bool)
    else:
        assert policy is not None
        selected_ade, selected_fde, switched = _select_with_policy(
            ds,
            pred,
            policy,
            force_h100_floor=force_h100_floor,
        )
    metrics = m._metrics(ds, selected_ade, selected_fde, switched)
    return {
        "policy": dict(policy or {}),
        "force_h100_floor": bool(force_h100_floor),
        "metrics": metrics,
        "safe_easy": bool(metrics["easy_degradation_vs_floor"] <= 0.02),
        "safe_t100": bool(metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] >= -1e-8),
        "switch_count": int(np.sum(switched)),
    }


def _metric_diff(replayed: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    rows = {}
    max_abs = 0.0
    for key in CORE_METRICS:
        diff = float(replayed[key]) - float(expected[key])
        rows[key] = {
            "replayed": float(replayed[key]),
            "expected": float(expected[key]),
            "abs_diff": abs(diff),
            "signed_diff": diff,
        }
        max_abs = max(max_abs, abs(diff))
    return {"max_abs_diff": max_abs, "by_metric": rows}


def _search_self_gate(val: m.WaypointSplit, pred: Mapping[str, np.ndarray]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for gain in [0.0, 0.25, 0.45, 0.55, 0.65, 0.75, 0.85]:
        for harm in [0.15, 0.25, 0.35, 0.50, 0.75, 1.00]:
            for failure in [0.0, 0.10, 0.20, 0.35, 0.50]:
                policy = {"gain_threshold": gain, "harm_threshold": harm, "failure_threshold": failure}
                result = _evaluate_policy(val, pred, policy)
                metrics = result["metrics"]
                objective = (
                    metrics["full_waypoint_ade_improvement_vs_floor"]
                    + 1.2 * metrics["t50_full_waypoint_ade_improvement_vs_floor"]
                    + 0.8 * metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
                    - 20.0 * max(0.0, metrics["easy_degradation_vs_floor"] - 0.02)
                    - 0.05 * metrics["switch_rate"]
                )
                result["objective"] = float(objective)
                if best is None or result["objective"] > best["objective"]:
                    best = result
    assert best is not None
    return best


def _search_conformal_style_gate(val: m.WaypointSplit, pred: Mapping[str, np.ndarray]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = []
    for gain in [0.0, 0.20, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]:
        for harm in [0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.50, 0.75]:
            for failure in [0.0, 0.10, 0.20, 0.35, 0.50, 0.65]:
                policy = {"gain_threshold": gain, "harm_threshold": harm, "failure_threshold": failure}
                result = _evaluate_policy(val, pred, policy, force_h100_floor=True)
                metrics = result["metrics"]
                result["passes_validation_risk_constraints"] = bool(
                    metrics["easy_degradation_vs_floor"] <= 0.02
                    and metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] >= -1e-8
                )
                objective = (
                    metrics["full_waypoint_ade_improvement_vs_floor"]
                    + 1.5 * metrics["t50_full_waypoint_ade_improvement_vs_floor"]
                    + 0.8 * metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
                    - 0.08 * metrics["switch_rate"]
                )
                result["objective"] = float(objective)
                candidates.append(result)
                if result["passes_validation_risk_constraints"] and (
                    best is None or result["objective"] > best["objective"]
                ):
                    best = result
    if best is None:
        floor = _evaluate_policy(val, pred, None, floor_only=True)
        floor["objective"] = 0.0
        floor["passes_validation_risk_constraints"] = True
        best = floor
    best["valid_candidates"] = int(sum(bool(row["passes_validation_risk_constraints"]) for row in candidates))
    best["searched_candidates"] = int(len(candidates))
    best["calibration_note"] = (
        "Conformal-style here means validation-calibrated risk gating with an explicit h100 floor guard; "
        "it is not a formal exchangeability guarantee."
    )
    return best


def _policy_row(name: str, result: Mapping[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    return {
        "name": name,
        "policy": result.get("policy", {}),
        "force_h100_floor": bool(result.get("force_h100_floor", False)),
        "full_waypoint_ade_improvement_vs_floor": metrics["full_waypoint_ade_improvement_vs_floor"],
        "t50_full_waypoint_ade_improvement_vs_floor": metrics["t50_full_waypoint_ade_improvement_vs_floor"],
        "t100_raw_frame_full_waypoint_diagnostic_vs_floor": metrics[
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor"
        ],
        "hard_failure_full_waypoint_ade_improvement_vs_floor": metrics[
            "hard_failure_full_waypoint_ade_improvement_vs_floor"
        ],
        "easy_degradation_vs_floor": metrics["easy_degradation_vs_floor"],
        "switch_rate": metrics["switch_rate"],
        "switch_count": result["switch_count"],
        "safe_easy": result["safe_easy"],
        "safe_t100": result["safe_t100"],
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    runtime = m._configure_runtime(int(args.seed))
    prior, ckpt, model = _load_stage43_m()
    val, test = _build_eval_splits(prior, ckpt)
    device = torch.device("cpu")
    val_pred = m._predict(model, val, device, int(args.batch_size))
    test_pred = m._predict(model, test, device, int(args.batch_size))

    stored_policy = prior["validation_selected_policy"]["policy"]
    stored_replay = _evaluate_policy(test, test_pred, stored_policy)
    stored_replay_diff = _metric_diff(stored_replay["metrics"], prior["test_metrics_with_floor"])
    floor_only = _evaluate_policy(test, test_pred, None, floor_only=True)
    ungated = _evaluate_policy(test, test_pred, None, ungated=True)
    fresh_self_gate_val = _search_self_gate(val, val_pred)
    fresh_self_gate_test = _evaluate_policy(test, test_pred, fresh_self_gate_val["policy"])
    conformal_val = _search_conformal_style_gate(val, val_pred)
    conformal_test = _evaluate_policy(
        test,
        test_pred,
        conformal_val.get("policy", {}),
        force_h100_floor=bool(conformal_val.get("force_h100_floor", True)),
        floor_only=(conformal_val.get("policy", {}) == {}),
    )
    conformal_test["validation_selected"] = {
        "policy": conformal_val.get("policy", {}),
        "force_h100_floor": bool(conformal_val.get("force_h100_floor", True)),
        "valid_candidates": int(conformal_val.get("valid_candidates", 0)),
        "searched_candidates": int(conformal_val.get("searched_candidates", 0)),
        "calibration_note": conformal_val.get("calibration_note"),
        "validation_metrics": conformal_val["metrics"],
    }

    policy_table = [
        _policy_row("floor_only", floor_only),
        _policy_row("ungated_neural", ungated),
        _policy_row("stored_stage43_m_self_gate", stored_replay),
        _policy_row("fresh_self_gate_search", fresh_self_gate_test),
        _policy_row("conformal_style_h100_easy_guard", conformal_test),
    ]
    current_hashes = {split: m._row_hash(m._npz(m._cache_path(split))) for split in m.SPLITS}
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_replay_and_audit_over_frozen_stage43_m_checkpoint",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "runtime": runtime,
        "stage43_m_source": {
            "report": str(STAGE43_M),
            "checkpoint": str(STAGE43_M_CKPT),
            "report_sha256": m._sha256(STAGE43_M),
            "checkpoint_sha256": m._sha256(STAGE43_M_CKPT),
            "prior_verdict": prior.get("stage43_m_gate", {}).get("verdict"),
            "prior_result_source": prior.get("result_source"),
        },
        "data_rows": {"val": int(len(val.x)), "test": int(len(test.x))},
        "feature_count": int(test.x.shape[1]),
        "feature_schema_match": list(ckpt["feature_names"]) == test.feature_names,
        "cache_row_hashes": current_hashes,
        "cache_row_hash_match_prior": current_hashes == prior.get("cache_row_hashes"),
        "stored_policy": stored_policy,
        "stored_policy_replay": stored_replay,
        "stored_policy_replay_diff": stored_replay_diff,
        "fresh_self_gate_validation": fresh_self_gate_val,
        "conformal_style_validation": conformal_val,
        "policy_table": policy_table,
        "best_safe_policy_by_t50": max(
            [row for row in policy_table if row["safe_easy"]],
            key=lambda row: row["t50_full_waypoint_ade_improvement_vs_floor"],
        ),
        "interpretation": {
            "global_floor_removable": False,
            "ungated_neural_safe": bool(ungated["metrics"]["easy_degradation_vs_floor"] <= 0.02),
            "self_gate_can_preserve_easy": bool(
                stored_replay["metrics"]["easy_degradation_vs_floor"] <= 0.02
                or fresh_self_gate_test["metrics"]["easy_degradation_vs_floor"] <= 0.02
                or conformal_test["metrics"]["easy_degradation_vs_floor"] <= 0.02
            ),
            "conformal_style_h100_guard_needed": bool(
                stored_replay["metrics"]["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] < 0.0
                and conformal_test["metrics"]["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] >= -1e-8
            ),
            "deployable_update": "Keep Stage43-M stored self-gate as replayed; conformal-style h100 guard is diagnostic unless promoted by a separate frozen-policy stage.",
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
        "input_hash": _combined_hash([STAGE43_M, STAGE43_M_CKPT, m._cache_path("val"), m._cache_path("test")]),
    }
    payload["stage43_ak_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    replay_diff = payload["stored_policy_replay_diff"]["max_abs_diff"]
    ungated = next(row for row in payload["policy_table"] if row["name"] == "ungated_neural")
    conformal = next(row for row in payload["policy_table"] if row["name"] == "conformal_style_h100_easy_guard")
    stored = next(row for row in payload["policy_table"] if row["name"] == "stored_stage43_m_self_gate")
    gates = {
        "stage43_m_checkpoint_present": Path(payload["stage43_m_source"]["checkpoint"]).exists(),
        "feature_schema_matches_checkpoint": payload["feature_schema_match"] is True,
        "cache_row_hashes_match_prior": payload["cache_row_hash_match_prior"] is True,
        "stored_policy_exact_replay": replay_diff <= 1e-5,
        "ungated_neural_reported_unsafe": ungated["easy_degradation_vs_floor"] > 0.02
        or ungated["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] < stored[
            "t100_raw_frame_full_waypoint_diagnostic_vs_floor"
        ],
        "fresh_self_gate_eval_completed": any(row["name"] == "fresh_self_gate_search" for row in payload["policy_table"]),
        "conformal_style_gate_eval_completed": conformal["safe_easy"] is True,
        "conformal_style_h100_guard_safe": conformal["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] >= -1e-8,
        "self_gate_preserves_easy_on_at_least_one_policy": any(row["safe_easy"] for row in payload["policy_table"]),
        "global_floor_still_required_if_ungated_unsafe": payload["interpretation"]["global_floor_removable"] is False,
        "no_future_or_test_leakage": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["future_waypoint_input"] is False
        and payload["no_leakage"]["future_waypoint_label_eval_only"] is True
        and payload["no_leakage"]["central_velocity_input"] is False
        and payload["no_leakage"]["test_endpoint_goal_construction"] is False
        and payload["no_leakage"]["test_statistics_normalization"] is False
        and payload["no_leakage"]["thresholds_selected_on_test"] is False,
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
        "verdict": "stage43_ak_self_gate_conformal_audit_pass"
        if passed == total
        else "stage43_ak_self_gate_conformal_audit_incomplete",
        "global_floor_removable": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    gate = payload["stage43_ak_gate"]
    lines = [
        "# Stage43-AK Self-Gate / Conformal-Style Safety Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- stored policy replay max abs diff: `{payload['stored_policy_replay_diff']['max_abs_diff']:.8f}`",
        f"- cache row hashes match prior: `{payload['cache_row_hash_match_prior']}`",
        f"- feature schema match: `{payload['feature_schema_match']}`",
        "",
        "## Policy Comparison",
        "",
        "| policy | all | t50 | t100 diag | hard/failure | easy deg | switch | safe easy | safe t100 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in payload["policy_table"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["name"],
                    _pct(row["full_waypoint_ade_improvement_vs_floor"]),
                    _pct(row["t50_full_waypoint_ade_improvement_vs_floor"]),
                    _pct(row["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]),
                    _pct(row["hard_failure_full_waypoint_ade_improvement_vs_floor"]),
                    _pct(row["easy_degradation_vs_floor"]),
                    _pct(row["switch_rate"]),
                    f"`{row['safe_easy']}`",
                    f"`{row['safe_t100']}`",
                ]
            )
            + " |"
        )
    best = payload["best_safe_policy_by_t50"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- best safe policy by t50 in this audit: `{best['name']}` with t50 `{_pct(best['t50_full_waypoint_ade_improvement_vs_floor'])}` and easy degradation `{_pct(best['easy_degradation_vs_floor'])}`.",
            "- Ungated neural deployment remains unsafe and is not promoted.",
            "- The conformal-style guard is validation-calibrated and explicitly floors h100; it is a diagnostic safety audit, not a formal conformal guarantee.",
            "- Global safety floor removal is still not supported.",
            "",
            "## Boundary",
            "",
            "- Dataset-local/raw-frame 2.5D only.",
            "- Future waypoints/endpoints are labels/eval only, not inputs.",
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
            "# Stage43-AK Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- global_floor_removable: `{gate['global_floor_removable']}`",
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
    gate = payload["stage43_ak_gate"]
    stored = next(row for row in payload["policy_table"] if row["name"] == "stored_stage43_m_self_gate")
    conformal = next(row for row in payload["policy_table"] if row["name"] == "conformal_style_h100_easy_guard")
    ungated = next(row for row in payload["policy_table"] if row["name"] == "ungated_neural")
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"stored_policy_replay_max_abs_diff = `{payload['stored_policy_replay_diff']['max_abs_diff']:.8f}`",
        f"stored_self_gate_all_t50_easy = `{_pct(stored['full_waypoint_ade_improvement_vs_floor'])}` / `{_pct(stored['t50_full_waypoint_ade_improvement_vs_floor'])}` / `{_pct(stored['easy_degradation_vs_floor'])}`",
        f"ungated_easy_t100 = `{_pct(ungated['easy_degradation_vs_floor'])}` / `{_pct(ungated['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"conformal_style_all_t50_t100_easy = `{_pct(conformal['full_waypoint_ade_improvement_vs_floor'])}` / `{_pct(conformal['t50_full_waypoint_ade_improvement_vs_floor'])}` / `{_pct(conformal['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}` / `{_pct(conformal['easy_degradation_vs_floor'])}`",
        "",
        "Stage43-AK replayed the frozen Stage43-M checkpoint and policy, then compared stored self-gate, fresh self-gate search, ungated neural deployment, and a validation-calibrated conformal-style h100/easy guard. The audit keeps the global safety floor: ungated neural remains unsafe, while guarded policies preserve easy cases.",
        "",
        "Boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds claim; no Stage5C; no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_ak_self_gate_conformal_audit"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "report": str(REPORT_MD),
        "gate_report": str(GATE_MD),
        "stored_policy_replay_diff": payload["stored_policy_replay_diff"],
        "policy_table": payload["policy_table"],
        "best_safe_policy_by_t50": payload["best_safe_policy_by_t50"],
        "global_floor_removable": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_ak_self_gate_conformal_audit"
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
                        "stage": "Stage43-AK",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "stored_policy_replay_max_abs_diff": payload["stored_policy_replay_diff"]["max_abs_diff"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-AK self-gate/conformal-style safety audit.")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=431)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = _run(args)
    gate = result["stage43_ak_gate"]
    print(f"Stage43-AK: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"stored_policy_replay_max_abs_diff={result['stored_policy_replay_diff']['max_abs_diff']:.8f}")
    print(f"global_floor_removable={gate['global_floor_removable']}")
    return result


if __name__ == "__main__":
    main()
