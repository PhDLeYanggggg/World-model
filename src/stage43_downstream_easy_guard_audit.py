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
    _target_vec,
    _trajectory_error,
)
from src.stage43_full_waypoint_latent_robustness_audit import (
    STAGE43_M_JSON,
    _load_model,
    _standardize_from_checkpoint,
)
from src.stage43_latent_adapter_downstream_heads import (
    REPORT_JSON as STAGE43_CA_JSON,
    _fit_heads,
    _head_eval,
    _load_adapter,
    _predict_heads,
)
from src.stage43_latent_transition_adapter_repair import (
    REPORT_JSON as STAGE43_BZ_JSON,
    _adapter_predict,
)
from src.stage43_latent_transition_consistency_audit import _predict_transition_latents


REPORT_JSON = OUT_DIR / "stage43_downstream_easy_guard_audit.json"
REPORT_MD = OUT_DIR / "stage43_downstream_easy_guard_audit.md"
GATE_MD = OUT_DIR / "stage43_stage_cb_downstream_easy_guard_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_CB_DOWNSTREAM_EASY_GUARD_AUDIT"
SOURCE = "fresh_stage43_cb_downstream_easy_guard_audit"
SELECTED_VARIANT = "identity_stage43m_adapter_z"
EPS = 1e-8


def _encode_selected_variant(base_model: torch.nn.Module, adapter: Any, ds: Any, *, batch_size: int) -> np.ndarray:
    pred = _predict_transition_latents(base_model, ds, batch_size=int(batch_size))
    adapter_latent = _adapter_predict(adapter, ds.x, pred["z_t"], batch_size=int(batch_size))
    return np.concatenate([pred["z_t"], pred["z_next"], adapter_latent], axis=1).astype(np.float32)


def _candidate_errors(ds: Any, pred: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return _trajectory_error(ds, pred["waypoint"])


def _disagreement_features(ds: Any, pred: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    diff = pred["waypoint"].astype(np.float64) - ds.floor_waypoint_delta.astype(np.float64)
    mean_disagreement = np.linalg.norm(diff, axis=2).mean(axis=1).astype(np.float32)
    endpoint_disagreement = np.linalg.norm(diff[:, -1, :], axis=1).astype(np.float32)
    return {
        "model_floor_mean_disagreement": mean_disagreement,
        "model_floor_endpoint_disagreement": endpoint_disagreement,
    }


def _select_with_easy_guard(
    ds: Any,
    pred: Mapping[str, np.ndarray],
    candidate_ade: np.ndarray,
    candidate_fde: np.ndarray,
    disagreement: Mapping[str, np.ndarray],
    policy: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    allow = (
        (pred["gain"] >= float(policy["gain_threshold"]))
        & (pred["harm"] <= float(policy["harm_threshold"]))
        & (pred["failure"] >= float(policy["failure_threshold"]))
        & (disagreement["model_floor_mean_disagreement"] <= float(policy["disagreement_threshold"]))
        & (
            disagreement["model_floor_endpoint_disagreement"]
            <= float(policy["endpoint_disagreement_threshold"])
        )
    )
    selected_ade = np.where(allow, candidate_ade, ds.floor_ade).astype(np.float32)
    selected_fde = np.where(allow, candidate_fde, ds.floor_fde).astype(np.float32)
    return selected_ade, selected_fde, allow.astype(bool)


def _search_easy_guard_policy(ds: Any, pred: Mapping[str, np.ndarray]) -> dict[str, Any]:
    candidate_ade, candidate_fde = _candidate_errors(ds, pred)
    disagreement = _disagreement_features(ds, pred)
    dis_q = np.quantile(disagreement["model_floor_mean_disagreement"], [0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90])
    end_q = np.quantile(disagreement["model_floor_endpoint_disagreement"], [0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 0.90])
    best: dict[str, Any] | None = None
    evaluated = 0
    validation_safe = 0
    for gain in [0.25, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92, 0.97]:
        for harm in [0.03, 0.05, 0.08, 0.10, 0.15, 0.25, 0.35]:
            for failure in [0.10, 0.20, 0.35, 0.50, 0.65, 0.80]:
                for dis in dis_q:
                    for endpoint in end_q:
                        policy = {
                            "gain_threshold": float(gain),
                            "harm_threshold": float(harm),
                            "failure_threshold": float(failure),
                            "disagreement_threshold": float(dis),
                            "endpoint_disagreement_threshold": float(endpoint),
                        }
                        selected_ade, selected_fde, switched = _select_with_easy_guard(
                            ds, pred, candidate_ade, candidate_fde, disagreement, policy
                        )
                        metrics = _metrics(ds, selected_ade, selected_fde, switched)
                        evaluated += 1
                        if metrics["easy_degradation_vs_floor"] > 0.005:
                            continue
                        if metrics["switch_rate"] > 0.25:
                            continue
                        validation_safe += 1
                        objective = (
                            metrics["full_waypoint_ade_improvement_vs_floor"]
                            + metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
                            + 0.50 * metrics["t50_full_waypoint_ade_improvement_vs_floor"]
                            - 0.05 * metrics["switch_rate"]
                        )
                        row = {
                            "policy": policy,
                            "metrics": metrics,
                            "objective": float(objective),
                            "mode": "validation_easy_guard_search",
                        }
                        if best is None or row["objective"] > best["objective"]:
                            best = row
    if best is None:
        selected_ade = ds.floor_ade.copy()
        selected_fde = ds.floor_fde.copy()
        switched = np.zeros(len(ds.x), dtype=bool)
        return {
            "policy": {
                "gain_threshold": 1.01,
                "harm_threshold": -0.01,
                "failure_threshold": 1.01,
                "disagreement_threshold": 0.0,
                "endpoint_disagreement_threshold": 0.0,
            },
            "metrics": _metrics(ds, selected_ade, selected_fde, switched),
            "objective": 0.0,
            "mode": "validation_floor_fallback",
            "evaluated_policies": int(evaluated),
            "validation_safe_policies": int(validation_safe),
            "diagnostic": "no_validation_easy_safe_policy_found",
        }
    best["evaluated_policies"] = int(evaluated)
    best["validation_safe_policies"] = int(validation_safe)
    return best


def _apply_policy(ds: Any, pred: Mapping[str, np.ndarray], policy: Mapping[str, float]) -> dict[str, Any]:
    candidate_ade, candidate_fde = _candidate_errors(ds, pred)
    disagreement = _disagreement_features(ds, pred)
    selected_ade, selected_fde, switched = _select_with_easy_guard(
        ds, pred, candidate_ade, candidate_fde, disagreement, policy
    )
    return {
        "candidate_mean_ade": float(np.mean(candidate_ade)),
        "candidate_mean_fde": float(np.mean(candidate_fde)),
        "disagreement_summary": {
            "mean": float(np.mean(disagreement["model_floor_mean_disagreement"])),
            "p50": float(np.quantile(disagreement["model_floor_mean_disagreement"], 0.50)),
            "p90": float(np.quantile(disagreement["model_floor_mean_disagreement"], 0.90)),
            "endpoint_mean": float(np.mean(disagreement["model_floor_endpoint_disagreement"])),
            "endpoint_p50": float(np.quantile(disagreement["model_floor_endpoint_disagreement"], 0.50)),
            "endpoint_p90": float(np.quantile(disagreement["model_floor_endpoint_disagreement"], 0.90)),
        },
        "metrics": _metrics(ds, selected_ade, selected_fde, switched),
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "switched": switched,
    }


def _slice_summary(ds: Any, selected_ade: np.ndarray, selected_fde: np.ndarray, switched: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"domain": {}, "horizon": {}, "source_file": {}}
    for axis, values in [
        ("domain", ds.domain.astype(str)),
        ("horizon", ds.horizon.astype(str)),
        ("source_file", ds.source_file.astype(str)),
    ]:
        for value in sorted(set(values.tolist())):
            mask = values == value
            if int(mask.sum()) == 0:
                continue
            sub = type(
                "Slice",
                (),
                {
                    "x": ds.x[mask],
                    "floor_ade": ds.floor_ade[mask],
                    "floor_fde": ds.floor_fde[mask],
                    "hard": ds.hard[mask],
                    "failure": ds.failure[mask],
                    "easy": ds.easy[mask],
                    "horizon": ds.horizon[mask],
                },
            )()
            out[axis][str(value)] = _metrics(sub, selected_ade[mask], selected_fde[mask], switched[mask])
    return out


def _gap_summary(val_metrics: Mapping[str, Any], test_metrics: Mapping[str, Any]) -> dict[str, float]:
    keys = [
        "full_waypoint_ade_improvement_vs_floor",
        "t50_full_waypoint_ade_improvement_vs_floor",
        "hard_failure_full_waypoint_ade_improvement_vs_floor",
        "easy_degradation_vs_floor",
        "switch_rate",
    ]
    return {key: float(test_metrics[key] - val_metrics[key]) for key in keys}


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    val = payload["validation_easy_guard"]
    test = payload["test_once"]
    gates = {
        "stage43_ca_precondition_seen": payload["stage43_ca_precondition"]["verdict"]
        in {
            "stage43_ca_latent_adapter_downstream_heads_pass",
            "stage43_ca_latent_adapter_downstream_heads_partial_lift",
        },
        "fresh_guard_replay_completed": payload["result_source"] == "fresh_validation_only_easy_guard_replay",
        "train_only_heads_refit": payload["protocol"]["train_only_heads_refit"] is True,
        "future_labels_eval_only": payload["no_leakage"]["future_labels_as_inputs"] is False
        and payload["no_leakage"]["future_labels_train_eval_only"] is True,
        "no_test_threshold_tuning": payload["no_leakage"]["test_threshold_tuning"] is False,
        "inference_safe_guard_features_only": payload["no_leakage"]["guard_uses_future_labels"] is False
        and payload["no_leakage"]["guard_uses_test_endpoints"] is False,
        "validation_easy_safe_policy_found": val["metrics"]["easy_degradation_vs_floor"] <= 0.005
        and val["metrics"]["switch_rate"] > 0.0,
        "test_easy_preserved": test["metrics"]["easy_degradation_vs_floor"] <= 0.02,
        "protected_lift_vs_floor": test["metrics"]["full_waypoint_ade_improvement_vs_floor"] > 0.0
        or test["metrics"]["t50_full_waypoint_ade_improvement_vs_floor"] > 0.0
        or test["metrics"]["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0,
        "validation_test_easy_gap_reported": "easy_degradation_vs_floor" in payload["validation_test_gap"],
        "domain_horizon_source_breakdown_reported": bool(payload["test_once"]["slice_summary"].get("domain"))
        and bool(payload["test_once"]["slice_summary"].get("horizon"))
        and bool(payload["test_once"]["slice_summary"].get("source_file")),
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
        "long_objective_kept_active": payload["long_objective_complete"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    if passed == total:
        verdict = "stage43_cb_downstream_easy_guard_pass"
    elif gates["validation_easy_safe_policy_found"] and gates["protected_lift_vs_floor"]:
        verdict = "stage43_cb_downstream_easy_guard_val_safe_test_easy_mismatch"
    else:
        verdict = "stage43_cb_downstream_easy_guard_diagnostic_incomplete"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_cb_gate"]
    val = payload["validation_easy_guard"]["metrics"]
    test = payload["test_once"]["metrics"]
    return [
        "# Stage43-CB Downstream Easy Guard Audit",
        "",
        f"- source: `{SOURCE}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- selected latent variant: `{payload['selected_variant']}`",
        f"- evaluated validation policies: `{payload['validation_easy_guard']['evaluated_policies']}`",
        f"- validation-safe policies: `{payload['validation_easy_guard']['validation_safe_policies']}`",
        "- deployable policy changed: `False`",
        "",
        "## Validation-Selected Policy",
        "",
        f"- policy: `{payload['validation_easy_guard']['policy']}`",
        f"- validation all improvement: `{val['full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- validation t50 improvement: `{val['t50_full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- validation hard/failure improvement: `{val['hard_failure_full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- validation easy degradation: `{val['easy_degradation_vs_floor']:.4f}`",
        f"- validation switch rate: `{val['switch_rate']:.4f}`",
        "",
        "## Test Once",
        "",
        f"- test all improvement: `{test['full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- test t50 improvement: `{test['t50_full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- test hard/failure improvement: `{test['hard_failure_full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- test easy degradation: `{test['easy_degradation_vs_floor']:.4f}`",
        f"- test switch rate: `{test['switch_rate']:.4f}`",
        "",
        "## Validation-Test Gap",
        "",
        f"- easy degradation gap: `{payload['validation_test_gap']['easy_degradation_vs_floor']:.4f}`",
        f"- all improvement gap: `{payload['validation_test_gap']['full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- t50 improvement gap: `{payload['validation_test_gap']['t50_full_waypoint_ade_improvement_vs_floor']:.4f}`",
        "",
        "## Interpretation",
        "",
        "- Stage43-CB refits the downstream heads on train only, then searches a stricter easy guard on validation only.",
        "- The guard uses predicted failure/gain/harm plus model-vs-floor rollout disagreement. These are inference-safe quantities.",
        "- The test set is evaluated once with the validation-selected policy.",
        "- A validation-safe policy can still harm test easy rows. That is a deployment blocker, not a success.",
        "- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| `{key}` | `{bool(value)}` |" for key, value in gate["gates"].items()],
        "",
    ]


def _write_reports(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_cb_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-CB Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
        ],
    )
    test = payload["test_once"]["metrics"]
    world_lines = [
        "# Stage43 Current World-Model Gate",
        "",
        f"- source: `{SOURCE}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "- deployable policy changed: `False`",
        f"- test all improvement: `{test['full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- test t50 improvement: `{test['t50_full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- test hard/failure improvement: `{test['hard_failure_full_waypoint_ade_improvement_vs_floor']:.4f}`",
        f"- test easy degradation: `{test['easy_degradation_vs_floor']:.4f}`",
        "- long objective complete: `False`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "",
        "## Current Boundary",
        "",
        "- Stage43-CB is an easy-safety transfer audit for downstream latent heads.",
        "- It does not change the deployable model when test easy preservation fails.",
        "- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| `{key}` | `{bool(value)}` |" for key, value in gate["gates"].items()],
        "",
    ]
    write_json(WORLD_GATE_JSON, _jsonable(gate))
    write_md(WORLD_GATE_MD, world_lines)


def _update_summary_files(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_cb_gate"]
    val = payload["validation_easy_guard"]["metrics"]
    test = payload["test_once"]["metrics"]
    block = [
        f"## {SECTION}",
        "",
        f"source = `{SOURCE}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        "deployable_policy_changed = `False`",
        "",
        "Stage43-CB reruns the Stage43-CA selected latent downstream heads with a validation-only easy guard using predicted risk and model-vs-floor disagreement.",
        f"Validation all / hard / easy: `{val['full_waypoint_ade_improvement_vs_floor']:.4f}` / `{val['hard_failure_full_waypoint_ade_improvement_vs_floor']:.4f}` / `{val['easy_degradation_vs_floor']:.4f}`.",
        f"Test all / t50 / hard / easy: `{test['full_waypoint_ade_improvement_vs_floor']:.4f}` / `{test['t50_full_waypoint_ade_improvement_vs_floor']:.4f}` / `{test['hard_failure_full_waypoint_ade_improvement_vs_floor']:.4f}` / `{test['easy_degradation_vs_floor']:.4f}`.",
        f"Validation-test easy degradation gap: `{payload['validation_test_gap']['easy_degradation_vs_floor']:.4f}`.",
        "",
        "Interpretation: downstream latent heads still show all/hard signal, but easy-safety does not reliably transfer from validation to test. Deployment remains unchanged.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, block)
    state = read_json(RESEARCH_STATE, {})
    state["stage43_cb_downstream_easy_guard_audit"] = {
        "source": SOURCE,
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "validation_metrics": val,
        "test_metrics": test,
        "validation_test_gap": payload["validation_test_gap"],
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "deployable_policy_changed": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_cb_downstream_easy_guard_audit"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, _jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                _jsonable(
                    {
                        "event": SOURCE,
                        "verdict": gate["verdict"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def run_downstream_easy_guard_audit(*, batch_size: int = 8192, ridge: float = 1e-2) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage43m = read_json(STAGE43_M_JSON, {})
    stage43bz = read_json(STAGE43_BZ_JSON, {})
    stage43ca = read_json(STAGE43_CA_JSON, {})
    checkpoint, ckpt, base_model = _load_model(stage43m)
    adapter_path = Path(stage43bz.get("adapter_checkpoint", OUT_DIR / "checkpoints/stage43_latent_transition_adapter_repair.pt"))
    train = _standardize_from_checkpoint(_build_split("train", max_rows=None, seed=int(ckpt.get("seed", 431))), ckpt)
    val = _standardize_from_checkpoint(_build_split("val", max_rows=None, seed=int(ckpt.get("seed", 431))), ckpt)
    test = _standardize_from_checkpoint(_build_split("test", max_rows=None, seed=int(ckpt.get("seed", 431))), ckpt)
    probe = _predict_transition_latents(base_model, train, batch_size=int(batch_size))
    adapter = _load_adapter(adapter_path, train.x.shape[1], probe["z_t"].shape[1])
    train_latent = np.concatenate(
        [
            probe["z_t"],
            probe["z_next"],
            _adapter_predict(adapter, train.x, probe["z_t"], batch_size=int(batch_size)),
        ],
        axis=1,
    ).astype(np.float32)
    val_latent = _encode_selected_variant(base_model, adapter, val, batch_size=int(batch_size))
    test_latent = _encode_selected_variant(base_model, adapter, test, batch_size=int(batch_size))
    weights = _fit_heads(train_latent, train, ridge=float(ridge))
    val_pred = _predict_heads(val_latent, weights)
    test_pred = _predict_heads(test_latent, weights)
    val_policy = _search_easy_guard_policy(val, val_pred)
    val_apply = _apply_policy(val, val_pred, val_policy["policy"])
    test_apply = _apply_policy(test, test_pred, val_policy["policy"])
    val_summary = {
        "policy": val_policy["policy"],
        "metrics": val_apply["metrics"],
        "objective": float(val_policy["objective"]),
        "mode": val_policy["mode"],
        "evaluated_policies": int(val_policy["evaluated_policies"]),
        "validation_safe_policies": int(val_policy["validation_safe_policies"]),
        "candidate_mean_ade": val_apply["candidate_mean_ade"],
        "candidate_mean_fde": val_apply["candidate_mean_fde"],
        "disagreement_summary": val_apply["disagreement_summary"],
    }
    test_summary = {
        "policy": val_policy["policy"],
        "metrics": test_apply["metrics"],
        "candidate_mean_ade": test_apply["candidate_mean_ade"],
        "candidate_mean_fde": test_apply["candidate_mean_fde"],
        "disagreement_summary": test_apply["disagreement_summary"],
        "slice_summary": _slice_summary(test, test_apply["selected_ade"], test_apply["selected_fde"], test_apply["switched"]),
        "bootstrap": _bootstrap_ci(test, test_apply["selected_ade"], test_apply["selected_fde"], n=1000, seed=977),
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_validation_only_easy_guard_replay",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "stage43_m_precondition": {
            "verdict": stage43m.get("stage43_m_gate", {}).get("verdict"),
            "checkpoint": str(checkpoint),
        },
        "stage43_bz_precondition": {
            "verdict": stage43bz.get("stage43_bz_gate", {}).get("verdict"),
            "adapter_checkpoint": str(adapter_path),
        },
        "stage43_ca_precondition": {
            "verdict": stage43ca.get("stage43_ca_gate", {}).get("verdict"),
            "selected_adapter_variant": stage43ca.get("selected_adapter_variant"),
            "ca_test_metrics": stage43ca.get("variants", {})
            .get(stage43ca.get("selected_adapter_variant", ""), {})
            .get("protected", {})
            .get("test_metrics_with_floor", {}),
        },
        "protocol": {
            "train_only_heads_refit": True,
            "ridge": float(ridge),
            "batch_size": int(batch_size),
            "selected_variant": SELECTED_VARIANT,
            "target_vec_shape": list(_target_vec(train).shape),
            "num_workers": 0,
        },
        "rows": {"train": int(len(train.x)), "val": int(len(val.x)), "test": int(len(test.x))},
        "selected_variant": SELECTED_VARIANT,
        "head_eval": {
            "validation": _head_eval(val, val_pred),
            "test": _head_eval(test, test_pred),
        },
        "validation_easy_guard": val_summary,
        "test_once": test_summary,
        "validation_test_gap": _gap_summary(val_apply["metrics"], test_apply["metrics"]),
        "no_leakage": {
            "future_labels_as_inputs": False,
            "future_labels_train_eval_only": True,
            "future_waypoint_input": False,
            "future_endpoint_input": False,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_threshold_tuning": False,
            "test_statistics_normalization": False,
            "guard_uses_future_labels": False,
            "guard_uses_test_endpoints": False,
        },
        "claim_boundary": {
            "deployable_policy_changed": False,
            "metric_or_seconds_claim": False,
            "true_3d_claim": False,
            "foundation_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "long_objective_complete": False,
    }
    payload["stage43_cb_gate"] = _gate(payload)
    _write_reports(payload)
    _update_summary_files(payload)
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Audit validation-only easy guard transfer for Stage43-CA downstream heads.")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--ridge", type=float, default=1e-2)
    args = parser.parse_args(argv)
    payload = run_downstream_easy_guard_audit(batch_size=args.batch_size, ridge=args.ridge)
    gate = payload["stage43_cb_gate"]
    test = payload["test_once"]["metrics"]
    print(f"Stage43-CB: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"test_all={test['full_waypoint_ade_improvement_vs_floor']:.4f}")
    print(f"test_t50={test['t50_full_waypoint_ade_improvement_vs_floor']:.4f}")
    print(f"test_hard={test['hard_failure_full_waypoint_ade_improvement_vs_floor']:.4f}")
    print(f"test_easy={test['easy_degradation_vs_floor']:.4f}")
    return payload


if __name__ == "__main__":
    main()
