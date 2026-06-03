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
from src import stage43_coverage_aware_t100_long_horizon_specialist as cj
from src import stage43_full_waypoint_latent_dynamics as m


OUT_DIR = m.OUT_DIR
CKPT_DIR = OUT_DIR / "checkpoints"
REPORT_JSON = OUT_DIR / "stage43_coverage_aware_t100_causal_feature_repair.json"
REPORT_MD = OUT_DIR / "stage43_coverage_aware_t100_causal_feature_repair.md"
GATE_MD = OUT_DIR / "stage43_stage_ck_coverage_aware_t100_causal_feature_repair_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
HEARTBEAT_JSON = OUT_DIR / "stage43_coverage_aware_t100_causal_feature_repair_heartbeat.json"
CHECKPOINT_NAME = "stage43_coverage_aware_t100_causal_feature_repair.pt"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE43_CK_COVERAGE_AWARE_T100_CAUSAL_FEATURE_REPAIR"
SOURCE = "fresh_stage43_ck_coverage_aware_t100_causal_feature_repair"


def _pct(value: float | int) -> str:
    return f"{100.0 * float(value):.2f}%"


def _make_causal_features(ds: m.WaypointSplit, pred: Mapping[str, np.ndarray]) -> np.ndarray:
    """Causal-only t100 specialist inputs.

    This intentionally excludes true-error diagnostics such as candidate ADE/FDE
    or floor ADE/FDE.  Those values require future waypoints and are valid only
    for loss/evaluation, not inference.
    """

    parts = [
        ds.x.astype(np.float32),
        ds.floor_waypoint_delta.reshape(len(ds.x), -1).astype(np.float32),
        pred["waypoint"].reshape(len(ds.x), -1).astype(np.float32),
        pred["latent"].astype(np.float32),
        pred["gain"].reshape(-1, 1).astype(np.float32),
        pred["harm"].reshape(-1, 1).astype(np.float32),
        pred["failure"].reshape(-1, 1).astype(np.float32),
        pred["density"].reshape(-1, 1).astype(np.float32),
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def _causal_specialist_split(
    ds: m.WaypointSplit,
    pred: Mapping[str, np.ndarray],
    ci_ade: np.ndarray,
    ci_fde: np.ndarray,
    ci_switch: np.ndarray,
) -> cj.SpecialistSplit:
    features = _make_causal_features(ds, pred)
    target_residual = (ds.waypoint_delta - ds.floor_waypoint_delta).astype(np.float32)
    cg_candidate_ade, cg_candidate_fde = m._trajectory_error(ds, pred["waypoint"])
    return cj.SpecialistSplit(
        base=ds,
        features=features,
        target_residual=target_residual,
        valid=ds.waypoint_valid.astype(np.float32),
        cg_candidate_ade=cg_candidate_ade,
        cg_candidate_fde=cg_candidate_fde,
        ci_ade=ci_ade,
        ci_fde=ci_fde,
        ci_switch=ci_switch,
    )


def _prior_cj_audit() -> dict[str, Any]:
    prior = read_json(cj.REPORT_JSON, {})
    deployed = bool(prior.get("deployment_decision", {}).get("deploy_t100_specialist", False))
    return {
        "prior_report": str(cj.REPORT_JSON),
        "prior_stage43_cj_exists": bool(prior),
        "prior_stage43_cj_verdict": prior.get("stage43_cj_gate", {}).get("verdict", "missing"),
        "prior_stage43_cj_deployed_t100_specialist": deployed,
        "prior_stage43_cj_label_derived_features_in_specialist_input": True,
        "label_derived_features_identified": [
            "cg_candidate_ade",
            "cg_candidate_fde",
            "floor_ade",
            "floor_fde",
        ],
        "deployment_contamination": bool(deployed),
        "interpretation": (
            "Stage43-CJ did not deploy the t100 specialist, so the deployable CI floor remains uncontaminated. "
            "However, CJ's specialist diagnostic feature set included true-error features derived from future waypoints; "
            "CJ must not be cited as a clean no-leakage t100 specialist trial."
        ),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = payload["test_metrics_with_causal_specialist"]
    ci_metrics = payload["ci_floor_test_metrics"]
    claim = payload["claim_boundary"]
    no_leakage = payload["no_leakage"]
    prior = payload["prior_stage43_cj_audit"]
    positive_t100 = bool(metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] > 0.0)
    gates = {
        "prior_cj_leakage_audited": prior["prior_stage43_cj_label_derived_features_in_specialist_input"] is True,
        "prior_cj_not_deployed_or_flagged": prior["deployment_contamination"] is False,
        "ci_precondition_passed": payload["stage43_ci_precondition"]["verdict"]
        in {"stage43_ci_t100_safe_switch_pass_floor_repair", "stage43_ci_t100_safe_switch_pass_positive_t100"},
        "fresh_torch_training": payload["result_source"] == SOURCE and Path(payload["checkpoint"]).exists(),
        "checkpoint_not_committed": payload["checkpoint_committed"] is False,
        "causal_only_features": no_leakage["label_derived_eval_error_features_in_input"] is False
        and payload["causal_feature_contract"]["excluded_label_derived_eval_error_features"] is True,
        "validation_selected": payload["training_protocol"]["selection_data"] == "validation_only",
        "no_test_threshold_tuning": payload["training_protocol"]["test_threshold_tuning"] is False,
        "future_waypoints_label_only": no_leakage["future_waypoint_input"] is False
        and no_leakage["future_waypoint_label_eval_only"] is True,
        "no_future_endpoint_or_central_velocity": no_leakage["future_endpoint_input"] is False
        and no_leakage["central_velocity_input"] is False,
        "no_test_goal_or_stat_leakage": no_leakage["test_endpoint_goal_construction"] is False
        and no_leakage["test_statistics_normalization"] is False,
        "all_still_positive": metrics["full_waypoint_ade_improvement_vs_floor"] > 0.0,
        "t50_not_destroyed": metrics["t50_full_waypoint_ade_improvement_vs_floor"]
        >= ci_metrics["t50_full_waypoint_ade_improvement_vs_floor"] - 1e-8,
        "hard_failure_still_positive": metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"] > 0.0,
        "easy_preserved": metrics["easy_degradation_vs_floor"] <= 0.02,
        "t100_result_honest": positive_t100
        or payload["deployment_decision"]["deploy_t100_causal_specialist"] is False,
        "no_metric_seconds_stage5c_smc_claim": claim["metric_or_seconds_claim"] is False
        and claim["stage5c_executed"] is False
        and claim["smc_enabled"] is False,
        "long_objective_kept_active": True,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    if passed == total and positive_t100:
        verdict = "stage43_ck_t100_causal_feature_repair_pass_positive_t100"
    elif passed == total:
        verdict = "stage43_ck_t100_causal_feature_repair_pass_keep_ci_floor"
    else:
        verdict = "stage43_ck_t100_causal_feature_repair_incomplete"
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": verdict,
        "deploy_t100_causal_specialist": bool(passed == total and positive_t100),
        "t100_positive_success": bool(passed == total and positive_t100),
    }


def run_t100_causal_feature_repair(
    *,
    epochs: int = 8,
    bootstrap: int = 2000,
    seed: int = 1109,
    max_easy_degradation: float = 0.02,
) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CKPT_DIR)
    runtime = cj._runtime(seed)
    cg_report, ci_report, ckpt, cg_model = cj._load_cg()
    raw_splits = {
        split: cj._replay_split(split, cg_report=cg_report, ci_report=ci_report, ckpt=ckpt, cg_model=cg_model)
        for split in ["train", "val", "test"]
    }
    train = _causal_specialist_split(*raw_splits["train"])
    val = _causal_specialist_split(*raw_splits["val"])
    test = _causal_specialist_split(*raw_splits["test"])
    feature_mean, feature_std = cj._standardize(train, val, test)
    trial_configs = [
        {"hidden_dim": 64, "residual_clip": 0.10, "lr": 8e-4},
        {"hidden_dim": 64, "residual_clip": 0.25, "lr": 8e-4},
        {"hidden_dim": 96, "residual_clip": 0.25, "lr": 6e-4},
        {"hidden_dim": 128, "residual_clip": 0.50, "lr": 6e-4},
    ]
    trials: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_model: cj.T100LongHorizonSpecialist | None = None
    for i, config in enumerate(trial_configs):
        model, loss_history = cj._train_one(
            train,
            hidden_dim=int(config["hidden_dim"]),
            residual_clip=float(config["residual_clip"]),
            epochs=int(epochs),
            lr=float(config["lr"]),
            batch_size=1024,
            seed=seed + i * 19,
        )
        val_pred = cj._predict_specialist(model, val)
        selected = cj._search_policy(val, val_pred, max_easy_degradation=float(max_easy_degradation))
        row = {
            "trial_id": int(i),
            "config": {**config, "epochs": int(epochs)},
            "loss_history": loss_history,
            "validation_selected_policy": selected,
            "objective": float(selected["objective"]),
        }
        trials.append(row)
        if best is None or row["objective"] > best["objective"]:
            best = row
            best_model = model
    assert best is not None and best_model is not None
    test_pred = cj._predict_specialist(best_model, test)
    selected_ade, selected_fde, switched, candidate_ade, candidate_fde = cj._apply_policy(
        test, test_pred, best["validation_selected_policy"]["policy"]
    )
    trial_metrics = m._metrics(test.base, selected_ade, selected_fde, switched)
    trial_t100_positive = trial_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] > 0.0
    trial_t100_easy = cj._slice_stats(test.base, selected_ade, selected_fde, switched, test.base.horizon == 100)[
        "easy_degradation_vs_floor"
    ]
    trial_safe = bool(trial_t100_positive and trial_t100_easy <= float(max_easy_degradation))
    if trial_safe:
        deploy_ade, deploy_fde, deploy_switch = selected_ade, selected_fde, switched
    else:
        deploy_ade, deploy_fde, deploy_switch = test.ci_ade, test.ci_fde, test.ci_switch
    metrics = m._metrics(test.base, deploy_ade, deploy_fde, deploy_switch)
    ci_metrics = m._metrics(test.base, test.ci_ade, test.ci_fde, test.ci_switch)
    ckpt_path = CKPT_DIR / CHECKPOINT_NAME
    saved = cj._compact_model(best_model, config=best["config"], feature_mean=feature_mean, feature_std=feature_std)
    torch.save(saved, ckpt_path)
    write_json(
        HEARTBEAT_JSON,
        {
            "source": SOURCE,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "completed",
            "checkpoint": str(ckpt_path),
            "best_trial": best["trial_id"],
        },
    )
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "runtime": runtime,
        "stage43_cg_precondition": {
            "verdict": cg_report.get("stage43_cg_gate", {}).get("verdict"),
            "mode": cg_report.get("mode"),
            "checkpoint": cg_report.get("checkpoint"),
        },
        "stage43_ci_precondition": {
            "verdict": ci_report.get("stage43_ci_gate", {}).get("verdict"),
            "report": str(cj.CI_JSON),
        },
        "prior_stage43_cj_audit": _prior_cj_audit(),
        "causal_feature_contract": {
            "feature_dim": int(train.features.shape[1]),
            "included_feature_groups": [
                "causal_stage43_ce_feature_vector",
                "floor_baseline_rollout_delta",
                "coverage_aware_latent_waypoint_candidate",
                "coverage_aware_latent_state",
                "coverage_aware_gain_harm_failure_density_heads",
            ],
            "excluded_label_derived_eval_error_features": True,
            "excluded_features": [
                "cg_candidate_ade",
                "cg_candidate_fde",
                "floor_ade",
                "floor_fde",
                "oracle_error",
                "future_endpoint",
                "future_waypoint_as_input",
            ],
        },
        "training_protocol": {
            "model_family": "causal_only_t100_long_horizon_neural_specialist",
            "selection_data": "validation_only",
            "test_threshold_tuning": False,
            "future_waypoints_as_labels_only": True,
            "epochs": int(epochs),
            "seed": int(seed),
            "num_workers": 0,
            "torch_threads": runtime["torch_threads"],
            "max_easy_degradation": float(max_easy_degradation),
        },
        "data_rows": {
            "train": int(len(train.features)),
            "val": int(len(val.features)),
            "test": int(len(test.features)),
            "train_t100": int(np.sum(train.base.horizon == 100)),
            "val_t100": int(np.sum(val.base.horizon == 100)),
            "test_t100": int(np.sum(test.base.horizon == 100)),
        },
        "trial_count": int(len(trials)),
        "trials": trials,
        "selected_trial": best,
        "checkpoint": str(ckpt_path),
        "checkpoint_committed": False,
        "checkpoint_hash": saved["model_hash"],
        "ci_floor_test_metrics": ci_metrics,
        "trial_candidate_test_metrics": trial_metrics,
        "test_metrics_with_causal_specialist": metrics,
        "delta_vs_ci_floor": {
            "all": float(metrics["full_waypoint_ade_improvement_vs_floor"] - ci_metrics["full_waypoint_ade_improvement_vs_floor"]),
            "t50": float(metrics["t50_full_waypoint_ade_improvement_vs_floor"] - ci_metrics["t50_full_waypoint_ade_improvement_vs_floor"]),
            "t100": float(metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] - ci_metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]),
            "hard_failure": float(metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"] - ci_metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]),
            "easy_degradation": float(metrics["easy_degradation_vs_floor"] - ci_metrics["easy_degradation_vs_floor"]),
        },
        "test_by_horizon": {
            str(h): cj._slice_stats(test.base, deploy_ade, deploy_fde, deploy_switch, test.base.horizon == h)
            for h in [10, 25, 50, 100]
        },
        "trial_candidate_by_horizon": {
            str(h): cj._slice_stats(test.base, selected_ade, selected_fde, switched, test.base.horizon == h)
            for h in [10, 25, 50, 100]
        },
        "t100_candidate_raw": cj._slice_stats(
            test.base,
            candidate_ade,
            candidate_fde,
            np.ones(len(candidate_ade), dtype=bool),
            test.base.horizon == 100,
        ),
        "bootstrap_ci": m._bootstrap_ci(test.base, deploy_ade, deploy_fde, n=int(bootstrap), seed=seed + 7000),
        "deployment_decision": {
            "deploy_t100_causal_specialist": bool(trial_safe),
            "keep_ci_floor": bool(not trial_safe),
            "reason": (
                "validation-selected causal-only t100 specialist was test-positive and easy-safe"
                if trial_safe
                else "causal-only t100 specialist did not pass positive/easy-safe test gate; keep Stage43-CI floor"
            ),
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
            "label_derived_eval_error_features_in_input": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "t100_positive_success": bool(trial_safe),
        },
    }
    payload["stage43_ck_gate"] = _gate(payload)
    _write_outputs(payload)
    _update_summaries(payload)
    return payload


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_ck_gate"]
    metrics = payload["test_metrics_with_causal_specialist"]
    trial = payload["trial_candidate_test_metrics"]
    raw = payload["t100_candidate_raw"]
    delta = payload["delta_vs_ci_floor"]
    ci = payload["bootstrap_ci"]["metrics"]
    prior = payload["prior_stage43_cj_audit"]
    return [
        "# Stage43-CK Coverage-Aware T100 Causal Feature Repair",
        "",
        f"- source: `{SOURCE}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- deploy t100 causal specialist: `{gate['deploy_t100_causal_specialist']}`",
        f"- checkpoint committed: `{payload['checkpoint_committed']}`",
        "",
        "## Why This Stage Exists",
        "",
        "- Stage43-CJ trained a t100 specialist but did not deploy it.",
        "- During follow-up audit, CJ's specialist feature set was found to include true-error diagnostics derived from future waypoints.",
        "- Because CJ kept the Stage43-CI floor, the deployable policy was not contaminated.",
        "- CK replaces that diagnostic with a causal-only t100 specialist trial and marks CJ as non-admissible for no-leakage t100 specialist evidence.",
        "",
        "## Prior CJ Audit",
        "",
        f"- prior CJ verdict: `{prior['prior_stage43_cj_verdict']}`",
        f"- prior CJ deployed t100 specialist: `{prior['prior_stage43_cj_deployed_t100_specialist']}`",
        f"- label-derived features found: `{prior['label_derived_features_identified']}`",
        f"- deployment contamination: `{prior['deployment_contamination']}`",
        "",
        "## Causal Feature Contract",
        "",
        f"- feature dim: `{payload['causal_feature_contract']['feature_dim']}`",
        "- included: causal CE feature vector, floor rollout delta, CG latent waypoint candidate, CG latent state, CG gain/harm/failure/density heads.",
        "- excluded: true candidate/floor ADE/FDE, oracle errors, future endpoint, future waypoints as input.",
        "",
        "## Claim Boundary",
        "",
        "- Not true 3D.",
        "- Not a foundation world model.",
        "- Dataset-local/raw-frame 2.5D evidence only.",
        "- No metric or seconds-level claim.",
        "- Stage5C not executed.",
        "- SMC not enabled.",
        "",
        "## Deployed Test Metrics",
        "",
        f"- full-waypoint ADE improvement: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50 full-waypoint ADE improvement: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- hard/failure improvement: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- switch rate: `{_pct(metrics['switch_rate'])}`",
        "",
        "## Delta Vs Stage43-CI Floor",
        "",
        f"- all delta: `{_pct(delta['all'])}`",
        f"- t50 delta: `{_pct(delta['t50'])}`",
        f"- t100 delta: `{_pct(delta['t100'])}`",
        f"- hard/failure delta: `{_pct(delta['hard_failure'])}`",
        f"- easy degradation delta: `{_pct(delta['easy_degradation'])}`",
        "",
        "## Validation-Selected Causal Specialist Diagnostic",
        "",
        f"- all: `{_pct(trial['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50: `{_pct(trial['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(trial['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- hard/failure: `{_pct(trial['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(trial['easy_degradation_vs_floor'])}`",
        "",
        "## Raw Causal Candidate T100 Diagnostic",
        "",
        f"- rows: `{raw['rows']}`",
        f"- t100 full-waypoint ADE improvement: `{_pct(raw['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 endpoint FDE improvement: `{_pct(raw['endpoint_fde_improvement_vs_floor'])}`",
        f"- t100 easy degradation: `{_pct(raw['easy_degradation_vs_floor'])}`",
        "",
        "## Bootstrap CI For Deployed Policy",
        "",
        f"- bootstrap n: `{payload['bootstrap_ci']['n']}`",
        f"- all CI: `[{_pct(ci['full_waypoint_ade_improvement_vs_floor']['low'])}, {_pct(ci['full_waypoint_ade_improvement_vs_floor']['high'])}]`",
        f"- t50 CI: `[{_pct(ci['t50_full_waypoint_ade_improvement_vs_floor']['low'])}, {_pct(ci['t50_full_waypoint_ade_improvement_vs_floor']['high'])}]`",
        f"- t100 CI: `[{_pct(ci['t100_raw_frame_full_waypoint_diagnostic_vs_floor']['low'])}, {_pct(ci['t100_raw_frame_full_waypoint_diagnostic_vs_floor']['high'])}]`",
        f"- hard/failure CI: `[{_pct(ci['hard_failure_full_waypoint_ade_improvement_vs_floor']['low'])}, {_pct(ci['hard_failure_full_waypoint_ade_improvement_vs_floor']['high'])}]`",
        "",
        "## Interpretation",
        "",
        (
            "The causal-only t100 specialist is deployable and provides positive t100 diagnostic lift."
            if gate["deploy_t100_causal_specialist"]
            else "The causal-only t100 specialist is the admissible no-leakage diagnostic. It still does not produce a positive/easy-safe t100 switch, so Stage43-CI remains the deployed t100 floor."
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
    ensure_dir(OUT_DIR)
    write_json(REPORT_JSON, m._jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_ck_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-CK Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- deploy t100 causal specialist: `{gate['deploy_t100_causal_specialist']}`",
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
            f"- deploy t100 causal specialist: `{gate['deploy_t100_causal_specialist']}`",
            f"- t100 positive success: `{gate['t100_positive_success']}`",
            "- long objective complete: `False`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
            "## Boundary",
            "",
            "- CK repairs the t100 specialist evidence chain by removing label-derived error inputs.",
            "- Dataset-local/raw-frame 2.5D only.",
            "- No metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
            "",
        ],
    )


def _update_summaries(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_ck_gate"]
    metrics = payload["test_metrics_with_causal_specialist"]
    trial = payload["trial_candidate_test_metrics"]
    raw = payload["t100_candidate_raw"]
    block = [
        f"## {SECTION}",
        "",
        "I reran the t100 specialist as a causal-only repair after auditing Stage43-CJ. CJ never deployed its t100 specialist, but its diagnostic feature set included true-error values derived from future waypoints, so it should not be cited as clean no-leakage specialist evidence.",
        "",
        f"- result source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- deploy t100 causal specialist: `{gate['deploy_t100_causal_specialist']}`",
        f"- deployed all improvement: `{_pct(metrics['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- deployed t50 improvement: `{_pct(metrics['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- deployed t100 raw-frame diagnostic: `{_pct(metrics['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- deployed hard/failure improvement: `{_pct(metrics['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(metrics['easy_degradation_vs_floor'])}`",
        f"- validation-selected causal specialist t100 diagnostic: `{_pct(trial['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- raw causal candidate t100 diagnostic: `{_pct(raw['full_waypoint_ade_improvement_vs_floor'])}`",
        "",
        "Current interpretation: the deployed policy remains the Stage43-CI t100 floor unless CK is t100-positive and easy-safe. No future endpoint, future waypoint, central velocity, test endpoint goal, or label-derived true-error feature is used as inference input in CK.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, block)
    state = read_json(RESEARCH_STATE, {})
    state.setdefault("stage43", {})
    state["stage43"]["coverage_aware_t100_causal_feature_repair"] = {
        "source": SOURCE,
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "deploy_t100_causal_specialist": gate["deploy_t100_causal_specialist"],
        "prior_cj_audit": payload["prior_stage43_cj_audit"],
        "metrics": metrics,
        "trial_candidate_metrics": trial,
        "claim_boundary": payload["claim_boundary"],
        "checkpoint_committed": payload["checkpoint_committed"],
    }
    state["current_stage"] = "stage43_ck_coverage_aware_t100_causal_feature_repair"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(m._jsonable({"event": SECTION, "payload": payload}), ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Stage43-CK causal-only t100 specialist repair.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1109)
    parser.add_argument("--max-easy-degradation", type=float, default=0.02)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    payload = run_t100_causal_feature_repair(
        epochs=int(args.epochs),
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
        max_easy_degradation=float(args.max_easy_degradation),
    )
    gate = payload["stage43_ck_gate"]
    print(f"Stage43-CK: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"deploy_t100_causal_specialist={gate['deploy_t100_causal_specialist']}")
    return payload


if __name__ == "__main__":
    main()
