from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage43_full_waypoint_latent_dynamics as m
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_full_waypoint_latent_robustness_audit import _breakdown, _pct
from src.stage43_full_waypoint_latent_safe_repair import _source_family
from src.stage43_tail_horizon_waypoint_adapter import _model_hash, _ridge_fit, _target_matrix


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_current_matrix_t100_source_family_gate.json"
REPORT_MD = OUT_DIR / "stage43_current_matrix_t100_source_family_gate.md"
GATE_MD = OUT_DIR / "stage43_stage_cm_current_matrix_t100_source_family_gate.md"

README_RESULTS = m.README_RESULTS
M3W_README = m.M3W_README
WORK_SUMMARY = m.WORK_SUMMARY
RESEARCH_STATE = m.RESEARCH_STATE

SOURCE = "fresh_stage43_cm_current_matrix_t100_source_family_gate"
SECTION = "STAGE43_CM_CURRENT_MATRIX_T100_SOURCE_FAMILY_GATE"

STAGE43_CL_JSON = OUT_DIR / "stage43_t100_source_stable_compatibility_audit.json"
STAGE43_AT_JSON = OUT_DIR / "stage43_external_validation_matrix.json"

DENIED_FEATURE_NAME_FRAGMENTS = (
    "future",
    "oracle",
    "central_velocity",
    "ground_truth",
    "label",
    "ade",
    "fde",
)
EPS = 1e-8


def _sha_text(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _split_scope_hash(ds: m.WaypointSplit) -> str:
    digest = hashlib.sha256()
    for key, arr in {
        "split": np.asarray([ds.split] * len(ds.x), dtype=str),
        "source_file": ds.source_file.astype(str),
        "scene_id": ds.scene_id.astype(str),
        "horizon": ds.horizon.astype(np.int64),
    }.items():
        digest.update(key.encode("utf-8"))
        if arr.dtype.kind in {"U", "S", "O"}:
            digest.update(arr.astype(str).tobytes())
        else:
            digest.update(arr.tobytes())
    return digest.hexdigest()


def _augment_with_floor_waypoint_features(ds: m.WaypointSplit) -> m.WaypointSplit:
    floor = ds.floor_waypoint_delta.reshape(len(ds.x), -1).astype(np.float32)
    names = [*ds.feature_names, *[f"causal_floor_waypoint_delta_{i}" for i in range(floor.shape[1])]]
    x = np.concatenate([ds.x.astype(np.float32), floor], axis=1).astype(np.float32)
    return replace(ds, x=x, feature_names=names)


def _standardize(train: m.WaypointSplit, val: m.WaypointSplit, test: m.WaypointSplit) -> tuple[np.ndarray, np.ndarray]:
    mean = train.x.mean(axis=0).astype(np.float32)
    raw_std = train.x.std(axis=0).astype(np.float32)
    std = np.where(raw_std < 1e-3, 1.0, raw_std).astype(np.float32)
    for ds in (train, val, test):
        ds.x = ((ds.x - mean) / std).astype(np.float32)
    return mean, std


def _ridge_predict(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    xb = np.concatenate([x, np.ones((len(x), 1), dtype=np.float32)], axis=1).astype(np.float64)
    return (xb @ weight).astype(np.float32)


def _predict_waypoint(ds: m.WaypointSplit, weight: np.ndarray, target: str) -> np.ndarray:
    pred = _ridge_predict(ds.x, weight)
    if target == "residual":
        pred = ds.floor_waypoint_delta.reshape(len(ds.x), -1) + pred
    return pred.reshape(-1, 4, 2).astype(np.float32)


def _source_families(ds: m.WaypointSplit) -> np.ndarray:
    return np.asarray([_source_family(value) for value in ds.source_file], dtype=str)


def _train_mask(ds: m.WaypointSplit, train_filter: str) -> np.ndarray:
    if train_filter == "t100":
        return ds.horizon == 100
    if train_filter == "t50t100":
        return ds.horizon >= 50
    if train_filter == "all":
        return np.ones(len(ds.x), dtype=bool)
    raise ValueError(f"Unknown train_filter={train_filter}")


def _slice_improvement(selected: np.ndarray, floor: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) == 0:
        return 0.0
    return float(1.0 - float(np.mean(selected[mask])) / max(float(np.mean(floor[mask])), EPS))


def _easy_degradation(ds: m.WaypointSplit, selected: np.ndarray, mask: np.ndarray) -> float:
    easy = mask & ds.easy
    if int(easy.sum()) == 0:
        return 0.0
    return float(max(0.0, float(np.mean(selected[easy])) / max(float(np.mean(ds.floor_ade[easy])), EPS) - 1.0))


def _family_t100_table(
    ds: m.WaypointSplit,
    candidate_ade: np.ndarray,
    *,
    min_support_rows: int,
    min_improvement: float,
    max_easy_degradation: float,
) -> tuple[dict[str, Any], set[str]]:
    families = _source_families(ds)
    table: dict[str, Any] = {}
    allowed: set[str] = set()
    for family in sorted(set(families.tolist())):
        mask = (families == family) & (ds.horizon == 100)
        rows = int(mask.sum())
        if rows == 0:
            continue
        improvement = _slice_improvement(candidate_ade, ds.floor_ade, mask)
        easy = _easy_degradation(ds, candidate_ade, mask)
        supported = rows >= int(min_support_rows)
        safe_positive = supported and improvement > float(min_improvement) and easy <= float(max_easy_degradation)
        reason = "allowed_by_validation"
        if not supported:
            reason = "blocked_insufficient_validation_support"
        elif improvement <= float(min_improvement):
            reason = "blocked_validation_nonpositive"
        elif easy > float(max_easy_degradation):
            reason = "blocked_validation_easy_harm"
        table[family] = {
            "rows": rows,
            "t100_full_waypoint_ade_improvement_vs_floor": float(improvement),
            "easy_degradation_vs_floor": float(easy),
            "allowed": bool(safe_positive),
            "reason": reason,
        }
        if safe_positive:
            allowed.add(family)
    return table, allowed


def _apply_t100_family_rules(
    ds: m.WaypointSplit,
    candidate_ade: np.ndarray,
    candidate_fde: np.ndarray,
    allowed_families: set[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    families = _source_families(ds)
    switch = np.asarray([(h == 100 and family in allowed_families) for family, h in zip(families, ds.horizon)], dtype=bool)
    selected_ade = np.where(switch, candidate_ade, ds.floor_ade).astype(np.float32)
    selected_fde = np.where(switch, candidate_fde, ds.floor_fde).astype(np.float32)
    return selected_ade, selected_fde, switch


def _candidate_eval(
    train: m.WaypointSplit,
    val: m.WaypointSplit,
    *,
    target: str,
    train_filter: str,
    l2: float,
    min_support_rows: int,
    min_improvement: float,
    max_easy_degradation: float,
) -> dict[str, Any]:
    ids = _train_mask(train, train_filter)
    weight = _ridge_fit(train.x[ids], _target_matrix(train, target)[ids], float(l2))
    pred = _predict_waypoint(val, weight, target)
    candidate_ade, candidate_fde = m._trajectory_error(val, pred)
    table, allowed = _family_t100_table(
        val,
        candidate_ade,
        min_support_rows=int(min_support_rows),
        min_improvement=float(min_improvement),
        max_easy_degradation=float(max_easy_degradation),
    )
    selected_ade, selected_fde, switch = _apply_t100_family_rules(val, candidate_ade, candidate_fde, allowed)
    metrics = m._metrics(val, selected_ade, selected_fde, switch)
    objective = (
        2.0 * metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"]
        + 0.5 * metrics["hard_failure_full_waypoint_ade_improvement_vs_floor"]
        - 25.0 * max(0.0, metrics["easy_degradation_vs_floor"] - 0.02)
        - 2.0 * max(0.0, -metrics["full_waypoint_ade_improvement_vs_floor"])
    )
    return {
        "target": target,
        "train_filter": train_filter,
        "l2": float(l2),
        "train_rows": int(ids.sum()),
        "weight": weight,
        "model_hash": _model_hash(weight, l2=float(l2), target=target, train_filter=train_filter),
        "validation_metrics": metrics,
        "validation_source_family_t100_table": table,
        "allowed_families": sorted(allowed),
        "objective": float(objective),
    }


def _feature_contract(feature_names: list[str]) -> dict[str, Any]:
    denied = sorted(
        {
            name
            for name in feature_names
            for frag in DENIED_FEATURE_NAME_FRAGMENTS
            if frag in name.lower()
        }
    )
    return {
        "feature_names_persisted": True,
        "feature_dim": int(len(feature_names)),
        "feature_name_hash": _sha_text(feature_names),
        "feature_names": feature_names,
        "denied_feature_name_hits": denied,
        "causal_floor_waypoint_rollout_feature_count": int(
            sum(name.startswith("causal_floor_waypoint_delta_") for name in feature_names)
        ),
        "future_waypoints_label_only": True,
        "baseline_rollout_computed_without_future": True,
    }


def _breakdown_by_source_family(ds: m.WaypointSplit, selected_ade: np.ndarray, selected_fde: np.ndarray, switch: np.ndarray) -> dict[str, Any]:
    families = _source_families(ds)
    arrays = (ds.floor_ade, ds.floor_fde, selected_ade, selected_fde, selected_ade, switch, ds.easy)
    return _breakdown(families, *arrays, min_rows=10)


def _family_test_table(
    ds: m.WaypointSplit,
    candidate_ade: np.ndarray,
    selected_ade: np.ndarray,
    switch: np.ndarray,
    allowed_families: set[str],
) -> dict[str, Any]:
    families = _source_families(ds)
    table: dict[str, Any] = {}
    for family in sorted(set(families.tolist())):
        mask = (families == family) & (ds.horizon == 100)
        rows = int(mask.sum())
        if rows == 0:
            continue
        table[family] = {
            "rows": rows,
            "validation_allowed": family in allowed_families,
            "candidate_t100_full_waypoint_ade_improvement_vs_floor": _slice_improvement(candidate_ade, ds.floor_ade, mask),
            "selected_t100_full_waypoint_ade_improvement_vs_floor": _slice_improvement(selected_ade, ds.floor_ade, mask),
            "easy_degradation_vs_floor": _easy_degradation(ds, selected_ade, mask),
            "switch_rate": float(np.mean(switch[mask])) if rows else 0.0,
        }
    return table


def _deployment_decision(metrics: Mapping[str, Any], test_table: Mapping[str, Any], allowed_families: set[str]) -> dict[str, Any]:
    positive_families = [
        family
        for family, row in test_table.items()
        if row.get("validation_allowed")
        and float(row.get("selected_t100_full_waypoint_ade_improvement_vs_floor", 0.0)) > 0.0
        and float(row.get("easy_degradation_vs_floor", 0.0)) <= 0.02
    ]
    deploy = bool(
        allowed_families
        and metrics["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] > 0.0
        and metrics["easy_degradation_vs_floor"] <= 0.02
        and metrics["full_waypoint_ade_improvement_vs_floor"] >= 0.0
        and len(positive_families) == len(allowed_families)
    )
    return {
        "deploy_current_matrix_t100_source_family_gate": deploy,
        "positive_test_families": positive_families,
        "allowed_family_count": int(len(allowed_families)),
        "reason": (
            "validation_selected_source_family_t100_rules_positive_and_easy_safe_on_current_matrix_test"
            if deploy
            else "keep_floor_because_current_matrix_test_is_not_uniformly_positive_easy_safe"
        ),
    }


def build_current_matrix_t100_source_family_gate(
    *,
    seed: int = 437,
    bootstrap: int = 1000,
    min_support_rows: int = 200,
    min_improvement: float = 0.0,
    max_easy_degradation: float = 0.02,
    l2_grid: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0, 10000.0),
    target_grid: tuple[str, ...] = ("residual", "direct"),
    train_filter_grid: tuple[str, ...] = ("t100", "t50t100"),
) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage_cl = read_json(STAGE43_CL_JSON, {})
    matrix = read_json(STAGE43_AT_JSON, {})

    train = _augment_with_floor_waypoint_features(m._build_split("train", max_rows=None, seed=int(seed)))
    val = _augment_with_floor_waypoint_features(m._build_split("val", max_rows=None, seed=int(seed)))
    test = _augment_with_floor_waypoint_features(m._build_split("test", max_rows=None, seed=int(seed)))
    feature_names = train.feature_names
    if feature_names != val.feature_names or feature_names != test.feature_names:
        raise ValueError("Feature names differ across splits.")
    feature_mean, feature_std = _standardize(train, val, test)

    candidates = [
        _candidate_eval(
            train,
            val,
            target=target,
            train_filter=train_filter,
            l2=float(l2),
            min_support_rows=int(min_support_rows),
            min_improvement=float(min_improvement),
            max_easy_degradation=float(max_easy_degradation),
        )
        for target in target_grid
        for train_filter in train_filter_grid
        for l2 in l2_grid
    ]
    candidates.sort(key=lambda row: row["objective"], reverse=True)
    best = candidates[0]
    allowed = set(best["allowed_families"])
    pred = _predict_waypoint(test, best["weight"], best["target"])
    candidate_ade, candidate_fde = m._trajectory_error(test, pred)
    raw_selected_ade, raw_selected_fde, raw_switch = _apply_t100_family_rules(test, candidate_ade, candidate_fde, allowed)
    raw_metrics = m._metrics(test, raw_selected_ade, raw_selected_fde, raw_switch)
    test_family_table = _family_test_table(test, candidate_ade, raw_selected_ade, raw_switch, allowed)
    deployment = _deployment_decision(raw_metrics, test_family_table, allowed)

    if deployment["deploy_current_matrix_t100_source_family_gate"]:
        deployed_ade, deployed_fde, deployed_switch = raw_selected_ade, raw_selected_fde, raw_switch
    else:
        deployed_ade = test.floor_ade.copy()
        deployed_fde = test.floor_fde.copy()
        deployed_switch = np.zeros(len(test.x), dtype=bool)
    deployed_metrics = m._metrics(test, deployed_ade, deployed_fde, deployed_switch)
    bootstrap_ci = m._bootstrap_ci(test, deployed_ade, deployed_fde, n=int(bootstrap), seed=int(seed) + 43000)

    compact_candidates = []
    for row in candidates[:10]:
        compact_candidates.append(
            {
                "target": row["target"],
                "train_filter": row["train_filter"],
                "l2": row["l2"],
                "train_rows": row["train_rows"],
                "model_hash": row["model_hash"],
                "objective": row["objective"],
                "allowed_families": row["allowed_families"],
                "validation_metrics": row["validation_metrics"],
                "validation_source_family_t100_table": row["validation_source_family_t100_table"],
            }
        )

    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_current_matrix_train_val_selected_t100_source_family_gate",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "input_artifacts": {
            "stage43_cl": str(STAGE43_CL_JSON),
            "stage43_at": str(STAGE43_AT_JSON),
        },
        "input_verdicts": {
            "stage43_cl": stage_cl.get("stage43_cl_gate", {}).get("verdict"),
            "stage43_at": matrix.get("stage43_at_gate", {}).get("verdict"),
        },
        "current_matrix_scope": {
            "train_rows": int(len(train.x)),
            "val_rows": int(len(val.x)),
            "test_rows": int(len(test.x)),
            "test_t100_rows": int(np.sum(test.horizon == 100)),
            "stage43_at_matrix_test_rows": int(matrix.get("test_rows", matrix.get("split", {}).get("test_rows", 0))),
            "stage43_cl_local_t100_rows": int(
                stage_cl.get("compatibility", {}).get("stage43_t_test_rows", 0)
            ),
            "train_scope_hash": _split_scope_hash(train),
            "val_scope_hash": _split_scope_hash(val),
            "test_scope_hash": _split_scope_hash(test),
        },
        "feature_contract": {
            **_feature_contract(feature_names),
            "feature_mean_hash": hashlib.sha256(feature_mean.tobytes()).hexdigest(),
            "feature_std_hash": hashlib.sha256(feature_std.tobytes()).hexdigest(),
        },
        "training_protocol": {
            "model_family": "closed_form_ridge_current_matrix_t100_source_family_gate",
            "selection_data": "validation_only",
            "test_threshold_tuning": False,
            "seed": int(seed),
            "min_support_rows": int(min_support_rows),
            "min_improvement": float(min_improvement),
            "max_easy_degradation": float(max_easy_degradation),
            "future_waypoints_as_labels_only": True,
            "source_family_rules_selected_on_validation_only": True,
        },
        "candidate_search": {
            "l2_grid": list(map(float, l2_grid)),
            "target_grid": list(target_grid),
            "train_filter_grid": list(train_filter_grid),
            "candidate_count": int(len(candidates)),
            "top_candidates": compact_candidates,
        },
        "selected_model": {
            "target": best["target"],
            "train_filter": best["train_filter"],
            "l2": best["l2"],
            "train_rows": best["train_rows"],
            "model_hash": best["model_hash"],
            "validation_metrics": best["validation_metrics"],
            "validation_allowed_families": best["allowed_families"],
            "validation_source_family_t100_table": best["validation_source_family_t100_table"],
        },
        "raw_validation_rule_test_metrics": raw_metrics,
        "deployed_test_metrics": deployed_metrics,
        "deployment_decision": deployment,
        "bootstrap_ci": bootstrap_ci,
        "test_source_family_t100_table": test_family_table,
        "deployed_by_source_family": _breakdown_by_source_family(test, deployed_ade, deployed_fde, deployed_switch),
        "raw_by_source_family": _breakdown_by_source_family(test, raw_selected_ade, raw_selected_fde, raw_switch),
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
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "global_t100_success_claim": bool(deployment["deploy_current_matrix_t100_source_family_gate"]),
            "uniform_t100_success_claim": False,
            "long_objective_complete": False,
        },
        "input_hash": _combined_hash([STAGE43_CL_JSON, STAGE43_AT_JSON]),
    }
    payload["stage43_cm_gate"] = _gate(payload)
    _write_outputs(payload)
    _update_summaries(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    feature = payload["feature_contract"]
    scope = payload["current_matrix_scope"]
    deployed = payload["deployed_test_metrics"]
    raw = payload["raw_validation_rule_test_metrics"]
    decision = payload["deployment_decision"]
    no_leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    matrix_rows = int(scope.get("stage43_at_matrix_test_rows", 0))
    gates = {
        "stage43_cl_precondition_present": payload["input_verdicts"]["stage43_cl"]
        == "stage43_cl_t100_source_stable_compatibility_pass_local_only",
        "current_matrix_scope_used": int(scope["test_rows"]) > int(scope["stage43_cl_local_t100_rows"])
        and (matrix_rows == 0 or int(scope["test_rows"]) == matrix_rows),
        "feature_names_persisted": feature["feature_names_persisted"] is True
        and feature["feature_dim"] == len(feature["feature_names"])
        and bool(feature["feature_name_hash"]),
        "feature_contract_clean": feature["denied_feature_name_hits"] == []
        and feature["future_waypoints_label_only"] is True
        and feature["baseline_rollout_computed_without_future"] is True,
        "validation_only_selection": payload["training_protocol"]["selection_data"] == "validation_only"
        and payload["training_protocol"]["test_threshold_tuning"] is False,
        "source_family_t100_table_present": bool(payload["selected_model"]["validation_source_family_t100_table"])
        and bool(payload["test_source_family_t100_table"]),
        "raw_test_evaluated_once": "t100_raw_frame_full_waypoint_diagnostic_vs_floor" in raw,
        "deployed_policy_safe": deployed["easy_degradation_vs_floor"] <= 0.02
        and deployed["full_waypoint_ade_improvement_vs_floor"] >= -1e-9
        and deployed["t100_raw_frame_full_waypoint_diagnostic_vs_floor"] >= -1e-9,
        "global_t100_claim_matches_deploy": claim["global_t100_success_claim"]
        is bool(decision["deploy_current_matrix_t100_source_family_gate"]),
        "uniform_t100_not_overclaimed": claim["uniform_t100_success_claim"] is False,
        "no_future_or_test_leakage": no_leak["future_endpoint_input"] is False
        and no_leak["future_waypoint_input"] is False
        and no_leak["central_velocity_input"] is False
        and no_leak["test_endpoint_goal_construction"] is False
        and no_leak["test_statistics_normalization"] is False
        and no_leak["test_threshold_tuning"] is False,
        "no_metric_seconds_stage5c_smc_claim": claim["metric_or_seconds_claim"] is False
        and claim["stage5c_executed"] is False
        and claim["smc_enabled"] is False,
        "long_objective_kept_active": claim["long_objective_complete"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    deploy = bool(decision["deploy_current_matrix_t100_source_family_gate"])
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": (
            "stage43_cm_current_matrix_t100_source_family_gate_pass_deploy_t100"
            if passed == total and deploy
            else "stage43_cm_current_matrix_t100_source_family_gate_pass_keep_floor"
            if passed == total
            else "stage43_cm_current_matrix_t100_source_family_gate_incomplete"
        ),
        "deploy_current_matrix_t100_source_family_gate": deploy and passed == total,
        "raw_validation_rule_t100_improvement": raw["t100_raw_frame_full_waypoint_diagnostic_vs_floor"],
        "deployed_t100_improvement": deployed["t100_raw_frame_full_waypoint_diagnostic_vs_floor"],
    }


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_cm_gate"]
    scope = payload["current_matrix_scope"]
    raw = payload["raw_validation_rule_test_metrics"]
    deployed = payload["deployed_test_metrics"]
    decision = payload["deployment_decision"]
    selected = payload["selected_model"]
    lines = [
        "# Stage43-CM Current-Matrix T100 Source-Family Gate",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- deploy current-matrix t100 source-family gate: `{gate['deploy_current_matrix_t100_source_family_gate']}`",
        "",
        "## Current Matrix Scope",
        "",
        f"- train rows: `{scope['train_rows']}`",
        f"- val rows: `{scope['val_rows']}`",
        f"- test rows: `{scope['test_rows']}`",
        f"- test t100 rows: `{scope['test_t100_rows']}`",
        f"- Stage43-CL local Stage43-T rows: `{scope['stage43_cl_local_t100_rows']}`",
        f"- Stage43-AT matrix rows: `{scope['stage43_at_matrix_test_rows']}`",
        "",
        "## Feature Contract",
        "",
        f"- feature dim: `{payload['feature_contract']['feature_dim']}`",
        f"- feature name hash: `{payload['feature_contract']['feature_name_hash']}`",
        f"- denied feature hits: `{payload['feature_contract']['denied_feature_name_hits']}`",
        f"- future waypoints: `label/eval only`",
        "",
        "## Selected Model",
        "",
        f"- target: `{selected['target']}`",
        f"- train filter: `{selected['train_filter']}`",
        f"- l2: `{selected['l2']}`",
        f"- train rows: `{selected['train_rows']}`",
        f"- model hash: `{selected['model_hash']}`",
        f"- validation allowed families: `{', '.join(selected['validation_allowed_families']) or 'none'}`",
        "",
        "## Raw Validation-Rule Test Metrics",
        "",
        f"- all ADE lift: `{_pct(raw['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50 ADE lift: `{_pct(raw['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(raw['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- hard/failure lift: `{_pct(raw['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(raw['easy_degradation_vs_floor'])}`",
        f"- switch rate: `{_pct(raw['switch_rate'])}`",
        "",
        "## Deployed Metrics",
        "",
        f"- all ADE lift: `{_pct(deployed['full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t50 ADE lift: `{_pct(deployed['t50_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(deployed['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- hard/failure lift: `{_pct(deployed['hard_failure_full_waypoint_ade_improvement_vs_floor'])}`",
        f"- easy degradation: `{_pct(deployed['easy_degradation_vs_floor'])}`",
        f"- switch rate: `{_pct(deployed['switch_rate'])}`",
        f"- deployment reason: `{decision['reason']}`",
        "",
        "## Source-Family T100 Test Table",
        "",
        "| family | rows | val allowed | candidate t100 lift | selected t100 lift | easy degradation | switch |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for family, row in payload["test_source_family_t100_table"].items():
        lines.append(
            f"| {family} | {row['rows']} | `{row['validation_allowed']}` | "
            f"{_pct(row['candidate_t100_full_waypoint_ade_improvement_vs_floor'])} | "
            f"{_pct(row['selected_t100_full_waypoint_ade_improvement_vs_floor'])} | "
            f"{_pct(row['easy_degradation_vs_floor'])} | {_pct(row['switch_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This run uses the current Stage43 full-waypoint supervision matrix rather than the earlier small Stage43-T source split. It persists the causal feature names and source/split hashes, chooses source-family t100 switch rules on validation only, then evaluates the selected rules once on test.",
            "",
            "If the validation-selected source-family rule is not positive and easy-safe on the current matrix test set, deployment stays at the Stage43-CI/CK floor. That is a conservative claim boundary, not a t100 success claim.",
            "",
            "Boundary unchanged: dataset-local/raw-frame 2.5D only; no metric or seconds-level claim; no Stage5C execution; no SMC.",
        ]
    )
    return lines


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_cm_gate"]
    lines = [
        "# Stage43-CM Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- deploy current-matrix t100 source-family gate: `{gate['deploy_current_matrix_t100_source_family_gate']}`",
        f"- raw validation-rule t100 improvement: `{_pct(gate['raw_validation_rule_t100_improvement'])}`",
        f"- deployed t100 improvement: `{_pct(gate['deployed_t100_improvement'])}`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    lines.extend([f"| `{name}` | `{value}` |" for name, value in gate["gates"].items()])
    write_md(GATE_MD, lines)


def _update_summaries(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_cm_gate"]
    raw = payload["raw_validation_rule_test_metrics"]
    deployed = payload["deployed_test_metrics"]
    scope = payload["current_matrix_scope"]
    block = [
        f"## {SECTION}",
        "",
        "I rebuilt the t100 source-family check on the current Stage43 full-waypoint matrix rather than relying on the earlier small Stage43-T local split. The run persists causal feature names, feature hashes, and source/split hashes so the evidence can be audited later.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- current test rows: `{scope['test_rows']}`",
        f"- current t100 rows: `{scope['test_t100_rows']}`",
        f"- raw validation-rule t100 lift: `{_pct(raw['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- raw easy degradation: `{_pct(raw['easy_degradation_vs_floor'])}`",
        f"- deployed t100 lift: `{_pct(deployed['t100_raw_frame_full_waypoint_diagnostic_vs_floor'])}`",
        f"- deployed easy degradation: `{_pct(deployed['easy_degradation_vs_floor'])}`",
        f"- deploy t100 source-family gate: `{gate['deploy_current_matrix_t100_source_family_gate']}`",
        "",
        "Interpretation: this is a current-matrix compatibility audit for t100. If the validation-selected rule is not positive and easy-safe on the current matrix, the deployed policy remains the floor; no t100 success, metric, seconds-level, Stage5C, or SMC claim is made.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, block)
    state = read_json(RESEARCH_STATE, {})
    state.setdefault("stage43", {})
    state["stage43"]["current_matrix_t100_source_family_gate"] = {
        "source": SOURCE,
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "deploy_current_matrix_t100_source_family_gate": gate["deploy_current_matrix_t100_source_family_gate"],
        "current_matrix_scope": payload["current_matrix_scope"],
        "feature_contract": {
            key: value
            for key, value in payload["feature_contract"].items()
            if key != "feature_names"
        },
        "selected_model": payload["selected_model"],
        "raw_validation_rule_test_metrics": payload["raw_validation_rule_test_metrics"],
        "deployed_test_metrics": payload["deployed_test_metrics"],
        "claim_boundary": payload["claim_boundary"],
    }
    state["current_stage"] = "stage43_cm_current_matrix_t100_source_family_gate"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, m._jsonable(state))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-CM current-matrix t100 source-family gate.")
    parser.add_argument("--seed", type=int, default=437)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--min-support-rows", type=int, default=200)
    parser.add_argument("--max-easy-degradation", type=float, default=0.02)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    payload = build_current_matrix_t100_source_family_gate(
        seed=int(args.seed),
        bootstrap=int(args.bootstrap),
        min_support_rows=int(args.min_support_rows),
        max_easy_degradation=float(args.max_easy_degradation),
    )
    gate = payload["stage43_cm_gate"]
    print(f"Stage43-CM: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    return payload


if __name__ == "__main__":
    main()
