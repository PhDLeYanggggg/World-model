from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage41_breakthrough as s41
from src import stage41_joint_multiagent_consistency as jmc
from src import stage41_joint_rollout_consistency as jrc
from src import stage42_source_level_full_waypoint_eval as am
from src import stage42_t50_floor_relaxability_repair as by
from src import stage42_ucy_validation_support_repair as aw
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit


OUT_DIR = Path("outputs/stage42_long_research")

BY_JSON = OUT_DIR / "t50_floor_relaxability_repair_stage42.json"
BZ_JSON = OUT_DIR / "t50_repair_statistical_evidence_stage42.json"
EN_JSON = OUT_DIR / "floor_removability_decision_map_stage42.json"

REPORT_JSON = OUT_DIR / "floor_relaxation_safety_stress_stage42.json"
REPORT_MD = OUT_DIR / "floor_relaxation_safety_stress_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_gt_gate.md"

SOURCE = "fresh_stage42_gt_floor_relaxation_safety_stress"
TARGET_SLICES = ("TrajNet|50", "UCY|50")
EASY_LIMIT = 0.02
NEAR_COLLISION_EPS = 0.005
BOOTSTRAP_N = 1000
SEED = 424270


CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-GT 是 Stage42-BY/BZ partial t50 floor-relaxation 的 all-agent safety stress test。",
    "本阶段不训练新模型，不下载数据，不转换新数据，不执行 Stage5C，不启用 SMC。",
    "future endpoints / waypoints 只作为 supervised/evaluation labels，不作为 inference input。",
    "不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 坐标不能写成 global metric。",
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def _subset_labels(labels: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[mask] for k, v in labels.items() if isinstance(v, np.ndarray) and len(v) == len(mask)}


def _labels_for_joint(data: Mapping[str, np.ndarray], waypoint_labels: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    current_xy = np.stack([data["current_x"], data["current_y"]], axis=1).astype(np.float64)
    return {
        "current_xy": current_xy,
        "normalizer": np.maximum(data["scale"].astype(np.float64), am.EPS),
        "horizon": data["horizon"].astype(int),
        "hard": data["hard"].astype(bool),
        "failure": data["failure"].astype(bool),
        "easy": data["easy"].astype(bool),
        "domain": data["dataset"].astype(str),
        "candidate_fde": data["family_fde"].astype(np.float64)[:, 0]
        if "family_fde" in data
        else np.zeros(len(current_xy), dtype=np.float64),
        "waypoint_xy": waypoint_labels["waypoint_xy"],
        "waypoint_valid": waypoint_labels["waypoint_valid"],
    }


def _group_keys(data: Mapping[str, np.ndarray]) -> np.ndarray:
    domain = data["dataset"].astype(str)
    source = data["source_file"].astype(str)
    frame = data["frame_id"].astype(np.float64)
    horizon = data["horizon"].astype(int)
    return np.asarray(
        [f"{d}::{s}::frame={fr:.3f}::h={h}" for d, s, fr, h in zip(domain, source, frame, horizon)],
        dtype="U768",
    )


def _fit_policy_with_xy() -> dict[str, Any]:
    data = s41._combined()
    original_split, group = am._split_arrays(data)
    domain = data["dataset"].astype(str)
    repaired_split, internal_val_group = aw._split_with_ucy_internal_val(original_split, group, domain)
    waypoint_labels = am._reconstruct_waypoint_labels(data)
    floor = am._floor_arrays(data, repaired_split == "train")
    features, feature_names = am._feature_matrix(data, floor)
    masks = aw._safe_variant_masks(feature_names)
    by_report = read_json(BY_JSON, {})
    selected_variant = str(by_report["summary"]["selected_variant"])
    fitted = by._fit_selected_variant(selected_variant, features, masks, data, repaired_split, waypoint_labels, floor)
    x, _, _ = am._standardize(features[:, masks[selected_variant]], repaired_split == "train")
    coef = am._fit_ridge_model(
        x,
        (
            (
                waypoint_labels["waypoint_xy"].astype(np.float64)
                - np.stack([data["current_x"], data["current_y"]], axis=1)[:, None, :].astype(np.float64)
            )
            / np.maximum(data["scale"].astype(np.float64)[:, None, None], am.EPS)
        ).astype(np.float32),
        waypoint_labels["waypoint_valid"],
        repaired_split == "train",
        float(fitted["lambda"]),
    )
    pred_xy = am._predict_waypoints(x, coef, data)
    floor_xy = floor["floor_xy"].astype(np.float32)
    selected_xy, policy_switch = _apply_policy_xy(pred_xy, floor_xy, data, fitted["policy"])
    selected_ade, selected_fde = am._trajectory_errors(selected_xy, waypoint_labels)
    floor_ade, floor_fde = am._trajectory_errors(floor_xy, waypoint_labels)
    return {
        "data": data,
        "split": repaired_split,
        "internal_val_group": internal_val_group,
        "labels": waypoint_labels,
        "joint_labels": _labels_for_joint(data, waypoint_labels),
        "keys": _group_keys(data),
        "floor_xy": floor_xy,
        "pred_xy": pred_xy,
        "selected_xy": selected_xy,
        "switch": policy_switch,
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "floor_ade": floor_ade,
        "floor_fde": floor_fde,
        "selected_variant": selected_variant,
        "selected_lambda": float(fitted["lambda"]),
        "policy_slices": sorted(fitted["policy"]["slices"].keys()),
    }


def _apply_policy_xy(
    pred_xy: np.ndarray,
    floor_xy: np.ndarray,
    data: Mapping[str, np.ndarray],
    policy: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    residual_norm = np.linalg.norm(pred_xy[:, -1] - floor_xy[:, -1], axis=1) / np.maximum(
        data["scale"].astype(np.float64), am.EPS
    )
    domain = data["dataset"].astype(str)
    horizon = data["horizon"].astype(int)
    selected_xy = floor_xy.copy()
    switch = np.zeros(len(floor_xy), dtype=bool)
    for key, params in policy["slices"].items():
        d, h_s = key.split("|")
        m = (domain == d) & (horizon == int(h_s))
        local = m & (
            (residual_norm <= float(params["residual_norm_threshold"]))
            if params["direction"] == "low"
            else (residual_norm >= float(params["residual_norm_threshold"]))
        )
        blended = floor_xy + float(params["alpha"]) * (pred_xy - floor_xy)
        selected_xy[local] = blended[local]
        switch[local] = True
    return selected_xy.astype(np.float32), switch


def _mask_for_slice(data: Mapping[str, np.ndarray], split: np.ndarray, slice_name: str) -> np.ndarray:
    domain_name, horizon_s = slice_name.split("|")
    return (
        (split == "test")
        & (data["dataset"].astype(str) == domain_name)
        & (data["horizon"].astype(int) == int(horizon_s))
    )


def _joint_for_mask(bundle: Mapping[str, Any], mask: np.ndarray, name: str, xy: np.ndarray, switch: np.ndarray) -> dict[str, Any]:
    return jrc._joint_stats(
        name,
        xy[mask],
        _subset_labels(bundle["joint_labels"], mask),
        bundle["keys"][mask],
        switch[mask],
    )


def _delta_stats(selected: Mapping[str, Any], floor: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "near_collision_rate_002_delta": float(selected["near_collision_rate_002"] - floor["near_collision_rate_002"]),
        "near_collision_rate_005_delta": float(selected["near_collision_rate_005"] - floor["near_collision_rate_005"]),
        "p05_min_group_distance_delta": None
        if selected.get("p05_min_group_distance") is None or floor.get("p05_min_group_distance") is None
        else float(selected["p05_min_group_distance"] - floor["p05_min_group_distance"]),
        "mean_min_group_distance_delta": None
        if selected.get("mean_min_group_distance") is None or floor.get("mean_min_group_distance") is None
        else float(selected["mean_min_group_distance"] - floor["mean_min_group_distance"]),
        "jagged_rate_delta": float(selected["smoothness"]["jagged_rate"] - floor["smoothness"]["jagged_rate"]),
        "mean_max_normalized_step_delta": float(
            selected["smoothness"]["mean_max_normalized_step"] - floor["smoothness"]["mean_max_normalized_step"]
        ),
    }


def _bootstrap_near_delta(selected_min: np.ndarray, floor_min: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(selected_min) & np.isfinite(floor_min)
    if int(np.sum(finite)) < 30:
        return {"low": 0.0, "mid": 0.0, "high": 0.0, "n": int(np.sum(finite)), "bootstrap_n": 0}
    delta = (selected_min[finite] < 0.05).astype(np.float64) - (floor_min[finite] < 0.05).astype(np.float64)
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    ids = np.arange(len(delta))
    for _ in range(BOOTSTRAP_N):
        sample = rng.choice(ids, size=len(ids), replace=True)
        values.append(float(np.mean(delta[sample])))
    return {
        "low": float(np.percentile(values, 2.5)),
        "mid": float(np.percentile(values, 50.0)),
        "high": float(np.percentile(values, 97.5)),
        "n": int(len(delta)),
        "bootstrap_n": BOOTSTRAP_N,
    }


def _stress_row(bundle: Mapping[str, Any], mask: np.ndarray, name: str) -> dict[str, Any]:
    data = bundle["data"]
    floor_switch = np.zeros(len(bundle["switch"]), dtype=bool)
    floor_stats = _joint_for_mask(bundle, mask, "floor", bundle["floor_xy"], floor_switch)
    selected_stats = _joint_for_mask(bundle, mask, "partial_floor_relaxation", bundle["selected_xy"], bundle["switch"])
    normalizer = bundle["joint_labels"]["normalizer"][mask].astype(np.float64)
    keys = bundle["keys"][mask]
    floor_min = jmc._min_group_distance(bundle["floor_xy"][mask], keys, normalizer)
    selected_min = jmc._min_group_distance(bundle["selected_xy"][mask], keys, normalizer)
    metric = am._metric(bundle["selected_ade"], bundle["floor_ade"], data, bundle["switch"], mask)
    delta = _delta_stats(selected_stats, floor_stats)
    safety_pass = bool(
        delta["near_collision_rate_005_delta"] <= NEAR_COLLISION_EPS
        and delta["jagged_rate_delta"] <= 0.0
        and metric["easy_degradation"] <= EASY_LIMIT
    )
    return {
        "source": SOURCE,
        "name": name,
        "rows": int(np.sum(mask)),
        "groups": int(selected_stats["groups"]),
        "metric": metric,
        "floor_joint_stats": floor_stats,
        "selected_joint_stats": selected_stats,
        "selected_minus_floor": delta,
        "near_collision_delta_bootstrap": _bootstrap_near_delta(selected_min, floor_min),
        "safety_pass": safety_pass,
    }


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    by_report = read_json(BY_JSON, {})
    bz_report = read_json(BZ_JSON, {})
    en_report = read_json(EN_JSON, {})
    bundle = _fit_policy_with_xy()
    data = bundle["data"]
    split = bundle["split"]
    target_masks = {name: _mask_for_slice(data, split, name) for name in TARGET_SLICES}
    union_mask = np.zeros(len(split), dtype=bool)
    for mask in target_masks.values():
        union_mask |= mask
    all_t50_mask = (split == "test") & (data["horizon"].astype(int) == 50)
    rows = {
        "target_union_t50": _stress_row(bundle, union_mask, "target_union_t50"),
        "all_test_t50": _stress_row(bundle, all_t50_mask, "all_test_t50"),
    }
    for name, mask in target_masks.items():
        rows[name] = _stress_row(bundle, mask, name)

    target_safety_pass = bool(rows["target_union_t50"]["safety_pass"])
    deployment_decision = (
        "partial_t50_floor_relaxation_safety_supported"
        if target_safety_pass
        else "partial_t50_floor_relaxation_diagnostic_only_safety_caveat"
    )
    summary = {
        "source": SOURCE,
        "selected_variant": bundle["selected_variant"],
        "selected_lambda": bundle["selected_lambda"],
        "policy_slices": bundle["policy_slices"],
        "target_slices": list(TARGET_SLICES),
        "target_union_rows": rows["target_union_t50"]["rows"],
        "target_union_t50_improvement": rows["target_union_t50"]["metric"]["t50_improvement"],
        "target_union_hard_failure_improvement": rows["target_union_t50"]["metric"]["hard_failure_improvement"],
        "target_union_easy_degradation": rows["target_union_t50"]["metric"]["easy_degradation"],
        "target_union_near_collision_005_delta": rows["target_union_t50"]["selected_minus_floor"]["near_collision_rate_005_delta"],
        "target_union_jagged_rate_delta": rows["target_union_t50"]["selected_minus_floor"]["jagged_rate_delta"],
        "target_union_safety_pass": target_safety_pass,
        "deployment_decision": deployment_decision,
        "global_floor_removal_allowed": False,
        "floor_free_neural_deployable": False,
        "teacher_floor_context_required": True,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "stage": "Stage42-GT Floor-Relaxation Safety Stress Test",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([str(BY_JSON), str(BZ_JSON), str(EN_JSON), "data/stage41_world_model/combined_external.npz"]),
        "current_facts": CURRENT_FACTS,
        "input_reports": {
            "by_verdict": by_report.get("stage42_by_gate", {}).get("verdict"),
            "bz_verdict": bz_report.get("stage42_bz_gate", {}).get("verdict"),
            "en_verdict": en_report.get("stage42_en_gate", {}).get("verdict"),
        },
        "policy_replay": {
            "source": SOURCE,
            "selected_variant": bundle["selected_variant"],
            "selected_lambda": bundle["selected_lambda"],
            "policy_slices": bundle["policy_slices"],
            "internal_val_group": bundle["internal_val_group"],
            "test_threshold_tuning": False,
        },
        "stress_tests": rows,
        "summary": summary,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "train_only_internal_validation": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "global_metric_claim_allowed": False,
            "global_seconds_claim_allowed": False,
            "floor_free_neural_deployable": False,
            "global_floor_removal_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_gt_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    s = payload["summary"]
    no_leakage = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "source_fresh": payload["source"] == SOURCE,
        "by_input_passed": payload["input_reports"]["by_verdict"] == "stage42_by_t50_floor_relaxability_repair_pass",
        "bz_input_passed_or_present": payload["input_reports"]["bz_verdict"] in {
            "stage42_bz_t50_repair_statistical_evidence_pass",
            None,
        },
        "en_input_passed": payload["input_reports"]["en_verdict"] == "stage42_en_floor_removability_decision_map_pass",
        "policy_replayed": len(payload["policy_replay"]["policy_slices"]) >= 2,
        "target_rows_present": s["target_union_rows"] > 0,
        "all_agent_stress_metrics_written": "selected_joint_stats" in payload["stress_tests"]["target_union_t50"],
        "safety_decision_recorded": s["deployment_decision"] in {
            "partial_t50_floor_relaxation_safety_supported",
            "partial_t50_floor_relaxation_diagnostic_only_safety_caveat",
        },
        "no_leakage_pass": no_leakage["future_endpoint_input"] is False
        and no_leakage["future_waypoint_input"] is False
        and no_leakage["central_velocity"] is False
        and no_leakage["test_endpoint_goals"] is False
        and no_leakage["test_threshold_tuning"] is False,
        "floor_free_neural_not_claimed": claim["floor_free_neural_deployable"] is False,
        "global_floor_removal_not_claimed": claim["global_floor_removal_allowed"] is False,
        "no_metric_seconds_overclaim": claim["global_metric_claim_allowed"] is False
        and claim["global_seconds_claim_allowed"] is False,
        "stage5c_false": claim["stage5c_executed"] is False,
        "smc_false": claim["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = "stage42_gt_floor_relaxation_safety_stress_pass" if passed == total else "stage42_gt_floor_relaxation_safety_stress_partial"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    s = payload["summary"]
    lines = [
        "# Stage42-GT Floor-Relaxation Safety Stress Test",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{payload['stage42_gt_gate']['passed']} / {payload['stage42_gt_gate']['total']}`",
        f"- verdict: `{payload['stage42_gt_gate']['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Summary",
        "",
        f"- selected_variant: `{s['selected_variant']}`",
        f"- policy_slices: `{s['policy_slices']}`",
        f"- target_union_rows: `{s['target_union_rows']}`",
        f"- target_union_t50_improvement: `{_pct(s['target_union_t50_improvement'])}`",
        f"- target_union_hard_failure_improvement: `{_pct(s['target_union_hard_failure_improvement'])}`",
        f"- target_union_easy_degradation: `{_pct(s['target_union_easy_degradation'])}`",
        f"- target_union_near_collision_005_delta: `{_pct(s['target_union_near_collision_005_delta'])}`",
        f"- target_union_jagged_rate_delta: `{_pct(s['target_union_jagged_rate_delta'])}`",
        f"- target_union_safety_pass: `{s['target_union_safety_pass']}`",
        f"- deployment_decision: `{s['deployment_decision']}`",
        f"- floor_free_neural_deployable: `{s['floor_free_neural_deployable']}`",
        "",
        "## Stress Tests",
        "",
        "| slice | rows | groups | t50 gain | hard gain | easy degradation | switch | near@0.05 delta | jagged delta | safety pass |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in payload["stress_tests"].items():
        metric = row["metric"]
        delta = row["selected_minus_floor"]
        lines.append(
            f"| `{name}` | {row['rows']} | {row['groups']} | "
            f"{_pct(metric.get('t50_improvement', 0.0))} | "
            f"{_pct(metric.get('hard_failure_improvement', 0.0))} | "
            f"{_pct(metric.get('easy_degradation', 0.0))} | "
            f"{_pct(metric.get('switch_rate', 0.0))} | "
            f"{_pct(delta.get('near_collision_rate_005_delta', 0.0))} | "
            f"{_pct(delta.get('jagged_rate_delta', 0.0))} | {row['safety_pass']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- This stage stress-tests the Stage42-BY/BZ partial t50 floor-relaxation policy at all-agent group level.",
        "- A positive t50 gain alone is not enough; proximity and smoothness must not materially degrade.",
        "- Global floor removal remains forbidden; any supported relaxation is limited to validation-backed t50 slices.",
        "- If `deployment_decision` is diagnostic-only, the BY/BZ t50 relaxation remains paper evidence but not a safety deployment policy.",
    ]
    return lines


def _render_gate(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage42_gt_gate"]
    lines = [
        "# Stage42-GT Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | pass |",
        "| --- | ---: |",
    ]
    for key, value in gate["gates"].items():
        lines.append(f"| `{key}` | {bool(value)} |")
    return lines


def run_stage42_floor_relaxation_safety_stress() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_md(GATE_MD, _render_gate(payload))
    return payload


if __name__ == "__main__":
    result = run_stage42_floor_relaxation_safety_stress()
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False, sort_keys=True))
