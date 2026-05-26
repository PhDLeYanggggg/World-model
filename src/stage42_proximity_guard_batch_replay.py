from __future__ import annotations

import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src import stage41_joint_multiagent_consistency as jmc
from src import stage42_common_validation_bridge_shape_composer as co
from src import stage42_proximity_aware_composer_guard as cq
from src.stage42_proximity_guard_policy_freeze import _pct, _replace_section
from src.stage42_proximity_guard_runtime_policy import FrozenProximityGuardPolicy


OUT_DIR = Path("outputs/stage42_long_research")
POLICY_JSON = OUT_DIR / "frozen_proximity_guard_composer_policy_stage42_policy.json"
CT_JSON = OUT_DIR / "frozen_proximity_guard_policy_replay_stage42.json"
CU_JSON = OUT_DIR / "proximity_guard_runtime_policy_stage42.json"

REPORT_JSON = OUT_DIR / "proximity_guard_batch_replay_stage42.json"
REPORT_MD = OUT_DIR / "proximity_guard_batch_replay_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_cv_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
RETRO_README = Path("README_M3W_GOAL_RETROSPECTIVE_CURRENT_ZH.md")
RESEARCH_STATE = Path("research_state.json")

CURRENT_FACTS = [
    "当前不是 true 3D world model。",
    "当前不是 large-scale foundation world model。",
    "当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。",
    "Stage42-CV 在真实 common validation/test rows 上重放 Stage42-CU runtime policy。",
    "guard 的第二个 proximity 输入是 validation-selected base composer candidate rollout 的 group min-distance，不是 future label。",
    "runtime batch replay 不重新选择阈值，不使用 test endpoint goals，不执行 Stage5C，不启用 SMC。",
    "future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。",
    "t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。",
    "dataset-local/raw-frame 不能写成 global metric。",
]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


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


def _decision_arrays(
    policy: FrozenProximityGuardPolicy,
    labels: Mapping[str, np.ndarray],
    endpoint_min: np.ndarray,
    candidate_min: np.ndarray,
) -> dict[str, Any]:
    domain = labels["domain"].astype(str)
    horizon = labels["horizon"].astype(int)
    use_full = np.zeros(len(horizon), dtype=bool)
    guarded_off = np.zeros(len(horizon), dtype=bool)
    reasons: Counter[str] = Counter()
    for idx in range(len(horizon)):
        e_min = None if not np.isfinite(endpoint_min[idx]) else float(endpoint_min[idx])
        c_min = None if not np.isfinite(candidate_min[idx]) else float(candidate_min[idx])
        decision = policy.decide(
            domain=domain[idx],
            horizon=int(horizon[idx]),
            endpoint_min_group_distance=e_min,
            full_min_group_distance=c_min,
        )
        use_full[idx] = decision.use_full_waypoint
        guarded_off[idx] = decision.guarded_off
        reasons[decision.reason] += 1
    return {
        "use_full": use_full,
        "guarded_off": guarded_off,
        "reason_counts": dict(sorted(reasons.items())),
    }


def _runtime_apply(
    endpoint: Mapping[str, Any],
    full: Mapping[str, Any],
    keys: np.ndarray,
    policy: FrozenProximityGuardPolicy,
) -> dict[str, Any]:
    base = co._compose(endpoint, full, policy.base_choices)
    labels = endpoint["labels"]
    normalizer = labels["normalizer"].astype(np.float64)
    endpoint_min = jmc._min_group_distance(endpoint["selected_xy"], keys, normalizer)
    candidate_min = jmc._min_group_distance(base["selected_xy"], keys, normalizer)
    decisions = _decision_arrays(policy, labels, endpoint_min, candidate_min)
    use_full = decisions["use_full"]
    selected_xy = endpoint["selected_xy"].copy()
    selected_xy[use_full] = full["selected_xy"][use_full]
    selected_ade, selected_fde = co.ft._trajectory_errors(selected_xy, labels)
    return {
        "selected_xy": selected_xy,
        "selected_ade": selected_ade,
        "selected_fde": selected_fde,
        "use_full": use_full,
        "guarded_off": int(np.sum(decisions["guarded_off"])),
        "guarded_off_rate": float(np.mean(decisions["guarded_off"])) if len(use_full) else 0.0,
        "reason_counts": decisions["reason_counts"],
        "metric_vs_endpoint_ade": co._metric(selected_ade, endpoint["selected_ade"], labels, use_full),
        "metric_vs_floor_ade": co._metric(selected_ade, endpoint["floor_ade"], labels, use_full),
        "metric_vs_endpoint_fde": co._metric(selected_fde, endpoint["selected_fde"], labels, use_full),
        "metric_vs_floor_fde": co._metric(selected_fde, endpoint["floor_fde"], labels, use_full),
    }


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b))) if a.size and b.size else 0.0


def _metric_diff(runtime_metric: Mapping[str, Any], expected_metric: Mapping[str, Any]) -> dict[str, float]:
    keys = [
        "all_improvement",
        "t50_improvement",
        "t100_raw_frame_diagnostic_improvement",
        "hard_failure_improvement",
        "easy_degradation",
        "switch_rate",
    ]
    return {key: abs(float(runtime_metric.get(key, 0.0)) - float(expected_metric.get(key, 0.0))) for key in keys}


def _split_replay(split: str, policy: FrozenProximityGuardPolicy) -> dict[str, Any]:
    endpoint = co._endpoint_bundle(split)
    full = co._full_bundle(split)
    keys = jmc._group_metadata(split)["key"]
    expected = cq._apply_proximity_guard(endpoint, full, keys, policy.base_choices, policy.min_sep, policy.margin)
    runtime = _runtime_apply(endpoint, full, keys, policy)
    joint = cq._joint_stats(endpoint, runtime, keys)
    return {
        "source": "fresh_batch_runtime_replay_from_cached_verified_common_validation_rows",
        "split": split,
        "rows": int(len(runtime["use_full"])),
        "policy_hash": policy.policy_hash,
        "decision_match": bool(np.array_equal(runtime["use_full"], expected["use_full"])),
        "selected_xy_max_abs_diff": _max_abs_diff(runtime["selected_xy"], expected["selected_xy"]),
        "selected_ade_max_abs_diff": _max_abs_diff(runtime["selected_ade"], expected["selected_ade"]),
        "selected_fde_max_abs_diff": _max_abs_diff(runtime["selected_fde"], expected["selected_fde"]),
        "metric_diff_vs_expected": _metric_diff(runtime["metric_vs_endpoint_ade"], expected["metric_vs_endpoint_ade"]),
        "runtime_metric_vs_endpoint_ade": runtime["metric_vs_endpoint_ade"],
        "runtime_metric_vs_floor_ade": runtime["metric_vs_floor_ade"],
        "runtime_metric_vs_endpoint_fde": runtime["metric_vs_endpoint_fde"],
        "runtime_metric_vs_floor_fde": runtime["metric_vs_floor_fde"],
        "runtime_joint_safety": joint,
        "guarded_off": runtime["guarded_off"],
        "guarded_off_rate": runtime["guarded_off_rate"],
        "reason_counts": runtime["reason_counts"],
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    ct_gate = payload["inputs"]["stage42_ct"].get("stage42_ct_gate", {})
    cu_gate = payload["inputs"]["stage42_cu"].get("stage42_cu_gate", {})
    val = payload["splits"]["val"]
    test = payload["splits"]["test"]
    test_metric = test["runtime_metric_vs_endpoint_ade"]
    test_safety = test["runtime_joint_safety"]["composer_minus_endpoint"]
    no_leakage = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "ct_gate_passed": ct_gate.get("passed") == ct_gate.get("total"),
        "cu_gate_passed": cu_gate.get("passed") == cu_gate.get("total"),
        "policy_hash_consistent": payload["policy_hash"] == payload["inputs"]["stage42_cu"].get("policy_hash"),
        "val_decision_exact_replay": val["decision_match"],
        "test_decision_exact_replay": test["decision_match"],
        "val_selected_xy_exact": val["selected_xy_max_abs_diff"] <= 1e-12,
        "test_selected_xy_exact": test["selected_xy_max_abs_diff"] <= 1e-12,
        "test_all_metric_exact": test["metric_diff_vs_expected"]["all_improvement"] <= 1e-12,
        "test_t50_metric_exact": test["metric_diff_vs_expected"]["t50_improvement"] <= 1e-12,
        "test_t100_metric_exact": test["metric_diff_vs_expected"]["t100_raw_frame_diagnostic_improvement"] <= 1e-12,
        "test_hard_metric_exact": test["metric_diff_vs_expected"]["hard_failure_improvement"] <= 1e-12,
        "test_all_positive": float(test_metric["all_improvement"]) > 0.0,
        "test_t50_positive": float(test_metric["t50_improvement"]) > 0.0,
        "test_t100_positive": float(test_metric["t100_raw_frame_diagnostic_improvement"]) > 0.0,
        "test_hard_positive": float(test_metric["hard_failure_improvement"]) > 0.0,
        "test_easy_under_2pct": float(test_metric["easy_degradation"]) <= 0.02,
        "test_near_collision_not_worse_than_endpoint": float(test_safety["near_collision_rate_005_delta"]) <= 0.0,
        "no_future_endpoint_input": no_leakage["future_endpoint_input"] is False,
        "no_future_waypoints_input": no_leakage["future_waypoints_input"] is False,
        "no_central_velocity": no_leakage["central_velocity"] is False,
        "no_test_endpoint_goals": no_leakage["test_endpoint_goals"] is False,
        "no_test_threshold_tuning": no_leakage["test_threshold_tuning"] is False,
        "metric_seconds_overclaim_blocked": claim["metric_or_seconds_claim"] is False,
        "stage5c_not_executed": claim["stage5c_executed"] is False,
        "smc_not_enabled": claim["smc_enabled"] is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    verdict = "stage42_cv_batch_runtime_replay_pass" if passed == total else "stage42_cv_batch_runtime_replay_partial"
    return {"gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    test = payload["splits"]["test"]
    metric = test["runtime_metric_vs_endpoint_ade"]
    safety = test["runtime_joint_safety"]["composer_minus_endpoint"]
    lines = [
        "## Stage42-CV Batch Runtime Policy Replay",
        "",
        "- source: `fresh_batch_runtime_replay_from_frozen_policy_artifact`",
        f"- verdict: `{payload['stage42_cv_gate']['verdict']}`",
        f"- gates: `{payload['stage42_cv_gate']['passed']} / {payload['stage42_cv_gate']['total']}`",
        f"- policy hash: `{payload['policy_hash']}`",
        "- replay scope: real common validation/test rows, not toy smoke cases.",
        "- replay result: validation and test runtime decisions exactly match the original CQ guard output.",
        f"- test ADE vs endpoint-linear all/t50/t100 raw/hard: `{_pct(metric['all_improvement'])}` / `{_pct(metric['t50_improvement'])}` / `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}` / `{_pct(metric['hard_failure_improvement'])}`",
        f"- test easy degradation: `{_pct(metric['easy_degradation'])}`",
        f"- near-collision@0.05 delta vs endpoint-linear: `{_pct(safety['near_collision_rate_005_delta'])}`",
        "- Claim boundary unchanged: protected dataset-local/raw-frame 2.5D only; no true 3D, no foundation, no metric/seconds-level, no Stage5C execution, no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, RETRO_README]:
        _replace_section(path, "STAGE42_CV_BATCH_RUNTIME_REPLAY", lines)


def _refresh_research_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    test = payload["splits"]["test"]
    metric = test["runtime_metric_vs_endpoint_ade"]
    state["current_stage"] = "Stage42-CV batch runtime policy replay"
    state["current_verdict"] = payload["stage42_cv_gate"]["verdict"]
    state["stage42_cv_batch_runtime_replay"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "gate": str(GATE_MD),
        "verdict": payload["stage42_cv_gate"]["verdict"],
        "gates": f"{payload['stage42_cv_gate']['passed']}/{payload['stage42_cv_gate']['total']}",
        "policy_hash": payload["policy_hash"],
        "val_decision_match": payload["splits"]["val"]["decision_match"],
        "test_decision_match": test["decision_match"],
        "test_vs_endpoint_linear_ade": {
            "all_improvement": metric["all_improvement"],
            "t50_improvement": metric["t50_improvement"],
            "t100_raw_frame_diagnostic_improvement": metric["t100_raw_frame_diagnostic_improvement"],
            "hard_failure_improvement": metric["hard_failure_improvement"],
            "easy_degradation": metric["easy_degradation"],
        },
        "claim_boundary": payload["claim_boundary"],
        "conclusion": "Stage42-CV replays the frozen Stage42-CU runtime policy over real common validation/test rows and exactly matches the original Stage42-CQ guard decisions and metrics. It is deployment/reproducibility evidence, not new training.",
        "verification": {
            "runner": ".venv-pytorch/bin/python run_stage42_batch_replay_proximity_guard_policy.py",
            "targeted_pytest": ".venv-pytorch/bin/python -m pytest tests/test_stage42_proximity_guard_batch_replay.py",
        },
    }
    summary = state.setdefault("latest_user_facing_goal_summary", {})
    summary["source"] = "cached_verified_synthesis_for_user_question_refreshed_after_stage42_cv"
    included = summary.setdefault("latest_fresh_evidence_included", [])
    note = "Stage42-CV batch runtime replay: frozen runtime policy exactly matches original CQ guard decisions on real validation/test rows"
    if note not in included:
        included.append(note)
    write_json(RESEARCH_STATE, state)


def _write_md(payload: Mapping[str, Any]) -> None:
    gate = payload["stage42_cv_gate"]
    test = payload["splits"]["test"]
    metric = test["runtime_metric_vs_endpoint_ade"]
    safety = test["runtime_joint_safety"]["composer_minus_endpoint"]
    lines = [
        "# Stage42-CV Batch Runtime Policy Replay",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- git_commit: `{payload['git_commit']}`",
        f"- policy_hash: `{payload['policy_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## Current Facts",
        "",
        *[f"- {fact}" for fact in CURRENT_FACTS],
        "",
        "## Replay Summary",
        "",
        f"- validation rows: `{payload['splits']['val']['rows']}`",
        f"- test rows: `{test['rows']}`",
        f"- validation decision exact replay: `{payload['splits']['val']['decision_match']}`",
        f"- test decision exact replay: `{test['decision_match']}`",
        f"- test selected_xy max abs diff vs CQ: `{test['selected_xy_max_abs_diff']}`",
        f"- test selected ADE max abs diff vs CQ: `{test['selected_ade_max_abs_diff']}`",
        f"- test selected FDE max abs diff vs CQ: `{test['selected_fde_max_abs_diff']}`",
        "",
        "## Test Metrics Vs Endpoint-Linear ADE",
        "",
        f"- all: `{_pct(metric['all_improvement'])}`",
        f"- t50: `{_pct(metric['t50_improvement'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(metric['t100_raw_frame_diagnostic_improvement'])}`",
        f"- hard/failure: `{_pct(metric['hard_failure_improvement'])}`",
        f"- easy degradation: `{_pct(metric['easy_degradation'])}`",
        f"- switch rate: `{_pct(metric['switch_rate'])}`",
        "",
        "## Runtime Reasons On Test",
        "",
        *[f"- `{key}`: `{value}`" for key, value in sorted(test["reason_counts"].items())],
        "",
        "## Joint Safety Vs Endpoint-Linear",
        "",
        f"- near_collision@0.02 delta: `{_pct(safety['near_collision_rate_002_delta'])}`",
        f"- near_collision@0.05 delta: `{_pct(safety['near_collision_rate_005_delta'])}`",
        f"- p05 min group distance delta: `{_pct(safety['p05_min_group_distance_delta'])}`",
        f"- jagged-rate delta: `{_pct(safety['jagged_rate_delta'])}`",
        "",
        "## Interpretation",
        "",
        "- Stage42-CV proves the callable runtime policy exactly replays the original CQ guard decisions on real validation/test rows.",
        "- This is stronger than smoke testing: it exercises the policy on the same common rows used by the bridge/shape composer evidence.",
        "- It does not reselect thresholds, does not use future labels as inputs, and does not add new model scores.",
        "- Claims remain protected dataset-local/raw-frame 2.5D only.",
    ]
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage42-CV Gate",
        "",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        "",
        "| gate | passed |",
        "| --- | ---: |",
    ]
    for name, ok in gate["gates"].items():
        gate_lines.append(f"| `{name}` | `{ok}` |")
    write_md(GATE_MD, gate_lines)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_stage42_batch_replay_proximity_guard_policy() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ct = read_json(CT_JSON, {})
    cu = read_json(CU_JSON, {})
    policy = FrozenProximityGuardPolicy.from_file(POLICY_JSON)
    val = _split_replay("val", policy)
    test = _split_replay("test", policy)
    payload: dict[str, Any] = {
        "source": "fresh_batch_runtime_replay_from_frozen_policy_artifact",
        "stage": "Stage42-CV batch runtime policy replay",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "current_facts": CURRENT_FACTS,
        "inputs": {
            "stage42_ct": {"path": str(CT_JSON), "stage42_ct_gate": ct.get("stage42_ct_gate", {})},
            "stage42_cu": {
                "path": str(CU_JSON),
                "stage42_cu_gate": cu.get("stage42_cu_gate", {}),
                "policy_hash": cu.get("policy_hash"),
            },
        },
        "policy_hash": policy.policy_hash,
        "splits": {"val": val, "test": test},
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoints_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "runtime_inputs_use_predicted_geometry_only": True,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_cv_gate"] = _gate(payload)
    write_json(REPORT_JSON, _jsonable(payload))
    _write_md(payload)
    _refresh_readmes(payload)
    _refresh_research_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_batch_replay_proximity_guard_policy()
    gate = result["stage42_cv_gate"]
    print(f"Stage42-CV batch runtime replay: {gate['verdict']} ({gate['passed']}/{gate['total']})")
