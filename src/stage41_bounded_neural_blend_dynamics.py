from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_breakthrough as s41
from src import stage41_full_trajectory_world_state as ft
from src import stage41_joint_rollout_consistency as jrc
from src import stage41_teacher_guided_evidence as evidence
from src import stage41_teacher_guided_proposal as tgp


OUT_DIR = tgp.OUT_DIR
REPORT_JSON = OUT_DIR / "stage41_bounded_neural_blend_dynamics.json"
REPORT_MD = OUT_DIR / "stage41_bounded_neural_blend_dynamics.md"
TEST_COLLISION_CEILING = 0.01
EPS = 1e-6


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
    return value


def _append_ledger(step: str, status: str, started: float, inputs: Sequence[str | Path], outputs: Sequence[str | Path]) -> None:
    ensure_dir(OUT_DIR)
    entry = {
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "step": step,
        "source": "fresh_run",
        "status": status,
        "wall_time_s": time.perf_counter() - started,
        "input_hash": _combined_hash(inputs),
        "output_hash": _combined_hash(outputs),
        "git_commit": _git_commit(),
    }
    with (OUT_DIR / "run_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


def _load_frozen_model() -> tuple[str, dict[str, Any], float]:
    checkpoint, policy, min_sep = evidence._selected_checkpoint_policy_guard()
    return checkpoint, policy, float(min_sep)


def _bundle(split: str, checkpoint: str, policy: Mapping[str, Any], min_sep: float) -> dict[str, Any]:
    data = tgp._bundle(split)
    pred = tgp._predict(checkpoint, data)
    raw_switch = tgp._policy_switch(pred, policy)
    repaired_switch, repaired_off = jrc._apply_proximity_guard(
        data["floor_xy"],
        data["neural_xy"],
        data["labels"],
        data["keys"],
        raw_switch.astype(bool),
        min_sep,
    )
    return {
        **data,
        "teacher_raw_switch": raw_switch.astype(bool),
        "teacher_repaired_switch": repaired_switch.astype(bool),
        "teacher_repaired_guarded_off": np.asarray([repaired_off], dtype=np.int64),
        "teacher_repaired_min_sep": np.asarray([float(min_sep)], dtype=np.float32),
    }


def _alpha_vector(data: Mapping[str, Any], policy: Mapping[str, Any]) -> np.ndarray:
    labels = data["labels"]
    horizon = labels["horizon"].astype(int)
    domain = labels["domain"].astype(str)
    if policy["type"] == "global":
        alpha = np.full(len(horizon), float(policy["alpha"]), dtype=np.float64)
    elif policy["type"] == "horizon":
        by_h = {int(k): float(v) for k, v in policy["alpha_by_horizon"].items()}
        alpha = np.asarray([by_h.get(int(h), 0.0) for h in horizon], dtype=np.float64)
    elif policy["type"] == "domain_horizon":
        nested = {
            str(d): {int(k): float(v) for k, v in row.items()}
            for d, row in policy["alpha_by_domain_horizon"].items()
        }
        alpha = np.asarray([nested.get(str(d), {}).get(int(h), 0.0) for d, h in zip(domain, horizon)], dtype=np.float64)
    else:
        raise ValueError(f"unknown policy type: {policy['type']}")
    gate = str(policy.get("gate", "all"))
    if gate == "teacher_raw_switch":
        alpha = alpha * data["teacher_raw_switch"].astype(np.float64)
    elif gate == "teacher_repaired_switch":
        alpha = alpha * data["teacher_repaired_switch"].astype(np.float64)
    elif gate == "teacher_prob_050":
        alpha = alpha * (data["teacher_prob"].astype(np.float64) >= 0.50)
    elif gate == "teacher_prob_070":
        alpha = alpha * (data["teacher_prob"].astype(np.float64) >= 0.70)
    elif gate != "all":
        raise ValueError(f"unknown gate: {gate}")
    return alpha


def _candidate_policies() -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    for alpha in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40]:
        policies.append({"type": "global", "alpha": alpha})
        policies.append({"type": "global", "alpha": alpha, "gate": "teacher_raw_switch"})
        policies.append({"type": "global", "alpha": alpha, "gate": "teacher_repaired_switch"})
        policies.append({"type": "global", "alpha": alpha, "gate": "teacher_prob_070"})
    horizon_templates = [
        {10: 0.02, 25: 0.03, 50: 0.06, 100: 0.06},
        {10: 0.03, 25: 0.05, 50: 0.10, 100: 0.10},
        {10: 0.05, 25: 0.08, 50: 0.15, 100: 0.15},
        {10: 0.05, 25: 0.10, 50: 0.20, 100: 0.20},
        {10: 0.08, 25: 0.12, 50: 0.25, 100: 0.25},
    ]
    for row in horizon_templates:
        policies.append({"type": "horizon", "alpha_by_horizon": row})
        policies.append({"type": "horizon", "alpha_by_horizon": row, "gate": "teacher_raw_switch"})
        policies.append({"type": "horizon", "alpha_by_horizon": row, "gate": "teacher_repaired_switch"})
        policies.append({"type": "horizon", "alpha_by_horizon": row, "gate": "teacher_prob_070"})
    domains = ["ETH_UCY", "TrajNet", "UCY"]
    for base in horizon_templates[1:]:
        policies.append(
            {
                "type": "domain_horizon",
                "alpha_by_domain_horizon": {
                    d: {h: (a * (0.75 if d == "ETH_UCY" else 1.0 if d == "TrajNet" else 0.85)) for h, a in base.items()}
                    for d in domains
                },
            }
        )
    return policies


def _evaluate_blend(data: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    labels = data["labels"]
    alpha = _alpha_vector(data, policy)
    floor_xy = data["floor_xy"].astype(np.float64)
    neural_xy = data["neural_xy"].astype(np.float64)
    blended_xy = floor_xy + alpha[:, None, None] * (neural_xy - floor_xy)
    blended_ade, _blended_fde = ft._trajectory_errors(blended_xy, labels)
    floor_ade = data["floor_ade"].astype(np.float64)
    intervention = alpha > EPS
    ds = {
        "horizon": labels["horizon"],
        "hard": labels["hard"],
        "failure": labels["failure"],
        "easy": labels["easy"],
        "domain": labels["domain"],
        "candidate_fde": labels["candidate_fde"],
    }
    metrics = s41._metrics(blended_ade, floor_ade, ds, intervention)
    floor_stats = jrc._joint_stats("floor", floor_xy, labels, data["keys"], np.zeros(len(alpha), dtype=bool))
    blend_stats = jrc._joint_stats("bounded_neural_blend", blended_xy, labels, data["keys"], intervention)
    metrics["alpha_mean"] = float(np.mean(alpha))
    metrics["alpha_positive_rate"] = float(np.mean(intervention))
    metrics["collision_delta_vs_floor_005"] = float(blend_stats["near_collision_rate_005"] - floor_stats["near_collision_rate_005"])
    metrics["smoothness_jagged_delta"] = float(blend_stats["smoothness"]["jagged_rate"] - floor_stats["smoothness"]["jagged_rate"])
    return {
        "policy": _jsonable(policy),
        "metrics": metrics,
        "floor_stats": floor_stats,
        "blend_stats": blend_stats,
    }


def _score(metrics: Mapping[str, Any], collision_ceiling: float) -> float:
    return (
        float(metrics.get("all_improvement", 0.0))
        + 1.4 * float(metrics.get("t50_improvement", 0.0))
        + 1.0 * float(metrics.get("t100_improvement", 0.0))
        + 1.2 * float(metrics.get("hard_failure_improvement", 0.0))
        - 35.0 * max(0.0, float(metrics.get("easy_degradation", 1.0)) - 0.02)
        - 10.0 * max(0.0, float(metrics.get("collision_delta_vs_floor_005", 1.0)) - collision_ceiling)
        - 2.0 * max(0.0, float(metrics.get("smoothness_jagged_delta", 1.0)) - 0.01)
    )


def _eligible(metrics: Mapping[str, Any], collision_ceiling: float) -> bool:
    return bool(
        metrics.get("all_improvement", 0.0) > 0
        and (
            metrics.get("t50_improvement", 0.0) > 0
            or metrics.get("hard_failure_improvement", 0.0) > 0
        )
        and metrics.get("easy_degradation", 1.0) <= 0.02
        and metrics.get("collision_delta_vs_floor_005", 1.0) <= collision_ceiling
        and metrics.get("alpha_mean", 0.0) > 0.0
    )


def _select_policy(train: Mapping[str, Any], val: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for policy in _candidate_policies():
        ev = _evaluate_blend(val, policy)
        train_ev = _evaluate_blend(train, policy)
        metrics = ev["metrics"]
        train_metrics = train_ev["metrics"]
        train_stress_safe = bool(
            train_metrics.get("easy_degradation", 1.0) <= 0.02
            and train_metrics.get("collision_delta_vs_floor_005", 1.0) <= TEST_COLLISION_CEILING
        )
        rows.append(
            {
                "policy": ev["policy"],
                "metrics": metrics,
                "train_stress_metrics": train_metrics,
                "train_stress_safe": train_stress_safe,
                "eligible": _eligible(metrics, TEST_COLLISION_CEILING) and train_stress_safe,
                "score": _score(metrics, TEST_COLLISION_CEILING),
            }
        )
    pool = [row for row in rows if row["eligible"]] or rows
    selected = max(pool, key=lambda row: row["score"])
    return {"selected": selected, "candidates": rows}


def _select_safe_switch_policy(train: Mapping[str, Any], val: Mapping[str, Any]) -> dict[str, Any]:
    """Select a continuous blend only inside the already validated switch set."""

    rows: list[dict[str, Any]] = []
    for policy in _candidate_policies():
        if policy.get("gate") != "teacher_repaired_switch":
            continue
        ev = _evaluate_blend(val, policy)
        train_ev = _evaluate_blend(train, policy)
        metrics = ev["metrics"]
        train_metrics = train_ev["metrics"]
        train_stress_safe = bool(
            train_metrics.get("easy_degradation", 1.0) <= 0.02
            and train_metrics.get("collision_delta_vs_floor_005", 1.0) <= TEST_COLLISION_CEILING
        )
        rows.append(
            {
                "policy": ev["policy"],
                "metrics": metrics,
                "train_stress_metrics": train_metrics,
                "train_stress_safe": train_stress_safe,
                "eligible": _eligible(metrics, TEST_COLLISION_CEILING) and train_stress_safe,
                "score": _score(metrics, TEST_COLLISION_CEILING),
            }
        )
    pool = [row for row in rows if row["eligible"]] or rows
    selected = max(pool, key=lambda row: row["score"])
    return {"selected": selected, "candidates": rows, "constraint": "teacher_repaired_switch_only"}


def run_bounded_neural_blend_dynamics() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    checkpoint, teacher_policy, min_sep = _load_frozen_model()
    train = _bundle("train", checkpoint, teacher_policy, min_sep)
    val = _bundle("val", checkpoint, teacher_policy, min_sep)
    selection = _select_policy(train, val)
    safe_switch_selection = _select_safe_switch_policy(train, val)
    test = _bundle("test", checkpoint, teacher_policy, min_sep)
    selected_policy = (selection["selected"] or {}).get("policy") or {}
    test_eval = _evaluate_blend(test, selected_policy)
    metrics = test_eval["metrics"]
    deployable = _eligible(metrics, TEST_COLLISION_CEILING)
    safe_switch_policy = (safe_switch_selection["selected"] or {}).get("policy") or {}
    safe_switch_eval = _evaluate_blend(test, safe_switch_policy)
    safe_switch_metrics = safe_switch_eval["metrics"]
    safe_switch_deployable = _eligible(safe_switch_metrics, TEST_COLLISION_CEILING)
    result = {
        "source": "fresh_run",
        "protocol": "bounded_neural_blend_dynamics",
        "checkpoint": checkpoint,
        "teacher_policy": teacher_policy,
        "teacher_repaired_min_sep": min_sep,
        "validation_selection": selection,
        "test_policy": selected_policy,
        "test_metrics": metrics,
        "test_floor_stats": test_eval["floor_stats"],
        "test_blend_stats": test_eval["blend_stats"],
        "bounded_neural_blend_deployable": deployable,
        "non_fallback_continuous_neural_contribution": bool(deployable and metrics.get("alpha_mean", 0.0) > 0.0),
        "safe_switch_validation_selection": safe_switch_selection,
        "safe_switch_test_policy": safe_switch_policy,
        "safe_switch_test_metrics": safe_switch_metrics,
        "safe_switch_test_floor_stats": safe_switch_eval["floor_stats"],
        "safe_switch_test_blend_stats": safe_switch_eval["blend_stats"],
        "safe_switch_bounded_neural_blend_deployable": safe_switch_deployable,
        "safe_switch_non_fallback_continuous_neural_contribution": bool(
            safe_switch_deployable and safe_switch_metrics.get("alpha_mean", 0.0) > 0.0
        ),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_labels_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "blend_policy_selected_on_val": True,
            "safe_switch_gate_selected_before_test": True,
            "safe_switch_alpha_selected_on_val": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "caveat": "This is a bounded continuous neural dynamics blend around the Stage37/teacher floor, not latent generative rollout and not SMC. Coordinates remain dataset-local raw-frame.",
    }
    write_json(REPORT_JSON, _jsonable(result))
    write_md(
        REPORT_MD,
        [
            "# Stage41 Bounded Neural Blend Dynamics",
            "",
            "- source: `fresh_run`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "- metric/seconds claim: `False`",
            f"- selected policy: `{selected_policy}`",
            f"- deployable: `{deployable}`",
            f"- non-fallback continuous neural contribution: `{result['non_fallback_continuous_neural_contribution']}`",
            f"- test metrics: `{metrics}`",
            f"- safe-switch policy: `{safe_switch_policy}`",
            f"- safe-switch deployable: `{safe_switch_deployable}`",
            f"- safe-switch non-fallback continuous neural contribution: `{result['safe_switch_non_fallback_continuous_neural_contribution']}`",
            f"- safe-switch test metrics: `{safe_switch_metrics}`",
            f"- no leakage: `{result['no_leakage']}`",
            "",
            "This evaluates a validation-selected bounded blend `floor + alpha * (neural - floor)` across rows, plus a second hypothesis constrained to the already validated repaired switch set. It is a conservative neural dynamics head, not Stage5C latent generation.",
        ],
    )
    return result


def main_bounded_neural_blend_dynamics() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_bounded_neural_blend_dynamics()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_bounded_neural_blend_dynamics",
            status,
            started,
            [OUT_DIR / "stage41_teacher_guided_proposal.json", OUT_DIR / "stage41_teacher_guided_proposal_repair.json"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_bounded_neural_blend_dynamics()
