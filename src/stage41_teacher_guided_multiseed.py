from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_group_consistency_multiseed as gcms
from src import stage41_joint_rollout_consistency as jrc
from src import stage41_teacher_guided_proposal as tgp
from src import stage41_teacher_guided_proposal_repair as repair


OUT_DIR = tgp.OUT_DIR
REPORT_JSON = OUT_DIR / "stage41_teacher_guided_multiseed.json"
REPORT_MD = OUT_DIR / "stage41_teacher_guided_multiseed.md"
SEEDS = [11, 17, 23]
VAL_COLLISION_CEILING = repair.VAL_COLLISION_CEILING
TEST_COLLISION_CEILING = repair.TEST_COLLISION_CEILING


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


def _trial_for_seed(seed: int) -> dict[str, Any]:
    return {
        "name": f"teacher_proposal_balanced_seed{seed}",
        "width": 96,
        "dropout": 0.08,
        "lr": 8e-4,
        "teacher_w": 1.2,
        "gain_w": 0.6,
        "harm_w": 1.0,
        "hard_w": 1.5,
        "seed": int(seed),
    }


def _positive_domains(metrics: Mapping[str, Any]) -> int:
    return sum(
        1
        for row in (metrics.get("by_domain") or {}).values()
        if row.get("all_improvement", 0.0) > 0
        or row.get("t50_improvement", 0.0) > 0
        or row.get("hard_failure_improvement", 0.0) > 0
    )


def _candidate_eligible(metrics: Mapping[str, Any], collision_delta: float, switch_rate: float, ceiling: float) -> bool:
    return bool(
        metrics.get("all_improvement", 0.0) > 0
        and metrics.get("t50_improvement", 0.0) > 0
        and metrics.get("t100_improvement", 0.0) > 0
        and metrics.get("hard_failure_improvement", 0.0) > 0
        and metrics.get("easy_degradation", 1.0) <= 0.02
        and collision_delta <= ceiling
        and switch_rate > 0.0
    )


def _guard_score(metrics: Mapping[str, Any], collision_delta: float, switch_rate: float, ceiling: float) -> float:
    return (
        float(metrics.get("all_improvement", 0.0))
        + 1.25 * float(metrics.get("t50_improvement", 0.0))
        + float(metrics.get("t100_improvement", 0.0))
        + 1.15 * float(metrics.get("hard_failure_improvement", 0.0))
        - 35.0 * max(0.0, float(metrics.get("easy_degradation", 1.0)) - 0.02)
        - 12.0 * max(0.0, collision_delta - ceiling)
        - 0.10 * switch_rate
    )


def _select_guard(data: Mapping[str, Any], raw_switch: np.ndarray) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for min_sep in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.16]:
        guarded, guarded_off = jrc._apply_proximity_guard(
            data["floor_xy"],
            data["neural_xy"],
            data["labels"],
            data["keys"],
            raw_switch.astype(bool),
            min_sep,
        )
        ev = tgp._evaluate_switch(data, guarded, f"val_teacher_multiseed_guard_{min_sep}")
        metrics = ev["selected_metrics"]
        switch_rate = float(np.mean(guarded))
        collision_delta = float(ev["collision_delta_005"])
        eligible = _candidate_eligible(metrics, collision_delta, switch_rate, VAL_COLLISION_CEILING)
        candidates.append(
            {
                "min_sep": min_sep,
                "guarded_off": int(guarded_off),
                "metrics": metrics,
                "collision_delta_005": collision_delta,
                "switch_rate": switch_rate,
                "eligible": eligible,
                "score": _guard_score(metrics, collision_delta, switch_rate, VAL_COLLISION_CEILING),
            }
        )
    pool = [row for row in candidates if row["eligible"]] or candidates
    return {"selected": max(pool, key=lambda row: row["score"]), "candidates": candidates}


def _replicate_seed(seed: int, train: Mapping[str, Any], val: Mapping[str, Any], test: Mapping[str, Any]) -> dict[str, Any]:
    trained = tgp._train_trial(_trial_for_seed(seed), train, val)
    pred_val = tgp._predict(trained["checkpoint"], val)
    policy, val_candidates = tgp._fit_policy(pred_val, val)
    raw_val_switch = tgp._policy_switch(pred_val, policy)
    guard = _select_guard(val, raw_val_switch)
    pred_test = tgp._predict(trained["checkpoint"], test)
    raw_test_switch = tgp._policy_switch(pred_test, policy)
    raw_eval = tgp._evaluate_switch(test, raw_test_switch, f"teacher_multiseed_raw_seed{seed}")
    guarded_switch, guarded_off = jrc._apply_proximity_guard(
        test["floor_xy"],
        test["neural_xy"],
        test["labels"],
        test["keys"],
        raw_test_switch,
        float((guard.get("selected") or {}).get("min_sep", 0.0)),
    )
    repaired_eval = tgp._evaluate_switch(test, guarded_switch, f"teacher_multiseed_repaired_seed{seed}")
    metrics = dict(repaired_eval["selected_metrics"])
    switch_rate = float(np.mean(guarded_switch))
    collision_delta = float(repaired_eval["collision_delta_005"])
    metrics["collision_delta_vs_floor_005"] = collision_delta
    deployable = _candidate_eligible(metrics, collision_delta, switch_rate, TEST_COLLISION_CEILING)
    return {
        "source": "fresh_run",
        "seed": int(seed),
        "train": trained,
        "selected_policy": policy,
        "val_candidates_count": len(val_candidates),
        "selected_guard": guard.get("selected"),
        "test_guarded_off": int(guarded_off),
        "raw_test_metrics": raw_eval["selected_metrics"],
        "raw_collision_delta_vs_floor_005": raw_eval["collision_delta_005"],
        "test_metrics": metrics,
        "multi_agent_metrics": repaired_eval["multi_agent_metrics"],
        "positive_external_domains": _positive_domains(metrics),
        "deployable": deployable,
        "no_leakage": {
            "teacher_switch_inference_input": False,
            "teacher_switch_train_label_only": True,
            "future_waypoints_input": False,
            "future_waypoints_label_eval_only": True,
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "policy_selected_on_val": True,
            "proximity_guard_selected_on_val": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def _summarize_metric(rows: list[Mapping[str, Any]], key: str) -> dict[str, float]:
    vals = np.asarray([float(row["test_metrics"].get(key, 0.0)) for row in rows], dtype=np.float64)
    return {
        "mean": float(vals.mean()) if len(vals) else 0.0,
        "std": float(vals.std(ddof=0)) if len(vals) else 0.0,
        "min": float(vals.min()) if len(vals) else 0.0,
        "max": float(vals.max()) if len(vals) else 0.0,
    }


def run_teacher_guided_multiseed() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    train = tgp._bundle("train")
    val = tgp._bundle("val")
    test = tgp._bundle("test")
    seed_results = [_replicate_seed(seed, train, val, test) for seed in SEEDS]
    metric_summary = {
        key: _summarize_metric(seed_results, key)
        for key in [
            "all_improvement",
            "t50_improvement",
            "t100_improvement",
            "hard_failure_improvement",
            "easy_degradation",
            "switch_rate",
            "collision_delta_vs_floor_005",
        ]
    }
    positive_domain_counts = [int(row["positive_external_domains"]) for row in seed_results]
    replication_pass = bool(
        seed_results
        and all(row.get("deployable") for row in seed_results)
        and metric_summary["all_improvement"]["min"] > 0
        and metric_summary["t50_improvement"]["min"] > 0
        and metric_summary["t100_improvement"]["min"] > 0
        and metric_summary["hard_failure_improvement"]["min"] > 0
        and metric_summary["easy_degradation"]["max"] <= 0.02
        and metric_summary["collision_delta_vs_floor_005"]["max"] <= TEST_COLLISION_CEILING
        and min(positive_domain_counts or [0]) >= 2
    )
    result = {
        "source": "fresh_run",
        "protocol": "teacher_guided_proposal_multiseed_safety_repair",
        "seeds": SEEDS,
        "validation_collision_ceiling": VAL_COLLISION_CEILING,
        "test_collision_ceiling": TEST_COLLISION_CEILING,
        "seed_results": seed_results,
        "metric_summary": metric_summary,
        "positive_domain_counts": positive_domain_counts,
        "replication_pass": replication_pass,
        "no_leakage": {
            "teacher_switch_inference_input": False,
            "teacher_switch_train_label_only": True,
            "future_waypoints_input": False,
            "future_waypoints_label_eval_only": True,
            "future_endpoint_input": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "policy_selected_on_val": True,
            "proximity_guard_selected_on_val": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "caveat": "This is multi-seed replication of a teacher-guided neural proposal under Stage37 safety fallback. It remains dataset-local raw-frame 2.5D and does not execute Stage5C or SMC.",
    }
    write_json(REPORT_JSON, _jsonable(result))
    write_md(
        REPORT_MD,
        [
            "# Stage41 Teacher-Guided Proposal Multi-Seed Replication",
            "",
            "- source: `fresh_run`",
            f"- seeds: `{SEEDS}`",
            f"- validation collision ceiling: `{VAL_COLLISION_CEILING}`",
            f"- test collision ceiling: `{TEST_COLLISION_CEILING}`",
            f"- replication pass: `{replication_pass}`",
            f"- metric summary: `{metric_summary}`",
            f"- positive domain counts: `{positive_domain_counts}`",
            f"- no leakage: `{result['no_leakage']}`",
            "",
            "Each seed is trained fresh, selects policy and proximity guard on validation, and evaluates test once. Future waypoints are labels/eval only.",
        ],
    )
    return result


def main_teacher_guided_multiseed() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_teacher_guided_multiseed()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_teacher_guided_multiseed",
            status,
            started,
            [OUT_DIR / "stage41_teacher_guided_proposal_repair.json", OUT_DIR / "stage41_teacher_guided_evidence.json"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_teacher_guided_multiseed()
