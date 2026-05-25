from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_group_consistency_distiller as gcd
from src.stage41_group_consistency_multiseed import _jsonable, _positive_domains, _summarize_metric


OUT_DIR = gcd.OUT_DIR
REPORT_JSON = OUT_DIR / "stage41_group_consistency_multiseed_repair.json"
REPORT_MD = OUT_DIR / "stage41_group_consistency_multiseed_repair.md"
INPUT_JSON = OUT_DIR / "stage41_group_consistency_multiseed.json"
VAL_COLLISION_CEILING = 0.005
TEST_COLLISION_CEILING = 0.01


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


def _candidate_eligible(metrics: Mapping[str, Any], switch_rate: float, collision_ceiling: float) -> bool:
    return bool(
        metrics.get("all_improvement", 0.0) > 0
        and metrics.get("t50_improvement", 0.0) > 0
        and metrics.get("t100_improvement", 0.0) > 0
        and metrics.get("hard_failure_improvement", 0.0) > 0
        and metrics.get("easy_degradation", 1.0) <= 0.02
        and metrics.get("collision_delta_vs_floor_005", 1.0) <= collision_ceiling
        and switch_rate > 0.0
    )


def _safety_buffer_score(candidate: Mapping[str, Any], collision_ceiling: float) -> float:
    metrics = candidate["metrics"]
    switch_rate = float(candidate.get("switch_rate", 0.0))
    collision_delta = float(metrics.get("collision_delta_vs_floor_005", 1.0))
    return (
        gcd._score(metrics)
        - 20.0 * max(0.0, collision_delta - collision_ceiling)
        - 0.15 * switch_rate
    )


def _fit_safety_buffer_policy(scores: Mapping[str, np.ndarray], val: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _, candidates = gcd._fit_policy(scores, val)
    eligible = [
        row
        for row in candidates
        if _candidate_eligible(row["metrics"], float(row.get("switch_rate", 0.0)), VAL_COLLISION_CEILING)
    ]
    pool = eligible or candidates
    best = max(pool, key=lambda row: _safety_buffer_score(row, VAL_COLLISION_CEILING))
    return {
        "type": "group_consistency_distiller_safety_buffer",
        **best["policy"],
        "val_eligible": bool(eligible),
        "val_collision_ceiling": VAL_COLLISION_CEILING,
        "val_selected_metrics": best["metrics"],
    }, candidates


def _deployable(metrics: Mapping[str, Any], switch_rate: float) -> bool:
    return _candidate_eligible(metrics, switch_rate, TEST_COLLISION_CEILING)


def _repair_seed(seed_row: Mapping[str, Any], val: Mapping[str, np.ndarray], test: Mapping[str, np.ndarray]) -> dict[str, Any]:
    checkpoint = seed_row["train"]["checkpoint"]
    scores_val = gcd._predict(checkpoint, val)
    policy, candidates = _fit_safety_buffer_policy(scores_val, val)
    scores_test = gcd._predict(checkpoint, test)
    test_metrics, test_switch = gcd._policy_metrics(scores_test, test, policy)
    switch_rate = float(np.mean(test_switch))
    return {
        "seed": int(seed_row["seed"]),
        "checkpoint": checkpoint,
        "previous_deployable": bool(seed_row.get("deployable")),
        "previous_policy": seed_row.get("selected_policy"),
        "previous_test_metrics": seed_row.get("test_metrics"),
        "selected_policy": policy,
        "val_candidates_count": len(candidates),
        "test_metrics": test_metrics,
        "positive_external_domains": _positive_domains(test_metrics),
        "deployable": _deployable(test_metrics, switch_rate),
        "no_leakage": {
            "future_waypoints_input": False,
            "future_labels_eval_only": True,
            "train_gain_safe_unsafe_labels_only": True,
            "test_threshold_tuning": False,
            "policy_selected_on_val": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def run_group_consistency_multiseed_repair() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"Missing prior multiseed report: {INPUT_JSON}")
    prior = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    checkpoint, repaired_policy, policy_source = gcd._load_policy_and_checkpoint()
    val = gcd._bundle("val", checkpoint, repaired_policy)
    test = gcd._bundle("test", checkpoint, repaired_policy)
    seed_results = [_repair_seed(row, val, test) for row in prior.get("seed_results", [])]
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
        "protocol": "group_consistency_distiller_multiseed_safety_buffer_repair",
        "policy_source": policy_source,
        "prior_replication_pass": bool(prior.get("replication_pass")),
        "repair_reason": "Prior multi-seed replication had stable positive FDE gains but one seed exceeded the near-proximity safety delta by a small margin.",
        "validation_collision_ceiling": VAL_COLLISION_CEILING,
        "test_collision_ceiling": TEST_COLLISION_CEILING,
        "seeds": [int(row["seed"]) for row in seed_results],
        "seed_results": seed_results,
        "metric_summary": metric_summary,
        "positive_domain_counts": positive_domain_counts,
        "replication_pass": replication_pass,
        "no_leakage": {
            "future_waypoints_input": False,
            "future_labels_eval_only": True,
            "train_gain_safe_unsafe_labels_only": True,
            "test_threshold_tuning": False,
            "policy_selected_on_val": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "caveat": "This repairs deployment policy selection only. It does not execute Stage5C or SMC, and remains dataset-local raw-frame 2.5D.",
    }
    write_json(REPORT_JSON, _jsonable(result))
    write_md(
        REPORT_MD,
        [
            "# Stage41 Group Consistency Multi-Seed Safety-Buffer Repair",
            "",
            "- source: `fresh_run`",
            f"- prior replication pass: `{result['prior_replication_pass']}`",
            f"- validation collision ceiling: `{VAL_COLLISION_CEILING}`",
            f"- test collision ceiling: `{TEST_COLLISION_CEILING}`",
            f"- replication pass after repair: `{replication_pass}`",
            f"- metric summary: `{metric_summary}`",
            f"- positive domain counts: `{positive_domain_counts}`",
            f"- no leakage: `{result['no_leakage']}`",
            "",
            "This is a validation-selected conservative deployment repair over already-trained seed checkpoints. Test thresholds are not tuned on test; future labels remain label/eval only.",
        ],
    )
    return result


def main_group_consistency_multiseed_repair() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_group_consistency_multiseed_repair()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_group_consistency_multiseed_repair",
            status,
            started,
            [INPUT_JSON, OUT_DIR / "stage41_group_consistency_distiller.json"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_group_consistency_multiseed_repair()
