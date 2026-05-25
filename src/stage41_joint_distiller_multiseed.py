from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, write_json, write_md
from src import stage41_joint_policy_distillation as jpd


OUT_DIR = jpd.OUT_DIR
REPORT_JSON = OUT_DIR / "stage41_joint_policy_distillation_multiseed.json"
REPORT_MD = OUT_DIR / "stage41_joint_policy_distillation_multiseed.md"
SEEDS = [11, 17, 23]


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


def _summarize_metric(rows: list[Mapping[str, Any]], key: str) -> dict[str, float]:
    vals = np.asarray([float(row["test_metrics"].get(key, 0.0)) for row in rows], dtype=np.float64)
    return {
        "mean": float(vals.mean()) if len(vals) else 0.0,
        "std": float(vals.std(ddof=0)) if len(vals) else 0.0,
        "min": float(vals.min()) if len(vals) else 0.0,
        "max": float(vals.max()) if len(vals) else 0.0,
    }


def _positive_domains(metrics: Mapping[str, Any]) -> int:
    return sum(
        1
        for row in (metrics.get("by_domain") or {}).values()
        if row.get("all_improvement", 0.0) > 0
        or row.get("t50_improvement", 0.0) > 0
        or row.get("hard_failure_improvement", 0.0) > 0
    )


def _replicate_trial(seed: int) -> dict[str, Any]:
    trial = {
        "name": f"joint_distill_nobase_balanced_seed{seed}",
        "width": 96,
        "dropout": 0.08,
        "lr": 8e-4,
        "gain_w": 1.0,
        "switch_w": 1.0,
        "harm_w": 1.2,
        "hard_w": 1.5,
        "seed": int(seed),
    }
    trained = jpd._train_trial(trial)
    scores_val, data_val = jpd._predict_checkpoint(trained["checkpoint"], "val")
    policy, val_metrics = jpd._fit_policy(scores_val, data_val, "distiller_only")
    scores_test, data_test = jpd._predict_checkpoint(trained["checkpoint"], "test")
    test_metrics = jpd._evaluate(scores_test, data_test, policy)
    return {
        "seed": int(seed),
        "source": trained.get("source"),
        "checkpoint": trained.get("checkpoint"),
        "heartbeat": trained.get("heartbeat"),
        "policy": policy,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "positive_external_domains": _positive_domains(test_metrics),
        "no_leakage": {
            "base_switch_input": False,
            "future_waypoints_input": False,
            "future_labels_eval_only": True,
            "test_threshold_tuning": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
    }


def run_joint_distiller_multiseed() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    seed_results = [_replicate_trial(seed) for seed in SEEDS]
    metric_summary = {
        key: _summarize_metric(seed_results, key)
        for key in [
            "all_improvement",
            "t50_improvement",
            "t100_improvement",
            "hard_failure_improvement",
            "easy_degradation",
            "switch_rate",
        ]
    }
    positive_domain_counts = [int(row["positive_external_domains"]) for row in seed_results]
    replication_pass = bool(
        metric_summary["all_improvement"]["min"] > 0
        and metric_summary["t50_improvement"]["min"] > 0
        and metric_summary["hard_failure_improvement"]["min"] > 0
        and metric_summary["easy_degradation"]["max"] <= 0.02
        and min(positive_domain_counts or [0]) >= 2
    )
    result = {
        "source": "fresh_run",
        "protocol": "no_base_switch_joint_distiller_multiseed",
        "seeds": SEEDS,
        "seed_results": seed_results,
        "metric_summary": metric_summary,
        "positive_domain_counts": positive_domain_counts,
        "replication_pass": replication_pass,
        "no_leakage": {
            "base_switch_input": False,
            "future_waypoints_input": False,
            "future_labels_eval_only": True,
            "test_threshold_tuning": False,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
        "caveat": "This is multi-seed replication of the no-base-switch distiller. It still does not fix UCY fallback-only behavior or make metric/seconds/foundation claims.",
    }
    write_json(REPORT_JSON, _jsonable(result))
    write_md(
        REPORT_MD,
        [
            "# Stage41 Joint Distiller Multi-Seed Replication",
            "",
            "- source: `fresh_run`",
            f"- seeds: `{SEEDS}`",
            f"- replication pass: `{replication_pass}`",
            f"- metric summary: `{metric_summary}`",
            f"- positive domain counts: `{positive_domain_counts}`",
            f"- no leakage: `{result['no_leakage']}`",
            "",
            "UCY remains fallback-only in the main candidate; this replication checks seed stability, not UCY repair.",
        ],
    )
    return result


def main_joint_distiller_multiseed() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_joint_distiller_multiseed()
        status = "ok"
    finally:
        jpd._append_ledger(
            "stage41_joint_distiller_multiseed",
            status,
            started,
            [OUT_DIR / "stage41_joint_policy_distillation.json"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_joint_distiller_multiseed()
