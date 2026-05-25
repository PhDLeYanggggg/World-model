from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_bounded_neural_blend_dynamics as blend
from src import stage41_teacher_guided_multiseed as tgms
from src import stage41_teacher_guided_proposal as tgp


OUT_DIR = tgp.OUT_DIR
REPORT_JSON = OUT_DIR / "stage41_composite_tail_multiseed.json"
REPORT_MD = OUT_DIR / "stage41_composite_tail_multiseed.md"


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


def _seed_inputs(seed_row: Mapping[str, Any]) -> tuple[int, str, dict[str, Any], float]:
    seed = int(seed_row["seed"])
    checkpoint = str(((seed_row.get("train") or {}).get("checkpoint")) or "")
    policy = dict(seed_row.get("selected_policy") or {})
    min_sep = float(((seed_row.get("selected_guard") or {}).get("min_sep")) or 0.0)
    if not checkpoint or not policy:
        raise ValueError(f"missing checkpoint/policy for seed {seed}")
    return seed, checkpoint, policy, min_sep


def _metric_delta(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, float]:
    return {
        "all_delta": float(a.get("all_improvement", 0.0) - b.get("all_improvement", 0.0)),
        "t50_delta": float(a.get("t50_improvement", 0.0) - b.get("t50_improvement", 0.0)),
        "t100_delta": float(a.get("t100_improvement", 0.0) - b.get("t100_improvement", 0.0)),
        "hard_delta": float(a.get("hard_failure_improvement", 0.0) - b.get("hard_failure_improvement", 0.0)),
        "easy_delta": float(a.get("easy_degradation", 0.0) - b.get("easy_degradation", 0.0)),
    }


def _replicate_seed(seed_row: Mapping[str, Any]) -> dict[str, Any]:
    seed, checkpoint, teacher_policy, min_sep = _seed_inputs(seed_row)
    val = blend._bundle("val", checkpoint, teacher_policy, min_sep)
    selection = blend._select_safe_switch_policy(val, val)
    policy = (selection.get("selected") or {}).get("policy") or {}
    test = blend._bundle("test", checkpoint, teacher_policy, min_sep)
    test_eval = blend._evaluate_blend(test, policy)
    metrics = dict(test_eval["metrics"])
    teacher_metrics = dict(seed_row.get("test_metrics") or {})
    delta = _metric_delta(metrics, teacher_metrics)
    deployable = blend._eligible(metrics, blend.TEST_COLLISION_CEILING)
    positive_domains = tgms._positive_domains(metrics)
    return {
        "source": "fresh_run",
        "seed": seed,
        "checkpoint": checkpoint,
        "teacher_policy": teacher_policy,
        "teacher_guard_min_sep": min_sep,
        "composite_validation_selection": selection,
        "selected_policy": policy,
        "test_metrics": metrics,
        "test_floor_stats": test_eval["floor_stats"],
        "test_blend_stats": test_eval["blend_stats"],
        "teacher_repair_seed_metrics": teacher_metrics,
        "delta_vs_seed_teacher_repair": delta,
        "positive_external_domains": positive_domains,
        "deployable": bool(deployable),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_labels_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "policy_selected_on_val": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def _summarize(rows: Sequence[Mapping[str, Any]], path: Sequence[str]) -> dict[str, float]:
    vals = []
    for row in rows:
        cur: Any = row
        for key in path:
            cur = cur.get(key, {}) if isinstance(cur, Mapping) else {}
        vals.append(float(cur or 0.0))
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(arr.mean()) if len(arr) else 0.0,
        "std": float(arr.std(ddof=0)) if len(arr) else 0.0,
        "min": float(arr.min()) if len(arr) else 0.0,
        "max": float(arr.max()) if len(arr) else 0.0,
    }


def run_composite_tail_multiseed() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    teacher_multiseed = read_json(OUT_DIR / "stage41_teacher_guided_multiseed.json", {})
    seed_rows = teacher_multiseed.get("seed_results") or []
    if not seed_rows:
        raise FileNotFoundError("Run teacher-guided multiseed before composite-tail multiseed.")
    seed_results = [_replicate_seed(row) for row in seed_rows]
    metric_summary = {
        key: _summarize(seed_results, ["test_metrics", key])
        for key in [
            "all_improvement",
            "t50_improvement",
            "t100_improvement",
            "hard_failure_improvement",
            "easy_degradation",
            "alpha_mean",
            "switch_rate",
            "collision_delta_vs_floor_005",
        ]
    }
    delta_summary = {
        key: _summarize(seed_results, ["delta_vs_seed_teacher_repair", key])
        for key in ["all_delta", "t50_delta", "t100_delta", "hard_delta", "easy_delta"]
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
        and metric_summary["collision_delta_vs_floor_005"]["max"] <= blend.TEST_COLLISION_CEILING
        and min(positive_domain_counts or [0]) >= 2
    )
    strict_delta_vs_teacher_pass = bool(
        delta_summary["all_delta"]["min"] > 0
        and delta_summary["t50_delta"]["min"] > 0
        and delta_summary["t100_delta"]["min"] > 0
        and delta_summary["hard_delta"]["min"] > 0
        and delta_summary["easy_delta"]["max"] <= 0.0
    )
    result = {
        "source": "fresh_run",
        "protocol": "composite_tail_bounded_neural_multiseed",
        "seed_source": "cached_verified_teacher_guided_multiseed_checkpoints_re_evaluated_fresh",
        "seeds": [int(row["seed"]) for row in seed_results],
        "seed_results": seed_results,
        "metric_summary": metric_summary,
        "delta_vs_teacher_repair_summary": delta_summary,
        "positive_domain_counts": positive_domain_counts,
        "replication_pass": replication_pass,
        "strict_delta_vs_teacher_repair_pass": strict_delta_vs_teacher_pass,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_labels_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "policy_selected_on_val": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "caveat": "Each seed uses its own previously trained teacher-guided checkpoint and validation-selected guard, then selects composite-tail policy on that seed's validation split and evaluates test once. This remains dataset-local raw-frame 2.5D.",
    }
    write_json(REPORT_JSON, _jsonable(result))
    write_md(
        REPORT_MD,
        [
            "# Stage41 Composite-Tail Multi-Seed Evidence",
            "",
            "- source: `fresh_run`",
            f"- seed source: `{result['seed_source']}`",
            f"- seeds: `{result['seeds']}`",
            f"- replication pass: `{replication_pass}`",
            f"- strict delta vs teacher repair pass: `{strict_delta_vs_teacher_pass}`",
            f"- metric summary: `{metric_summary}`",
            f"- delta vs teacher repair summary: `{delta_summary}`",
            f"- positive domain counts: `{positive_domain_counts}`",
            f"- no leakage: `{result['no_leakage']}`",
            "",
            "Composite-tail is selected on validation for each seed-specific checkpoint and evaluated on test once. It is not Stage5C or SMC.",
        ],
    )
    return result


def main_composite_tail_multiseed() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_composite_tail_multiseed()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_composite_tail_multiseed",
            status,
            started,
            [OUT_DIR / "stage41_teacher_guided_multiseed.json", OUT_DIR / "stage41_composite_tail_evidence.json"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_composite_tail_multiseed()
