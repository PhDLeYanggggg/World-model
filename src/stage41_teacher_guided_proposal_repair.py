from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_joint_rollout_consistency as jrc
from src import stage41_teacher_guided_proposal as tgp


OUT_DIR = tgp.OUT_DIR
REPORT_JSON = OUT_DIR / "stage41_teacher_guided_proposal_repair.json"
REPORT_MD = OUT_DIR / "stage41_teacher_guided_proposal_repair.md"
VAL_COLLISION_CEILING = 0.007
TEST_COLLISION_CEILING = 0.01


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


def _score(metrics: Mapping[str, Any], collision_delta: float) -> float:
    return (
        float(metrics.get("all_improvement", 0.0))
        + 1.25 * float(metrics.get("t50_improvement", 0.0))
        + 1.0 * float(metrics.get("t100_improvement", 0.0))
        + 1.15 * float(metrics.get("hard_failure_improvement", 0.0))
        - 35.0 * max(0.0, float(metrics.get("easy_degradation", 1.0)) - 0.02)
        - 12.0 * max(0.0, collision_delta - VAL_COLLISION_CEILING)
    )


def _load_selected() -> tuple[str, dict[str, Any]]:
    report = read_json(OUT_DIR / "stage41_teacher_guided_proposal.json", {})
    selected = str(report.get("selected_trial"))
    trial = ((report.get("trial_reports") or {}).get(selected) or {})
    ckpt = ((trial.get("train") or {}).get("checkpoint"))
    policy = report.get("selected_policy") or {}
    if not ckpt or not policy:
        raise FileNotFoundError("Run stage41_teacher_guided_proposal before repair.")
    return str(ckpt), dict(policy)


def _bundle_for_split(checkpoint: str | Path, policy: Mapping[str, Any], split: str) -> dict[str, Any]:
    data = tgp._bundle(split)
    pred = tgp._predict(checkpoint, data)
    raw_switch = tgp._policy_switch(pred, policy)
    return {"data": data, "pred": pred, "raw_switch": raw_switch}


def _eval(data: Mapping[str, Any], switch: np.ndarray, name: str) -> dict[str, Any]:
    return tgp._evaluate_switch(data, switch, name)


def _select_guard(checkpoint: str | Path, policy: Mapping[str, Any]) -> dict[str, Any]:
    val = _bundle_for_split(checkpoint, policy, "val")
    candidates: list[dict[str, Any]] = []
    for min_sep in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.16]:
        guarded, guarded_off = jrc._apply_proximity_guard(
            val["data"]["floor_xy"],
            val["data"]["neural_xy"],
            val["data"]["labels"],
            val["data"]["keys"],
            val["raw_switch"],
            min_sep,
        )
        ev = _eval(val["data"], guarded, f"val_teacher_repair_{min_sep}")
        metrics = ev["selected_metrics"]
        eligible = bool(
            metrics.get("all_improvement", 0.0) > 0
            and metrics.get("t50_improvement", 0.0) > 0
            and metrics.get("t100_improvement", 0.0) > 0
            and metrics.get("hard_failure_improvement", 0.0) > 0
            and metrics.get("easy_degradation", 1.0) <= 0.02
            and ev["collision_delta_005"] <= VAL_COLLISION_CEILING
            and float(np.mean(guarded)) > 0
        )
        candidates.append(
            {
                "min_sep": min_sep,
                "guarded_off": guarded_off,
                "metrics": metrics,
                "collision_delta_005": ev["collision_delta_005"],
                "switch_rate": float(np.mean(guarded)),
                "eligible": eligible,
                "score": _score(metrics, ev["collision_delta_005"]),
            }
        )
    pool = [row for row in candidates if row["eligible"]] or candidates
    best = max(pool, key=lambda row: row["score"])
    return {"selected": best, "candidates": candidates}


def run_teacher_guided_proposal_repair() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    checkpoint, policy = _load_selected()
    guard = _select_guard(checkpoint, policy)
    test = _bundle_for_split(checkpoint, policy, "test")
    raw_eval = _eval(test["data"], test["raw_switch"], "raw_teacher_guided_proposal")
    guarded_switch, guarded_off = jrc._apply_proximity_guard(
        test["data"]["floor_xy"],
        test["data"]["neural_xy"],
        test["data"]["labels"],
        test["data"]["keys"],
        test["raw_switch"],
        float((guard.get("selected") or {}).get("min_sep", 0.0)),
    )
    repaired_eval = _eval(test["data"], guarded_switch, "repaired_teacher_guided_proposal")
    group_repair = read_json(OUT_DIR / "stage41_group_consistency_multiseed_repair.json", {})
    group_summary = group_repair.get("metric_summary") or {}
    group_basis = {
        "all_improvement": (group_summary.get("all_improvement") or {}).get("mean", 0.0),
        "t50_improvement": (group_summary.get("t50_improvement") or {}).get("mean", 0.0),
        "t100_improvement": (group_summary.get("t100_improvement") or {}).get("mean", 0.0),
        "hard_failure_improvement": (group_summary.get("hard_failure_improvement") or {}).get("mean", 0.0),
        "easy_degradation": (group_summary.get("easy_degradation") or {}).get("max", 1.0),
        "collision_delta_vs_floor_005": (group_summary.get("collision_delta_vs_floor_005") or {}).get("max", 1.0),
    }
    metrics = repaired_eval["selected_metrics"]
    lift = {
        "all_delta": float(metrics.get("all_improvement", 0.0) - float(group_basis.get("all_improvement") or 0.0)),
        "t50_delta": float(metrics.get("t50_improvement", 0.0) - float(group_basis.get("t50_improvement") or 0.0)),
        "t100_delta": float(metrics.get("t100_improvement", 0.0) - float(group_basis.get("t100_improvement") or 0.0)),
        "hard_delta": float(metrics.get("hard_failure_improvement", 0.0) - float(group_basis.get("hard_failure_improvement") or 0.0)),
        "easy_delta": float(metrics.get("easy_degradation", 0.0) - float(group_basis.get("easy_degradation") or 0.0)),
    }
    deployable = bool(
        metrics.get("all_improvement", 0.0) > 0
        and metrics.get("t50_improvement", 0.0) > 0
        and metrics.get("t100_improvement", 0.0) > 0
        and metrics.get("hard_failure_improvement", 0.0) > 0
        and metrics.get("easy_degradation", 1.0) <= 0.02
        and repaired_eval["collision_delta_005"] <= TEST_COLLISION_CEILING
        and float(np.mean(guarded_switch)) > 0
    )
    improves_current = bool(deployable and lift["all_delta"] > 0 and lift["t50_delta"] > 0 and lift["hard_delta"] > 0)
    result = {
        "source": "fresh_run",
        "protocol_status": "teacher_guided_proposal_proximity_repair",
        "checkpoint": checkpoint,
        "base_policy": policy,
        "validation_guard": guard,
        "test_guarded_off": int(guarded_off),
        "raw_test_metrics": raw_eval["selected_metrics"],
        "raw_collision_delta_005": raw_eval["collision_delta_005"],
        "test_metrics": metrics,
        "multi_agent_metrics": repaired_eval["multi_agent_metrics"],
        "collision_delta_vs_floor_005": repaired_eval["collision_delta_005"],
        "switch_rate": float(np.mean(guarded_switch)),
        "current_best_group_consistency_basis": group_basis,
        "lift_over_current_group_consistency_basis": lift,
        "teacher_guided_proposal_repair_deployable": deployable,
        "teacher_guided_proposal_repair_improves_current_deployable": improves_current,
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
        "caveat": "This repairs a teacher-guided neural proposal with a validation-selected proximity guard. It remains dataset-local raw-frame 2.5D and is not Stage5C/SMC.",
    }
    write_json(REPORT_JSON, _jsonable(result))
    write_md(
        REPORT_MD,
        [
            "# Stage41 Teacher-Guided Proposal Safety Repair",
            "",
            "- source: `fresh_run`",
            f"- selected guard: `{guard.get('selected')}`",
            f"- deployable: `{deployable}`",
            f"- improves current deployable: `{improves_current}`",
            f"- raw test metrics: `{raw_eval['selected_metrics']}`",
            f"- raw collision delta @0.05: `{raw_eval['collision_delta_005']}`",
            f"- repaired test metrics: `{metrics}`",
            f"- repaired collision delta @0.05: `{repaired_eval['collision_delta_005']}`",
            f"- switch rate: `{float(np.mean(guarded_switch))}`",
            f"- lift over current group-consistency basis: `{lift}`",
            f"- no leakage: `{result['no_leakage']}`",
            "",
            "The neural proposal is selected on validation and then repaired with a validation-selected proximity guard. Test is evaluated once.",
        ],
    )
    return result


def main_teacher_guided_proposal_repair() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_teacher_guided_proposal_repair()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_teacher_guided_proposal_repair",
            status,
            started,
            [OUT_DIR / "stage41_teacher_guided_proposal.json"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_teacher_guided_proposal_repair()
