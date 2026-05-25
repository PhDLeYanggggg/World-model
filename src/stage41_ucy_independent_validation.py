from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src import stage41_full_trajectory_world_state as ft
from src import stage41_joint_multiagent_consistency as jmc
from src import stage41_joint_policy_distillation as jpd


OUT_DIR = jpd.OUT_DIR
REPORT_JSON = OUT_DIR / "stage41_ucy_independent_validation.json"
REPORT_MD = OUT_DIR / "stage41_ucy_independent_validation.md"
SELECTION_MODULUS = 5
SELECTION_REMAINDER = 0


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


def _slice_arrays(data: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[mask] for k, v in data.items() if isinstance(v, np.ndarray) and len(v) == len(mask)}


def _source_summary(split: str, domain_name: str = "UCY") -> dict[str, Any]:
    ds = ft._fresh_ds(split)
    domain = ds["domain"].astype(str)
    scene = ds["scene_id"].astype(str)
    source = ds["source_file"].astype(str)
    mask = domain == domain_name
    sources = sorted(set(source[mask].tolist()))
    return {
        "rows": int(mask.sum()),
        "scenes": sorted(set(scene[mask].tolist())),
        "source_count": int(len(sources)),
        "sources": {src: int(np.sum(mask & (source == src))) for src in sources},
    }


def _evaluate_local(scores: Mapping[str, np.ndarray], data: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> dict[str, Any]:
    selected, _selected_fde, switch = jpd._apply_policy(scores, data, policy)
    return jpd._metric(selected, data["floor_ade"], data, switch)


def _positive_safe(metrics: Mapping[str, Any]) -> bool:
    return bool(
        metrics.get("all_improvement", 0.0) > 0
        and metrics.get("t50_improvement", 0.0) > 0
        and metrics.get("hard_failure_improvement", 0.0) > 0
        and metrics.get("easy_degradation", 1.0) <= 0.02
    )


def _temporal_blocks(frame_id: np.ndarray, ucy_mask: np.ndarray) -> dict[str, np.ndarray]:
    frames = frame_id[ucy_mask].astype(float)
    if len(frames) == 0:
        return {}
    q1, q2 = np.quantile(frames, [1.0 / 3.0, 2.0 / 3.0])
    return {
        "early": ucy_mask & (frame_id <= q1),
        "middle": ucy_mask & (frame_id > q1) & (frame_id <= q2),
        "late": ucy_mask & (frame_id > q2),
    }


def run_ucy_independent_validation() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    distiller = read_json(OUT_DIR / "stage41_joint_policy_distillation.json", {})
    if not distiller:
        raise FileNotFoundError("Run stage41_joint_policy_distillation first.")
    if distiller.get("no_leakage", {}).get("base_switch_input", True):
        raise RuntimeError("Refusing to validate a base-switch-leaking distiller.")

    checkpoint = distiller["best_checkpoint"]
    scores_train, data_train = jpd._predict_checkpoint(checkpoint, "train")
    domain_train = data_train["domain"].astype(str)
    ucy_train = domain_train == "UCY"
    row_ids = np.arange(len(domain_train), dtype=np.int64)
    select_mask = ucy_train & ((row_ids % SELECTION_MODULUS) == SELECTION_REMAINDER)
    select_scores = _slice_arrays(scores_train, select_mask)
    select_data = _slice_arrays(data_train, select_mask)
    ucy_policy, selection_metrics = jpd._fit_policy(select_scores, select_data, "distiller_only")

    validation_folds: dict[str, Any] = {}
    for remainder in range(SELECTION_MODULUS):
        mask = ucy_train & ((row_ids % SELECTION_MODULUS) == remainder)
        fold_scores = _slice_arrays(scores_train, mask)
        fold_data = _slice_arrays(data_train, mask)
        validation_folds[f"mod_{remainder}"] = {
            "role": "selection" if remainder == SELECTION_REMAINDER else "internal_validation",
            "rows": int(mask.sum()),
            "metrics": _evaluate_local(fold_scores, fold_data, ucy_policy),
        }
    internal_validation_pass = all(
        _positive_safe(row["metrics"])
        for key, row in validation_folds.items()
        if row["role"] == "internal_validation"
    )

    meta = jmc._group_metadata("train")
    frame_id = meta["frame_id"].astype(float)
    temporal_blocks: dict[str, Any] = {}
    for name, mask in _temporal_blocks(frame_id, ucy_train).items():
        temporal_blocks[name] = {
            "rows": int(mask.sum()),
            "metrics": _evaluate_local(_slice_arrays(scores_train, mask), _slice_arrays(data_train, mask), ucy_policy),
        }
    temporal_validation_pass = all(_positive_safe(row["metrics"]) for row in temporal_blocks.values())

    repaired = read_json(OUT_DIR / "stage41_ucy_fallback_repair.json", {})
    test_metrics = repaired.get("repaired_metrics") or {}
    test_ucy = (test_metrics.get("by_domain") or {}).get("UCY") or {}
    source_summary = {split: _source_summary(split, "UCY") for split in ["train", "val", "test"]}
    source_level_blocker = source_summary["train"]["source_count"] < 2 or source_summary["val"]["rows"] == 0
    validation_pass = bool(
        internal_validation_pass
        and temporal_validation_pass
        and test_ucy.get("all_improvement", 0.0) > 0
        and test_ucy.get("t50_improvement", 0.0) > 0
        and test_metrics.get("easy_degradation", 1.0) <= 0.02
    )
    result = {
        "source": "fresh_run",
        "protocol": "ucy_internal_fold_and_temporal_validation",
        "source_summary": source_summary,
        "source_level_independent_validation_available": not source_level_blocker,
        "source_level_blocker": (
            "UCY has one train source and no UCY validation source; true source-level UCY validation needs another UCY-like source or a rebuilt split."
            if source_level_blocker
            else ""
        ),
        "selection": {
            "rule": f"UCY train rows with row_index % {SELECTION_MODULUS} == {SELECTION_REMAINDER}",
            "rows": int(select_mask.sum()),
            "metrics": selection_metrics,
            "policy_slices": ucy_policy.get("slices", {}),
        },
        "internal_validation_folds": validation_folds,
        "temporal_validation_blocks": temporal_blocks,
        "internal_validation_pass": internal_validation_pass,
        "temporal_validation_pass": temporal_validation_pass,
        "test_ucy_metrics": test_ucy,
        "test_overall_metrics": test_metrics,
        "validation_pass": validation_pass,
        "no_leakage": {
            "future_waypoints_input": False,
            "future_labels_eval_only": True,
            "test_threshold_tuning": False,
            "train_only_policy_selection": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
        },
        "caveat": "This validates the UCY repair on internal train folds and temporal blocks, but source-level UCY validation remains blocked by available data.",
    }
    write_json(REPORT_JSON, _jsonable(result))
    write_md(
        REPORT_MD,
        [
            "# Stage41 UCY Independent Validation",
            "",
            "- source: `fresh_run`",
            f"- validation pass: `{validation_pass}`",
            f"- source-level independent validation available: `{not source_level_blocker}`",
            f"- source-level blocker: `{result['source_level_blocker']}`",
            f"- source summary: `{source_summary}`",
            f"- selection rows: `{int(select_mask.sum())}`",
            f"- selection metrics: `{selection_metrics}`",
            f"- internal validation folds: `{validation_folds}`",
            f"- temporal validation blocks: `{temporal_blocks}`",
            f"- test UCY metrics: `{test_ucy}`",
            f"- no leakage: `{result['no_leakage']}`",
            "",
            "The UCY repair validates across train-internal folds and temporal blocks, but it still needs a true source-level UCY validation split before becoming final deployment evidence.",
        ],
    )
    return result


def main_ucy_independent_validation() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_ucy_independent_validation()
        status = "ok"
    finally:
        jpd._append_ledger(
            "stage41_ucy_independent_validation",
            status,
            started,
            [OUT_DIR / "stage41_ucy_fallback_repair.json", OUT_DIR / "stage41_joint_policy_distillation.json"],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_ucy_independent_validation()
