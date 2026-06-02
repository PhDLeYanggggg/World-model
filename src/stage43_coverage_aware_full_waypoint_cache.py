from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_source_level_full_waypoint_eval as waypoint_eval
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_full_waypoint_latent_dynamics import (
    OUT_DIR,
    README_RESULTS,
    RESEARCH_STATE,
    WORK_SUMMARY,
    M3W_README,
    _git_commit,
    _jsonable,
)
from src.stage43_full_waypoint_supervision_cache import _load_old_split, _row_hash, _sha256_file
from src.stage43_source_family_coverage_split_repair import REPORT_JSON as STAGE43_CE_JSON


CACHE_DIR = Path("data/stage43_ce_full_waypoint_supervision_cache")
REPORT_JSON = OUT_DIR / "stage43_coverage_aware_full_waypoint_cache.json"
REPORT_MD = OUT_DIR / "stage43_coverage_aware_full_waypoint_cache.md"
GATE_MD = OUT_DIR / "stage43_stage_cf_coverage_aware_full_waypoint_cache_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_CF_COVERAGE_AWARE_FULL_WAYPOINT_CACHE"
SOURCE = "fresh_stage43_cf_coverage_aware_full_waypoint_cache"
OLD_SPLITS = ["train", "val", "test"]
NEW_SPLITS = ["train", "val", "test"]


def _concat_for_split(assignments: Mapping[str, str], new_split: str) -> dict[str, np.ndarray]:
    rows: dict[str, list[np.ndarray]] = {
        "old_split": [],
        "local_row": [],
        "dataset": [],
        "scene_id": [],
        "source_file": [],
        "agent_id": [],
        "frame_id": [],
        "current_x": [],
        "current_y": [],
        "future_endpoint_x": [],
        "future_endpoint_y": [],
        "horizon": [],
        "dt_frame_step": [],
        "track_length": [],
        "valid_mask": [],
        "scale": [],
        "hard": [],
        "failure": [],
        "easy": [],
    }
    for old_split in OLD_SPLITS:
        geo, labels = _load_old_split(old_split)
        sources = geo["source_file"].astype(str)
        mask = np.asarray([assignments[str(source)] == new_split for source in sources], dtype=bool)
        ids = np.where(mask)[0]
        if len(ids) == 0:
            continue
        rows["old_split"].append(np.asarray([old_split] * len(ids)))
        rows["local_row"].append(ids.astype(np.int64))
        for key in [
            "dataset",
            "scene_id",
            "source_file",
            "agent_id",
            "frame_id",
            "current_x",
            "current_y",
            "future_endpoint_x",
            "future_endpoint_y",
            "horizon",
            "dt_frame_step",
            "track_length",
            "valid_mask",
        ]:
            rows[key].append(geo[key][ids])
        for key in ["scale", "hard", "failure", "easy"]:
            rows[key].append(labels[key][ids])
    if not rows["horizon"]:
        raise ValueError(f"No rows found for repaired split {new_split}")
    return {key: np.concatenate(value, axis=0) for key, value in rows.items()}


def _cache_path(split: str) -> Path:
    return CACHE_DIR / f"stage43_ce_full_waypoint_supervision_{split}.npz"


def _write_split_cache(split: str, data: Mapping[str, np.ndarray]) -> dict[str, Any]:
    labels = waypoint_eval._reconstruct_waypoint_labels(data)
    current_xy = np.stack([data["current_x"], data["current_y"]], axis=1).astype(np.float32)
    future_xy = np.stack([data["future_endpoint_x"], data["future_endpoint_y"]], axis=1).astype(np.float32)
    endpoint_diff = np.linalg.norm(labels["waypoint_xy"][:, -1, :].astype(np.float64) - future_xy.astype(np.float64), axis=1)
    valid_last = labels["waypoint_valid"][:, -1].astype(bool)
    max_endpoint_diff = float(np.max(endpoint_diff[valid_last])) if int(np.sum(valid_last)) else 0.0
    path = _cache_path(split)
    np.savez_compressed(
        path,
        old_split=data["old_split"].astype(str),
        local_row=data["local_row"].astype(np.int64),
        dataset=data["dataset"].astype(str),
        scene_id=data["scene_id"].astype(str),
        source_file=data["source_file"].astype(str),
        agent_id=data["agent_id"].astype(np.int64),
        frame_id=data["frame_id"].astype(np.float64),
        horizon=data["horizon"].astype(np.int16),
        current_xy=current_xy,
        future_xy=future_xy,
        waypoint_xy=labels["waypoint_xy"].astype(np.float32),
        waypoint_valid=labels["waypoint_valid"].astype(bool),
        missing_track=labels["missing_track"].astype(bool),
        scale=data["scale"].astype(np.float32),
        hard=data["hard"].astype(bool),
        failure=data["failure"].astype(bool),
        easy=data["easy"].astype(bool),
        valid_mask=data["valid_mask"].astype(bool),
        dt_frame_step=data["dt_frame_step"].astype(np.float32),
        track_length=data["track_length"].astype(np.float32),
    )
    horizon = data["horizon"].astype(int)
    domain = data["dataset"].astype(str)
    scene = data["scene_id"].astype(str)
    source = data["source_file"].astype(str)
    return {
        "split": split,
        "cache_path": str(path),
        "cache_sha256": _sha256_file(path),
        "row_hash": _row_hash(data),
        "rows": int(len(horizon)),
        "domains": {str(k): int(v) for k, v in zip(*np.unique(domain, return_counts=True))},
        "scene_count": int(len(set(scene.tolist()))),
        "source_count": int(len(set(source.tolist()))),
        "horizon_counts": {str(int(k)): int(v) for k, v in sorted(Counter(horizon.tolist()).items())},
        "full_waypoint_rows": int(np.sum(np.any(labels["waypoint_valid"], axis=1))),
        "all_waypoints_valid_rows": int(np.sum(np.all(labels["waypoint_valid"], axis=1))),
        "missing_track_rows": int(np.sum(labels["missing_track"])),
        "max_endpoint_diff_last_waypoint": max_endpoint_diff,
        "hard_rows": int(np.sum(data["hard"].astype(bool))),
        "failure_rows": int(np.sum(data["failure"].astype(bool))),
        "easy_rows": int(np.sum(data["easy"].astype(bool))),
    }


def _source_sets(split_arrays: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, set[str]]:
    return {split: set(arr["source_file"].astype(str).tolist()) for split, arr in split_arrays.items()}


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summaries = payload["split_summaries"]
    expected = payload["stage43_ce_precondition"].get("split_rows", {})
    leakage = payload["no_leakage"]
    coverage = payload["coverage_summary"]
    gates = {
        "stage43_ce_precondition_ready": payload["stage43_ce_precondition"]["verdict"]
        == "stage43_ce_source_family_coverage_split_repair_ready",
        "cache_files_written": all(Path(row["cache_path"]).exists() for row in summaries.values()),
        "cache_rows_match_ce_assignment": all(int(summaries[split]["rows"]) == int(expected.get(split, -1)) for split in NEW_SPLITS),
        "train_val_test_rows_present": all(int(row["rows"]) > 0 for row in summaries.values()),
        "full_waypoint_labels_present": all(int(row["full_waypoint_rows"]) > 0 for row in summaries.values()),
        "endpoint_alignment_pass": all(float(row["max_endpoint_diff_last_waypoint"]) <= 1e-4 for row in summaries.values()),
        "source_splits_disjoint": payload["source_overlap_counts"]["train_val"] == 0
        and payload["source_overlap_counts"]["train_test"] == 0
        and payload["source_overlap_counts"]["val_test"] == 0,
        "validation_covers_test_source_families": coverage["test_families_without_validation_support"] == []
        and coverage["test_domain_families_without_validation_support"] == [],
        "no_future_waypoint_input": leakage["future_waypoint_input"] is False
        and leakage["future_waypoint_label_eval_only"] is True,
        "no_test_goal_or_stat_leakage": leakage["test_endpoint_goal_construction"] is False
        and leakage["test_statistics_normalization"] is False,
        "cache_not_committed_boundary": payload["cache_committed"] is False,
        "not_a_model_result_boundary_recorded": payload["claim_boundary"]["new_training_or_evaluation_not_run"] is True,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
        "long_objective_kept_active": payload["long_objective_complete"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    verdict = "stage43_cf_coverage_aware_full_waypoint_cache_ready" if passed == total else "stage43_cf_coverage_aware_full_waypoint_cache_incomplete"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_cf_gate"]
    return [
        "# Stage43-CF Coverage-Aware Full-Waypoint Cache",
        "",
        f"- source: `{SOURCE}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- cache dir: `{payload['cache_dir']}`",
        "- cache committed: `False`",
        "- new model training run: `False`",
        "",
        "## Purpose",
        "",
        "- Stage43-CE produced a coverage-aware source assignment but did not rebuild training labels.",
        "- Stage43-CF materializes the repaired full-waypoint supervision cache for train/val/test under that assignment.",
        "- Future endpoint/full-waypoint data remain labels/evaluation targets only; they are not model inputs.",
        "",
        "## Split Summary",
        "",
        "| split | rows | domains | sources | scenes | horizons | full waypoint rows | all-waypoint rows | missing tracks | endpoint diff max | hard | failure | easy |",
        "| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| {split} | {row['rows']} | `{row['domains']}` | {row['source_count']} | {row['scene_count']} | `{row['horizon_counts']}` | {row['full_waypoint_rows']} | {row['all_waypoints_valid_rows']} | {row['missing_track_rows']} | {row['max_endpoint_diff_last_waypoint']:.8f} | {row['hard_rows']} | {row['failure_rows']} | {row['easy_rows']} |"
            for split, row in payload["split_summaries"].items()
        ],
        "",
        "## Cache Files",
        "",
        "| split | path | sha256 | row hash |",
        "| --- | --- | --- | --- |",
        *[
            f"| {split} | `{row['cache_path']}` | `{row['cache_sha256']}` | `{row['row_hash']}` |"
            for split, row in payload["split_summaries"].items()
        ],
        "",
        "## Leakage Boundary",
        "",
        "- Source files are disjoint across train/val/test.",
        "- No future endpoint/waypoint is used as an inference input.",
        "- No central velocity, test endpoint goal construction, or test-statistics normalization is introduced.",
        "- This cache is local derived data and is intentionally not committed.",
        "",
        "## Claim Boundary",
        "",
        "- This is a cache rebuild, not a new model result.",
        "- Dataset-local/raw-frame 2.5D only.",
        "- No metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
        "",
        "## Next Required Step",
        "",
        "- Train/evaluate the Stage43 full-waypoint latent dynamics model on this repaired cache.",
        "- Keep the broad external stress matrix as diagnostic evidence; this repaired split is coverage-aware and narrower.",
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        "",
    ]


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_cf_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-CF Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
        ],
    )
    write_json(WORLD_GATE_JSON, _jsonable(gate))
    write_md(
        WORLD_GATE_MD,
        [
            "# Stage43 Current World-Model Gate",
            "",
            f"- source: `{SOURCE}`",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            "- deployable policy changed: `False`",
            "- new model training run: `False`",
            "- long objective complete: `False`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
            "## Current Boundary",
            "",
            "- Stage43-CF rebuilds the full-waypoint supervision cache under the CE coverage-aware split.",
            "- It is not a model result and does not replace the current safety floor.",
            "- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
            "",
        ],
    )


def _update_summaries(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_cf_gate"]
    split = payload["split_summaries"]
    block = [
        f"## {SECTION}",
        "",
        f"source = `{SOURCE}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        "deployable_policy_changed = `False`",
        "new_model_training_run = `False`",
        "cache_committed = `False`",
        "",
        "Stage43-CF materializes the CE coverage-aware source split into a local full-waypoint supervision cache. Future endpoints and waypoints are labels/eval targets only, not inference inputs.",
        "",
        f"Cache rows train/val/test = `{split['train']['rows']}` / `{split['val']['rows']}` / `{split['test']['rows']}`.",
        f"Cache dir = `{payload['cache_dir']}`.",
        "",
        "Interpretation: this closes the repaired-split cache blocker. It is not a model result; the next step is training/evaluating latent dynamics on this repaired cache.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, block)
    state = read_json(RESEARCH_STATE, {})
    state["stage43_cf_coverage_aware_full_waypoint_cache"] = {
        "source": SOURCE,
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "cache_dir": payload["cache_dir"],
        "cache_committed": False,
        "split_summaries": split,
        "source_overlap_counts": payload["source_overlap_counts"],
        "coverage_summary": payload["coverage_summary"],
        "no_leakage": payload["no_leakage"],
        "report": str(REPORT_MD),
        "new_model_training_run": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_cf_coverage_aware_full_waypoint_cache"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, _jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"source": SOURCE, "verdict": gate["verdict"], "generated_at_utc": payload["generated_at_utc"]}, ensure_ascii=False) + "\n")


def run_coverage_aware_full_waypoint_cache() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CACHE_DIR)
    ce = read_json(STAGE43_CE_JSON, {})
    assignments = {str(k): str(v) for k, v in ce.get("coverage_split", {}).get("source_assignments", {}).items()}
    split_arrays: dict[str, dict[str, np.ndarray]] = {}
    split_summaries: dict[str, Any] = {}
    for split in NEW_SPLITS:
        data = _concat_for_split(assignments, split)
        split_arrays[split] = data
        split_summaries[split] = _write_split_cache(split, data)
    sets = _source_sets(split_arrays)
    overlaps = {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_cache_rebuild_from_stage43_ce_assignment",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "stage43_ce_precondition": {
            "verdict": ce.get("stage43_ce_gate", {}).get("verdict"),
            "assignment_hash": ce.get("coverage_split", {}).get("assignment_hash"),
            "split_rows": {split: int(ce.get("split_summary", {}).get(split, {}).get("rows", -1)) for split in NEW_SPLITS},
            "report": str(STAGE43_CE_JSON),
        },
        "coverage_summary": ce.get("coverage_summary", {}),
        "cache_dir": str(CACHE_DIR),
        "cache_committed": False,
        "split_summaries": split_summaries,
        "source_overlap_counts": overlaps,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_waypoint_label_eval_only": True,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "source_level_split_disjoint": all(v == 0 for v in overlaps.values()),
        },
        "claim_boundary": {
            "new_training_or_evaluation_not_run": True,
            "true_3d_claim": False,
            "foundation_claim": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "long_objective_complete": False,
    }
    payload["stage43_cf_gate"] = _gate(payload)
    _write_outputs(payload)
    _update_summaries(payload)
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Stage43-CF coverage-aware full-waypoint cache rebuild.")
    parser.parse_args(argv)
    payload = run_coverage_aware_full_waypoint_cache()
    gate = payload["stage43_cf_gate"]
    print(f"Stage43-CF: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    for split, row in payload["split_summaries"].items():
        print(f"{split}: rows={row['rows']} full_waypoint={row['full_waypoint_rows']} sha={row['cache_sha256'][:12]}")
    return payload


if __name__ == "__main__":
    main()
