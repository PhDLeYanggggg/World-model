from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_source_level_full_waypoint_eval as am
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_protected_latent_state_model import _git_commit, _jsonable
from src.stage43_source_level_heldout_split import REPORT_JSON as SPLIT_JSON


OUT_DIR = Path("outputs/stage43_latent_state")
CACHE_DIR = Path("data/stage43_full_waypoint_supervision_cache")
DATA35 = Path("data/stage35_selective_transfer")

REPORT_JSON = OUT_DIR / "stage43_full_waypoint_supervision_cache.json"
REPORT_MD = OUT_DIR / "stage43_full_waypoint_supervision_cache.md"
GATE_MD = OUT_DIR / "stage43_stage_l_full_waypoint_supervision_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_L_FULL_WAYPOINT_SUPERVISION_CACHE"
SOURCE = "fresh_stage43_l_full_waypoint_supervision_cache"
OLD_SPLITS = ["train", "val", "test"]
NEW_SPLITS = ["train", "val", "test"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_hash(parts: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in ["old_split", "local_row", "dataset", "scene_id", "source_file", "agent_id", "frame_id", "horizon"]:
        arr = parts[key]
        digest.update(key.encode("utf-8"))
        if arr.dtype.kind in {"U", "S", "O"}:
            digest.update(np.asarray(arr).astype(str).tobytes())
        else:
            digest.update(np.asarray(arr).tobytes())
    return digest.hexdigest()


def _load_old_split(split: str) -> tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]:
    geo = np.load(DATA35 / f"expanded_external_{split}.npz", allow_pickle=False)
    labels = np.load(DATA35 / f"labels_{split}.npz", allow_pickle=False)
    return geo, labels


def _concat_for_new_split(assignments: Mapping[str, str], new_split: str) -> dict[str, np.ndarray]:
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
        source = geo["source_file"].astype(str)
        ids = np.where(np.asarray([assignments[str(src)] == new_split for src in source], dtype=bool))[0]
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
    return {key: np.concatenate(value, axis=0) for key, value in rows.items()}


def _cache_path(split: str) -> Path:
    return CACHE_DIR / f"stage43_full_waypoint_supervision_{split}.npz"


def _write_split_cache(split: str, data: Mapping[str, np.ndarray]) -> dict[str, Any]:
    labels = am._reconstruct_waypoint_labels(data)
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
    h = data["horizon"].astype(int)
    domain = data["dataset"].astype(str)
    scene = data["scene_id"].astype(str)
    return {
        "split": split,
        "cache_path": str(path),
        "cache_sha256": _sha256_file(path),
        "row_hash": _row_hash(data),
        "rows": int(len(h)),
        "domains": {str(k): int(v) for k, v in zip(*np.unique(domain, return_counts=True))},
        "scene_count": int(len(set(scene.tolist()))),
        "source_count": int(len(set(data["source_file"].astype(str).tolist()))),
        "horizon_counts": {str(int(k)): int(v) for k, v in sorted(Counter(h.tolist()).items())},
        "full_waypoint_rows": int(np.sum(np.any(labels["waypoint_valid"], axis=1))),
        "all_waypoints_valid_rows": int(np.sum(np.all(labels["waypoint_valid"], axis=1))),
        "missing_track_rows": int(np.sum(labels["missing_track"])),
        "max_endpoint_diff_last_waypoint": max_endpoint_diff,
        "hard_rows": int(np.sum(data["hard"].astype(bool))),
        "failure_rows": int(np.sum(data["failure"].astype(bool))),
        "easy_rows": int(np.sum(data["easy"].astype(bool))),
    }


def _source_sets(split_summaries: Mapping[str, Mapping[str, Any]], split_arrays: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, set[str]]:
    return {split: set(arr["source_file"].astype(str).tolist()) for split, arr in split_arrays.items()}


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summaries = payload["split_summaries"]
    leakage = payload["no_leakage"]
    gates = {
        "stage43_source_split_precondition_passed": payload["source_split_precondition"]["verdict"] == "stage43_f_source_level_split_ready",
        "cache_files_written": all(Path(row["cache_path"]).exists() for row in summaries.values()),
        "train_val_test_rows_present": all(int(row["rows"]) > 0 for row in summaries.values()),
        "train_val_full_waypoint_labels_present": summaries["train"]["full_waypoint_rows"] > 0
        and summaries["val"]["full_waypoint_rows"] > 0,
        "test_full_waypoint_labels_present": summaries["test"]["full_waypoint_rows"] > 0,
        "endpoint_alignment_pass": all(float(row["max_endpoint_diff_last_waypoint"]) <= 1e-4 for row in summaries.values()),
        "source_splits_disjoint": payload["source_overlap_counts"]["train_val"] == 0
        and payload["source_overlap_counts"]["train_test"] == 0
        and payload["source_overlap_counts"]["val_test"] == 0,
        "no_future_waypoint_input": leakage["future_waypoint_input"] is False
        and leakage["future_waypoint_label_eval_only"] is True,
        "no_test_goal_or_stat_leakage": leakage["test_endpoint_goal_construction"] is False
        and leakage["test_statistics_normalization"] is False,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_l_full_waypoint_supervision_cache_pass"
        if passed == total
        else "stage43_l_full_waypoint_supervision_cache_incomplete",
        "full_waypoint_supervised_training_ready": passed == total,
    }


def run_full_waypoint_supervision_cache() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CACHE_DIR)
    split_report = read_json(SPLIT_JSON, {})
    assignments = {str(k): str(v) for k, v in split_report.get("source_assignments", {}).items()}
    split_arrays: dict[str, dict[str, np.ndarray]] = {}
    split_summaries: dict[str, Any] = {}
    for split in NEW_SPLITS:
        data = _concat_for_new_split(assignments, split)
        split_arrays[split] = data
        split_summaries[split] = _write_split_cache(split, data)
    sets = _source_sets(split_summaries, split_arrays)
    overlaps = {
        "train_val": len(sets["train"] & sets["val"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["val"] & sets["test"]),
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_run_freeze_source_level_full_waypoint_supervision_labels",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "source_split_precondition": split_report.get("stage43_f_gate", {}),
        "source_split_row_hash": split_report.get("pool", {}).get("row_hash"),
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
            "true_3d": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage43_l_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    gate = payload["stage43_l_gate"]
    lines = [
        "# Stage43-L Full-Waypoint Supervision Cache",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- full-waypoint supervised training ready: `{gate['full_waypoint_supervised_training_ready']}`",
        f"- cache dir: `{payload['cache_dir']}`",
        "- cache committed: `False`",
        "",
        "## Why This Stage Exists",
        "",
        "Stage43-B had endpoint/failure/gain/harm latent-state training ready, but full-waypoint supervised latent training was blocked because train/val/test full-waypoint labels were not frozen under the Stage43 source-level split. Stage43-L freezes those labels as a local cache and records row hashes, source disjointness, and leakage boundaries.",
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
        "- Future waypoints and endpoints are labels/evaluation targets only.",
        "- They are not inference inputs.",
        "- Source files are disjoint across train/val/test.",
        "- No test endpoint goal construction, no central velocity input, no test-statistics normalization.",
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        "",
        "Conclusion: Stage43-L closes the Stage43-B full-waypoint supervision cache blocker for local source-level raw-frame training. It does not execute Stage5C or SMC and does not make metric/seconds/true-3D/foundation claims.",
    ]
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage43-L Full-Waypoint Supervision Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- full-waypoint supervised training ready: `{gate['full_waypoint_supervised_training_ready']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ],
    )
    _update_readmes(payload)


def _update_readmes(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_l_gate"]
    test = payload["split_summaries"]["test"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"full_waypoint_supervised_training_ready = `{gate['full_waypoint_supervised_training_ready']}`",
        "",
        "Stage43-L freezes source-level train/val/test full-waypoint supervision labels under the Stage43 source split. The cache is local and intentionally not committed. This closes the Stage43-B blocker for supervised full-waypoint latent-state training while keeping future waypoints as labels/eval only.",
        "",
        f"Rows: train `{payload['split_summaries']['train']['rows']}`, val `{payload['split_summaries']['val']['rows']}`, test `{test['rows']}`. Test full-waypoint rows `{test['full_waypoint_rows']}`; source overlaps train/val/test `{payload['source_overlap_counts']}`.",
        "",
        "No Stage5C, no SMC, no metric/seconds/true-3D/foundation claim.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(RESEARCH_STATE, {})
    state["stage43_l_full_waypoint_supervision_cache"] = {
        "source": payload["source"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "full_waypoint_supervised_training_ready": gate["full_waypoint_supervised_training_ready"],
        "cache_dir": payload["cache_dir"],
        "cache_committed": False,
        "split_summaries": payload["split_summaries"],
        "source_overlap_counts": payload["source_overlap_counts"],
        "report": str(REPORT_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    write_json(RESEARCH_STATE, state)
    ensure_dir(LEDGER_JSONL.parent)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                _jsonable(
                    {
                        "stage": "Stage43-L",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)
    return run_full_waypoint_supervision_cache()


if __name__ == "__main__":
    result = main()
    gate = result["stage43_l_gate"]
    print(f"Stage43-L full-waypoint supervision cache: {gate['verdict']} ({gate['passed']}/{gate['total']})")
