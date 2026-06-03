from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src import stage42_source_level_full_waypoint_eval as am
from src import stage43_t100_source_scene_support_split_repair as cp
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_full_waypoint_latent_dynamics import (
    M3W_README,
    OUT_DIR,
    README_RESULTS,
    RESEARCH_STATE,
    WORK_SUMMARY,
    _git_commit,
    _jsonable,
)
from src.stage43_full_waypoint_supervision_cache import _load_old_split, _sha256_file


CACHE_DIR = Path("data/stage43_cp_t100_source_scene_support_cache")
REPORT_JSON = OUT_DIR / "stage43_t100_source_scene_supported_supervision_cache.json"
REPORT_MD = OUT_DIR / "stage43_t100_source_scene_supported_supervision_cache.md"
GATE_MD = OUT_DIR / "stage43_stage_cq_t100_source_scene_supported_supervision_gate.md"

SECTION = "STAGE43_CQ_T100_SOURCE_SCENE_SUPPORTED_SUPERVISION_CACHE"
SOURCE = "fresh_stage43_cq_t100_source_scene_supported_supervision_cache"
OLD_SPLITS = ["train", "val", "test"]
NEW_SPLITS = ["train", "val", "test"]
HORIZON = 100


def _row_hash(data: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in ["old_split", "local_row", "dataset", "scene_id", "source_file", "agent_id", "frame_id", "horizon"]:
        arr = np.asarray(data[key])
        digest.update(key.encode("utf-8"))
        if arr.dtype.kind in {"U", "S", "O"}:
            digest.update(arr.astype(str).tobytes())
        else:
            digest.update(arr.tobytes())
    return digest.hexdigest()


def _agent_key(source_file: str, agent_id: str | int) -> str:
    return f"{source_file}||{agent_id}"


def _cache_path(split: str) -> Path:
    return CACHE_DIR / f"stage43_cp_t100_supervision_{split}.npz"


def _concat_for_cp_split(assignments_by_agent: Mapping[str, str], new_split: str) -> dict[str, np.ndarray]:
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
        agent = geo["agent_id"].astype(str)
        horizon = geo["horizon"].astype(np.int64)
        split_ids = np.asarray(
            [assignments_by_agent.get(_agent_key(src, ag), "train") == new_split for src, ag in zip(source, agent)],
            dtype=bool,
        )
        ids = np.where(split_ids & (horizon == HORIZON))[0]
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


def _assignment_by_agent_from_pool(pool: Mapping[str, np.ndarray]) -> tuple[dict[str, str], str]:
    assignments, _plan = cp._assign_agent_disjoint_source_supported(pool)
    assignment_by_agent: dict[str, str] = {}
    for src, ag, split in zip(pool["source_file"].astype(str), pool["agent_id"].astype(str), assignments.astype(str)):
        assignment_by_agent[_agent_key(src, ag)] = str(split)
    return assignment_by_agent, cp._assignment_hash(assignments)


def _write_split_cache(split: str, data: Mapping[str, np.ndarray], *, assignment_hash: str) -> dict[str, Any]:
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
        cp_assignment_hash=np.asarray([assignment_hash]),
        split_protocol=np.asarray(["source_scene_supported_agent_disjoint_t100"]),
    )
    h = data["horizon"].astype(int)
    domain = data["dataset"].astype(str)
    source = data["source_file"].astype(str)
    scene = data["scene_id"].astype(str)
    source_agent = {_agent_key(src, ag) for src, ag in zip(source.tolist(), data["agent_id"].astype(str).tolist())}
    return {
        "split": split,
        "cache_path": str(path),
        "cache_sha256": _sha256_file(path),
        "row_hash": _row_hash(data),
        "rows": int(len(h)),
        "domains": {str(k): int(v) for k, v in zip(*np.unique(domain, return_counts=True))},
        "source_count": int(len(set(source.tolist()))),
        "scene_count": int(len(set(scene.tolist()))),
        "source_agent_count": int(len(source_agent)),
        "horizon_counts": {str(int(k)): int(v) for k, v in sorted(Counter(h.tolist()).items())},
        "full_waypoint_rows": int(np.sum(np.any(labels["waypoint_valid"], axis=1))),
        "all_waypoints_valid_rows": int(np.sum(np.all(labels["waypoint_valid"], axis=1))),
        "missing_track_rows": int(np.sum(labels["missing_track"])),
        "max_endpoint_diff_last_waypoint": max_endpoint_diff,
        "hard_rows": int(np.sum(data["hard"].astype(bool))),
        "failure_rows": int(np.sum(data["failure"].astype(bool))),
        "easy_rows": int(np.sum(data["easy"].astype(bool))),
    }


def _overlap_counts(split_arrays: Mapping[str, Mapping[str, np.ndarray]], key: str) -> dict[str, int]:
    sets: dict[str, set[str]] = {}
    for split, data in split_arrays.items():
        if key == "source_agent":
            sets[split] = {
                _agent_key(src, ag)
                for src, ag in zip(data["source_file"].astype(str).tolist(), data["agent_id"].astype(str).tolist())
            }
        elif key == "row_key":
            sets[split] = {
                f"{old}||{row}"
                for old, row in zip(data["old_split"].astype(str).tolist(), data["local_row"].astype(str).tolist())
            }
        else:
            sets[split] = set(data[key].astype(str).tolist())
    return {
        "train_val": int(len(sets["train"] & sets["val"])),
        "train_test": int(len(sets["train"] & sets["test"])),
        "val_test": int(len(sets["val"] & sets["test"])),
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summaries = payload["split_summaries"]
    leakage = payload["no_leakage"]
    gates = {
        "stage43_cp_precondition_passed": payload["stage43_cp_precondition"]["verdict"]
        == "stage43_cp_t100_source_scene_support_split_ready",
        "cache_files_written": all(Path(row["cache_path"]).exists() for row in summaries.values()),
        "t100_only_cache": all(row["horizon_counts"] == {"100": row["rows"]} for row in summaries.values()),
        "train_val_test_rows_present": all(int(row["rows"]) > 0 for row in summaries.values()),
        "full_waypoint_labels_present": all(int(row["full_waypoint_rows"]) > 0 for row in summaries.values()),
        "endpoint_alignment_pass": all(float(row["max_endpoint_diff_last_waypoint"]) <= 1e-4 for row in summaries.values()),
        "source_or_scene_support_preserved": payload["support_summary"]["source_or_scene_supported_ratio"] >= 0.99,
        "source_agent_disjoint": leakage["source_agent_disjoint"] is True,
        "row_disjoint": leakage["row_disjoint"] is True,
        "source_scene_overlap_reported_as_protocol": leakage["source_scene_overlap_intentional_for_support_protocol"] is True,
        "no_future_waypoint_input": leakage["future_waypoint_input"] is False
        and leakage["future_waypoint_label_eval_only"] is True,
        "no_test_goal_or_stat_leakage": leakage["test_endpoint_goal_construction"] is False
        and leakage["test_statistics_normalization"] is False,
        "cache_not_committed_boundary": payload["cache_committed"] is False,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
        "long_objective_kept_active": payload["long_objective_complete"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = (
        "stage43_cq_t100_source_scene_supported_supervision_cache_pass"
        if passed == total
        else "stage43_cq_t100_source_scene_supported_supervision_cache_incomplete"
    )
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": verdict,
        "t100_supported_supervised_training_ready": passed == total,
    }


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_cq_gate"]
    return [
        "# Stage43-CQ T100 Source/Scene-Supported Supervision Cache",
        "",
        f"- source: `{SOURCE}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- t100 supported supervised training ready: `{gate['t100_supported_supervised_training_ready']}`",
        f"- cache dir: `{payload['cache_dir']}`",
        "- cache committed: `False`",
        "",
        "## Split Summary",
        "",
        "| split | rows | domains | sources | scenes | source-agents | full waypoint rows | all-waypoint rows | endpoint diff max | hard | failure | easy |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| {split} | {row['rows']} | `{row['domains']}` | {row['source_count']} | {row['scene_count']} | {row['source_agent_count']} | {row['full_waypoint_rows']} | {row['all_waypoints_valid_rows']} | {row['max_endpoint_diff_last_waypoint']:.8f} | {row['hard_rows']} | {row['failure_rows']} | {row['easy_rows']} |"
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
        "## Protocol Boundary",
        "",
        f"- CP assignment hash: `{payload['cp_assignment_hash']}`",
        f"- source-agent overlap counts: `{payload['no_leakage']['source_agent_overlap_counts']}`",
        f"- row overlap counts: `{payload['no_leakage']['row_overlap_counts']}`",
        f"- source overlap counts: `{payload['no_leakage']['source_file_overlap_counts']}`",
        f"- scene overlap counts: `{payload['no_leakage']['scene_overlap_counts']}`",
        "- source/scene overlap is intentional for this supported protocol; source-agent tracks and rows remain disjoint.",
        "- Future endpoints and waypoints are labels/eval only, not inference inputs.",
        "",
        "## Interpretation",
        "",
        "- Stage43-CQ turns the Stage43-CP support manifest into a t100-only supervised cache.",
        "- This still does not solve t100; it only removes the previous validation-support blocker for the next t100 learner.",
        "- The stricter heldout current split remains floor-only at t100 until a model passes its safety gates.",
        "- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
        "",
    ]


def _write_reports(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_cq_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-CQ Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- t100 supported supervised training ready: `{gate['t100_supported_supervised_training_ready']}`",
            "- cache committed: `False`",
            "- Stage5C executed: `False`",
            "- SMC enabled: `False`",
            "",
            "| gate | pass |",
            "| --- | --- |",
            *[f"| `{key}` | `{bool(value)}` |" for key, value in gate["gates"].items()],
            "",
        ],
    )
    readme_block = [
        "## Stage43-CQ: t100 supported supervision cache",
        "",
        "I froze a t100-only supervised cache for the Stage43-CP source/scene-supported protocol. This is the data artifact needed before training another t100 learner; it is not a new model result.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- train / val / test t100 rows: `{payload['split_summaries']['train']['rows']} / {payload['split_summaries']['val']['rows']} / {payload['split_summaries']['test']['rows']}`",
        f"- row disjoint: `{payload['no_leakage']['row_disjoint']}`",
        f"- source-agent disjoint: `{payload['no_leakage']['source_agent_disjoint']}`",
        f"- cache committed: `{payload['cache_committed']}`",
        "",
        "The cache keeps future waypoints and endpoints as labels only. The current heldout t100 policy is still floor-only; this cache just makes the next t100 training attempt legitimate.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, readme_block)

    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage43_cq_t100_source_scene_supported_supervision_cache"
    state["current_verdict"] = gate["verdict"]
    stage = state.setdefault("stage43_long_research_execution", {})
    stage["t100_source_scene_supported_supervision_cache"] = {
        "source": SOURCE,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "cp_assignment_hash": payload["cp_assignment_hash"],
        "cache_committed": False,
        "split_summaries": payload["split_summaries"],
        "no_leakage": payload["no_leakage"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, _jsonable(state))


def build_t100_source_scene_supported_supervision_cache() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CACHE_DIR)
    cp_payload = read_json(cp.REPORT_JSON, {})
    pool = cp._concat_pool()
    assignments_by_agent, assignment_hash = _assignment_by_agent_from_pool(pool)
    split_arrays: dict[str, dict[str, np.ndarray]] = {}
    split_summaries: dict[str, Any] = {}
    for split in NEW_SPLITS:
        data = _concat_for_cp_split(assignments_by_agent, split)
        split_arrays[split] = data
        split_summaries[split] = _write_split_cache(split, data, assignment_hash=assignment_hash)

    leakage = {
        "row_overlap_counts": _overlap_counts(split_arrays, "row_key"),
        "source_agent_overlap_counts": _overlap_counts(split_arrays, "source_agent"),
        "source_file_overlap_counts": _overlap_counts(split_arrays, "source_file"),
        "scene_overlap_counts": _overlap_counts(split_arrays, "scene_id"),
        "row_disjoint": all(value == 0 for value in _overlap_counts(split_arrays, "row_key").values()),
        "source_agent_disjoint": all(value == 0 for value in _overlap_counts(split_arrays, "source_agent").values()),
        "source_scene_overlap_intentional_for_support_protocol": True,
        "future_endpoint_input": False,
        "future_waypoint_input": False,
        "future_waypoint_label_eval_only": True,
        "central_velocity_input": False,
        "test_endpoint_goal_construction": False,
        "test_statistics_normalization": False,
    }
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_t100_only_source_scene_supported_full_waypoint_supervision_cache",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "stage43_cp_precondition": {
            "report": str(cp.REPORT_JSON),
            "verdict": cp_payload.get("stage43_cp_gate", {}).get("verdict"),
            "assignment_hash": cp_payload.get("assignment_hash"),
        },
        "cp_assignment_hash": assignment_hash,
        "cache_dir": str(CACHE_DIR),
        "cache_committed": False,
        "split_summaries": split_summaries,
        "support_summary": cp_payload.get("support_summary", {}),
        "no_leakage": leakage,
        "claim_boundary": {
            "new_model_training_or_evaluation_not_run": True,
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "long_objective_complete": False,
    }
    payload["stage43_cq_gate"] = _gate(payload)
    _write_reports(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    _args = parser.parse_args()
    payload = build_t100_source_scene_supported_supervision_cache()
    gate = payload["stage43_cq_gate"]
    print(f"Stage43-CQ: {gate['verdict']} ({gate['passed']}/{gate['total']})")


if __name__ == "__main__":
    main()
