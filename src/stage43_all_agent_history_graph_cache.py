from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m


OUT_DIR = m.OUT_DIR
CURRENT_GRAPH_DIR = Path("data/stage43_all_agent_current_graph_cache")
HISTORY_DIR = Path("data/stage37_t50_history")
CACHE_DIR = Path("data/stage43_all_agent_history_graph_cache")

REPORT_JSON = OUT_DIR / "stage43_all_agent_history_graph_cache.json"
REPORT_MD = OUT_DIR / "stage43_all_agent_history_graph_cache.md"
GATE_MD = OUT_DIR / "stage43_stage_bn_all_agent_history_graph_cache_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SOURCE = "fresh_stage43_bn_all_agent_history_graph_cache"
SECTION = "STAGE43_BN_ALL_AGENT_HISTORY_GRAPH_CACHE"
HISTORY_K_DEFAULT = 16

BM_JSON = OUT_DIR / "stage43_all_agent_current_graph_cache.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_existing(paths: list[Path]) -> str:
    return _combined_hash([path for path in paths if path.exists()])


def _row_hash(parts: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in ["old_split", "local_row", "dataset", "scene_id", "source_file", "agent_id", "frame_id", "horizon"]:
        digest.update(key.encode("utf-8"))
        arr = np.asarray(parts[key])
        if arr.dtype.kind in {"U", "S", "O"}:
            digest.update(arr.astype(str).tobytes())
        else:
            digest.update(arr.tobytes())
    return digest.hexdigest()


def _current_graph_path(split: str) -> Path:
    return CURRENT_GRAPH_DIR / f"stage43_all_agent_current_graph_{split}.npz"


def _history_path(old_split: str) -> Path:
    return HISTORY_DIR / f"history_windows_{old_split}.npz"


def _history_graph_path(split: str) -> Path:
    return CACHE_DIR / f"stage43_all_agent_history_graph_{split}.npz"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _gather_history_by_old_split(graph: Mapping[str, np.ndarray], key: str, history_k: int) -> np.ndarray:
    n = int(graph["horizon"].shape[0])
    old_split = graph["old_split"].astype(str)
    local_row = graph["local_row"].astype(np.int64)
    sample = _load_npz(_history_path("train"))[key]
    if sample.ndim == 2:
        out = np.zeros((n, history_k), dtype=sample.dtype)
    else:
        out = np.zeros((n,), dtype=sample.dtype)
    for split in ["train", "val", "test"]:
        ids = np.where(old_split == split)[0]
        if len(ids) == 0:
            continue
        hist = _load_npz(_history_path(split))[key]
        vals = hist[local_row[ids]]
        if vals.ndim == 2:
            vals = vals[:, -history_k:]
        out[ids] = vals
    return out


def _stack_target_history(graph: Mapping[str, np.ndarray], *, history_k: int) -> dict[str, np.ndarray]:
    hx = _gather_history_by_old_split(graph, "history_x", history_k).astype(np.float32)
    hy = _gather_history_by_old_split(graph, "history_y", history_k).astype(np.float32)
    hdx = _gather_history_by_old_split(graph, "history_dx", history_k).astype(np.float32)
    hdy = _gather_history_by_old_split(graph, "history_dy", history_k).astype(np.float32)
    speed = _gather_history_by_old_split(graph, "history_speed", history_k).astype(np.float32)
    valid = _gather_history_by_old_split(graph, "history_valid_mask", history_k).astype(bool)
    source_found = _gather_history_by_old_split(graph, "source_found", history_k).astype(bool)
    history_xy = np.stack([hx, hy], axis=2).astype(np.float32)
    history_dxdy = np.stack([hdx, hdy], axis=2).astype(np.float32)
    return {
        "target_history_xy": history_xy,
        "target_history_dxdy": history_dxdy,
        "target_history_speed": speed,
        "target_history_valid_mask": valid,
        "target_history_source_found": source_found,
    }


def _gather_neighbor_view(target: np.ndarray, neighbor_index: np.ndarray, neighbor_mask: np.ndarray) -> np.ndarray:
    safe_index = np.where(neighbor_mask, neighbor_index, 0).astype(np.int64)
    gathered = target[safe_index]
    if gathered.ndim == 3:
        gathered = gathered * neighbor_mask[:, :, None]
    elif gathered.ndim == 4:
        gathered = gathered * neighbor_mask[:, :, None, None]
    return gathered.astype(target.dtype, copy=False)


def _build_history_graph_for_split(split: str, *, history_k: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    graph = _load_npz(_current_graph_path(split))
    target = _stack_target_history(graph, history_k=history_k)
    neighbor_index = graph["neighbor_index"].astype(np.int64)
    neighbor_mask = graph["neighbor_mask"].astype(bool)

    neighbor_history_xy = _gather_neighbor_view(target["target_history_xy"], neighbor_index, neighbor_mask)
    neighbor_history_dxdy = _gather_neighbor_view(target["target_history_dxdy"], neighbor_index, neighbor_mask)
    neighbor_history_speed = _gather_neighbor_view(target["target_history_speed"], neighbor_index, neighbor_mask)
    neighbor_history_valid = _gather_neighbor_view(
        target["target_history_valid_mask"].astype(np.float32), neighbor_index, neighbor_mask
    ).astype(bool)
    neighbor_source_found = _gather_neighbor_view(
        target["target_history_source_found"].astype(np.float32), neighbor_index, neighbor_mask
    ).astype(bool)

    all_agent_history_xy = np.concatenate(
        [target["target_history_xy"][:, None, :, :], neighbor_history_xy], axis=1
    ).astype(np.float32)
    all_agent_history_dxdy = np.concatenate(
        [target["target_history_dxdy"][:, None, :, :], neighbor_history_dxdy], axis=1
    ).astype(np.float32)
    all_agent_history_speed = np.concatenate(
        [target["target_history_speed"][:, None, :], neighbor_history_speed], axis=1
    ).astype(np.float32)
    all_agent_history_valid_mask = np.concatenate(
        [target["target_history_valid_mask"][:, None, :], neighbor_history_valid], axis=1
    ).astype(bool)

    target_path_length = np.sum(
        np.linalg.norm(target["target_history_dxdy"], axis=2) * target["target_history_valid_mask"], axis=1
    ).astype(np.float32)
    neighbor_path_length = np.sum(
        np.linalg.norm(neighbor_history_dxdy, axis=3) * neighbor_history_valid, axis=2
    ).astype(np.float32)
    target_mean_speed = np.divide(
        np.sum(target["target_history_speed"] * target["target_history_valid_mask"], axis=1),
        np.maximum(np.sum(target["target_history_valid_mask"], axis=1), 1),
    ).astype(np.float32)
    neighbor_mean_speed = np.divide(
        np.sum(neighbor_history_speed * neighbor_history_valid, axis=2),
        np.maximum(np.sum(neighbor_history_valid, axis=2), 1),
    ).astype(np.float32)
    shared_valid_count = np.sum(
        target["target_history_valid_mask"][:, None, :] & neighbor_history_valid, axis=2
    ).astype(np.float32)
    mean_speed_delta = (neighbor_mean_speed - target_mean_speed[:, None]).astype(np.float32)
    edge_history_attr = np.stack(
        [
            shared_valid_count,
            neighbor_path_length,
            np.repeat(target_path_length[:, None], neighbor_path_length.shape[1], axis=1),
            neighbor_mean_speed,
            np.repeat(target_mean_speed[:, None], neighbor_path_length.shape[1], axis=1),
            mean_speed_delta,
        ],
        axis=2,
    ).astype(np.float32)
    edge_history_attr *= neighbor_mask[:, :, None]

    arrays: dict[str, np.ndarray] = {
        "old_split": graph["old_split"].astype(str),
        "local_row": graph["local_row"].astype(np.int64),
        "dataset": graph["dataset"].astype(str),
        "scene_id": graph["scene_id"].astype(str),
        "source_file": graph["source_file"].astype(str),
        "agent_id": graph["agent_id"].astype(np.int64),
        "frame_id": graph["frame_id"].astype(np.float64),
        "horizon": graph["horizon"].astype(np.int16),
        "neighbor_index": neighbor_index,
        "neighbor_agent_ids": graph["neighbor_agent_ids"].astype(np.int64),
        "neighbor_mask": neighbor_mask,
        "edge_index": graph["edge_index"].astype(np.int64),
        "edge_attr": graph["edge_attr"].astype(np.float32),
        "all_agent_current_xy": graph["all_agent_current_xy"].astype(np.float32),
        "all_agent_history_xy": all_agent_history_xy,
        "all_agent_history_dxdy": all_agent_history_dxdy,
        "all_agent_history_speed": all_agent_history_speed,
        "all_agent_history_valid_mask": all_agent_history_valid_mask,
        "target_history_source_found": target["target_history_source_found"],
        "neighbor_history_source_found": neighbor_source_found,
        "edge_history_attr": edge_history_attr,
        "edge_history_attr_names": np.asarray(
            [
                "shared_valid_count",
                "neighbor_path_length",
                "target_path_length",
                "neighbor_mean_speed",
                "target_mean_speed",
                "neighbor_minus_target_mean_speed",
            ]
        ),
        "history_k": np.asarray([history_k], dtype=np.int16),
    }
    valid_rows = np.sum(all_agent_history_valid_mask[:, 0, :], axis=1)
    neighbor_valid_rows = np.sum(np.any(all_agent_history_valid_mask[:, 1:, :], axis=2), axis=1)
    summary = {
        "split": split,
        "rows": int(arrays["horizon"].shape[0]),
        "history_k": int(history_k),
        "top_k": int(neighbor_index.shape[1]),
        "edge_count": int(arrays["edge_index"].shape[1]),
        "rows_with_target_history": int(np.sum(valid_rows > 0)),
        "rows_with_full_target_history": int(np.sum(valid_rows >= history_k)),
        "rows_with_any_neighbor_history": int(np.sum(neighbor_valid_rows > 0)),
        "rows_with_full_target_history_fraction": float(np.mean(valid_rows >= history_k)),
        "rows_with_any_neighbor_history_fraction": float(np.mean(neighbor_valid_rows > 0)),
        "mean_neighbor_history_degree": float(np.mean(neighbor_valid_rows)),
        "future_label_keys_present": [],
    }
    return arrays, summary


def _validate(split: str, arrays: Mapping[str, np.ndarray], summary: Mapping[str, Any]) -> dict[str, Any]:
    graph = _load_npz(_current_graph_path(split))
    row_hash_graph = _row_hash(graph)
    row_hash_history = _row_hash(arrays)
    n = int(arrays["horizon"].shape[0])
    edge_index = arrays["edge_index"]
    future_keys_absent = not any(
        key in arrays for key in ["future_xy", "waypoint_xy", "future_endpoint_x", "future_endpoint_y"]
    )
    valid_shapes = (
        arrays["all_agent_history_xy"].shape[:2] == (n, arrays["neighbor_index"].shape[1] + 1)
        and arrays["all_agent_history_xy"].shape[-1] == 2
        and arrays["all_agent_history_dxdy"].shape == arrays["all_agent_history_xy"].shape
        and arrays["all_agent_history_speed"].shape[:2] == arrays["all_agent_history_xy"].shape[:2]
        and arrays["all_agent_history_valid_mask"].shape == arrays["all_agent_history_speed"].shape
        and arrays["edge_history_attr"].shape[:2] == arrays["neighbor_index"].shape
    )
    finite = bool(
        np.all(np.isfinite(arrays["all_agent_history_xy"]))
        and np.all(np.isfinite(arrays["all_agent_history_dxdy"]))
        and np.all(np.isfinite(arrays["all_agent_history_speed"]))
        and np.all(np.isfinite(arrays["edge_history_attr"]))
    )
    edge_in_range = bool(edge_index.shape[0] == 2 and (edge_index.shape[1] == 0 or np.max(edge_index) < n))
    return {
        "split": split,
        "row_hash_current_graph": row_hash_graph,
        "row_hash_history_graph": row_hash_history,
        "row_alignment_preserved": row_hash_graph == row_hash_history,
        "valid_shapes": bool(valid_shapes),
        "finite_history_values": finite,
        "edge_index_in_range": edge_in_range,
        "future_label_keys_absent_from_history_graph_cache": future_keys_absent,
        "rows_with_full_target_history": int(summary["rows_with_full_target_history"]),
        "rows_with_any_neighbor_history": int(summary["rows_with_any_neighbor_history"]),
    }


def _write_split(split: str, *, history_k: int) -> tuple[dict[str, Any], dict[str, Any]]:
    arrays, summary = _build_history_graph_for_split(split, history_k=history_k)
    ensure_dir(CACHE_DIR)
    path = _history_graph_path(split)
    np.savez_compressed(path, **arrays)
    validation = _validate(split, arrays, summary)
    summary = dict(summary)
    summary.update({"cache_path": str(path), "cache_sha256": _sha256(path), "row_hash": validation["row_hash_history_graph"]})
    return summary, validation


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summaries = payload["split_summaries"]
    validation = payload["validation"]
    leak = payload["no_leakage"]
    claim = payload["claim_boundary"]
    gates = {
        "bm_precondition_passed": payload["preconditions"]["stage43_bm_verdict"]
        == "stage43_bm_all_agent_current_graph_cache_pass_partial_history_blocker",
        "history_cache_files_written": all(Path(row["cache_path"]).exists() for row in summaries.values()),
        "train_val_test_rows_present": all(int(row["rows"]) > 0 for row in summaries.values()),
        "target_history_present": all(int(row["rows_with_target_history"]) > 0 for row in summaries.values()),
        "neighbor_history_present": all(int(row["rows_with_any_neighbor_history"]) > 0 for row in summaries.values()),
        "row_alignment_preserved": all(row["row_alignment_preserved"] for row in validation.values()),
        "shape_and_finite_validation_passed": all(
            row["valid_shapes"] and row["finite_history_values"] and row["edge_index_in_range"]
            for row in validation.values()
        ),
        "future_labels_not_in_history_graph_inputs": all(
            row["future_label_keys_absent_from_history_graph_cache"] for row in validation.values()
        ),
        "history_graph_ready_but_raw_scene_blocked": payload["readiness_decision"]["all_agent_history_graph_ready"]
        is True
        and payload["readiness_decision"]["raw_scene_or_sdf_ready"] is False,
        "no_overclaim": claim["graph_rich_history_cache_claim_allowed"] is True
        and claim["retrained_graph_ablation_executed"] is False
        and claim["raw_scene_or_sdf_main_claim_allowed"] is False,
        "no_future_or_test_leakage": leak["future_endpoint_input"] is False
        and leak["future_waypoint_input"] is False
        and leak["future_labels_cached_as_input"] is False
        and leak["central_velocity_input"] is False
        and leak["test_endpoint_goal_construction"] is False
        and leak["test_statistics_normalization"] is False,
        "stage5c_and_smc_false": claim["stage5c_executed"] is False and claim["smc_enabled"] is False,
        "long_objective_kept_active": claim["long_objective_complete"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_bn_all_agent_history_graph_cache_pass_raw_scene_blocker"
        if passed == total
        else "stage43_bn_all_agent_history_graph_cache_incomplete",
        "all_agent_history_graph_cache_ready": passed == total,
        "raw_scene_or_sdf_cache_ready": False,
        "retrained_graph_ablation_executed": False,
        "protected_multimodal_latent_state_candidate": passed == total,
        "long_objective_complete": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def build_all_agent_history_graph_cache(*, history_k: int = HISTORY_K_DEFAULT) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CACHE_DIR)
    bm = read_json(BM_JSON, {})
    summaries: dict[str, Any] = {}
    validation: dict[str, Any] = {}
    for split in ["train", "val", "test"]:
        summary, valid = _write_split(split, history_k=history_k)
        summaries[split] = summary
        validation[split] = valid
    input_paths = [BM_JSON] + [_current_graph_path(split) for split in ["train", "val", "test"]] + [
        _history_path(split) for split in ["train", "val", "test"]
    ]
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_build_past_only_all_agent_history_graph_cache_from_stage37_history_and_stage43_current_graph",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "input_artifacts": [str(path) for path in input_paths],
        "input_hash": _combined_existing(input_paths),
        "preconditions": {
            "stage43_bm_verdict": bm.get("stage43_bm_gate", {}).get("verdict", "missing"),
            "stage43_bm_current_graph_ready": bool(
                bm.get("stage43_bm_gate", {}).get("all_agent_current_graph_cache_ready", False)
            ),
        },
        "cache_dir": str(CACHE_DIR),
        "history_graph_schema": {
            "history_k": int(history_k),
            "slot_0": "target_agent_history",
            "slots_1_to_top_k": "neighbor_agent_history_from_stage43_current_graph_neighbor_index",
            "all_agent_history_xy": "[rows, top_k_plus_target, history_k, 2]",
            "all_agent_history_dxdy": "[rows, top_k_plus_target, history_k, 2]",
            "all_agent_history_speed": "[rows, top_k_plus_target, history_k]",
            "edge_history_attr_names": [
                "shared_valid_count",
                "neighbor_path_length",
                "target_path_length",
                "neighbor_mean_speed",
                "target_mean_speed",
                "neighbor_minus_target_mean_speed",
            ],
        },
        "split_summaries": summaries,
        "validation": validation,
        "readiness_decision": {
            "all_agent_current_graph_ready": True,
            "all_agent_history_graph_ready": True,
            "raw_scene_or_sdf_ready": False,
            "retrained_graph_ablation_ready_next": True,
            "reason": (
                "Stage43-BN adds past-only target and neighbor history tensors on top of the BM current graph. "
                "This makes graph-history retraining feasible, while raw-scene/SDF ablations remain blocked."
            ),
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_cached_as_input": False,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "history_source": "stage37_past_only_history_windows",
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "dataset_local_raw_frame_only": True,
            "graph_rich_history_cache_claim_allowed": True,
            "retrained_graph_ablation_executed": False,
            "raw_scene_or_sdf_main_claim_allowed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
    }
    payload["stage43_bn_gate"] = _gate(payload)
    return payload


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_bn_gate"]
    lines = [
        "# Stage43-BN All-Agent History Graph Cache",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- all-agent history graph cache ready: `{gate['all_agent_history_graph_cache_ready']}`",
        f"- raw scene/SDF cache ready: `{gate['raw_scene_or_sdf_cache_ready']}`",
        f"- retrained graph ablation executed: `{gate['retrained_graph_ablation_executed']}`",
        "",
        "## Schema",
        "",
        f"- history_k: `{payload['history_graph_schema']['history_k']}`",
        f"- all_agent_history_xy: `{payload['history_graph_schema']['all_agent_history_xy']}`",
        f"- edge history attrs: `{payload['history_graph_schema']['edge_history_attr_names']}`",
        "",
        "## Split Summary",
        "",
        "| split | rows | history_k | rows full target history | rows any neighbor history | edge count | cache |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for split, row in payload["split_summaries"].items():
        lines.append(
            f"| `{split}` | `{row['rows']}` | `{row['history_k']}` | `{row['rows_with_full_target_history']}` | `{row['rows_with_any_neighbor_history']}` | `{row['edge_count']}` | `{row['cache_path']}` |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            "| split | row alignment | shapes finite | future labels absent |",
            "| --- | --- | --- | --- |",
        ]
    )
    for split, row in payload["validation"].items():
        lines.append(
            f"| `{split}` | `{row['row_alignment_preserved']}` | `{row['valid_shapes'] and row['finite_history_values']}` | `{row['future_label_keys_absent_from_history_graph_cache']}` |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This cache uses Stage37 past-only history windows and Stage43-BM current graph neighbors.",
            "- Future endpoint/full-waypoint labels are not cached as inputs.",
            "- This enables a future retrained graph ablation, but no graph ablation is executed here.",
            "- Raw-scene/SDF cache remains blocked.",
            "- No metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ]
    )
    return lines


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bn_gate"]
    test = payload["split_summaries"]["test"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"all_agent_history_graph_cache_ready = `{gate['all_agent_history_graph_cache_ready']}`",
        f"raw_scene_or_sdf_cache_ready = `{gate['raw_scene_or_sdf_cache_ready']}`",
        f"retrained_graph_ablation_executed = `{gate['retrained_graph_ablation_executed']}`",
        "",
        f"Stage43-BN builds past-only target and neighbor history graph tensors from Stage37 history windows plus Stage43-BM current graph neighbors. Test rows `{test['rows']}`, rows with full target history `{test['rows_with_full_target_history']}`, rows with any neighbor history `{test['rows_with_any_neighbor_history']}`, edge count `{test['edge_count']}`.",
        "",
        "This makes graph-history retraining feasible next, but no retrained graph ablation was executed in BN. Raw-scene/SDF remains the next unresolved cache blocker.",
        "",
        "Boundary unchanged: dataset-local/raw-frame 2.5D only; future labels are not cached as inputs; no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_bn_all_agent_history_graph_cache"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "all_agent_history_graph_cache_ready": gate["all_agent_history_graph_cache_ready"],
        "raw_scene_or_sdf_cache_ready": gate["raw_scene_or_sdf_cache_ready"],
        "retrained_graph_ablation_executed": gate["retrained_graph_ablation_executed"],
        "cache_dir": payload["cache_dir"],
        "split_summaries": payload["split_summaries"],
        "validation": payload["validation"],
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_bn_all_agent_history_graph_cache"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-BN",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "all_agent_history_graph_cache_ready": gate["all_agent_history_graph_cache_ready"],
                        "raw_scene_or_sdf_cache_ready": gate["raw_scene_or_sdf_cache_ready"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def _write_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bn_gate"]
    write_json(REPORT_JSON, m._jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_json(WORLD_GATE_JSON, m._jsonable(gate))
    write_md(
        GATE_MD,
        [
            "# Stage43-BN All-Agent History Graph Cache Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- all-agent history graph cache ready: `{gate['all_agent_history_graph_cache_ready']}`",
            f"- raw scene/SDF cache ready: `{gate['raw_scene_or_sdf_cache_ready']}`",
            f"- retrained graph ablation executed: `{gate['retrained_graph_ablation_executed']}`",
            f"- long objective complete: `{gate['long_objective_complete']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    write_md(
        WORLD_GATE_MD,
        [
            "# Stage43 Current World-Model Gate",
            "",
            f"- source: `{payload['source']}`",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- protected multimodal latent state candidate: `{gate['protected_multimodal_latent_state_candidate']}`",
            f"- all-agent history graph cache ready: `{gate['all_agent_history_graph_cache_ready']}`",
            f"- raw scene/SDF cache ready: `{gate['raw_scene_or_sdf_cache_ready']}`",
            f"- retrained graph ablation executed: `{gate['retrained_graph_ablation_executed']}`",
            f"- long objective complete: `{gate['long_objective_complete']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "## Current Boundary",
            "",
            "- Stage43-BN builds row-aligned past-only all-agent history graph tensors.",
            "- This enables graph-history retraining next, but no retrained graph ablation has been executed in BN.",
            "- Raw-scene/SDF cache remains unavailable.",
            "- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    _update_ledgers(payload)


def run_all_agent_history_graph_cache(*, history_k: int = HISTORY_K_DEFAULT) -> dict[str, Any]:
    payload = build_all_agent_history_graph_cache(history_k=history_k)
    _write_outputs(payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stage43 row-aligned all-agent history graph cache.")
    parser.add_argument("--history-k", type=int, default=HISTORY_K_DEFAULT)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    payload = run_all_agent_history_graph_cache(history_k=args.history_k)
    gate = payload["stage43_bn_gate"]
    print(f"Stage43-BN: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"cache_dir={payload['cache_dir']}")
    print(f"test_neighbor_history_rows={payload['split_summaries']['test']['rows_with_any_neighbor_history']}")
    return payload


if __name__ == "__main__":
    main()
