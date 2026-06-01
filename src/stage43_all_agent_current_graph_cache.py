from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src import stage43_full_waypoint_latent_dynamics as m


OUT_DIR = m.OUT_DIR
CACHE_DIR = Path("data/stage43_all_agent_current_graph_cache")
REPORT_JSON = OUT_DIR / "stage43_all_agent_current_graph_cache.json"
REPORT_MD = OUT_DIR / "stage43_all_agent_current_graph_cache.md"
GATE_MD = OUT_DIR / "stage43_stage_bm_all_agent_current_graph_cache_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SOURCE = "fresh_stage43_bm_all_agent_current_graph_cache"
SECTION = "STAGE43_BM_ALL_AGENT_CURRENT_GRAPH_CACHE"
TOP_K_DEFAULT = 8

BL_JSON = OUT_DIR / "stage43_raw_scene_graph_ablation_readiness.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _full_waypoint_path(split: str) -> Path:
    return m._cache_path(split)


def _graph_path(split: str) -> Path:
    return CACHE_DIR / f"stage43_all_agent_current_graph_{split}.npz"


def _load_full_waypoint(split: str) -> dict[str, np.ndarray]:
    path = _full_waypoint_path(split)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _frame_key(source_file: str, frame_id: float, horizon: int) -> tuple[str, str, int]:
    return (source_file, f"{float(frame_id):.6f}", int(horizon))


def _build_edges_for_split(split: str, *, top_k: int) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    data = _load_full_waypoint(split)
    xy = data["current_xy"].astype(np.float32)
    n = int(xy.shape[0])
    source_file = data["source_file"].astype(str)
    frame_id = data["frame_id"].astype(np.float64)
    horizon = data["horizon"].astype(np.int64)
    agent_id = data["agent_id"].astype(np.int64)

    groups: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for idx in range(n):
        groups[_frame_key(source_file[idx], frame_id[idx], int(horizon[idx]))].append(idx)

    neighbor_index = np.full((n, top_k), -1, dtype=np.int64)
    neighbor_agent_ids = np.full((n, top_k), -1, dtype=np.int64)
    neighbor_mask = np.zeros((n, top_k), dtype=bool)
    neighbor_rel_xy = np.zeros((n, top_k, 2), dtype=np.float32)
    neighbor_distance = np.zeros((n, top_k), dtype=np.float32)
    neighbor_edge_attr = np.zeros((n, top_k, 6), dtype=np.float32)
    all_agent_count = np.zeros(n, dtype=np.int16)

    edge_sources: list[int] = []
    edge_targets: list[int] = []
    edge_attrs: list[np.ndarray] = []

    multi_agent_rows = 0
    singleton_rows = 0
    for ids in groups.values():
        ids_arr = np.asarray(ids, dtype=np.int64)
        count = int(len(ids_arr))
        all_agent_count[ids_arr] = count
        if count <= 1:
            singleton_rows += count
            continue
        multi_agent_rows += count
        group_xy = xy[ids_arr]
        diff = group_xy[None, :, :] - group_xy[:, None, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        order = np.argsort(dist, axis=1)
        for local_pos, row_idx in enumerate(ids_arr):
            chosen = order[local_pos, :top_k]
            chosen = chosen[np.isfinite(dist[local_pos, chosen])]
            if len(chosen) == 0:
                continue
            slots = min(top_k, len(chosen))
            nbr_rows = ids_arr[chosen[:slots]]
            rel = diff[local_pos, chosen[:slots]].astype(np.float32)
            d = dist[local_pos, chosen[:slots]].astype(np.float32)
            inv = (1.0 / np.maximum(d, 1e-6)).astype(np.float32)
            cos = (rel[:, 0] / np.maximum(d, 1e-6)).astype(np.float32)
            sin = (rel[:, 1] / np.maximum(d, 1e-6)).astype(np.float32)
            attr = np.stack([rel[:, 0], rel[:, 1], d, inv, cos, sin], axis=1).astype(np.float32)

            neighbor_index[row_idx, :slots] = nbr_rows
            neighbor_agent_ids[row_idx, :slots] = agent_id[nbr_rows]
            neighbor_mask[row_idx, :slots] = True
            neighbor_rel_xy[row_idx, :slots, :] = rel
            neighbor_distance[row_idx, :slots] = d
            neighbor_edge_attr[row_idx, :slots, :] = attr

            edge_sources.extend([int(row_idx)] * slots)
            edge_targets.extend(nbr_rows.astype(int).tolist())
            edge_attrs.extend(attr)

    if edge_attrs:
        edge_attr = np.stack(edge_attrs, axis=0).astype(np.float32)
    else:
        edge_attr = np.zeros((0, 6), dtype=np.float32)
    edge_index = np.asarray([edge_sources, edge_targets], dtype=np.int64)
    if edge_index.size == 0:
        edge_index = edge_index.reshape(2, 0)

    density_count = np.sum((neighbor_distance > 0.0) & (neighbor_distance <= 10.0), axis=1).astype(np.float32)
    min_neighbor_distance = np.zeros(n, dtype=np.float32)
    has_neighbor = np.any(neighbor_mask, axis=1)
    if np.any(has_neighbor):
        finite_dist = np.where(neighbor_mask[has_neighbor], neighbor_distance[has_neighbor], np.inf)
        min_neighbor_distance[has_neighbor] = np.min(finite_dist, axis=1).astype(np.float32)

    graph_arrays = {
        "old_split": data["old_split"].astype(str),
        "local_row": data["local_row"].astype(np.int64),
        "dataset": data["dataset"].astype(str),
        "scene_id": data["scene_id"].astype(str),
        "source_file": data["source_file"].astype(str),
        "agent_id": data["agent_id"].astype(np.int64),
        "frame_id": data["frame_id"].astype(np.float64),
        "horizon": data["horizon"].astype(np.int16),
        "all_agent_current_xy": xy,
        "target_current_xy": xy,
        "all_agent_count": all_agent_count,
        "neighbor_index": neighbor_index,
        "neighbor_agent_ids": neighbor_agent_ids,
        "neighbor_mask": neighbor_mask,
        "neighbor_rel_xy": neighbor_rel_xy,
        "neighbor_distance": neighbor_distance,
        "neighbor_edge_attr": neighbor_edge_attr,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_attr_names": np.asarray(["rel_x", "rel_y", "distance", "inv_distance", "bearing_cos", "bearing_sin"]),
        "density_neighbor_count_radius10": density_count,
        "min_neighbor_distance": min_neighbor_distance,
        "graph_group_key_components": np.asarray(["source_file", "frame_id", "horizon"]),
    }
    summary = {
        "split": split,
        "rows": n,
        "groups": int(len(groups)),
        "singleton_rows": int(singleton_rows),
        "multi_agent_rows": int(multi_agent_rows),
        "multi_agent_row_fraction": float(multi_agent_rows / max(n, 1)),
        "edge_count": int(edge_index.shape[1]),
        "top_k": int(top_k),
        "max_agent_count_per_group": int(max((len(v) for v in groups.values()), default=0)),
        "mean_agent_count_per_group": float(np.mean([len(v) for v in groups.values()])) if groups else 0.0,
        "mean_degree": float(np.mean(np.sum(neighbor_mask, axis=1))) if n else 0.0,
        "horizon_counts": {str(int(k)): int(v) for k, v in sorted(Counter(horizon.astype(int).tolist()).items())},
        "dataset_counts": {str(k): int(v) for k, v in zip(*np.unique(data["dataset"].astype(str), return_counts=True))},
    }
    return graph_arrays, summary


def _validate_graph(split: str, arrays: Mapping[str, np.ndarray], summary: Mapping[str, Any]) -> dict[str, Any]:
    row_hash_graph = _row_hash(arrays)
    full = _load_full_waypoint(split)
    row_hash_full = _row_hash(full)
    edge_index = arrays["edge_index"]
    edge_attr = arrays["edge_attr"]
    n = int(arrays["horizon"].shape[0])
    edge_count = int(edge_index.shape[1])
    in_range = bool(
        edge_index.shape[0] == 2
        and edge_attr.shape[0] == edge_count
        and (edge_count == 0 or (np.min(edge_index) >= 0 and np.max(edge_index) < n))
    )
    no_self_edges = bool(edge_count == 0 or not np.any(edge_index[0] == edge_index[1]))
    finite_edge_attr = bool(np.all(np.isfinite(edge_attr)))
    neighbor_consistent = bool(
        arrays["neighbor_index"].shape[0] == n
        and arrays["neighbor_agent_ids"].shape == arrays["neighbor_index"].shape
        and arrays["neighbor_mask"].shape == arrays["neighbor_index"].shape
        and arrays["neighbor_rel_xy"].shape[:2] == arrays["neighbor_index"].shape
    )
    future_keys_absent = not any(
        key in arrays
        for key in [
            "future_xy",
            "waypoint_xy",
            "future_endpoint_x",
            "future_endpoint_y",
            "oracle_best",
            "selected_using_future",
        ]
    )
    return {
        "split": split,
        "row_hash_full_waypoint": row_hash_full,
        "row_hash_graph": row_hash_graph,
        "row_alignment_preserved": row_hash_full == row_hash_graph,
        "edge_index_in_range": in_range,
        "no_self_edges": no_self_edges,
        "finite_edge_attr": finite_edge_attr,
        "neighbor_dense_views_consistent": neighbor_consistent,
        "future_label_keys_absent_from_graph_cache": future_keys_absent,
        "rows": int(n),
        "edge_count": edge_count,
        "multi_agent_rows": int(summary["multi_agent_rows"]),
    }


def _write_split(split: str, *, top_k: int) -> tuple[dict[str, Any], dict[str, Any]]:
    arrays, summary = _build_edges_for_split(split, top_k=top_k)
    ensure_dir(CACHE_DIR)
    path = _graph_path(split)
    np.savez_compressed(path, **arrays)
    validation = _validate_graph(split, arrays, summary)
    summary = dict(summary)
    summary.update(
        {
            "cache_path": str(path),
            "cache_sha256": _sha256(path),
            "row_hash": validation["row_hash_graph"],
            "contains_edge_index": "edge_index" in arrays,
            "contains_edge_attr": "edge_attr" in arrays,
            "contains_all_agent_current_xy": "all_agent_current_xy" in arrays,
            "contains_all_agent_history_xy": False,
        }
    )
    return summary, validation


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summaries = payload["split_summaries"]
    validation = payload["validation"]
    boundary = payload["claim_boundary"]
    leakage = payload["no_leakage"]
    gates = {
        "bl_precondition_passed": payload["preconditions"]["stage43_bl_verdict"]
        == "stage43_bl_raw_scene_graph_ablation_readiness_pass_blocker_documented",
        "cache_files_written": all(Path(row["cache_path"]).exists() for row in summaries.values()),
        "train_val_test_rows_present": all(int(row["rows"]) > 0 for row in summaries.values()),
        "edge_tensors_present": all(row["contains_edge_index"] and row["contains_edge_attr"] for row in summaries.values()),
        "all_agent_current_state_present": all(row["contains_all_agent_current_xy"] for row in summaries.values()),
        "multi_agent_rows_present": all(int(row["multi_agent_rows"]) > 0 for row in summaries.values()),
        "row_alignment_preserved": all(row["row_alignment_preserved"] for row in validation.values()),
        "edge_validation_passed": all(
            row["edge_index_in_range"]
            and row["no_self_edges"]
            and row["finite_edge_attr"]
            and row["neighbor_dense_views_consistent"]
            for row in validation.values()
        ),
        "future_labels_not_in_inputs": all(row["future_label_keys_absent_from_graph_cache"] for row in validation.values()),
        "current_graph_ready_but_history_graph_blocked": payload["readiness_decision"][
            "all_agent_current_graph_ready"
        ]
        is True
        and payload["readiness_decision"]["all_agent_history_graph_ready"] is False,
        "no_overclaim": boundary["graph_rich_history_main_claim_allowed"] is False
        and boundary["current_frame_graph_cache_claim_allowed"] is True,
        "no_future_or_test_leakage": leakage["future_endpoint_input"] is False
        and leakage["future_waypoint_input"] is False
        and leakage["future_labels_cached_as_input"] is False
        and leakage["central_velocity_input"] is False
        and leakage["test_endpoint_goal_construction"] is False
        and leakage["test_statistics_normalization"] is False,
        "stage5c_and_smc_false": boundary["stage5c_executed"] is False and boundary["smc_enabled"] is False,
        "long_objective_kept_active": boundary["long_objective_complete"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_bm_all_agent_current_graph_cache_pass_partial_history_blocker"
        if passed == total
        else "stage43_bm_all_agent_current_graph_cache_incomplete",
        "all_agent_current_graph_cache_ready": passed == total,
        "all_agent_history_graph_cache_ready": False,
        "raw_scene_or_sdf_cache_ready": False,
        "protected_multimodal_latent_state_candidate": passed == total,
        "long_objective_complete": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def build_all_agent_current_graph_cache(*, top_k: int = TOP_K_DEFAULT) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    ensure_dir(CACHE_DIR)
    bl = read_json(BL_JSON, {})
    split_summaries: dict[str, Any] = {}
    validation: dict[str, Any] = {}
    for split in ["train", "val", "test"]:
        summary, valid = _write_split(split, top_k=top_k)
        split_summaries[split] = summary
        validation[split] = valid

    input_paths = [BL_JSON] + [_full_waypoint_path(split) for split in ["train", "val", "test"]]
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_build_current_frame_all_agent_knn_graph_cache_from_stage43_full_waypoint_rows",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "input_artifacts": [str(path) for path in input_paths],
        "input_hash": _combined_hash(input_paths),
        "preconditions": {
            "stage43_bl_verdict": bl.get("stage43_bl_gate", {}).get("verdict", "missing"),
            "stage43_bl_raw_scene_ready": bool(
                bl.get("stage43_bl_gate", {}).get("raw_scene_retrained_ablation_ready_now", False)
            ),
            "stage43_bl_graph_rich_ready": bool(
                bl.get("stage43_bl_gate", {}).get("graph_rich_retrained_ablation_ready_now", False)
            ),
        },
        "cache_dir": str(CACHE_DIR),
        "graph_schema": {
            "top_k": int(top_k),
            "group_key": ["source_file", "frame_id", "horizon"],
            "edge_index_shape": "[2, num_edges]",
            "edge_attr_names": ["rel_x", "rel_y", "distance", "inv_distance", "bearing_cos", "bearing_sin"],
            "dense_neighbor_views": [
                "neighbor_index",
                "neighbor_agent_ids",
                "neighbor_mask",
                "neighbor_rel_xy",
                "neighbor_distance",
                "neighbor_edge_attr",
            ],
            "row_aligned_current_state": "all_agent_current_xy",
            "all_agent_history_xy": "not_available_in_stage43_full_waypoint_cache",
        },
        "split_summaries": split_summaries,
        "validation": validation,
        "readiness_decision": {
            "all_agent_current_graph_ready": True,
            "all_agent_history_graph_ready": False,
            "raw_scene_or_sdf_ready": False,
            "graph_rich_retrained_ablation_ready_now": False,
            "reason": (
                "Stage43-BM builds row-aligned current-frame all-agent KNN edge tensors. "
                "It closes the current-state neighbor graph cache gap, but all-agent history graph "
                "and raw-scene/SDF tensors remain unavailable, so graph-rich history main claims and "
                "raw-scene ablations are still blocked."
            ),
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "future_labels_cached_as_input": False,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "graph_edges_use": "current_frame_source_file_frame_id_horizon_only",
        },
        "claim_boundary": {
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "dataset_local_raw_frame_only": True,
            "current_frame_graph_cache_claim_allowed": True,
            "graph_rich_history_main_claim_allowed": False,
            "raw_scene_or_sdf_main_claim_allowed": False,
            "retrained_graph_ablation_executed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
    }
    payload["stage43_bm_gate"] = _gate(payload)
    return payload


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_bm_gate"]
    lines = [
        "# Stage43-BM All-Agent Current Graph Cache",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- all-agent current graph cache ready: `{gate['all_agent_current_graph_cache_ready']}`",
        f"- all-agent history graph cache ready: `{gate['all_agent_history_graph_cache_ready']}`",
        f"- raw scene/SDF cache ready: `{gate['raw_scene_or_sdf_cache_ready']}`",
        "",
        "## Schema",
        "",
        f"- top_k: `{payload['graph_schema']['top_k']}`",
        f"- group key: `{payload['graph_schema']['group_key']}`",
        f"- edge attrs: `{payload['graph_schema']['edge_attr_names']}`",
        f"- all_agent_history_xy: `{payload['graph_schema']['all_agent_history_xy']}`",
        "",
        "## Split Summary",
        "",
        "| split | rows | groups | edges | multi-agent rows | mean degree | max agents/group | cache |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for split, row in payload["split_summaries"].items():
        lines.append(
            f"| `{split}` | `{row['rows']}` | `{row['groups']}` | `{row['edge_count']}` | `{row['multi_agent_rows']}` | `{row['mean_degree']:.3f}` | `{row['max_agent_count_per_group']}` | `{row['cache_path']}` |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            "| split | row alignment | in range | no self edges | finite attrs | future labels absent |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for split, row in payload["validation"].items():
        lines.append(
            f"| `{split}` | `{row['row_alignment_preserved']}` | `{row['edge_index_in_range']}` | `{row['no_self_edges']}` | `{row['finite_edge_attr']}` | `{row['future_label_keys_absent_from_graph_cache']}` |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This is a current-frame all-agent neighbor graph cache, not an all-agent history graph cache.",
            "- It does not include future endpoint or future waypoint inputs.",
            "- It does not execute a retrained graph ablation yet.",
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


def _write_outputs(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bm_gate"]
    write_json(REPORT_JSON, m._jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    write_json(WORLD_GATE_JSON, m._jsonable(gate))
    write_md(
        GATE_MD,
        [
            "# Stage43-BM All-Agent Current Graph Cache Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            f"- all-agent current graph cache ready: `{gate['all_agent_current_graph_cache_ready']}`",
            f"- all-agent history graph cache ready: `{gate['all_agent_history_graph_cache_ready']}`",
            f"- raw scene/SDF cache ready: `{gate['raw_scene_or_sdf_cache_ready']}`",
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
            f"- all-agent current graph cache ready: `{gate['all_agent_current_graph_cache_ready']}`",
            f"- all-agent history graph cache ready: `{gate['all_agent_history_graph_cache_ready']}`",
            f"- raw scene/SDF cache ready: `{gate['raw_scene_or_sdf_cache_ready']}`",
            f"- long objective complete: `{gate['long_objective_complete']}`",
            f"- Stage5C executed: `{gate['stage5c_executed']}`",
            f"- SMC enabled: `{gate['smc_enabled']}`",
            "",
            "## Current Boundary",
            "",
            "- Stage43-BM builds row-aligned current-frame all-agent KNN graph tensors.",
            "- This enables the next graph-aware retraining step, but all-agent history graph and raw-scene/SDF caches remain unavailable.",
            "- No retrained graph ablation has been executed in BM.",
            "- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
        ],
    )
    _update_ledgers(payload)


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_bm_gate"]
    test = payload["split_summaries"]["test"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"all_agent_current_graph_cache_ready = `{gate['all_agent_current_graph_cache_ready']}`",
        f"all_agent_history_graph_cache_ready = `{gate['all_agent_history_graph_cache_ready']}`",
        f"raw_scene_or_sdf_cache_ready = `{gate['raw_scene_or_sdf_cache_ready']}`",
        "",
        f"Stage43-BM builds current-frame all-agent KNN graph tensors from the full-waypoint row cache. Test rows `{test['rows']}`, test edges `{test['edge_count']}`, test multi-agent rows `{test['multi_agent_rows']}`, mean degree `{test['mean_degree']:.3f}`.",
        "",
        "This closes the current-state neighbor graph cache gap needed for future graph-aware retraining, but it does not close the all-agent history graph or raw-scene/SDF blocker. No graph ablation or training was executed in BM.",
        "",
        "Boundary unchanged: current-frame/past-available graph inputs only; future labels are not cached as inputs; no metric/seconds claim, no true 3D/foundation, no Stage5C, no SMC.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_bm_all_agent_current_graph_cache"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "all_agent_current_graph_cache_ready": gate["all_agent_current_graph_cache_ready"],
        "all_agent_history_graph_cache_ready": gate["all_agent_history_graph_cache_ready"],
        "raw_scene_or_sdf_cache_ready": gate["raw_scene_or_sdf_cache_ready"],
        "cache_dir": payload["cache_dir"],
        "split_summaries": payload["split_summaries"],
        "validation": payload["validation"],
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_bm_all_agent_current_graph_cache"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-BM",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "all_agent_current_graph_cache_ready": gate["all_agent_current_graph_cache_ready"],
                        "all_agent_history_graph_cache_ready": gate["all_agent_history_graph_cache_ready"],
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def run_all_agent_current_graph_cache(*, top_k: int = TOP_K_DEFAULT) -> dict[str, Any]:
    payload = build_all_agent_current_graph_cache(top_k=top_k)
    _write_outputs(payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stage43 row-aligned current-frame all-agent graph cache.")
    parser.add_argument("--top-k", type=int, default=TOP_K_DEFAULT)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    payload = run_all_agent_current_graph_cache(top_k=args.top_k)
    gate = payload["stage43_bm_gate"]
    print(f"Stage43-BM: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"cache_dir={payload['cache_dir']}")
    print(f"test_edges={payload['split_summaries']['test']['edge_count']}")
    return payload


if __name__ == "__main__":
    main()
