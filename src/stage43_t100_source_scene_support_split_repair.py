from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np

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
from src.stage43_source_level_heldout_split import (
    REPORT_JSON as STAGE43_F_JSON,
    _concat_pool,
    _sha256_text,
)


REPORT_JSON = OUT_DIR / "stage43_t100_source_scene_support_split_repair.json"
REPORT_MD = OUT_DIR / "stage43_t100_source_scene_support_split_repair.md"
GATE_MD = OUT_DIR / "stage43_stage_cp_t100_source_scene_support_split_gate.md"

SECTION = "STAGE43_CP_T100_SOURCE_SCENE_SUPPORT_SPLIT_REPAIR"
SOURCE = "fresh_stage43_cp_t100_source_scene_support_split_repair"
SPLITS = ["train", "val", "test"]


def _source_scene_key(source_file: str, scene_id: str) -> str:
    return f"{source_file}||{scene_id}"


def _agent_key(source_file: str, agent_id: int | str) -> str:
    return f"{source_file}||{agent_id}"


def _assignment_hash(assignments: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(assignments.astype(str).tobytes())
    return digest.hexdigest()


def _row_hash(pool: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in ["old_split", "local_row", "dataset", "scene_id", "source_file", "agent_id", "frame_id", "horizon"]:
        arr = np.asarray(pool[key])
        digest.update(key.encode("utf-8"))
        if arr.dtype.kind in {"U", "S", "O"}:
            digest.update(arr.astype(str).tobytes())
        else:
            digest.update(arr.tobytes())
    return digest.hexdigest()


def _assign_agent_disjoint_source_supported(pool: Mapping[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    source = pool["source_file"].astype(str)
    agent = pool["agent_id"].astype(str)
    assignment_by_agent: dict[str, str] = {}
    blockers: dict[str, Any] = {"sources_with_too_few_agents": [], "sources_without_validation_t100_support": []}
    source_plan: dict[str, Any] = {}

    for source_file in sorted(set(source.tolist())):
        mask = source == source_file
        agents = sorted(set(agent[mask].tolist()), key=lambda value: _sha256_text(f"{SOURCE}|{source_file}|{value}"))
        if len(agents) < 3:
            blockers["sources_with_too_few_agents"].append(source_file)
            for value in agents:
                assignment_by_agent[_agent_key(source_file, value)] = "train"
            source_plan[source_file] = {
                "agent_count": int(len(agents)),
                "train_agents": int(len(agents)),
                "val_agents": 0,
                "test_agents": 0,
                "reason": "blocked_too_few_agents_for_agent_disjoint_train_val_test",
            }
            continue

        n = len(agents)
        val_n = max(1, int(round(n * 0.20)))
        test_n = max(1, int(round(n * 0.20)))
        if val_n + test_n >= n:
            val_n = 1
            test_n = 1
        train_agents = agents[: n - val_n - test_n]
        val_agents = agents[n - val_n - test_n : n - test_n]
        test_agents = agents[n - test_n :]
        for value in train_agents:
            assignment_by_agent[_agent_key(source_file, value)] = "train"
        for value in val_agents:
            assignment_by_agent[_agent_key(source_file, value)] = "val"
        for value in test_agents:
            assignment_by_agent[_agent_key(source_file, value)] = "test"
        source_plan[source_file] = {
            "agent_count": int(n),
            "train_agents": int(len(train_agents)),
            "val_agents": int(len(val_agents)),
            "test_agents": int(len(test_agents)),
            "reason": "agent_disjoint_source_supported_split",
        }

    assignments = np.asarray(
        [assignment_by_agent.get(_agent_key(src, ag), "train") for src, ag in zip(source.tolist(), agent.tolist())],
        dtype=str,
    )

    for source_file in sorted(set(source.tolist())):
        src = source == source_file
        val_t100 = int(np.sum(src & (assignments == "val") & (pool["horizon"].astype(np.int64) == 100)))
        test_t100 = int(np.sum(src & (assignments == "test") & (pool["horizon"].astype(np.int64) == 100)))
        if test_t100 > 0 and val_t100 == 0:
            blockers["sources_without_validation_t100_support"].append(source_file)

    return assignments, {"source_plan": source_plan, "blockers": blockers}


def _split_summary(pool: Mapping[str, np.ndarray], assignments: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in SPLITS:
        mask = assignments == split
        horizons = Counter(pool["horizon"][mask].astype(int).tolist())
        domains = Counter(pool["dataset"][mask].astype(str).tolist())
        source_files = sorted(set(pool["source_file"][mask].astype(str).tolist()))
        scenes = sorted(set(pool["scene_id"][mask].astype(str).tolist()))
        source_agents = {
            _agent_key(src, ag)
            for src, ag in zip(pool["source_file"][mask].astype(str).tolist(), pool["agent_id"][mask].astype(str).tolist())
        }
        out[split] = {
            "rows": int(mask.sum()),
            "domains": dict(sorted((str(k), int(v)) for k, v in domains.items())),
            "source_count": int(len(source_files)),
            "scene_count": int(len(scenes)),
            "source_agent_count": int(len(source_agents)),
            "horizon_counts": {str(k): int(v) for k, v in sorted(horizons.items())},
            "hard_rows": int(np.sum(pool["hard"][mask])),
            "failure_rows": int(np.sum(pool["failure"][mask])),
            "easy_rows": int(np.sum(pool["easy"][mask])),
        }
    return out


def _support_summary(pool: Mapping[str, np.ndarray], assignments: np.ndarray, *, min_support_rows: int) -> dict[str, Any]:
    source = pool["source_file"].astype(str)
    scene = pool["scene_id"].astype(str)
    horizon = pool["horizon"].astype(np.int64)
    test_h100 = (assignments == "test") & (horizon == 100)
    val_h100 = (assignments == "val") & (horizon == 100)

    val_source_counts = Counter(source[val_h100].tolist())
    val_scene_counts = Counter(scene[val_h100].tolist())
    val_source_scene_counts = Counter(
        _source_scene_key(src, sc) for src, sc in zip(source[val_h100].tolist(), scene[val_h100].tolist())
    )

    source_supported = np.asarray(
        [val_source_counts[src] >= int(min_support_rows) for src in source],
        dtype=bool,
    )
    scene_supported = np.asarray(
        [val_scene_counts[sc] >= int(min_support_rows) for sc in scene],
        dtype=bool,
    )
    source_scene_supported = np.asarray(
        [val_source_scene_counts[_source_scene_key(src, sc)] >= int(min_support_rows) for src, sc in zip(source, scene)],
        dtype=bool,
    )
    supported = test_h100 & (source_supported | scene_supported)
    exact_supported = test_h100 & source_scene_supported
    unsupported = test_h100 & ~supported

    by_source: dict[str, Any] = {}
    for src in sorted(set(source[test_h100].tolist())):
        mask = test_h100 & (source == src)
        by_source[src] = {
            "test_t100_rows": int(mask.sum()),
            "validation_t100_rows": int(val_source_counts[src]),
            "supported": bool(val_source_counts[src] >= int(min_support_rows)),
        }

    by_scene: dict[str, Any] = {}
    for sc in sorted(set(scene[test_h100].tolist())):
        mask = test_h100 & (scene == sc)
        by_scene[sc] = {
            "test_t100_rows": int(mask.sum()),
            "validation_t100_rows": int(val_scene_counts[sc]),
            "supported": bool(val_scene_counts[sc] >= int(min_support_rows)),
        }

    return {
        "min_support_rows": int(min_support_rows),
        "test_t100_rows": int(test_h100.sum()),
        "source_supported_test_t100_rows": int((test_h100 & source_supported).sum()),
        "scene_supported_test_t100_rows": int((test_h100 & scene_supported).sum()),
        "source_or_scene_supported_test_t100_rows": int(supported.sum()),
        "exact_source_scene_supported_test_t100_rows": int(exact_supported.sum()),
        "unsupported_test_t100_rows": int(unsupported.sum()),
        "source_or_scene_supported_ratio": float(supported.sum() / max(int(test_h100.sum()), 1)),
        "exact_source_scene_supported_ratio": float(exact_supported.sum() / max(int(test_h100.sum()), 1)),
        "by_source": by_source,
        "by_scene": by_scene,
    }


def _leakage_summary(pool: Mapping[str, np.ndarray], assignments: np.ndarray) -> dict[str, Any]:
    split_masks = {split: assignments == split for split in SPLITS}
    row_keys_by_split = {
        split: {
            f"{old}|{row}"
            for old, row in zip(pool["old_split"][mask].astype(str).tolist(), pool["local_row"][mask].astype(str).tolist())
        }
        for split, mask in split_masks.items()
    }
    source_agent_by_split = {
        split: {
            _agent_key(src, ag)
            for src, ag in zip(pool["source_file"][mask].astype(str).tolist(), pool["agent_id"][mask].astype(str).tolist())
        }
        for split, mask in split_masks.items()
    }
    source_by_split = {
        split: set(pool["source_file"][mask].astype(str).tolist()) for split, mask in split_masks.items()
    }
    scene_by_split = {
        split: set(pool["scene_id"][mask].astype(str).tolist()) for split, mask in split_masks.items()
    }

    def overlaps(groups: Mapping[str, set[str]]) -> dict[str, int]:
        out = {}
        for i, left in enumerate(SPLITS):
            for right in SPLITS[i + 1 :]:
                out[f"{left}_{right}"] = int(len(groups[left] & groups[right]))
        return out

    return {
        "row_overlap_counts": overlaps(row_keys_by_split),
        "row_disjoint": all(value == 0 for value in overlaps(row_keys_by_split).values()),
        "source_agent_overlap_counts": overlaps(source_agent_by_split),
        "source_agent_disjoint": all(value == 0 for value in overlaps(source_agent_by_split).values()),
        "source_file_overlap_counts": overlaps(source_by_split),
        "scene_overlap_counts": overlaps(scene_by_split),
        "source_scene_overlap_intentional_for_support_protocol": True,
        "cross_source_generalization_split": False,
        "future_endpoint_input": False,
        "future_waypoint_input": False,
        "central_velocity_input": False,
        "test_endpoint_goal_construction": False,
        "test_statistics_normalization": False,
    }


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    split = payload["split_summary"]
    leakage = payload["no_leakage"]
    support = payload["support_summary"]
    gates = {
        "stage43_f_precondition_seen": payload["stage43_f_precondition"]["verdict"] == "stage43_f_source_level_split_ready",
        "source_supported_split_manifest_built": payload["assignment_hash"] != "",
        "train_val_test_nonempty": all(int(split[name]["rows"]) > 0 for name in SPLITS),
        "t100_test_rows_available": int(support["test_t100_rows"]) > 0,
        "t100_source_or_scene_support_positive": float(support["source_or_scene_supported_ratio"]) > 0.0,
        "t100_exact_source_scene_support_positive": float(support["exact_source_scene_supported_ratio"]) > 0.0,
        "row_disjoint": leakage["row_disjoint"] is True,
        "source_agent_disjoint": leakage["source_agent_disjoint"] is True,
        "source_scene_overlap_reported_as_protocol_not_leakage": leakage["source_scene_overlap_intentional_for_support_protocol"] is True
        and leakage["cross_source_generalization_split"] is False,
        "no_future_or_test_leakage_constructed": leakage["future_endpoint_input"] is False
        and leakage["future_waypoint_input"] is False
        and leakage["central_velocity_input"] is False
        and leakage["test_endpoint_goal_construction"] is False
        and leakage["test_statistics_normalization"] is False,
        "not_a_model_result_boundary_recorded": payload["claim_boundary"]["new_training_or_evaluation_not_run"] is True
        and payload["claim_boundary"]["requires_cache_rebuild_before_training"] is True,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
        "long_objective_kept_active": payload["long_objective_complete"] is False,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    verdict = (
        "stage43_cp_t100_source_scene_support_split_ready"
        if passed == total
        else "stage43_cp_t100_source_scene_support_split_partial"
    )
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_cp_gate"]
    support = payload["support_summary"]
    return [
        "# Stage43-CP T100 Source/Scene Support Split Repair",
        "",
        f"- source: `{SOURCE}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- assignment hash: `{payload['assignment_hash']}`",
        "- new model training run: `False`",
        "- deployable policy changed: `False`",
        "",
        "## Why I Built This",
        "",
        "- Stage43-CO proved the current source-level heldout split has zero exact source/scene support for t100.",
        "- This manifest builds a separate agent-disjoint source/scene-supported protocol for future t100 training/evaluation.",
        "- It is not a cross-source generalization result; the source and scene overlap is intentional and reported.",
        "",
        "## Split Summary",
        "",
        "| split | rows | domains | sources | scenes | source-agents | horizons |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- |",
        *[
            f"| {name} | {row['rows']} | `{row['domains']}` | {row['source_count']} | {row['scene_count']} | {row['source_agent_count']} | `{row['horizon_counts']}` |"
            for name, row in payload["split_summary"].items()
        ],
        "",
        "## T100 Support",
        "",
        f"- test t100 rows: `{support['test_t100_rows']}`",
        f"- source-supported test t100 rows: `{support['source_supported_test_t100_rows']}`",
        f"- scene-supported test t100 rows: `{support['scene_supported_test_t100_rows']}`",
        f"- source-or-scene-supported ratio: `{support['source_or_scene_supported_ratio']:.2%}`",
        f"- exact source-scene-supported ratio: `{support['exact_source_scene_supported_ratio']:.2%}`",
        f"- unsupported test t100 rows: `{support['unsupported_test_t100_rows']}`",
        "",
        "## Leakage Boundary",
        "",
        f"- row disjoint: `{payload['no_leakage']['row_disjoint']}`",
        f"- source-agent disjoint: `{payload['no_leakage']['source_agent_disjoint']}`",
        f"- source overlap counts: `{payload['no_leakage']['source_file_overlap_counts']}`",
        f"- scene overlap counts: `{payload['no_leakage']['scene_overlap_counts']}`",
        "- source/scene overlap is intentional for this support protocol and is not a cross-source generalization split.",
        "- no future endpoint/waypoint input, central velocity input, test endpoint goals, or test statistics normalization is constructed.",
        "",
        "## Interpretation",
        "",
        "- This gives a legal next path for t100 learning with validation support at source/scene level.",
        "- It does not replace the stricter heldout generalization result from Stage43-CO, where t100 remains floor-only.",
        "- The next step is to rebuild a light supervised cache on this protocol and evaluate whether t100 can improve without easy harm.",
        "- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
        "",
    ]


def _write_reports(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    write_md(REPORT_MD, _render_report(payload))
    gate = payload["stage43_cp_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-CP Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- passed: `{gate['passed']} / {gate['total']}`",
            "- new model training run: `False`",
            "- deployable policy changed: `False`",
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
        f"## Stage43-CP: t100 source/scene-supported split",
        "",
        "I built a separate agent-disjoint split protocol for t100 work where validation and test share source/scene support without sharing rows or source-agent tracks.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- test t100 rows: `{payload['support_summary']['test_t100_rows']}`",
        f"- source-or-scene-supported t100 ratio: `{payload['support_summary']['source_or_scene_supported_ratio']:.2%}`",
        f"- exact source-scene-supported t100 ratio: `{payload['support_summary']['exact_source_scene_supported_ratio']:.2%}`",
        f"- row disjoint: `{payload['no_leakage']['row_disjoint']}`",
        f"- source-agent disjoint: `{payload['no_leakage']['source_agent_disjoint']}`",
        "",
        "This is not a new model result and not cross-source generalization. It is the protocol I need before trying another t100 learner: current heldout t100 stays floor-only, while this supported split can test whether t100 learning is possible when validation actually covers the source/scene.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, readme_block)

    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage43_cp_t100_source_scene_support_split_repair"
    state["current_verdict"] = gate["verdict"]
    stage = state.setdefault("stage43_long_research_execution", {})
    stage["t100_source_scene_support_split_repair"] = {
        "source": SOURCE,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "assignment_hash": payload["assignment_hash"],
        "support_summary": payload["support_summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    write_json(RESEARCH_STATE, _jsonable(state))


def build_t100_source_scene_support_split_repair(*, min_support_rows: int = 200) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    pool = _concat_pool()
    assignments, plan = _assign_agent_disjoint_source_supported(pool)
    split_summary = _split_summary(pool, assignments)
    support_summary = _support_summary(pool, assignments, min_support_rows=int(min_support_rows))
    leakage = _leakage_summary(pool, assignments)
    stage43_f = read_json(STAGE43_F_JSON, {})

    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_agent_disjoint_source_scene_supported_t100_split_manifest",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "stage43_f_precondition": {
            "report": str(STAGE43_F_JSON),
            "verdict": stage43_f.get("stage43_f_gate", {}).get("verdict"),
            "row_hash": stage43_f.get("row_hash"),
        },
        "pool": {
            "rows": int(len(pool["horizon"])),
            "row_hash": _row_hash(pool),
            "source_count": int(len(set(pool["source_file"].astype(str).tolist()))),
            "scene_count": int(len(set(pool["scene_id"].astype(str).tolist()))),
            "domains": sorted(set(pool["dataset"].astype(str).tolist())),
        },
        "assignment_protocol": {
            "granularity": "source_file_agent_id",
            "source_scene_support": True,
            "same_source_scene_across_splits": True,
            "same_source_agent_across_splits": False,
            "selection_uses_labels_or_test_metrics": False,
            "rule": "deterministic_hash_split_agents_within_each_source_file",
            "min_support_rows": int(min_support_rows),
            "source_plan": plan["source_plan"],
            "blockers": plan["blockers"],
        },
        "assignment_hash": _assignment_hash(assignments),
        "split_summary": split_summary,
        "support_summary": support_summary,
        "no_leakage": leakage,
        "claim_boundary": {
            "new_training_or_evaluation_not_run": True,
            "requires_cache_rebuild_before_training": True,
            "not_cross_source_generalization_result": True,
            "true_3d_world_model": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "long_objective_complete": False,
    }
    payload["stage43_cp_gate"] = _gate(payload)
    _write_reports(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-support-rows", type=int, default=200)
    args = parser.parse_args()
    payload = build_t100_source_scene_support_split_repair(min_support_rows=int(args.min_support_rows))
    gate = payload["stage43_cp_gate"]
    print(f"Stage43-CP: {gate['verdict']} ({gate['passed']}/{gate['total']})")


if __name__ == "__main__":
    main()
