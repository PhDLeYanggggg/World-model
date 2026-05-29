from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section


OUT_DIR = Path("outputs/stage43_latent_state")
DATA35 = Path("data/stage35_selective_transfer")

REPORT_JSON = OUT_DIR / "stage43_source_level_heldout_split.json"
REPORT_MD = OUT_DIR / "stage43_source_level_heldout_split.md"
GATE_MD = OUT_DIR / "stage43_stage_f_source_level_split_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE43_F_SOURCE_LEVEL_HELDOUT_SPLIT"
SOURCE = "fresh_stage43_f_source_level_heldout_split"
DOMAINS = ["ETH_UCY", "TrajNet", "UCY"]
SPLITS = ["train", "val", "test"]


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_split(split: str) -> dict[str, np.ndarray]:
    geo_path = DATA35 / f"expanded_external_{split}.npz"
    labels_path = DATA35 / f"labels_{split}.npz"
    geo = np.load(geo_path, allow_pickle=False)
    labels = np.load(labels_path, allow_pickle=False)
    n = len(geo["horizon"])
    old_split = np.asarray([split] * n)
    local_row = np.arange(n, dtype=np.int64)
    return {
        "old_split": old_split,
        "local_row": local_row,
        "dataset": geo["dataset"].astype(str),
        "scene_id": geo["scene_id"].astype(str),
        "source_file": geo["source_file"].astype(str),
        "agent_id": geo["agent_id"].astype(np.int64),
        "frame_id": geo["frame_id"].astype(np.float64),
        "horizon": geo["horizon"].astype(np.int64),
        "track_length": geo["track_length"].astype(np.float32),
        "hard": labels["hard"].astype(bool),
        "failure": labels["failure"].astype(bool),
        "easy": labels["easy"].astype(bool),
        "scale": labels["scale"].astype(np.float32),
    }


def _concat_pool() -> dict[str, np.ndarray]:
    parts = [_load_split(split) for split in SPLITS]
    keys = parts[0].keys()
    return {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}


def _source_summaries(pool: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for source_file in sorted(set(pool["source_file"].tolist())):
        mask = pool["source_file"] == source_file
        domains = sorted(set(pool["dataset"][mask].tolist()))
        scenes = sorted(set(pool["scene_id"][mask].tolist()))
        old_splits = sorted(set(pool["old_split"][mask].tolist()))
        horizons = Counter(pool["horizon"][mask].astype(int).tolist())
        summaries[source_file] = {
            "source_file": source_file,
            "source_id": _sha256_text(source_file)[:16],
            "domain": domains[0] if len(domains) == 1 else "mixed",
            "domains": domains,
            "scenes": scenes,
            "old_splits": old_splits,
            "rows": int(mask.sum()),
            "horizon_counts": {str(k): int(v) for k, v in sorted(horizons.items())},
            "hard_rows": int(np.sum(pool["hard"][mask])),
            "failure_rows": int(np.sum(pool["failure"][mask])),
            "easy_rows": int(np.sum(pool["easy"][mask])),
            "agent_count": int(len(set(pool["agent_id"][mask].astype(int).tolist()))),
            "frame_min": float(np.min(pool["frame_id"][mask])) if int(mask.sum()) else 0.0,
            "frame_max": float(np.max(pool["frame_id"][mask])) if int(mask.sum()) else 0.0,
            "track_length_mean": float(np.mean(pool["track_length"][mask])) if int(mask.sum()) else 0.0,
            "scale_median": float(np.median(pool["scale"][mask])) if int(mask.sum()) else 0.0,
        }
    return summaries


def _assign_sources(summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    by_domain: dict[str, list[str]] = defaultdict(list)
    for source_file, row in summaries.items():
        by_domain[str(row["domain"])].append(source_file)
    assignments: dict[str, str] = {}
    domain_plan: dict[str, Any] = {}
    for domain in sorted(by_domain):
        sources = sorted(by_domain[domain], key=lambda s: _sha256_text(f"{domain}:{s}"))
        n = len(sources)
        if n < 3:
            # Keep a hard blocker visible instead of fabricating a full split.
            test_n = 1 if n >= 1 else 0
            val_n = 0
        else:
            test_n = max(1, int(round(n * 0.2)))
            val_n = max(1, int(round(n * 0.2)))
            if test_n + val_n >= n:
                test_n = 1
                val_n = 1 if n >= 3 else 0
        test_sources = sources[:test_n]
        val_sources = sources[test_n : test_n + val_n]
        train_sources = sources[test_n + val_n :]
        for source_file in train_sources:
            assignments[source_file] = "train"
        for source_file in val_sources:
            assignments[source_file] = "val"
        for source_file in test_sources:
            assignments[source_file] = "test"
        domain_plan[domain] = {
            "source_count": n,
            "train_sources": [_sha256_text(s)[:16] for s in train_sources],
            "val_sources": [_sha256_text(s)[:16] for s in val_sources],
            "test_sources": [_sha256_text(s)[:16] for s in test_sources],
            "train_source_files": train_sources,
            "val_source_files": val_sources,
            "test_source_files": test_sources,
        }
    return {"assignments": assignments, "domain_plan": domain_plan}


def _split_summary(pool: Mapping[str, np.ndarray], assignments: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    assigned = np.asarray([assignments[str(source)] for source in pool["source_file"]])
    for split in SPLITS:
        mask = assigned == split
        domains = sorted(set(pool["dataset"][mask].tolist())) if int(mask.sum()) else []
        scenes = sorted(set(pool["scene_id"][mask].tolist())) if int(mask.sum()) else []
        sources = sorted(set(pool["source_file"][mask].tolist())) if int(mask.sum()) else []
        out[split] = {
            "rows": int(mask.sum()),
            "domains": domains,
            "domain_counts": {
                str(k): int(v)
                for k, v in zip(*np.unique(pool["dataset"][mask], return_counts=True))
            }
            if int(mask.sum())
            else {},
            "scene_count": int(len(scenes)),
            "source_count": int(len(sources)),
            "source_ids": [_sha256_text(s)[:16] for s in sources],
            "horizon_counts": {
                str(int(k)): int(v)
                for k, v in zip(*np.unique(pool["horizon"][mask].astype(int), return_counts=True))
            }
            if int(mask.sum())
            else {},
            "hard_rows": int(np.sum(pool["hard"][mask])),
            "failure_rows": int(np.sum(pool["failure"][mask])),
            "easy_rows": int(np.sum(pool["easy"][mask])),
        }
    return out


def _row_hash(pool: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in ["old_split", "local_row", "dataset", "scene_id", "source_file", "agent_id", "frame_id", "horizon"]:
        arr = pool[key]
        digest.update(key.encode("utf-8"))
        digest.update(np.asarray(arr).astype(str).tobytes() if arr.dtype.kind in {"U", "S", "O"} else np.asarray(arr).tobytes())
    return digest.hexdigest()


def _leakage(pool: Mapping[str, np.ndarray], assignments: Mapping[str, str]) -> dict[str, Any]:
    source_sets = {split: set() for split in SPLITS}
    scene_sets = {split: set() for split in SPLITS}
    row_keys = set()
    row_duplicate_count = 0
    for old_split, local_row, source, scene in zip(
        pool["old_split"].tolist(),
        pool["local_row"].astype(int).tolist(),
        pool["source_file"].tolist(),
        pool["scene_id"].tolist(),
    ):
        split = assignments[str(source)]
        source_sets[split].add(str(source))
        scene_sets[split].add(str(scene))
        key = (str(old_split), int(local_row))
        if key in row_keys:
            row_duplicate_count += 1
        row_keys.add(key)
    source_overlap = {
        f"{a}_{b}": sorted(source_sets[a] & source_sets[b])
        for i, a in enumerate(SPLITS)
        for b in SPLITS[i + 1 :]
    }
    scene_overlap = {
        f"{a}_{b}": sorted(scene_sets[a] & scene_sets[b])
        for i, a in enumerate(SPLITS)
        for b in SPLITS[i + 1 :]
    }
    return {
        "source_file_disjoint": all(len(v) == 0 for v in source_overlap.values()),
        "source_overlap_counts": {k: len(v) for k, v in source_overlap.items()},
        "scene_overlap_counts": {k: len(v) for k, v in scene_overlap.items()},
        "scene_overlap_allowed_for_source_level_split": True,
        "row_duplicate_count": int(row_duplicate_count),
        "row_overlap_pass": row_duplicate_count == 0,
        "future_endpoint_input": False,
        "future_waypoint_input": False,
        "central_velocity_input": False,
        "test_endpoint_goals_constructed": False,
        "test_statistics_normalization": False,
        "split_granularity": "source_file_level",
    }


def _artifact_hashes() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in SPLITS:
        out[split] = {}
        for name in ["expanded_external", "labels"]:
            path = DATA35 / (f"{name}_{split}.npz" if name == "expanded_external" else f"labels_{split}.npz")
            out[split][name] = {"path": str(path), "exists": path.exists(), "sha256": _sha256_file(path) if path.exists() else ""}
    return out


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    split_summary = payload["split_summary"]
    leakage = payload["no_leakage"]
    domains_with_test = {
        domain
        for domain, plan in payload["domain_plan"].items()
        if len(plan["test_source_files"]) > 0 and len(plan["train_source_files"]) > 0
    }
    gates = {
        "input_pool_loaded": payload["pool"]["rows"] > 0,
        "all_required_domains_present": set(payload["pool"]["domains"]) >= set(DOMAINS),
        "source_level_split_built": payload["split_granularity"] == "source_file_level",
        "train_val_test_nonempty": all(split_summary[split]["rows"] > 0 for split in SPLITS),
        "each_domain_has_test_and_train_sources": domains_with_test >= set(DOMAINS),
        "test_contains_all_domains": set(split_summary["test"]["domains"]) >= set(DOMAINS),
        "source_file_disjoint": leakage["source_file_disjoint"] is True,
        "row_overlap_pass": leakage["row_overlap_pass"] is True,
        "no_future_or_test_leakage_constructed": leakage["future_endpoint_input"] is False
        and leakage["future_waypoint_input"] is False
        and leakage["central_velocity_input"] is False
        and leakage["test_endpoint_goals_constructed"] is False
        and leakage["test_statistics_normalization"] is False,
        "old_split_reuse_boundary_recorded": payload["claim_boundary"]["old_split_pool_reused_for_new_stage43_split"] is True
        and payload["claim_boundary"]["previous_stage43_checkpoint_not_official_for_new_split"] is True,
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
        "verdict": "stage43_f_source_level_split_ready" if passed == total else "stage43_f_source_level_split_partial",
        "next_action": "train/evaluate Stage43 latent model on this source-level split; old Stage43-C checkpoint remains UCY-heldout evidence only",
    }


def build_source_level_split() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    pool = _concat_pool()
    summaries = _source_summaries(pool)
    assignment_payload = _assign_sources(summaries)
    assignments = assignment_payload["assignments"]
    split_summary = _split_summary(pool, assignments)
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_split_manifest_from_existing_stage35_36_37_artifacts",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "split_granularity": "source_file_level",
        "pool": {
            "rows": int(len(pool["horizon"])),
            "domains": sorted(set(pool["dataset"].tolist())),
            "domain_counts": {str(k): int(v) for k, v in zip(*np.unique(pool["dataset"], return_counts=True))},
            "source_count": int(len(summaries)),
            "scene_count": int(len(set(pool["scene_id"].tolist()))),
            "horizon_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(pool["horizon"].astype(int), return_counts=True))},
            "row_hash": _row_hash(pool),
        },
        "artifact_hashes": _artifact_hashes(),
        "source_summaries": summaries,
        "source_assignments": {source: assignments[source] for source in sorted(assignments)},
        "domain_plan": assignment_payload["domain_plan"],
        "split_summary": split_summary,
        "no_leakage": _leakage(pool, assignments),
        "claim_boundary": {
            "old_split_pool_reused_for_new_stage43_split": True,
            "previous_stage43_checkpoint_not_official_for_new_split": True,
            "new_training_or_evaluation_not_run": True,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "true_3d": False,
            "foundation_world_model": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage43_f_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    gate = payload["stage43_f_gate"]
    lines = [
        "# Stage43-F Source-Level Heldout Split",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- split granularity: `{payload['split_granularity']}`",
        f"- row hash: `{payload['pool']['row_hash']}`",
        "",
        "## Pool",
        "",
        f"- rows: `{payload['pool']['rows']}`",
        f"- domains: `{payload['pool']['domains']}`",
        f"- domain counts: `{payload['pool']['domain_counts']}`",
        f"- source count: `{payload['pool']['source_count']}`",
        f"- scene count: `{payload['pool']['scene_count']}`",
        f"- horizon counts: `{payload['pool']['horizon_counts']}`",
        "",
        "## New Source-Level Split",
        "",
        "| split | rows | domains | sources | scenes | horizons | hard | failure | easy |",
        "| --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |",
        *[
            f"| {split} | {row['rows']} | `{row['domains']}` | {row['source_count']} | {row['scene_count']} | `{row['horizon_counts']}` | {row['hard_rows']} | {row['failure_rows']} | {row['easy_rows']} |"
            for split, row in payload["split_summary"].items()
        ],
        "",
        "## Leakage Boundary",
        "",
        f"- source files disjoint: `{payload['no_leakage']['source_file_disjoint']}`",
        f"- row overlap pass: `{payload['no_leakage']['row_overlap_pass']}`",
        f"- scene overlap counts: `{payload['no_leakage']['scene_overlap_counts']}`",
        "- scene overlap is reported rather than hidden because this is a source-file-level split, not a strict scene-level split.",
        "- no future endpoint/waypoint input, central velocity input, test endpoint goals, or test statistics normalization is constructed by this manifest.",
        "",
        "## Claim Boundary",
        "",
        "- This is a fresh split manifest, not a new model training/evaluation result.",
        "- The old Stage43-C checkpoint remains UCY-heldout evidence only and is not official for this new source-level split.",
        "- Coordinates remain dataset-local/raw-frame; no metric or seconds-level claim is made.",
        "- Stage5C and SMC remain disabled.",
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
    ]
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage43-F Source-Level Split Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- next action: {gate['next_action']}",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ],
    )
    _update_readmes(payload)


def _update_readmes(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_f_gate"]
    split = payload["split_summary"]
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        "",
        "Stage43-F builds the source-file-level heldout split required by Stage43-E. It reuses the existing Stage35/36/37 external artifacts as a data pool, but creates a new split manifest where ETH_UCY, TrajNet, and UCY all appear in test through disjoint source files.",
        "",
        f"Pool rows `{payload['pool']['rows']}`, domains `{payload['pool']['domain_counts']}`, row hash `{payload['pool']['row_hash']}`.",
        "",
        f"New split rows: train `{split['train']['rows']}`, val `{split['val']['rows']}`, test `{split['test']['rows']}`. Test domains `{split['test']['domains']}`.",
        "",
        "Important boundary: this is not a new model result. The old Stage43-C checkpoint remains UCY-heldout evidence only; a new Stage43 latent model must be trained/evaluated on this split before any multi-domain latent claim is allowed.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(RESEARCH_STATE, {})
    state["stage43_f_source_level_heldout_split"] = {
        "source": payload["source"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "pool": payload["pool"],
        "split_summary": payload["split_summary"],
        "no_leakage": payload["no_leakage"],
        "claim_boundary": payload["claim_boundary"],
        "report": str(REPORT_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    write_json(RESEARCH_STATE, state)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)
    return build_source_level_split()


if __name__ == "__main__":
    result = main()
    gate = result["stage43_f_gate"]
    print(f"Stage43-F source-level split: {gate['verdict']} ({gate['passed']}/{gate['total']})")
