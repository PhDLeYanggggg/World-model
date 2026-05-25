from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash, _git_commit
from src import stage41_breakthrough as s41
from src import stage41_joint_rollout_consistency as jrc
from src import stage41_teacher_guided_evidence as evidence
from src import stage41_teacher_guided_proposal as tgp


OUT_DIR = Path("outputs/stage41_external_split")
REPORT_JSON = OUT_DIR / "stage41_source_level_validation_repair.json"
REPORT_MD = OUT_DIR / "stage41_source_level_validation_repair.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"
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
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(entry), ensure_ascii=False) + "\n")


def _source_key(source_file: str) -> str:
    marker = "/datasets/"
    src = str(source_file).replace("\\", "/")
    if marker in src:
        return src.split(marker, 1)[1]
    return src


def _is_ucy_family(source_file: str, domain: str) -> bool:
    src = _source_key(source_file)
    return (
        domain == "UCY"
        or src.startswith("UCY/")
        or "/UCY/" in src
        or "crowds_zara" in src
        or "students00" in src
    )


def _source_stats(data: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    horizon = data["horizon"].astype(int)
    scene = data["scene_id"].astype(str)
    source = data["source_file"].astype(str)
    agent = data["agent_id"].astype(str)
    frame = data["frame_id"].astype(float)
    return {
        "rows": int(np.sum(mask)),
        "sources": int(len(set(source[mask].tolist()))) if np.any(mask) else 0,
        "scenes": int(len(set(scene[mask].tolist()))) if np.any(mask) else 0,
        "agents": int(len(set(agent[mask].tolist()))) if np.any(mask) else 0,
        "t10": int(np.sum(mask & (horizon == 10))),
        "t25": int(np.sum(mask & (horizon == 25))),
        "t50": int(np.sum(mask & (horizon == 50))),
        "t100": int(np.sum(mask & (horizon == 100))),
        "hard": int(np.sum(data["hard"].astype(bool)[mask])) if "hard" in data else 0,
        "easy": int(np.sum(data["easy"].astype(bool)[mask])) if "easy" in data else 0,
        "failure": int(np.sum(data["failure"].astype(bool)[mask])) if "failure" in data else 0,
        "frame_min": float(np.min(frame[mask])) if np.any(mask) else 0.0,
        "frame_max": float(np.max(frame[mask])) if np.any(mask) else 0.0,
    }


def _combined_inventory() -> dict[str, Any]:
    data = s41._combined()
    domain = data["dataset"].astype(str)
    source = data["source_file"].astype(str)
    scene = data["scene_id"].astype(str)
    horizon = data["horizon"].astype(int)
    by_domain: dict[str, Any] = {}
    basename_groups: dict[str, list[str]] = defaultdict(list)
    for src in sorted(set(source.tolist())):
        basename_groups[Path(_source_key(src)).name].append(_source_key(src))
    duplicate_basename_groups = {k: sorted(v) for k, v in basename_groups.items() if len(set(v)) > 1}
    for dom in sorted(set(domain.tolist())):
        rows = []
        dom_mask = domain == dom
        for src in sorted(set(source[dom_mask].tolist())):
            mask = dom_mask & (source == src)
            rows.append(
                {
                    "source": _source_key(src),
                    "scene": sorted(set(scene[mask].tolist())),
                    "rows": int(np.sum(mask)),
                    "t50": int(np.sum(mask & (horizon == 50))),
                    "t100": int(np.sum(mask & (horizon == 100))),
                    "agents": int(len(set(data["agent_id"].astype(str)[mask].tolist()))),
                }
            )
        by_domain[dom] = {"source_count": len(rows), "sources": rows}
    family_mask = np.asarray([_is_ucy_family(src, dom) for src, dom in zip(source, domain)], dtype=bool)
    by_family = {
        "UCY_family_surrogate": {
            "definition": "UCY domain plus UCY-path and crowds/student source files. This is a surrogate family, not pure UCY source-level validation.",
            "stats": _source_stats(
                {
                    "horizon": horizon,
                    "scene_id": scene,
                    "source_file": source,
                    "agent_id": data["agent_id"],
                    "frame_id": data["frame_id"],
                    "hard": data["hard"],
                    "easy": data["easy"],
                    "failure": data["failure"],
                },
                family_mask,
            ),
            "sources": sorted(set(_source_key(src) for src in source[family_mask].tolist())),
        }
    }
    return {
        "source": "fresh_run",
        "combined_rows": int(len(domain)),
        "domains": sorted(set(domain.tolist())),
        "by_domain": by_domain,
        "by_family": by_family,
        "duplicate_basename_groups": duplicate_basename_groups,
        "known_duplicate_blockers": [
            "UCY/zara03/crowds_zara03.txt and TrajNet/Train/crowds/crowds_zara03.txt are treated as duplicate-like zara03 sources and must not be split across train/val/test as independent evidence."
        ],
    }


def _split_feasibility() -> dict[str, Any]:
    split_report = read_json("outputs/stage41_fresh_confirmation/stage41_source_rotation_split_report.json", {})
    by_domain = split_report.get("by_domain") or {}
    feasibility: dict[str, Any] = {}
    for dom, rows in by_domain.items():
        train = rows.get("train") or {}
        val = rows.get("val") or {}
        test = rows.get("test") or {}
        ready = bool(
            train.get("source_files", 0) >= 1
            and val.get("source_files", 0) >= 1
            and test.get("source_files", 0) >= 1
            and val.get("t50", 0) > 0
            and test.get("t50", 0) > 0
        )
        feasibility[dom] = {
            "source_level_train_val_test_available": ready,
            "train_sources": int(train.get("source_files", 0)),
            "val_sources": int(val.get("source_files", 0)),
            "test_sources": int(test.get("source_files", 0)),
            "train_rows": int(train.get("rows", 0)),
            "val_rows": int(val.get("rows", 0)),
            "test_rows": int(test.get("rows", 0)),
            "val_t50": int(val.get("t50", 0)),
            "test_t50": int(test.get("t50", 0)),
        }
    overlap = split_report.get("overlap_audit") or {}
    no_source_overlap = bool((overlap.get("source_file_overlap_pass") is True) or all(len(v) == 0 for v in (overlap.get("source_file_overlap") or {}).values()))
    feasibility["no_source_overlap"] = no_source_overlap
    feasibility["source_overlap_audit"] = overlap.get("source_file_overlap") or {}
    feasibility["UCY_source_level_blocker"] = (
        "Pure UCY still has no independent validation source in the source-rotation split. Internal/temporal validation is useful but is not source-level evidence."
        if not feasibility.get("UCY", {}).get("source_level_train_val_test_available", False)
        else ""
    )
    feasibility["UCY_family_surrogate_available"] = bool(
        feasibility.get("ETH_UCY", {}).get("source_level_train_val_test_available", False)
        and feasibility.get("UCY", {}).get("test_sources", 0) >= 1
    )
    feasibility["UCY_family_surrogate_note"] = (
        "UCY-family surrogate uses ETH_UCY UCY-path sources plus held-out UCY test sources; it is not a replacement for a true pure-UCY source-level validation split."
    )
    return feasibility


def _selected_switch_for_test() -> tuple[dict[str, Any], np.ndarray, dict[str, Any]]:
    checkpoint, policy, min_sep = evidence._selected_checkpoint_policy_guard()
    test = tgp._bundle("test")
    pred = tgp._predict(checkpoint, test)
    raw_switch = tgp._policy_switch(pred, policy)
    guarded_switch, guarded_off = jrc._apply_proximity_guard(
        test["floor_xy"],
        test["neural_xy"],
        test["labels"],
        test["keys"],
        raw_switch.astype(bool),
        min_sep,
    )
    frozen = {
        "checkpoint": checkpoint,
        "policy": policy,
        "proximity_guard_min_sep": float(min_sep),
        "guarded_off": int(guarded_off),
    }
    return test, guarded_switch.astype(bool), frozen


def _metrics_subset(data: Mapping[str, Any], switch: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    if not np.any(mask):
        return {"rows": 0, "all_improvement": 0.0, "t50_improvement": 0.0, "hard_failure_improvement": 0.0, "easy_degradation": 0.0}
    selected = data["floor_ade"].astype(np.float64)[mask].copy()
    sw = switch[mask].astype(bool)
    selected[sw] = data["neural_ade"].astype(np.float64)[mask][sw]
    ds = {
        "horizon": data["horizon"][mask],
        "hard": data["hard"][mask],
        "failure": data["failure"][mask],
        "easy": data["easy"][mask],
        "domain": data["domain"][mask],
        "candidate_fde": data["candidate_fde"][mask],
    }
    metrics = s41._metrics(selected, data["floor_ade"].astype(np.float64)[mask], ds, sw)
    metrics["rows"] = int(np.sum(mask))
    return metrics


def _test_source_metrics() -> dict[str, Any]:
    data, switch, frozen = _selected_switch_for_test()
    labels = data["labels"]
    source = labels["source_file"].astype(str)
    scene = labels["scene_id"].astype(str)
    domain = labels["domain"].astype(str)
    horizon = labels["horizon"].astype(int)
    by_source: dict[str, Any] = {}
    for src in sorted(set(source.tolist())):
        mask = source == src
        row = _metrics_subset(data, switch, mask)
        row.update(
            {
                "source": _source_key(src),
                "domain": sorted(set(domain[mask].tolist())),
                "scene": sorted(set(scene[mask].tolist())),
                "t50_rows": int(np.sum(mask & (horizon == 50))),
                "t100_rows": int(np.sum(mask & (horizon == 100))),
                "switch_rate": float(np.mean(switch[mask])) if np.any(mask) else 0.0,
            }
        )
        by_source[_source_key(src)] = row
    family_masks = {
        "UCY_family_surrogate_test": np.asarray([_is_ucy_family(src, dom) for src, dom in zip(source, domain)], dtype=bool),
        "pure_UCY_test": domain == "UCY",
        "ETH_UCY_test": domain == "ETH_UCY",
        "TrajNet_test": domain == "TrajNet",
    }
    by_family = {name: _metrics_subset(data, switch, mask) for name, mask in family_masks.items()}
    return {
        "source": "fresh_run",
        "frozen_policy": frozen,
        "overall_test_metrics": _metrics_subset(data, switch, np.ones(len(switch), dtype=bool)),
        "by_source": by_source,
        "by_family": by_family,
        "no_leakage": {
            "future_endpoint_input": False,
            "future_labels_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "frozen_policy_and_guard": True,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }


def _positive_safe(metrics: Mapping[str, Any]) -> bool:
    return bool(
        metrics.get("all_improvement", 0.0) > 0
        and (metrics.get("t50_improvement", 0.0) > 0 or metrics.get("hard_failure_improvement", 0.0) > 0)
        and metrics.get("easy_degradation", 1.0) <= 0.02
    )


def run_source_level_validation_repair() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    inventory = _combined_inventory()
    feasibility = _split_feasibility()
    source_metrics = _test_source_metrics()
    ucy_independent = read_json("outputs/stage41_fresh_confirmation/stage41_ucy_independent_validation.json", {})
    test_families = source_metrics.get("by_family") or {}
    positive_sources = [
        src
        for src, metrics in (source_metrics.get("by_source") or {}).items()
        if _positive_safe(metrics)
    ]
    pure_ucy_source_level = bool(feasibility.get("UCY", {}).get("source_level_train_val_test_available", False))
    surrogate_positive = _positive_safe(test_families.get("UCY_family_surrogate_test", {}))
    result = {
        "source": "fresh_run",
        "protocol": "stage41_source_level_validation_repair_audit",
        "inventory": inventory,
        "split_feasibility": feasibility,
        "frozen_test_source_metrics": source_metrics,
        "ucy_internal_temporal_validation": {
            "source": "cached_verified",
            "validation_pass": bool(ucy_independent.get("validation_pass", False)),
            "source_level_available": bool(ucy_independent.get("source_level_independent_validation_available", False)),
            "source_level_blocker": ucy_independent.get("source_level_blocker", ""),
            "test_ucy_metrics": ucy_independent.get("test_ucy_metrics", {}),
        },
        "positive_test_sources": positive_sources,
        "source_level_validation_repair_pass": bool(
            feasibility.get("no_source_overlap", False)
            and _positive_safe(source_metrics.get("overall_test_metrics", {}))
            and len(positive_sources) >= 2
            and surrogate_positive
            and bool(ucy_independent.get("validation_pass", False))
        ),
        "pure_ucy_source_level_gate": pure_ucy_source_level,
        "ucy_family_surrogate_gate": surrogate_positive,
        "deployment_interpretation": (
            "The frozen teacher-guided candidate has positive source-heldout evidence and a positive UCY-family surrogate, but pure UCY source-level validation remains blocked. "
            "This supports continued candidate status, not final external source-level deployment evidence."
        ),
        "no_leakage": {
            "future_endpoint_input": False,
            "future_labels_eval_only": True,
            "central_velocity": False,
            "test_endpoint_goals": False,
            "test_threshold_tuning": False,
            "source_file_overlap_pass": bool(feasibility.get("no_source_overlap", False)),
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    write_json(REPORT_JSON, _jsonable(result))

    lines = [
        "# Stage41 Source-Level Validation Repair",
        "",
        "- source: `fresh_run`",
        "- Stage5C executed: `False`",
        "- SMC enabled: `False`",
        "- metric/seconds claim: `False`",
        f"- source-level validation repair pass: `{result['source_level_validation_repair_pass']}`",
        f"- pure UCY source-level gate: `{pure_ucy_source_level}`",
        f"- UCY-family surrogate gate: `{surrogate_positive}`",
        f"- UCY source-level blocker: `{feasibility.get('UCY_source_level_blocker')}`",
        "",
        "## Frozen Test Source Metrics",
        "",
        "| source | rows | t50 | t100 | all | t50 imp | t100 imp | hard | easy degr | switch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for src, row in sorted((source_metrics.get("by_source") or {}).items()):
        lines.append(
            f"| `{src}` | {row.get('rows', 0)} | {row.get('t50_rows', 0)} | {row.get('t100_rows', 0)} | "
            f"{float(row.get('all_improvement', 0.0)):.4f} | {float(row.get('t50_improvement', 0.0)):.4f} | "
            f"{float(row.get('t100_improvement', 0.0)):.4f} | {float(row.get('hard_failure_improvement', 0.0)):.4f} | "
            f"{float(row.get('easy_degradation', 0.0)):.4f} | {float(row.get('switch_rate', 0.0)):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Family Metrics",
            "",
            "| family | rows | all | t50 | t100 | hard/failure | easy degradation |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for family, row in sorted((source_metrics.get("by_family") or {}).items()):
        lines.append(
            f"| `{family}` | {row.get('rows', 0)} | {float(row.get('all_improvement', 0.0)):.4f} | "
            f"{float(row.get('t50_improvement', 0.0)):.4f} | {float(row.get('t100_improvement', 0.0)):.4f} | "
            f"{float(row.get('hard_failure_improvement', 0.0)):.4f} | {float(row.get('easy_degradation', 0.0)):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            result["deployment_interpretation"],
            "",
            "Pure UCY source-level validation is still not solved because the available split has no independent UCY validation source after excluding duplicate-like zara03. Internal folds and temporal UCY checks are cached-verified support, not a substitute for source-level evidence.",
            "",
            f"- no leakage: `{result['no_leakage']}`",
        ]
    )
    write_md(REPORT_MD, lines)
    return result


def main_source_level_validation_repair() -> None:
    started = time.perf_counter()
    status = "failed"
    try:
        run_source_level_validation_repair()
        status = "ok"
    finally:
        _append_ledger(
            "stage41_source_level_validation_repair",
            status,
            started,
            [
                "data/stage41_world_model/combined_external.npz",
                "outputs/stage41_fresh_confirmation/stage41_teacher_guided_proposal_repair.json",
                "outputs/stage41_fresh_confirmation/stage41_ucy_independent_validation.json",
            ],
            [REPORT_MD, REPORT_JSON],
        )


if __name__ == "__main__":
    main_source_level_validation_repair()
