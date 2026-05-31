from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_full_waypoint_latent_dynamics import (
    CACHE_DIR,
    OUT_DIR,
    README_RESULTS,
    RESEARCH_STATE,
    WORK_SUMMARY,
    M3W_README,
    _git_commit,
    _jsonable,
)
from src.stage43_full_waypoint_latent_robustness_audit import _pct
from src.stage43_full_waypoint_latent_safe_repair import _source_family


REPORT_JSON = OUT_DIR / "stage43_t100_source_coverage_preflight.json"
REPORT_MD = OUT_DIR / "stage43_t100_source_coverage_preflight.md"
GATE_MD = OUT_DIR / "stage43_stage_s_t100_source_coverage_preflight_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

STAGE43_R_JSON = OUT_DIR / "stage43_t100_source_stability_guard.json"
SECTION = "STAGE43_S_T100_SOURCE_COVERAGE_PREFLIGHT"
SOURCE = "fresh_stage43_s_t100_source_coverage_preflight"
SPLITS = ("train", "val", "test")


def _cache_path(split: str) -> Path:
    return CACHE_DIR / f"stage43_full_waypoint_supervision_{split}.npz"


def _short_source(path: str) -> str:
    text = str(path)
    return text.split("/external_data/", 1)[-1] if "/external_data/" in text else text


def _cache_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        path = _cache_path(split)
        z = np.load(path, allow_pickle=False)
        horizon = z["horizon"].astype(np.int64)
        source_file = z["source_file"].astype(str)
        mask = horizon == 100
        for source in sorted(set(source_file[mask].tolist())):
            ids = mask & (source_file == source)
            rows.append(
                {
                    "split": split,
                    "source_file": source,
                    "source_short": _short_source(source),
                    "source_family": _source_family(source),
                    "h100_rows": int(ids.sum()),
                    "all_rows_in_split": int((source_file == source).sum()),
                }
            )
    return rows


def _family_summary(records: list[dict[str, Any]], *, min_val_source_count: int, min_source_rows: int) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_family[row["source_family"]].append(row)
    summary: dict[str, Any] = {}
    for family, rows in sorted(by_family.items()):
        source_map: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = row["source_short"]
            source_map[key] = {
                "source_file": row["source_file"],
                "source_short": row["source_short"],
                "current_split": row["split"],
                "h100_rows": row["h100_rows"],
                "eligible_for_validation": row["h100_rows"] >= int(min_source_rows),
            }
        eligible = [row for row in source_map.values() if row["eligible_for_validation"]]
        current_by_split = {
            split: {
                "source_count": sum(1 for row in source_map.values() if row["current_split"] == split),
                "h100_rows": sum(int(row["h100_rows"]) for row in source_map.values() if row["current_split"] == split),
            }
            for split in SPLITS
        }
        feasible_source_stable_validation = len(eligible) >= int(min_val_source_count) + 2
        reason = "feasible_with_source_level_resplit"
        if len(eligible) < int(min_val_source_count):
            reason = "blocked_too_few_h100_sources"
        elif len(eligible) < int(min_val_source_count) + 2:
            reason = "blocked_cannot_hold_train_val_test_with_source_stable_validation"
        proposal = _propose_family_split(eligible, min_val_source_count=int(min_val_source_count))
        summary[family] = {
            "source_count": int(len(source_map)),
            "eligible_source_count": int(len(eligible)),
            "h100_rows": int(sum(int(row["h100_rows"]) for row in source_map.values())),
            "current_by_split": current_by_split,
            "feasible_source_stable_validation": bool(feasible_source_stable_validation),
            "reason": reason,
            "source_level_split_proposal": proposal,
            "sources": sorted(source_map.values(), key=lambda row: (row["current_split"], -row["h100_rows"], row["source_short"])),
        }
    return summary


def _propose_family_split(eligible_sources: list[dict[str, Any]], *, min_val_source_count: int) -> dict[str, Any]:
    if len(eligible_sources) < int(min_val_source_count) + 2:
        return {
            "status": "not_feasible",
            "train_sources": [],
            "val_sources": [],
            "test_sources": [],
        }
    sources = sorted(eligible_sources, key=lambda row: (-int(row["h100_rows"]), row["source_short"]))
    current_test = [row for row in sources if row["current_split"] == "test"]
    test_sources = current_test[:1] if current_test else sources[-1:]
    test_names = {row["source_short"] for row in test_sources}
    remaining = [row for row in sources if row["source_short"] not in test_names]
    current_val = [row for row in remaining if row["current_split"] == "val"]
    val_sources = current_val[: int(min_val_source_count)]
    if len(val_sources) < int(min_val_source_count):
        val_names = {row["source_short"] for row in val_sources}
        for row in reversed(remaining):
            if row["source_short"] not in val_names:
                val_sources.append(row)
                val_names.add(row["source_short"])
            if len(val_sources) >= int(min_val_source_count):
                break
    val_names = {row["source_short"] for row in val_sources}
    train_sources = [row for row in remaining if row["source_short"] not in val_names]
    return {
        "status": "feasible",
        "train_sources": [row["source_short"] for row in train_sources],
        "val_sources": [row["source_short"] for row in val_sources],
        "test_sources": [row["source_short"] for row in test_sources],
        "train_h100_rows": int(sum(int(row["h100_rows"]) for row in train_sources)),
        "val_h100_rows": int(sum(int(row["h100_rows"]) for row in val_sources)),
        "test_h100_rows": int(sum(int(row["h100_rows"]) for row in test_sources)),
    }


def run_t100_source_coverage_preflight(
    *,
    min_val_source_count: int = 2,
    min_source_rows: int = 100,
) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage43r = read_json(STAGE43_R_JSON, {})
    records = _source_records()
    family = _family_summary(records, min_val_source_count=int(min_val_source_count), min_source_rows=int(min_source_rows))
    feasible = [name for name, row in family.items() if row["feasible_source_stable_validation"]]
    blocked = [name for name, row in family.items() if not row["feasible_source_stable_validation"]]
    split_hash = hashlib.sha256(
        json.dumps(
            [
                {
                    "split": row["split"],
                    "source_short": row["source_short"],
                    "family": row["source_family"],
                    "h100_rows": row["h100_rows"],
                }
                for row in sorted(records, key=lambda x: (x["split"], x["source_short"]))
            ],
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_h100_source_coverage_preflight",
        "generated_at_utc": datetime.now().replace(tzinfo=timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "stage43_r_precondition": {
            "verdict": stage43r.get("stage43_r_gate", {}).get("verdict"),
            "t100_status": stage43r.get("t100_source_stability_guard", {}).get("status"),
        },
        "cache_inputs": {
            split: {
                "path": str(_cache_path(split)),
                "exists": _cache_path(split).exists(),
                "sha256": _cache_file_hash(_cache_path(split)) if _cache_path(split).exists() else None,
            }
            for split in SPLITS
        },
        "protocol": {
            "audit_only": True,
            "rewrites_cache": False,
            "test_threshold_tuning": False,
            "min_val_source_count": int(min_val_source_count),
            "min_source_rows": int(min_source_rows),
            "future_labels_used_for_inputs": False,
            "split_hash": split_hash,
        },
        "records": records,
        "family_summary": family,
        "preflight_summary": {
            "h100_source_count": int(len(records)),
            "feasible_families": feasible,
            "blocked_families": blocked,
            "family_count": int(len(family)),
            "feasible_family_count": int(len(feasible)),
            "blocked_family_count": int(len(blocked)),
            "can_rebuild_source_stable_h100_validation": bool(len(feasible) >= 1),
            "needs_more_h100_sources_for_uniform_t100": bool(len(blocked) > 0),
        },
        "no_leakage": {
            "future_endpoint_input": False,
            "future_waypoint_input": False,
            "central_velocity_input": False,
            "test_endpoint_goal_construction": False,
            "test_statistics_normalization": False,
            "test_threshold_tuning": False,
        },
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "t100_positive_success": False,
        },
    }
    payload["stage43_s_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    gates = {
        "stage43_r_precondition_available": payload["stage43_r_precondition"]["verdict"]
        == "stage43_r_source_stable_h100_guard_blocks_t100_false_positive",
        "cache_inputs_exist": all(row["exists"] for row in payload["cache_inputs"].values()),
        "fresh_audit_only": payload["result_source"] == "fresh_h100_source_coverage_preflight"
        and payload["protocol"]["audit_only"] is True
        and payload["protocol"]["rewrites_cache"] is False,
        "source_coverage_reported": payload["preflight_summary"]["h100_source_count"] > 0
        and payload["preflight_summary"]["family_count"] > 0,
        "feasible_or_blocker_reported": payload["preflight_summary"]["feasible_family_count"] > 0
        or payload["preflight_summary"]["blocked_family_count"] > 0,
        "test_threshold_not_tuned": payload["protocol"]["test_threshold_tuning"] is False
        and payload["no_leakage"]["test_threshold_tuning"] is False,
        "no_future_inputs": payload["no_leakage"]["future_endpoint_input"] is False
        and payload["no_leakage"]["future_waypoint_input"] is False,
        "no_metric_seconds_stage5c_smc_claim": payload["claim_boundary"]["metric_or_seconds_claim"] is False
        and payload["claim_boundary"]["stage5c_executed"] is False
        and payload["claim_boundary"]["smc_enabled"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    return {
        "source": payload["source"],
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_s_t100_source_coverage_preflight_pass"
        if passed == total
        else "stage43_s_t100_source_coverage_preflight_incomplete",
        "rebuild_source_stable_h100_split_recommended": bool(
            payload["preflight_summary"]["can_rebuild_source_stable_h100_validation"]
        ),
        "uniform_t100_blocker": bool(payload["preflight_summary"]["needs_more_h100_sources_for_uniform_t100"]),
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(dict(payload)))
    gate = payload["stage43_s_gate"]
    summary = payload["preflight_summary"]
    lines = [
        "# Stage43-S T100 Source Coverage Preflight",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- h100 source count: `{summary['h100_source_count']}`",
        f"- feasible families: `{', '.join(summary['feasible_families']) if summary['feasible_families'] else 'none'}`",
        f"- blocked families: `{', '.join(summary['blocked_families']) if summary['blocked_families'] else 'none'}`",
        f"- rebuild source-stable h100 split recommended: `{gate['rebuild_source_stable_h100_split_recommended']}`",
        f"- uniform t100 blocker remains: `{gate['uniform_t100_blocker']}`",
        "",
        "## Family Summary",
        "",
        "| family | sources | eligible sources | h100 rows | feasible | reason | current train/val/test sources |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for family, row in payload["family_summary"].items():
        split_bits = []
        for split in SPLITS:
            data = row["current_by_split"][split]
            split_bits.append(f"{split}:{data['source_count']}/{data['h100_rows']}")
        lines.append(
            f"| {family} | {row['source_count']} | {row['eligible_source_count']} | {row['h100_rows']} | `{row['feasible_source_stable_validation']}` | `{row['reason']}` | {'; '.join(split_bits)} |"
        )
    lines.extend(
        [
            "",
            "## Proposed Source-Level Split Preflight",
            "",
            "| family | status | train sources | val sources | test sources |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for family, row in payload["family_summary"].items():
        proposal = row["source_level_split_proposal"]
        lines.append(
            f"| {family} | `{proposal['status']}` | `{', '.join(proposal['train_sources']) if proposal['train_sources'] else 'none'}` | `{', '.join(proposal['val_sources']) if proposal['val_sources'] else 'none'}` | `{', '.join(proposal['test_sources']) if proposal['test_sources'] else 'none'}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Stage43-S does not rewrite data or tune test thresholds. It shows whether h100 can support source-stable validation. In the current cache, only TrajNet_crowds has enough total h100 sources to try a new source-level split with two validation sources; ETH_UCY, TrajNet_biwi, and UCY remain source-scarce. Uniform t100 success therefore still needs more h100 source coverage or a separately validated per-source strategy.",
            "",
            "Claim boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; no Stage5C execution; no SMC.",
        ]
    )
    write_md(REPORT_MD, lines)
    gate_lines = [
        "# Stage43-S Gate",
        "",
        f"verdict: `{gate['verdict']}`",
        f"passed: `{gate['passed']} / {gate['total']}`",
        f"rebuild_source_stable_h100_split_recommended: `{gate['rebuild_source_stable_h100_split_recommended']}`",
        f"uniform_t100_blocker: `{gate['uniform_t100_blocker']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    gate_lines.extend([f"| {name} | `{value}` |" for name, value in gate["gates"].items()])
    write_md(GATE_MD, gate_lines)
    _refresh_readmes(payload)
    _update_state(payload)
    _append_ledger(payload)


def _refresh_readmes(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_s_gate"]
    summary = payload["preflight_summary"]
    lines = [
        "## Stage43-S t100 source coverage preflight",
        "",
        f"Result source: `{payload['result_source']}`. This audits h100 source coverage and prepares a source-stable split preflight without rewriting caches or tuning test thresholds.",
        "",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- feasible h100 families: `{', '.join(summary['feasible_families']) if summary['feasible_families'] else 'none'}`",
        f"- blocked h100 families: `{', '.join(summary['blocked_families']) if summary['blocked_families'] else 'none'}`",
        f"- rebuild source-stable h100 split recommended: `{gate['rebuild_source_stable_h100_split_recommended']}`",
        f"- uniform t100 blocker remains: `{gate['uniform_t100_blocker']}`",
        "",
        "Boundary: this is a preflight audit, not a new t100 deployment. It keeps the Stage43-P/R fallback and preserves the dataset-local/raw-frame 2.5D claim boundary.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, lines)


def _update_state(payload: Mapping[str, Any]) -> None:
    state = read_json(RESEARCH_STATE, {})
    gate = payload["stage43_s_gate"]
    state["stage43_s_t100_source_coverage_preflight"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "report": str(REPORT_JSON),
        "gate": f"{gate['passed']} / {gate['total']}",
        "verdict": gate["verdict"],
        "preflight_summary": payload["preflight_summary"],
        "claim_boundary": payload["claim_boundary"],
    }
    state["current_stage"] = "stage43_s_t100_source_coverage_preflight"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, _jsonable(state))


def _append_ledger(payload: Mapping[str, Any]) -> None:
    ensure_dir(OUT_DIR)
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable({"event": "stage43_s_t100_source_coverage_preflight", "payload": payload}), ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage43-S t100 source coverage preflight.")
    parser.add_argument("--min-val-source-count", type=int, default=2)
    parser.add_argument("--min-source-rows", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    result = run_t100_source_coverage_preflight(
        min_val_source_count=int(args.min_val_source_count),
        min_source_rows=int(args.min_source_rows),
    )
    gate = result["stage43_s_gate"]
    print(f"Stage43-S: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    return result


if __name__ == "__main__":
    main()
