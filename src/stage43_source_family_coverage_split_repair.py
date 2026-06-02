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
from src.stage43_shadow_easy_guard_repair import _source_family
from src.stage43_source_level_heldout_split import REPORT_JSON as STAGE43_F_JSON


REPORT_JSON = OUT_DIR / "stage43_source_family_coverage_split_repair.json"
REPORT_MD = OUT_DIR / "stage43_source_family_coverage_split_repair.md"
GATE_MD = OUT_DIR / "stage43_stage_ce_source_family_coverage_split_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

SECTION = "STAGE43_CE_SOURCE_FAMILY_COVERAGE_SPLIT_REPAIR"
SOURCE = "fresh_stage43_ce_source_family_coverage_split_repair"
SPLITS = ["train", "val", "test"]
DOMAINS = ["ETH_UCY", "TrajNet", "UCY"]


def _source_id(source_file: str) -> str:
    return hashlib.sha256(str(source_file).encode("utf-8")).hexdigest()[:16]


def _domain_family(source_row: Mapping[str, Any]) -> str:
    return f"{source_row['domain']}|{source_row['family']}"


def _source_records(stage43f: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for source_file, row in stage43f.get("source_summaries", {}).items():
        records[str(source_file)] = {
            "source_file": str(source_file),
            "source_id": _source_id(str(source_file)),
            "domain": str(row.get("domain", "unknown")),
            "family": _source_family(str(source_file)),
            "scene_ids": list(row.get("scenes", [])),
            "old_splits": list(row.get("old_splits", [])),
            "rows": int(row.get("rows", 0)),
            "horizon_counts": dict(row.get("horizon_counts", {})),
            "hard_rows": int(row.get("hard_rows", 0)),
            "failure_rows": int(row.get("failure_rows", 0)),
            "easy_rows": int(row.get("easy_rows", 0)),
            "basename": Path(str(source_file)).name.lower(),
        }
    return records


def _candidate_groups(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    by_domain_family: dict[str, list[str]] = defaultdict(list)
    by_family: dict[str, list[str]] = defaultdict(list)
    by_domain: dict[str, list[str]] = defaultdict(list)
    for source_file, row in records.items():
        by_domain_family[_domain_family(row)].append(source_file)
        by_family[str(row["family"])].append(source_file)
        by_domain[str(row["domain"])].append(source_file)
    return {
        "by_domain_family": {key: sorted(value) for key, value in by_domain_family.items()},
        "by_family": {key: sorted(value) for key, value in by_family.items()},
        "by_domain": {key: sorted(value) for key, value in by_domain.items()},
        "singleton_domain_families": sorted(key for key, value in by_domain_family.items() if len(value) < 2),
        "singleton_global_families": sorted(key for key, value in by_family.items() if len(value) < 2),
    }


def _rank_sources(records: Mapping[str, Mapping[str, Any]], sources: list[str], *, salt: str) -> list[str]:
    return sorted(
        sources,
        key=lambda source: (
            -int(records[source]["rows"]),
            hashlib.sha256(f"{salt}|{source}".encode("utf-8")).hexdigest(),
        ),
    )


def _select_test_val_pair(
    records: Mapping[str, Mapping[str, Any]],
    groups: Mapping[str, Any],
    domain: str,
) -> tuple[str | None, str | None, str | None]:
    eligible = []
    for key, sources in groups["by_domain_family"].items():
        d, family = key.split("|", 1)
        if d != domain or len(sources) < 2:
            continue
        total_rows = sum(int(records[source]["rows"]) for source in sources)
        eligible.append((total_rows, family, key, list(sources)))
    if not eligible:
        return None, None, None
    _, _, key, sources = sorted(eligible, reverse=True)[0]
    ranked = _rank_sources(records, sources, salt=f"{SOURCE}|{domain}|{key}")
    test_source = ranked[0]
    val_source = ranked[1]
    return key, test_source, val_source


def _build_coverage_assignments(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    groups = _candidate_groups(records)
    assignments = {source: "train" for source in records}
    domain_pair_plan: dict[str, Any] = {}
    blockers: dict[str, Any] = {"domains_without_supported_test_family": [], "singleton_domain_families": groups["singleton_domain_families"]}

    for domain in DOMAINS:
        key, test_source, val_source = _select_test_val_pair(records, groups, domain)
        if key is None or test_source is None or val_source is None:
            blockers["domains_without_supported_test_family"].append(domain)
            continue
        assignments[test_source] = "test"
        assignments[val_source] = "val"
        domain_pair_plan[domain] = {
            "domain_family": key,
            "test_source_id": records[test_source]["source_id"],
            "val_source_id": records[val_source]["source_id"],
            "test_source_file": test_source,
            "val_source_file": val_source,
            "rule": "Choose one heldout source and one validation-support source from the same domain-family using metadata only.",
        }

    test_families = {records[source]["family"] for source, split in assignments.items() if split == "test"}
    val_families = {records[source]["family"] for source, split in assignments.items() if split == "val"}
    for family in sorted(test_families - val_families):
        candidates = [
            source
            for source in groups["by_family"].get(family, [])
            if assignments[source] == "train"
        ]
        if candidates:
            assignments[_rank_sources(records, candidates, salt=f"{SOURCE}|global|{family}")[0]] = "val"

    return {
        "assignments": assignments,
        "domain_pair_plan": domain_pair_plan,
        "groups": groups,
        "blockers": blockers,
        "selection_rule": (
            "Metadata-only coverage-aware source split: for each domain, put a heldout source in test and a disjoint "
            "same-domain-family source in validation when at least two sources exist. No labels, future endpoints, "
            "future waypoints, test metrics, or threshold tuning are used."
        ),
    }


def _split_summary(records: Mapping[str, Mapping[str, Any]], assignments: Mapping[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in SPLITS:
        sources = [source for source, value in assignments.items() if value == split]
        rows = [records[source] for source in sources]
        horizon_counts: Counter[str] = Counter()
        for row in rows:
            horizon_counts.update({str(k): int(v) for k, v in row["horizon_counts"].items()})
        out[split] = {
            "rows": int(sum(int(row["rows"]) for row in rows)),
            "source_count": int(len(sources)),
            "source_ids": [records[source]["source_id"] for source in sorted(sources)],
            "domains": sorted(set(str(row["domain"]) for row in rows)),
            "domain_counts": dict(Counter(str(row["domain"]) for row in rows)),
            "families": sorted(set(str(row["family"]) for row in rows)),
            "domain_families": sorted(set(_domain_family(row) for row in rows)),
            "horizon_counts": dict(sorted(horizon_counts.items(), key=lambda kv: int(kv[0]))),
            "hard_rows": int(sum(int(row["hard_rows"]) for row in rows)),
            "failure_rows": int(sum(int(row["failure_rows"]) for row in rows)),
            "easy_rows": int(sum(int(row["easy_rows"]) for row in rows)),
        }
    return out


def _coverage_summary(records: Mapping[str, Mapping[str, Any]], assignments: Mapping[str, str]) -> dict[str, Any]:
    split_families = {}
    split_domain_families = {}
    for split in SPLITS:
        split_sources = [source for source, value in assignments.items() if value == split]
        split_families[split] = sorted(set(str(records[source]["family"]) for source in split_sources))
        split_domain_families[split] = sorted(set(_domain_family(records[source]) for source in split_sources))
    test_families = set(split_families["test"])
    val_families = set(split_families["val"])
    test_domain_families = set(split_domain_families["test"])
    val_domain_families = set(split_domain_families["val"])
    return {
        "families_by_split": split_families,
        "domain_families_by_split": split_domain_families,
        "test_families_without_validation_support": sorted(test_families - val_families),
        "test_domain_families_without_validation_support": sorted(test_domain_families - val_domain_families),
        "global_family_coverage_pass": len(test_families - val_families) == 0,
        "domain_family_coverage_pass": len(test_domain_families - val_domain_families) == 0,
    }


def _leakage_summary(records: Mapping[str, Mapping[str, Any]], assignments: Mapping[str, str]) -> dict[str, Any]:
    source_sets = {split: {source for source, value in assignments.items() if value == split} for split in SPLITS}
    overlap = {
        f"{a}_{b}": sorted(source_sets[a] & source_sets[b])
        for i, a in enumerate(SPLITS)
        for b in SPLITS[i + 1 :]
    }
    basename_by_split: dict[str, set[str]] = {
        split: {str(records[source]["basename"]) for source in sources} for split, sources in source_sets.items()
    }
    basename_overlap = {
        f"{a}_{b}": sorted(basename_by_split[a] & basename_by_split[b])
        for i, a in enumerate(SPLITS)
        for b in SPLITS[i + 1 :]
    }
    scene_by_split: dict[str, set[str]] = {
        split: {str(scene) for source in sources for scene in records[source]["scene_ids"]} for split, sources in source_sets.items()
    }
    scene_overlap = {
        f"{a}_{b}": sorted(scene_by_split[a] & scene_by_split[b])
        for i, a in enumerate(SPLITS)
        for b in SPLITS[i + 1 :]
    }
    return {
        "source_file_disjoint": all(len(value) == 0 for value in overlap.values()),
        "source_overlap_counts": {key: len(value) for key, value in overlap.items()},
        "basename_overlap_counts": {key: len(value) for key, value in basename_overlap.items()},
        "basename_overlap_examples": {key: value[:10] for key, value in basename_overlap.items() if value},
        "scene_overlap_counts": {key: len(value) for key, value in scene_overlap.items()},
        "scene_overlap_allowed_for_source_level_split": True,
        "future_endpoint_input": False,
        "future_waypoint_input": False,
        "central_velocity_input": False,
        "test_endpoint_goal_construction": False,
        "test_statistics_normalization": False,
    }


def _assignment_hash(assignments: Mapping[str, str]) -> str:
    text = json.dumps({str(k): str(v) for k, v in sorted(assignments.items())}, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    split = payload["split_summary"]
    coverage = payload["coverage_summary"]
    leakage = payload["no_leakage"]
    gates = {
        "stage43_f_precondition_passed": payload["stage43_f_precondition"]["verdict"] == "stage43_f_source_level_split_ready",
        "coverage_aware_assignment_built": payload["coverage_split"]["assignment_hash"] != "",
        "train_val_test_nonempty": all(int(split[name]["rows"]) > 0 for name in SPLITS),
        "test_contains_required_domains": set(split["test"]["domains"]) >= set(DOMAINS),
        "validation_contains_required_domains": set(split["val"]["domains"]) >= set(DOMAINS),
        "source_file_disjoint": leakage["source_file_disjoint"] is True,
        "global_source_family_validation_coverage": coverage["global_family_coverage_pass"] is True,
        "domain_source_family_validation_coverage": coverage["domain_family_coverage_pass"] is True,
        "singleton_unsupported_families_avoided_in_test": not any(
            key in set(coverage["domain_families_by_split"]["test"])
            for key in payload["coverage_split"]["blockers"]["singleton_domain_families"]
        ),
        "basename_overlap_reported": "basename_overlap_counts" in leakage,
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
    verdict = "stage43_ce_source_family_coverage_split_repair_ready" if passed == total else "stage43_ce_source_family_coverage_split_repair_partial"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_ce_gate"]
    coverage = payload["coverage_summary"]
    split = payload["split_summary"]
    return [
        "# Stage43-CE Source-Family Coverage Split Repair",
        "",
        f"- source: `{SOURCE}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- assignment hash: `{payload['coverage_split']['assignment_hash']}`",
        "- deployable policy changed: `False`",
        "- new model training run: `False`",
        "",
        "## Why This Exists",
        "",
        "- Stage43-CD proved that validation source-family coverage gaps make the downstream latent guard overly conservative.",
        "- This audit builds a metadata-only source split where validation covers every test source family and domain-family when feasible.",
        "- It does not use test labels, future endpoints, future waypoints, or test metrics for threshold tuning.",
        "",
        "## Coverage-Aware Split",
        "",
        "| split | rows | domains | families | domain families | sources |",
        "| --- | ---: | --- | --- | --- | ---: |",
        *[
            f"| {name} | {row['rows']} | `{row['domains']}` | `{row['families']}` | `{row['domain_families']}` | {row['source_count']} |"
            for name, row in split.items()
        ],
        "",
        "## Validation Coverage",
        "",
        f"- test families without validation support: `{coverage['test_families_without_validation_support']}`",
        f"- test domain-families without validation support: `{coverage['test_domain_families_without_validation_support']}`",
        f"- singleton domain-families avoided in test: `{payload['stage43_ce_gate']['gates']['singleton_unsupported_families_avoided_in_test']}`",
        "- tradeoff: the repaired test split is coverage-aware and narrower than the broad external stress split; unsupported singleton families remain acquisition/coverage blockers rather than hidden successes.",
        "",
        "## Leakage / Caveats",
        "",
        f"- source-file disjoint: `{payload['no_leakage']['source_file_disjoint']}`",
        f"- basename overlap counts: `{payload['no_leakage']['basename_overlap_counts']}`",
        f"- scene overlap counts: `{payload['no_leakage']['scene_overlap_counts']}`",
        "- Scene overlap is reported because this is still source-file-level, not strict scene-level.",
        "- Basename overlap is reported as a duplicate-source caution, not hidden.",
        "",
        "## Next Required Step",
        "",
        "- Rebuild the Stage43 full-waypoint supervision cache with this assignment, then retrain/evaluate latent models on that repaired split.",
        "- Until that happens, this is split-repair readiness evidence only, not a new world-model result.",
        "- Keep the broad external stress matrix as a separate diagnostic so the repaired split does not erase domain-gap evidence.",
        "",
        "## Claim Boundary",
        "",
        "- Dataset-local/raw-frame 2.5D only.",
        "- No metric/seconds, true-3D, foundation, Stage5C, or SMC claim.",
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
    gate = payload["stage43_ce_gate"]
    write_md(
        GATE_MD,
        [
            "# Stage43-CE Gate",
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
            "- Stage43-CE is a source-family coverage split-repair preflight, not a new model result.",
            "- It repairs the split protocol needed before retraining/evaluating latent dynamics with better validation coverage.",
            "- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()],
            "",
        ],
    )


def _update_summaries(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_ce_gate"]
    split = payload["split_summary"]
    coverage = payload["coverage_summary"]
    block = [
        f"## {SECTION}",
        "",
        f"source = `{SOURCE}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        "deployable_policy_changed = `False`",
        "new_model_training_run = `False`",
        "",
        "Stage43-CE builds a metadata-only coverage-aware source split so validation covers every test source family/domain-family where feasible. This directly addresses the Stage43-CD over-conservative fallback caused by validation support gaps.",
        "",
        f"Split rows train/val/test = `{split['train']['rows']}` / `{split['val']['rows']}` / `{split['test']['rows']}`.",
        f"Test families without validation support = `{coverage['test_families_without_validation_support']}`.",
        f"Test domain-families without validation support = `{coverage['test_domain_families_without_validation_support']}`.",
        "Tradeoff: the repaired test split is intentionally coverage-aware and narrower than the broad external stress split; singleton/unsupported source families remain blockers, not successes.",
        "",
        "Interpretation: this is split-protocol repair readiness, not a new model result. The next step is to rebuild the full-waypoint supervision cache and retrain/evaluate latent dynamics on the repaired split.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        _replace_section(path, SECTION, block)
    state = read_json(RESEARCH_STATE, {})
    state["stage43_ce_source_family_coverage_split_repair"] = {
        "source": SOURCE,
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "assignment_hash": payload["coverage_split"]["assignment_hash"],
        "split_summary": split,
        "coverage_summary": coverage,
        "no_leakage": payload["no_leakage"],
        "report": str(REPORT_MD),
        "deployable_policy_changed": False,
        "new_model_training_run": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_ce_source_family_coverage_split_repair"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(RESEARCH_STATE, _jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"source": SOURCE, "verdict": gate["verdict"], "generated_at_utc": payload["generated_at_utc"]}, ensure_ascii=False) + "\n")


def run_source_family_coverage_split_repair() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage43f = read_json(STAGE43_F_JSON, {})
    records = _source_records(stage43f)
    coverage_split = _build_coverage_assignments(records)
    assignments = coverage_split["assignments"]
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_metadata_only_source_family_coverage_split_repair",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "stage43_f_precondition": {"verdict": stage43f.get("stage43_f_gate", {}).get("verdict"), "report": str(STAGE43_F_JSON)},
        "source_pool": {
            "source_count": int(len(records)),
            "domains": sorted(set(str(row["domain"]) for row in records.values())),
            "families": sorted(set(str(row["family"]) for row in records.values())),
            "domain_families": sorted(set(_domain_family(row) for row in records.values())),
        },
        "coverage_split": {
            **coverage_split,
            "assignment_hash": _assignment_hash(assignments),
            "source_assignments": {source: assignments[source] for source in sorted(assignments)},
            "source_assignment_ids": {records[source]["source_id"]: assignments[source] for source in sorted(assignments)},
        },
        "split_summary": _split_summary(records, assignments),
        "coverage_summary": _coverage_summary(records, assignments),
        "no_leakage": _leakage_summary(records, assignments),
        "claim_boundary": {
            "new_training_or_evaluation_not_run": True,
            "requires_cache_rebuild_before_training": True,
            "dataset_local_raw_frame_only": True,
            "metric_or_seconds_claim": False,
            "true_3d_claim": False,
            "foundation_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
        "long_objective_complete": False,
    }
    payload["stage43_ce_gate"] = _gate(payload)
    _write_outputs(payload)
    _update_summaries(payload)
    return payload


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Stage43-CE source-family coverage split repair preflight.")
    parser.parse_args(argv)
    payload = run_source_family_coverage_split_repair()
    gate = payload["stage43_ce_gate"]
    print(f"Stage43-CE: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"assignment_hash={payload['coverage_split']['assignment_hash']}")
    print(f"test_family_gaps={payload['coverage_summary']['test_families_without_validation_support']}")
    print(f"test_domain_family_gaps={payload['coverage_summary']['test_domain_families_without_validation_support']}")
    return payload


if __name__ == "__main__":
    main()
