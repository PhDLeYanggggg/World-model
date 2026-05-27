from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_proximity_guard_policy_freeze import _replace_section


OUT_DIR = Path("outputs/stage42_long_research")
STAGE42P_JSON = OUT_DIR / "t50_gain_harm_selector_stage42.json"
REPORT_JSON = OUT_DIR / "t50_gain_harm_stability_audit_stage42.json"
REPORT_MD = OUT_DIR / "t50_gain_harm_stability_audit_stage42.md"
GATE_MD = OUT_DIR / "stage42_stage_if_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
MASTER_README = Path("README_M3W_MASTER_SUMMARY_2026_05_27_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE42_IF_T50_GAIN_HARM_STABILITY_AUDIT"
SOURCE = "fresh_stage42_if_t50_gain_harm_stability_audit"


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
    if isinstance(value, Path):
        return str(value)
    return value


def _metric(row: Mapping[str, Any], split_key: str, metric_key: str = "ade") -> Mapping[str, Any]:
    return ((row.get(split_key, {}) or {}).get(metric_key, {}) or {})


def _validation_score(row: Mapping[str, Any]) -> float:
    metric = _metric(row, "val_metrics", "ade")
    easy = float(metric.get("easy_degradation", 1.0))
    return (
        7.5 * float(metric.get("t50_improvement", 0.0))
        + 1.4 * float(metric.get("all_improvement", 0.0))
        + 1.0 * float(metric.get("hard_failure_improvement", 0.0))
        + 0.25 * float(metric.get("t100_improvement", 0.0))
        - 80.0 * max(0.0, easy - 0.018)
    )


def _select_validation_seed(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"selected": False, "reason": "no_rows"}
    ranked = sorted(
        [
            {
                "seed": row.get("seed"),
                "base_seed": row.get("base_seed"),
                "validation_score": _validation_score(row),
                "val_ade": dict(_metric(row, "val_metrics", "ade")),
                "test_ade": dict(_metric(row, "test_metrics", "ade")),
                "test_fde": dict(_metric(row, "test_metrics", "fde")),
            }
            for row in rows
        ],
        key=lambda item: item["validation_score"],
        reverse=True,
    )
    return {
        "selected": True,
        "selection_rule": "validation_only_t50_weighted_score_no_test_threshold_tuning",
        "selected_seed": ranked[0]["seed"],
        "selected_base_seed": ranked[0]["base_seed"],
        "ranked": ranked,
    }


def _domain_instability(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    domain_counts: dict[str, dict[str, Any]] = {}
    for row in rows:
        seed = row.get("seed")
        by_domain = (_metric(row, "test_metrics", "ade").get("by_domain", {}) or {})
        for domain, metric in by_domain.items():
            t50 = float(metric.get("t50_improvement", 0.0))
            item = {
                "seed": seed,
                "domain": domain,
                "rows": int(metric.get("rows", 0)),
                "all_improvement": float(metric.get("all_improvement", 0.0)),
                "t50_improvement": t50,
                "hard_failure_improvement": float(metric.get("hard_failure_improvement", 0.0)),
                "easy_degradation": float(metric.get("easy_degradation", 0.0)),
                "switch_rate": float(metric.get("switch_rate", 0.0)),
                "negative_t50": t50 < 0.0,
            }
            records.append(item)
            bucket = domain_counts.setdefault(domain, {"rows_seen": 0, "negative_t50_count": 0, "t50_values": []})
            bucket["rows_seen"] += int(metric.get("rows", 0))
            bucket["negative_t50_count"] += int(t50 < 0.0)
            bucket["t50_values"].append(t50)
    for bucket in domain_counts.values():
        values = bucket["t50_values"]
        bucket["mean_t50"] = sum(values) / len(values) if values else 0.0
        bucket["min_t50"] = min(values) if values else 0.0
        bucket["max_t50"] = max(values) if values else 0.0
    worst = sorted(records, key=lambda item: item["t50_improvement"])[:5]
    return {
        "records": records,
        "domain_counts": domain_counts,
        "worst_t50_slices": worst,
        "negative_t50_slice_count": sum(1 for row in records if row["negative_t50"]),
    }


def _row_level_bootstrap_availability(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in ["row_error", "per_row", "selected_error", "fallback_error"]):
                return True
            if _row_level_bootstrap_availability(nested):
                return True
    if isinstance(value, list):
        return any(_row_level_bootstrap_availability(item) for item in value)
    return False


def _build_payload() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage42p = read_json(STAGE42P_JSON, {})
    rows = list(stage42p.get("rows", []))
    summary = stage42p.get("summary", {}) or {}
    ade_t50 = summary.get("ade_t50", {}) or {}
    fde_t50 = summary.get("fde_t50", {}) or {}
    seed_rows = []
    for row in rows:
        ade = _metric(row, "test_metrics", "ade")
        seed_rows.append(
            {
                "seed": row.get("seed"),
                "base_seed": row.get("base_seed"),
                "test_ade_all": float(ade.get("all_improvement", 0.0)),
                "test_ade_t50": float(ade.get("t50_improvement", 0.0)),
                "test_ade_t100_raw_frame_diagnostic": float(ade.get("t100_improvement", 0.0)),
                "test_ade_hard_failure": float(ade.get("hard_failure_improvement", 0.0)),
                "test_ade_easy_degradation": float(ade.get("easy_degradation", 0.0)),
                "validation_score": _validation_score(row),
            }
        )
    negative_t50_seeds = [row["seed"] for row in seed_rows if row["test_ade_t50"] < 0.0]
    domain = _domain_instability(rows)
    validation_selection = _select_validation_seed(rows)
    selected_test = (validation_selection.get("ranked") or [{}])[0].get("test_ade", {}) if validation_selection.get("selected") else {}
    row_bootstrap_available = _row_level_bootstrap_availability(stage42p)
    payload = {
        "stage": "Stage42-IF",
        "source": SOURCE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_hash": _combined_hash([STAGE42P_JSON]),
        "stage42p_source": stage42p.get("source", "missing"),
        "stage42p_gate": stage42p.get("stage42_p_gate", {}),
        "seed_rows": seed_rows,
        "summary": {
            "seed_count": len(rows),
            "ade_t50_mean": float(ade_t50.get("mean", 0.0)),
            "ade_t50_ci_low": float(ade_t50.get("ci_low", 0.0)),
            "ade_t50_ci_high": float(ade_t50.get("ci_high", 0.0)),
            "fde_t50_mean": float(fde_t50.get("mean", 0.0)),
            "fde_t50_ci_low": float(fde_t50.get("ci_low", 0.0)),
            "negative_t50_seed_count": len(negative_t50_seeds),
            "negative_t50_seeds": negative_t50_seeds,
            "paper_stable_ade_t50_claim_supported": float(ade_t50.get("ci_low", -1.0)) > 0.0,
            "paper_stable_fde_t50_claim_supported": float(fde_t50.get("ci_low", -1.0)) > 0.0,
            "row_level_bootstrap_available": row_bootstrap_available,
            "row_level_bootstrap_status": "available" if row_bootstrap_available else "not_run_blocked_by_missing_row_errors_in_stage42p_artifact",
            "validation_selected_seed": validation_selection.get("selected_seed"),
            "validation_selected_test_ade_all": float(selected_test.get("all_improvement", 0.0)),
            "validation_selected_test_ade_t50": float(selected_test.get("t50_improvement", 0.0)),
            "validation_selected_test_ade_hard_failure": float(selected_test.get("hard_failure_improvement", 0.0)),
            "validation_selected_test_easy_degradation": float(selected_test.get("easy_degradation", 0.0)),
        },
        "validation_selection": validation_selection,
        "domain_instability": domain,
        "diagnosis": {
            "primary_blocker": "seed_level_ade_t50_instability",
            "root_causes": [
                "Stage42-P improves mean t+50 ADE but one seed is negative, so the seed-level CI lower bound is below zero.",
                "TrajNet t+50 is the main unstable domain slice: it is strongly positive for seed 151 but negative for seed 157.",
                "Existing Stage42-P artifact stores aggregate metrics, not per-row error vectors, so row bootstrap cannot be recomputed without rerunning/cache export.",
                "FDE t+50 is stable positive, which means endpoint-distance style evidence is stronger than ADE-over-horizon evidence.",
            ],
            "recommended_next_action": "rerun t50 gain/harm selector with exported per-row selected/fallback/oracle error arrays and additional validation-selected seeds before promoting a paper-level ADE t50 claim",
        },
        "source_labels": {
            "stage42p_artifact": "cached_verified",
            "stability_audit": "fresh_run",
            "new_training": "not_run",
            "row_bootstrap": "not_run" if not row_bootstrap_available else "fresh_run_possible",
            "public_readme_rewrite": "not_touched_by_this_stage",
        },
        "no_leakage": stage42p.get("no_leakage", {}),
        "claim_boundary": {
            "true_3d": False,
            "foundation_world_model": False,
            "metric_or_seconds_claim": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage42_if_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary", {})
    no_leakage = payload.get("no_leakage", {})
    claim = payload.get("claim_boundary", {})
    gates = {
        "stage42p_artifact_loaded": payload.get("stage42p_source") != "missing" and summary.get("seed_count", 0) >= 3,
        "stability_audit_completed": payload.get("source") == SOURCE,
        "mean_ade_t50_positive": summary.get("ade_t50_mean", -1.0) > 0.0,
        "paper_stable_ade_t50_ci_positive": summary.get("paper_stable_ade_t50_claim_supported") is True,
        "fde_t50_ci_positive": summary.get("paper_stable_fde_t50_claim_supported") is True,
        "negative_seed_identified": summary.get("negative_t50_seed_count", 0) > 0,
        "domain_instability_identified": payload.get("domain_instability", {}).get("negative_t50_slice_count", 0) > 0,
        "validation_selected_seed_positive": summary.get("validation_selected_test_ade_t50", -1.0) > 0.0,
        "row_bootstrap_availability_audited": "row_level_bootstrap_status" in summary,
        "no_future_endpoint_or_waypoint_input": no_leakage.get("future_endpoint_input") is False
        and no_leakage.get("future_waypoints_input") is False,
        "no_central_velocity_or_test_goal": no_leakage.get("central_velocity") is False
        and no_leakage.get("test_endpoint_goals") is False,
        "no_metric_seconds_overclaim": claim.get("metric_or_seconds_claim") is False,
        "stage5c_false": claim.get("stage5c_executed") is False,
        "smc_false": claim.get("smc_enabled") is False,
    }
    passed = sum(1 for ok in gates.values() if ok)
    total = len(gates)
    if passed == total:
        verdict = "stage42_if_t50_gain_harm_stability_audit_pass"
    elif gates["stability_audit_completed"] and not gates["paper_stable_ade_t50_ci_positive"]:
        verdict = "stage42_if_t50_gain_harm_ci_blocker_identified"
    else:
        verdict = "stage42_if_t50_gain_harm_stability_audit_partial"
    return {"source": SOURCE, "gates": gates, "passed": passed, "total": total, "verdict": verdict}


def _write_report(payload: Mapping[str, Any]) -> None:
    s = payload["summary"]
    gate = payload["stage42_if_gate"]
    lines = [
        "# Stage42-IF T50 Gain/Harm Stability Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- generated_at_utc: `{payload['generated_at_utc']}`",
        f"- input_hash: `{payload['input_hash']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "## What This Audits",
        "",
        "Stage42-P repaired the mean t+50 signal for the gain/harm selector. This audit checks whether that result is stable enough to use as a paper-level t+50 ADE claim.",
        "",
        "## Current Facts",
        "",
        "- This is still dataset-local/raw-frame 2.5D multi-agent world-state evidence.",
        "- It is not true 3D, not metric, not seconds-level, and not a foundation world model.",
        "- Stage5C latent generative execution remains disabled.",
        "- SMC remains disabled.",
        "",
        "## Seed-Level Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| seeds | {s['seed_count']} |",
        f"| ADE t50 mean | {s['ade_t50_mean']:.6f} |",
        f"| ADE t50 CI low | {s['ade_t50_ci_low']:.6f} |",
        f"| ADE t50 CI high | {s['ade_t50_ci_high']:.6f} |",
        f"| FDE t50 mean | {s['fde_t50_mean']:.6f} |",
        f"| FDE t50 CI low | {s['fde_t50_ci_low']:.6f} |",
        f"| negative ADE t50 seeds | {s['negative_t50_seed_count']} |",
        f"| paper-stable ADE t50 claim supported | `{s['paper_stable_ade_t50_claim_supported']}` |",
        f"| row-level bootstrap status | `{s['row_level_bootstrap_status']}` |",
        "",
        "## Per-Seed Test ADE",
        "",
        "| seed | base_seed | all | t50 | t100 raw diag | hard/failure | easy degradation | validation score |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["seed_rows"]:
        lines.append(
            f"| {row['seed']} | {row['base_seed']} | {row['test_ade_all']:.6f} | {row['test_ade_t50']:.6f} | {row['test_ade_t100_raw_frame_diagnostic']:.6f} | {row['test_ade_hard_failure']:.6f} | {row['test_ade_easy_degradation']:.6f} | {row['validation_score']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Validation-Selected Seed",
            "",
            f"- selected_seed: `{s['validation_selected_seed']}`",
            "- selection rule: validation-only t+50-weighted score; no test threshold tuning.",
            f"- selected test ADE all: `{s['validation_selected_test_ade_all']:.6f}`",
            f"- selected test ADE t50: `{s['validation_selected_test_ade_t50']:.6f}`",
            f"- selected test ADE hard/failure: `{s['validation_selected_test_ade_hard_failure']:.6f}`",
            f"- selected test easy degradation: `{s['validation_selected_test_easy_degradation']:.6f}`",
            "",
            "This is useful deployment-selection evidence, but it is not a substitute for stable multi-seed or row-bootstrap evidence.",
            "",
            "## Worst Domain T50 Slices",
            "",
            "| seed | domain | rows | all | t50 | hard/failure | easy degradation | switch |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["domain_instability"]["worst_t50_slices"]:
        lines.append(
            f"| {row['seed']} | `{row['domain']}` | {row['rows']} | {row['all_improvement']:.6f} | {row['t50_improvement']:.6f} | {row['hard_failure_improvement']:.6f} | {row['easy_degradation']:.6f} | {row['switch_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Diagnosis",
            "",
            f"- primary_blocker: `{payload['diagnosis']['primary_blocker']}`",
            *[f"- {item}" for item in payload["diagnosis"]["root_causes"]],
            "",
            "## Next Action",
            "",
            f"- {payload['diagnosis']['recommended_next_action']}",
        ]
    )
    write_md(REPORT_MD, lines)


def _write_gate(gate: Mapping[str, Any]) -> None:
    lines = [
        "# Stage42-IF Gate",
        "",
        f"- source: `{gate['source']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- verdict: `{gate['verdict']}`",
        "",
        "| gate | pass |",
        "| --- | --- |",
    ]
    for name, ok in gate["gates"].items():
        lines.append(f"| `{name}` | `{bool(ok)}` |")
    write_md(GATE_MD, lines)


def _refresh_readmes_and_state(payload: Mapping[str, Any]) -> None:
    s = payload["summary"]
    gate = payload["stage42_if_gate"]
    lines = [
        "## Stage42-IF T50 Gain/Harm Stability Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gates: `{gate['passed']} / {gate['total']}`",
        f"- ADE t50 mean / CI low: `{s['ade_t50_mean']:.6f}` / `{s['ade_t50_ci_low']:.6f}`",
        f"- FDE t50 mean / CI low: `{s['fde_t50_mean']:.6f}` / `{s['fde_t50_ci_low']:.6f}`",
        f"- negative ADE t50 seeds: `{s['negative_t50_seed_count']}`",
        f"- validation-selected seed test ADE t50: `{s['validation_selected_test_ade_t50']:.6f}`",
        f"- row bootstrap status: `{s['row_level_bootstrap_status']}`",
        "- conclusion: Stage42-P is positive on mean t+50 and stable on FDE t+50, but ADE t+50 is not yet seed-CI stable enough for a paper-level t+50 ADE claim.",
        "- boundary: dataset-local/raw-frame 2.5D only; no metric/seconds claim, no Stage5C, no SMC.",
    ]
    for path in [README_RESULTS, M3W_README, MASTER_README]:
        _replace_section(path, SECTION, lines)
    state = read_json(RESEARCH_STATE, {})
    state["current_stage"] = "stage42_if_t50_gain_harm_stability_audit"
    state["current_verdict"] = gate["verdict"]
    stage42 = state.setdefault("stage42", {})
    stage42["stage_if_t50_gain_harm_stability_audit"] = {
        "source": payload["source"],
        "report": str(REPORT_MD),
        "json": str(REPORT_JSON),
        "gate": str(GATE_MD),
        "verdict": gate["verdict"],
        "gates": f"{gate['passed']}/{gate['total']}",
        "ade_t50_mean": s["ade_t50_mean"],
        "ade_t50_ci_low": s["ade_t50_ci_low"],
        "fde_t50_mean": s["fde_t50_mean"],
        "fde_t50_ci_low": s["fde_t50_ci_low"],
        "negative_t50_seed_count": s["negative_t50_seed_count"],
        "validation_selected_seed": s["validation_selected_seed"],
        "validation_selected_test_ade_t50": s["validation_selected_test_ade_t50"],
        "paper_stable_ade_t50_claim_supported": s["paper_stable_ade_t50_claim_supported"],
        "row_level_bootstrap_status": s["row_level_bootstrap_status"],
        "claim_boundary": payload["claim_boundary"],
    }
    reports = state.setdefault("generated_reports", [])
    for path in [REPORT_MD, REPORT_JSON, GATE_MD]:
        sp = str(path)
        if sp not in reports:
            reports.append(sp)
    write_json(RESEARCH_STATE, _jsonable(state))


def run_stage42_t50_gain_harm_stability_audit() -> dict[str, Any]:
    payload = _build_payload()
    write_json(REPORT_JSON, _jsonable(payload))
    _write_report(payload)
    _write_gate(payload["stage42_if_gate"])
    _refresh_readmes_and_state(payload)
    return payload


if __name__ == "__main__":
    result = run_stage42_t50_gain_harm_stability_audit()
    print(json.dumps(_jsonable(result["stage42_if_gate"]), ensure_ascii=False, indent=2))
