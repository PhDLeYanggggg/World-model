from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_protected_latent_state_model import OUT_DIR, _git_commit, _jsonable
from src.stage43_unit_consistent_safe_switch import REPORT_JSON as STAGE43I_JSON


REPORT_JSON = OUT_DIR / "stage43_source_level_caveat_audit.json"
REPORT_MD = OUT_DIR / "stage43_source_level_caveat_audit.md"
GATE_MD = OUT_DIR / "stage43_stage_j_source_level_caveat_gate.md"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE43_J_SOURCE_LEVEL_CAVEAT_AUDIT"
SOURCE = "fresh_stage43_j_source_level_caveat_audit"


def _source_rows(stage43i: Mapping[str, Any]) -> list[dict[str, Any]]:
    sources = stage43i.get("deployment_policy", {}).get("source_metrics", {})
    rows: list[dict[str, Any]] = []
    for source_id, row in sorted(sources.items()):
        metrics = row.get("metrics", {})
        rows.append(
            {
                "source_id": source_id,
                "domains": row.get("domains", []),
                "scenes": row.get("scenes", []),
                "rows": int(metrics.get("rows", 0)),
                "all_improvement_vs_floor": float(metrics.get("all_improvement_vs_floor", 0.0)),
                "t50_improvement_vs_floor": float(metrics.get("t50_improvement_vs_floor", 0.0)),
                "t100_raw_frame_diagnostic_vs_floor": float(metrics.get("t100_raw_frame_diagnostic_vs_floor", 0.0)),
                "hard_failure_improvement_vs_floor": float(metrics.get("hard_failure_improvement_vs_floor", 0.0)),
                "easy_degradation_vs_floor": float(metrics.get("easy_degradation_vs_floor", 0.0)),
                "switch_rate": float(metrics.get("switch_rate", 0.0)),
            }
        )
    return rows


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    gates = {
        "stage43_i_passed": payload["stage43_i_precondition"]["verdict"] == "stage43_i_unit_consistent_safe_switch_pass",
        "source_metrics_present": payload["source_count"] > 0,
        "nonpositive_source_detected": payload["nonpositive_source_count"] > 0,
        "uniform_source_claim_blocked": payload["source_uniform_candidate"] is False,
        "domain_level_claim_preserved": payload["domain_level_candidate"] is True,
        "no_test_tuned_repair_attempted": payload["repair_attempted"] is False,
        "claim_boundary_recorded": payload["claim_boundary"]["uniform_per_source_claim"] is False
        and payload["claim_boundary"]["metric_or_seconds_claim"] is False
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
        "verdict": "stage43_j_source_level_caveat_mapped" if passed == total else "stage43_j_source_level_caveat_incomplete",
        "source_uniform_candidate": False,
        "domain_level_candidate": True,
    }


def run_audit() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    stage43i = read_json(STAGE43I_JSON, {})
    rows = _source_rows(stage43i)
    nonpositive = [row for row in rows if row["all_improvement_vs_floor"] <= 0.0]
    negative_t50 = [row for row in rows if row["t50_improvement_vs_floor"] <= 0.0]
    worst = min(rows, key=lambda r: r["all_improvement_vs_floor"]) if rows else None
    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_run_source_level_caveat_audit_from_stage43_i_outputs",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "stage43_i_precondition": {
            "path": str(STAGE43I_JSON),
            "verdict": stage43i.get("stage43_i_gate", {}).get("verdict"),
            "gate": f"{stage43i.get('stage43_i_gate', {}).get('passed')} / {stage43i.get('stage43_i_gate', {}).get('total')}",
            "policy": stage43i.get("deployment_policy", {}).get("name"),
            "policy_hash": stage43i.get("deployment_policy", {}).get("policy_hash"),
        },
        "source_count": len(rows),
        "source_rows": rows,
        "nonpositive_source_count": len(nonpositive),
        "nonpositive_sources": nonpositive,
        "nonpositive_t50_source_count": len(negative_t50),
        "worst_source": worst,
        "source_uniform_candidate": False,
        "domain_level_candidate": True,
        "repair_attempted": False,
        "recommended_next_action": {
            "action": "stage43_k_source_slice_repair_without_test_threshold_tuning",
            "reason": "Stage43-I passes domain-level safety, but one small TrajNet source is slightly negative and several source-level t50 slices are floor-only.",
            "allowed_methods": [
                "train/validation-only source-family gate",
                "source-conditioned uncertainty gate",
                "source-level conformal risk guard",
                "retrain with source-balanced objective",
            ],
            "forbidden_methods": [
                "disable a test source by source id",
                "choose thresholds from test source metrics",
                "claim uniform source-level success before repair",
            ],
        },
        "claim_boundary": {
            "domain_level_candidate": True,
            "uniform_per_source_claim": False,
            "dataset_local_raw_frame_only": True,
            "t100_raw_frame_diagnostic_only": True,
            "metric_or_seconds_claim": False,
            "true_3d": False,
            "foundation_world_model": False,
            "stage5c_executed": False,
            "smc_enabled": False,
        },
    }
    payload["stage43_j_gate"] = _gate(payload)
    _write_outputs(payload)
    return payload


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, _jsonable(payload))
    gate = payload["stage43_j_gate"]
    worst = payload["worst_source"] or {}
    lines = [
        "# Stage43-J Source-Level Caveat Audit",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- source uniform candidate: `{gate['source_uniform_candidate']}`",
        f"- domain-level candidate: `{gate['domain_level_candidate']}`",
        f"- repair attempted: `{payload['repair_attempted']}`",
        "",
        "## Finding",
        "",
        "Stage43-I is a unit-consistent domain-level protected latent candidate, but it is not a uniform per-source success claim. This audit intentionally blocks that overclaim before the result is used in a paper package.",
        "",
        f"- source count: `{payload['source_count']}`",
        f"- nonpositive all-improvement source count: `{payload['nonpositive_source_count']}`",
        f"- nonpositive t50 source count: `{payload['nonpositive_t50_source_count']}`",
        f"- worst source: `{worst.get('source_id')}`",
        f"- worst source all improvement: `{float(worst.get('all_improvement_vs_floor', 0.0)):.6f}`",
        f"- worst source t50 improvement: `{float(worst.get('t50_improvement_vs_floor', 0.0)):.6f}`",
        f"- worst source easy degradation: `{float(worst.get('easy_degradation_vs_floor', 0.0)):.6f}`",
        "",
        "## Source Metrics",
        "",
        "| source | domains | scenes | rows | all | t50 | t100 raw | hard/failure | easy degradation | switch |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *[
            f"| {row['source_id']} | {','.join(row['domains'])} | {','.join(row['scenes'])} | {row['rows']} | {row['all_improvement_vs_floor']:.6f} | {row['t50_improvement_vs_floor']:.6f} | {row['t100_raw_frame_diagnostic_vs_floor']:.6f} | {row['hard_failure_improvement_vs_floor']:.6f} | {row['easy_degradation_vs_floor']:.6f} | {row['switch_rate']:.6f} |"
            for row in payload["source_rows"]
        ],
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
        *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        "",
        "Conclusion: Stage43-I should be described as a protected domain-level candidate with source-level caveats. The next repair must use train/validation-only source-family gating or source-balanced retraining, not test-source threshold tuning.",
    ]
    write_md(REPORT_MD, lines)
    write_md(
        GATE_MD,
        [
            "# Stage43-J Source-Level Caveat Gate",
            "",
            f"- verdict: `{gate['verdict']}`",
            f"- gate: `{gate['passed']} / {gate['total']}`",
            f"- source uniform candidate: `{gate['source_uniform_candidate']}`",
            f"- domain-level candidate: `{gate['domain_level_candidate']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| {name} | {bool(value)} |" for name, value in gate["gates"].items()],
        ],
    )
    _update_readmes(payload)


def _update_readmes(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_j_gate"]
    worst = payload["worst_source"] or {}
    section = [
        f"## {SECTION}",
        "",
        f"source = `{payload['source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"source_uniform_candidate = `{gate['source_uniform_candidate']}`",
        f"domain_level_candidate = `{gate['domain_level_candidate']}`",
        "",
        "Stage43-J audits the Stage43-I source-level slices and blocks a uniform per-source claim. Stage43-I remains a unit-consistent domain-level protected latent candidate, but one small TrajNet source is slightly negative and multiple source t50 slices remain floor-only.",
        "",
        f"Worst source `{worst.get('source_id')}`: all `{float(worst.get('all_improvement_vs_floor', 0.0)):.6f}`, t50 `{float(worst.get('t50_improvement_vs_floor', 0.0)):.6f}`, easy degradation `{float(worst.get('easy_degradation_vs_floor', 0.0)):.6f}`.",
        "",
        "Next allowed repair: source-family gate or source-balanced retraining selected on train/validation only. Forbidden: disabling a test source by source id or tuning thresholds from test source metrics.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(RESEARCH_STATE, {})
    state["stage43_j_source_level_caveat_audit"] = {
        "source": payload["source"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "source_uniform_candidate": gate["source_uniform_candidate"],
        "domain_level_candidate": gate["domain_level_candidate"],
        "source_count": payload["source_count"],
        "nonpositive_source_count": payload["nonpositive_source_count"],
        "nonpositive_t50_source_count": payload["nonpositive_t50_source_count"],
        "worst_source": payload["worst_source"],
        "recommended_next_action": payload["recommended_next_action"],
        "report": str(REPORT_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    write_json(RESEARCH_STATE, state)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)
    return run_audit()


if __name__ == "__main__":
    result = main()
    gate = result["stage43_j_gate"]
    print(f"Stage43-J source caveat audit: {gate['verdict']} ({gate['passed']}/{gate['total']})")
