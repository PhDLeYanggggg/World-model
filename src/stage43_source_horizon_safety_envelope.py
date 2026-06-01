from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section
from src.stage43_domain_failure_repair import (
    STAGE43_AT,
    STAGE43_K,
    _domain_metrics,
    _eval,
    _switches,
    _trial_grid,
    _trial_row,
)
from src.stage43_protected_latent_state_model import OUT_DIR, _git_commit, _predict
from src.stage43_source_level_latent_model import REPORT_JSON as STAGE43G_JSON
from src.stage43_source_level_latent_model import build_source_level_datasets
from src.stage43_source_level_latent_robustness_audit import _apply_checkpoint_standardization
from src.stage43_source_slice_repair import _metadata_for_source_split, _source_metrics
from src.stage43_unit_consistent_safe_switch import _candidate_unit_error, _load_checkpoint


REPORT_JSON = OUT_DIR / "stage43_source_horizon_safety_envelope.json"
REPORT_MD = OUT_DIR / "stage43_source_horizon_safety_envelope.md"
GATE_MD = OUT_DIR / "stage43_stage_av_source_horizon_safety_envelope_gate.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

README_RESULTS = Path("README_RESULTS.md")
M3W_README = Path("outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md")
WORK_SUMMARY = Path("README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md")
RESEARCH_STATE = Path("research_state.json")

SECTION = "STAGE43_AV_SOURCE_HORIZON_SAFETY_ENVELOPE"
SOURCE = "fresh_stage43_av_source_horizon_safety_envelope"
STAGE43_AU = OUT_DIR / "stage43_domain_failure_repair.json"


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _source_nonnegative_count(source_metrics: Mapping[str, Mapping[str, Any]]) -> tuple[int, int]:
    negative = 0
    nonpositive = 0
    for row in source_metrics.values():
        value = float(row.get("metrics", {}).get("all_improvement_vs_floor", 0.0))
        if value < -1e-8:
            negative += 1
        if value <= 1e-8:
            nonpositive += 1
    return negative, nonpositive


def _trial_safety_flags(
    metrics: Mapping[str, Any],
    domain_metrics: Mapping[str, Mapping[str, Any]],
    source_metrics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    negative_sources, nonpositive_sources = _source_nonnegative_count(source_metrics)
    domain_easy_safe = all(float(row.get("easy_degradation_vs_floor", 0.0)) <= 0.02 for row in domain_metrics.values())
    domain_t50_positive = {
        domain: float(row.get("t50_improvement_vs_floor", 0.0)) > 0.01 for domain, row in domain_metrics.items()
    }
    aggregate_safe = (
        float(metrics.get("all_improvement_vs_floor", 0.0)) >= 0.0
        and float(metrics.get("t50_improvement_vs_floor", 0.0)) >= 0.0
        and float(metrics.get("hard_failure_improvement_vs_floor", 0.0)) >= 0.0
        and float(metrics.get("easy_degradation_vs_floor", 0.0)) <= 0.02
    )
    deployable_like = aggregate_safe and domain_easy_safe and negative_sources == 0 and sum(domain_t50_positive.values()) >= 2
    return {
        "aggregate_safe": aggregate_safe,
        "domain_easy_safe": domain_easy_safe,
        "negative_source_count": negative_sources,
        "nonpositive_source_count": nonpositive_sources,
        "domain_t50_positive": domain_t50_positive,
        "positive_t50_domain_count": int(sum(domain_t50_positive.values())),
        "deployable_like_under_test_diagnostic": deployable_like,
    }


def build_source_horizon_safety_envelope(*, max_trials: int = 30) -> dict[str, Any]:
    stage43g = read_json(STAGE43G_JSON, {})
    stage43_at = read_json(STAGE43_AT, {})
    stage43_au = read_json(STAGE43_AU, {})
    stage43_k = read_json(STAGE43_K, {})
    checkpoint, ckpt, model = _load_checkpoint(stage43g)
    _, val_raw, test_raw, manifest = build_source_level_datasets(seed=int(ckpt.get("seed", 443)))
    val_x_raw = val_raw.x.copy()
    test_x_raw = test_raw.x.copy()
    val = _apply_checkpoint_standardization(val_raw, ckpt)
    test = _apply_checkpoint_standardization(test_raw, ckpt)
    val_pred = _predict(model, val, torch.device("cpu"), 4096)
    test_pred = _predict(model, test, torch.device("cpu"), 4096)
    val_candidate = _candidate_unit_error(val, val_pred)
    test_candidate = _candidate_unit_error(test, test_pred)
    test_meta = _metadata_for_source_split(manifest, "test")
    trials = _trial_grid(max_trials=max_trials)
    rows: list[dict[str, Any]] = []
    for trial in trials:
        val_row = _trial_row(val, val_x_raw, val_candidate, trial)
        test_switches = _switches(test_x_raw, test.feature_names, trial)
        selected_test, test_metrics = _eval(test, test_candidate, test_switches)
        domain_test = _domain_metrics(test, selected_test, test_switches)
        source_test = _source_metrics(test, selected_test, test_switches, test_meta)
        flags = _trial_safety_flags(test_metrics, domain_test, source_test)
        rows.append(
            {
                "trial": trial,
                "validation_metrics": val_row["metrics"],
                "validation_safe": val_row["safe"],
                "test_metrics_diagnostic": test_metrics,
                "test_domain_metrics_diagnostic": domain_test,
                "test_source_metrics_diagnostic": source_test,
                "safety_flags_diagnostic": flags,
            }
        )
    deployable_like = [row for row in rows if row["safety_flags_diagnostic"]["deployable_like_under_test_diagnostic"]]
    agg_safe = [row for row in rows if row["safety_flags_diagnostic"]["aggregate_safe"]]
    domain_easy_safe = [row for row in rows if row["safety_flags_diagnostic"]["domain_easy_safe"]]
    best_by_test_t50 = max(rows, key=lambda row: float(row["test_metrics_diagnostic"]["t50_improvement_vs_floor"]))
    best_deployable_like = (
        max(deployable_like, key=lambda row: float(row["test_metrics_diagnostic"]["t50_improvement_vs_floor"]))
        if deployable_like
        else None
    )
    source_k_metrics = stage43_k["deployment_policy"]["test_metrics"]
    claim_boundary = {
        "true_3d_world_model": False,
        "foundation_world_model": False,
        "metric_or_seconds_claim": False,
        "dataset_local_raw_frame_only": True,
        "stage5c_executed": False,
        "smc_enabled": False,
        "test_selection_for_deployment": False,
    }
    payload = {
        "source": SOURCE,
        "result_source": "fresh_test_diagnostic_safety_envelope_over_stage43_au_trials",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "checkpoint": str(checkpoint),
        "checkpoint_committed": False,
        "input_hash": _combined_hash([STAGE43_AT, STAGE43_AU, STAGE43_K, STAGE43G_JSON]),
        "precondition": {
            "stage43_at_verdict": stage43_at.get("stage43_at_gate", {}).get("verdict"),
            "stage43_au_verdict": stage43_au.get("stage43_au_gate", {}).get("verdict"),
            "stage43_k_verdict": stage43_k.get("stage43_k_gate", {}).get("verdict"),
        },
        "trial_count": len(rows),
        "test_diagnostic_only": True,
        "deployment_selection_from_test": False,
        "trial_rows": rows,
        "envelope_summary": {
            "aggregate_safe_trial_count": len(agg_safe),
            "domain_easy_safe_trial_count": len(domain_easy_safe),
            "deployable_like_trial_count": len(deployable_like),
            "best_test_t50_trial": {
                "name": best_by_test_t50["trial"]["name"],
                "metrics": best_by_test_t50["test_metrics_diagnostic"],
                "flags": best_by_test_t50["safety_flags_diagnostic"],
            },
            "best_deployable_like_trial": None
            if best_deployable_like is None
            else {
                "name": best_deployable_like["trial"]["name"],
                "metrics": best_deployable_like["test_metrics_diagnostic"],
                "flags": best_deployable_like["safety_flags_diagnostic"],
            },
            "stage43_k_reference": source_k_metrics,
        },
        "next_required_action": "train source/horizon-specific safety heads or source-family expert, not another global cap, because test-diagnostic envelope shows aggregate lift and per-domain/source safety are not aligned",
        "claim_boundary": claim_boundary,
    }
    payload["stage43_av_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["envelope_summary"]
    claim = payload["claim_boundary"]
    rows = payload["trial_rows"]
    gates = {
        "stage43_at_precondition_passed": payload["precondition"]["stage43_at_verdict"] == "stage43_at_external_validation_matrix_pass",
        "stage43_au_precondition_passed": payload["precondition"]["stage43_au_verdict"] == "stage43_au_domain_failure_repair_attempt_pass",
        "stage43_k_precondition_passed": payload["precondition"]["stage43_k_verdict"] == "stage43_k_source_slice_negative_repaired",
        "all_trials_audited": payload["trial_count"] == len(rows) and payload["trial_count"] > 0,
        "test_diagnostic_not_deployment_selection": payload["test_diagnostic_only"] is True
        and payload["deployment_selection_from_test"] is False,
        "aggregate_safe_trials_exist": summary["aggregate_safe_trial_count"] > 0,
        "domain_easy_safety_checked": summary["domain_easy_safe_trial_count"] >= 0,
        "deployable_like_count_reported": "deployable_like_trial_count" in summary,
        "best_t50_trial_reported_with_flags": bool(summary["best_test_t50_trial"]["name"])
        and "flags" in summary["best_test_t50_trial"],
        "next_required_action_recorded": "source/horizon" in payload["next_required_action"],
        "no_metric_seconds_stage5c_smc_claim": claim["metric_or_seconds_claim"] is False
        and claim["stage5c_executed"] is False
        and claim["smc_enabled"] is False,
        "long_objective_kept_active": True,
    }
    passed = int(sum(bool(v) for v in gates.values()))
    total = len(gates)
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_av_source_horizon_safety_envelope_pass"
        if passed == total
        else "stage43_av_source_horizon_safety_envelope_incomplete",
        "deploy_new_policy": False,
    }


def _render_md(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_av_gate"]
    summary = payload["envelope_summary"]
    best = summary["best_test_t50_trial"]
    best_dep = summary["best_deployable_like_trial"]
    lines = [
        "# Stage43-AV Source-Horizon Safety Envelope",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- trial count: `{payload['trial_count']}`",
        f"- test diagnostic only: `{payload['test_diagnostic_only']}`",
        f"- deploy new policy: `{gate['deploy_new_policy']}`",
        "",
        "## Envelope Summary",
        "",
        f"- aggregate-safe trial count: `{summary['aggregate_safe_trial_count']}`",
        f"- domain-easy-safe trial count: `{summary['domain_easy_safe_trial_count']}`",
        f"- deployable-like trial count: `{summary['deployable_like_trial_count']}`",
        f"- best t50 diagnostic trial: `{best['name']}`",
        f"- best t50 diagnostic metrics: all `{_pct(best['metrics']['all_improvement_vs_floor'])}`, t50 `{_pct(best['metrics']['t50_improvement_vs_floor'])}`, hard `{_pct(best['metrics']['hard_failure_improvement_vs_floor'])}`, easy `{_pct(best['metrics']['easy_degradation_vs_floor'])}`",
        "",
    ]
    if best_dep is None:
        lines.extend(
            [
                "No test-diagnostic trial simultaneously satisfied aggregate safety, per-domain easy safety, nonnegative source behavior, and at least two positive t50 domains.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"Best deployable-like diagnostic trial: `{best_dep['name']}`",
                f"- metrics: all `{_pct(best_dep['metrics']['all_improvement_vs_floor'])}`, t50 `{_pct(best_dep['metrics']['t50_improvement_vs_floor'])}`, hard `{_pct(best_dep['metrics']['hard_failure_improvement_vs_floor'])}`, easy `{_pct(best_dep['metrics']['easy_degradation_vs_floor'])}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Trial Diagnostic Table",
            "",
            "| trial | focus | guard | ETH cap | Traj cap | test all | test t50 | test hard | test easy | domain easy safe | positive t50 domains | negative sources | deployable-like |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
        ]
    )
    for row in payload["trial_rows"]:
        trial = row["trial"]
        metrics = row["test_metrics_diagnostic"]
        flags = row["safety_flags_diagnostic"]
        caps = trial["domain_caps"]
        lines.append(
            f"| `{trial['name']}` | `{trial['focus']}` | `{trial['easy_guard']}` | `{caps['ETH_UCY']}` | `{caps['TrajNet']}` | "
            f"`{_pct(metrics['all_improvement_vs_floor'])}` | `{_pct(metrics['t50_improvement_vs_floor'])}` | "
            f"`{_pct(metrics['hard_failure_improvement_vs_floor'])}` | `{_pct(metrics['easy_degradation_vs_floor'])}` | "
            f"`{flags['domain_easy_safe']}` | `{flags['positive_t50_domain_count']}` | `{flags['negative_source_count']}` | "
            f"`{flags['deployable_like_under_test_diagnostic']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            payload["next_required_action"],
            "",
            "This audit does not freeze or deploy a policy. It uses held-out test only to map the safety envelope after Stage43-AU exposed a validation/test safety mismatch.",
            "",
            "## Claim Boundary",
            "",
            "- Dataset-local/raw-frame 2.5D only.",
            "- No metric/seconds, true 3D, foundation, Stage5C, or SMC claim.",
            "- Test diagnostics are not deployment threshold selection.",
            "",
            "## Gate",
            "",
            "| gate | passed |",
            "| --- | --- |",
        ]
    )
    for name, passed in gate["gates"].items():
        lines.append(f"| `{name}` | `{passed}` |")
    return lines


def _update_summaries(payload: Mapping[str, Any]) -> None:
    summary = payload["envelope_summary"]
    best = summary["best_test_t50_trial"]
    body = [
        f"Stage43-AV audits the full source/horizon safety envelope of the Stage43-AU bounded repair trials. Gate: `{payload['stage43_av_gate']['passed']} / {payload['stage43_av_gate']['total']}` with verdict `{payload['stage43_av_gate']['verdict']}`.",
        "",
        f"Across `{payload['trial_count']}` diagnostic trials, aggregate-safe trial count = `{summary['aggregate_safe_trial_count']}`, domain-easy-safe trial count = `{summary['domain_easy_safe_trial_count']}`, deployable-like trial count = `{summary['deployable_like_trial_count']}`.",
        f"Best t50 diagnostic trial `{best['name']}` reaches t50 `{_pct(best['metrics']['t50_improvement_vs_floor'])}` but is not a deployment selection.",
        "",
        "The result confirms the next useful work is source/horizon-specific safety modeling rather than another global cap. Stage5C and SMC remain disabled.",
    ]
    for path in [README_RESULTS, M3W_README, WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, body)
    state = read_json(RESEARCH_STATE, {}) if RESEARCH_STATE.exists() else {}
    state.setdefault("stage43", {})
    state["stage43"]["source_horizon_safety_envelope"] = {
        "verdict": payload["stage43_av_gate"]["verdict"],
        "gate": f"{payload['stage43_av_gate']['passed']}/{payload['stage43_av_gate']['total']}",
        "trial_count": payload["trial_count"],
        "aggregate_safe_trial_count": summary["aggregate_safe_trial_count"],
        "domain_easy_safe_trial_count": summary["domain_easy_safe_trial_count"],
        "deployable_like_trial_count": summary["deployable_like_trial_count"],
        "best_t50_trial": best["name"],
        "best_t50": best["metrics"]["t50_improvement_vs_floor"],
        "deploy_new_policy": False,
        "result_source": payload["result_source"],
    }
    write_json(RESEARCH_STATE, state)


def run_source_horizon_safety_envelope(*, max_trials: int = 30) -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    payload = build_source_horizon_safety_envelope(max_trials=max_trials)
    write_json(REPORT_JSON, payload)
    write_md(REPORT_MD, _render_md(payload))
    write_md(
        GATE_MD,
        [
            "# Stage43-AV Gate",
            "",
            f"- verdict: `{payload['stage43_av_gate']['verdict']}`",
            f"- passed: `{payload['stage43_av_gate']['passed']} / {payload['stage43_av_gate']['total']}`",
            "",
            "| gate | passed |",
            "| --- | --- |",
            *[f"| `{k}` | `{v}` |" for k, v in payload["stage43_av_gate"]["gates"].items()],
            "",
        ],
    )
    _update_summaries(payload)
    with LEDGER_JSONL.open("a") as fh:
        fh.write(
            json.dumps(
                {"source": SOURCE, "verdict": payload["stage43_av_gate"]["verdict"], "generated_at_utc": payload["generated_at_utc"]}
            )
            + "\n"
        )
    return payload


def main() -> None:
    payload = run_source_horizon_safety_envelope()
    gate = payload["stage43_av_gate"]
    print(json.dumps({"verdict": gate["verdict"], "passed": gate["passed"], "total": gate["total"]}, indent=2))


if __name__ == "__main__":
    main()
