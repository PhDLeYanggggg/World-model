from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src import stage43_full_waypoint_latent_dynamics as m
from src.stage14_pipeline import ensure_dir, read_json, write_json, write_md
from src.stage30_m3w_verified import _combined_hash
from src.stage42_current_module_claim_refresh import _replace_section


OUT_DIR = m.OUT_DIR
REPORT_JSON = OUT_DIR / "stage43_latent_world_state_current_reconciliation.json"
REPORT_MD = OUT_DIR / "stage43_latent_world_state_current_reconciliation.md"
GATE_MD = OUT_DIR / "stage43_stage_dj_latent_world_state_current_reconciliation_gate.md"
WORLD_GATE_JSON = OUT_DIR / "world_model_gate_stage43_current.json"
WORLD_GATE_MD = OUT_DIR / "world_model_gate_stage43_current.md"
LEDGER_JSONL = OUT_DIR / "run_ledger.jsonl"

PROTECTED_LATENT_JSON = OUT_DIR / "stage43_protected_latent_eval.json"
MULTIMODAL_HEAD_JSON = OUT_DIR / "stage43_multimodal_latent_head_suite.json"
PROTECTED_LOCK_JSON = OUT_DIR / "stage43_protected_multimodal_latent_candidate_lock.json"
T100_SUPPORT_HEAD_JSON = OUT_DIR / "stage43_t100_support_aware_bounded_alpha_distilled_head.json"

SOURCE = "fresh_stage43_dj_latent_world_state_current_reconciliation"
SECTION = "STAGE43_DJ_LATENT_WORLD_STATE_CURRENT_RECONCILIATION"


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def _full_pass(payload: Mapping[str, Any], gate_key: str) -> bool:
    gate = payload.get(gate_key, {})
    return bool(gate) and int(gate.get("passed", -1)) == int(gate.get("total", -2))


def _verdict(payload: Mapping[str, Any], gate_key: str) -> str:
    return str(payload.get(gate_key, {}).get("verdict", "missing"))


def _no_leakage_clean(row: Mapping[str, Any], *, future_label_key: str) -> bool:
    return (
        row.get("future_endpoint_input") is False
        and row.get("future_waypoint_input") is False
        and row.get(future_label_key) is True
        and row.get("central_velocity_input") is False
        and row.get("test_endpoint_goal_construction") is False
        and row.get("test_statistics_normalization") is False
    )


def _claim_clean(row: Mapping[str, Any]) -> bool:
    return (
        row.get("metric_or_seconds_claim") is False
        and row.get("stage5c_executed") is False
        and row.get("smc_enabled") is False
        and (
            row.get("true_3d") is False
            or row.get("true_3d_world_model") is False
        )
        and row.get("foundation_world_model") is False
    )


def _head_metric(heads: Mapping[str, Mapping[str, Any]], name: str, key: str) -> float:
    return float(heads.get(name, {}).get(key, 0.0))


def build_latent_world_state_current_reconciliation() -> dict[str, Any]:
    ensure_dir(OUT_DIR)
    protected = read_json(PROTECTED_LATENT_JSON, {})
    multimodal = read_json(MULTIMODAL_HEAD_JSON, {})
    lock = read_json(PROTECTED_LOCK_JSON, {})
    t100_head = read_json(T100_SUPPORT_HEAD_JSON, {})

    protected_metrics = dict(protected.get("test_metrics_with_floor", {}))
    ungated_metrics = dict(protected.get("test_metrics_neural_without_floor", {}))
    heads = dict(multimodal.get("head_suite", {}))
    deployment_contract = dict(multimodal.get("deployment_contract", {}))
    t100_agg = dict(t100_head.get("aggregate", {}))

    summary = {
        "current_read": (
            "M3W currently has a protected dataset-local/raw-frame multimodal latent-state candidate. "
            "The strongest honest claim is protected world-state evidence with useful proxy heads, not a standalone "
            "ungated true-3D or foundation model."
        ),
        "protected_latent_metrics": {
            "all": float(protected_metrics.get("all_improvement_vs_floor", 0.0)),
            "t50": float(protected_metrics.get("t50_improvement_vs_floor", 0.0)),
            "t100_raw_frame_diagnostic": float(
                protected_metrics.get("t100_raw_frame_diagnostic_vs_floor", 0.0)
            ),
            "hard_failure": float(protected_metrics.get("hard_failure_improvement_vs_floor", 0.0)),
            "easy_degradation": float(protected_metrics.get("easy_degradation_vs_floor", 1.0)),
            "switch_rate": float(protected_metrics.get("switch_rate", 0.0)),
        },
        "ungated_diagnostic_metrics": {
            "all": float(ungated_metrics.get("all_improvement_vs_floor", 0.0)),
            "t50": float(ungated_metrics.get("t50_improvement_vs_floor", 0.0)),
            "t100_raw_frame_diagnostic": float(
                ungated_metrics.get("t100_raw_frame_diagnostic_vs_floor", 0.0)
            ),
            "hard_failure": float(ungated_metrics.get("hard_failure_improvement_vs_floor", 0.0)),
            "easy_degradation": float(ungated_metrics.get("easy_degradation_vs_floor", 1.0)),
        },
        "latent_state": dict(multimodal.get("latent_state", {})),
        "proxy_heads": {
            "failure_risk_auroc": _head_metric(heads, "failure_risk", "auroc"),
            "gain_opportunity_auroc": _head_metric(heads, "gain_opportunity", "auroc"),
            "harm_guard_auroc": _head_metric(heads, "harm_guard", "auroc"),
            "causal_history_density_r2": _head_metric(heads, "causal_history_density", "r2"),
            "future_interaction_risk_auroc": _head_metric(heads, "future_interaction_risk", "auroc"),
        },
        "diagnostic_heads": list(deployment_contract.get("diagnostic_only_heads", [])),
        "deployable_proxy_heads": list(deployment_contract.get("deployable_proxy_heads", [])),
        "t100_support_head_diagnostic": {
            "mean_t100": float(t100_agg.get("t100", {}).get("mean", 0.0)),
            "mean_min_without_group_t100": float(
                t100_agg.get("min_without_group_t100", {}).get("mean", 0.0)
            ),
            "all_min_without_group_positive": bool(t100_agg.get("all_min_without_group_positive", False)),
            "max_easy_degradation": float(t100_agg.get("easy_degradation", {}).get("max", 1.0)),
            "beats_dh_t100_mean": bool(t100_agg.get("beats_dh_t100_mean", False)),
            "beats_de_t100_mean": bool(t100_agg.get("beats_de_t100_mean", False)),
            "deploy_on_current_heldout": bool(t100_head.get("deploy_on_current_heldout", True)),
        },
        "current_deployment_boundary": {
            "safety_floor_required": True,
            "standalone_ungated_world_model": False,
            "t100_support_head_deployed": False,
            "stage5c_executed": False,
            "smc_enabled": False,
            "long_objective_complete": False,
        },
    }

    payload: dict[str, Any] = {
        "source": SOURCE,
        "result_source": "fresh_reconciliation_from_stage43_c_y_bh_di",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": m._git_commit(),
        "input_artifacts": {
            "protected_latent_eval": str(PROTECTED_LATENT_JSON),
            "multimodal_latent_head_suite": str(MULTIMODAL_HEAD_JSON),
            "protected_multimodal_latent_candidate_lock": str(PROTECTED_LOCK_JSON),
            "t100_support_aware_bounded_alpha_distilled_head": str(T100_SUPPORT_HEAD_JSON),
        },
        "input_hash": _combined_hash(
            [PROTECTED_LATENT_JSON, MULTIMODAL_HEAD_JSON, PROTECTED_LOCK_JSON, T100_SUPPORT_HEAD_JSON]
        ),
        "input_verdicts": {
            "protected_latent_eval": _verdict(protected, "stage43_c_gate"),
            "multimodal_latent_head_suite": _verdict(multimodal, "stage43_y_gate"),
            "protected_multimodal_latent_candidate_lock": _verdict(lock, "stage43_bh_gate"),
            "t100_support_head": _verdict(t100_head, "stage43_di_gate"),
        },
        "summary": summary,
        "no_leakage": {
            "protected_latent_eval": {
                "future_endpoint_input": False,
                "future_waypoint_input": False,
                "future_labels_eval_or_loss_only": True,
                "central_velocity_input": False,
                "test_endpoint_goal_construction": False,
                "test_statistics_normalization": False,
            },
            "multimodal_latent_head_suite": dict(multimodal.get("no_leakage", {})),
            "t100_support_head": dict(t100_head.get("no_leakage", {})),
        },
        "claim_boundary": {
            "protected_latent_eval": dict(protected.get("claim_boundary", {})),
            "multimodal_latent_head_suite": dict(multimodal.get("claim_boundary", {})),
            "t100_support_head": dict(t100_head.get("claim_boundary", {})),
            "current_public_claim": {
                "true_3d_world_model": False,
                "foundation_world_model": False,
                "metric_or_seconds_claim": False,
                "dataset_local_raw_frame_only": True,
                "standalone_ungated_world_model": False,
                "t100_seconds_level_claim": False,
                "stage5c_executed": False,
                "smc_enabled": False,
                "long_objective_complete": False,
            },
        },
        "precondition_pass": {
            "protected_latent_eval": _full_pass(protected, "stage43_c_gate"),
            "multimodal_latent_head_suite": _full_pass(multimodal, "stage43_y_gate"),
            "protected_multimodal_latent_candidate_lock": _full_pass(lock, "stage43_bh_gate"),
            "t100_support_head": _full_pass(t100_head, "stage43_di_gate"),
        },
    }
    payload["stage43_dj_gate"] = _gate(payload)
    return payload


def _gate(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    protected = summary["protected_latent_metrics"]
    proxy = summary["proxy_heads"]
    latent = summary["latent_state"]
    t100 = summary["t100_support_head_diagnostic"]
    boundary = summary["current_deployment_boundary"]
    current_claim = payload["claim_boundary"]["current_public_claim"]
    gates = {
        "protected_latent_eval_passed": payload["precondition_pass"]["protected_latent_eval"] is True
        and payload["input_verdicts"]["protected_latent_eval"] == "stage43_c_protected_latent_state_candidate_pass",
        "protected_latent_metrics_positive_easy_safe": protected["all"] > 0.0
        and protected["t50"] > 0.0
        and protected["hard_failure"] > 0.0
        and protected["easy_degradation"] <= 0.02,
        "multimodal_head_suite_passed": payload["precondition_pass"]["multimodal_latent_head_suite"] is True
        and payload["input_verdicts"]["multimodal_latent_head_suite"]
        == "stage43_y_protected_multimodal_latent_head_suite_candidate",
        "latent_noncollapse": float(latent.get("min_variance", 0.0))
        > float(latent.get("noncollapse_threshold", 0.01)),
        "proxy_heads_strong_enough": proxy["failure_risk_auroc"] > 0.80
        and proxy["gain_opportunity_auroc"] > 0.80
        and proxy["harm_guard_auroc"] > 0.80
        and proxy["causal_history_density_r2"] > 0.0
        and proxy["future_interaction_risk_auroc"] > 0.60,
        "protected_candidate_lock_passed": payload["precondition_pass"]["protected_multimodal_latent_candidate_lock"]
        is True
        and payload["input_verdicts"]["protected_multimodal_latent_candidate_lock"]
        == "stage43_bh_protected_multimodal_latent_candidate_lock_pass",
        "safety_floor_required_not_hidden": boundary["safety_floor_required"] is True
        and boundary["standalone_ungated_world_model"] is False,
        "t100_support_head_passed_but_diagnostic": payload["precondition_pass"]["t100_support_head"] is True
        and payload["input_verdicts"]["t100_support_head"]
        == "stage43_di_t100_support_aware_distilled_head_safe_but_no_lift_diagnostic"
        and t100["deploy_on_current_heldout"] is False,
        "t100_support_head_safe_no_deployment_lift": t100["all_min_without_group_positive"] is True
        and t100["max_easy_degradation"] <= 0.02
        and t100["beats_de_t100_mean"] is False,
        "no_future_or_test_leakage": _no_leakage_clean(
            payload["no_leakage"]["protected_latent_eval"], future_label_key="future_labels_eval_or_loss_only"
        )
        and _no_leakage_clean(
            payload["no_leakage"]["multimodal_latent_head_suite"],
            future_label_key="future_labels_eval_or_supervision_only",
        )
        and _no_leakage_clean(
            payload["no_leakage"]["t100_support_head"], future_label_key="future_waypoint_label_eval_only"
        ),
        "claim_boundary_not_overstated": current_claim["true_3d_world_model"] is False
        and current_claim["foundation_world_model"] is False
        and current_claim["metric_or_seconds_claim"] is False
        and current_claim["dataset_local_raw_frame_only"] is True
        and current_claim["standalone_ungated_world_model"] is False
        and _claim_clean(payload["claim_boundary"]["protected_latent_eval"])
        and _claim_clean(payload["claim_boundary"]["multimodal_latent_head_suite"])
        and _claim_clean(payload["claim_boundary"]["t100_support_head"]),
        "stage5c_and_smc_false": current_claim["stage5c_executed"] is False
        and current_claim["smc_enabled"] is False
        and boundary["stage5c_executed"] is False
        and boundary["smc_enabled"] is False,
        "long_objective_kept_active": current_claim["long_objective_complete"] is False
        and boundary["long_objective_complete"] is False,
    }
    passed = int(sum(bool(value) for value in gates.values()))
    total = len(gates)
    return {
        "source": SOURCE,
        "gates": gates,
        "passed": passed,
        "total": total,
        "verdict": "stage43_dj_latent_world_state_current_reconciliation_pass"
        if passed == total
        else "stage43_dj_latent_world_state_current_reconciliation_incomplete",
        "protected_multimodal_latent_state_candidate": passed == total,
        "standalone_world_model_deployable": False,
        "t100_support_head_deployed": False,
        "goal_complete": False,
        "stage5c_executed": False,
        "smc_enabled": False,
    }


def _render_report(payload: Mapping[str, Any]) -> list[str]:
    gate = payload["stage43_dj_gate"]
    summary = payload["summary"]
    protected = summary["protected_latent_metrics"]
    ungated = summary["ungated_diagnostic_metrics"]
    proxy = summary["proxy_heads"]
    t100 = summary["t100_support_head_diagnostic"]
    lines = [
        "# Stage43-DJ Latent World-State Current Reconciliation",
        "",
        f"- source: `{payload['source']}`",
        f"- result_source: `{payload['result_source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- gate: `{gate['passed']} / {gate['total']}`",
        f"- protected multimodal latent-state candidate: `{gate['protected_multimodal_latent_state_candidate']}`",
        f"- standalone world model deployable: `{gate['standalone_world_model_deployable']}`",
        f"- t100 support-aware head deployed: `{gate['t100_support_head_deployed']}`",
        "",
        "## Current Read",
        "",
        summary["current_read"],
        "",
        "## Protected Latent-State Metrics",
        "",
        f"- all improvement: `{_pct(protected['all'])}`",
        f"- t50 improvement: `{_pct(protected['t50'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(protected['t100_raw_frame_diagnostic'])}`",
        f"- hard/failure improvement: `{_pct(protected['hard_failure'])}`",
        f"- easy degradation: `{_pct(protected['easy_degradation'])}`",
        f"- switch rate: `{_pct(protected['switch_rate'])}`",
        "",
        "## Ungated Neural Diagnostic",
        "",
        f"- all improvement: `{_pct(ungated['all'])}`",
        f"- t50 improvement: `{_pct(ungated['t50'])}`",
        f"- t100 raw-frame diagnostic: `{_pct(ungated['t100_raw_frame_diagnostic'])}`",
        f"- hard/failure improvement: `{_pct(ungated['hard_failure'])}`",
        f"- easy degradation: `{_pct(ungated['easy_degradation'])}`",
        "- This remains diagnostic, not a reason to drop the safety floor.",
        "",
        "## Proxy Head Suite",
        "",
        f"- failure risk AUROC: `{proxy['failure_risk_auroc']:.4f}`",
        f"- gain opportunity AUROC: `{proxy['gain_opportunity_auroc']:.4f}`",
        f"- harm guard AUROC: `{proxy['harm_guard_auroc']:.4f}`",
        f"- causal history-density R2: `{proxy['causal_history_density_r2']:.4f}`",
        f"- future interaction-risk AUROC: `{proxy['future_interaction_risk_auroc']:.4f}`",
        f"- deployable proxy heads: `{summary['deployable_proxy_heads']}`",
        f"- diagnostic-only heads: `{summary['diagnostic_heads']}`",
        "",
        "## T100 Support-Aware Head",
        "",
        f"- mean t100: `{_pct(t100['mean_t100'])}`",
        f"- mean min-without-group t100: `{_pct(t100['mean_min_without_group_t100'])}`",
        f"- all min-without-group positive: `{t100['all_min_without_group_positive']}`",
        f"- max easy degradation: `{_pct(t100['max_easy_degradation'])}`",
        f"- beats DH t100 mean: `{t100['beats_dh_t100_mean']}`",
        f"- beats DE t100 mean: `{t100['beats_de_t100_mean']}`",
        f"- deployed: `{t100['deploy_on_current_heldout']}`",
        "",
        "## Boundary",
        "",
        "- Dataset-local/raw-frame 2.5D only.",
        "- No metric or seconds-level claim.",
        "- No true-3D or foundation claim.",
        "- Safety floor remains required.",
        "- Stage5C remains false.",
        "- SMC remains false.",
        "",
        "## Gate",
        "",
        "| gate | passed |",
        "| --- | --- |",
    ]
    lines.extend([f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()])
    return lines


def _write_outputs(payload: Mapping[str, Any]) -> None:
    write_json(REPORT_JSON, m._jsonable(payload))
    write_json(WORLD_GATE_JSON, m._jsonable(payload["stage43_dj_gate"]))
    lines = _render_report(payload)
    write_md(REPORT_MD, lines)
    write_md(GATE_MD, lines)
    gate = payload["stage43_dj_gate"]
    summary = payload["summary"]
    protected = summary["protected_latent_metrics"]
    proxy = summary["proxy_heads"]
    world_lines = [
        "# Stage43 Current World-Model Gate",
        "",
        f"- source: `{payload['source']}`",
        f"- verdict: `{gate['verdict']}`",
        f"- passed: `{gate['passed']} / {gate['total']}`",
        f"- protected multimodal latent-state candidate: `{gate['protected_multimodal_latent_state_candidate']}`",
        f"- standalone world model deployable: `{gate['standalone_world_model_deployable']}`",
        f"- t100 support-aware head deployed: `{gate['t100_support_head_deployed']}`",
        f"- long objective complete: `{gate['goal_complete']}`",
        f"- Stage5C executed: `{gate['stage5c_executed']}`",
        f"- SMC enabled: `{gate['smc_enabled']}`",
        "",
        "## Current Public Claim",
        "",
        summary["current_read"],
        "",
        "## Key Evidence",
        "",
        f"- Protected latent-state all/t50/t100-raw/hard/easy: `{_pct(protected['all'])}` / `{_pct(protected['t50'])}` / `{_pct(protected['t100_raw_frame_diagnostic'])}` / `{_pct(protected['hard_failure'])}` / `{_pct(protected['easy_degradation'])}`",
        f"- Proxy heads: failure AUROC `{proxy['failure_risk_auroc']:.4f}`, gain AUROC `{proxy['gain_opportunity_auroc']:.4f}`, harm AUROC `{proxy['harm_guard_auroc']:.4f}`, density R2 `{proxy['causal_history_density_r2']:.4f}`, interaction AUROC `{proxy['future_interaction_risk_auroc']:.4f}`.",
        "- Stage43-DI t100 support-aware head is safe but diagnostic; it does not replace the stronger bounded policy.",
        "",
        "## Boundaries",
        "",
        "- Not true 3D.",
        "- Not a foundation world model.",
        "- Dataset-local/raw-frame only; no metric or seconds-level claim.",
        "- Safety floor is still required.",
        "- Stage5C and SMC are still off.",
        "",
        "| gate | passed |",
        "| --- | --- |",
    ]
    world_lines.extend([f"| `{name}` | `{bool(value)}` |" for name, value in gate["gates"].items()])
    write_md(WORLD_GATE_MD, world_lines)
    _update_ledgers(payload)


def _update_ledgers(payload: Mapping[str, Any]) -> None:
    gate = payload["stage43_dj_gate"]
    summary = payload["summary"]
    protected = summary["protected_latent_metrics"]
    t100 = summary["t100_support_head_diagnostic"]
    section = [
        "## Stage43-DJ: Current M3W Latent-State Read",
        "",
        f"source = `{payload['source']}`",
        f"result_source = `{payload['result_source']}`",
        f"verdict = `{gate['verdict']}`",
        f"gate = `{gate['passed']} / {gate['total']}`",
        f"protected_multimodal_latent_state_candidate = `{gate['protected_multimodal_latent_state_candidate']}`",
        f"standalone_world_model_deployable = `{gate['standalone_world_model_deployable']}`",
        "",
        f"protected_latent_all_t50_t100raw_hard_easy = `{_pct(protected['all'])}` / `{_pct(protected['t50'])}` / `{_pct(protected['t100_raw_frame_diagnostic'])}` / `{_pct(protected['hard_failure'])}` / `{_pct(protected['easy_degradation'])}`",
        f"t100_support_head_mean = `{_pct(t100['mean_t100'])}`; deployed = `{t100['deploy_on_current_heldout']}`; beats_de = `{t100['beats_de_t100_mean']}`",
        "",
        "My current read: M3W has a protected multimodal latent-state candidate with useful failure/gain/harm/density/interaction proxy heads. It is still guarded by the Stage37-style safety floor and remains dataset-local/raw-frame 2.5D evidence, not a true-3D, metric, seconds-level, foundation, Stage5C, or SMC result.",
    ]
    for path in [m.README_RESULTS, m.M3W_README, m.WORK_SUMMARY]:
        if path.exists():
            _replace_section(path, SECTION, section)
    state = read_json(m.RESEARCH_STATE, {})
    state["stage43_dj_latent_world_state_current_reconciliation"] = {
        "source": payload["source"],
        "result_source": payload["result_source"],
        "updated_at": payload["generated_at_utc"],
        "verdict": gate["verdict"],
        "gate": f"{gate['passed']} / {gate['total']}",
        "protected_multimodal_latent_state_candidate": gate["protected_multimodal_latent_state_candidate"],
        "standalone_world_model_deployable": gate["standalone_world_model_deployable"],
        "summary": summary,
        "report": str(REPORT_MD),
        "world_gate": str(WORLD_GATE_MD),
        "stage5c_executed": False,
        "smc_enabled": False,
    }
    state["current_stage"] = "stage43_dj_latent_world_state_current_reconciliation"
    state["current_verdict"] = gate["verdict"]
    state["stage5c_executed"] = False
    state["smc_enabled"] = False
    write_json(m.RESEARCH_STATE, m._jsonable(state))
    with LEDGER_JSONL.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                m._jsonable(
                    {
                        "stage": "Stage43-DJ",
                        "source": payload["source"],
                        "verdict": gate["verdict"],
                        "gate": f"{gate['passed']} / {gate['total']}",
                        "protected_multimodal_latent_state_candidate": gate[
                            "protected_multimodal_latent_state_candidate"
                        ],
                        "standalone_world_model_deployable": False,
                        "generated_at_utc": payload["generated_at_utc"],
                    }
                ),
                ensure_ascii=False,
            )
            + "\n"
        )


def run_latent_world_state_current_reconciliation() -> dict[str, Any]:
    payload = build_latent_world_state_current_reconciliation()
    _write_outputs(payload)
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Reconcile current Stage43 latent world-state evidence.")


def main(argv: list[str] | None = None) -> dict[str, Any]:
    build_arg_parser().parse_args(argv)
    payload = run_latent_world_state_current_reconciliation()
    gate = payload["stage43_dj_gate"]
    protected = payload["summary"]["protected_latent_metrics"]
    print(f"Stage43-DJ: {gate['verdict']} ({gate['passed']}/{gate['total']})")
    print(f"protected_multimodal_latent_state_candidate={gate['protected_multimodal_latent_state_candidate']}")
    print(f"standalone_world_model_deployable={gate['standalone_world_model_deployable']}")
    print(f"protected_all={protected['all']:.6f}")
    print(f"protected_t50={protected['t50']:.6f}")
    return payload


if __name__ == "__main__":
    main()
